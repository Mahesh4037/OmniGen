import os
import io
import base64
import gc
import uuid
import logging
from pathlib import Path
from flask import Flask, render_template, request, redirect, url_for, send_from_directory

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from diffusers import StableDiffusionXLPipeline, LTXPipeline
from diffusers.utils import export_to_video
from huggingface_hub import login
from PIL import Image
from dotenv import load_dotenv
load_dotenv()

# Configure environment variables
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"
os.environ["XFORMERS_ENABLE_MEM_EFFICIENT_ATTENTION"] = "1"

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Hugging Face authentication
hf_token = os.getenv("HF_TOKEN")
if not hf_token:
    raise RuntimeError("HF_TOKEN not found in .env file. Please create one with your Hugging Face token.")
login(token=hf_token)

# Flask app setup
app = Flask(__name__)

# Configuration
OUTPUT_DIR = Path.cwd() / "video_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# Global model cache
model_cache = {"model": None, "tokenizer": None, "type": None}

# Global history
history = []

def unload_model():
    """Unload current model and free memory"""
    if model_cache["model"]:
        logger.info(f"Unloading {model_cache['type']} pipeline")
        del model_cache["model"]
        if model_cache["tokenizer"]:
            del model_cache["tokenizer"]
        gc.collect()
        torch.cuda.empty_cache()
        model_cache.update({"model": None, "tokenizer": None, "type": None})


def load_model(model_type):
    """Load the specified model type into memory"""
    if model_cache["type"] == model_type and model_cache["model"]:
        return model_cache["model"], model_cache["tokenizer"]

    unload_model()
    tokenizer = None

    if model_type == "text":
        # Qwen 2 instruction-tuned 1.5B parameters model
        model_id = "Qwen/Qwen2-1.5B-Instruct"
        quant_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_enable_fp32_cpu_offload=True
        )
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=quant_config,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )

    elif model_type == "image":
        model = StableDiffusionXLPipeline.from_pretrained(
            "segmind/SSD-1B",
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16"
        ).to("cuda")

    elif model_type == "video":
        model = LTXPipeline.from_pretrained(
            "Lightricks/LTX-Video",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        )
        model.enable_attention_slicing()
        model.enable_model_cpu_offload()
        model.safety_checker = lambda images, **kwargs: (images, [False] * len(images))

    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    model_cache.update({"model": model, "tokenizer": tokenizer, "type": model_type})
    return model, tokenizer


def add_prompt_to_history(prompt, model_type):
    history.insert(0, {"prompt": prompt, "type": model_type})

@app.route("/clear-history")
def clear_history():
    history.clear()
    return redirect(url_for("index"))

# Main routes
@app.route("/")
def index():
    return render_template("index.html", history=history)

@app.route("/text")
def text_generation():
    return render_template("text_generation.html", history=history)

@app.route("/image")
def image_generation():
    return render_template("image_generation.html", history=history)

@app.route("/video")
def video_generation():
    return render_template("video_generation.html", history=history)

@app.route("/generate/text", methods=["POST"])
def gen_text():
    prompt = request.form["prompt"].strip()
    add_prompt_to_history(prompt, "text")

    # Parameters from HTML form
    max_tokens = min(int(request.form.get("max_length", 512)), 1024)
    temp = max(min(float(request.form.get("temperature", 0.7)), 2.0), 0.1)
    top_p = max(min(float(request.form.get("top_p", 0.9)), 1.0), 0.1)

    # Load and run model
    model, tokenizer = load_model("text")
    messages = [
        {"role": "system", "content": "You are Qwen, a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    prompt_input = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt_input, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=temp,
            top_p=top_p
        )
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return render_template("text_generation.html", generated_text=decoded, history=history)

@app.route("/generate/image", methods=["POST"])
def gen_image():
    prompt = request.form["prompt"].strip()
    add_prompt_to_history(prompt, "image")

    # Parameters from HTML form
    width = int(request.form.get("width", 1024))
    height = int(request.form.get("height", 1024))
    steps = min(int(request.form.get("steps", 35)), 50)
    guidance_scale = float(request.form.get("guidance_scale", 7.5))

    pipe, _ = load_model("image")
    try:
        result = pipe(
            prompt,
            height=height,
            width=width,
            num_inference_steps=steps,
            guidance_scale=guidance_scale
        )
        image = result.images[0]

        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode()
        return render_template("image_generation.html", img_data=encoded, history=history)
    except Exception as e:
        logger.exception("Image generation failed")
        return render_template("image_generation.html", error=str(e), history=history)

@app.route("/generate/video", methods=["POST"])
def gen_video():
    prompt = request.form["prompt"].strip()
    add_prompt_to_history(prompt, "video")

    # Parameters from HTML form
    width = int(request.form.get("width", 768))
    height = int(request.form.get("height", 512))
    num_frames = int(request.form.get("num_frames", 121))
    num_steps = int(request.form.get("num_inference_steps", 40))
    negative_prompt = request.form.get(
        "negative_prompt",
        "blurry, low quality, pixelated, noisy, grainy, washed out, overexposed, underexposed, lens flare, flickering, glitch, artifacting, distorted faces, malformed hands, motion blur, color banding, jagged edges, cartoon, sketch, monochrome, text, logo, camera shake, unnatural lighting"
    )

    try:
        pipe, _ = load_model("video")
        result = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_frames=num_frames,
            num_inference_steps=num_steps
        )
        video_frames = result.frames[0]

        job_id = uuid.uuid4().hex
        job_dir = OUTPUT_DIR / job_id
        job_dir.mkdir()
        out_file = job_dir / "output.mp4"
        export_to_video(video_frames, str(out_file), fps=int(request.form.get("fps", 24)))

        video_url = f"/video-output/{job_id}/output.mp4"
        return render_template("video_generation.html", video_url=video_url, history=history)
    except Exception as e:
        logger.exception("Video generation failed")
        return render_template("video_generation.html", error=str(e), history=history)

@app.route("/video-output/<job>/<filename>")
def serve_video(job, filename):
    return send_from_directory(str(OUTPUT_DIR / job), filename)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
