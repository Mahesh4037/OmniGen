# 🌐 OmniGen AI - Multimodal AI Generation Platform

**OmniGen AI** is a fully local, Flask-based multimodal generative AI web application that empowers users to generate text, images, and videos using state-of-the-art open-source models from Hugging Face. It provides a clean, interactive UI and supports user-supplied Hugging Face tokens via `.env`, enabling full control and flexibility.

---

## 🚀 Features

- 🧠 **Text Generation**: Powered by [Qwen/Qwen2-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2-1.5B-Instruct) for multilingual, code-aware, and instruction-following text outputs.

- 🎨 **Image Generation**: Uses [segmind/SSD-1B](https://huggingface.co/segmind/SSD-1B), a fast and lightweight Stable Diffusion model for ultra-HD images.

- 🎥 **Video Generation**: Integrates [Lightricks/LTX-Video](https://huggingface.co/Lightricks/LTX-Video), a cutting-edge video diffusion model.

- 🔐 **Secure API Access**: Uses Hugging Face tokens from a `.env` file to authenticate securely.
- 🗃️ **History Tracking**: Stores generated content with timestamps and filenames.
- 💡 **Frontend**: Clean UI with advanced UX support (can be customized for themes, loaders, etc.)

---

## 📂 Folder Structure

```
├── app.py               # Main Flask backend
├── .env                 # Your Hugging Face token (not committed)
├── templates/           # HTML frontend templates
│   ├── base.html
│   ├── index.html
│   ├── text_generation.html
│   ├── image_generation.html
│   └── video_generation.html
├── static/              # Optional: custom CSS/JS or generated media
└── README.md            # This file
```

---

## 🛠️ Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/OmniGen-AI.git
cd OmniGen-AI
```

### 2. Create and Activate a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Add Your Hugging Face Token

Create a `.env` file in the root directory:

```bash
echo "HF_TOKEN=your_huggingface_token_here" > .env
```

> 🔐 **Never share your token or commit ****\`\`**** to GitHub.**

You can generate a token here: [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)

---

## ▶️ Running the App

```bash
python app.py
```

Visit: [http://localhost:5000](http://localhost:5000)

---

## 🤖 Models Used

| Task  | Model Name               | License    | Link                                                                                               |
| ----- | ------------------------ | ---------- | -------------------------------------------------------------------------------------------------- |
| Text  | Qwen/Qwen2-1.5B-Instruct | Apache 2.0 | [https://huggingface.co/Qwen/Qwen2-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2-1.5B-Instruct) |
| Image | segmind/SSD-1B           | MIT        | [https://huggingface.co/segmind/SSD-1B](https://huggingface.co/segmind/SSD-1B)                     |
| Video | Lightricks/LTX-Video     | MIT        | [https://huggingface.co/Lightricks/LTX-Video](https://huggingface.co/Lightricks/LTX-Video)         |

All models are publicly available and open-source.

---

## 📜 License

This project is licensed under the **MIT License**.

```
MIT License

Copyright (c) 2025 Mahesh Singla

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND.
```

---

## 🙌 Acknowledgements

- Hugging Face 🤗 for providing model APIs and hosting.
- Open source contributors for each model.
- Flask community for the lightweight web framework.

---

## 📧 Contact

**Author:** Mahesh Singla\
**Email:** [ssingla2004@gmail.com](mailto\:ssingla2004@gmail.com)\
**GitHub:** [github.com/Maheshsingla4037](https://github.com/Maheshsingla4037)

Feel free to open issues or contribute to the project!

---

> 💡 *“AI is not just the future — it's your canvas. Generate boldly.”*

