document.addEventListener('DOMContentLoaded', () => {
    const themeToggle = document.getElementById('themeToggle');
    const savedTheme = localStorage.getItem('theme') || 'light';
  
    const applyTheme = (isDark) => {
      document.documentElement.classList.toggle('dark', isDark);
      localStorage.setItem('theme', isDark ? 'dark' : 'light');
    };
  
    if (themeToggle) {
      themeToggle.addEventListener('click', () => {
        const isDark = document.documentElement.classList.toggle('dark');
        applyTheme(isDark);
      });
    }
  
    // Mobile menu
    const mobileMenuButton = document.getElementById('mobile-menu-button');
    const mobileMenu = document.getElementById('mobile-menu');
  
    if (mobileMenuButton && mobileMenu) {
      mobileMenuButton.addEventListener('click', (e) => {
        e.stopPropagation();
        mobileMenu.classList.toggle('hidden');
      });
  
      document.addEventListener('click', (e) => {
        if (!mobileMenu.contains(e.target) && !mobileMenuButton.contains(e.target)) {
          mobileMenu.classList.add('hidden');
        }
      });
    }
  
    // Smooth scroll
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
      anchor.addEventListener('click', function (e) {
        e.preventDefault();
        const target = document.querySelector(this.getAttribute('href'));
        target?.scrollIntoView({ behavior: 'smooth' });
      });
    });
  
    // Form handling
    document.querySelectorAll('form').forEach(form => {
      form.addEventListener('submit', (e) => {
        const button = form.querySelector('button[type="submit"]');
        if (button) {
          button.disabled = true;
          button.classList.add('opacity-50', 'cursor-not-allowed');
        }
  
        const loadingState = form.closest('.glass-card')?.nextElementSibling?.querySelector('#loadingState');
        if (loadingState) loadingState.classList.remove('hidden');
  
        const resultsContainer = form.closest('.glass-card')?.nextElementSibling?.querySelector('#resultsContainer');
        if (resultsContainer) resultsContainer.classList.add('hidden');
      });
    });
  
    // Toast system
    window.showToast = (message, type = 'success') => {
      const toast = document.createElement('div');
      toast.className = `toast-message fixed bottom-4 right-4 px-4 py-2 rounded-lg text-sm transition-opacity ${
        type === 'success' 
          ? 'bg-green-100 text-green-800 dark:bg-green-900/20 dark:text-green-200' 
          : 'bg-red-100 text-red-800 dark:bg-red-900/20 dark:text-red-200'
      }`;
      toast.textContent = message;
      document.body.appendChild(toast);
  
      setTimeout(() => {
        toast.classList.add('opacity-0');
        setTimeout(() => toast.remove(), 300);
      }, 3000);
    };
  
    // Clipboard copy
    window.copyToClipboard = (text) => {
      navigator.clipboard.writeText(text).then(() => {
        showToast('Copied to clipboard!', 'success');
      }).catch(() => {
        showToast('Failed to copy text', 'error');
      });
    };
  
    // File download
    window.downloadFile = (data, filename) => {
      const link = document.createElement('a');
      link.href = data;
      link.download = filename;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
    };
  
    // Range Slider value label
    document.querySelectorAll('input[type="range"]').forEach(slider => {
      const display = document.createElement('div');
      display.className = 'text-sm text-gray-500 dark:text-gray-400 mt-1';
      slider.parentNode.appendChild(display);
  
      const updateDisplay = () => {
        display.textContent = slider.value;
        if (slider.name === 'temperature') {
          display.textContent += ' (Creativity)';
        } else if (slider.name === 'top_p') {
          display.textContent += ' (Focus)';
        }
      };
  
      slider.addEventListener('input', updateDisplay);
      updateDisplay();
    });
  
    // Video click play/pause
    document.querySelectorAll('video').forEach(video => {
      video.addEventListener('click', () => {
        video.paused ? video.play() : video.pause();
      });
    });
  
    // History card click
    document.querySelectorAll('.history-card').forEach(card => {
      card.addEventListener('click', (e) => {
        if (!e.target.closest('button')) {
          const prompt = card.dataset.prompt;
          // Optional: handle history click
        }
      });
    });
  
    // Initialize theme on load
    applyTheme(savedTheme === 'dark');
  });
  
  // Catch unexpected JS errors
  window.onerror = (message, source, lineno, colno, error) => {
    console.error(`Error: ${message} at ${source}:${lineno}:${colno}`);
    showToast('An unexpected error occurred', 'error');
    return true;
  };
  