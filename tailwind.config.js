/** @type {import('tailwindcss').Config} */
module.exports = {
  // Enable class-based dark mode toggling
  darkMode: 'class',
  content: [
    // All template and JS files where Tailwind classes are used
    './templates/**/*.{html,js}',
    './static/js/**/*.{html,js}',
    './src/**/*.{html,js,css}',
  ],
  darkMode: 'class', // Use `class="dark"` to toggle
  theme: {
    extend: {
      fontFamily: {
        'space-grotesk': ['"Space Grotesk"', 'sans-serif'],
      },
      colors: {
        'quantum-void': 'hsl(230, 45%, 5%)',
        'quantum-abyss': 'hsl(230, 40%, 10%)',
        'quantum-singularity': 'hsl(255, 65%, 55%)',
        'quantum-event-horizon': 'hsl(265, 75%, 65%)',
        'quantum-mist': 'hsl(240, 20%, 97%)',
        'quantum-twill': 'hsl(240, 15%, 85%)',
        'quantum-light': 'hsl(240, 20%, 99%)',
        'quantum-deep': 'hsl(240, 30%, 15%)',
      },
      boxShadow: {
        'quantum-lg': '0 10px 30px rgba(0,0,0,0.2)',
        'quantum-xl': '0 15px 40px rgba(0,0,0,0.3)',
        'quantum-2xl': '0 25px 50px rgba(0,0,0,0.35)',
      },
      fontFamily: {
        'space-grotesk': ['"Space Grotesk"', 'sans-serif'],
      },
      backdropBlur: {
        xl: '20px',
      },
    },
  },
  plugins: [
      require('@tailwindcss/forms'),
      require('@tailwindcss/typography'),
  ],
}