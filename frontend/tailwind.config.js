/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      fontFamily: {
        sans: ['Inter', 'HarmonyOS Sans', 'system-ui', 'sans-serif'],
      },
      colors: {
        ink: '#111827',
      },
      borderRadius: {
        '2xl': '16px',
        '3xl': '24px',
      },
      boxShadow: {
        'soft': '0 2px 20px rgba(0,0,0,0.04)',
        'glow': '0 0 0 3px rgba(110,168,254,0.15)',
      },
    },
  },
  plugins: [],
}
