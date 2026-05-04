/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        brand: {
          50:  '#eef3f8',
          100: '#d5e2ef',
          500: '#3b6fa0',
          700: '#1a3a5c',
          900: '#0d1e30',
        }
      }
    }
  },
  plugins: [],
}
