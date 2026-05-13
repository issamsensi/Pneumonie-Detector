/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx,ts,tsx}'],
  theme: {
    extend: {
      colors: {
        brand: {
          50: '#effaf8',
          100: '#d7f4f0',
          200: '#aee9e0',
          300: '#7ad7cb',
          400: '#3fbfb1',
          500: '#18a394',
          600: '#0f8378',
          700: '#0d6961',
          800: '#0c5450',
          900: '#0b4744',
        },
        medical: {
          50: '#eff6ff',
          100: '#dbeafe',
          200: '#bfdbfe',
          300: '#93c5fd',
          400: '#60a5fa',
          500: '#3b82f6',
          600: '#2563eb',
          700: '#1d4ed8',
          800: '#1e40af',
          900: '#1e3a8a',
        },
      },
      boxShadow: {
        soft: '0 10px 25px -12px rgba(2, 6, 23, 0.18)',
      },
    },
  },
  plugins: [],
}

