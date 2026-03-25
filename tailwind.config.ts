import type { Config } from 'tailwindcss';

/** Class-based dark mode (toggle via `dark` on `document.documentElement`). */
export default {
  darkMode: 'class',
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
} satisfies Config;
