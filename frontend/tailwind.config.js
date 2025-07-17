/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      colors: {
        primary: "#00f5ff",
        secondary: "#ff073a",
        warning: "#ff8c00",
        success: "#39ff14",
        danger: "#ff073a",
        info: "#00f5ff",
        "dark-bg": "#0a0a0a",
        "dark-surface": "#121212",
        "dark-border": "#333333",
      },
      fontFamily: {
        mono: ["JetBrains Mono", "monospace"],
      },
      animation: {
        "pulse-glow": "pulse-glow 2s ease-in-out infinite",
      },
      keyframes: {
        "pulse-glow": {
          "0%, 100%": {
            filter: "drop-shadow(0 0 5px var(--primary-color))",
          },
          "50%": {
            filter: "drop-shadow(0 0 15px var(--primary-color))",
          },
        },
      },
    },
  },
  plugins: [],
};
