/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  theme: {
    extend: {
      colors: {
        ivory: "#fbf7ef",
        linen: "#eee2d0",
        oat: "#d9c4aa",
        taupe: "#9d8268",
        cocoa: "#5f4938",
        ink: "#322921"
      },
      fontFamily: {
        display: ["Cormorant Garamond", "Georgia", "serif"],
        sans: ["Inter", "Avenir Next", "Segoe UI", "sans-serif"]
      },
      boxShadow: {
        soft: "0 24px 70px rgba(95, 73, 56, 0.12)"
      }
    }
  },
  plugins: []
};
