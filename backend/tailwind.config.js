/** Callas design tokens. Colors resolve through CSS custom properties set on :root in static/src/app.css. */
module.exports = {
  content: ["./templates/**/*.html", "./static/js/**/*.js"],
  theme: {
    colors: {
      transparent: "transparent",
      current: "currentColor",
      white: "#ffffff",
      ink: "var(--ink)",
      stock: "var(--stock)",
      alarm: "var(--alarm)",
      "alarm-deep": "var(--alarm-deep)",
      "alarm-tint": "var(--alarm-tint)",
      safe: "var(--safe)",
      "safe-tint": "var(--safe-tint)",
      rule: "var(--rule)",
      muted: "var(--muted)",
      paper: "var(--paper)",
    },
    borderRadius: { none: "0", DEFAULT: "0" },
    fontFamily: {
      display: ['"Kanit"', '"Sarabun"', '"Leelawadee UI"', "Tahoma", "sans-serif"],
      body: ['"Sarabun"', '"Leelawadee UI"', "Tahoma", "sans-serif"],
    },
    extend: {},
  },
};
