/** Callas design tokens. Colors resolve through CSS custom properties set on :root in static/src/app.css. */
module.exports = {
  content: ["./templates/**/*.html", "./static/js/**/*.js"],
  theme: {
    colors: {
      transparent: "transparent",
      current: "currentColor",
      white: "#ffffff",
      ground: "var(--ground)",
      surface: "var(--surface)",
      ink: "var(--ink)",
      text: "var(--text)",
      muted: "var(--muted)",
      rule: "var(--rule)",
      "rule-strong": "var(--rule-strong)",
      alarm: "var(--alarm)",
      "alarm-tint": "var(--alarm-tint)",
      "alarm-line": "var(--alarm-line)",
      safe: "var(--safe)",
      "safe-tint": "var(--safe-tint)",
      "safe-line": "var(--safe-line)",
      unsure: "var(--unsure)",
      "unsure-tint": "var(--unsure-tint)",
      "unsure-line": "var(--unsure-line)",
      paper: "var(--paper)",
    },
    borderRadius: { none: "0", DEFAULT: "10px", full: "9999px" },
    fontFamily: {
      display: ['"Kanit"', '"Sarabun"', '"Leelawadee UI"', "Tahoma", "sans-serif"],
      body: ['"Sarabun"', '"Leelawadee UI"', "Tahoma", "sans-serif"],
    },
    extend: {},
  },
};
