/* see: <https://docs.mathjax.org/en/latest/options/input/tex.html#tex-options> */
MathJax = {
  tex: {
    packages: {
      "[+]": ["noundefined", "autoload", "ams", "textmacros", "physics"],
    },
    inlineMath: [
      /* start/end delimiter pairs for in-line math */
      ["$", "$"],
      ["\\(", "\\)"],
    ],
    displayMath: [
      /* start/end delimiter pairs for display math */
      ["$$", "$$"],
      ["\\[", "\\]"],
    ],
    tags: "none", // or 'ams' or 'all'
  },
};
