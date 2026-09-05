/* The 16-token vocabulary of the toy language. */
window.TINY_TOKENS = [
  { emoji: "\u{1F311}", name: "new moon" }, // 0
  { emoji: "\u2604\uFE0F",  name: "comet" },     // 1
  { emoji: "\u2B50",        name: "star" },      // 2
  { emoji: "\u2600\uFE0F",  name: "sun" },       // 3
  { emoji: "\u{1F319}", name: "crescent" }, // 4
  { emoji: "\u26A1",        name: "bolt" },      // 5
  { emoji: "\u{1F525}", name: "fire" },     // 6
  { emoji: "\u{1F4A7}", name: "drop" },     // 7
  { emoji: "\u{1F331}", name: "seedling" }, // 8
  { emoji: "\u{1F338}", name: "blossom" },  // 9
  { emoji: "\u{1F34E}", name: "apple" },    // 10
  { emoji: "\u{1F98B}", name: "butterfly" },// 11
  { emoji: "\u{1F41F}", name: "fish" },     // 12
  { emoji: "\u{1F408}", name: "cat" },      // 13
  { emoji: "\u{1F422}", name: "turtle" },   // 14
  { emoji: "\u{1F41D}", name: "bee" },      // 15
].map((t, i) => ({ ...t, id: i, hue: Math.round(i * 360 / 16) }));

/* Sequence used for the default demo — follows the toy language's rule
   "next token = one-back token + 1 (mod 16)": 4,9,5,10,6,11,7,12,(8)…        */
window.TINY_DEMO = [4, 9, 5, 10, 6, 11, 7, 12];
