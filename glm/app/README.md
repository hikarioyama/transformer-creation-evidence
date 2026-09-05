# The Next-Token Machine

An **interactive 3D browser app** that dissects how a tiny decoder-only Transformer
predicts the next token — with **real inference running live in your browser** and
every number on screen coming from the actual computation.

> **This is an educational toy model.** 2 layers · 2 attention heads · hidden size 16 ·
> vocabulary 16 · context 16. It was trained once, offline; its weights are frozen into
> the page. No training happens in the browser and no external APIs are used.

Screenshot omitted from this public release; visual audit evidence is summarized separately.

## Run it

No build step, no install needed to *use* the app:

```bash
# option A — just open it (works from [REDACTED_LOCAL_FILE_URL] too)
open index.html            # macOS
xdg-open index.html        # Linux

# option B — tiny local server
python3 -m http.server 8000
# → http://localhost:8000
```

Everything (three.js r128, weights, token data) is vendored locally — the page runs fully offline.

## The toy language

The model was trained on one deterministic rule:

```
next token = (token one step back) + 1   (mod 16)
```

so sequences look like `a, b, a+1, b+1, a+2, b+2, …`. The **only** way to predict the
next token is to look one step back — so the trained attention heads develop a crisp
"previous-token" circuit (layer-0 heads put ~68% / ~98% of their attention mass on
offset −1), which is exactly what the 3D view shows you. Edit the sequence however you
like: the attention spike follows the weights, not the data.

## What you can do

- **Step through 15 stages** (left panel, arrow keys, or ▶ play): tokens → embeddings →
  LayerNorm → Q/K/V → attention → residual → MLP, for both blocks, → final LN → logits → softmax.
- **Orbit/zoom** the 3D pipeline (drag / scroll). Every slab renders a real tensor:
  height ∝ |value|, cyan = negative, amber = positive; each slab's label shows its true numeric range.
- **Click any token** (chips or attention posts) to select a position; all inspector panels follow.
- **Inspect actual intermediates** in the right panel: attention heatmaps with exact weights,
  raw score rows, Q/K/V vectors, 64-dim MLP activations, weight matrices with a value readout.
- **Edit the sequence** (bottom bar): click a chip to swap its token, pop, generate with
  the model's own top prediction, or reshuffle.
- **Toggle KV caching** and append tokens: cached positions freeze (ice-blue) in the Q/K/V
  views and only the newest column is computed. The **Verify** tab runs *both* paths over the
  current sequence and compares all logits and probabilities — they match **bitwise** (max |Δ| = 0).

## Architecture & exact math

GPT-style pre-LN decoder, implemented from scratch in plain JavaScript (`js/model.js`):

```
X⁰[i]   = E[token_i] + P[i]                         E,P: learned 16×16 tables
block b: X′ = X + Attn(LN₁(X)) ;  X = X′ + MLP(LN₂(X))
Attn:    Q = x̂·Wq, K = x̂·Wk, V = x̂·Wv   (16→16, no biases)
         2 heads × head_dim 8;  score[i][j] = Q_i·K_j / √8   (j ≤ i, causal mask)
         A = softmax(score) ;  out = concat_h(A_h V_h)·Wo
MLP:     h = x̂·W1+b1 (16→64);  GELU(h) (tanh approx);  ·W2+b2 (64→16)
LN:      γ⊙(x−μ)/√(σ²+1e-5)+β per position
output:  logits = LN_f(X)·Wuᵀ  (16 scores);  P(next) = softmax(logits)
```

Weights were produced by `tools/train.py` (numpy, hand-derived backprop **verified by
numerical gradient check**, worst rel. err 1.4e-7; full-batch Adam, 2000 steps, final
loss 3e-5, 100% next-token accuracy). They are exported as exact float64 JSON
(`js/weights.js`), so the browser reproduces the numpy computation bit-for-bit.

## Verification

| test | command | result |
|---|---|---|
| JS engine vs numpy reference logits/attention | `npm run test:model` | max Δ ≈ 7e-15 |
| KV-cached incremental vs full recompute | `npm run test:model` | max Δ = **0.0** (bitwise) |
| prefill + append, cache invalidation semantics | `npm run test:model` | pass |
| headless browser: all 15 steps, tabs, editing, cache, in-app parity | `npm run test:ui` | pass, zero console errors |
| edge cases: length 1, max context 16, position-0 edit | `npm run test:edge` | pass |
| model correctness on random sequences | inside edge test | 40/40 argmax = rule |

UI tests use Playwright + bundled Chromium: `npx playwright install chromium` if needed.

## Files

```
index.html            the app (open this)
css/style.css         UI styling
js/model.js           the inference engine — all transformer math, from scratch
js/weights.js         frozen trained weights (generated; do not edit)
js/tokens.js          the 16-token vocabulary + demo sequence
js/viz3d.js           three.js scene: tensor slabs, attention arcs, bars, camera
js/app.js             steps, explanations, inspectors, sequence editor, cache logic
tools/train.py        offline training + export (numpy, gradient-checked)
tools/debug_grad.py   per-parameter gradient diagnostics
test/                 node + headless-browser test suites
vendor/               three.js r128 + OrbitControls (vendored)
```

## Honest limitations

- The task is deliberately trivial; the interesting part is the *circuit* (previous-token
  attention), not the language.
- The model has 16 positions of context; the "language" has no beginning-of-string token,
  so position 0's prediction is unsupervised noise.
- Layer 2 mostly re-mixes layer 1's output — with 2 layers there isn't room for much else.
  That's part of the lesson: capacity shapes circuits.
- Cached and uncached paths are bitwise identical here because sums run in a fixed order;
  a real GPU kernel may differ by float reassociation (~1e-7), which is what "tolerance"
  is for. The app still reports the true measured max |Δ|.
