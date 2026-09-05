# Small Signals · Transformer Observatory

An interactive **3D visualization of a real decoder-only Transformer** computing a
next-token prediction — entirely in your browser, with the model math implemented
from scratch (no ML libraries, no training, no network calls at runtime).

> **This is an educational toy model.** It has 2 layers, 2 attention heads, hidden
> size 16, and a 16-token vocabulary. Its weights are **fixed random values
> (untrained)** — predictions are arbitrary by construction. The point is to make
> every number of a genuine Transformer forward pass visible and inspectable.

## Run it

Any static file server works (ES modules refuse `[REDACTED_LOCAL_FILE_URL] URLs):

```bash
cd tiny-transformer-3d
python3 -m http.server 8080 --bind 127.0.0.1
# or: npx serve .
```

Open <http://localhost:8080>. No build step, no dependencies to install; Three.js
is vendored in `vendor/` and served locally.

## What you can do

- **Step through all 18 computation stages** (embed → 2 transformer blocks →
  final norm → logits → softmax) with prev/next, play/pause, and keyboard ←/→.
- **Inspect actual intermediate values** in the 3D residual-stream scene (8
  stations; hover any cell for its exact Float64 value) and in the tensor
  inspector (click any cell for full 17-digit precision).
- **Edit the input**: change any token via its dropdown, append/remove tokens,
  or greedy-append the model's own prediction. Context limit: 12 tokens.
- **Toggle KV caching**: with the cache on, an edit or append recomputes only
  the invalidated suffix. The status bar and the "KV cache / proof" tab show the
  reused-vs-computed split and continuously audit the active path against a full
  uncached forward pass (tolerance 10⁻¹⁰; observed max |Δ| is exactly 0).
- **Attention microscope**: full causal attention matrices per layer/head, plus a
  per-pair breakdown (Q·K, scaling, masking, softmax, A×V contribution).
- **Export** the complete trace — weights, config, every intermediate tensor,
  KV cache — as JSON.

## Architecture

```
token IDs (1–12) → embed(id) + sinusoidal PE        [16 dims]
  × 2 blocks, pre-norm:
      u = LayerNorm(x);  Q,K,V = uW_Q, uW_K, uW_V   [16×16 each, 2 heads × 8]
      A = softmax(QKᵀ/√8 + causal −∞ mask); ctx = A V;  r = x + (ctx)W_O
      v = LayerNorm(r);  r = r + GELU(v W₁) W₂      [16 → 32 → 16]
z = LayerNorm(x_final);  logits = z W_U;  P = softmax(logits)
```

No biases, dropout, or learned norm parameters; ε = 10⁻⁵; population variance;
tanh-approximation GELU; untied embedding/output matrices; Float64 throughout.

## Reproducibility

Seed `20250308` via Mulberry32. Embeddings ~ uniform [−0.65, 0.65]; projection
weights Xavier-uniform (√(6/(fan_in+fan_out))). `WEIGHTS` is computed once at
load and is byte-identical on every run (unit-tested).

## Tests

```bash
npm test                                  # 9 unit tests: math, masking, cache equivalence
python3 tests/numpy_reference.py          # independent NumPy re-implementation cross-check
npm start &                               # serve, then:
node tests/browser.mjs                    # optional Playwright integration test
```

The browser test needs Playwright once: `npm install playwright && npx playwright
install chromium`. It drives the real UI (all 18 stages, cache toggle equivalence,
edits, export, mobile layout) and asserts **zero page errors and zero external
requests**. Screenshots land in `tests/artifacts/`.

## Files

| File | Purpose |
|---|---|
| `model.js` | All model math: RNG, embeddings, LayerNorm, attention, MLP, full forward, `KVDecoder`, audits |
| `stages.js` | The 18 stage definitions (explanations, formulas, tensor selectors) |
| `scene.js` | Three.js observatory: instanced value tiles, attention arcs, picking |
| `app.js` | UI wiring, stepping, inspector tables, cache report, export |
| `index.html` / `style.css` | Layout and visual design |
| `vendor/` | Three.js + OrbitControls (served locally, no CDN) |
