/*
 * Node test suite for the TinyModel engine.
 *   1. JS logits & attention vs numpy reference (float64 exact round-trip)
 *   2. KV-cached incremental forward vs full forward
 *   3. cache invalidation semantics
 * Run:  node test/test_model.js
 */
global.window = {};                       // weights.js expects a browser-ish global
require('../js/weights.js');
const TM = require('../js/model.js');
const ref = require('./test_reference.json');

const model = TM.load(window.TINY_WEIGHTS);
let failures = 0;
const ok = (cond, label, extra) => {
  console.log((cond ? '  PASS ' : '  FAIL ') + label + (extra ? '  ' + extra : ''));
  if (!cond) failures++;
};

console.log('== 1. parity vs numpy reference ==');
for (const tc of ref) {
  const tr = TM.forwardFull(model, tc.ids);
  let dLogit = 0, dAttn = 0;
  for (let i = 0; i < tc.ids.length; i++)
    for (let v = 0; v < model.V; v++)
      dLogit = Math.max(dLogit, Math.abs(tr.logits[i][v] - tc.logits[i][v]));
  for (let l = 0; l < model.L; l++)
    for (let h = 0; h < model.H; h++)
      for (let i = 0; i < tc.ids.length; i++) {
        const r = tr.blocks[l].attn[h][i];
        for (let j = 0; j <= i; j++)
          dAttn = Math.max(dAttn, Math.abs(r[j] - tc.attn[`L${l}H${h}`][i][j]));
      }
  ok(dLogit < 1e-9 && dAttn < 1e-9,
     `seq len ${tc.ids.length}: maxΔlogit=${dLogit.toExponential(2)} maxΔattn=${dAttn.toExponential(2)}`);
}

console.log('== 2. probs sanity ==');
{
  const tr = TM.forwardFull(model, ref[0].ids);
  let worst = 0;
  for (const p of tr.probs) {
    const s = p.reduce((a, b) => a + b, 0);
    worst = Math.max(worst, Math.abs(s - 1));
    for (const q of p) if (q < 0 || q > 1) worst = Infinity;
  }
  ok(worst < 1e-12, `prob rows sum to 1 (worst=${worst.toExponential(2)})`);
}

console.log('== 3. KV cache vs full recompute ==');
for (const tc of ref) {
  const ids = tc.ids;
  const full = TM.forwardFull(model, ids);
  const cache = TM.createCache(model);
  let worst = 0;
  let worstWhere = '';
  for (let t = 1; t <= ids.length; t++) {
    const inc = TM.forwardCached(model, ids.slice(0, t), cache);
    const i = t - 1;
    for (let v = 0; v < model.V; v++) {
      const d = Math.abs(inc.logits[i][v] - full.logits[i][v]);
      if (d > worst) { worst = d; worstWhere = `logit pos ${i} dim ${v}`; }
      const dp = Math.abs(inc.probs[i][v] - full.probs[i][v]);
      if (dp > worst) { worst = dp; worstWhere = `prob pos ${i} dim ${v}`; }
    }
    // attention rows must match too
    for (let l = 0; l < model.L; l++)
      for (let h = 0; h < model.H; h++)
        for (let j = 0; j <= i; j++) {
          const d = Math.abs(inc.attnFull[l][h][i][j] - full.blocks[l].attn[h][i][j]);
          if (d > worst) { worst = d; worstWhere = `attn L${l}H${h} [${i},${j}]`; }
        }
  }
  ok(worst < 1e-9, `cached==uncached for len ${ids.length}: maxΔ=${worst.toExponential(3)} (${worstWhere})`);
}

console.log('== 4. one-shot incremental append ==');
{
  const ids = ref[0].ids;
  const full = TM.forwardFull(model, ids);
  const cache = TM.createCache(model);
  // prefill with 6 tokens, then append the rest in one call
  TM.forwardCached(model, ids.slice(0, 6), cache);
  const inc = TM.forwardCached(model, ids, cache);
  let worst = 0;
  for (const i of inc.fresh)
    for (let v = 0; v < model.V; v++)
      worst = Math.max(worst, Math.abs(inc.logits[i][v] - full.logits[i][v]));
  ok(worst < 1e-9, `prefill+append maxΔ=${worst.toExponential(3)}`);
  ok(inc.fresh.length === ids.length - 6 && inc.cachedCount === 6,
     `fresh=${inc.fresh.length}, cached=${inc.cachedCount}`);
}

console.log('== 5. cache invalidation ==');
{
  const ids = ref[1].ids;
  const cache = TM.createCache(model);
  TM.forwardCached(model, ids, cache);
  ok(TM.cacheMatches(cache, ids), 'matching prefix accepted');
  ok(!TM.cacheMatches(cache, ids.slice(0, 5)), 'shorter prefix rejected');
  const edited = ids.slice(); edited[2] = (edited[2] + 1) % model.V;
  ok(!TM.cacheMatches(cache, edited), 'edited old token rejected');
  ok(!TM.cacheMatches(cache, [ids[0], ids[1], (ids[2] + 3) % model.V, ...ids.slice(3)]), 'different suffix rejected');
  TM.resetCache(cache);
  ok(cache.pos === 0, 'resetCache empties cache');
  const inc = TM.forwardCached(model, ids, cache);
  ok(inc.fresh.length === ids.length, 'after reset everything is fresh');
}

console.log('== 6. verifyParity helper ==');
{
  const r = TM.verifyParity(model, ref[2].ids);
  ok(r.ok, `verifyParity maxΔ=${r.maxAbsDiff.toExponential(3)}`);
}

console.log(failures === 0 ? '\nALL TESTS PASSED' : `\n${failures} TEST(S) FAILED`);
process.exit(failures === 0 ? 0 : 1);
