/*
 * TinyModel — the complete inference engine for a 2-layer decoder-only Transformer.
 *
 *   vocab=16 · d_model=16 · 2 heads (head_dim=8) · 2 pre-LN blocks · MLP 16→64→16
 *   GELU (tanh approx) · learned positional embeddings · max context 16
 *
 * The math is implemented from scratch here (no ML libraries). All reductions run
 * in ascending index order so that the KV-cached and uncached paths produce
 * *bitwise identical* float64 results.
 *
 * Weights come from js/weights.js (trained offline by tools/train.py, fixed forever).
 */
(function (root) {
  'use strict';

  // ---------- constants (mirrored from weights meta) ----------
  const EPS = 1e-5;

  // ---------- tiny linalg ----------
  function gelu(x) {
    const c = Math.sqrt(2.0 / Math.PI);
    return 0.5 * x * (1.0 + Math.tanh(c * (x + 0.044715 * x * x * x)));
  }

  // softmax over arr[0..n-1]; masked entries (j>i) are simply never included
  function softmaxRow(arr, n) {
    let m = -Infinity;
    for (let j = 0; j < n; j++) if (arr[j] > m) m = arr[j];
    let s = 0;
    const out = new Array(n);
    for (let j = 0; j < n; j++) { out[j] = Math.exp(arr[j] - m); s += out[j]; }
    for (let j = 0; j < n; j++) out[j] /= s;
    return out;
  }

  function softmaxVec(v) {
    let m = -Infinity;
    for (let i = 0; i < v.length; i++) if (v[i] > m) m = v[i];
    let s = 0;
    const out = new Array(v.length);
    for (let i = 0; i < v.length; i++) { out[i] = Math.exp(v[i] - m); s += out[i]; }
    for (let i = 0; i < v.length; i++) out[i] /= s;
    return out;
  }

  // row vector @ matrix (matrix stored row-major, shape [nIn][nOut])
  // y[e] = sum_d x[d] * M[d*nOut + e], ascending d
  function matvec(x, M, nIn, nOut) {
    const y = new Array(nOut).fill(0);
    for (let d = 0; d < nIn; d++) {
      const xd = x[d];
      const base = d * nOut;
      for (let e = 0; e < nOut; e++) y[e] += xd * M[base + e];
    }
    return y;
  }

  function layernorm(x, g, b) {
    let mu = 0;
    for (let d = 0; d < x.length; d++) mu += x[d];
    mu /= x.length;
    let vr = 0;
    for (let d = 0; d < x.length; d++) { const t = x[d] - mu; vr += t * t; }
    vr /= x.length;
    const rstd = 1.0 / Math.sqrt(vr + EPS);
    const y = new Array(x.length), xhat = new Array(x.length);
    for (let d = 0; d < x.length; d++) {
      xhat[d] = (x[d] - mu) * rstd;
      y[d] = g[d] * xhat[d] + b[d];
    }
    return { y, xhat, mu, rstd };
  }

  // ---------- model ----------
  function load(tinyWeights) {
    const meta = tinyWeights.meta, w = tinyWeights.weights;
    const flat = (name) => Float64Array.from(w[name].flat());
    const model = {
      meta,
      V: meta.V, D: meta.D, H: meta.H, HD: meta.HD, L: meta.L, F: meta.F,
      CTX: meta.CTX, SCALE: meta.SCALE,
      tokEmb: flat('tok_emb'),
      posEmb: flat('pos_emb'),
      blocks: [],
      lnfG: flat('lnf_g'), lnfB: flat('lnf_b'),
      Wu: flat('Wu'),
    };
    for (let l = 0; l < meta.L; l++) {
      model.blocks.push({
        ln1G: flat(`block${l}.ln1_g`), ln1B: flat(`block${l}.ln1_b`),
        Wq: flat(`block${l}.Wq`), Wk: flat(`block${l}.Wk`),
        Wv: flat(`block${l}.Wv`), Wo: flat(`block${l}.Wo`),
        ln2G: flat(`block${l}.ln2_g`), ln2B: flat(`block${l}.ln2_b`),
        W1: flat(`block${l}.W1`), b1: flat(`block${l}.b1`),
        W2: flat(`block${l}.W2`), b2: flat(`block${l}.b2`),
      });
    }
    return model;
  }

  // ---------- attention over one query row ----------
  // qRow: [D]; kRows: array of [D] (length = i+1); returns per-head rows
  function attnRow(qRow, kRows, vRows, i, model, collect) {
    const { H, HD, D, SCALE } = model;
    const scores = [], attn = [], headOut = [];
    for (let h = 0; h < H; h++) {
      const off = h * HD;
      const s = new Array(i + 1);
      for (let j = 0; j <= i; j++) {
        let dotv = 0;
        const kr = kRows[j];
        for (let d = 0; d < HD; d++) dotv += qRow[off + d] * kr[off + d];
        s[j] = dotv * SCALE;
      }
      const a = softmaxRow(s, i + 1);
      const o = new Array(HD).fill(0);
      for (let j = 0; j <= i; j++) {
        const aj = a[j], vr = vRows[j];
        for (let d = 0; d < HD; d++) o[d] += aj * vr[off + d];
      }
      scores.push(s); attn.push(a); headOut.push(o);
    }
    if (collect) collect(scores, attn, headOut);
    return headOut;
  }

  function concatHeads(headOut, D, HD) {
    const row = new Array(D);
    for (let h = 0; h < headOut.length; h++)
      for (let d = 0; d < HD; d++) row[h * HD + d] = headOut[h][d];
    return row;
  }

  function processRow(model, xRow, block, collect) {
    // one position through one block; returns {x1, x2, ...intermediates}
    const D = model.D, F = model.F;
    const ln1 = layernorm(xRow, block.ln1G, block.ln1B);
    const q = matvec(ln1.y, block.Wq, D, D);
    const k = matvec(ln1.y, block.Wk, D, D);
    const v = matvec(ln1.y, block.Wv, D, D);
    const headOut = attnRow(q, [k], [v], 0, model, collect); // placeholder, replaced below
    return { ln1, q, k, v, headOut };
  }

  // ---------- full (uncached) forward ----------
  function forwardFull(model, ids) {
    const { D, V, L, F } = model;
    const T = ids.length;
    if (T > model.CTX) throw new Error(`sequence length ${T} > max context ${model.CTX}`);
    const trace = {
      ids: ids.slice(), T, fromCache: false,
      tokEmb: [], posEmb: [], embed: [],
      blocks: [],
      lnf: [], logits: [], probs: [],
    };
    for (let i = 0; i < T; i++) {
      const te = Array.from(model.tokEmb.subarray(ids[i] * D, (ids[i] + 1) * D));
      const pe = Array.from(model.posEmb.subarray(i * D, (i + 1) * D));
      const e = new Array(D);
      for (let d = 0; d < D; d++) e[d] = te[d] + pe[d];
      trace.tokEmb.push(te); trace.posEmb.push(pe); trace.embed.push(e);
    }
    let x = trace.embed.map(r => r.slice());
    for (let l = 0; l < L; l++) {
      const block = model.blocks[l];
      const bt = {
        ln1: [], q: [], k: [], v: [],
        scores: [], attn: [],        // [H][T][<=T]  (row i has i+1 entries)
        headOut: [], attnConcat: [], attnProj: [],
        x1: [], ln2: [], mlpHidden: [], mlpAct: [], mlpOut: [], x2: [],
      };
      // project all rows first so attention can read any j <= i
      for (let i = 0; i < T; i++) {
        const ln1 = layernorm(x[i], block.ln1G, block.ln1B);
        bt.ln1.push(ln1.y);
        bt.q.push(matvec(ln1.y, block.Wq, D, D));
        bt.k.push(matvec(ln1.y, block.Wk, D, D));
        bt.v.push(matvec(ln1.y, block.Wv, D, D));
      }
      for (let i = 0; i < T; i++) {
        const headOut = attnRow(bt.q[i], bt.k, bt.v, i, model);
        const concat = concatHeads(headOut, D, model.HD);
        const proj = matvec(concat, block.Wo, D, D);
        const x1 = new Array(D), x2 = new Array(D);
        for (let d = 0; d < D; d++) x1[d] = x[i][d] + proj[d];
        const ln2 = layernorm(x1, block.ln2G, block.ln2B);
        const h = matvec(ln2.y, block.W1, D, F);
        for (let f = 0; f < F; f++) h[f] += block.b1[f];
        const g = new Array(F);
        for (let f = 0; f < F; f++) g[f] = gelu(h[f]);
        const mo = matvec(g, block.W2, F, D);
        for (let d = 0; d < D; d++) mo[d] += block.b2[d];
        for (let d = 0; d < D; d++) x2[d] = x1[d] + mo[d];
        bt.headOut.push(headOut); bt.attnConcat.push(concat); bt.attnProj.push(proj);
        bt.x1.push(x1); bt.ln2.push(ln2.y); bt.mlpHidden.push(h); bt.mlpAct.push(g);
        bt.mlpOut.push(mo); bt.x2.push(x2);
      }
      // per-head score/attn matrices for the inspector (filled below)
      bt.x2_in = x.map(r => r.slice());
      x = bt.x2;
      trace.blocks.push(bt);
    }
    // per-head score/attn rows for inspection (cheap, T<=16)
    for (let l = 0; l < L; l++) {
      const bt = trace.blocks[l];
      for (let h = 0; h < model.H; h++) { bt.scores.push([]); bt.attn.push([]); }
      for (let i = 0; i < T; i++) {
        for (let h = 0; h < model.H; h++) {
          const off = h * model.HD, s = new Array(i + 1);
          for (let j = 0; j <= i; j++) {
            let dotv = 0;
            for (let d = 0; d < model.HD; d++) dotv += bt.q[i][off + d] * bt.k[j][off + d];
            s[j] = dotv * model.SCALE;
          }
          bt.scores[h][i] = s;
          bt.attn[h][i] = softmaxRow(s, i + 1);
        }
      }
    }
    for (let i = 0; i < T; i++) {
      const lnf = layernorm(x[i], model.lnfG, model.lnfB);
      const lg = matvec(lnf.y, model.Wu, D, V);
      trace.lnf.push(lnf.y);
      trace.logits.push(lg);
      trace.probs.push(softmaxVec(lg));
    }
    trace.xFinal = x;
    return trace;
  }

  // ---------- KV cache ----------
  function createCache(model) {
    return {
      model, pos: 0, prefixIds: [],
      layers: model.blocks.map(() => ({ kRows: [], vRows: [] })), // rows: [D]
      attnRows: model.blocks.map(() => []),  // attnRows[l][i] = [H][i+1]
    };
  }

  function cacheMatches(cache, ids) {
    if (cache.pos > ids.length) return false;
    for (let i = 0; i < cache.pos; i++) if (cache.prefixIds[i] !== ids[i]) return false;
    return true;
  }

  /*
   * Incremental forward: positions [0, cache.pos) are served from the cache,
   * positions [cache.pos, T) are computed fresh. cache is updated in place.
   * Returns a partial trace for the fresh positions plus full assembled K/V and
   * attention matrices (fresh + cached) for visualization.
   */
  function forwardCached(model, ids, cache) {
    if (!cacheMatches(cache, ids))
      throw new Error('cache does not match the sequence prefix — clear it first');
    const { D, V, L, F, H, HD } = model;
    const T = ids.length;
    const start = cache.pos;
    const fresh = [];
    for (let i = start; i < T; i++) fresh.push(i);

    const result = {
      ids: ids.slice(), T, fromCache: true, start, fresh,
      cachedCount: start,
      blocks: [],
      lnf: {}, logits: {}, probs: {},
    };

    for (let l = 0; l < L; l++) {
      const block = model.blocks[l];
      const cl = cache.layers[l];
      const bt = { ln1: {}, q: {}, k: {}, v: {}, scores: [], attn: [], headOut: {}, attnConcat: {}, attnProj: {}, x1: {}, ln2: {}, mlpHidden: {}, mlpAct: {}, mlpOut: {}, x2: {} };
      const xin = {};   // residual input rows for fresh positions
      for (const i of fresh) xin[i] = (l === 0)
        ? (() => {
            const te = Array.from(model.tokEmb.subarray(ids[i] * D, (ids[i] + 1) * D));
            const pe = Array.from(model.posEmb.subarray(i * D, (i + 1) * D));
            const e = new Array(D);
            for (let d = 0; d < D; d++) e[d] = te[d] + pe[d];
            return e;
          })()
        : prevX2[i];

      for (const i of fresh) {
        const ln1 = layernorm(xin[i], block.ln1G, block.ln1B);
        const q = matvec(ln1.y, block.Wq, D, D);
        const k = matvec(ln1.y, block.Wk, D, D);
        const v = matvec(ln1.y, block.Wv, D, D);
        bt.ln1[i] = ln1.y; bt.q[i] = q; bt.k[i] = k; bt.v[i] = v;
        cl.kRows[i] = k; cl.vRows[i] = v;
      }
      for (const i of fresh) {
        const headOut = attnRow(bt.q[i], cl.kRows, cl.vRows, i, model);
        const concat = concatHeads(headOut, D, HD);
        const proj = matvec(concat, block.Wo, D, D);
        const x1 = new Array(D), x2 = new Array(D);
        for (let d = 0; d < D; d++) x1[d] = xin[i][d] + proj[d];
        const ln2 = layernorm(x1, block.ln2G, block.ln2B);
        const h = matvec(ln2.y, block.W1, D, F);
        for (let f = 0; f < F; f++) h[f] += block.b1[f];
        const g = new Array(F);
        for (let f = 0; f < F; f++) g[f] = gelu(h[f]);
        const mo = matvec(g, block.W2, F, D);
        for (let d = 0; d < D; d++) mo[d] += block.b2[d];
        for (let d = 0; d < D; d++) x2[d] = x1[d] + mo[d];
        bt.headOut[i] = headOut; bt.attnConcat[i] = concat; bt.attnProj[i] = proj;
        bt.x1[i] = x1; bt.ln2[i] = ln2.y; bt.mlpHidden[i] = h; bt.mlpAct[i] = g;
        bt.mlpOut[i] = mo; bt.x2[i] = x2;
        // attention rows for the inspector (kept per-position, keyed by head)
        const rowsH = [];
        for (let h = 0; h < H; h++) {
          const off = h * HD, s = new Array(i + 1);
          const qRow = bt.q[i];
          for (let j = 0; j <= i; j++) {
            let dotv = 0;
            for (let d = 0; d < HD; d++) dotv += qRow[off + d] * cl.kRows[j][off + d];
            s[j] = dotv * model.SCALE;
          }
          rowsH.push({ scores: s, attn: softmaxRow(s, i + 1) });
        }
        bt.scores[i] = rowsH.map(r => r.scores);
        bt.attn[i] = rowsH.map(r => r.attn);
        cache.attnRows[l][i] = rowsH;
      }
      result.blocks.push(bt);
      if (l === 0) prevX2 = bt.x2;
    }

    for (const i of fresh) {
      const lnf = layernorm(result.blocks[L - 1].x2[i], model.lnfG, model.lnfB);
      const lg = matvec(lnf.y, model.Wu, D, V);
      result.lnf[i] = lnf.y;
      result.logits[i] = lg;
      result.probs[i] = softmaxVec(lg);
    }

    cache.pos = T;
    cache.prefixIds = ids.slice();

    // assembled full views for the viz
    result.kAll = cache.layers.map(cl => cl.kRows.slice());
    result.vAll = cache.layers.map(cl => cl.vRows.slice());
    result.attnFull = cache.attnRows.map(rows => {
      const m = [];
      for (let h = 0; h < H; h++) {
        const mh = [];
        for (let i = 0; i < T; i++) {
          const r = rows[i] ? rows[i][h].attn : null;
          mh.push(r);
        }
        mh._scores = [];
        for (let i = 0; i < T; i++) mh._scores.push(rows[i] ? rows[i][h].scores : null);
        m.push(mh);
      }
      return m;
    });
    return result;
  }
  let prevX2 = null;

  // reset helper
  function resetCache(cache) {
    cache.pos = 0; cache.prefixIds = [];
    cache.layers.forEach(cl => { cl.kRows = []; cl.vRows = []; });
    cache.attnRows.forEach(a => { a.length = 0; });
  }

  // ---------- verification: cached vs uncached ----------
  function verifyParity(model, ids) {
    const full = forwardFull(model, ids);
    const cache = createCache(model);
    let worst = 0, worstAt = null;
    const perPos = [];
    for (let t = 1; t <= ids.length; t++) {
      const prefix = ids.slice(0, t);
      const inc = forwardCached(model, prefix, cache);
      const i = t - 1;
      for (let v = 0; v < model.V; v++) {
        const d = Math.abs(inc.logits[i][v] - full.logits[i][v]);
        if (d > worst) { worst = d; worstAt = [i, v]; }
      }
      const p1 = inc.probs[i], p2 = full.probs[i];
      for (let v = 0; v < model.V; v++) {
        const d = Math.abs(p1[v] - p2[v]);
        if (d > worst) { worst = d; worstAt = [i, v]; }
      }
      perPos.push({ pos: i, maxDiff: 0 });
    }
    return { maxAbsDiff: worst, worstAt, ok: worst < 1e-9, full, perPos };
  }

  const api = {
    gelu, softmaxRow, softmaxVec, layernorm, matvec,
    load, forwardFull, createCache, forwardCached, cacheMatches, resetCache, verifyParity,
  };

  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.TinyModel = api;
})(typeof window !== 'undefined' ? window : globalThis);
