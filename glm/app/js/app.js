/*
 * app.js — UI wiring: stepping, explanations, inspectors, sequence editing, KV cache.
 */
(function () {
  'use strict';
  const TM = window.TinyModel, VIZ = window.Viz3D, TOK = window.TINY_TOKENS;
  const model = TM.load(window.TINY_WEIGHTS);

  // ---------------------------------------------------------------- state
  const state = {
    ids: window.TINY_DEMO.slice(),
    selPos: window.TINY_DEMO.length - 1,
    step: 0,
    head: 0,
    attL: 0,
    kvOn: false,
    cache: TM.createCache(model),
    trace: null,          // display trace, always full-shaped
    fresh: new Set(),     // positions computed last recompute
    frozen: new Set(),    // positions served from cache last recompute
    cacheStatus: 'no cache activity yet',
    lastVerify: null,
    playing: false, playTimer: null,
  };

  const fmt = (v, d = 3) => (v < 0 ? '−' : '') + Math.abs(v).toFixed(d);
  const norm = v => Math.sqrt(v.reduce((a, x) => a + x * x, 0));
  const emoji = id => TOK[id].emoji;
  const pct = p => (100 * p).toFixed(1) + '%';
  const mono = s => `<span class="mono">${s}</span>`;

  // ---------------------------------------------------------------- steps
  const STAGE_KEYS = ['embed', 'b0.ln', 'b0.qkv', 'b0.attn', 'b0.add', 'b0.mlp',
                      'b1.ln', 'b1.qkv', 'b1.attn', 'b1.add', 'b1.mlp', 'lnf', 'out'];

  function topK(vals, k) {
    return vals.map((v, i) => [i, v]).sort((a, b) => b[1] - a[1]).slice(0, k);
  }

  const STEPS = [
    {
      stage: 'embed', title: 'The input',
      explain(c) {
        return `A decoder-only Transformer sees a <b>sequence of tokens</b>. Our toy vocabulary has exactly
        <b>16 tokens</b>; the sequence below has <b>${c.T}</b>. The model's entire job: predict the <b>next</b> token.
        <div class="note">This is a <b>toy language</b> with one rule you can check:
        the next token is always <b>one more than the token one step back</b> (…14, 15, 0, 1…).
        A real language model learned grammar from trillions of words; this one learned a single rule —
        but the machinery is the same.</div>
        Click any token chip (or a token post in the attention stage) to select a position —
        every panel then shows that position's real numbers.`;
      },
    },
    {
      stage: 'embed', title: 'Embedding: token + position',
      explain(c) {
        const i = c.selPos, id = c.ids[i];
        const te = c.trace.tokEmb[i], pe = c.trace.posEmb[i], em = c.trace.embed[i];
        return `Each token ID indexes a row of the learned embedding matrix <b>E</b> (16×16). Networks have no
        built-in sense of order, so a second learned table <b>P</b> injects <i>where</i> each token sits.
        They are simply added:
        <div class="formula">X⁰[i] = E[token<sub>i</sub>] + P[i]</div>
        For ${emoji(id)} at position ${i}: ‖E[${id}]‖ = ${mono(fmt(norm(te)))},
        ‖P[${i}]‖ = ${mono(fmt(norm(pe)))} → ‖X⁰[${i}]‖ = ${mono(fmt(norm(em)))}.
        <div class="note">From here on, each position carries a 16-dimensional <b>residual stream</b> that
        every layer reads from and writes back into.</div>`;
      },
    },
    {
      stage: 'b0.ln', title: 'Block 1 · LayerNorm',
      explain(c) {
        const i = c.selPos;
        const x = c.trace.blocks[0].x2_in ? c.trace.embed[i] : c.trace.embed[i];
        const ln = c.trace.blocks[0].ln1[i];
        return `Each block starts by normalizing every position's stream to mean 0 and variance 1, then applying
        learned gain <b>γ</b> and bias <b>β</b>:
        <div class="formula">LN(x) = γ ⊙ (x − μ)/√(σ² + ε) + β,  ε = 1e−5</div>
        Position ${i}: μ = ${mono(fmt(c.trace.embed[i].reduce((a, b) => a + b, 0) / 16))},
        σ = ${mono(fmt(Math.sqrt(c.trace.embed[i].reduce((a, v) => a + (v - c.trace.embed[i].reduce((a2, b) => a2 + b, 0) / 16) ** 2, 0) / 16)))}
        → after LN: mean ${mono(fmt(ln.reduce((a, b) => a + b, 0) / 16))}.
        <div class="note">LayerNorm keeps activations in a stable range no matter how large the residual
        stream gets — one reason deep Transformers train at all.</div>`;
      },
    },
    {
      stage: 'b0.qkv', title: 'Block 1 · Queries, Keys, Values',
      explain(c) {
        const i = c.selPos, b = c.trace.blocks[0];
        return `The normalized stream is projected by three learned matrices:
        <div class="formula">Q = x̂·W<sub>q</sub>&nbsp;&nbsp;K = x̂·W<sub>k</sub>&nbsp;&nbsp;V = x̂·W<sub>v</sub></div>
        <b>Q</b> ("what am I looking for?"), <b>K</b> ("what do I contain?"), <b>V</b> ("what I'll hand over if
        you attend to me"). Each 16-dim vector splits into <b>2 heads</b> of 8 dims — two independent
        attention patterns in parallel: head 0 uses dims 0–7, head 1 dims 8–15.
        <div class="note">For ${emoji(c.ids[i])} at position ${i}:
        ‖Q‖ = ${mono(fmt(norm(b.q[i])))}, ‖K‖ = ${mono(fmt(norm(b.k[i])))}, ‖V‖ = ${mono(fmt(norm(b.v[i])))}.</div>`;
      },
    },
    {
      stage: 'b0.attn', title: 'Block 1 · Attention — the heart',
      explain(c) {
        const i = c.selPos, l = 0, h = c.head, b = c.trace.blocks[l];
        const row = b.attn[h][i];
        let best = 0; for (let j = 1; j < row.length; j++) if (row[j] > row[best]) best = j;
        const s = b.scores[h][i];
        return `Position ${i}'s query is compared with <b>every key at or before it</b> — the <b>causal mask</b>
        hides the future, which is exactly what makes this <i>decoder-only</i>:
        <div class="formula">score[i][j] = Q<sub>i</sub>·K<sub>j</sub> / √8&nbsp;&nbsp;(j ≤ i)
        &nbsp;&nbsp;→&nbsp;&nbsp; A[i][·] = softmax(score)</div>
        Head ${h} for ${emoji(c.ids[i])}: scores over j≤${i} range
        ${mono(fmt(Math.min(...s)))} … ${mono(fmt(Math.max(...s)))};
        after softmax the winner is position <b>${best}</b> ${emoji(c.ids[best])} with
        <b>${pct(row[best])}</b> of the attention${best === i - 1 ? ' — <b>the previous token</b>. This head learned precisely the “look one step back” skill this language demands.' : best === i ? ' — itself.' : '.'}
        The output is the V-vectors weighted by A, heads concatenated, then projected by W<sub>o</sub>.
        <div class="note">Switch heads below or in the Attention inspector — and try editing the sequence:
        the “previous token” spike follows the edit, because it comes from the weights, not from the data.</div>`;
      },
    },
    {
      stage: 'b0.add', title: 'Block 1 · Residual add',
      explain(c) {
        const i = c.selPos, b = c.trace.blocks[0];
        const before = c.trace.embed[i], after = b.x1[i];
        let delta = 0; for (let d = 0; d < 16; d++) delta += (after[d] - before[d]) ** 2;
        return `Attention never replaces the stream — it <b>adds</b> its findings:
        <div class="formula">X¹′ = X⁰ + Attn(X⁰)</div>
        Position ${i}: ‖X⁰‖ = ${mono(fmt(norm(before)))} → ‖X¹′‖ = ${mono(fmt(norm(after)))};
        the attention update moved the vector by ${mono(fmt(Math.sqrt(delta)))}.
        <div class="note">Residual connections are the backbone of every Transformer: each layer <i>refines</i>
        the stream instead of overwriting it, letting information survive all the way to the output.</div>`;
      },
    },
    {
      stage: 'b0.mlp', title: 'Block 1 · MLP + residual',
      explain(c) {
        const i = c.selPos, b = c.trace.blocks[0];
        const hid = b.mlpAct[i];
        const active = hid.filter(v => v > 0.5).length;
        const mx = Math.max(...hid);
        return `Each position now runs a small two-layer network <i>independently</i>:
        <div class="formula">MLP(x) = GELU(x·W₁ + b₁)·W₂ + b₂&nbsp;&nbsp;(16 → 64 → 16)</div>
        <b>GELU</b> is a smooth gate: small values pass through nearly linearly, large ones are damped.
        Attention <i>moves</i> information between positions; the MLP <i>computes</i> with what arrived —
        in this model it implements much of the “previous token + 1” mapping.
        <div class="note">Position ${i}: ${active}/64 hidden units above 0.5, peak activation
        ${mono(fmt(mx))}. Then X¹ = X¹′ + MLP(X¹′) finishes the block.</div>`;
      },
    },
    {
      stage: 'b1.ln', title: 'Block 2 · LayerNorm',
      explain(c) { return `Block 2 replays the same recipe with its own weights. LayerNorm first:
        mean ≈ 0, variance ≈ 1 per position, then γ, β.`; },
    },
    {
      stage: 'b1.qkv', title: 'Block 2 · Queries, Keys, Values',
      explain(c) {
        const b = c.trace.blocks[1], i = c.selPos;
        return `Fresh matrices W<sub>q</sub>, W<sub>k</sub>, W<sub>v</sub> produce block 2's own Q, K, V.
        ‖Q‖ = ${mono(fmt(norm(b.q[i])))}, ‖K‖ = ${mono(fmt(norm(b.k[i])))}, ‖V‖ = ${mono(fmt(norm(b.v[i])))} for position ${i}.`;
      },
    },
    {
      stage: 'b1.attn', title: 'Block 2 · Attention',
      explain(c) {
        const i = c.selPos, h = c.head, b = c.trace.blocks[1];
        const row = b.attn[h][i];
        let self = row[row.length - 1];
        return `Second-layer attention. In this trained model, head ${h} at position ${i} keeps
        <b>${pct(row[row.length - 1])}</b> on the current token and spreads the rest softly over earlier ones —
        the decisive “look one back” already happened in block 1, so block 2 mostly <i>re-mixes</i> what
        block 1 wrote.
        <div class="note">Compare the two layers in the Attention inspector: layer 0 = sharp offset −1,
        layer 1 = soft blend. Real LLMs stack dozens of layers building up circuits this way.</div>`;
      },
    },
    {
      stage: 'b1.add', title: 'Block 2 · Residual add',
      explain(c) { return `X²′ = X¹ + Attn₂(X¹). The stream is refined once more before the final MLP.`; },
    },
    {
      stage: 'b1.mlp', title: 'Block 2 · MLP + residual → X²',
      explain(c) { return `The second MLP polishes the representation, and
        <div class="formula">X² = X²′ + MLP₂(X²′)</div>
        closes block 2. With only two layers, most of the useful circuit lives in block 1 — you can see
        it in how much less the block-2 attention update changes the stream.`; },
    },
    {
      stage: 'lnf', title: 'Final LayerNorm',
      explain(c) { return `Before unembedding, one more LayerNorm puts the stream into a canonical scale —
        the same trick that stabilized the blocks now stabilizes the logits.`; },
    },
    {
      stage: 'out', title: 'Unembed → logits',
      explain(c) {
        const lg = c.trace.logits[c.T - 1];
        const top = topK(lg, 3).map(([i, v]) => `${emoji(i)} ${mono(fmt(v, 2))}`).join(', ');
        return `<div class="formula">logits = LN(X<sub>last</sub>)·W<sub>u</sub>ᵀ&nbsp;&nbsp;→&nbsp;&nbsp;16 raw scores</div>
        One score per vocabulary token — unnormalized. Last position reads:
        top-3 = ${top}.
        <div class="note">Only the <b>last</b> position's logits are needed for “predict the next token” —
        but during training every position produced one, and all of them were supervised.</div>`;
      },
    },
    {
      stage: 'out', title: 'Softmax → next-token prediction',
      explain(c) {
        const p = c.trace.probs[c.T - 1];
        let pred = 0; for (let i = 1; i < p.length; i++) if (p[i] > p[pred]) pred = i;
        const trueNext = c.T >= 2 ? (c.ids[c.T - 2] + 1) % 16 : null;
        return `<div class="formula">P(next) = softmax(logits)</div>
        The model predicts <b>${emoji(pred)} (t${pred})</b> with ${mono(pct(p[pred]))} confidence.
        ${trueNext != null ? `The toy language's rule says the true next token is
        <b>${emoji(trueNext)} (t${trueNext})</b> — “one back + 1”${pred === trueNext ? ', and the model nails it. ✅' : ' — the model disagrees here; edit the sequence and watch the attention adjust.'}` : 'With a single token there is no “one step back” — the model can only guess.'}
        <div class="note"><b>Now play:</b> append tokens with the KV cache on and watch only the newest
        column light up — cached positions are frozen ice-blue. Then open the <b>Verify</b> panel:
        cached and full recomputation agree to the last bit.</div>`;
      },
    },
  ];

  // ---------------------------------------------------------------- recompute
  function allSet(n) { const s = new Set(); for (let i = 0; i < n; i++) s.add(i); return s; }

  function partialToFull(res) {
    // build a full-shaped trace from a cached run that computed everything (prefill)
    const t = { ids: res.ids, T: res.T, fromCache: true,
      tokEmb: [], posEmb: [], embed: [], blocks: [], lnf: [], logits: [], probs: [] };
    for (let i = 0; i < res.T; i++) {
      const te = Array.from(model.tokEmb.subarray(res.ids[i] * model.D, (res.ids[i] + 1) * model.D));
      const pe = Array.from(model.posEmb.subarray(i * model.D, (i + 1) * model.D));
      const e = new Array(model.D);
      for (let d = 0; d < model.D; d++) e[d] = te[d] + pe[d];
      t.tokEmb.push(te); t.posEmb.push(pe); t.embed.push(e);
    }
    for (let l = 0; l < model.L; l++) {
      const rb = res.blocks[l];
      t.blocks.push({
        ln1: [], q: [], k: [], v: [], scores: rb.scores && [], attn: rb.attn && [],
        headOut: [], attnConcat: [], attnProj: [], x1: [], ln2: [],
        mlpHidden: [], mlpAct: [], mlpOut: [], x2: [], x2_in: [],
      });
      const tb = t.blocks[l];
      for (let i = 0; i < res.T; i++) {
        tb.ln1.push(rb.ln1[i]); tb.q.push(rb.q[i]); tb.k.push(rb.k[i]); tb.v.push(rb.v[i]);
        tb.headOut.push(rb.headOut[i]); tb.attnConcat.push(rb.attnConcat[i]);
        tb.attnProj.push(rb.attnProj[i]); tb.x1.push(rb.x1[i]); tb.ln2.push(rb.ln2[i]);
        tb.mlpHidden.push(rb.mlpHidden[i]); tb.mlpAct.push(rb.mlpAct[i]);
        tb.mlpOut.push(rb.mlpOut[i]); tb.x2.push(rb.x2[i]);
        tb.x2_in.push(l === 0 ? t.embed[i] : t.blocks[l - 1].x2[i]);
      }
      for (let h = 0; h < model.H; h++) {
        tb.scores[h] = []; tb.attn[h] = [];
        for (let i = 0; i < res.T; i++) {
          tb.scores[h].push(rb.scores[i] ? rb.scores[i][h] : null);
          tb.attn[h].push(rb.attn[i] ? rb.attn[i][h] : null);
        }
      }
    }
    for (let i = 0; i < res.T; i++) { t.lnf.push(res.lnf[i]); t.logits.push(res.logits[i]); t.probs.push(res.probs[i]); }
    return t;
  }

  function mergePartial(prev, res) {
    // copy the fresh rows into the previous full trace (cached positions unchanged)
    for (let l = 0; l < model.L; l++) {
      const rb = res.blocks[l], pb = prev.blocks[l];
      for (const i of res.fresh) {
        pb.ln1[i] = rb.ln1[i]; pb.q[i] = rb.q[i]; pb.k[i] = rb.k[i]; pb.v[i] = rb.v[i];
        pb.headOut[i] = rb.headOut[i]; pb.attnConcat[i] = rb.attnConcat[i];
        pb.attnProj[i] = rb.attnProj[i]; pb.x1[i] = rb.x1[i]; pb.ln2[i] = rb.ln2[i];
        pb.mlpHidden[i] = rb.mlpHidden[i]; pb.mlpAct[i] = rb.mlpAct[i];
        pb.mlpOut[i] = rb.mlpOut[i]; pb.x2[i] = rb.x2[i];
        for (let h = 0; h < model.H; h++) {
          pb.scores[h][i] = rb.scores[i] ? rb.scores[i][h] : null;
          pb.attn[h][i] = res.attnFull[l][h][i] ? res.attnFull[l][h][i].slice() : null;
        }
      }
    }
    for (const i of res.fresh) {
      prev.lnf[i] = res.lnf[i]; prev.logits[i] = res.logits[i]; prev.probs[i] = res.probs[i];
    }
    prev.T = res.T; prev.ids = res.ids.slice();
  }

  function recompute(kind) {
    const ids = state.ids;
    if (!state.kvOn) {
      state.trace = TM.forwardFull(model, ids);
      state.fresh = allSet(ids.length); state.frozen = new Set();
      TM.resetCache(state.cache);
      state.cacheStatus = kind === 'edit'
        ? 'no cache — recomputed all positions'
        : 'no cache — full forward every time';
    } else if (state.cache.pos === ids.length - 1 && TM.cacheMatches(state.cache, ids)) {
      try {
        const res = TM.forwardCached(model, ids, state.cache);
        mergePartial(state.trace, res);
        state.fresh = new Set(res.fresh);
        state.frozen = new Set(); for (let i = 0; i < res.cachedCount; i++) state.frozen.add(i);
        state.cacheStatus = `⚡ KV cache: reused ${res.cachedCount} cached rows, computed ${res.fresh.length} new`;
      } catch (err) {
        console.warn('cached path failed, falling back to full recompute:', err);
        TM.resetCache(state.cache);
        state.trace = TM.forwardFull(model, ids);
        state.fresh = allSet(ids.length); state.frozen = new Set();
        state.cacheStatus = 'fallback: full recompute';
      }
    } else {
      TM.resetCache(state.cache);
      const res = TM.forwardCached(model, ids, state.cache);
      state.trace = partialToFull(res);
      state.fresh = new Set(res.fresh); state.frozen = new Set();
      state.cacheStatus = ids.length === 1
        ? 'KV cache: computed the first position'
        : `KV cache prefill: computed all ${ids.length} positions`;
    }
    pushTraceToViz();
    renderAll();
  }

  // ---------------------------------------------------------------- viz sync
  function stageIdx(key) { return STAGE_KEYS.indexOf(key); }

  function pushTraceToViz() {
    const t = state.trace, N = t.T;
    if (VIZ.stageCount() === 0 || vizN !== N || vizIdsChanged()) rebuildViz();
    const style = { freshRows: state.fresh, frozenRows: state.frozen };
    const st = i => VIZ.stages[i];
    st(0).slabs[0].setValues(t.tokEmb, {});
    st(0).slabs[1].setValues(t.posEmb, {});
    st(0).slabs[2].setValues(t.embed, style);
    st(1).slabs[0].setValues(t.blocks[0].ln1, style);
    st(2).slabs[0].setValues(t.blocks[0].q, style);
    st(2).slabs[1].setValues(t.blocks[0].k, style);
    st(2).slabs[2].setValues(t.blocks[0].v, style);
    st(3).slabs[0].setValues(t.blocks[0].attnConcat, style);
    st(3).slabs[1].setValues(t.blocks[0].attnProj, style);
    st(4).slabs[0].setValues(t.blocks[0].x1, style);
    st(5).slabs[0].setValues(t.blocks[0].mlpAct, style);
    st(5).slabs[1].setValues(t.blocks[0].mlpOut, style);
    st(5).slabs[2].setValues(t.blocks[0].x2, style);
    st(6).slabs[0].setValues(t.blocks[1].ln1, style);
    st(7).slabs[0].setValues(t.blocks[1].q, style);
    st(7).slabs[1].setValues(t.blocks[1].k, style);
    st(7).slabs[2].setValues(t.blocks[1].v, style);
    st(8).slabs[0].setValues(t.blocks[1].attnConcat, style);
    st(8).slabs[1].setValues(t.blocks[1].attnProj, style);
    st(9).slabs[0].setValues(t.blocks[1].x1, style);
    st(10).slabs[0].setValues(t.blocks[1].mlpAct, style);
    st(10).slabs[1].setValues(t.blocks[1].mlpOut, style);
    st(10).slabs[2].setValues(t.blocks[1].x2, style);
    st(11).slabs[0].setValues(t.lnf, {});
    const bars = VIZ.bars();
    bars[0].setValues(t.logits[N - 1], null);
    let am = 0; for (let i = 1; i < model.V; i++) if (t.probs[N - 1][i] > t.probs[N - 1][am]) am = i;
    bars[1].setValues(t.probs[N - 1], am);
    updateArcs();
    updateSelectionMarkers();
  }

  let vizN = -1, vizIdsSig = '';
  function vizIdsChanged() {
    const sig = state.ids.join(',');
    if (sig !== vizIdsSig) { vizIdsSig = sig; return true; }
    return false;
  }
  function rebuildViz() {
    vizN = state.ids.length;
    VIZ.buildAll(state.ids, { L: model.L, V: model.V });
    vizIdsSig = state.ids.join(',');
  }

  function updateArcs() {
    const stp = STEPS[state.step];
    if (!stp.stage.endsWith('.attn')) {
      // no arcs outside attention stages — clear them
      VIZ.setArcs(-1, 0, null, null);
      return;
    }
    const layer = stp.stage.startsWith('b0') ? 0 : 1;
    VIZ.setArcs(layer, state.head, state.trace.blocks[layer].attn[state.head], state.selPos, { showRing: true });
  }

  function updateSelectionMarkers() {
    const i = state.step, st = VIZ.stages[i];
    if (!st) return;
    st.slabs.forEach((s, si) => {
      if (s.transposed) s.highlightRow(null);
      else s.highlightRow(state.selPos);
    });
  }

  // ---------------------------------------------------------------- rendering (panels)
  const $ = id => document.getElementById(id);

  function renderAll() {
    renderStepList();
    renderExplanation();
    renderSeqBar();
    renderCacheBadge();
    renderInspector();
  }

  function renderStepList() {
    const el = $('stepList');
    el.innerHTML = STEPS.map((s, i) =>
      `<div class="step ${i === state.step ? 'cur' : i < state.step ? 'done' : ''}" data-i="${i}">
         <div class="step-dot">${i < state.step ? '✓' : i + 1}</div><div>${s.title}</div></div>`).join('');
    el.querySelectorAll('.step').forEach(d =>
      d.addEventListener('click', () => gotoStep(+d.dataset.i)));
  }

  function renderExplanation() {
    const stp = STEPS[state.step];
    $('stepTitle').textContent = `Step ${state.step + 1}/${STEPS.length} — ${stp.title}`;
    $('stepBody').innerHTML = stp.explain({
      trace: state.trace, ids: state.ids, selPos: state.selPos, head: state.head,
      T: state.ids.length,
    });
    $('stepPos').textContent = (state.step + 1) + ' / ' + STEPS.length;
    $('btnPrev').disabled = state.step === 0;
    $('btnNext').disabled = state.step === STEPS.length - 1;
  }

  function renderSeqBar() {
    const bar = $('seqChips');
    bar.innerHTML = '';
    state.ids.forEach((id, i) => {
      const t = TOK[id];
      const chip = document.createElement('div');
      chip.className = 'chip' + (i === state.selPos ? ' sel' : '');
      chip.style.setProperty('--h', t.hue);
      chip.innerHTML = `<span class="ce">${t.emoji}</span><span class="ci">t${t.id}</span><span class="cp">${i}</span>`;
      chip.title = `${t.name} (token ${t.id}) at position ${i} — click to edit`;
      chip.addEventListener('click', e => openPalette(e, i));
      bar.appendChild(chip);
    });
    $('seqLen').textContent = `${state.ids.length}/${model.CTX}`;
    $('btnAppend').disabled = state.ids.length >= model.CTX;
  }

  function renderCacheBadge() {
    $('cacheState').innerHTML = state.kvOn
      ? `<span class="kvs kvs-on">KV cache ON</span>` : `<span class="kvs kvs-off">KV cache OFF</span>`;
    $('cacheStatus').textContent = state.cacheStatus;
  }

  // palette popover for editing a chip
  let paletteEl = null, paletteFor = -1;
  function openPalette(ev, pos) {
    closePalette();
    paletteFor = pos;
    paletteEl = document.createElement('div');
    paletteEl.className = 'palette';
    TOK.forEach(t => {
      const b = document.createElement('button');
      b.className = 'pal-btn'; b.style.setProperty('--h', t.hue);
      b.innerHTML = `<span>${t.emoji}</span><span class="pi">${t.id}</span>`;
      b.title = t.name;
      b.addEventListener('click', () => {
        if (state.ids[paletteFor] !== t.id) {
          state.ids[paletteFor] = t.id;
          closePalette();
          recompute('edit');
        } else closePalette();
      });
      paletteEl.appendChild(b);
    });
    document.body.appendChild(paletteEl);
    const r = ev.currentTarget.getBoundingClientRect();
    paletteEl.style.left = Math.min(window.innerWidth - 260, r.left) + 'px';
    paletteEl.style.top = (r.top - 130) + 'px';
    setTimeout(() => document.addEventListener('pointerdown', paletteAway), 0);
  }
  function paletteAway(e) { if (paletteEl && !paletteEl.contains(e.target)) closePalette(); }
  function closePalette() {
    if (paletteEl) { paletteEl.remove(); paletteEl = null; }
    document.removeEventListener('pointerdown', paletteAway);
  }

  // ---------------------------------------------------------------- inspector
  function renderInspector() {
    const tab = document.querySelector('.tab.active').dataset.tab;
    $('tab-attention').style.display = tab === 'attention' ? '' : 'none';
    $('tab-values').style.display = tab === 'values' ? '' : 'none';
    $('tab-weights').style.display = tab === 'weights' ? '' : 'none';
    $('tab-verify').style.display = tab === 'verify' ? '' : 'none';
    if (tab === 'attention') renderAttentionTab();
    if (tab === 'values') renderValuesTab();
    if (tab === 'weights') renderWeightsTab();
    if (tab === 'verify') renderVerifyTab();
  }

  function vecTable(vec, labels, digits, cls) {
    return `<div class="vec ${cls || ''}">` + vec.map((v, d) =>
      `<div class="vec-cell" title="${d}: ${v}"><span class="vec-d">${labels ? labels[d] : d}</span><span>${fmt(v, digits)}</span></div>`).join('') + '</div>';
  }

  function renderValuesTab() {
    const stp = STEPS[state.step], i = state.selPos, t = state.trace;
    let title = '', body = '', note = '';
    const key = stp.stage;
    if (key === 'embed') {
      title = `Residual stream X⁰ at position ${i} (${TOK[t.ids[i]].name})`;
      body = rowTriple('E[' + t.ids[i] + ']', t.tokEmb[i], 'P[' + i + ']', t.posEmb[i], 'X⁰[' + i + ']', t.embed[i]);
      note = 'E row (token) + P row (position), added dimension-wise.';
    } else if (key.endsWith('.ln')) {
      const l = key.startsWith('b0') ? 0 : 1;
      const b = t.blocks[l];
      const inp = l === 0 ? t.embed[i] : t.blocks[0].x2[i];
      title = `LayerNorm ${l + 1} at position ${i}`;
      body = rowTriple('before', inp, 'after', b.ln1[i]);
      const mu = inp.reduce((a, x) => a + x, 0) / 16;
      const vr = inp.reduce((a, x) => a + (x - mu) ** 2, 0) / 16;
      note = `mean ${fmt(mu)} → 0 · std ${fmt(Math.sqrt(vr))} → 1 (then γ, β)`;
    } else if (key.endsWith('.qkv')) {
      const l = key.startsWith('b0') ? 0 : 1, b = t.blocks[l];
      title = `Q · K · V at position ${i} — head 0 = dims 0–7, head 1 = dims 8–15`;
      body = rowTriple('Q', b.q[i], 'K', b.k[i], 'V', b.v[i]);
      note = `‖Q‖=${fmt(norm(b.q[i]))} ‖K‖=${fmt(norm(b.k[i]))} ‖V‖=${fmt(norm(b.v[i]))}`;
    } else if (key.endsWith('.attn')) {
      const l = key.startsWith('b0') ? 0 : 1, b = t.blocks[l], h = state.head;
      const row = b.attn[h][i];
      title = `Attention, layer ${l + 1} head ${h} — query row ${i} (${TOK[t.ids[i]].name})`;
      body = `
        <div class="ins-sub">raw scores Q·K/√8 (j ≤ ${i})</div>
        ${scoreRow(b.scores[h][i], t.ids)}
        <div class="ins-sub">softmax → attention weights (Σ = ${(row.reduce((a, b) => a + b, 0)).toFixed(6)})</div>
        ${scoreRow(row, t.ids)}
        <div class="ins-sub">weighted sum of V (this row out, dims ${h * 8}–${h * 8 + 7} = head ${h})</div>
        ${vecTable(b.headOut[i][h], null, 3)}`;
      note = 'Arcs in the 3D view draw exactly these weights.';
    } else if (key.endsWith('.add')) {
      const l = key.startsWith('b0') ? 0 : 1, b = t.blocks[l];
      const before = l === 0 ? t.embed[i] : t.blocks[0].x2[i];
      title = `Residual add at position ${i} (block ${l + 1})`;
      body = rowTriple('stream in', before, 'attention update', b.attnProj[i], 'stream out', b.x1[i]);
      note = 'stream out = stream in + attention update (dimension-wise)';
    } else if (key.endsWith('.mlp')) {
      const l = key.startsWith('b0') ? 0 : 1, b = t.blocks[l];
      title = `MLP at position ${i} (block ${l + 1}) — 64 hidden units after GELU`;
      body = `${vecTable(b.mlpAct[i], null, 3, 'wrap')}
        <div class="ins-sub">MLP output (16)</div>${vecTable(b.mlpOut[i], null, 3)}
        <div class="ins-sub">stream out x${l + 2 === 1 ? '¹' : '²'}</div>${vecTable(b.x2[i], null, 3)}`;
      note = 'hidden = GELU(LN(x)·W₁+b₁); each position computed independently.';
    } else if (key === 'lnf') {
      title = `Final LayerNorm at position ${i}`;
      body = rowTriple('before', t.blocks[1].x2[i], 'after', t.lnf[i]);
      note = 'the same recipe as inside the blocks — canonical scale for the output projection';
    } else if (key === 'out') {
      const lg = t.logits[t.T - 1], pr = t.probs[t.T - 1];
      title = 'Last position — what the model predicts next';
      body = `<div class="ins-sub">logits (16)</div>${vecTable(lg, null, 2)}
        <div class="ins-sub">probabilities (16)</div>
        <div class="prob-list">${topK(pr, 16).map(([tok, p]) =>
          `<div class="prob-row"><span class="pe">${TOK[tok].emoji}</span>
           <span class="pn">t${tok}</span>
           <div class="pbar"><div style="width:${(p * 100).toFixed(1)}%"></div></div>
           <span class="pv mono">${pct(p)}</span></div>`).join('')}</div>`;
      note = 'Sorted by probability. This is the full next-token distribution.';
    }
    $('valuesTitle').textContent = title;
    $('valuesBody').innerHTML = body;
    $('valuesNote').textContent = note;
  }

  function scoreRow(row, ids) {
    return `<div class="score-row">` + row.map((v, j) =>
      `<div class="score-cell" title="j=${j}: ${v}"><span class="sd">${j}</span><span class="sv" style="opacity:${0.35 + 0.65 * Math.min(1, v)}">${fmt(v, 3)}</span></div>`).join('') + '</div>';
  }

  function rowTriple(l1, v1, l2, v2, l3, v3) {
    const block = (l, v, hl) => `<div class="rowt ${hl ? 'hl' : ''}"><div class="rowt-l">${l}</div>${vecTable(v, null, 3)}</div>`;
    return block(l1, v1) + block(l2, v2, true) + (l3 ? block(l3, v3) : '');
  }

  // ----- attention tab
  let attCanvasCtx = null;
  function renderAttentionTab() {
    const sel = $('attLayerHead');
    sel.innerHTML = [0, 1].flatMap(l => [0, 1].map(h =>
      `<button class="lh-btn ${(state.attL === l && state.head === h) ? 'on' : ''}" data-l="${l}" data-h="${h}">
        L${l + 1}·H${h + 1}</button>`)).join('');
    sel.querySelectorAll('.lh-btn').forEach(b => b.addEventListener('click', () => {
      state.attL = +b.dataset.l; state.head = +b.dataset.h;
      renderAttentionTab(); updateArcs();
    }));
    drawAttnHeatmap(state.attL, state.head);
    const i = Math.min(state.selPos, state.trace.T - 1);
    const row = state.trace.blocks[state.attL].attn[state.head][i];
    let best = 0; for (let j = 1; j < row.length; j++) if (row[j] > row[best]) best = j;
    $('attSummary').innerHTML =
      `Layer ${state.attL + 1}, head ${state.head}: ${TOK[state.trace.ids[i]].emoji} at position ${i}
       attends most to <b>position ${best}</b> ${TOK[state.trace.ids[best]].emoji} with
       <b>${pct(row[best])}</b>${best === i - 1 ? ' — the previous token: the circuit this toy model was trained to find.' : '.'}`;
  }

  function drawAttnHeatmap(layer, head) {
    const cv = $('attnCanvas');
    const ctx = cv.getContext('2d');
    const T = state.trace.T, cell = Math.min(26, Math.floor(300 / T));
    cv.width = T * cell + 58; cv.height = T * cell + 58;
    ctx.fillStyle = '#0a101d'; ctx.fillRect(0, 0, cv.width, cv.height);
    ctx.font = '11px ui-monospace, monospace'; ctx.textBaseline = 'middle';
    for (let i = 0; i < T; i++) {
      ctx.fillStyle = '#93a7c4';
      ctx.textAlign = 'right';
      ctx.fillText(i, 46, 58 + i * cell + cell / 2);
      const te = TOK[state.trace.ids[i]].emoji;
      ctx.fillText(te, 58 + i * cell + cell / 2, 46);
    }
    const rows = state.trace.blocks[layer].attn[head];
    let vmax = 0; rows.forEach(r => r && r.forEach(v => vmax = Math.max(vmax, v)));
    for (let i = 0; i < T; i++) {
      const r = rows[i]; if (!r) continue;
      for (let j = 0; j < r.length; j++) {
        const t = r[j] / (vmax || 1);
        const c = Math.round(8 + t * 210), cc = Math.round(20 + t * 150);
        ctx.fillStyle = `rgb(${c},${cc},${Math.round(60 + 40 * (1 - t))})`;
        ctx.fillRect(58 + j * cell, 58 + i * cell, cell - 1.5, cell - 1.5);
        if (r[j] > 0.005) {
          ctx.fillStyle = t > 0.55 ? '#04121f' : '#cfe0f5';
          ctx.textAlign = 'center';
          ctx.fillText(r[j].toFixed(2), 58 + j * cell + cell / 2, 58 + i * cell + cell / 2 + 1);
        }
      }
      // causal mask cells
      for (let j = i + 1; j < T; j++) {
        ctx.fillStyle = 'rgba(255,255,255,0.03)';
        ctx.fillRect(58 + j * cell, 58 + i * cell, cell - 1.5, cell - 1.5);
      }
    }
    ctx.fillStyle = '#5c7396'; ctx.textAlign = 'left';
    ctx.fillText('query i ↓ · key j →', 58, 16);
    cv.onmousemove = e => {
      const r = cv.getBoundingClientRect();
      const x = Math.floor((e.clientX - r.left - 58) / cell), y = Math.floor((e.clientY - r.top - 58) / cell);
      if (x >= 0 && y >= 0 && x <= y && x < T && y < T && rows[y]) {
        $('attHover').innerHTML = `A[${y}][${x}] = <b>${rows[y][x].toFixed(4)}</b> — position ${y} (${TOK[state.trace.ids[y]].emoji}) → position ${x} (${TOK[state.trace.ids[x]].emoji})`;
      } else $('attHover').textContent = 'hover a cell for the exact weight';
    };
  }

  // ----- weights tab
  const WEIGHT_KEYS = ['tok_emb', 'pos_emb', 'Wu',
    'block0.Wq', 'block0.Wk', 'block0.Wv', 'block0.Wo', 'block0.W1', 'block0.W2',
    'block1.Wq', 'block1.Wk', 'block1.Wv', 'block1.Wo', 'block1.W1', 'block1.W2'];
  function renderWeightsTab() {
    const sel = $('weightSel');
    if (!sel.options.length) {
      sel.innerHTML = WEIGHT_KEYS.map(k => `<option value="${k}">${k}</option>`).join('');
      sel.addEventListener('change', renderWeightsTab);
      $('weightCanvas').onmousemove = wHover;
    }
    const key = sel.value;
    const w = window.TINY_WEIGHTS.weights[key];
    const rows = w.length, cols = w[0].length;
    const cv = $('weightCanvas');
    const cell = Math.floor(280 / Math.max(rows, cols));
    cv.dataset.cell = cell;
    cv.width = cols * cell + 4; cv.height = rows * cell + 4;
    const ctx = cv.getContext('2d');
    ctx.fillStyle = '#0a101d'; ctx.fillRect(0, 0, cv.width, cv.height);
    let vmax = 1e-9;
    for (const r of w) for (const v of r) vmax = Math.max(vmax, Math.abs(v));
    for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) {
      const v = w[r][c], t = Math.abs(v) / vmax;
      ctx.fillStyle = v < 0
        ? `rgb(${Math.round(30 - 10 * t)},${Math.round(90 + 120 * t)},${Math.round(160 + 95 * t)})`
        : `rgb(${Math.round(160 + 95 * t)},${Math.round(90 + 70 * t)},${Math.round(40 - 10 * t)})`;
      ctx.fillRect(2 + c * cell, 2 + r * cell, cell - 0.5, cell - 0.5);
    }
    $('weightInfo').textContent = `shape ${rows}×${cols} · range ±${vmax.toFixed(3)} · row = input dim, col = output dim`;
  }
  function wHover(e) {
    const cv = $('weightCanvas'), r = cv.getBoundingClientRect();
    const key = $('weightSel').value, w = window.TINY_WEIGHTS.weights[key];
    const cell = cv.dataset.cell ? +cv.dataset.cell : cv.width / w[0].length;
    const c = Math.floor((e.clientX - r.left - 2) / cell);
    const rr = Math.floor((e.clientY - r.top - 2) / cell);
    if (c >= 0 && c < w[0].length && rr >= 0 && rr < w.length)
      $('weightInfo2').textContent = `${key}[${rr}][${c}] = ${w[rr][c].toFixed(5)}`;
  }

  // ----- verify tab
  function renderVerifyTab() {
    const v = state.lastVerify;
    $('verifyBtn').onclick = () => {
      $('verifyOut').innerHTML = '<span class="dim">running both paths…</span>';
      setTimeout(() => {
        const r = TM.verifyParity(model, state.ids);
        state.lastVerify = r;
        const d = r.maxAbsDiff;
        $('verifyOut').innerHTML = r.ok
          ? `<div class="verdict ok">✓ cached ≡ uncached — max |Δ| = ${d === 0 ? '0 (bitwise identical)' : d.toExponential(2)} over ${state.ids.length} positions × 16 logits+probs</div>`
          : `<div class="verdict bad">✗ mismatch: max |Δ| = ${d.toExponential(2)} at ${r.worstAt}</div>`;
      }, 30);
    };
    if (v) {
      $('verifyOut').innerHTML = v.ok
        ? `<div class="verdict ok">✓ max |Δ| = ${v.maxAbsDiff === 0 ? '0 (bitwise identical)' : v.maxAbsDiff.toExponential(2)}</div>`
        : `<div class="verdict bad">✗ max |Δ| = ${v.maxAbsDiff.toExponential(2)}</div>`;
    }
  }

  // ---------------------------------------------------------------- actions
  // one computation stage per step (steps 13 & 14 share the 'out' stage)
  const STEP2STAGE = ['embed', 'embed', 'b0.ln', 'b0.qkv', 'b0.attn', 'b0.add', 'b0.mlp',
                      'b1.ln', 'b1.qkv', 'b1.attn', 'b1.add', 'b1.mlp', 'lnf', 'out', 'out'];

  function gotoStep(i) {
    state.step = Math.max(0, Math.min(STEPS.length - 1, i));
    VIZ.setStageFocus(stageIdx(STEP2STAGE[state.step]), false);
    updateArcs();
    updateSelectionMarkers();
    renderAll();
    if (state.step === STEPS.length - 1) stopPlay();
  }

  function setSeq(ids, kind) {
    state.ids = ids;
    state.selPos = ids.length - 1;
    recompute(kind || 'edit');
    gotoStep(state.step); // re-focus camera in case stage list rebuilt
  }

  function appendToken(id) {
    if (state.ids.length >= model.CTX) return;
    const ids = state.ids.concat([id]);
    const wasAppend = ids.slice(0, -1).every((v, i) => v === state.ids[i]);
    state.selPos = ids.length - 1;
    state.ids = ids;
    recompute(wasAppend ? 'append' : 'edit');
  }

  function stopPlay() {
    state.playing = false;
    clearInterval(state.playTimer);
    $('btnPlay').textContent = '▶ play';
  }

  // ---------------------------------------------------------------- boot
  function boot() {
    VIZ.init(document.getElementById('canvas3d'), {
      onTokenPicked: pos => { state.selPos = pos; updateArcs(); updateSelectionMarkers(); renderAll(); },
    });
    rebuildViz();
    recompute('init');

    $('btnPrev').addEventListener('click', () => { stopPlay(); gotoStep(state.step - 1); });
    $('btnNext').addEventListener('click', () => { stopPlay(); gotoStep(state.step + 1); });
    $('btnPlay').addEventListener('click', () => {
      if (state.playing) { stopPlay(); return; }
      state.playing = true;
      $('btnPlay').textContent = '⏸ pause';
      state.playTimer = setInterval(() => {
        if (state.step >= STEPS.length - 1) stopPlay();
        else gotoStep(state.step + 1);
      }, 3400);
      gotoStep(state.step + 1);
    });

    document.querySelectorAll('.tab').forEach(t => t.addEventListener('click', () => {
      document.querySelectorAll('.tab').forEach(x => x.classList.remove('active'));
      t.classList.add('active');
      renderInspector();
    }));

    $('btnRandom').addEventListener('click', () => {
      const n = Math.min(model.CTX, 4 + Math.floor(Math.random() * 5));
      setSeq(Array.from({ length: n }, () => Math.floor(Math.random() * 16)), 'edit');
    });
    $('btnDemo').addEventListener('click', () => setSeq(window.TINY_DEMO.slice(), 'edit'));
    $('btnAppend').addEventListener('click', () => {
      const p = state.trace.probs[state.ids.length - 1];
      let am = 0; for (let i = 1; i < p.length; i++) if (p[i] > p[am]) am = i;
      appendToken(am);
    });
    $('btnPop').addEventListener('click', () => {
      if (state.ids.length <= 1) return;
      setSeq(state.ids.slice(0, -1), 'edit');
    });
    $('kvToggle').addEventListener('change', e => {
      state.kvOn = e.target.checked;
      recompute('toggle');
    });
    $('btnCacheInfo').addEventListener('click', () => {
      document.querySelector('.tab[data-tab="verify"]').click();
    });

    window.addEventListener('keydown', e => {
      if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;
      if (e.key === 'ArrowRight') { stopPlay(); gotoStep(state.step + 1); }
      if (e.key === 'ArrowLeft') { stopPlay(); gotoStep(state.step - 1); }
      if (e.key === ' ') { e.preventDefault(); $('btnPlay').click(); }
    });

    document.getElementById('introClose').addEventListener('click', () => {
      document.getElementById('intro').remove();
    });
  }

  window.addEventListener('DOMContentLoaded', boot);
  window.__app = state; // exposed for tests/debugging
})();
