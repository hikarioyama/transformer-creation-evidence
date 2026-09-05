/*
 * Viz3D — three.js scene for the tiny transformer.
 *
 * Visual language:
 *   · pipeline runs along -Z: each computation stage is an island of "slabs"
 *   · a slab renders a [positions × channels] tensor as a landscape of boxes
 *     (height ∝ |value|, cyan = negative, amber = positive)
 *   · attention stages show token posts with arcs whose thickness/opacity ∝ weight
 *   · the camera flies between stages as you step through the computation
 */
window.Viz3D = (function () {
  'use strict';

  const BG = 0x05070f;
  const STAGE_GAP = 15;
  const COL_SPACING = 1.05;      // token columns inside a slab
  const CHAN_SPACING = 0.62;     // channel rows inside a slab (along z)
  const CELL = 0.52;             // box footprint
  const MAX_H = 4.2;

  const NEG1 = new THREE.Color('#0b1526'), NEG2 = new THREE.Color('#38bdf8');
  const POS1 = new THREE.Color('#0b1526'), POS2 = new THREE.Color('#fb923c');
  const FROZEN = new THREE.Color('#5b6b85');
  const FRESHTINT = new THREE.Color('#ffffff');

  // ---------------------------------------------------------------- helpers
  function valueColor(v, vmax) {
    const t = Math.min(1, Math.abs(v) / (vmax || 1));
    const c = (v < 0 ? NEG1.clone().lerp(NEG2, Math.pow(t, 0.7))
                     : POS1.clone().lerp(POS2, Math.pow(t, 0.7)));
    c.multiplyScalar(0.4 + 0.6 * t);
    return c;
  }

  function makeTextSprite(draw, w, h, worldW, worldH) {
    const cv = document.createElement('canvas');
    cv.width = w; cv.height = h;
    draw(cv.getContext('2d'), w, h);
    const tex = new THREE.CanvasTexture(cv);
    tex.minFilter = THREE.LinearFilter;
    const mat = new THREE.SpriteMaterial({ map: tex, transparent: true, depthWrite: false });
    const sp = new THREE.Sprite(mat);
    sp.scale.set(worldW, worldH, 1);
    sp.userData.redraw = (fn) => {
      const ctx = cv.getContext('2d');
      ctx.clearRect(0, 0, w, h);
      fn(ctx, w, h);
      tex.needsUpdate = true;
    };
    return sp;
  }

  function labelSprite(line1, line2, worldW) {
    const sp = makeTextSprite((ctx, w, h) => drawLabel(ctx, w, h, line1, line2), 640, 160, worldW || 10, (worldW || 10) / 4);
    return sp;
  }
  function drawLabel(ctx, w, h, line1, line2) {
    ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
    ctx.font = '600 52px system-ui, sans-serif';
    ctx.fillStyle = '#dce8ff';
    ctx.fillText(line1, w / 2, line2 ? h * 0.34 : h * 0.5);
    if (line2) {
      ctx.font = '400 34px ui-monospace, monospace';
      ctx.fillStyle = '#7f95b5';
      ctx.fillText(line2, w / 2, h * 0.72);
    }
  }

  function disposeObject(root) {
    root.traverse(o => {
      if (o.geometry) o.geometry.dispose();
      if (o.material) {
        const mats = Array.isArray(o.material) ? o.material : [o.material];
        mats.forEach(m => { if (m.map) m.map.dispose(); m.dispose(); });
      }
    });
  }

  // ---------------------------------------------------------------- Slab
  class Slab {
    /*
     * opts: {rows (positions), cols (channels), transposed, colSpacing, chanSpacing,
     *        x, z, y0, name, sub, cellScale}
     * normal orientation: positions along X, channels along Z
     * transposed:         channels along X, positions along Z (MLP carpet)
     */
    constructor(opts) {
      this.rows = opts.rows; this.cols = opts.cols;
      this.transposed = !!opts.transposed;
      this.colSpacing = opts.colSpacing || (this.transposed ? 0.30 : COL_SPACING);
      this.chanSpacing = opts.chanSpacing || (this.transposed ? 1.15 : CHAN_SPACING);
      this.maxH = opts.maxH || MAX_H;
      this.vmax = 1;
      this.data = null;
      this._name = opts.name; this._sub = opts.sub;

      const n = this.rows * this.cols;
      const geo = new THREE.BoxGeometry(CELL, 1, CELL);
      const mat = new THREE.MeshBasicMaterial({ vertexColors: false });
      this.mesh = new THREE.InstancedMesh(geo, mat, n);
      this.mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
      for (let i = 0; i < n; i++) this.mesh.setColorAt(i, new THREE.Color(0x000000));

      // dark base plate
      const w = this.cols * this.colSpacing, d = this.rows * this.chanSpacing;
      const plate = new THREE.Mesh(
        new THREE.BoxGeometry(w + 1.4, 0.12, d + 1.4),
        new THREE.MeshBasicMaterial({ color: 0x0a1220, transparent: true, opacity: 0.85 }));
      plate.position.y = -0.10;

      this.label = labelSprite(opts.name, opts.sub, this.transposed ? 12 : 11);

      this.group = new THREE.Group();
      this.group.add(plate, this.mesh, this.label);
      this.group.position.set(opts.x || 0, opts.y0 || 0, opts.z || 0);

      // selection marker (glowing plane behind selected token column)
      const mkGeo = new THREE.PlaneGeometry(this.colSpacing * 0.98, 5.0);
      const mkMat = new THREE.MeshBasicMaterial({
        color: 0x67e8f9, transparent: true, opacity: 0.16, side: THREE.DoubleSide,
        depthWrite: false, blending: THREE.AdditiveBlending });
      this.marker = new THREE.Mesh(mkGeo, mkMat);
      this.marker.visible = false;
      this.group.add(this.marker);

      this._m4 = new THREE.Matrix4();
      this._zeroRows = null; this._frozenRows = null; this._freshRows = null;
    }

    setLabel(line1, line2) { this.label.userData.redraw((ctx, w, h) => drawLabel(ctx, w, h, line1, line2)); }

    /* data: array of rows (length rows) each length cols, or object {rowIdx: row}
       for partial updates (untouched rows keep old values). */
    setValues(data, style) {
      style = style || {};
      this._frozenRows = style.frozenRows || null;
      this._freshRows = style.freshRows || null;
      const full = new Array(this.rows);
      for (let r = 0; r < this.rows; r++)
        full[r] = (data && data[r] !== undefined) ? data[r] : (this.data ? this.data[r] : null);
      this.data = full;
      // per-slab value range (honest auto-scale, shown in the label)
      let vmax = 1e-9;
      for (let r = 0; r < this.rows; r++) {
        const row = full[r]; if (!row) continue;
        for (let c = 0; c < this.cols; c++) { const a = Math.abs(row[c]); if (a > vmax) vmax = a; }
      }
      this.vmax = vmax;

      const m = new THREE.Matrix4();
      for (let r = 0; r < this.rows; r++) {
        const row = full[r];
        for (let c = 0; c < this.cols; c++) {
          const idx = r * this.cols + c;
          let x, z;
          if (this.transposed) { x = (c - (this.cols - 1) / 2) * this.colSpacing; z = (r - (this.rows - 1) / 2) * this.chanSpacing; }
          else { x = (r - (this.rows - 1) / 2) * this.colSpacing; z = (c - (this.cols - 1) / 2) * this.chanSpacing; }
          let h = 0.08, col = new THREE.Color(0x101826);
          if (row) {
            const v = row[c];
            const frozen = this._frozenRows && this._frozenRows.has(r);
            const fresh = this._freshRows && this._freshRows.has(r);
            h = Math.max(0.08, Math.min(this.maxH, Math.abs(v) / this.vmax * this.maxH * 0.92 + 0.08));
            col = valueColor(v, this.vmax);
            if (frozen) col.lerp(FROZEN, 0.72).multiplyScalar(0.8);
            else if (fresh) col.lerp(FRESHTINT, 0.22);
          }
          m.makeScale(1, h, 1);
          m.setPosition(x, h / 2, z);
          this.mesh.setMatrixAt(idx, m);
          this.mesh.setColorAt(idx, col);
        }
      }
      this.mesh.instanceMatrix.needsUpdate = true;
      if (this.mesh.instanceColor) this.mesh.instanceColor.needsUpdate = true;
      // keep the label honest: show this tensor's actual numeric range
      this.setLabel(this._name, `${this._sub} · range ±${vmax.toFixed(2)}`);
    }

    highlightRow(r) {
      if (r == null) { this.marker.visible = false; return; }
      this.marker.visible = true;
      const x = this.transposed ? 0 : (r - (this.rows - 1) / 2) * this.colSpacing;
      this.marker.position.set(x, 2.0, this.transposed ? (r - (this.rows - 1) / 2) * this.chanSpacing : 0);
    }

    setOpacity(op) {
      this.mesh.material.opacity = op; this.mesh.material.transparent = op < 1;
      this.mesh.material.depthWrite = op >= 0.99;
      this.label.material.opacity = Math.min(1, op * 1.4);
      this.marker.material.opacity = 0.16 * op;
      this.mesh.count = op <= 0.01 ? 0 : this.rows * this.cols;
    }

    dispose() { disposeObject(this.group); }
  }

  // ---------------------------------------------------------------- bars (logits/probs)
  class BarRow {
    /* n bars along X; each bar: height ∝ value; emoji sprite + value text above */
    constructor(opts) {
      this.n = opts.n; this.spacing = opts.spacing || 0.8;
      this.maxH = opts.maxH || 5;
      this.mode = opts.mode; // 'logit' | 'prob'
      this.group = new THREE.Group();
      this.group.position.set(opts.x || 0, opts.y0 || 0, opts.z || 0);
      this.bars = []; this.valueSprites = []; this.emojiSprites = [];
      const geo = new THREE.BoxGeometry(0.62, 1, 0.62);
      for (let i = 0; i < this.n; i++) {
        const mat = new THREE.MeshBasicMaterial({ color: 0x223 });
        const bar = new THREE.Mesh(geo, mat);
        const x = (i - (this.n - 1) / 2) * this.spacing;
        bar.position.x = x;
        this.group.add(bar); this.bars.push(bar);

        const vs = makeTextSprite((ctx, w, h) => {
          ctx.clearRect(0, 0, w, h);
        }, 128, 64, 1.35, 0.68);
        vs.position.set(x, 1.2, 0);
        this.group.add(vs); this.valueSprites.push(vs);

        const t = window.TINY_TOKENS[i];
        const es = makeTextSprite((ctx, w, h) => {
          ctx.font = '64px system-ui, "Apple Color Emoji", "Segoe UI Emoji", "Noto Color Emoji", sans-serif';
          ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
          ctx.fillText(t.emoji, w / 2, h / 2 + 4);
        }, 128, 128, 1.15, 1.15);
        es.position.set(x, -1.15, 0);
        this.group.add(es); this.emojiSprites.push(es);
      }
    }

    setValues(values, argmax) {
      let vmax = 1e-9, vmin = 0;
      for (const v of values) { vmax = Math.max(vmax, Math.abs(v)); if (this.mode === 'logit') vmin = Math.min(vmin, v); }
      for (let i = 0; i < this.n; i++) {
        const v = values[i];
        const bar = this.bars[i];
        let h;
        if (this.mode === 'logit') h = Math.max(0.06, Math.abs(v) / vmax * this.maxH * 0.9);
        else h = Math.max(0.06, v / Math.max(1e-9, Math.max(...values)) * this.maxH);
        bar.scale.set(1, h, 1);
        bar.position.y = (this.mode === 'logit' ? (v >= 0 ? h / 2 : -h / 2) : h / 2);
        let col;
        if (this.mode === 'logit') col = valueColor(v, vmax);
        else {
          const t = v / Math.max(1e-9, Math.max(...values));
          col = new THREE.Color('#133b2c').lerp(new THREE.Color(i === argmax ? '#fbbf24' : '#34d399'), Math.pow(t, 0.6));
        }
        bar.material.color.copy(col);
        const txt = this.mode === 'logit' ? v.toFixed(2) : (100 * v).toFixed(1) + '%';
        const big = (this.mode === 'prob' && i === argmax) || (this.mode === 'logit' && Math.abs(v) === Math.max(...values.map(Math.abs)));
        this.valueSprites[i].userData.redraw((ctx, w, h) => {
          ctx.font = (big ? '700 44px' : '500 36px') + ' ui-monospace, monospace';
          ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
          ctx.fillStyle = big ? (i === argmax ? '#fbbf24' : '#e8f2ff') : '#9db1cc';
          ctx.fillText(txt, w / 2, h / 2);
        });
        this.valueSprites[i].position.y = (this.mode === 'logit' ? (v >= 0 ? h + 0.45 : -h - 0.45) : h + 0.45);
        const dim = this.mode === 'prob' && i !== argmax && v < 0.004;
        this.emojiSprites[i].material.opacity = dim ? 0.35 : 1;
      }
    }

    setOpacity(op) {
      this.group.traverse(o => {
        if (o.material) { o.material.transparent = true; o.material.opacity = o.material._baseOp != null ? o.material._baseOp * op : op; }
      });
    }
  }

  // ---------------------------------------------------------------- module state
  let renderer, scene, camera, controls, raycaster, container;
  let rootGroup, stages = [], chips = [], posts = [], arcGroup, tween = null;
  let pickables = [], onTokenPicked = null, selectedPos = null;
  let stars;

  const ARC_COLORS = [new THREE.Color('#4fc3f7'), new THREE.Color('#ffb74d')];

  function valueToX(i, n, spacing) { return (i - (n - 1) / 2) * spacing; }

  // ---------------------------------------------------------------- init
  function init(dom, callbacks) {
    container = dom;
    onTokenPicked = callbacks.onTokenPicked || (() => {});

    renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setSize(dom.clientWidth, dom.clientHeight);
    renderer.setClearColor(BG);
    dom.appendChild(renderer.domElement);

    scene = new THREE.Scene();
    scene.fog = new THREE.Fog(BG, 34, 105);

    camera = new THREE.PerspectiveCamera(52, dom.clientWidth / dom.clientHeight, 0.1, 600);
    camera.position.set(0, 12, 20);

    controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true; controls.dampingFactor = 0.08;
    controls.maxPolarAngle = 1.48; controls.minDistance = 5; controls.maxDistance = 70;
    controls.target.set(0, 1, 0);

    // floor grid + stars
    const grid = new THREE.GridHelper(420, 84, 0x1c2940, 0x121b2c);
    grid.position.y = -2.02;
    scene.add(grid);
    const starGeo = new THREE.BufferGeometry();
    const sp = [];
    for (let i = 0; i < 450; i++) {
      sp.push((Math.random() - 0.5) * 320, Math.random() * 80 - 8, (Math.random() - 0.5) * 420 - 60);
    }
    starGeo.setAttribute('position', new THREE.Float32BufferAttribute(sp, 3));
    stars = new THREE.Points(starGeo, new THREE.PointsMaterial({
      color: 0x9db8e8, size: 0.55, transparent: true, opacity: 0.45, sizeAttenuation: true }));
    scene.add(stars);

    rootGroup = new THREE.Group();
    scene.add(rootGroup);

    raycaster = new THREE.Raycaster();
    bindPointer();

    window.addEventListener('resize', onResize);
    animate();
  }

  function onResize() {
    const w = container.clientWidth, h = container.clientHeight;
    camera.aspect = w / h; camera.updateProjectionMatrix();
    renderer.setSize(w, h);
  }

  // ---------------------------------------------------------------- build
  function buildAll(ids, meta) {
    // clear
    if (rootGroup.children.length) {
      rootGroup.traverse(o => { if (o.geometry) o.geometry.dispose(); });
      while (rootGroup.children.length) {
        const c = rootGroup.children.pop();
        disposeObject(c);
      }
    }
    stages = []; chips = []; posts = []; pickables = [];
    const N = ids.length;

    const slabDefs = [];  // {stage, factory}
    const S = (i) => -i * STAGE_GAP;

    // ---- stage 0: embeddings
    const embedStage = new THREE.Group(); embedStage.position.z = S(0);
    const slTok = new Slab({ rows: N, cols: 16, colSpacing: 0.62, chanSpacing: 0.42, x: -10.5, z: 0, name: 'E[token]', sub: 'token embedding row' });
    const slPos = new Slab({ rows: N, cols: 16, colSpacing: 0.62, chanSpacing: 0.42, x: 0, z: 0, name: 'P[pos]', sub: 'position embedding row' });
    const slX0  = new Slab({ rows: N, cols: 16, x: 11.5, z: 0, name: 'X⁰ = E + P', sub: 'residual stream, layer 0 input' });
    embedStage.add(slTok.group, slPos.group, slX0.group);
    // token chips
    for (let i = 0; i < N; i++) {
      const t = window.TINY_TOKENS[ids[i]];
      const chip = makeTextSprite((ctx, w, h) => {
        const r = 26;
        ctx.beginPath(); ctx.roundRect(6, 6, w - 12, h - 12, r);
        ctx.fillStyle = `hsl(${t.hue}, 45%, 16%, 0.92)`; ctx.fill();
        ctx.lineWidth = 5; ctx.strokeStyle = `hsl(${t.hue}, 85%, 68%)`; ctx.stroke();
        ctx.font = '84px system-ui, "Apple Color Emoji", "Segoe UI Emoji", "Noto Color Emoji", sans-serif';
        ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
        ctx.fillText(t.emoji, w / 2, h * 0.42);
        ctx.font = '700 34px ui-monospace, monospace';
        ctx.fillStyle = `hsl(${t.hue}, 80%, 78%)`;
        ctx.fillText('t' + t.id, w / 2, h * 0.82);
      }, 192, 224, 1.9, 2.22);
      chip.position.set(valueToX(i, N, COL_SPACING), 2.1, 6.4);
      chip.userData.pos = i;
      embedStage.add(chip); chips.push(chip); pickables.push(chip);
    }
    rootGroup.add(embedStage);
    stages.push({ group: embedStage, slabs: [slTok, slPos, slX0], key: 'embed' });

    // ---- per-block stages
    for (let b = 0; b < meta.L; b++) {
      const B = 1 + b * 5;
      // LN
      const lnStage = mkSimpleStage(S(B), `LayerNorm · block ${b + 1}`, [
        { rows: N, name: `LN(X${sup(b)})`, sub: 'γ·x̂+β, per position' }]);
      stages.push({ ...lnStage, key: `b${b}.ln` });
      // QKV
      const qkvStage = new THREE.Group(); qkvStage.position.z = S(B + 1);
      const slQ = new Slab({ rows: N, cols: 16, colSpacing: 0.62, chanSpacing: 0.42, x: -11, z: 0, name: 'Q = LN·Wq', sub: 'dims 0–7 head 0 · 8–15 head 1' });
      const slK = new Slab({ rows: N, cols: 16, colSpacing: 0.62, chanSpacing: 0.42, x: 0, z: 0, name: 'K = LN·Wk', sub: 'what each token offers' });
      const slV = new Slab({ rows: N, cols: 16, colSpacing: 0.62, chanSpacing: 0.42, x: 11, z: 0, name: 'V = LN·Wv', sub: 'what it hands over' });
      qkvStage.add(slQ.group, slK.group, slV.group);
      rootGroup.add(qkvStage);
      stages.push({ group: qkvStage, slabs: [slQ, slK, slV], key: `b${b}.qkv` });
      // attention
      const atStage = new THREE.Group(); atStage.position.z = S(B + 2);
      const postGroup = new THREE.Group(); postGroup.position.set(0, 0, 3.4);
      for (let i = 0; i < N; i++) {
        const t = window.TINY_TOKENS[ids[i]];
        const post = new THREE.Mesh(
          new THREE.CylinderGeometry(0.22, 0.3, 1.7, 12),
          new THREE.MeshBasicMaterial({ color: new THREE.Color(`hsl(${t.hue}, 60%, 45%)`) }));
        post.position.set(valueToX(i, N, 1.25), 0.85, 0);
        const es = makeTextSprite((ctx, w, h) => {
          ctx.beginPath(); ctx.arc(w / 2, h / 2, w / 2 - 8, 0, Math.PI * 2);
          ctx.fillStyle = `hsl(${t.hue}, 45%, 14%, 0.95)`; ctx.fill();
          ctx.lineWidth = 4; ctx.strokeStyle = `hsl(${t.hue}, 85%, 66%)`; ctx.stroke();
          ctx.font = '58px system-ui, "Apple Color Emoji", "Segoe UI Emoji", "Noto Color Emoji", sans-serif';
          ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
          ctx.fillText(t.emoji, w / 2, h / 2 + 2);
        }, 128, 128, 1.5, 1.5);
        es.position.set(valueToX(i, N, 1.25), 2.6, 0);
        es.userData.pos = i;
        postGroup.add(post, es); posts.push(es); pickables.push(es);
      }
      atStage.add(postGroup);
      const slConcat = new Slab({ rows: N, cols: 16, colSpacing: 0.62, chanSpacing: 0.42, x: -5.5, z: -2.5, name: 'Σ weights · V', sub: 'concat heads' });
      const slProj = new Slab({ rows: N, cols: 16, colSpacing: 0.62, chanSpacing: 0.42, x: 6.5, z: -2.5, name: '·Wo', sub: 'attention output' });
      atStage.add(slConcat.group, slProj.group);
      arcGroup = new THREE.Group(); atStage.add(arcGroup);
      rootGroup.add(atStage);
      stages.push({ group: atStage, slabs: [slConcat, slProj], key: `b${b}.attn`, arcLayer: b });
      // residual add
      const addStage = mkSimpleStage(S(B + 3), `Residual add · block ${b + 1}`, [
        { rows: N, name: `X${sup(b + 1)}′ = X${sup(b)} + Attn`, sub: 'residual stream grows' }]);
      stages.push({ ...addStage, key: `b${b}.add` });
      // MLP
      const mlpStage = new THREE.Group(); mlpStage.position.z = S(B + 4);
      const slH = new Slab({ rows: N, cols: 64, transposed: true, colSpacing: 0.26, chanSpacing: 1.0, x: 0, z: 0, name: 'MLP hidden (64)', sub: 'LN(x)·W1 → GELU', maxH: 2.6 });
      const slMO = new Slab({ rows: N, cols: 16, colSpacing: 0.62, chanSpacing: 0.42, x: -8.5, z: -6.5, name: '·W2 + b2', sub: 'back to 16 dims' });
      const slX2 = new Slab({ rows: N, cols: 16, colSpacing: 0.62, chanSpacing: 0.42, x: 8.5, z: -6.5, name: `X${sup(b + 1)} = X′ + MLP`, sub: 'block output' });
      mlpStage.add(slH.group, slMO.group, slX2.group);
      rootGroup.add(mlpStage);
      stages.push({ group: mlpStage, slabs: [slH, slMO, slX2], key: `b${b}.mlp` });
    }

    // ---- final LN
    const lnfStage = mkSimpleStage(S(1 + meta.L * 5), 'final LayerNorm', [
      { rows: N, name: 'LN(X)final', sub: 'ready for unembedding' }]);
    stages.push({ ...lnfStage, key: 'lnf' });

    // ---- output stage
    const outStage = new THREE.Group(); outStage.position.z = S(2 + meta.L * 5);
    const logitsBar = new BarRow({ n: meta.V, x: -10.5, z: 0, mode: 'logit', maxH: 4.6 });
    const probBar = new BarRow({ n: meta.V, x: 10.5, z: 0, mode: 'prob', maxH: 4.6 });
    const lblL = labelSprite('logits  = LN·Wuᵀ', 'one raw score per token', 9); lblL.position.set(-10.5, 6.6, 0);
    const lblP = labelSprite('softmax → P(next)', 'probabilities sum to 1', 9); lblP.position.set(10.5, 6.6, 0);
    outStage.add(logitsBar.group, probBar.group, lblL, lblP);
    rootGroup.add(outStage);
    stages.push({ group: outStage, slabs: [], bars: [logitsBar, probBar], key: 'out' });

    // focus stage 0 instantly
    setStageFocus(0, true);
  }

  function sup(i) { return i === 0 ? '⁰' : i === 1 ? '¹' : '²'; }

  function mkSimpleStage(z, title, slabs) {
    const g = new THREE.Group(); g.position.z = z;
    const list = slabs.map((s, i) => {
      const sl = new Slab({ rows: s.rows, cols: 16, x: (i - (slabs.length - 1) / 2) * 12, z: 0, name: s.name, sub: s.sub });
      g.add(sl.group);
      return sl;
    });
    rootGroup.add(g);
    return { group: g, slabs: list };
  }

  // ---------------------------------------------------------------- stage focus / camera
  function setStageFocus(idx, instant) {
    stages.forEach((st, i) => {
      const d = Math.abs(i - idx);
      const op = d === 0 ? 1 : d === 1 ? 0.5 : 0.16;
      st.slabs.forEach(s => s.setOpacity(op));
      (st.bars || []).forEach(b => b.setOpacity(op));
    });
    flyTo(stageCam(idx), instant);
  }

  function stageCam(idx) {
    const z = -idx * STAGE_GAP;
    const key = stages[idx] ? stages[idx].key : '';
    if (key === 'out')
      return { pos: [0, 11.5, z + 27], tgt: [0, 2.4, z - 1] };
    if (key.endsWith('.attn'))
      return { pos: [0, 13.5, z + 15], tgt: [0, 1.5, z - 1] };
    if (key.endsWith('.mlp'))
      return { pos: [0, 13.5, z + 19], tgt: [0, 1.2, z - 3] };
    return { pos: [0, 11.5, z + 15.5], tgt: [0, 1.2, z - 1.5] };
  }

  function flyTo(dest, instant) {
    const p0 = camera.position.clone(), t0 = controls.target.clone();
    const p1 = new THREE.Vector3(...dest.pos), t1 = new THREE.Vector3(...dest.tgt);
    if (instant) { camera.position.copy(p1); controls.target.copy(t1); controls.update(); return; }
    tween = { p0, t0, p1, t1, start: performance.now(), dur: 700 };
  }

  // ---------------------------------------------------------------- arcs
  function setArcs(layer, head, attnRows, selRow, opts) {
    opts = opts || {};
    if (!arcGroup) return;
    while (arcGroup.children.length) {
      const c = arcGroup.children.pop();
      if (c.geometry) c.geometry.dispose();
      if (c.material) c.material.dispose();
    }
    const st = stages.find(s => s.arcLayer === layer);
    if (!st || !attnRows) return;
    const stageZ = st.group.position.z + 3.4;  // posts plane
    const N = attnRows.length;
    for (let i = 0; i < N; i++) {
      const row = attnRows[i]; if (!row) continue;
      const isSel = selRow == null || i === selRow;
      for (let j = 0; j < row.length; j++) {
        const w = row[j];
        if (w < 0.006) continue;
        const x0 = valueToX(i, N, 1.25), x1 = valueToX(j, N, 1.25);
        const y0 = 2.0;
        const lift = 1.1 + w * 5.2;
        const mid = new THREE.Vector3((x0 + x1) / 2, y0 + lift, 0);
        const curve = new THREE.QuadraticBezierCurve3(
          new THREE.Vector3(x0, y0, 0), mid, new THREE.Vector3(x1, y0, 0));
        const radius = 0.02 + w * (isSel ? 0.16 : 0.07);
        const tube = new THREE.Mesh(
          new THREE.TubeGeometry(curve, 22, radius, 6, false),
          new THREE.MeshBasicMaterial({
            color: ARC_COLORS[head], transparent: true,
            opacity: (isSel ? 0.25 + w * 0.75 : 0.05 + w * 0.1),
            depthWrite: false, blending: THREE.AdditiveBlending }));
        arcGroup.add(tube);
      }
    }
    // highlight ring on selected post
    if (selRow != null && opts.showRing) {
      const ring = new THREE.Mesh(
        new THREE.TorusGeometry(0.55, 0.07, 8, 24),
        new THREE.MeshBasicMaterial({ color: 0xfbbf24, transparent: true, opacity: 0.9 }));
      ring.position.set(valueToX(selRow, N, 1.25), 2.6, 0);
      ring.userData.isRing = true;
      arcGroup.add(ring);
    }
  }

  // ---------------------------------------------------------------- pointer
  let downXY = null;
  function bindPointer() {
    const el = renderer.domElement;
    el.addEventListener('pointerdown', e => { downXY = [e.clientX, e.clientY]; });
    el.addEventListener('pointerup', e => {
      if (!downXY) return;
      const dx = e.clientX - downXY[0], dy = e.clientY - downXY[1];
      downXY = null;
      if (dx * dx + dy * dy > 25) return;      // was a drag
      const hit = pick(e);
      if (hit && hit.userData.pos != null && onTokenPicked) onTokenPicked(hit.userData.pos);
    });
    el.addEventListener('pointermove', e => {
      const hit = pick(e);
      el.style.cursor = hit ? 'pointer' : '';
      const hovered = hit && hit.userData.pos != null ? hit : null;
      chips.forEach(c => c.scale.setScalar(c === hovered ? 1.18 : 1));
      posts.forEach(p => p.scale.setScalar(p === hovered ? 1.22 : 1));
    });
  }

  function pick(e) {
    const rect = renderer.domElement.getBoundingClientRect();
    const ndc = new THREE.Vector2(
      ((e.clientX - rect.left) / rect.width) * 2 - 1,
      -((e.clientY - rect.top) / rect.height) * 2 + 1);
    raycaster.setFromCamera(ndc, camera);
    const hits = raycaster.intersectObjects(pickables, false);
    return hits.length ? hits[0].object : null;
  }

  // ---------------------------------------------------------------- loop
  const easeIO = t => t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
  function animate() {
    requestAnimationFrame(animate);
    if (tween) {
      const k = Math.min(1, (performance.now() - tween.start) / tween.dur);
      const e = easeIO(k);
      camera.position.lerpVectors(tween.p0, tween.p1, e);
      controls.target.lerpVectors(tween.t0, tween.t1, e);
      if (k >= 1) tween = null;
    }
    stars.rotation.y += 0.00012;
    controls.update();
    renderer.render(scene, camera);
  }

  // ---------------------------------------------------------------- public API
  return {
    init,
    buildAll,
    get stages() { return stages; },
    slab(stageIdx, i) { const st = stages[stageIdx]; return st && st.slabs ? st.slabs[i] : null; },
    bars() { const st = stages[stages.length - 1]; return st.bars; },
    setStageFocus,
    setArcs,
    setSelected(pos) { selectedPos = pos; },
    stageCount() { return stages.length; },
    stageIndexForStep(stepKey) { return stages.findIndex(s => s.key === stepKey); },
  };
})();
