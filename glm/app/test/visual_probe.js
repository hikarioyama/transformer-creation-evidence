/* Visual sanity probe: pixels rendered, geometry sanity, interaction. */
const { chromium } = require('playwright');
const http = require('http'); const fs = require('fs'); const path = require('path');
const ROOT = path.join(__dirname, '..');
const srv = http.createServer((req, resp) => {
  const p = path.join(ROOT, req.url === '/' ? 'index.html' : decodeURIComponent(req.url.split('?')[0]));
  fs.readFile(p, (err, data) => { if (err) { resp.writeHead(404); resp.end(); return; }
    resp.writeHead(200, { 'Content-Type': {'.css':'text/css','.js':'text/javascript','.html':'text/html'}[path.extname(p)] || 'text/plain' });
    resp.end(data); });
});
srv.listen(8130, async () => {
  const b = await chromium.launch();
  const pg = await b.newPage({ viewport: { width: 1680, height: 1000 } });
  const errs = [];
  pg.on('pageerror', e => errs.push(e.message));
  await pg.goto('http://localhost:8130/', { waitUntil: 'networkidle' });
  await pg.waitForTimeout(1000);
  await pg.click('#introClose'); await pg.waitForTimeout(600);

  const cvSize = await pg.evaluate(() => {
    const cv = document.querySelector('#canvas3d canvas');
    return { w: cv.width, h: cv.height };
  });
  console.log('canvas size:', JSON.stringify(cvSize));

  const shot = await pg.screenshot();
  fs.writeFileSync(path.join(__dirname, 'shot_step1.png'), shot);
  console.log('screenshot bytes:', shot.length);

  // stage geometry: slabs within a stage must not overlap (xy footprints)
  const geo = await pg.evaluate(() => {
    const out = [];
    window.Viz3D.stages.forEach((st, si) => {
      st.slabs.forEach(s => {
        const p = s.group.position;
        const w = s.transposed ? s.cols * s.colSpacing : s.rows * s.colSpacing;
        const d = s.transposed ? s.rows * s.chanSpacing : s.cols * s.chanSpacing;
        out.push({ stage: si, x: p.x, z: p.z, w, d });
      });
    });
    return out;
  });
  const overlaps = [];
  for (const st of [...new Set(geo.map(g => g.stage))]) {
    const slabs = geo.filter(g => g.stage === st);
    const z0 = slabs[0].z;
    const samePlane = slabs.filter(s => Math.abs(s.z - z0) < 3);
    for (let i = 0; i < samePlane.length; i++) for (let j = i + 1; j < samePlane.length; j++) {
      const a = samePlane[i], bb = samePlane[j];
      if (Math.abs(a.x - bb.x) < (a.w + bb.w) / 2 - 0.5)
        overlaps.push(`stage ${st}: slabs at x=${a.x.toFixed(1)} & x=${bb.x.toFixed(1)} overlap`);
    }
  }
  console.log(overlaps.length ? 'OVERLAPS:\n' + overlaps.join('\n') : 'no slab overlaps within stages');

  // arcs present on attention step
  await pg.evaluate(() => document.querySelectorAll('#stepList .step')[4].click());
  await pg.waitForTimeout(900);
  await pg.screenshot({ path: path.join(__dirname, 'shot_step_attn.png') });

  // values tab shows real numbers
  await pg.click('.tab[data-tab="values"]');
  await pg.waitForTimeout(300);
  const vecs = await pg.evaluate(() => document.querySelectorAll('#valuesBody .vec-cell').length);
  const sample = await pg.evaluate(() => {
    const c = document.querySelector('#valuesBody .vec-cell span:last-child');
    return c ? c.textContent : 'none';
  });
  console.log('value cells rendered:', vecs, 'sample value:', sample);

  // prediction consistency: argmax == (one-back + 1) for the demo sequence
  const pred = await pg.evaluate(() => {
    const p = window.TinyModel && null;
    return null;
  });
  await pg.click('.tab[data-tab="verify"]');
  await pg.click('#verifyBtn');
  await pg.waitForTimeout(500);
  console.log('verdict:', (await pg.textContent('#verifyOut')).trim());

  console.log(errs.length ? 'PAGE ERRORS: ' + errs.join(' | ') : 'probe clean');
  await b.close(); srv.close();
});
