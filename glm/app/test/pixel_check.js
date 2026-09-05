/* Pixel-level check of the WebGL output via in-page canvas decoding. */
const { chromium } = require('playwright');
const http = require('http'); const fs = require('fs'); const path = require('path');
const ROOT = path.join(__dirname, '..');
const srv = http.createServer((req, resp) => {
  const p = path.join(ROOT, req.url === '/' ? 'index.html' : decodeURIComponent(req.url.split('?')[0]));
  fs.readFile(p, (err, data) => { if (err) { resp.writeHead(404); resp.end(); return; }
    resp.writeHead(200, { 'Content-Type': {'.css':'text/css','.js':'text/javascript','.html':'text/html'}[path.extname(p)] || 'text/plain' });
    resp.end(data); });
});
srv.listen(8131, async () => {
  const b = await chromium.launch();
  const pg = await b.newPage({ viewport: { width: 1680, height: 1000 } });
  await pg.goto('http://localhost:8131/', { waitUntil: 'networkidle' });
  await pg.waitForTimeout(900);
  await pg.click('#introClose'); await pg.waitForTimeout(700);

  async function shotStats(name, clip) {
    const buf = await pg.screenshot(clip ? { clip } : {});
    const b64 = buf.toString('base64');
    return await pg.evaluate(async (b64) => {
      const img = new Image();
      await new Promise((res, rej) => { img.onload = res; img.onerror = rej; img.src = 'data:image/png;base64,' + b64; });
      const cv = document.createElement('canvas');
      cv.width = img.width; cv.height = img.height;
      const ctx = cv.getContext('2d');
      ctx.drawImage(img, 0, 0);
      const d = ctx.getImageData(0, 0, cv.width, cv.height).data;
      let lit = 0, total = d.length / 4, bright = 0, colored = 0;
      let sumR = 0, sumG = 0, sumB = 0;
      for (let i = 0; i < d.length; i += 4) {
        const r = d[i], g = d[i + 1], bl = d[i + 2];
        const lum = 0.2126 * r + 0.7152 * g + 0.0722 * bl;
        if (lum > 18) lit++;
        if (lum > 90) bright++;
        const mx = Math.max(r, g, bl), mn = Math.min(r, g, bl);
        if (mx > 40 && (mx - mn) > 25) colored++;   // saturated (slab colors)
        sumR += r; sumG += g; sumB += bl;
      }
      return { name, lit: (100 * lit / total).toFixed(1) + '%', bright: (100 * bright / total).toFixed(1) + '%',
               colored: (100 * colored / total).toFixed(1) + '%' };
    }, b64);
  }

  console.log(await shotStats('step1-embed(center, panels excluded)', { x: 420, y: 120, width: 820, height: 700 }));
  await pg.evaluate(() => document.querySelectorAll('#stepList .step')[4].click());
  await pg.waitForTimeout(900);
  console.log(await shotStats('step5-attention-arcs(center)', { x: 420, y: 120, width: 820, height: 700 }));
  await pg.evaluate(() => document.querySelectorAll('#stepList .step')[14].click());
  await pg.waitForTimeout(900);
  console.log(await shotStats('step15-output-bars(center)', { x: 420, y: 120, width: 820, height: 700 }));
  await pg.screenshot({ path: path.join(__dirname, 'shot_final.png') });

  await b.close(); srv.close();
});
