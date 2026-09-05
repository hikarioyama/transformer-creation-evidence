/* Edge cases: length-1 sequence, max-context, position-0 edit, generation to 16. */
const { chromium } = require('playwright');
const http = require('http'); const fs = require('fs'); const path = require('path');
const ROOT = path.join(__dirname, '..');
const srv = http.createServer((req, resp) => {
  const p = path.join(ROOT, req.url === '/' ? 'index.html' : decodeURIComponent(req.url.split('?')[0]));
  fs.readFile(p, (err, data) => { if (err) { resp.writeHead(404); resp.end(); return; }
    resp.writeHead(200, { 'Content-Type': {'.css':'text/css','.js':'text/javascript','.html':'text/html'}[path.extname(p)] || 'text/plain' });
    resp.end(data); });
});
srv.listen(8133, async () => {
  const b = await chromium.launch();
  const pg = await b.newPage({ viewport: { width: 1680, height: 1000 } });
  const errs = [];
  pg.on('pageerror', e => errs.push(e.message));
  pg.on('console', m => { if (m.type() === 'error') errs.push('console: ' + m.text()); });
  await pg.goto('http://localhost:8133/', { waitUntil: 'networkidle' });
  await pg.waitForTimeout(800);
  await pg.click('#introClose');

  const walk = async () => {           // step through everything on every edge case
    await pg.evaluate(() => {
      document.querySelectorAll('#stepList .step').forEach((el, i) => {
        setTimeout(() => el.click(), i * 60);
      });
    });
    await pg.waitForTimeout(60 * 16 + 400);
  };

  // 1. shrink to length 1
  for (let i = 0; i < 7; i++) { await pg.click('#btnPop'); await pg.waitForTimeout(120); }
  const len1 = await pg.evaluate(() => document.querySelectorAll('#seqChips .chip').length);
  console.log('len after 7 pops:', len1);
  await walk();
  // attention row at length 1 must be [1]
  await pg.evaluate(() => document.querySelectorAll('#stepList .step')[4].click());
  await pg.waitForTimeout(300);
  await pg.click('.tab[data-tab="values"]');
  await pg.waitForTimeout(300);
  const attn1 = await pg.evaluate(() => document.querySelectorAll('#valuesBody .score-row')[1].querySelector('.sv').textContent);
  console.log('attention weight at len 1 (expect 1.000):', attn1);

  // 2. grow to max context 16 with cache ON, one prediction at a time
  await pg.click('#kvToggle');
  await pg.waitForTimeout(300);
  let last = '';
  for (let i = 0; i < 15; i++) {
    const disabled = await pg.evaluate(() => document.getElementById('btnAppend').disabled);
    if (disabled) break;
    await pg.click('#btnAppend');
    await pg.waitForTimeout(150);
    last = await pg.textContent('#cacheStatus');
  }
  const lenMax = await pg.evaluate(() => document.querySelectorAll('#seqChips .chip').length);
  const appendDisabled = await pg.evaluate(() => document.getElementById('btnAppend').disabled);
  console.log('grew to max context:', lenMax, '| append disabled:', appendDisabled, '| last cache status:', last.trim());

  // 3. verify at max length with everything cached
  await pg.click('.tab[data-tab="verify"]');
  await pg.click('#verifyBtn');
  await pg.waitForTimeout(500);
  console.log('verify at len 16:', (await pg.textContent('#verifyOut')).trim());

  // 4. edit position 0 via palette (cache must invalidate): pick a token != current id
  await pg.click('#seqChips .chip:nth-child(1)');
  await pg.waitForTimeout(150);
  const pickIdx = await pg.evaluate(() => {
    const cur = window.__app ? window.__app.ids[0] : null;
    return cur == null ? 3 : (cur + 5) % 16;
  });
  await pg.click(`.palette .pal-btn:nth-child(${pickIdx + 1})`);
  await pg.waitForTimeout(300);
  console.log('after pos-0 edit:', (await pg.textContent('#cacheStatus')).trim());
  await pg.click('.tab[data-tab="verify"]');
  await pg.click('#verifyBtn');
  await pg.waitForTimeout(500);
  console.log('verify after edit:', (await pg.textContent('#verifyOut')).trim());
  await walk();

  // 5. prediction correctness on a fresh rule-following sequence (model's job)
  const check = await pg.evaluate(() => {
    const TM = window.TinyModel, model = TM.load(window.TINY_WEIGHTS);
    let good = 0, n = 0;
    for (let k = 0; k < 40; k++) {
      const a = Math.floor(Math.random() * 16), bb = Math.floor(Math.random() * 16);
      const ids = [a, bb]; for (let j = 2; j < 10; j++) ids.push((ids[j - 2] + 1) % 16);
      const tr = TM.forwardFull(model, ids);
      const p = tr.probs[ids.length - 1];
      let am = 0; for (let i = 1; i < p.length; i++) if (p[i] > p[am]) am = i;
      const truth = (ids[ids.length - 2] + 1) % 16;
      if (am === truth) good++; n++;
    }
    return good + '/' + n;
  });
  console.log('model predicts rule correctly on random sequences:', check);

  console.log(errs.length ? 'FAILURES:\n' + errs.join('\n') : 'EDGE CASES CLEAN');
  await b.close(); srv.close();
  process.exit(errs.length ? 1 : 0);
});
