/*
 * Headless UI smoke test with Playwright + bundled Chromium.
 *   - loads index.html over a local http server
 *   - fails on any console error / page error
 *   - clicks through all steps, tabs, sequence edits, cache toggle
 *   - runs the parity check in-app
 *   - takes screenshots for visual inspection
 * Run:  node test/ui_test.js
 */
const { chromium } = require('playwright');
const http = require('http');
const fs = require('fs');
const path = require('path');

const ROOT = path.join(__dirname, '..');
const MIME = { '.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css', '.json': 'application/json' };

function serve(port) {
  return new Promise(res => {
    const srv = http.createServer((req, resp) => {
      const p = path.join(ROOT, req.url === '/' ? 'index.html' : decodeURIComponent(req.url.split('?')[0]));
      fs.readFile(p, (err, data) => {
        if (err) { resp.writeHead(404); resp.end('nope'); return; }
        resp.writeHead(200, { 'Content-Type': MIME[path.extname(p)] || 'application/octet-stream' });
        resp.end(data);
      });
    });
    srv.listen(port, () => res(srv));
  });
}

(async () => {
  const srv = await serve(8123);
  const browser = await chromium.launch();
  const page = await browser.newPage({ viewport: { width: 1680, height: 1000 } });
  const errors = [];
  page.on('console', m => { if (m.type() === 'error') errors.push('console: ' + m.text()); });
  page.on('pageerror', e => errors.push('pageerror: ' + e.message));

  await page.goto('http://localhost:8123/', { waitUntil: 'networkidle' });
  await page.waitForTimeout(1200);

  // intro visible? close it
  const intro = await page.$('#intro');
  if (intro) { await page.click('#introClose'); await page.waitForTimeout(400); }

  // ---- step through all steps (btnNext is disabled on the last step)
  const nSteps = await page.evaluate(() => document.querySelectorAll('#stepList .step').length);
  console.log('steps found:', nSteps);
  for (let i = 0; i < nSteps - 1; i++) {
    await page.click('#btnNext');
    await page.waitForTimeout(120);
  }
  await page.screenshot({ path: 'test/shot_last_step.png' });

  // ---- tabs
  for (const t of ['values', 'weights', 'verify', 'attention']) {
    await page.click(`.tab[data-tab="${t}"]`);
    await page.waitForTimeout(150);
  }
  await page.screenshot({ path: 'test/shot_inspector.png' });

  // ---- run in-app parity check
  await page.click('.tab[data-tab="verify"]');
  await page.click('#verifyBtn');
  await page.waitForTimeout(600);
  const verdict = await page.textContent('#verifyOut');
  console.log('verify verdict:', verdict.trim().slice(0, 110));
  if (!/✓/.test(verdict)) errors.push('parity verdict not OK: ' + verdict);

  // ---- back to a middle step; toggle cache ON; append prediction twice
  await page.evaluate(() => { document.querySelectorAll('#stepList .step')[13].click(); });
  await page.waitForTimeout(300);
  await page.click('#kvToggle');
  await page.waitForTimeout(300);
  const status1 = await page.textContent('#cacheStatus');
  console.log('after toggle:', status1.trim());
  const lenBefore = await page.evaluate(() => document.querySelectorAll('#seqChips .chip').length);
  await page.click('#btnAppend');
  await page.waitForTimeout(400);
  await page.click('#btnAppend');
  await page.waitForTimeout(400);
  const lenAfter = await page.evaluate(() => document.querySelectorAll('#seqChips .chip').length);
  const status2 = await page.textContent('#cacheStatus');
  console.log('append with cache:', lenBefore, '->', lenAfter, '|', status2.trim());
  if (lenAfter !== lenBefore + 2) errors.push('append did not add 2 chips');
  if (!/reused/.test(status2)) errors.push('cache status does not report reuse: ' + status2);

  // ---- edit an old token (palette)
  await page.click('#seqChips .chip:nth-child(2)');
  await page.waitForTimeout(200);
  await page.click('.palette .pal-btn:nth-child(3)');
  await page.waitForTimeout(400);
  const status3 = await page.textContent('#cacheStatus');
  console.log('after edit:', status3.trim());
  if (!/prefill|computed/.test(status3)) errors.push('edit did not invalidate cache: ' + status3);

  // ---- verify again after edits
  await page.click('.tab[data-tab="verify"]');
  await page.click('#verifyBtn');
  await page.waitForTimeout(600);
  const verdict2 = await page.textContent('#verifyOut');
  console.log('verify verdict 2:', verdict2.trim().slice(0, 110));
  if (!/✓/.test(verdict2)) errors.push('parity verdict 2 not OK: ' + verdict2);

  // ---- random + demo buttons
  await page.click('#btnRandom'); await page.waitForTimeout(300);
  await page.click('#btnDemo'); await page.waitForTimeout(300);

  // keyboard nav
  await page.keyboard.press('ArrowLeft');
  await page.waitForTimeout(200);

  // screenshots: attention step with arcs
  await page.evaluate(() => {
    const steps = document.querySelectorAll('#stepList .step');
    steps[4].click(); // block1 attention
  });
  await page.waitForTimeout(1100);
  await page.screenshot({ path: 'test/shot_attention.png' });

  await browser.close();
  srv.close();

  if (errors.length) {
    console.log('\nUI TEST FAILURES:');
    errors.forEach(e => console.log('  -', e));
    process.exit(1);
  } else {
    console.log('\nUI SMOKE TEST PASSED (no console/page errors)');
  }
})().catch(e => { console.error('FATAL', e); process.exit(1); });
