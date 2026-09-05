// Optional browser integration test. See README for installation instructions.
import assert from 'node:assert/strict';
import {mkdir} from 'node:fs/promises';
const {chromium}=await import(process.env.PLAYWRIGHT_MODULE || 'playwright');
const browser=await chromium.launch({headless:true,args:['--no-sandbox','--enable-unsafe-swiftshader']});
const page=await browser.newPage({viewport:{width:1440,height:1100},deviceScaleFactor:1});
const errors=[],external=[];page.on('pageerror',e=>errors.push(String(e)));page.on('request',r=>{if(!r.url().startsWith('http://localhost:8080')&&!r.url().startsWith('data:')&&!r.url().startsWith('blob:'))external.push(r.url());});
await page.goto('http://localhost:8080');await page.waitForFunction(()=>window.observatorySnapshot);
const snap=()=>page.evaluate(()=>window.observatorySnapshot());
let s=await snap();assert.ok(s.sceneAvailable,'WebGL scene should initialize');assert.equal(s.ids.length,7);assert.equal(s.comparison.logits,0);
await mkdir('tests/artifacts',{recursive:true});await page.screenshot({path:'tests/artifacts/desktop.png',fullPage:true});
// Exercise actual 3D navigation and selection, not just canvas creation.
await page.locator('.scene-row-label').nth(2).click();assert.equal((await snap()).focus,2);
await page.locator('.scene-label').nth(2).click();assert.equal((await snap()).index,3);
const canvas=page.locator('#viewport canvas'),beforeOrbit=await canvas.screenshot(),box=await canvas.boundingBox();await page.mouse.move(box.x+box.width*.5,box.y+box.height*.4);await page.mouse.down();await page.mouse.move(box.x+box.width*.65,box.y+box.height*.5,{steps:12});await page.mouse.up();await page.waitForTimeout(350);assert.notDeepEqual(await canvas.screenshot(),beforeOrbit,'Orbit drag changes the 3D view');await page.locator('#resetCamera').click();
for(let i=0;i<18;i++){await page.locator(`[data-step="${i}"]`).click();assert.equal((await snap()).index,i);assert.ok(await page.locator('#tensorTable td').count()>0);}
await page.locator('#tensorTable td button').first().click();assert.match(await page.locator('#cellDetail').innerText(),/Float64/);
const before=(await snap()).probs;await page.locator('#cacheToggle').uncheck();assert.deepEqual((await snap()).probs,before);await page.locator('#cacheToggle').check();assert.deepEqual((await snap()).probs,before);
await page.locator('#generate').click();s=await snap();assert.equal(s.ids.length,8);assert.equal(s.work.computed,1);assert.equal(s.work.reused,7);assert.equal(s.comparison.logits,0);
await page.locator('[data-position="2"]').selectOption('4');s=await snap();assert.equal(s.work.reused,2);assert.equal(s.work.computed,6);assert.equal(s.comparison.logits,0);
await page.locator('#removeToken').click();s=await snap();assert.equal(s.ids.length,7);assert.equal(s.work.computed,0);
await page.locator('#attentionTab').click();await page.locator('#attLayer').selectOption({value:'1'});await page.locator('#attHead').selectOption({value:'1'});assert.equal(await page.locator('#attLayer').inputValue(),'1');assert.match(await page.locator('#attentionDetail').innerText(),/Layer 2 · Head 1/);await page.locator('[data-query="1"][data-key="4"]').click();assert.match(await page.locator('#attentionDetail').innerText(),/After causal mask = −∞/);assert.match(await page.locator('#attentionDetail').innerText(),/Softmax probability = 0\.000/);
await page.screenshot({path:'tests/artifacts/attention.png',fullPage:true});
await page.locator('#attentionTab').focus();await page.keyboard.press('ArrowRight');assert.equal(await page.locator('#cacheTab').getAttribute('aria-selected'),'true');
await page.locator('#cacheTab').click();assert.match(await page.locator('#cacheReport').innerText(),/MATCH/);
const downloadPromise=page.waitForEvent('download');await page.locator('#export').click();const download=await downloadPromise;await download.saveAs('tests/artifacts/trace.json');
await page.locator('#aboutBtn').click();assert.ok(await page.locator('#about').isVisible());await page.keyboard.press('Escape');assert.ok(!(await page.locator('#about').isVisible()));
await page.locator('#resetPrompt').click();await page.locator('#play').click();await page.waitForTimeout(2600);assert.equal((await snap()).index,1);await page.locator('#play').click();
for(let i=0;i<5;i++)await page.locator('#appendToken').click();assert.equal((await snap()).ids.length,12);assert.ok(await page.locator('#appendToken').isDisabled());assert.ok(await page.locator('#generate').isDisabled());
for(let i=0;i<11;i++)await page.locator('#removeToken').click();assert.equal((await snap()).ids.length,1);assert.ok(await page.locator('#removeToken').isDisabled());
await page.locator('#resetPrompt').click();await page.locator('#tensorTab').click();
await page.setViewportSize({width:390,height:844});await page.waitForTimeout(300);await page.screenshot({path:'tests/artifacts/mobile.png',fullPage:true});
assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),'No horizontal page overflow on mobile');
await page.locator('#generate').click();assert.equal((await snap()).work.computed,1);
// The numerical app must remain usable without a WebGL context.
const fallback=await browser.newPage();fallback.on('pageerror',e=>errors.push(String(e)));await fallback.addInitScript(()=>{const original=HTMLCanvasElement.prototype.getContext;HTMLCanvasElement.prototype.getContext=function(type,...args){return type.includes('webgl')?null:original.call(this,type,...args);};});await fallback.goto('http://localhost:8080');await fallback.waitForFunction(()=>window.observatorySnapshot);assert.equal(await fallback.evaluate(()=>window.observatorySnapshot().sceneAvailable),false);assert.ok(await fallback.locator('#sceneFallback').isVisible());await fallback.locator('#generate').click();assert.equal(await fallback.evaluate(()=>window.observatorySnapshot().work.computed),1);await fallback.close();
assert.deepEqual(errors,[],'No browser exceptions');assert.deepEqual(external,[],'No external requests');
console.log('PASS: WebGL, all 18 stages, values, cache toggle, append/edit/remove, attention mask, export, dialog, playback, limits, mobile, WebGL fallback, no external requests.');
await browser.close();
