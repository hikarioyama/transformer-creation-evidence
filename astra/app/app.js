import {VOCAB,CONFIG,WEIGHTS,KVDecoder,forward,compareRuns,argmax,dot,traceJSON} from './model.js';
import {STAGES,tensor} from './stages.js';
import {Observatory} from './scene.js';
const $=s=>document.querySelector(s),esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const fmt=v=>v===-Infinity?'−∞':v.toFixed(3),prec=v=>v===-Infinity?'−Infinity':v.toPrecision(12);
const options=selected=>VOCAB.map((v,i)=>`<option value="${i}" ${i===selected?'selected':''}>${esc(v)}</option>`).join('');
let ids=[0,1,3,5,6,1,11],index=0,focus=6,cached=true,run,audit,comparison,tab='tensor',tensorKey='x',head=0,attLayer=0,keyPos=0,playing=null,work;
const decoder=new KVDecoder();
const scene=new Observatory($('#viewport'),i=>{stop();setStage(i);},p=>{focus=p;keyPos=Math.min(keyPos,focus);render();});
$('#addToken').innerHTML=options(12);
function recompute(reason){
  const incremental=decoder.run(ids),full=forward(ids);audit=incremental;run=cached?incremental:full;comparison=compareRuns(incremental,full);
  work={computed:run.computed,reused:run.reused,reason};focus=Math.min(focus,ids.length-1);keyPos=Math.min(keyPos,ids.length-1);
  render();
}
function setStage(i){index=Math.max(0,Math.min(STAGES.length-1,i));tensorKey=STAGES[index].tensors[0][0];if(STAGES[index].layer!==undefined)attLayer=STAGES[index].layer;render();}
function stop(){clearInterval(playing);playing=null;$('#play').textContent='▶ Play';}
function append(id,reason){if(ids.length>=CONFIG.maxContext)return;stop();ids.push(id);focus=ids.length-1;recompute(reason);}
function render(){
  const s=STAGES[index],winner=argmax(run.probs),atLimit=ids.length===CONFIG.maxContext;
  $('#tokens').innerHTML=ids.map((id,p)=>`<label class="token ${p===focus?'focus':''}"><small>p${p} · id ${id}</small><select aria-label="Token at position ${p}" data-position="${p}">${options(id)}</select></label>`).join('');
  $('#contextCount').textContent=`${ids.length} / ${CONFIG.maxContext} tokens`;
  $('#nextToken').textContent=VOCAB[winner];$('#nextToken').title=`${(run.probs[winner]*100).toFixed(4)}%`;
  $('#appendToken').disabled=$('#generate').disabled=atLimit;$('#removeToken').disabled=ids.length===1;
  $('#cacheMode').textContent=cached?'ON':'OFF';
  $('#status').textContent=`${comparison.logits<=1e-10&&comparison.probs<=1e-10?'✓':'✗'} Cache audit: max |Δ logits| = ${comparison.logits.toExponential(2)} · ${cached?'Cached':'Full'} path: ${work.computed} token row${work.computed===1?'':'s'} computed / layer, ${work.reused} reused. ${atLimit?'Context full — remove a token to continue.':work.reason}`;
  $('#stageGroup').textContent=`${s.group} / STEP ${String(index+1).padStart(2,'0')}`;$('#stageTitle').textContent=s.name;$('#formula').textContent=s.formula;$('#explanation').textContent=s.text;
  $('#sceneStage').textContent=`${s.group} / ${s.name}`;
  $('#focusToken').innerHTML=ids.map((id,p)=>`<option value="${p}" ${p===focus?'selected':''}>${p} · ${esc(VOCAB[id])}${p===ids.length-1?' / last':''}</option>`).join('');
  let norm=null;if(s.key==='n1.value')norm=run.rows[focus].blocks[s.layer].n1;if(s.key==='n2.value')norm=run.rows[focus].blocks[s.layer].n2;if(s.key==='final.value')norm=run.rows[focus].final;
  $('#normStats').textContent=norm?`p${focus}: μ = ${norm.mean.toFixed(6)} · σ² = ${norm.variance.toFixed(6)}`:`p${focus} can attend to ${focus+1} of ${ids.length} positions.`;
  $('#probabilities').innerHTML=run.probs.map((p,i)=>({p,i})).sort((a,b)=>b.p-a.p).map(({p,i})=>`<div class="prob-row ${i===winner?'winner':''}" title="id ${i}: probability ${prec(p)}; logit ${prec(run.logits[i])}"><span>${esc(VOCAB[i])}${i===winner?' ↗':''}</span><span>${(p*100).toFixed(1)}%</span><i class="bar" style="width:${p*100}%"></i></div>`).join('');
  $('#stepCount').textContent=`${String(index+1).padStart(2,'0')} / ${STAGES.length}`;$('#progressFill').style.width=`${(index+1)/STAGES.length*100}%`;$('#prev').disabled=index===0;$('#next').disabled=index===STAGES.length-1;
  $('#steps').innerHTML=['INPUT','LAYER 1','LAYER 2','OUTPUT'].map(g=>`<div class="step-group"><div class="eyebrow">${g}</div><div class="step-buttons">${STAGES.map((st,i)=>st.group===g?`<button data-step="${i}" class="${i===index?'active':i<index?'done':''}" title="${i+1}. ${st.name}" aria-label="Step ${i+1}: ${st.name}" ${i===index?'aria-current="step"':''}>${i+1}</button>`:'').join('')}</div></div>`).join('');
  if(!s.tensors.some(([k])=>k===tensorKey))tensorKey=s.tensors[0][0];
  $('#tensorSelect').innerHTML=s.tensors.map(([k,label])=>`<option value="${k}" ${k===tensorKey?'selected':''}>${label}</option>`).join('');
  $('#attLayer').value=attLayer;$('#attHead').value=head;
  renderTensor();renderAttention();renderCache();scene.update(run,index,focus,head,tab==='attention'?attLayer:null);
}
function cellColor(v,prob=false){if(v===-Infinity)return '#17212c';const t=prob?v:Math.tanh(v/2),a=[24,39,50],b=t<0?[164,105,63]:[63,140,118],f=Math.min(1,Math.abs(t));return `rgb(${a.map((x,i)=>Math.round(x+(b[i]-x)*f)).join(',')})`;}
function renderTensor(){
  const s=STAGES[index],matrix=tensor(run,s,tensorKey,WEIGHTS),weight=tensorKey.startsWith('w')||tensorKey==='unembed',prob=tensorKey==='probs',vocab=tensorKey==='logits'||prob||tensorKey==='unembed';
  $('#tensorShape').textContent=`[${matrix.length} × ${matrix[0].length}] · ${weight?'input × output':`position × ${vocab?'vocabulary':'feature'}`}`;
  $('#tensorTable').innerHTML=`<table><thead><tr><th>${weight?'in ↓ / out →':'p ↓ / d →'}</th>${matrix[0].map((_,d)=>`<th>${vocab?`${d} ${esc(VOCAB[d])}`:d}</th>`).join('')}</tr></thead><tbody>${matrix.map((row,p)=>`<tr class="${!weight&&p===focus?'selected-row':''}"><th>${weight?p:`${p} ${esc(VOCAB[ids[p]])}`}</th>${row.map((v,d)=>`<td><button data-row="${p}" data-col="${d}" style="--cell:${cellColor(v,prob)}" title="[${p}, ${d}] = ${prec(v)}" aria-label="Row ${p}, column ${d}: ${prec(v)}">${fmt(v)}</button></td>`).join('')}</tr>`).join('')}</tbody></table>`;
  $('#cellDetail').textContent=`${s.tensors.find(([k])=>k===tensorKey)[1]} · ${matrix.length*matrix[0].length} actual values · select a cell`;
}
function renderAttention(){
  $('#attentionTable').innerHTML=`<table><thead><tr><th>q ↓ / k →</th>${ids.map((id,p)=>`<th>${p}<br>${esc(VOCAB[id])}</th>`).join('')}</tr></thead><tbody>${run.rows.map((r,p)=>`<tr class="${p===focus?'selected-row':''}"><th>${p} ${esc(VOCAB[ids[p]])}</th>${ids.map((_,j)=>{const v=j<=p?r.blocks[attLayer].heads[head].probs[j]:0;return `<td><button data-query="${p}" data-key="${j}" style="--cell:${cellColor(v,true)};${p===focus&&j===keyPos?'border-color:#8df0cf':''}" title="q${p} → k${j}: ${prec(v)}${j>p?' (causal mask)':''}">${j>p?'🔒':v.toFixed(3)}</button></td>`;}).join('')}</tr>`).join('')}</tbody></table>`;
  const b=run.rows[focus].blocks[attLayer],kb=run.rows[keyPos].blocks[attLayer],q=b.q.slice(head*8,head*8+8),k=kb.k.slice(head*8,head*8+8),v=kb.v.slice(head*8,head*8+8),products=q.map((x,i)=>x*k[i]),raw=dot(q,k)/Math.sqrt(8),masked=keyPos>focus,prob=masked?0:b.heads[head].probs[keyPos];
  $('#attentionDetail').innerHTML=`<h3>p${focus} “${esc(VOCAB[ids[focus]])}” → p${keyPos} “${esc(VOCAB[ids[keyPos]])}”</h3><div>Layer ${attLayer+1} · Head ${head} · 8 features</div><div class="equation">Q = <code>[${q.map(fmt).join(', ')}]</code><br>K = <code>[${k.map(fmt).join(', ')}]</code></div><div class="equation">Σ QᵢKᵢ = <code>${prec(products.reduce((a,b)=>a+b,0))}</code><br>Scaled score / √8 = <code>${prec(raw)}</code><br>After causal mask = <code>${masked?'−∞':prec(raw)}</code><br>Softmax probability = <code>${prec(prob)}</code></div><div class="equation">V = <code>[${v.map(fmt).join(', ')}]</code><br>This key’s contribution, A × V:<br><code>[${v.map(x=>fmt(x*prob)).join(', ')}]</code></div><div class="equation">Sum over all allowed keys → head output:<br><code>[${b.heads[head].context.map(fmt).join(', ')}]</code><br>Attention row sum = <code>${b.heads[head].probs.reduce((a,b)=>a+b,0).toFixed(12)}</code></div>`;
}
function renderCache(){
  const pass=comparison.logits<=1e-10&&comparison.probs<=1e-10,n=ids.length,bytes=2*2*n*16*8;
  $('#cacheReport').innerHTML=`<div class="cache-summary"><div class="stat"><small>ALL-POSITION AUDIT</small><strong>${pass?'✓ MATCH':'✗ MISMATCH'}</strong></div><div class="stat"><small>MAX |Δ LOGITS|</small><strong>${comparison.logits.toExponential(2)}</strong></div><div class="stat"><small>ROWS COMPUTED / LAYER</small><strong>${work.computed} <span class="muted">/ ${n}</span></strong></div><div class="stat"><small>ACTIVE K/V PAYLOAD</small><strong>${cached?bytes.toLocaleString():0} <span class="muted">bytes</span></strong></div></div><p class="cache-copy">${cached?`<b>Cache ON.</b> ${work.reused} prefix positions reused; ${work.computed} new or invalidated positions computed in each layer. Only the new suffix runs embeddings, Q/K/V projections and MLPs.`:'<b>Cache OFF.</b> Every position is recomputed. The audit still maintains a separate incremental decoder to verify equivalence.'}<br>Max |Δ probabilities| = <span class="mono">${comparison.probs.toExponential(4)}</span>. Pass tolerance: <span class="mono">10⁻¹⁰</span>, checked over <b>all ${n} positions × 16 outputs</b>, not only the winning token.<br>K/V layout per layer: [position, head, feature] = [${n}, 2, 8] for each of K and V. ${bytes} bytes = 2 layers × K/V × ${n} positions × 16 features × 8 bytes (Float64); excludes JS container overhead and inspector traces. ${cached?'Mint = retained prefix; amber = computed suffix.':'Tiles below show the audit cache, not an active inference cache.'}</p>`;
  $('#cacheGrids').innerHTML=[0,1].map(l=>`<div class="cache-grid"><h3>Layer ${l+1} · ${cached?'inference':'audit'} K + V</h3><div class="cache-slots">${ids.map((id,p)=>`<div class="cache-slot ${p>=work.reused?'new':''}" title="K[${p}, 0] = ${prec(audit.rows[p].blocks[l].k[0])}; V[${p}, 0] = ${prec(audit.rows[p].blocks[l].v[0])}">p${p} ${esc(VOCAB[id])}<br>K 2×8<br>V 2×8</div>`).join('')}</div></div>`).join('');
}
$('#tokens').addEventListener('change',e=>{if(!e.target.matches('select'))return;stop();const p=+e.target.dataset.position;ids[p]=+e.target.value;focus=p;recompute(`Edit at p${p}: suffix invalidated.`);});
$('#appendToken').addEventListener('click',()=>append(+$('#addToken').value,'Appended one token.'));
$('#generate').addEventListener('click',()=>{const id=argmax(run.probs);append(id,'Greedy prediction appended; now predicting again.');setStage(STAGES.length-1);});
$('#removeToken').addEventListener('click',()=>{if(ids.length>1){stop();ids.pop();recompute('Removed final token; prefix remains valid.');}});
$('#resetPrompt').addEventListener('click',()=>{stop();ids=[0,1,3,5,6,1,11];focus=6;index=0;decoder.reset();recompute('Example restored; cache rebuilt.');});
$('#cacheToggle').addEventListener('change',e=>{cached=e.target.checked;recompute('Inference path changed; audit remains enabled.');});
$('#focusToken').addEventListener('change',e=>{focus=+e.target.value;render();});
$('#prev').addEventListener('click',()=>{stop();setStage(index-1);});$('#next').addEventListener('click',()=>{stop();setStage(index+1);});
$('#play').addEventListener('click',()=>{if(playing){stop();return;}if(index===STAGES.length-1)setStage(0);$('#play').textContent='Ⅱ Pause';playing=setInterval(()=>{if(index===STAGES.length-1){stop();return;}setStage(index+1);},2400);});
$('#steps').addEventListener('click',e=>{const b=e.target.closest('[data-step]');if(b){stop();setStage(+b.dataset.step);}});
$('#tensorSelect').addEventListener('change',e=>{tensorKey=e.target.value;renderTensor();});
$('#tensorTable').addEventListener('click',e=>{const b=e.target.closest('[data-row]');if(!b)return;const p=+b.dataset.row,d=+b.dataset.col,m=tensor(run,STAGES[index],tensorKey,WEIGHTS);$('#cellDetail').textContent=`${tensorKey}[${p}, ${d}] = ${m[p][d].toPrecision(17)} (Float64)`;});
$('.inspector-tabs').addEventListener('click',e=>{const b=e.target.closest('[data-tab]');if(!b)return;tab=b.dataset.tab;document.querySelectorAll('[data-tab]').forEach(el=>{const active=el===b;el.classList.toggle('active',active);el.setAttribute('aria-selected',active);el.tabIndex=active?0:-1;});['tensor','attention','cache'].forEach(t=>{$(`#${t}Pane`).hidden=t!==tab;});scene.update(run,index,focus,head,tab==='attention'?attLayer:null);});
$('.inspector-tabs').addEventListener('keydown',e=>{if(!['ArrowLeft','ArrowRight','Home','End'].includes(e.key))return;e.preventDefault();const tabs=[...document.querySelectorAll('[data-tab]')],i=tabs.indexOf(document.activeElement);const next=e.key==='Home'?0:e.key==='End'?2:(i+(e.key==='ArrowRight'?1:2))%3;tabs[next].focus();tabs[next].click();});
$('#attLayer').addEventListener('change',e=>{attLayer=+e.target.value;renderAttention();scene.update(run,index,focus,head,attLayer);});$('#attHead').addEventListener('change',e=>{head=+e.target.value;renderAttention();scene.update(run,index,focus,head,attLayer);});
$('#attentionTable').addEventListener('click',e=>{const b=e.target.closest('[data-query]');if(b){focus=+b.dataset.query;keyPos=+b.dataset.key;render();}});
$('#resetCamera').addEventListener('click',()=>scene.reset());$('#aboutBtn').addEventListener('click',()=>$('#about').showModal());$('#closeAbout').addEventListener('click',()=>$('#about').close());
$('#export').addEventListener('click',()=>{const data={model:'Untrained educational decoder-only Transformer',config:CONFIG,vocabulary:VOCAB,weights:WEIGHTS,selectedPath:cached?'cached':'full',work,comparison,trace:run,cache:decoder.cache,notes:'Cached historical attention rows contain only their causal prefix. Missing future scores are -Infinity; missing probabilities are zero. Weight orientation [in][out].'};const url=URL.createObjectURL(new Blob([traceJSON(data)],{type:'application/json'}));const a=document.createElement('a');a.href=url;a.download='small-signals-trace.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);});
document.addEventListener('keydown',e=>{if(['INPUT','SELECT','TEXTAREA','BUTTON'].includes(e.target.tagName)||$('#about').open)return;if(e.key==='ArrowRight'){e.preventDefault();stop();setStage(index+1);}if(e.key==='ArrowLeft'){e.preventDefault();stop();setStage(index-1);}});
recompute('Example loaded. Edit any token to experiment.');
// Read-only debugging snapshot; useful for reproducible browser integration tests.
window.observatorySnapshot=()=>({ids:ids.slice(),index,focus,cached,work:{...work},comparison:{...comparison},probs:run.probs.slice(),logits:run.logits.slice(),sceneAvailable:scene.available});
