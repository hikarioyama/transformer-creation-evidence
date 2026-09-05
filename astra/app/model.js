// Educational decoder-only Transformer. No ML library, training, or network calls.
// All arithmetic is JavaScript Number (IEEE-754 double). Matrices use [in][out].
export const VOCAB = ['<bos>', 'the', 'a', 'tiny', 'cat', 'robot', 'sees', 'likes', 'chases', 'red', 'blue', 'moon', 'star', 'and', '.', '<eos>'];
export const CONFIG = Object.freeze({layers: 2, heads: 2, hidden: 16, headDim: 8, ff: 32, vocab: 16, maxContext: 12, seed: 20250308, epsilon: 1e-5});
export function random(seed) { return () => { let t = seed += 0x6D2B79F5; t = Math.imul(t ^ t >>> 15, t | 1); t ^= t + Math.imul(t ^ t >>> 7, t | 61); return ((t ^ t >>> 14) >>> 0) / 4294967296; }; }
export const zeros = n => Array(n).fill(0);
export const add = (a,b) => a.map((v,i) => v+b[i]);
export const dot = (a,b) => a.reduce((s,v,i) => s+v*b[i],0);
export function linear(x,w) { return w[0].map((_,j) => x.reduce((s,v,i) => s+v*w[i][j],0)); }
export function softmax(x) { const m=Math.max(...x), e=x.map(v=>Math.exp(v-m)), s=e.reduce((a,b)=>a+b,0); return e.map(v=>v/s); }
export function layerNorm(x) { const mean=x.reduce((a,b)=>a+b,0)/x.length; const variance=x.reduce((s,v)=>s+(v-mean)**2,0)/x.length; return {value:x.map(v=>(v-mean)/Math.sqrt(variance+CONFIG.epsilon)),mean,variance}; }
export function gelu(x) { return .5*x*(1+Math.tanh(Math.sqrt(2/Math.PI)*(x+.044715*x**3))); }
export function positional(p) { return Array.from({length:16},(_,i)=> i%2 ? Math.cos(p/10000**((i-1)/16)) : Math.sin(p/10000**(i/16))); }
export function makeWeights(seed=CONFIG.seed) {
  const rng=random(seed);
  const matrix=(a,b,scale=Math.sqrt(6/(a+b)))=>Array.from({length:a},()=>Array.from({length:b},()=>(rng()*2-1)*scale));
  return {embedding:matrix(16,16,.65), blocks:Array.from({length:2},()=>({wq:matrix(16,16),wk:matrix(16,16),wv:matrix(16,16),wo:matrix(16,16),w1:matrix(16,32),w2:matrix(32,16)})),unembed:matrix(16,16)};
}
export const WEIGHTS=makeWeights();
function validate(ids) { if(!Array.isArray(ids)||ids.length<1||ids.length>CONFIG.maxContext||ids.some(i=>!Number.isInteger(i)||i<0||i>=16)) throw new Error('Expected 1–12 token IDs in [0, 15].'); }
function embed(id,p,w) { const token=w.embedding[id].slice(),position=positional(p); return {token,position,x:add(token,position)}; }
function attend(q, keys, values, p) {
  const heads=[];
  for(let h=0;h<2;h++) {
    const qh=q.slice(h*8,h*8+8);
    const scores=keys.map((k,j)=>j<=p?dot(qh,k.slice(h*8,h*8+8))/Math.sqrt(8):-Infinity);
    const probs=softmax(scores);
    const context=zeros(8);
    for(let j=0;j<values.length;j++) for(let d=0;d<8;d++) context[d]+=probs[j]*values[j][h*8+d];
    heads.push({scores,probs,context});
  }
  return {heads,context:heads.flatMap(h=>h.context)};
}
function finishBlock(x,n1,q,k,v,a,w) {
  const projected=linear(a.context,w.wo),residual1=add(x,projected),n2=layerNorm(residual1);
  const ffPre=linear(n2.value,w.w1),ffAct=ffPre.map(gelu),ffOut=linear(ffAct,w.w2),out=add(residual1,ffOut);
  return {input:x,n1,q,k,v,...a,projected,residual1,n2,ffPre,ffAct,ffOut,out};
}
function finishRow(row,w) { row.final=layerNorm(row.blocks[1].out); row.logits=linear(row.final.value,w.unembed); row.probs=softmax(row.logits); return row; }
// Full sequence evaluation. Future scores explicitly receive -Infinity BEFORE softmax.
export function forward(ids,w=WEIGHTS) {
  validate(ids);
  const rows=ids.map((id,p)=>({...embed(id,p,w),blocks:[]}));
  let xs=rows.map(r=>r.x);
  for(let l=0;l<2;l++) {
    const b=w.blocks[l],ns=xs.map(layerNorm),qs=ns.map(n=>linear(n.value,b.wq)),ks=ns.map(n=>linear(n.value,b.wk)),vs=ns.map(n=>linear(n.value,b.wv));
    xs=xs.map((x,p)=> {const block=finishBlock(x,ns[p],qs[p],ks[p],vs[p],attend(qs[p],ks,vs,p),b); rows[p].blocks.push(block); return block.out;});
  }
  rows.forEach(r=>finishRow(r,w));
  return {ids:ids.slice(),rows,logits:rows.at(-1).logits,probs:rows.at(-1).probs,computed:ids.length,reused:0};
}
// Cache stores K and V per layer, per absolute position. Earlier traces are retained
// only for the inspector; inference reads just K/V, never old Q or residual states.
export class KVDecoder {
  constructor(w=WEIGHTS) {this.w=w;this.reset();}
  reset() {this.ids=[];this.rows=[];this.cache=Array.from({length:2},()=>({keys:[],values:[]}));}
  append(id) {
    validate([...this.ids,id]);
    const p=this.ids.length,row={...embed(id,p,this.w),blocks:[]}; let x=row.x;
    for(let l=0;l<2;l++) {
      const b=this.w.blocks[l],n=layerNorm(x),q=linear(n.value,b.wq),k=linear(n.value,b.wk),v=linear(n.value,b.wv),c=this.cache[l];
      c.keys.push(k);c.values.push(v);
      const block=finishBlock(x,n,q,k,v,attend(q,c.keys,c.values,p),b);row.blocks.push(block);x=block.out;
    }
    finishRow(row,this.w);this.ids.push(id);this.rows.push(row);return row;
  }
  run(ids) {
    validate(ids);
    // On an edit, invalidate the edited position AND everything after it.
    let common=0;while(common<Math.min(ids.length,this.ids.length)&&ids[common]===this.ids[common]) common++;
    this.ids.length=common;this.rows.length=common;
    this.cache.forEach(c=>{c.keys.length=common;c.values.length=common;});
    for(let i=common;i<ids.length;i++) this.append(ids[i]);
    return {ids:ids.slice(),rows:this.rows.slice(),logits:this.rows.at(-1).logits,probs:this.rows.at(-1).probs,computed:ids.length-common,reused:common};
  }
}
export function difference(a,b) {return Math.max(...a.map((v,i)=>Math.abs(v-b[i])));}
export function compareRuns(a,b) { return {logits:Math.max(...a.rows.map((r,i)=>difference(r.logits,b.rows[i].logits))),probs:Math.max(...a.rows.map((r,i)=>difference(r.probs,b.rows[i].probs)))}; }
export const argmax=a=>a.indexOf(Math.max(...a));
// Trace export preserves masked infinities rather than silently turning them into null.
export const traceJSON=value=>JSON.stringify(value,(_,v)=>v===-Infinity?'-Infinity':v,2);
