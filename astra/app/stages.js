export const STAGES=[{
  name:'Token + position',short:'Embed',group:'INPUT',station:0,key:'x',formula:'xₚ = E[tokenₚ] + PE(p)',
  text:'A token is an ID, not a meaning. Look up its 16 fixed embedding values and add a sinusoidal position vector. This gives the same word a different representation at each position.',
  tensors:[['x','Embedding + position'],['token','Token embedding'],['position','Sinusoidal position']]
}];
for(let l=0;l<2;l++) {
  const base=1+l*3;
  const defs=[
    ['Normalize for attention','Norm 1','n1.value',base,'u = (x − mean(x)) / √(var(x) + 10⁻⁵)','Normalize each token across its 16 features. Mean and population variance are computed independently per row. Scale is fixed to 1 and bias to 0.',[['n1.value','Normalized input'],['input','Residual stream']]],
    ['Make queries, keys & values','Q · K · V','q',base,'Q = uWQ   K = uWK   V = uWV','Three separate linear projections. Split each 16-feature vector into two 8-feature heads. A query asks what to retrieve; keys are compared to it; values are what gets mixed.',[['q','Queries Q'],['k','Keys K'],['v','Values V'],['wq','Weight WQ'],['wk','Weight WK'],['wv','Weight WV']]],
    ['Causal attention','Attention','context',base+1,'A = softmax(QKᵀ / √8 + causal mask)\nheadₕ = AₕVₕ','Each query can only see itself and earlier positions. Future scores become −∞ before row-wise softmax. Each head forms a probability-weighted sum of its 8-feature value vectors.',[['context','Concatenated head outputs'],['q','Queries Q'],['k','Keys K'],['v','Values V']]],
    ['Project & add the skip','Skip 1','residual1',base+1,'a = concat(head₀, head₁)WO\nr = x + a','The two heads are concatenated, then mixed by a 16 × 16 projection. Add the original residual stream: attention updates the representation rather than replacing it.',[['residual1','After attention skip'],['projected','Attention projection'],['wo','Weight WO']]],
    ['Normalize for the MLP','Norm 2','n2.value',base+2,'v = LayerNorm(r)','A second, independent normalization prepares the residual stream for a position-wise feed-forward network. There is no communication between tokens in this sublayer.',[['n2.value','Normalized MLP input'],['residual1','Residual input']]],
    ['A tiny feed-forward network','MLP','ffOut',base+2,'f = GELU(vW₁)W₂   ·   16 → 32 → 16','Expand to 32 features, apply the tanh approximation to GELU, then contract to 16. This nonlinear transformation uses the same weights at every token position.',[['ffPre','MLP pre-activation (32)'],['ffAct','After GELU (32)'],['ffOut','MLP output (16)'],['w1','Weight W₁'],['w2','Weight W₂']]],
    ['Add the MLP skip','Skip 2','out',base+2,'xnext = r + f','Add the MLP update to the attention residual. '+(l===0?'These 16-feature rows now enter the second decoder block.':'The second block is complete. The last token’s representation summarizes its available context.'),[['out','Block output'],['ffOut','MLP update'],['residual1','Attention residual']]]
  ];
  for(const [name,short,key,station,formula,text,tensors] of defs) STAGES.push({name,short,key,station,formula,text,tensors,layer:l,group:`LAYER ${l+1}`});
}
STAGES.push(
  {name:'Final normalization',short:'Final norm',group:'OUTPUT',station:7,key:'final.value',formula:'z = LayerNorm(xfinal)',text:'Normalize each final residual vector before the vocabulary projection. Only the last position is used to choose the next token; earlier positions predict their own following token.',tensors:[['final.value','Final normalized states']]},
  {name:'Project into the vocabulary',short:'Logits',group:'OUTPUT',station:7,key:'logits',formula:'logits = zWU   ·   16 → 16',text:'Each of the 16 vocabulary entries gets a real-valued score. These are logits, not probabilities. The output matrix is independent of the token embedding matrix (untied weights).',tensors:[['logits','Vocabulary logits'],['unembed','Unembedding weights']]},
  {name:'Predict the next token',short:'Predict',group:'OUTPUT',station:7,key:'probs',formula:'P(next = i) = exp(logitᵢ − max) / Σⱼ exp(logitⱼ − max)',text:'A numerically stable softmax turns logits into a distribution over all 16 tokens. Greedy generation appends the highest-probability token. These untrained weights have no learned language ability.',tensors:[['probs','Vocabulary probabilities'],['logits','Vocabulary logits']]}
);
export function getPath(obj,key) {return key.split('.').reduce((o,k)=>o[k],obj);}
export function tensor(run,stage,key,weights) {
  if(key==='unembed') return weights.unembed;
  if(key.startsWith('w')) return weights.blocks[stage.layer][key];
  return run.rows.map(r=>getPath(stage.layer===undefined?r:r.blocks[stage.layer],key));
}
