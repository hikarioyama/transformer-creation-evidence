"""Independent batched NumPy reference, optional: python3 tests/numpy_reference.py.
The JS implementation's exported weights are input, NOT its math helpers.
Checks all positions and key intermediates for every prefix of 12 sequences.
"""
import json
import subprocess
import numpy as np
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
js = """
import {WEIGHTS,forward,random} from './model.js';
const r=random(4096), seq=Array.from({length:12},()=>Array.from({length:12},()=>Math.floor(r()*16)));
console.log(JSON.stringify({weights:WEIGHTS,runs:seq.flatMap(ids=>ids.map((_,i)=>forward(ids.slice(0,i+1))))}));
"""
data = json.loads(subprocess.check_output(['node', '--input-type=module', '-e', js], cwd=ROOT))
w = data['weights']
worst = 0.0
checks = 0

def check(a, b):
    global worst, checks
    error = float(np.max(np.abs(np.asarray(a) - np.asarray(b))))
    worst = max(worst, error)
    checks += 1
    assert error < 1e-10, error

def norm(x):
    return (x - x.mean(axis=-1, keepdims=True)) / np.sqrt(x.var(axis=-1, keepdims=True) + 1e-5)

def softmax(x):
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)

for run in data['runs']:
    ids, rows = run['ids'], run['rows']
    n = len(ids)
    angles = np.arange(n)[:, None] / 10000 ** (np.arange(0, 16, 2)[None, :] / 16)
    pe = np.empty((n, 16))
    pe[:, 0::2], pe[:, 1::2] = np.sin(angles), np.cos(angles)
    x = np.asarray(w['embedding'])[ids] + pe
    check(x, [r['x'] for r in rows])
    for l, block in enumerate(w['blocks']):
        b = {k: np.array(v) for k, v in block.items()}
        u = norm(x)
        q, k, v = (u @ b[key] for key in ('wq', 'wk', 'wv'))
        check(q, [r['blocks'][l]['q'] for r in rows])
        check(k, [r['blocks'][l]['k'] for r in rows])
        check(v, [r['blocks'][l]['v'] for r in rows])
        qh, kh, vh = (a.reshape(n, 2, 8).transpose(1, 0, 2) for a in (q, k, v))
        scores = qh @ kh.transpose(0, 2, 1) / np.sqrt(8)
        scores[:, np.triu_indices(n, 1)[0], np.triu_indices(n, 1)[1]] = -np.inf
        attn = softmax(scores)
        check(attn.transpose(1, 0, 2), [[h['probs'] for h in r['blocks'][l]['heads']] for r in rows])
        context = (attn @ vh).transpose(1, 0, 2).reshape(n, 16)
        check(context, [r['blocks'][l]['context'] for r in rows])
        residual = x + context @ b['wo']
        hidden = norm(residual) @ b['w1']
        activated = .5 * hidden * (1 + np.tanh(np.sqrt(2/np.pi) * (hidden + .044715*hidden**3)))
        x = residual + activated @ b['w2']
        check(hidden, [r['blocks'][l]['ffPre'] for r in rows])
        check(activated, [r['blocks'][l]['ffAct'] for r in rows])
        check(x, [r['blocks'][l]['out'] for r in rows])
    logits = norm(x) @ w['unembed']
    check(logits, [r['logits'] for r in rows])
    check(softmax(logits), [r['probs'] for r in rows])
print(f'PASS: {len(data["runs"])} full runs, {checks} tensor checks; max absolute error {worst:.3e}')
