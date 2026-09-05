#!/usr/bin/env python3
"""Per-parameter gradient diagnostics."""
import numpy as np, math
import train as T

# tiny config
T.V, T.D, T.H, T.HD, T.L, T.F, T.CTX = 5, 6, 2, 3, 2, 8, 6
T.rng = np.random.default_rng(7)
P = T.init_params()
B, Tn = 2, 5
ids = T.rng.integers(0, T.V, (B, Tn))
targets = T.rng.integers(0, T.V, (B, Tn))
valid = (np.arange(Tn) >= 1).astype(float)[None, :] * np.ones((B, Tn))
loss, G, _ = T.loss_and_grads(P, ids, targets, valid)

def scalar_loss(Px):
    l, _, _ = T.loss_and_grads(Px, ids, targets, valid)
    return l

EPS = 1e-5
def check(label, arr, grad, n=8):
    idxs = T.rng.choice(arr.size, size=min(n, arr.size), replace=False)
    worst = (0.0, None, 0, 0)
    for fi in idxs:
        i = np.unravel_index(fi, arr.shape)
        old = arr[i]
        arr[i] = old + EPS; lp = scalar_loss(P)
        arr[i] = old - EPS; lm = scalar_loss(P)
        arr[i] = old
        num = (lp - lm) / (2 * EPS)
        ana = grad[i]
        rel = abs(num - ana) / max(1e-7, abs(num) + abs(ana))
        if rel > worst[0]:
            worst = (rel, i, num, ana)
    flag = "OK " if worst[0] < 1e-5 else "FAIL"
    print(f"{flag} {label:24s} worst rel={worst[0]:.2e} at {worst[1]}  num={worst[2]:.5f} ana={worst[3]:.5f}")

for name in ["tok_emb", "pos_emb", "Wu", "lnf_g", "lnf_b"]:
    check(name, P[name], G[name])
for l in range(T.L):
    for name in ["Wq", "Wk", "Wv", "Wo", "W1", "b1", "b2", "ln1_g", "ln1_b", "ln2_g", "ln2_b"]:
        check(f"block{l}.{name}", P[f"block{l}"][name], G[f"block{l}"][name])
