#!/usr/bin/env python3
"""
Train a tiny decoder-only Transformer (2 layers, 2 heads, d_model=16, vocab=16,
max context 16) on a deterministic toy "language":

    t[k+1] = (t[k-1] + 1) mod 16

i.e. the next token is one more than the token one step back. Sequences look like
    a, b, a+1, b+1, a+2, b+2, ...
The ONLY way to predict the next token is to attend to the previous token, so the
trained model develops a crisp "previous-token" attention head -- perfect for an
educational visualization.

The script:
  1. builds the model + hand-rolled backprop (numerically gradient-checked)
  2. trains with full-batch Adam
  3. exports weights to js/weights.js (exact float64 round-trip via JSON)
  4. exports reference logits for JS parity tests to test/test_reference.json
  5. prints an attention-pattern summary (used to write honest UI copy)

Run:  python3 tools/train.py
"""
import json, math, sys
import numpy as np

rng = np.random.default_rng(1234)

# ----------------------------- config ---------------------------------------
V      = 16    # vocab
D      = 16    # d_model
H      = 2     # heads
HD     = D // H
L      = 2     # layers
F      = 64    # mlp hidden (4x)
CTX    = 16    # max context
EPS    = 1e-5
SCALE  = 1.0 / math.sqrt(HD)
GELU_C = math.sqrt(2.0 / math.pi)

# ----------------------------- basic ops ------------------------------------
def gelu(x):
    return 0.5 * x * (1.0 + np.tanh(GELU_C * (x + 0.044715 * x ** 3)))

def gelu_grad(x):
    t = GELU_C * (x + 0.044715 * x ** 3)
    th = np.tanh(t)
    return 0.5 * (1.0 + th) + 0.5 * x * (1.0 - th * th) * GELU_C * (1.0 + 3 * 0.044715 * x ** 2)

def softmax(x, axis=-1):
    m = x.max(axis=axis, keepdims=True)
    e = np.exp(x - m)
    return e / e.sum(axis=axis, keepdims=True)

def layernorm_forward(x, g, b):
    mu = x.mean(-1, keepdims=True)
    xc = x - mu
    var = (xc * xc).mean(-1, keepdims=True)
    rstd = 1.0 / np.sqrt(var + EPS)
    xhat = xc * rstd
    return g * xhat + b, (xhat, rstd)

def layernorm_backward(dy, g, cache):
    xhat, rstd = cache
    D = xhat.shape[-1]
    dg = (dy * xhat).reshape(-1, D).sum(0)
    db = dy.reshape(-1, D).sum(0)
    dxhat = dy * g
    # xhat = (x - mu) * rstd ; standard LN backward:
    # dx = rstd * (dxhat - mean(dxhat) - xhat * mean(dxhat * xhat))
    m1 = dxhat.mean(-1, keepdims=True)
    m2 = (dxhat * xhat).mean(-1, keepdims=True)
    dx = (dxhat - m1 - xhat * m2) * rstd
    return dx, dg, db

# ----------------------------- params ---------------------------------------
def init_params():
    P = {}
    P["tok_emb"] = rng.normal(0, 0.8, (V, D))
    P["pos_emb"] = rng.normal(0, 1.2, (CTX, D))
    for l in range(L):
        p = {}
        p["ln1_g"] = np.ones(D);  p["ln1_b"] = np.zeros(D)
        p["Wq"] = rng.normal(0, 0.35, (D, D)); p["Wk"] = rng.normal(0, 0.35, (D, D))
        p["Wv"] = rng.normal(0, 0.35, (D, D)); p["Wo"] = rng.normal(0, 0.35, (D, D))
        p["ln2_g"] = np.ones(D);  p["ln2_b"] = np.zeros(D)
        p["W1"] = rng.normal(0, 0.25, (D, F)); p["b1"] = np.zeros(F)
        p["W2"] = rng.normal(0, 0.06, (F, D)); p["b2"] = np.zeros(D)
        P[f"block{l}"] = p
    P["lnf_g"] = np.ones(D); P["lnf_b"] = np.zeros(D)
    P["Wu"] = rng.normal(0, 0.25, (D, V))
    return P

def split_heads(x):
    # [B,T,D] -> [B,H,T,HD]
    B, T, _ = x.shape
    return x.reshape(B, T, H, HD).transpose(0, 2, 1, 3)

def merge_heads(x):
    # [B,H,T,HD] -> [B,T,D]
    B, Hn, T, HDn = x.shape
    return x.transpose(0, 2, 1, 3).reshape(B, T, Hn * HDn)

# ----------------------------- forward --------------------------------------
def forward(P, ids, want_cache=False):
    """ids: [B,T] int. Returns logits [B,T,V] and a cache dict for backward/viz."""
    B, T = ids.shape
    C = {"ids": ids, "B": B, "T": T}
    x = P["tok_emb"][ids] + P["pos_emb"][:T][None, :, :]
    C["x0"] = x
    for l in range(L):
        p = P[f"block{l}"]
        c = {}
        ln1, ln1c = layernorm_forward(x, p["ln1_g"], p["ln1_b"])
        q = (ln1 @ p["Wq"]); k = (ln1 @ p["Wk"]); v = (ln1 @ p["Wv"])
        qh, kh, vh = split_heads(q), split_heads(k), split_heads(v)          # [B,H,T,HD]
        s = np.einsum("bhtd,bhsd->bhts", qh, kh) * SCALE                     # scores
        mask = np.triu(np.ones((T, T), bool), 1)
        s = np.where(mask, -1e30, s)
        a = softmax(s, axis=-1)                                              # attn weights
        oh = np.einsum("bhts,bhsd->bhtd", a, vh)
        o = merge_heads(oh) @ p["Wo"]
        x1 = x + o
        ln2, ln2c = layernorm_forward(x1, p["ln2_g"], p["ln2_b"])
        hpre = ln2 @ p["W1"] + p["b1"]
        gact = gelu(hpre)
        mlp = gact @ p["W2"] + p["b2"]
        x2 = x1 + mlp
        c.update(ln1=ln1, ln1c=ln1c, q=q, k=k, v=v, qh=qh, kh=kh, vh=vh, s=s, a=a,
                 oh=oh, o=o, x1=x1, ln2=ln2, ln2c=ln2c, hpre=hpre, gact=gact,
                 mlp=mlp, x2=x2)
        C[f"block{l}"] = c
        x = x2
    lnf, lnfc = layernorm_forward(x, P["lnf_g"], P["lnf_b"])
    logits = lnf @ P["Wu"]
    C["x_final"] = x; C["lnf"] = lnf; C["lnfc"] = lnfc; C["logits"] = logits
    return (logits, C) if want_cache else logits

# ----------------------------- backward -------------------------------------
def backward(P, C, dlogits):
    """dlogits: [B,T,V]. Returns grads dict with same structure as P."""
    B, T, _ = dlogits.shape
    G = {k: ({kk: np.zeros_like(vv) for kk, vv in v.items()} if isinstance(v, dict)
             else np.zeros_like(v))
         for k, v in P.items()}
    lnf = C["lnf"]
    G["Wu"] += np.einsum("btd,btv->dv", lnf, dlogits)
    dlnf = dlogits @ P["Wu"].T
    dx, dg, db = layernorm_backward(dlnf, P["lnf_g"], C["lnfc"])
    G["lnf_g"] += dg; G["lnf_b"] += db
    dx2 = dx
    for l in reversed(range(L)):
        p = P[f"block{l}"]; c = C[f"block{l}"]; g = G[f"block{l}"]
        # --- residual 2: x2 = x1 + mlp
        dmlp = dx2.copy()
        dx1 = dx2.copy()
        g["W2"] += np.einsum("btf,btd->fd", c["gact"], dmlp)
        g["b2"] += dmlp.sum((0, 1))
        dgact = dmlp @ p["W2"].T
        dhpre = dgact * gelu_grad(c["hpre"])
        g["W1"] += np.einsum("btd,btf->df", c["ln2"], dhpre)
        g["b1"] += dhpre.sum((0, 1))
        dln2 = dhpre @ p["W1"].T
        dx1_ln, dg2, db2 = layernorm_backward(dln2, p["ln2_g"], c["ln2c"])
        g["ln2_g"] += dg2; g["ln2_b"] += db2
        dx1 += dx1_ln
        # --- residual 1: x1 = x + o
        do = dx1.copy()
        dx = dx1.copy()
        g["Wo"] += np.einsum("btd,bte->de", merge_heads(c["oh"]), do)
        doh = split_heads(do @ p["Wo"].T)              # [B,T,D] -> [B,H,T,HD]
        a, vh = c["a"], c["vh"]
        dvh = np.einsum("bhts,bhtd->bhsd", a, doh)
        da = np.einsum("bhtd,bhsd->bhts", doh, vh)
        ds = a * (da - (da * a).sum(-1, keepdims=True))
        ds = np.where(np.triu(np.ones((T, T), bool), 1)[None, None], 0.0, ds)
        dqh = np.einsum("bhts,bhsd->bhtd", ds * SCALE, c["kh"])
        dkh = np.einsum("bhts,bhtd->bhsd", ds * SCALE, c["qh"])
        dq, dk, dv = merge_heads(dqh), merge_heads(dkh), merge_heads(dvh)  # [B,T,D]
        g["Wq"] += np.einsum("btd,bte->de", c["ln1"], dq)
        g["Wk"] += np.einsum("btd,bte->de", c["ln1"], dk)
        g["Wv"] += np.einsum("btd,bte->de", c["ln1"], dv)
        dln1 = dq @ p["Wq"].T + dk @ p["Wk"].T + dv @ p["Wv"].T
        dx_ln, dg1, db1 = layernorm_backward(dln1, p["ln1_g"], c["ln1c"])
        g["ln1_g"] += dg1; g["ln1_b"] += db1
        dx += dx_ln
        dx2 = dx  # propagate upstream gradient to the next (lower) block
    # embed
    np.add.at(G["tok_emb"], C["ids"].reshape(-1), dx.reshape(-1, D))
    G["pos_emb"][:T] += dx.sum(0)
    return G

# ----------------------------- loss -----------------------------------------
def loss_and_grads(P, ids, targets, valid):
    logits = forward(P, ids)
    B, T, _ = logits.shape
    probs = softmax(logits)
    picked = np.take_along_axis(probs, targets[:, :, None], axis=-1)[:, :, 0]
    ll = np.log(picked + 1e-12)
    loss = -(ll * valid).sum() / valid.sum()
    onehot = np.zeros_like(probs)
    np.put_along_axis(onehot, targets[:, :, None], 1.0, axis=-1)
    dlogits = (probs - onehot) * (valid / valid.sum())[:, :, None]
    G = backward(P, forward(P, ids, want_cache=True)[1], dlogits)
    return loss, G, logits

# ----------------------------- gradient check --------------------------------
def gradcheck():
    global V, D, H, HD, L, F, CTX, rng
    V, D, H, HD, L, F, CTX = 5, 6, 2, 3, 2, 8, 6
    EPSg = 1e-5
    rng = np.random.default_rng(7)
    P = init_params()
    B, T = 2, 5
    ids = rng.integers(0, V, (B, T))
    targets = rng.integers(0, V, (B, T))
    valid = (np.arange(T) >= 1).astype(float)[None, :] * np.ones((B, T))
    loss, G, _ = loss_and_grads(P, ids, targets, valid)

    def scalar_loss(Px):
        l, _, _ = loss_and_grads(Px, ids, targets, valid)
        return l

    worst = 0.0
    checks = 0
    for name in ["tok_emb", "pos_emb", "Wu", "lnf_g", "lnf_b"]:
        arr = P[name]
        flat_idx = rng.choice(arr.size, size=min(6, arr.size), replace=False)
        for fi in flat_idx:
            idx = np.unravel_index(fi, arr.shape)
            old = arr[idx]
            arr[idx] = old + EPSg; lp = scalar_loss(P)
            arr[idx] = old - EPSg; lm = scalar_loss(P)
            arr[idx] = old
            num = (lp - lm) / (2 * EPSg)
            ana = G[name][idx]
            rel = abs(num - ana) / max(1e-8, abs(num) + abs(ana))
            worst = max(worst, rel); checks += 1
    for l in range(L):
        for name in ["Wq", "Wk", "Wv", "Wo", "W1", "b1", "b2", "ln1_g", "ln1_b", "ln2_g", "ln2_b"]:
            arr = P[f"block{l}"][name]
            flat_idx = rng.choice(arr.size, size=min(6, arr.size), replace=False)
            for fi in flat_idx:
                idx = np.unravel_index(fi, arr.shape)
                old = arr[idx]
                arr[idx] = old + EPSg; lp = scalar_loss(P)
                arr[idx] = old - EPSg; lm = scalar_loss(P)
                arr[idx] = old
                num = (lp - lm) / (2 * EPSg)
                ana = G[f"block{l}"][name][idx]
                rel = abs(num - ana) / max(1e-8, abs(num) + abs(ana))
                worst = max(worst, rel); checks += 1
    print(f"gradcheck: {checks} params, worst rel err = {worst:.3e}", flush=True)
    # restore globals
    V, D, H, HD, L, F, CTX = 16, 16, 2, 8, 2, 64, 16
    rng = np.random.default_rng(1234)
    return worst < 1e-5

# ----------------------------- train -----------------------------------------
def train():
    B = 512
    ids = rng.integers(0, V, (B, CTX))
    targets = (ids - 1 + 1) % V  # placeholder, recomputed below
    # target at position i is (ids[i-1] + 1) % V for i>=1 ; position 0 masked
    targets = np.zeros_like(ids)
    targets[:, 1:] = (ids[:, :-1] + 1) % V
    valid = (np.arange(CTX) >= 1).astype(float)[None, :] * np.ones((B, CTX))

    P = init_params()
    m = {k: (np.zeros_like(v) if not isinstance(v, dict) else
             {kk: np.zeros_like(vv) for kk, vv in v.items()})
         for k, v in P.items()}
    v2 = {k: (np.zeros_like(v) if not isinstance(v, dict) else
              {kk: np.zeros_like(vv) for kk, vv in v.items()})
          for k, v in P.items()}

    def as_iter(P):
        for k, val in P.items():
            if isinstance(val, dict):
                for kk in val: yield (k, kk)
            else:
                yield (k, None)

    steps = 6000
    lr0, lr1 = 8e-3, 3e-4
    beta1, beta2, epsA = 0.9, 0.999, 1e-8
    for t in range(1, steps + 1):
        lr = lr1 + 0.5 * (lr0 - lr1) * (1 + math.cos(math.pi * t / steps))
        loss, G, logits = loss_and_grads(P, ids, targets, valid)
        for (k, kk) in as_iter(P):
            gk = G[k] if kk is None else G[k][kk]
            mk = m[k] if kk is None else m[k][kk]
            vk = v2[k] if kk is None else v2[k][kk]
            pk = P[k] if kk is None else P[k][kk]
            mk *= beta1; mk += (1 - beta1) * gk
            vk *= beta2; vk += (1 - beta2) * gk * gk
            mh = mk / (1 - beta1 ** t); vh = vk / (1 - beta2 ** t)
            pk -= lr * mh / (np.sqrt(vh) + epsA)
        if t % 500 == 0 or t == 1:
            pred = logits.argmax(-1)
            acc = ((pred == targets) * valid).sum() / valid.sum()
            last = logits[:, -1, :].argmax(-1)
            last_acc = (last == targets[:, -1]).mean()
            print(f"step {t:5d}  loss {loss:.4f}  acc {acc:.4f}  last-pos acc {last_acc:.4f}", flush=True)
        if acc > 0.9995 and t > 2000:
            print(f"converged at step {t}, loss {loss:.5f}")
            break
    return P

# ----------------------------- export ----------------------------------------
def flatten(P):
    out = {}
    for k, val in P.items():
        if isinstance(val, dict):
            for kk, vv in val.items():
                out[f"block{k}.{kk}" if not k.startswith("block") else f"{k}.{kk}"] = vv.tolist()
        else:
            out[k] = val.tolist()
    return out

def attention_summary(P):
    """Average attention mass per head as a function of offset (j-i), over a sample."""
    ids = rng.integers(0, V, (64, CTX))
    _, C = forward(P, ids, want_cache=True)
    print("\nattention pattern summary (mean mass by offset j-i):")
    summary = {}
    for l in range(L):
        a = C[f"block{l}"]["a"].mean(0)   # [H,T,T]
        for h in range(H):
            rows = []
            for off in range(-7, 1):
                i_idx = np.arange(CTX)
                j_idx = i_idx + off
                ok = j_idx >= 0
                if ok.sum() == 0: continue
                mass = a[h, i_idx[ok], j_idx[ok]].mean()
                rows.append((off, round(float(mass), 3)))
            summary[f"L{l}H{h}"] = rows
            print(f"  layer {l} head {h}: " + "  ".join(f"{off:+d}:{mass:.3f}" for off, mass in rows))
    return summary

def main():
    ok = gradcheck()
    if not ok:
        print("GRADIENT CHECK FAILED"); sys.exit(1)
    P = train()

    # ---- export weights (float64 exact round trip)
    flat = flatten(P)
    meta = {"V": V, "D": D, "H": H, "HD": HD, "L": L, "F": F, "CTX": CTX, "EPS": EPS,
            "SCALE": SCALE, "task": "t[k+1] = (t[k-1] + 1) mod 16",
            "seed": 1234}
    with open("js/weights.js", "w") as f:
        f.write("// Auto-generated by tools/train.py -- fixed, reproducible trained weights.\n")
        f.write("// Do not edit. Model: 2-layer decoder-only Transformer, d_model=16, 2 heads,\n")
        f.write("// vocab=16, max context=16, GELU-tanh MLP, pre-LayerNorm, learned positions.\n")
        f.write("window.TINY_WEIGHTS = " + json.dumps({"meta": meta, "weights": flat}) + ";\n")

    # ---- reference outputs for JS parity tests
    tests = []
    demo = [4, 9, 5, 10, 6, 11, 7, 12]
    seqs = [demo, [3, 14, 7, 0, 11, 5, 9, 2], [4, 9, 5, 10, 6, 11, 7, 12, 8, 13],
            [0, 1], [15, 3, 0, 4, 1, 5, 2, 6, 3, 7, 4, 8, 5, 9, 6, 10]]
    for s in seqs:
        arr = np.array([s])
        logits, C = forward(P, arr, want_cache=True)
        tests.append({"ids": s,
                      "logits": logits[0].tolist(),
                      "attn": {f"L{l}H{h}": C[f"block{l}"]["a"][0, h].tolist()
                               for l in range(L) for h in range(H)}})
    with open("test/test_reference.json", "w") as f:
        json.dump(tests, f)

    attention_summary(P)

    # quick demo print
    s = np.array([demo])
    logits, C = forward(P, s, want_cache=True)
    pr = softmax(logits)[0, -1]
    print("\ndemo sequence", demo)
    print("last-position top-3:", np.argsort(-pr)[:3], "probs:", np.round(np.sort(pr)[::-1][:3], 4))
    print("true next token:", (demo[-2] + 1) % V)
    print("wrote js/weights.js and test/test_reference.json")

if __name__ == "__main__":
    main()
