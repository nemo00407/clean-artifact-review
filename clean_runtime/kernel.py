"""Minimal deterministic transformer + LoRA training kernel (NumPy).

Non-semantic TCB: this module is trusted only to consume the declared
inputs handed to it by the runtime; its numerical behavior carries no
security obligation. Decoder-only pre-LN transformer, byte-level vocab,
LoRA adapters on the attention W_q / W_v projections, SGD with momentum.
All state is float32 and invocation-local; the only randomness source is
the TeeRng handle.
"""
import numpy as np

F32 = np.float32


# ---------------------------------------------------------------- primitives

def _layernorm_fwd(x, g, b, eps=1e-5):
    mu = x.mean(-1, keepdims=True)
    var = x.var(-1, keepdims=True)
    inv_std = 1.0 / np.sqrt(var + eps)
    xhat = (x - mu) * inv_std
    return (g * xhat + b).astype(F32), (xhat, inv_std, g)


def _layernorm_bwd(dy, cache):
    xhat, inv_std, g = cache
    dxhat = dy * g
    dx = inv_std * (dxhat - dxhat.mean(-1, keepdims=True)
                    - xhat * (dxhat * xhat).mean(-1, keepdims=True))
    dg = (dy * xhat).sum(axis=tuple(range(dy.ndim - 1)))
    db = dy.sum(axis=tuple(range(dy.ndim - 1)))
    return dx.astype(F32), dg.astype(F32), db.astype(F32)


def _linear_fwd(x, w):
    return x @ w


def _linear_bwd(x, w, dy):
    b, t, _ = x.shape
    xf = x.reshape(b * t, -1)
    dyf = dy.reshape(b * t, -1)
    return (dy @ w.T).astype(F32), (xf.T @ dyf).astype(F32)


def effective_weight(base_w, lora, name, scale):
    """W_eff = W + (alpha/r) * A @ B for adapted projections."""
    if lora is None or name + ".A" not in lora:
        return base_w
    return (base_w + scale * (lora[name + ".A"] @ lora[name + ".B"])).astype(F32)


def _split_heads(x, n_heads):
    b, t, d = x.shape
    return x.reshape(b, t, n_heads, d // n_heads).transpose(0, 2, 1, 3)


def _merge_heads(x):
    b, h, t, dh = x.shape
    return x.transpose(0, 2, 1, 3).reshape(b, t, h * dh)


# ------------------------------------------------------------------- forward

def forward(base, lora, tokens, config):
    """Causal-LM forward pass. Returns (loss, cache) for backward."""
    n_layers, n_heads = config["n_layers"], config["n_heads"]
    scale_lora = config["lora_alpha"] / config["lora_rank"]
    b, t = tokens.shape
    dh = config["d_model"] // n_heads
    att_scale = 1.0 / np.sqrt(dh)
    mask = np.triu(np.full((t, t), -1e9, dtype=F32), k=1)

    x = (base["embed"][tokens] + base["pos"][:t]).astype(F32)
    cache = {"tokens": tokens, "x0": x, "layers": []}
    for i in range(n_layers):
        p = f"l{i}."
        wq = effective_weight(base[p + "wq"], lora, p + "wq", scale_lora)
        wv = effective_weight(base[p + "wv"], lora, p + "wv", scale_lora)
        a, ln1_c = _layernorm_fwd(x, base[p + "ln1.g"], base[p + "ln1.b"])
        q = _split_heads(_linear_fwd(a, wq), n_heads)
        k = _split_heads(_linear_fwd(a, base[p + "wk"]), n_heads)
        v = _split_heads(_linear_fwd(a, wv), n_heads)
        scores = np.einsum("bhid,bhjd->bhij", q, k) * att_scale + mask
        scores -= scores.max(-1, keepdims=True)
        probs = np.exp(scores)
        probs /= probs.sum(-1, keepdims=True)
        o = _merge_heads(np.einsum("bhij,bhjd->bhid", probs, v))
        att = _linear_fwd(o, base[p + "wo"])
        h = x + att
        bi, ln2_c = _layernorm_fwd(h, base[p + "ln2.g"], base[p + "ln2.b"])
        z1 = _linear_fwd(bi, base[p + "w1"]) + base[p + "b1"]
        r1 = np.maximum(z1, 0.0)
        f = _linear_fwd(r1, base[p + "w2"]) + base[p + "b2"]
        x2 = h + f
        cache["layers"].append(
            dict(x=x, a=a, ln1=ln1_c, q=q, k=k, v=v, probs=probs, o=o,
                 wq=wq, wv=wv, h=h, bi=bi, ln2=ln2_c, z1=z1, r1=r1))
        x = x2.astype(F32)

    xf, lnf_c = _layernorm_fwd(x, base["lnf.g"], base["lnf.b"])
    logits = _linear_fwd(xf, base["wout"])
    cache.update(x_last=x, xf=xf, lnf=lnf_c)

    pred = logits[:, :-1].astype(np.float64)
    targets = tokens[:, 1:]
    pred -= pred.max(-1, keepdims=True)
    lse = np.log(np.exp(pred).sum(-1, keepdims=True))
    picked = np.take_along_axis(pred, targets[..., None].astype(np.int64), -1)
    count = b * (t - 1)
    loss = float((lse - picked).sum() / count)
    soft = np.exp(pred - lse)
    np.put_along_axis(soft, targets[..., None].astype(np.int64),
                      np.take_along_axis(soft, targets[..., None].astype(np.int64), -1) - 1.0, -1)
    dlogits = np.zeros_like(logits)
    dlogits[:, :-1] = (soft / count).astype(F32)
    cache["dlogits"] = dlogits
    return loss, cache


# ------------------------------------------------------------------ backward

def backward(base, lora, cache, config, train_base=False):
    """Backpropagate from the cached loss. Always returns LoRA grads when
    adapters are present; base-parameter grads only if train_base."""
    n_heads = config["n_heads"]
    scale_lora = config["lora_alpha"] / config["lora_rank"]
    dh = config["d_model"] // n_heads
    att_scale = 1.0 / np.sqrt(dh)
    grads = {}

    dlogits = cache["dlogits"]
    dxf, dwout = _linear_bwd(cache["xf"], base["wout"], dlogits)
    if train_base:
        grads["wout"] = dwout
    dx, dg, db = _layernorm_bwd(dxf, cache["lnf"])
    if train_base:
        grads["lnf.g"], grads["lnf.b"] = dg, db

    for i in reversed(range(config["n_layers"])):
        p = f"l{i}."
        c = cache["layers"][i]
        # FFN branch: x2 = h + f
        dh_res, df = dx, dx
        dr1, dw2 = _linear_bwd(c["r1"], base[p + "w2"], df)
        dz1 = dr1 * (c["z1"] > 0)
        dbi, dw1 = _linear_bwd(c["bi"], base[p + "w1"], dz1)
        if train_base:
            grads[p + "w2"] = dw2
            grads[p + "b2"] = df.sum((0, 1)).astype(F32)
            grads[p + "w1"] = dw1
            grads[p + "b1"] = dz1.sum((0, 1)).astype(F32)
        dln2, dg2, db2 = _layernorm_bwd(dbi, c["ln2"])
        if train_base:
            grads[p + "ln2.g"], grads[p + "ln2.b"] = dg2, db2
        dhh = dh_res + dln2
        # attention branch: h = x + att
        dx_res, datt = dhh, dhh
        do_m, dwo = _linear_bwd(c["o"], base[p + "wo"], datt)
        if train_base:
            grads[p + "wo"] = dwo
        do = _split_heads(do_m, n_heads)
        probs, q, k, v = c["probs"], c["q"], c["k"], c["v"]
        dprobs = np.einsum("bhid,bhjd->bhij", do, v)
        dv = np.einsum("bhij,bhid->bhjd", probs, do)
        ds = probs * (dprobs - (dprobs * probs).sum(-1, keepdims=True))
        dq = np.einsum("bhij,bhjd->bhid", ds, k) * att_scale
        dk = np.einsum("bhij,bhid->bhjd", ds, q) * att_scale
        dq_f, dk_f, dv_f = (_merge_heads(z).astype(F32) for z in (dq, dk, dv))
        da_q, dwq_eff = _linear_bwd(c["a"], c["wq"], dq_f)
        da_k, dwk = _linear_bwd(c["a"], base[p + "wk"], dk_f)
        da_v, dwv_eff = _linear_bwd(c["a"], c["wv"], dv_f)
        if lora is not None and p + "wq.A" in lora:
            grads[p + "wq.A"] = (scale_lora * dwq_eff @ lora[p + "wq.B"].T).astype(F32)
            grads[p + "wq.B"] = (scale_lora * lora[p + "wq.A"].T @ dwq_eff).astype(F32)
            grads[p + "wv.A"] = (scale_lora * dwv_eff @ lora[p + "wv.B"].T).astype(F32)
            grads[p + "wv.B"] = (scale_lora * lora[p + "wv.A"].T @ dwv_eff).astype(F32)
        if train_base:
            grads[p + "wq"], grads[p + "wk"], grads[p + "wv"] = dwq_eff, dwk, dwv_eff
        dln1, dg1, db1 = _layernorm_bwd(da_q + da_k + da_v, c["ln1"])
        if train_base:
            grads[p + "ln1.g"], grads[p + "ln1.b"] = dg1, db1
        dx = (dx_res + dln1).astype(F32)

    if train_base:
        dE = np.zeros_like(base["embed"])
        np.add.at(dE, cache["tokens"], dx)
        grads["embed"] = dE
        dpos = np.zeros_like(base["pos"])
        dpos[:dx.shape[1]] = dx.sum(0)
        grads["pos"] = dpos
    return grads


# ------------------------------------------------------------------ training

def lora_init(config, tee_rng):
    """A ~ N(0, 0.02), B = 0 on each layer's W_q / W_v (standard LoRA init)."""
    d, r = config["d_model"], config["lora_rank"]
    lora = {}
    for i in range(config["n_layers"]):
        for proj in ("wq", "wv"):
            lora[f"l{i}.{proj}.A"] = tee_rng.normal((d, r), scale=0.02)
            lora[f"l{i}.{proj}.B"] = np.zeros((r, d), dtype=F32)
    return lora


def base_init(config, tee_rng):
    d, ff, v, t = (config["d_model"], config["d_ff"], config["vocab"],
                   config["seq_len"])
    s = 0.02
    base = {"embed": tee_rng.normal((v, d), s), "pos": tee_rng.normal((t, d), s),
            "lnf.g": np.ones(d, dtype=F32), "lnf.b": np.zeros(d, dtype=F32),
            "wout": tee_rng.normal((d, v), s)}
    for i in range(config["n_layers"]):
        p = f"l{i}."
        for w in ("wq", "wk", "wv", "wo"):
            base[p + w] = tee_rng.normal((d, d), s)
        base[p + "w1"] = tee_rng.normal((d, ff), s)
        base[p + "b1"] = np.zeros(ff, dtype=F32)
        base[p + "w2"] = tee_rng.normal((ff, d), s)
        base[p + "b2"] = np.zeros(d, dtype=F32)
        base[p + "ln1.g"] = np.ones(d, dtype=F32)
        base[p + "ln1.b"] = np.zeros(d, dtype=F32)
        base[p + "ln2.g"] = np.ones(d, dtype=F32)
        base[p + "ln2.b"] = np.zeros(d, dtype=F32)
    return base


def sgd_step(params, grads, momentum_state, lr, momentum):
    for name, g in grads.items():
        buf = momentum_state.setdefault(name, np.zeros_like(g))
        buf *= momentum
        buf += g
        params[name] = (params[name] - lr * buf).astype(F32)


def train(base, lora, tokens_dataset, tee_rng, config, train_base=False):
    """Run config["epochs"] epochs of minibatch SGD. Mutates and returns the
    trainable parameter dict (lora, or base when train_base). Momentum state
    is invocation-local and discarded on return."""
    params = base if train_base else lora
    params = {k: np.array(v, dtype=F32) for k, v in params.items()}
    if train_base:
        base = params
    momentum_state = {}
    n = len(tokens_dataset)
    bs = config["batch_size"]
    losses = []
    for _ in range(config["epochs"]):
        order = tee_rng.permutation(n)
        epoch_loss, batches = 0.0, 0
        for s in range(0, n, bs):
            batch = tokens_dataset[order[s:s + bs]].astype(np.int64)
            loss, cache = forward(base, None if train_base else params,
                                  batch, config)
            grads = backward(base, None if train_base else params,
                             cache, config, train_base=train_base)
            sgd_step(params, grads, momentum_state, config["lr"],
                     config["momentum"])
            epoch_loss += loss
            batches += 1
        losses.append(epoch_loss / max(batches, 1))
    return params, losses
