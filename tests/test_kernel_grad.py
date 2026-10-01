"""Numerical gradient checks for the training kernel (tiny config)."""
import numpy as np

from clean_runtime import kernel
from clean_runtime.rng import TeeRng

CFG = {"n_layers": 1, "d_model": 8, "n_heads": 2, "d_ff": 16, "vocab": 17,
       "seq_len": 5, "lora_rank": 2, "lora_alpha": 4, "lr": 0.1,
       "momentum": 0.9, "epochs": 1, "batch_size": 2}


def _setup():
    rng = TeeRng(b"\x07" * 32)
    base = kernel.base_init(CFG, rng)
    lora = kernel.lora_init(CFG, rng)
    # make B nonzero so dA is informative
    for k in lora:
        if k.endswith(".B"):
            lora[k] = rng.normal(lora[k].shape, 0.02)
    tokens = np.array([[1, 5, 9, 13, 2], [3, 3, 8, 0, 16]], dtype=np.int64)
    return base, lora, tokens


def _loss(base, lora, tokens):
    loss, _ = kernel.forward(base, lora, tokens, CFG)
    return loss


def _directional_check(params, name, grads, feval, tol=3e-2, eps=2e-3):
    g = grads[name]
    direction = np.random.default_rng(11).normal(size=g.shape)
    direction /= np.linalg.norm(direction)
    orig = params[name].copy()
    params[name] = (orig + eps * direction).astype(np.float32)
    up = feval()
    params[name] = (orig - eps * direction).astype(np.float32)
    down = feval()
    params[name] = orig
    numeric = (up - down) / (2 * eps)
    analytic = float((g * direction).sum())
    denom = max(abs(numeric), abs(analytic), 1e-6)
    # float32 loss evaluation bounds the resolvable absolute error
    ok = abs(numeric - analytic) / denom < tol or abs(numeric - analytic) < 2e-6
    assert ok, f"{name}: numeric={numeric:.6g} analytic={analytic:.6g}"


def test_lora_grads():
    base, lora, tokens = _setup()
    _, cache = kernel.forward(base, lora, tokens, CFG)
    grads = kernel.backward(base, lora, cache, CFG)
    feval = lambda: _loss(base, lora, tokens)
    for name in ("l0.wq.A", "l0.wq.B", "l0.wv.A", "l0.wv.B"):
        _directional_check(lora, name, grads, feval)


def test_base_grads():
    base, lora, tokens = _setup()
    _, cache = kernel.forward(base, None, tokens, CFG)
    grads = kernel.backward(base, None, cache, CFG, train_base=True)
    feval = lambda: _loss(base, None, tokens)
    for name in ("wout", "embed", "pos", "l0.w1", "l0.b2", "l0.wk",
                 "l0.wo", "l0.ln1.g", "lnf.b"):
        _directional_check(base, name, grads, feval)


def test_training_reduces_loss():
    base, lora, tokens = _setup()
    data = np.tile(tokens, (8, 1)).astype(np.uint8)
    cfg = dict(CFG, epochs=30, lr=0.5)
    rng = TeeRng(b"\x09" * 32)
    before = _loss(base, lora, data[:2].astype(np.int64))
    new_lora, losses = kernel.train(base, lora, data, rng, cfg)
    assert losses[-1] < before - 0.05, (before, losses[-1])
