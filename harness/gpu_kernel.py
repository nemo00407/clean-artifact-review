"""A second payload for the same slot: LoRA on a 1B decoder, on the GPU.

`clean_runtime/kernel.py` is 204 lines of NumPy and carries, by the paper's
own accounting, no security obligation. If that is true then something else
can sit in the slot, and the guarantee should be unchanged. This module is
that something else: a different model, a different arithmetic, a different
device, behind the same five-argument interface.

    train(base, lora, tokens_dataset, tee_rng, config) -> (lora, losses)

What it does *not* get to change is how it is fed. Batches are drawn from
`tokens_dataset`, which the gateway hands over as a verified view, so every
fetch still re-derives the commitment taken at admission. Shuffling comes
from `tee_rng`, so the host has no seed to choose. Optimizer state is created
per call and discarded, as in the NumPy kernel, so two invocations given the
same inputs start from the same place.

The base model is frozen and public. Following Equation (1), an invocation
does not re-hash it: `base` carries the commitment C_base fixed at
deployment, and this module checks the weights it loads against it once.
"""
import hashlib
import json
import os
import pathlib

import numpy as np

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

F32 = np.float32
_CACHE = {}


def _torch():
    import torch
    if not _CACHE.get("configured"):
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
        _CACHE["configured"] = True
    return torch


def base_commitment(model_id):
    """C_base over the published weights, computed once at deployment."""
    from huggingface_hub import snapshot_download
    root = pathlib.Path(snapshot_download(model_id, allow_patterns=[
        "*.safetensors", "*.json", "tokenizer*"]))
    h = hashlib.sha256()
    for p in sorted(root.rglob("*.safetensors")):
        h.update(p.name.encode())
        with p.open("rb") as f:
            for chunk in iter(lambda: f.read(1 << 22), b""):
                h.update(chunk)
    return h.digest(), str(root)


def _model(config):
    """The frozen base, loaded once and checked against C_base."""
    key = config["model_id"]
    if key in _CACHE:
        return _CACHE[key]
    torch = _torch()
    from transformers import AutoModelForCausalLM
    digest, root = base_commitment(key)
    if bytes.fromhex(config["c_base_hex"]) != digest:
        raise ValueError("base model does not match the committed C_base")
    m = AutoModelForCausalLM.from_pretrained(
        root, dtype=torch.float32, attn_implementation="eager").cuda()
    for p in m.parameters():
        p.requires_grad_(False)
    m.eval()                       # frozen base: no dropout, no running stats
    _CACHE[key] = m
    return m


def _targets(model, config):
    """The projections the adapter attaches to, in a fixed order."""
    names = config["target_modules"]
    out = []
    for name, mod in model.named_modules():
        if name.split(".")[-1] in names and hasattr(mod, "weight"):
            out.append((name, mod))
    return sorted(out, key=lambda kv: kv[0])


def lora_init(config, tee_rng):
    """A ~ N(0, 0.02), B = 0, drawn from the attested randomness."""
    model = _model(config)
    r = config["lora_rank"]
    lora = {}
    for name, mod in _targets(model, config):
        din = mod.weight.shape[1]
        dout = mod.weight.shape[0]
        lora[f"{name}.A"] = tee_rng.normal((din, r), scale=0.02)
        lora[f"{name}.B"] = np.zeros((r, dout), dtype=F32)
    return lora


class _Adapted:
    """Wraps one projection as W + (B^T A^T) * scale, with A and B trainable."""

    def __init__(self, torch, mod, a, b, scale):
        self.mod, self.scale = mod, scale
        self.a = torch.nn.Parameter(torch.from_numpy(a).cuda())
        self.b = torch.nn.Parameter(torch.from_numpy(b).cuda())
        self.orig = mod.forward

        def fwd(x, _s=self):
            return _s.orig(x) + (x @ _s.a @ _s.b) * _s.scale
        mod.forward = fwd

    def detach(self):
        self.mod.forward = self.orig

    def numpy(self):
        return (self.a.detach().cpu().numpy().astype(F32),
                self.b.detach().cpu().numpy().astype(F32))


def train(base, lora, tokens_dataset, tee_rng, config, train_base=False):
    """One retraining pass over the admitted rows. Mirrors kernel.train."""
    if train_base:
        raise NotImplementedError("the GPU payload adapts only; the base is frozen")
    torch = _torch()
    model = _model(config)
    scale = config["lora_alpha"] / config["lora_rank"]

    adapters, order_keys = {}, []
    for name, mod in _targets(model, config):
        a, b = lora[f"{name}.A"], lora[f"{name}.B"]
        adapters[name] = _Adapted(torch, mod, np.array(a), np.array(b), scale)
        order_keys.append(name)

    params = []
    for name in order_keys:
        params += [adapters[name].a, adapters[name].b]
    opt = torch.optim.AdamW(params, lr=config["lr"], betas=(0.9, 0.999),
                            eps=1e-8, weight_decay=0.0)

    n = len(tokens_dataset)
    bs = config["batch_size"]
    losses = []
    try:
        for _ in range(config["epochs"]):
            order = tee_rng.permutation(n)
            total, nb = 0.0, 0
            for s in range(0, n, bs):
                rows = order[s:s + bs]
                batch = np.asarray(tokens_dataset[rows], dtype=np.int64)
                ids = torch.from_numpy(batch).cuda()
                out = model(input_ids=ids, labels=ids)
                opt.zero_grad(set_to_none=True)
                out.loss.backward()
                opt.step()
                total += float(out.loss.item())
                nb += 1
            losses.append(total / max(nb, 1))
        new = {}
        for name in order_keys:
            a, b = adapters[name].numpy()
            new[f"{name}.A"], new[f"{name}.B"] = a, b
    finally:
        for ad in adapters.values():
            ad.detach()
        torch.cuda.empty_cache()
    return new, losses
