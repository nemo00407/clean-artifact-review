"""Track 2 gate, second half - the same question on a real 1B model.

The synthetic gate showed the device can reproduce matmul, attention and a
backward pass bit for bit. That is necessary and not sufficient: a real
decoder stack reaches operators the stand-in never touches, and
torch.use_deterministic_algorithms raises rather than silently using a
nondeterministic kernel. Whichever way this goes, it is the answer -- a
RuntimeError here is a finding, not a bug to route around.

LoRA freezes the base, so no gradient reaches the embedding, which is where
most of the nondeterministic scatter kernels live. Whether that is enough is
what we are measuring.
"""
import hashlib
import json
import os
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

from .determinism_gate import adapter_hash, configure          # noqa: E402

MODEL = os.environ.get("GATE_MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")


def one_run(torch, steps, seed, batch=2, seqlen=256, rank=4):
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.float32, attn_implementation="eager")
    cfg = LoraConfig(r=rank, lora_alpha=16, lora_dropout=0.0, bias="none",
                     target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM")
    model = get_peft_model(model, cfg).cuda()
    model.train()

    g = torch.Generator(device="cpu").manual_seed(seed)
    vocab = model.config.vocab_size
    ids = torch.randint(0, vocab, (steps, batch, seqlen), generator=g).cuda()

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=1e-4)
    for i in range(steps):
        x = ids[i]
        out = model(input_ids=x, labels=x)
        opt.zero_grad(set_to_none=True)
        out.loss.backward()
        opt.step()

    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    h = adapter_hash(named)
    n_train = sum(p.numel() for _, p in named)
    n_total = sum(p.numel() for p in model.parameters())
    del model, ids
    torch.cuda.empty_cache()
    return h, float(out.loss.item()), n_train, n_total


def main():
    import torch
    out = {"model": MODEL, "torch": torch.__version__}
    if not torch.cuda.is_available():
        out["error"] = "no CUDA device"
        print(json.dumps(out, indent=1)); return 2
    out["device"] = torch.cuda.get_device_name(0)
    configure(torch)
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    out["steps"] = steps
    try:
        h1, l1, ntr, ntot = one_run(torch, steps, seed=1234)
        h2, l2, _, _ = one_run(torch, steps, seed=1234)
        h3, _, _, _ = one_run(torch, steps, seed=4321)
    except RuntimeError as e:
        out["error"] = f"RuntimeError: {e}"
        out["verdict"] = ("an operator in this stack has no deterministic "
                          "implementation on CUDA")
        print(json.dumps(out, indent=1)); return 2
    out.update({"trainable_params": ntr, "total_params": ntot,
                "run1": h1, "run2": h2, "different_seed": h3,
                "identical": h1 == h2, "criterion_discriminates": h1 != h3,
                "final_loss": l1, "loss_identical": l1 == l2,
                "peak_vram_gb": torch.cuda.max_memory_allocated() / 1e9})
    print(json.dumps(out, indent=1))
    return 0 if out["identical"] and out["criterion_discriminates"] else 1


if __name__ == "__main__":
    sys.exit(main())
