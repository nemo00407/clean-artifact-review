"""Track 2 gate - can this GPU reproduce a LoRA step bit for bit?

Assumption 3 asks the deployment for a deterministic execution mode. The
0.9M NumPy prototype gets it by pinning one BLAS thread. Whether a real
accelerator can give the same thing is the question that decides whether the
two-world test means anything at 1B parameters, so it is answered first and
on its own.

The criterion is the paper's: SHA-256 over the canonical serialization of the
adapter, equal or not equal. Nothing statistical.

What has to be switched off, and why each one matters:

  TF32            Ampere runs fp32 matmuls through tensor cores at reduced
                  mantissa by default. Same inputs, same result, but it is a
                  different arithmetic than fp32 and the reduction order is
                  not specified.
  cuDNN benchmark cuDNN picks an algorithm by timing candidates on the first
                  call, so the algorithm -- and its reduction order -- depends
                  on what else the machine was doing.
  SDPA fast paths flash and mem-efficient attention accumulate in an order
                  that depends on scheduling. Only the math backend is
                  deterministic.
  Atomics         several backward kernels accumulate gradients with atomics,
                  whose order is not defined. use_deterministic_algorithms
                  makes torch raise instead of silently using them.
  cuBLAS workspace without a fixed workspace, cuBLAS may split reductions
                  differently between calls in the same process.
"""
import hashlib
import json
import os
import sys

# Must be set before torch initialises CUDA.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def configure(torch):
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)


def adapter_hash(params):
    """The paper's criterion: a hash over a canonical serialization."""
    h = hashlib.sha256()
    for name, t in sorted(params):
        h.update(name.encode())
        h.update(t.detach().to("cpu", copy=True).contiguous().numpy().tobytes())
    return h.hexdigest()


def one_run(torch, steps, seed, device, dim=512, rank=4, batch=8, seqlen=128):
    """A LoRA-shaped update: frozen base, low-rank adapter, fixed data.

    Deliberately not a real model. The gate is about whether the arithmetic
    is reproducible on this device, and a stand-in that exercises matmul,
    attention and a backward pass answers that without a 2 GB download.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    g = torch.Generator(device=device).manual_seed(seed)

    base = torch.randn(dim, dim, generator=g, device=device)      # frozen
    a = torch.randn(rank, dim, generator=g, device=device) * 0.01
    b = torch.zeros(dim, rank, device=device)
    a.requires_grad_(True)
    b.requires_grad_(True)
    data = torch.randn(steps, batch, seqlen, dim, generator=g, device=device)

    opt = torch.optim.AdamW([a, b], lr=1e-3)
    for i in range(steps):
        x = data[i]
        w = base + b @ a
        q = x @ w
        k = x @ w
        v = x @ w
        y = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        loss = (y * y).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return adapter_hash([("lora_a", a), ("lora_b", b)]), float(loss.item())


def main():
    try:
        import torch
    except ImportError:
        print(json.dumps({"error": "torch not installed"}))
        return 2
    out = {"torch": torch.__version__, "cuda": torch.version.cuda,
           "device": None, "cublas_workspace": os.environ.get("CUBLAS_WORKSPACE_CONFIG")}
    if not torch.cuda.is_available():
        out["error"] = "no CUDA device"
        print(json.dumps(out, indent=1))
        return 2
    out["device"] = torch.cuda.get_device_name(0)
    configure(torch)

    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    out["steps"] = steps
    try:
        h1, l1 = one_run(torch, steps, seed=1234, device="cuda")
        h2, l2 = one_run(torch, steps, seed=1234, device="cuda")
        h3, _ = one_run(torch, steps, seed=4321, device="cuda")
    except RuntimeError as e:
        out["error"] = f"RuntimeError: {e}"
        print(json.dumps(out, indent=1))
        return 2

    out.update({
        "run1": h1, "run2": h2, "different_seed": h3,
        "identical": h1 == h2,
        "criterion_discriminates": h1 != h3,
        "final_loss": l1, "loss_identical": l1 == l2,
    })
    print(json.dumps(out, indent=1))
    return 0 if out["identical"] and out["criterion_discriminates"] else 1


if __name__ == "__main__":
    sys.exit(main())
