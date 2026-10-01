# Measurement platform

Recorded 2026-09-20T16:53Z. Every figure produced by `harness/` after
this date was taken here unless the module says otherwise.

## CPU and confidential computing

| | |
|---|---|
| CPU | Intel(R) Core(TM) i5-9600KF CPU @ 3.70GHz |
| Cores / threads | 6 |
| SGX | sgx sgx_lc (Flexible Launch Control, so self-signed enclaves need no Intel launch token) |
| EPC | EPC section 0x80200000-0x85ffffff, 94 MB |
| SGX stack | PSW 2.27.100.1-noble1, DCAP 1.24.100.2-noble1, aesmd active |
| Gramine | gramineproject/gramine:stable-jammy (run in Docker, so no host user needs to be in the `sgx` group) |

## GPU

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 3080 Ti, 12288 MiB |
| Compute capability | 8.6 (Ampere) |
| Driver | 595.84 |
| Confidential mode | **not available** --- CC mode begins with Hopper, so the GPU-TEE half of the architecture stays emulated |

## Memory and storage

| | |
|---|---|
| RAM | 15 GB total, about 10 GB available to a sweep when otherwise idle |
| Disk | 286G free |

## What this platform can and cannot show

The control plane can run in a real enclave: of the 521 security-relevant
runtime lines, `gateway`, `store`, `attest`, `runtime`, `commit`, `dlog`,
`rng` and `loader` are all CPU-side. The 204-line training kernel, which
carries no security obligation, is the part that stays outside on an Ampere
card. Nothing here can demonstrate confinement of a payload inside a GPU-TEE.
