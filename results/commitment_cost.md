# E7 - Commitment and Attestation Cost

Prototype adapter 16.0 KiB, base 1320.0 KiB. Hashing throughput 554 MB/s.

| operation | time |
|---|---:|
| composite C(W) = H(C_base || H(L)) | 42.0 us |
| full-model hash H(W_base || L) | 2438.0 us |
| ARC signing (Ed25519) | 32.8 us |

Composite commitment only re-hashes the adapter each unlearning (C_base is a deployment-time constant). Extrapolated to LLM scale (rank-16 LoRA ~ 0.2% of parameters):

| model | full-model hash | adapter-only C(W) | speedup |
|---|---:|---:|---:|
| 7B | 50502 ms | 101.0 ms | 500x |
| 70B | 505024 ms | 1010.0 ms | 500x |
