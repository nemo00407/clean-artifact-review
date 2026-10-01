# E16 - Two-world equality at 1B parameters

Payload: `harness/gpu_kernel.py`, LoRA rank 4 over `TinyLlama/TinyLlama-1.1B-Chat-v1.0` on the GPU. The gateway, read path, commitments, certificates and verifier are the runtime's own code, unchanged.

| configuration | records deleted | $L^A = L^B$ | verifies | seconds |
|---|---|---|---|---:|
| s0 slice1 n1 | 13 | yes | yes | 14.0 |
| s0 slice1 n2 | 13, 15 | yes | yes | 3.8 |
| s0 slice2 n1 | 25 | yes | yes | 2.7 |
| s0 slice2 n2 | 25, 27 | yes | yes | 2.5 |
| s0 slice3 n1 | 39 | yes | yes | 1.3 |
| s0 slice3 n2 | 39, 40 | yes | yes | 1.1 |
| s1 slice1 n1 | 22 | yes | yes | 4.1 |
| s1 slice1 n2 | 22, 26 | yes | yes | 3.8 |
| s1 slice2 n1 | 42 | yes | yes | 2.7 |
| s1 slice2 n2 | 42, 45 | yes | yes | 2.5 |
| s1 slice3 n1 | 56 | yes | yes | 1.3 |
| s1 slice3 n2 | 56, 57 | yes | yes | 1.1 |

12/12 configurations produced byte-identical adapters across the two worlds; 12/12 issued a certificate the independent verifier accepted.
