# E4 - Trusted Computing Base (LOC breakdown)

Effective source lines (non-blank, non-comment, docstrings excluded) of the CLEAN runtime. The training kernel is inside the TCB for confinement but is non-semantic: its numerical behavior carries no security obligation.

| module | SLOC | trusted for |
|---|---:|---|
| `__init__.py` | 0 | package doc / import policy |
| `attest.py` | 76 | MRENCLAVE, CP_Cert / ARC issuance timing |
| `commit.py` | 48 | canonical serialization and commitments |
| `device.py` | 23 | device boundary (single ingress; verification positioned after the transfer, so the transport needs no trust) |
| `dlog.py` | 28 | compound erasure (append-only deletion record; the enclave retains only its head) |
| `gateway.py` | 161 | input confinement (sole mediation point; physical exclusion of Delta) |
| `kernel.py` | 204 | non-semantic (numerical behavior carries no security obligation) |
| `loader.py` | 14 | clean-state selection / deletion-free input parsing |
| `rng.py` | 23 | deletion-independent attested randomness |
| `runtime.py` | 55 | closed-input state transition; ARC issued only after success |
| `store.py` | 93 | temporal input closure (block commitments; re-verification on every fetch from untrusted storage) |
| **total TCB** | **725** | |

Non-TCB components (not trusted for the erasure guarantee):

| component | SLOC | role |
|---|---:|---|
| `verifier/` | 141 | client-side checking, imports nothing from the runtime |
| `harness/` | 2817 | data generation, attacks, benchmarks |

The 725-line TCB is small enough to audit directly, in contrast to the millions of lines in a PyTorch/CUDA training stack or the circuit + trusted-setup TCB of SNARK-based verifiable unlearning. Input confinement is additionally checked mechanically by an AST audit (tests/test_confinement.py).
