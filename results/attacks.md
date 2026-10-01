# E3 - Attack Interception Matrix

Red-team attacks present genuinely enclave-signed ARCs over tampered inputs. The full verifier rejects every attack; the verifier ablation with the named binding removed then accepts, showing each defense is load-bearing.

| attack vector | CLEAN (full) | verifier ablation | defense |
|---|---|---|---|
| Retraining Skipping | REJECT | - | external forgery; caught by ARC unforgeability |
| Certificate Replay | REJECT | ACCEPT (vuln.) | freshness: nonce binding; NoNonce ablation accepts |
| Dataset Tampering | REJECT | ACCEPT (vuln.) | H(D_clean) recomputation; NoDataBind ablation accepts |
| Checkpoint Substitution | REJECT | ACCEPT (vuln.) | CP_Cert binding; NoCPCert ablation accepts |
| Unclean Checkpoint | REJECT | ACCEPT (vuln.) | clean-rollback judgment; NoClean ablation accepts |
| Mid-Invocation Shard Rollback | REJECT | ACCEPT (vuln.) | block re-verification on fetch; NoReverify ablation yields an ARC the full verifier accepts |
| PCIe Transfer Tampering | REJECT | ACCEPT (vuln.) | commitment checked after arrival on the device; NoReverify ablation yields an ARC the full verifier accepts |
| Configuration Steering | REJECT | ACCEPT (vuln.) | H(c) against the deployed configuration; NoConfigBind ablation accepts |
| Randomness Manipulation | REJECT (API) | - | no seed interface; r is enclave-derived and bound via H(r) in the ARC |

Honest run verifies (completeness). Verify() over 2000 calls: median 0.331 ms, p99 0.389 ms.

All attacks rejected by full verifier: True. Every structural ablation re-exposes its attack: True.