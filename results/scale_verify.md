# E14 - Verification at corpus scale

`Verify` is run whole at every point: the deletion request's membership in the manifest, the checkpoint certificate, the clean-prefix test, the compound-erasure test, the $H(\mathcal{D}_{clean})$ recomputation and both signature checks. **published** is `verifier/verify.py` unmodified, which takes the manifest as a list of (index, hex digest) pairs and sorts it before hashing. **streaming** is the same set of checks folded into one pass over the manifest, holding one Merkle subtree per height instead of the manifest.

| records | published | peak RSS | streaming | peak RSS |
|---:|---:|---:|---:|---:|
| 128 | 0.3 ms | 0.03 GB | 0.4 ms | 0.03 GB |
| 512 | 0.7 ms | 0.03 GB | 1.0 ms | 0.03 GB |
| 2,048 | 2.4 ms | 0.03 GB | 3.4 ms | 0.03 GB |
| 8,192 | 8.9 ms | 0.03 GB | 13.0 ms | 0.03 GB |
| 32,768 | 36.5 ms | 0.05 GB | 51.2 ms | 0.03 GB |
| 100,000 | 118.6 ms | 0.09 GB | 170.6 ms | 0.03 GB |
| 1,000,000 | 1.21 s | 0.60 GB | 1.56 s | 0.03 GB |
| 3,000,000 | 3.79 s | 1.75 GB | 4.85 s | 0.03 GB |
| 10,000,000 | 12.41 s | 5.31 GB | 16.25 s | 0.03 GB |
| 100,000,000 | --- | --- | 157.14 s | 0.03 GB |

Both curves are linear in time. What separates them is memory: the published verifier holds about 531 bytes per record -- an entry, its hex digest, the two index maps it builds and the sorted copy it hashes -- so a manifest of $10^9$ records would ask for roughly 531\,GB before the first hash is taken. The streaming pass holds 32\,MB at every point measured, including the largest.

It does not pay for that in time. The two timed regions do not contain the same work -- the published verifier is handed a list, while the streaming pass produces each entry as it consumes it, 83.17 s of the 157.14 s at 100,000,000 records. Net of producing the manifest, the streaming pass costs 0.64$\times$ the published verifier over the decades where the manifest dominates. Measured as they stand, the ratio is 1.4$\times$.

Times are the median over repetitions (50 at the small sizes down to a single run at $10^8$); peak RSS is what the verifier holds while running, the manifest included.

Under a 9\,GB address-space cap the published verifier raises `MemoryError` at 100,000,000 records. The streaming pass returns `accept` on the same certificate.

The two agree on the verdict, not merely on the timing. Every accepting point is already evidence that the folded root is bit-identical to the level-wise one, because the certificate binds a root the harness folded and the published verifier recomputes it level-wise before accepting. Giving one retained record the wrong digest makes both return `H(D_clean) does not match the manifest minus the request and the deletion record`.
