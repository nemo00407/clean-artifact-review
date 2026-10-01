# What the compared works actually report

Extracted from the cited papers themselves so the comparison table can be stated
like-for-like instead of as "seconds" and "heavy". Page-level figures only;
nothing here is inferred.

## Eisenhofer et al., *Verifiable and Provably Secure Machine Unlearning*
(`SNARK.pdf`, arXiv:2210.09126v3, S&P 2025)

Spartan over ristretto255, circuits compiled to R1CS via ZoKrates + CirC,
Poseidon as the in-circuit hash.

**Headline instantiation** (Table I): linear regression, 3 epochs, batch size 1,
a **synthetic dataset of 100 points with 10 features**, deleting 10 points.

| | retraining | amnesiac | optimization |
|---|---|---|---|
| proof of unlearning, R1CS constraints | 8,102,288 | 616,005 | 919,456 |
| Prove | 4m 58s | 2m 18s | 0m 53s |
| **Verify** | **1m 48s** | **0m 49s** | **0m 20s** |
| proof of training, Verify | 1m 36s | 1m 37s | 1m 35s |

**Proof-system variant matters.** The above is the NIZK variant. With the SNARK
variant of Spartan, verification drops to **< 1s** but proving rises to
33m 39s--34m 12s and the user runs a one-time preprocessing of 8m 6s--8m 12s.
Proof of non-membership is < 1s either way.

**Dataset scaling** (Appendix E, PMLB, linear regression, 3 epochs). Size is
#points x #features. They state a *linear* dependence of runtime on dataset size.

| dataset | points | features | R1CS | Prove | **Verify** |
|---|---:|---:|---:|---:|---:|
| Creditscore | 100 | 6 | 3,986,308 | 2m 22s | **0m 47s** |
| Patient | 88 | 8 | 4,579,718 | 2m 28s | 0m 53s |
| Cy Young | 92 | 10 | 5,903,988 | 3m 16s | 1m 9s |
| Corral | 160 | 6 | 6,347,236 | 3m 43s | 1m 15s |
| Lawsuit | 264 | 4 | 7,190,981 | 4m 7s | 1m 27s |
| Breast cancer | 286 | 9 | 16,514,048 | 9m 25s | 3m 18s |
| Monk3 | 554 | 6 | 21,841,281 | 13m 36s | **4m 32s** |

**Model scaling** (Table II), 100 points x 10 features throughout:
linear regression 1m 36s verify; logistic 1m 45s; NN(N=2) 3m 34s;
NN(N=4) 6m 42s. So *their* verification grows with the model as well.

**They name our design point as the alternative.** Section VII-A, "Trusted
Hardware": a TEE can replace the SNARK, and the concerns they list are trusting
the vendor, vendor availability for signing, **limited memory [32]**,
**applicability to ML tasks that involve GPU computation [62]**, and
side-channels. Our EPC measurement quantifies the third and W3 is the fourth.
They also place Weng et al. (PoUL) inside their framework.

## Abbaszadeh et al., *Zero-Knowledge Proofs of Training for DNNs* (Kaizen)
(`zkPoT.pdf`, CCS '24)

Sumcheck/GKR-style PoGD plus an aggregation scheme, recursion tree, Orion
polynomial commitments. Proof size and verifier time are **independent of the
number of iterations**.

Models and data: LeNet 61,706 params on MNIST (60,000 images); AlexNet 4.2M and
VGG-11 10.1M params on CIFAR-10 (60,000 images), both truncated.

At batch size N = 16:

| | LeNet (61.7K) | AlexNet (4.2M) | VGG-11 (10.1M) |
|---|---:|---:|---:|
| prover, per iteration | 193 s | 474 s | 882 s |
| peak prover memory | 134.4 GB | 269.1 GB | **466.3 GB** |
| proof size | 1.021 MB | 1.255 MB | 1.627 MB |
| **verifier** | **73 ms** | **86 ms** | **130 ms** |

Dataset Merkle root (MiMC-p^2/p^2): **18 s for MNIST, 218 s for CIFAR-10**, i.e.
for 60,000 records. Circuit sizes for the generic-IVC comparison: 429M gates
(LeNet), 1.6B (AlexNet), 5.1B (VGG-11).

## The comparison this licenses

The two baselines scale along **the opposite axis from CLEAN**, and that is the
substantive point rather than a ratio.

- Eisenhofer et al.: verification is linear in the dataset and grows with the
  model. 47 s at 100 records; 4m 32s at 554 records.
- Kaizen: verification is succinct in the dataset and the iteration count, and
  grows with the model. 73 ms at 61.7K params, 130 ms at 10.1M.
- CLEAN: verification is linear in the manifest and **constant in the model**.

So CLEAN is not uniformly cheaper to verify. Kaizen's verifier beats ours at a
large manifest, because ours reads the manifest and theirs does not. Where the
gap is decisive is the producing side and the memory it needs: 882 s and 466 GB
per iteration for a 10.1M-parameter model, against a certificate produced in
sub-millisecond time that does not grow with the model at all.

## Weng et al., *Proof of Unlearning: Definitions and Instantiation*
(`Proof_of_Unlearning_Definitions_and_Instantiation.pdf`, IEEE TIFS vol. 19, 2024)

Checked because the comparison table claims their constant verification cost is a design
property with no latency reported. **That is correct.** Section V-C states the
goal ("verification cost is expected to be small and constant") and the cuckoo
filter is what delivers it; the evaluation (Section VI-B) reports *storage* and
*learning/unlearning time*, not verification latency:

- Purchase dataset, 56,073 points in one shard, 2^16 buckets, 12-bit
  fingerprints, FPR ~0.003 -> filter 192 KB, `key_list` ~2.85 MB, so the
  authentication layer adds **~3.00 MB** per shard.
- Unlearning time by the slice the deleted point falls in (their Fig. 5a), with
  checkpoint and restore "negligible" against learning time (Fig. 5b).
- Baseline accuracy 95.14% on Purchase, 22 epochs, batch 1000.

They also hit our EPC wall: "due to the memory restriction of the SGX enclave,
the trained accuracy mentioned before is obtained by carefully tuning
hyperparameters during the SGD training." Their response to limited EPC is to
shrink the model; ours is to keep only commitments resident and measure where
that set stops fitting.
