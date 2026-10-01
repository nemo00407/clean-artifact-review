# CLEAN — Certified Learning Erasure with Attested Non-Interference

Artifact for the paper of the same title. It contains the runtime whose
boundary the paper's guarantee rests on, an independent verifier, the harness
that produces every number the paper reports, the Intel SGX build inputs, the
test suites, and the measurement outputs the tables and figures are drawn from.

The paper's claim is an information-flow property of an execution: an accepted
Attested Retraining Certificate (ARC) implies that the deleted records lay
outside the computation's input domain. Two things in this repository are worth
separating on a first read. `clean_runtime/` is the measured runtime — the part
the argument trusts, and the part an SGX measurement covers. `harness/` is the
untrusted host: it drives experiments, and it also plays the red team. The
verifier in `verifier/` imports nothing from the runtime, on purpose, so that a
verification pass never trusts runtime code.

```
clean_runtime/    the measured runtime: gateway, store, attest, commit, dlog,
                  rng, loader, device, and the training kernel it fronts
verifier/         the independent auditor, re-implementing every hash and
                  encoding it checks
harness/          untrusted host driver and red team; one module per experiment
tests/            regression and confinement tests (43 tests)
tools/            source-derived reports (trusted-computing-base line counts)
sgx/              Gramine manifests, the Docker build, and the scripts that
                  run the evaluation inside a real enclave
results/          measurements taken on the host
results-sgx/      the same experiments re-run inside the enclave
```

`results/README.md` is the index from result file to the paper's experiment
labels.

## What is not in here, and why

**Generated deployments.** `data/`, `data-1b/` and `data-sgx/` are built by the
harness, not shipped. Every corpus is synthetic and generated from a recorded
seed, so a deployment is reproduced rather than distributed. The commands below
build them.

**The enclave signing key.** `sgx/enclave-key.pem` is a private key and is not
distributed. Generate your own; the measurement that matters (`MRENCLAVE`) does
not depend on which key signs the image.

**Sealed enclave state.** `sgx/sealed/` holds the enclave's signing seed under
its sealing key. It is state, not source, and is recreated on the first run.

**Pretrained weights.** The 1.1B-parameter experiments fine-tune a publicly
available pretrained transformer, identified below and downloaded by the
harness. It is not redistributed here.

## Environment

Two tiers. Everything except the 1.1B-parameter experiments needs only NumPy
and `cryptography`; the rest is the standard library.

```bash
python3 -m venv .venv
.venv/bin/pip install numpy==2.5.2 cryptography==50.0.0 pytest==9.1.1
```

Measurements here were taken on Python 3.12.3. `requirements.txt` pins the same
versions, and additionally the PyTorch stack the 1.1B experiments need:

```bash
.venv/bin/pip install -r requirements.txt   # adds torch, transformers, peft, accelerate
```

`torch` must match the local CUDA; the measurements used `torch 2.6.0+cu124` on
a 12 GB consumer GPU. Without it, every CPU experiment still runs; the four
modules that need it (`harness/sisa_train_1b.py`, `harness/two_world_1b.py`,
`harness/retrain_cost_1b.py`, `harness/determinism_gate_llm.py`, through
`harness/gpu_kernel.py`) are the only ones that do not.

For the enclave half you additionally need a machine with Intel SGX and
Flexible Launch Control, the SGX platform software with `aesmd` running, and
Docker. `results/NOTES-hardware.md` records the platform every number was taken
on, including the SGX stack and EPC size.

Run every command below from the repository root.

## Quick check

Builds a deployment and runs the test suite. About two minutes on the recorded
platform, CPU only.

```bash
.venv/bin/python -m harness.sisa_train && .venv/bin/python -m pytest tests -q
```

`harness.sisa_train` performs the trusted initial lifecycle — sharding, slicing
and certifying a checkpoint at each slice boundary — and writes a deployment to
`data/`. The tests then exercise the runtime, the gateway's admission checks,
the verifier, payload confinement, and the two soundness properties that are
easiest to get wrong: that a deletion is not readmitted through rows compaction
has not reached yet, and that a restored sealed state does not buy a valid
certificate.

## Reproducing the paper's results

### Everything, in one command

```bash
.venv/bin/python -m harness.run_all
```

This rewrites `results/`. It runs the initial lifecycle first, then each
experiment in the order the paper presents them. The corpus-scale sweeps run
last because they reach 10^8 manifest records and take the longest.

### One experiment at a time

Each module writes its own files under `results/`. The paper's experiment
labels are in the first line of each output file and in `results/README.md`.

| command | paper label | produces |
|---|---|---|
| `python -m harness.two_world` | E1 | `noninterference.md` — two-world bitwise equality, swept |
| `python -m harness.attacks` | E3 | `attacks.md` |
| `python -m harness.attack_matrix` | E3b | `attack_matrix.md` — the attack × ablation matrix |
| `python -m harness.fetch_probe` | E9 | `abort_point.md` |
| `python -m harness.mutation` | E10 | `mutation.md` — mutation testing of the confinement audit |
| `python -m harness.compound` | E13 | `compound.md` — compound erasure and four restoration attempts |
| `python -m harness.cost_sweeps` | E11, E12 | `verification_cost.md`, `temporal_overhead.md` |
| `python -m harness.bench` | E7, E8 | `commitment_cost.md`, `rollback.md` |
| `python -m harness.cert_cost` | E17 | `cert_cost.md` |
| `python -m harness.scale_sweeps` | E14, E15 | `scale_verify.md`, `scale_prefix.md` |
| `python tools/loc_report.py` | E4 | `tcb_loc.md` — trusted-computing-base line counts |

`harness.scale_sweeps` accepts `--reuse` to keep the manifests a previous run
built instead of regenerating them.

`results/comparison.md` (E5) is the one table not measured here: it restates
what the compared systems report about themselves, and
`results/NOTES-baselines.md` records which claim came from which paper.

### At 1.1B parameters

Needs the PyTorch tier and a CUDA GPU. Determinism is a precondition rather
than an outcome, so the gate runs first and is meant to be read before the
sweep: two runs of one configuration must agree bitwise, and a run under
different enclave randomness must not.

```bash
.venv/bin/python -m harness.determinism_gate_llm     # the go/no-go gate
.venv/bin/python -m harness.sisa_train_1b            # builds data-1b/
.venv/bin/python -m harness.two_world_1b             # E16 -> two_world_1b.md
.venv/bin/python -m harness.retrain_cost_1b          # E18 -> retrain_1b.md
```

The 1.1B payload is a second implementation of the same interface, substituted
at the module binding the runtime already resolves. Nothing in
`clean_runtime/` changes between the two scales, which is the point of running
it: the obligation-carrying lines are the same bytes at both.

### Inside a real SGX enclave

First build the image and generate a signing key of your own:

```bash
docker build -t clean-sgx:1 sgx/
docker run --rm -v "$PWD/sgx":/work clean-sgx:1 \
  "gramine-sgx-gen-private-key /work/enclave-key.pem"
```

The staging step decides what the enclave measures, and it asks `git` rather
than the filesystem, so bytecode caches and build output cannot drift into the
measured image. If you obtained this artifact as an archive rather than as a
repository, initialise one first:

```bash
git init -q . && git add -A && git -c user.name=a -c user.email=a@b commit -qm artifact
```

Then run the evaluation inside the enclave:

```bash
sgx/run_in_enclave.sh                 # the whole lifecycle
sgx/run_in_enclave.sh harness.compound harness.attack_matrix   # selected modules
```

Results go to `results-sgx/`, and the deployment to `data-sgx/`, because a
deployment is bound to the image that produced it: the certificates name a
measurement, and the enclave's measurement is not the host's. The script prints
the `MRENCLAVE` of the signed image, which is the value
`gramine-sgx-sigstruct-view` reports and the value the certificates carry.

`ENCLAVE_SIZE` sets the enclave size and defaults to `4G`, deliberately larger
than the 94 MB EPC of the recorded platform, so that paging is measured rather
than avoided.

The expected outcome is that every verdict matches the host run byte for byte.
The boundary is what changed, not the results:

```bash
for f in compound.md attack_matrix.md abort_point.md noninterference.md; do
  diff -q results/$f results-sgx/$f && echo "$f identical"
done
```

One further script measures what a host gains by supplying a runtime of its own
choosing:

```bash
sgx/tamper_test.sh
```

It appends one comment line to a runtime module — a line the interpreter never
executes — and shows that the measurement changes, so `EGETKEY` derives a
different sealing key and the enclave cannot open its sealed signing seed at
all. It issues no certificate, rather than one an auditor would reject.

## Reading the evidence

`harness/attack_matrix.py` is the most direct place to see what the paper
claims. Each row is a host strategy; each column is the system with one binding
removed from the auditor, or with the runtime's read-path re-verification
disabled. A cell says whether the adversary obtained a certificate the auditor
accepted. No attack succeeds against the full system, and each attack succeeds
exactly where the defence that stops it has been removed.

`verifier/verify.py` is the whole audit: six conditions, no model parameter
read. Its `unsafe_skip` argument exists only to build those ablations, and
`known_head` is the freshness token an auditor carries between sessions.

`harness/mutation.py` measures the confinement audit by planting escapes in the
payload one at a time and reporting which the audit catches. The paper reports
the classes it misses as well as the ones it catches.

## Notes on determinism

The runtime's guarantee needs bit-exact reproducibility within one deployment,
not across platforms; the verifier evaluates hashes and signatures and
re-executes nothing. At 0.9M parameters the kernel is NumPy and deterministic
by construction. At 1.1B the contract is met by configuring deterministic
algorithms, fixed precision, a math-only attention backend and a fixed
accumulation order; `harness/determinism_gate_llm.py` is the check, and the
settings it applies are in that module.

Two runs of the same experiment on the same machine should therefore produce
identical `results/*.md` for every verdict. Timings will differ: they are
medians over repetitions, and the microsecond-scale ones move by a few percent
between runs, and further under load.
