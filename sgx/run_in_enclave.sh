#!/usr/bin/env bash
# Run the evaluation inside a real enclave.
#
# The staging step is what makes the measurement mean something. sgx.trusted_files
# measures every file under a trusted directory, so the tree handed to the
# enclave has to contain exactly the runtime, the verifier and the harness --
# no bytecode caches, no editor leftovers. Staging from git's view of the
# working tree gives that, and gives it reproducibly.
#
#   ./run_in_enclave.sh [module ...]
#
# With no arguments it runs the whole lifecycle: initial training, then the three
# evaluation modules whose verdicts the paper reports.
set -euo pipefail
cd "$(dirname "$0")"
TRAINER="$(cd .. && pwd)"
SIZE="${ENCLAVE_SIZE:-4G}"
MODULES=("$@")
if [ ${#MODULES[@]} -eq 0 ]; then
  # compound comes early because it is the one module that copies a deployment
  # and writes to the encrypted mount, so it is where a platform difference
  # shows up first, and it is short.
  MODULES=(harness.sisa_train harness.compound harness.attack_matrix
           harness.fetch_probe harness.two_world harness.cert_cost)
fi

# A deployment is bound to the image that produced it: the certificates name a
# measurement, and the enclave's measurement is not the host's. So the enclave
# gets its own storage and its own results, and the two can then be compared.
mkdir -p "$TRAINER/data-sgx/tmp" "$TRAINER/results-sgx"

# Gramine's unlink on an encrypted mount leaves the host file in place, so a
# scenario that isolated the sealed state can leave a directory behind that the
# next image cannot open. Clearing it is the host's job, not the enclave's.
rm -rf "$PWD"/sealed/isolated-*

# Provisioning a deployment provisions its sealed state. Whatever is in the
# mount was sealed under some earlier measurement, and an enclave with a
# different one can neither read it nor replace it, so the host clears it --
# which is the only thing a host can do about sealed state it cannot open.
if [[ " ${MODULES[*]} " == *" harness.sisa_train "* ]]; then
  rm -f "$PWD"/sealed/*
fi

echo "== staging the measured tree =="
rm -rf stage
mkdir -p stage build sealed
for pkg in clean_runtime verifier harness; do
  mkdir -p "stage/$pkg"
  # Tracked files plus untracked ones git would not ignore. Tracked alone is
  # tempting for reproducibility and wrong in practice: a module written but not
  # yet committed is silently left out of the image, and the enclave then fails
  # on an import rather than on anything to do with the experiment. `git` is
  # still what decides, so bytecode caches and ignored build output stay out.
  git -C "$TRAINER" ls-files --cached --others --exclude-standard "$pkg/*.py" \
    | sort -u | while read -r f; do
    cp "$TRAINER/$f" "stage/$f"
  done
done
find stage -name '__pycache__' -prune -exec rm -rf {} + 2>/dev/null || true
echo "   $(find stage -name '*.py' | wc -l) files, $(cat stage/*/*.py | wc -l) lines"

docker run --rm --device /dev/sgx_enclave \
  -v "$PWD/stage":/clean:ro \
  -v "$TRAINER/data-sgx":/data \
  -v "$TRAINER/results-sgx":/results \
  -v "$PWD/build":/build \
  -v "$PWD/sealed":/sealed \
  -v "$PWD/runtime.manifest.template":/runtime.manifest.template:ro \
  -v "$PWD/enclave-key.pem":/key.pem:ro \
  clean-sgx:1 "cd /build && gramine-manifest \
      -Dlog_level=error -Dentrypoint=/usr/bin/python3.10 \
      -Darch_libdir=/lib/x86_64-linux-gnu \
      -Denclave_size=$SIZE -Dra_type=none \
      /runtime.manifest.template runtime.manifest \
    && gramine-sgx-sign --key /key.pem --manifest runtime.manifest \
      --output runtime.manifest.sgx >/dev/null \
    && echo -n 'MRENCLAVE from the signed image: ' \
    && gramine-sgx-sigstruct-view runtime.sig \
       | sed -n 's/^ *mr_enclave: *//p' \
    && for m in ${MODULES[*]}; do \
         echo \"== \$m ==\" ; \
         gramine-sgx runtime -m \$m || exit 1 ; \
       done"
