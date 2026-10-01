#!/usr/bin/env bash
# What a host gains by editing the runtime it is asked to run.
#
# Track 1 showed with a throwaway probe that a different measurement cannot open
# sealed state. This asks the same question of the control plane itself, and the
# edit is the one a malicious host would actually make: change a line of the
# runtime, keep everything else, and run it.
#
# Step 1 runs the staged image and reads the sealed signing key.
# Step 2 appends one comment line to clean_runtime/store.py -- which changes
#        nothing the interpreter executes -- and runs again.
#
# The enclave in step 2 has a different MRENCLAVE, so EGETKEY hands it a
# different sealing key, so /sealed/enclave.sealed is not merely wrong for it:
# it cannot be opened. An enclave that cannot open the sealed key cannot sign,
# so it cannot issue a certificate at all.
set -uo pipefail
cd "$(dirname "$0")"
TRAINER="$(cd .. && pwd)"
RES="$TRAINER/results"

run () {   # $1 = label
  docker run --rm --device /dev/sgx_enclave \
    -v "$PWD/stage":/clean:ro \
    -v "$TRAINER/data-sgx":/data \
    -v "$TRAINER/results-sgx":/results \
    -v "$PWD/build":/build -v "$PWD/sealed":/sealed \
    -v "$PWD/runtime.manifest.template":/runtime.manifest.template:ro \
    -v "$PWD/enclave-key.pem":/key.pem:ro \
    clean-sgx:1 "cd /build && gramine-manifest \
        -Dlog_level=error -Dentrypoint=/usr/bin/python3.10 \
        -Darch_libdir=/lib/x86_64-linux-gnu \
        -Denclave_size=4G -Dra_type=none \
        /runtime.manifest.template runtime.manifest \
      && gramine-sgx-sign --key /key.pem --manifest runtime.manifest \
        --output runtime.manifest.sgx >/dev/null \
      && gramine-sgx-sigstruct-view runtime.sig | sed -n 's/^ *mr_enclave: */MRENCLAVE /p' \
      && gramine-sgx runtime -c 'import json, pathlib, sys
from harness import run_unlearn
from clean_runtime import gateway
print(\"REPORTED\", gateway.runtime_measurement().hex())
try:
    e = run_unlearn.load_enclave()
    print(\"SEALED_READ ok\")
    print(\"PK\", e.pk.hex())
except Exception as exc:
    print(\"SEALED_READ refused:\", type(exc).__name__, exc)'" 2>&1 \
  | grep -aE "^MRENCLAVE|^REPORTED|^SEALED_READ|^PK" | sed "s/^/[$1] /"
}

echo "== step 1: the staged image, unmodified =="
run step1 | tee /tmp/tamper_step1.txt

echo "== step 2: one comment line appended to clean_runtime/store.py =="
cp stage/clean_runtime/store.py /tmp/store.py.orig
echo "# a line the host added, which the interpreter never executes" \
  >> stage/clean_runtime/store.py
run step2 | tee /tmp/tamper_step2.txt
cp /tmp/store.py.orig stage/clean_runtime/store.py
echo "   (store.py restored)"

python3 - "$RES" <<'PY'
import json, pathlib, re, sys
def parse(p):
    out = {}
    for line in pathlib.Path(p).read_text().splitlines():
        line = re.sub(r"^\[\w+\] ", "", line)
        k, _, v = line.partition(" ")
        out[k] = v
    return out
s1, s2 = parse("/tmp/tamper_step1.txt"), parse("/tmp/tamper_step2.txt")
res = {
 "step1": s1, "step2": s2,
 "measurement_changed": s1.get("MRENCLAVE") != s2.get("MRENCLAVE"),
 "reported_matches_sigstruct": (s1.get("MRENCLAVE") == s1.get("REPORTED")
                                and s2.get("MRENCLAVE") == s2.get("REPORTED")),
 "step1_opened_sealed": s1.get("SEALED_READ", "").startswith("ok"),
 "step2_opened_sealed": s2.get("SEALED_READ", "").startswith("ok"),
}
out = pathlib.Path(sys.argv[1]) / "sgx_tamper.json"
out.write_text(json.dumps(res, indent=1))
print(json.dumps({k: v for k, v in res.items() if k.startswith(("m", "r", "s1", "s2"))
                  and not k.startswith("step")}, indent=1))
print("step1 sealed:", s1.get("SEALED_READ"))
print("step2 sealed:", s2.get("SEALED_READ"))
print("->", out)
PY
