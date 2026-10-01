#!/usr/bin/env bash
# Track 1 measurements. Writes JSON into ../results for the report to read.
#
# The sealing experiment runs in three deliberate steps because the question
# is not "does sealing work" but "is the key bound to the measurement":
#   1  a fresh enclave seals state
#   2  the same enclave, run again, reads it back
#   3  a different enclave (different manifest, hence different MRENCLAVE)
#      is offered the same file
set -euo pipefail
cd "$(dirname "$0")"
RES=../results
mkdir -p "$RES" sealed

probe () {  # $1 = enclave size, marks the measurement
  ./build.sh "$1" none 2>&1 | sed -n '/^{/,/^}/p'
}

echo "== sealing: is the key bound to the measurement? =="
rm -f sealed/*
S1=$(probe 1G); echo "$S1" > "$RES/sgx_seal_step1.json"
S2=$(probe 1G); echo "$S2" > "$RES/sgx_seal_step2.json"
S3=$(probe 2G); echo "$S3" > "$RES/sgx_seal_step3.json"
python3 - "$RES" <<'PY'
import json, sys, pathlib
r = pathlib.Path(sys.argv[1])
for i in (1, 2, 3):
    d = json.loads((r / f"sgx_seal_step{i}.json").read_text())
    s = d.get("sealing") or {}
    print(f"  step{i}  mr={d['measurement']['mr_enclave'][:16]}"
          f"  inherited={s.get('inherited_previous')}"
          f"  {'seal %.0f us / unseal %.0f us' % (s['seal_s']*1e6, s['unseal_s']*1e6) if s.get('seal_s') else s.get('inherit_error') or d.get('sealing_error','')}")
PY

echo "== EPC pressure on the retained commitment set =="
SIZES="1000 10000 100000 1000000 3000000 6000000 10000000"
APP=/app/epc.py APPARGS="$SIZES" ./build.sh 2G none 2>&1 \
  | grep '^JSON ' | cut -c6- > "$RES/sgx_epc.json"
docker run --rm -v "$PWD/app":/app:ro clean-sgx:1 \
  "python3 /app/epc.py $SIZES" 2>&1 | grep '^JSON ' | cut -c6- > "$RES/native_epc.json"
echo "  wrote $RES/sgx_epc.json, $RES/native_epc.json"
