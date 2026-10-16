#!/bin/sh
# Slow vector sets kept out of push CI (scripts/check.sh): the BLAKE3
# 102400-byte cases, one mode per file (each near one comptime fuel budget).
set -e
RCE=${RCE:-rce}
cd "$(dirname "$0")/.."
tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
for c in tests/hash/blake3_nightly_*.rce; do
  { cat lib/hash/blake3.rce; echo; cat "$c"; } > "$tmp/$(basename "$c")"
  "$RCE" test "$tmp/$(basename "$c")"
done
# lib/ml: every checks file emitted to Python, JavaScript and Rust with
# identical observations, then each tests/<area>/<name>_emit.rce inference
# run (emitted Rust, judged against reference posteriors).
for m in ad tensor prob; do
  for c in tests/ml/${m}_checks.rce tests/ml/${m}_checks_*.rce; do
    RCE="$RCE" python3 scripts/emitgate.py targets "lib/ml/$m.rce" "$c" python js rust
  done
done
for e in tests/*/*_emit.rce; do
  n=$(basename "$e" _emit.rce); a=$(basename "$(dirname "$e")")
  RCE="$RCE" python3 scripts/emitgate.py run "lib/$a/$n.rce" "$e"
done
