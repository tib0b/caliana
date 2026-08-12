#!/usr/bin/env bash
# Build the C++ differential-testing oracle from the original pystackreg sources.
#
# The pystackreg checkout the C++ comes from is deliberately not part of this
# repository — it is an unmodified upstream tree, kept beside the caliana
# checkout. Point PYSTACKREG at it if yours lives somewhere else; without it the
# differential tests skip themselves and the rest of `cargo test` still runs.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
root="$here/.."
pystackreg="${PYSTACKREG:-$root/../../pystackreg}"

if [ ! -d "$pystackreg/src" ]; then
  echo "no pystackreg checkout at $pystackreg — set PYSTACKREG to yours" >&2
  exit 1
fi

src="$pystackreg/src"
inc="$pystackreg/inc"

g++ -O2 -std=c++14 -I"$inc" \
  "$root/oracle/oracle.cpp" \
  "$src/TurboReg.cpp" \
  "$src/TurboRegImage.cpp" \
  "$src/TurboRegMask.cpp" \
  "$src/TurboRegPointHandler.cpp" \
  "$src/TurboRegTransform.cpp" \
  -o "$root/oracle/oracle"

echo "built $root/oracle/oracle"
