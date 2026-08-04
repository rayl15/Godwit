#!/usr/bin/env bash
# Test runner.
#
# Swift Testing ships inside the Command Line Tools rather than the toolchain's
# default search paths, so a CLT-only machine (no full Xcode) needs the
# framework path at compile time and two rpaths at link time. Xcode users do not
# need any of this, but passing it is harmless.
#
# Tests run serially: once Metal state exists, in-process parallelism makes
# GPU tests unreliable.

set -euo pipefail

cd "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/.."

developer_dir="$(xcode-select -p)"
frameworks="$developer_dir/Library/Developer/Frameworks"
interop_lib="$developer_dir/Library/Developer/usr/lib"

if [[ ! -d "$frameworks" ]]; then
  # Full Xcode keeps Swift Testing somewhere the toolchain already searches.
  exec swift test --no-parallel "$@"
fi

exec swift test --no-parallel \
  -Xswiftc -F -Xswiftc "$frameworks" \
  -Xlinker -rpath -Xlinker "$frameworks" \
  -Xlinker -rpath -Xlinker "$interop_lib" \
  "$@"
