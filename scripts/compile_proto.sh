#!/usr/bin/env bash
# Regenerate Python protobuf bindings from proto/schema.proto.
#
# Generated code is not committed (see .gitignore) -- run this after cloning
# and any time proto/schema.proto changes. CI should run it too and fail on
# a diff so a stale generated/ tree can never ship silently.
#
# The C++ target (--cpp_out) is intentionally not generated here yet; it has
# no consumer until src/ipc/ (the native ZeroMQ client) is implemented.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

if ! command -v protoc >/dev/null 2>&1; then
  echo "error: protoc not found. Install it first, e.g.:" >&2
  echo "  sudo apt-get install -y protobuf-compiler" >&2
  exit 1
fi

mkdir -p generated/python
protoc -I proto \
  --python_out=generated/python \
  --pyi_out=generated/python \
  proto/schema.proto

# generated/python isn't a package on its own; make it importable as
# `generated.python.schema_pb2` for scripts run from the repo root.
touch generated/__init__.py generated/python/__init__.py

echo "Generated bindings in generated/python/ from proto/schema.proto"
