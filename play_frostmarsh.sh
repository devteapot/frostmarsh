#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
SERVER="${FROSTMARSH_SERVER:-127.0.0.1:4433}"
cargo run --release -p frostmarsh-unified-client -- --server "$SERVER" "$@"
