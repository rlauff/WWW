#!/usr/bin/env bash


set -euo pipefail

command -v wasm-pack >/dev/null 2>&1 || {
  echo "wasm-pack not found. Install it with:  cargo install wasm-pack" >&2
  exit 1
}
rustup target list --installed 2>/dev/null | grep -q wasm32-unknown-unknown || {
  echo "Adding wasm32 target…"; rustup target add wasm32-unknown-unknown
}

cd rhombic_strips
echo "Building rhombic_strips wasm…"
wasm-pack build --target web --out-dir www/pkg --release

cd ..
cd subtext
echo "Building subtext wasm…"
wasm-pack build --target web

cd ..
cd quiver_mutations
echo "Building quiver_mutations wasm…"
wasm-pack build --target web
