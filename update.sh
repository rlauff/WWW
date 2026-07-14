#!/bin/bash
set -euo pipefail

echo "Pulling latest changes for the main website..."
git pull origin main 

echo "Updating subrepos to their latest upstream commits..."
git submodule update --remote --merge

echo "Checking dependencies..."
command -v wasm-pack >/dev/null 2>&1 || {
  echo "wasm-pack not found. Install it with: cargo install wasm-pack" >&2
  exit 1
}

rustup target list --installed 2>/dev/null | grep -q wasm32-unknown-unknown || {
  echo "Adding wasm32 target..."
  rustup target add wasm32-unknown-unknown
}

# Function to build and clean a specific submodule
build_module() {
  local dir=$1
  local out_dir=${2:-pkg} # Defaults to 'pkg' if no second argument is provided

  echo "Building $dir wasm..."
  (
    # The parentheses create a subshell. We cd in, build, clean, and 
    # when the subshell exits, we are automatically back in the root folder.
    cd "$dir"
    wasm-pack build --target web --out-dir "$out_dir" --release
    
    echo "Cleaning target directory for $dir..."
    rm -rf target
  )
}

echo "Recompiling WASM binaries..."
build_module "rhombic_strips" "www/pkg"
build_module "subtext"
build_module "quiver_mutations"

echo "Update and build complete!"
