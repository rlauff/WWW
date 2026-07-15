#!/bin/bash
set -euo pipefail

# 1. Update the main website
echo "Pulling latest changes for the main website..."
git pull origin main

# Define the live website directory as the current directory (WWW root)
LIVE_DIR="$PWD"

echo "Checking dependencies..."
command -v wasm-pack >/dev/null 2>&1 || {
  echo "wasm-pack not found. Install it with: cargo install wasm-pack" >&2
  exit 1
}
rustup target list --installed 2>/dev/null | grep -q wasm32-unknown-unknown || {
  echo "Adding wasm32 target..."; rustup target add wasm32-unknown-unknown
}

# 2. Delete the old deployments
echo "Removing old deployed tools..."
rm -rf "$LIVE_DIR/rhombic_strips" "$LIVE_DIR/subtext" "$LIVE_DIR/quiver_mutations"

# 3. Create a secure, temporary build environment
BUILD_DIR=$(mktemp -d)
echo "Created temporary workspace at $BUILD_DIR"

# Fail-safe cleanup: Ensure BUILD_DIR is deleted when script finishes or fails
trap 'echo "Cleaning up temporary workspace..."; rm -rf -- "$BUILD_DIR"' EXIT

cd "$BUILD_DIR"

# 4. Function to clone, build, and extract ONLY necessary web files
build_and_extract() {
  local repo_url=$1
  local repo_name=$2
  local wasm_out_dir=$3

  echo "==> Processing $repo_name..."
  
  # Clone shallowly for max speed
  git clone --depth 1 "$repo_url" "$repo_name"
  cd "$repo_name"

  # Build the WASM
  echo "Building WASM for $repo_name..."
  wasm-pack build --target web --out-dir "$wasm_out_dir" --release

  TARGET_DIR="$LIVE_DIR/$repo_name"
  echo "Extracting web files to $TARGET_DIR..."
  
  # Find and copy ONLY .html, .js, .css, .wasm, and .stx files.
  # We use -prune to completely ignore the .git and target directories to save time.
  find . -type d \( -name ".git" -o -name "target" \) -prune -o \
         -type f \( -name "*.html" -o -name "*.js" -o -name "*.css" -o -name "*.wasm" -o -name "*.stx" \) -print0 | 
  while IFS= read -r -d '' file; do
    # Remove the leading './' from the found file path
    clean_path="${file#./}"
    
    # Create the necessary subdirectories in the live folder
    mkdir -p "$TARGET_DIR/$(dirname "$clean_path")"
    
    # Copy the file over
    cp "$file" "$TARGET_DIR/$clean_path"
  done

  # Return to the temp workspace root
  cd "$BUILD_DIR"
}

# 5. Execute the Builds
build_and_extract "https://github.com/rlauff/rhombic_strips.git" "rhombic_strips" "www/pkg"
build_and_extract "https://github.com/rlauff/subtext.git" "subtext" "pkg"
build_and_extract "https://github.com/rlauff/quiver_mutations.git" "quiver_mutations" "pkg"

echo "Deploy complete! The temporary workspace will now be destroyed."
