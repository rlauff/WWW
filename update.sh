#!/bin/bash
set -euo pipefail

# Never hang on a credential prompt: a private/missing repo should fail loudly
export GIT_TERMINAL_PROMPT=0

# --------------------------------------------------------------------------
# Usage:
#   ./update.sh              full deploy (pull site, rebuild WASM, publish all)
#   ./update.sh --quick      skip the WASM projects, publish dead_or_alive only
#                            (aliases: -q, --only-dead-or-alive, --no-wasm)
# --------------------------------------------------------------------------
WASM=1
for arg in "$@"; do
  case "$arg" in
    -q|--quick|--no-wasm|--only-dead-or-alive) WASM=0 ;;
    -h|--help) sed -n '/^# Usage:/,/^# ---/p' "$0"; exit 0 ;;
    *) echo "unknown option: $arg (try --help)" >&2; exit 1 ;;
  esac
done
[ "$WASM" -eq 1 ] || echo "Quick mode: skipping the WASM projects."

# 1. Update the main website
echo "Pulling latest changes for the main website..."
git pull origin main

# Define the live website directory as the current directory (WWW root)
LIVE_DIR="$PWD"

if [ "$WASM" -eq 1 ]; then
  echo "Checking dependencies..."
  command -v wasm-pack >/dev/null 2>&1 || {
    echo "wasm-pack not found. Install it with: cargo install wasm-pack" >&2
    exit 1
  }
  rustup target list --installed 2>/dev/null | grep -q wasm32-unknown-unknown || {
    echo "Adding wasm32 target..."; rustup target add wasm32-unknown-unknown
  }
fi

# 2. Delete the old deployments
#    dead-or-alive is deliberately NOT deleted: reviewers move problems
#    between its candidates/ and accepted/ folders on the live server, and a
#    wipe would throw those decisions away. It is copied over in place below.
if [ "$WASM" -eq 1 ]; then
  echo "Removing old deployed tools..."
  rm -rf "$LIVE_DIR/rhombic_strips" "$LIVE_DIR/subtext" "$LIVE_DIR/quiver_mutations"
fi

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

# 4b. Static projects: nothing to compile, just publish the repo as-is.
#     cp -r (not the filtered find above) so the .json problem data and
#     manifests come along too. The repo's own layout is published verbatim
#     -- no directories are created, moved or removed here. Existing files
#     are overwritten, extra files on the server are left alone: that is
#     what preserves the review decisions made on the live server.
deploy_static() {
  local repo_url=$1
  local repo_name=$2

  echo "==> Processing $repo_name (static, no build)..."
  git clone --depth 1 "$repo_url" "$repo_name"

  local target="$LIVE_DIR/$repo_name"
  mkdir -p "$target"
  # publish the working tree, minus git's own bookkeeping
  rm -rf "$BUILD_DIR/$repo_name/.git"
  cp -r "$BUILD_DIR/$repo_name/." "$target/"
  find "$target" -name '*.cgi' -exec chmod 755 {} +

  if [ ! -f "$target/web/manifest.json" ] && \
     [ ! -f "$target/web/problems.json" ]; then
    echo "  WARNING: no manifest published for $repo_name --" \
         "the page will load but show no problems." >&2
  fi
  cd "$BUILD_DIR"
}

# 5. Execute the Builds
if [ "$WASM" -eq 1 ]; then
  build_and_extract "https://github.com/rlauff/rhombic_strips.git" "rhombic_strips" "www/pkg"
  build_and_extract "https://github.com/rlauff/subtext.git" "subtext" "pkg"
  build_and_extract "https://github.com/rlauff/quiver_mutations.git" "quiver_mutations" "pkg"
fi

deploy_static "https://github.com/rlauff/dead-or-alive.git" "dead-or-alive"

echo "Deploy complete! The temporary workspace will now be destroyed."

