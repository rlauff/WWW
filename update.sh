#!/bin/bash
# Exit immediately if a command exits with a non-zero status
set -e 

echo "Pulling latest changes for the main website..."
git pull origin main 

echo "Updating subrepos to their latest upstream commits..."
# --remote fetches the latest from the submodule's remote branch
# --merge merges it into the local submodule checkout
git submodule update --remote --merge

echo "Recompiling WASM binaries..."
./build_wasms.sh

echo "Update and build complete!"

echo "Delete target directories in submodules"
cd subtext/
rm -rf target
cd ../quiver_mutations/ 
rm -rf target
cd ../rhombic_strips/
rm -rf target
