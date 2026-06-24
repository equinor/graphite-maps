#!/bin/bash
# Vendor the cholmod-only subset of a scikit-sparse 0.5.x checkout into the repo
# root as a top-level `sksparse` package (graphite-maps only uses cholmod).
# Usage: vendor_scikit_sparse.sh <scikit-sparse-checkout-dir>
set -euo pipefail

src="${1:?usage: vendor_scikit_sparse.sh <scikit-sparse-checkout-dir>}"
dest="sksparse"

rm -rf "${dest}"
mkdir -p "${dest}"
for f in __init__.py utils.py cholmod.pyx cholmod.pxd; do
  cp "${src}/src/sksparse/${f}" "${dest}/${f}"
done
cp "${src}/LICENSE.txt" ./LICENSE_scikit_sparse.txt

echo "Vendored sksparse.cholmod subset into ./${dest}:"
ls -1 "${dest}"
