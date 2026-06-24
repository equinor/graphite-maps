#!/bin/bash
# Build SuiteSparse (cholmod + deps) from source in manylinux_2_28; yum's 4.4.6
# is too old for scikit-sparse >= 0.5 (needs SuiteSparse >= 7.4).
set -euo pipefail

SUITESPARSE_VERSION="${SUITESPARSE_VERSION:-v7.12.2}"
SUITESPARSE_PREFIX="${SUITESPARSE_PREFIX:-/opt/suitesparse}"

echo "Building SuiteSparse ${SUITESPARSE_VERSION} -> ${SUITESPARSE_PREFIX}"

yum install -y openblas-devel  # BLAS/LAPACK for the CHOLMOD Supernodal module

if ! command -v cmake >/dev/null 2>&1; then
  python3 -m pip install "cmake>=3.22"
fi

workdir="$(mktemp -d)"
trap 'rm -rf "${workdir}"' EXIT
pushd "${workdir}" >/dev/null

curl -fsSL \
  "https://github.com/DrTimothyAldenDavis/SuiteSparse/archive/refs/tags/${SUITESPARSE_VERSION}.tar.gz" \
  -o suitesparse.tar.gz
tar xzf suitesparse.tar.gz
cd SuiteSparse-*/

# Root CMake auto-adds cholmod's deps (suitesparse_config, amd, colamd, camd,
# ccolamd). METIS/Partition stays on for the "metis" ordering graphite-maps uses.
cmake -B build \
  -DCMAKE_INSTALL_PREFIX="${SUITESPARSE_PREFIX}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DSUITESPARSE_ENABLE_PROJECTS="cholmod" \
  -DSUITESPARSE_USE_CUDA=OFF \
  -DSUITESPARSE_USE_FORTRAN=OFF \
  -DSUITESPARSE_USE_OPENMP=ON \
  -DBUILD_STATIC_LIBS=OFF \
  -DBUILD_TESTING=OFF

cmake --build build --parallel "$(nproc)"
cmake --install build

popd >/dev/null

cholmod_header="${SUITESPARSE_PREFIX}/include/suitesparse/cholmod.h"
if [ ! -f "${cholmod_header}" ]; then
  echo "ERROR: ${cholmod_header} not found after install" >&2
  exit 1
fi
echo "Installed CHOLMOD version:"
grep -h "define CHOLMOD_MAIN_VERSION\|define CHOLMOD_SUB_VERSION\|define CHOLMOD_SUBSUB_VERSION" \
  "${cholmod_header}"
