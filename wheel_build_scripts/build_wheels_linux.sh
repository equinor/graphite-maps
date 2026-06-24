#!/bin/bash
# Build a manylinux_2_28 (RHEL8-compatible) graphite-maps wheel with a bundled,
# cholmod-only scikit-sparse 0.5.x, linked against a from-source SuiteSparse 7.x.
set -e

# The manylinux container runs as a different user than the host that owns
# /github/workspace. Without this, git (and setuptools_scm) refuses to operate.
git config --global --add safe.directory /github/workspace

if [ -z "$PYTHON_VERSION" ]; then
  echo "Please provide a Python version in the PYTHON_VERSION env variable"
  exit 1
fi

export SUITESPARSE_PREFIX=/opt/suitesparse
sh wheel_build_scripts/build_suitesparse_linux.sh

# Resolve the SuiteSparse lib dir (lib vs lib64) and let the build + auditwheel
# find libcholmod and its SuiteSparse/OpenBLAS dependencies.
suitesparse_lib="$(dirname "$(find "$SUITESPARSE_PREFIX" -name 'libcholmod.so*' | head -n1)")"
export SUITESPARSE_INCLUDE_DIR="$SUITESPARSE_PREFIX/include/suitesparse"
export SUITESPARSE_LIBRARY_DIR="$suitesparse_lib"
export LD_LIBRARY_PATH="$suitesparse_lib:/usr/lib64:${LD_LIBRARY_PATH:-}"

python_exec=$(which python"$PYTHON_VERSION")
$python_exec -m venv /tmp/myvenv
source /tmp/myvenv/bin/activate

python -m pip install --upgrade pip
pip install --upgrade setuptools wheel auditwheel "Cython>=3.0" "numpy>=2.0"

git clone --branch v0.5.0 --single-branch \
  https://github.com/scikit-sparse/scikit-sparse /tmp/scikit-sparse
sh wheel_build_scripts/vendor_scikit_sparse.sh /tmp/scikit-sparse

cp wheel_build_scripts/LICENSE_libsuitesparse_cholmod.txt ./LICENSE_libsuitesparse_cholmod.txt
cp wheel_build_scripts/LICENSE_libsuitesparse.txt ./LICENSE_libsuitesparse.txt

# Breaks the source install, fixes the wheel: see setup_wheels_only.py.
echo "adding wheel-only setup.py..."
mv wheel_build_scripts/setup_wheels_only.py setup.py

if [ "$IS_TAG" = "true" ]; then
  echo "Setting SETUPTOOLS_SCM_PRETEND_VERSION to ${GITHUB_REF_NAME}"
  export SETUPTOOLS_SCM_PRETEND_VERSION="${GITHUB_REF_NAME}"
fi

echo "Making wheel..."
pip wheel . --no-deps -w .

wheel_path=$(find . -maxdepth 1 -name "graphite*.whl")
auditwheel repair "$wheel_path" -w fixed_wheel
fixed_wheel_path=$(find fixed_wheel -name "*.whl")

echo "Replacing unrepaired wheel @ $wheel_path with repaired wheel @ $fixed_wheel_path"
rm "$wheel_path"
mv "$fixed_wheel_path" "$(dirname "$wheel_path")"
