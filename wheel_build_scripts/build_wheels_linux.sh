#!/bin/bash
set -e

if [ -z "$PYTHON_VERSION" ]; then
  echo "Please provide a Python version in the PYTHON_VERSION env variable"
  exit 1
fi

INSTALL_DIR=/github/workspace/deps_build

pushd /tmp

python_exec=$(which python$PYTHON_VERSION)
$python_exec -m venv myvenv
source ./myvenv/bin/activate

yum install -y suitesparse-devel
python -m pip install --upgrade pip
pip install --upgrade setuptools wheel auditwheel

git clone --branch v0.4.16 --single-branch https://github.com/scikit-sparse/scikit-sparse
LD_LIBRARY_PATH=/usr/lib64:$LD_LIBRARY_PATH pip install --editable scikit-sparse

popd
echo "copying sksparse into graphite_maps/sksparse"
cp -r /tmp/scikit-sparse/sksparse graphite_maps/sksparse
cp /tmp/scikit-sparse/LICENSE.txt graphite_maps/LICENSE_scikit_sparse.txt
cp wheel_build_scripts/LICENSE_libsuitesparse_cholmod.txt ./LICENSE_libsuitesparse_cholmod.txt
cp wheel_build_scripts/LICENSE_libsuitesparse.txt ./LICENSE_libsuitesparse.txt

# Breaks the install, fixes the wheel
echo "adding wheel-only setup.py..."
mv wheel_build_scripts/setup_wheels_only.py setup.py

echo "Making wheel..."

if [ "$IS_TAG" = "true" ]; then
  echo "Making dirty wheel appear clean..."
  echo "Setting SETUPTOOLS_SCM_PRETEND_VERSION to ${GITHUB_REF_NAME}"
  export SETUPTOOLS_SCM_PRETEND_VERSION="${GITHUB_REF_NAME}"
  echo "Set env SETUPTOOLS_SCM_PRETEND_VERSION to ${GITHUB_REF_NAME}"
fi
LD_LIBRARY_PATH=/usr/lib64:$LD_LIBRARY_PATH pip wheel .
echo "Done making wheel"

pip install auditwheel
wheel_path=$(find . -name "graphite*.whl")
auditwheel repair $wheel_path -w fixed_wheel
fixed_wheel_path=$(find fixed_wheel -name "*.whl")

echo "Replacing broken wheel @ $wheel_path with fixed wheel @ $fixed_wheel_path"
rm $wheel_path
mv $fixed_wheel_path $(dirname $wheel_path)
