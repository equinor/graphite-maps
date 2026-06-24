"""Wheel-only setup.py for building graphite-maps with a vendored, cholmod-only
scikit-sparse.

This file is moved to ./setup.py by the wheel build scripts (it intentionally
does not live at the repo root, so that a normal source/dev install stays pure
Python and depends on an externally installed scikit-sparse). It compiles the
``sksparse.cholmod`` Cython extension from the ``sksparse`` package that the
build scripts vendor into the repo root, linking against the system/bundled
SuiteSparse.
"""

import os

import numpy as np
from Cython.Build import cythonize
from setuptools import Extension, setup

INCLUDE_DIRS = [
    np.get_include(),
    "/usr/include/suitesparse",  # Debian/manylinux default
]
LIBRARY_DIRS = []

user_include_dir = os.getenv("SUITESPARSE_INCLUDE_DIR")
user_library_dir = os.getenv("SUITESPARSE_LIBRARY_DIR")
if user_include_dir:
    INCLUDE_DIRS.append(user_include_dir)
if user_library_dir:
    LIBRARY_DIRS.append(user_library_dir)

ext_modules = cythonize(
    [
        Extension(
            "sksparse.cholmod",
            ["sksparse/cholmod.pyx"],
            include_dirs=INCLUDE_DIRS,
            library_dirs=LIBRARY_DIRS,
            libraries=["cholmod"],
        )
    ],
    language_level="3",
)

setup(
    ext_modules=ext_modules,
    license_files=[
        "LICENSE",
        "LICENSE_scikit_sparse.txt",
        "LICENSE_libsuitesparse_cholmod.txt",
        "LICENSE_libsuitesparse.txt",
    ],
)
