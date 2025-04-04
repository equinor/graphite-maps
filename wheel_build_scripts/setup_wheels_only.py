import os

import numpy as np
from setuptools import Extension, setup

INCLUDE_DIRS = [
    # Debian's suitesparse-dev installs to
    np.get_include(),
    "/usr/include/suitesparse",
]
LIBRARY_DIRS = []

user_include_dir = os.getenv("SUITESPARSE_INCLUDE_DIR")
user_library_dir = os.getenv("SUITESPARSE_LIBRARY_DIR")
if user_include_dir:
    INCLUDE_DIRS.append(user_include_dir)

if user_library_dir:
    LIBRARY_DIRS.append(user_library_dir)

# Force platform-specific!
setup(
    ext_modules=[
        Extension(
            "sksparse.cholmod",
            ["graphite_maps/sksparse/cholmod.pyx"],
            include_dirs=INCLUDE_DIRS,
            library_dirs=LIBRARY_DIRS,
            libraries=["cholmod"],
        )
    ],
)
