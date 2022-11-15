import sys
import platform

from pybind11 import get_cmake_dir
from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

__version__ = "0.0.1"

if platform.processor() == "arm" or platform.processor() == "i386":
    extra_args = ['-O3', '-ftree-vectorize']
else:
    extra_args = ['-O3', '-march=native', '-mfpmath=sse']

ext_modules = [
    Pybind11Extension("pyMeDeCom.extensions",
        [
            "src/pybindings.cpp"
        ],
        define_macros = [('VERSION_INFO', __version__)],
        extra_compile_args = extra_args
        ),
]

setup(
    name="pyMeDeCom",
    packages=['pyMeDeCom'],
    version=__version__,
    author="Valentin Maurer",
    author_email="valentin.maurer@stud.uni-heidelberg.de",
    url="https://github.com/CompEpigen/pyMeDeCom",
    description="Decomposition of methylome data",
    long_description="Python implementation of https://github.com/lutsik/MeDeCom.",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    install_requires=["scikit-learn", "numpy", "pybind11"],
    python_requires=">=3.6",
    include_package_data=True,
    package_data={'': ['data/*.npz']},
)
