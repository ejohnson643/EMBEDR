#!/usr/bin/python
###############################################################################
## setup.py
###############################################################################
##
##    Setup file to build EMBEDR package.
##
##    Author: Eric Johnson
##    Date Last Modified: June 9, 2020
##
##    This setup file is adapted from the openTSNE setup.py file written by
##    Pavlin Poličar under the BSD 3-Clause License.
##
##    The modifications to compile the ANNOY library were adapted from the
##    setup file at github.com/spotify/annoy
##
###############################################################################
import distutils
import os
from os import path
import platform
import sys
import tempfile
import warnings

from distutils import ccompiler
from distutils.command.build_ext import build_ext
from distutils.errors import CompileError, LinkError
from distutils.sysconfig import customize_compiler

import setuptools
from setuptools import setup, Extension


class get_numpy_include:
    """Helper class to determine the numpy include path

    The idea here is that we want to use ``numpy.get_include()``, but we can't
    call it until we're sure numpy is installed.
    """

    def __str__(self):
        import numpy
        return numpy.get_include()


def get_include_dirs():
    """Quick function to get include directories for the compiler"""
    sysp = sys.prefix
    return (path.join(sysp, "include"), path.join(sysp, "Library", "include"))


def get_library_dirs():
    """Quick function to get library directories for the compiler"""
    sysp = sys.prefix
    return (path.join(sysp, "lib"), path.join(sysp, "Library", "lib"))


def has_c_library(library, extension=".c"):
    """Check whether a C/C++ library is available to the compiler.

    Parameters
    ----------
    library: str
        The name of the library that we want to check for.  For example, if we
        want FFTW3, then we need to link `fftw3.h`, so we supply `fftw3` as an
        input to this function.
    extensions: str (optional, default='.c')
        File extension of the library we want to check for.  If we want a C
        library, then the extension is `.c`, while for C++, any of `.cc`,
        `.cpp`, or `.cxx` are allowed.

    Returns
    -------
    bool
        T/F depending on whether the requested library is available.
    """
    with tempfile.TemporaryDirectory(dir=".") as directory:
        ## Make a temporary C/C++ file that looks for the requested library.
        fname = path.join(directory, f"{library}{extension}")
        with open(fname, "w") as f:
            f.write(f"#include <{library}.h>\n")
            f.write("int main() {}\n")

        ## Get a compiler
        compiler = ccompiler.new_compiler()
        customize_compiler(compiler)

        ## Add the include directories
        for inc_dir in get_include_dirs():
            compiler.add_include_dir(inc_dir)
        ## I don't know why we need this assertion.
        assert isinstance(compiler, ccompiler.CCompiler)

        ## Try to compile and link the library
        try:
            compiler.link_executable(compiler.compile([fname]), fname)
            return True
        except (CompileError, LinkError):
            return False


class MyBuildExt(build_ext):

    def build_extensions(self):
        ## We need to add a few things to the extension builder to keep track
        ## of the system architecture and to correctly compile the Annoy lib.

        ## First, lets set up compile and linker arguments
        extra_compile_args = []
        extra_link_args    = []

        ## Depending on the compiler, set the optimization level
        compiler = self.compiler.compiler_type
        if compiler == 'unix':
            extra_compile_args += ["-O3"]
        elif compiler == 'msvc':
            extra_compile_args += ["/Ox", "/fp:fast"]

        ## Poličar: "For some reason fast math causes segfaults on linus but
        ## works on mac"
        if compiler == 'unix' and platform.system() == 'Darwin':
            extra_compile_args += ["-ffast-math", "-fno-associative-math"]

        ## Make sure that the Annoy library can be found:
        annoy_ext = None
        for extension in extensions:
            if "annoy.annoylib" in extension.name:
                annoy_ext = extension
        assert annoy_ext is not None, "Annoy extension could not be found!"

        ## Then we set up Annoy-specific flags
        if compiler == 'unix':
            annoy_ext.extra_compile_args += ["-std=c++14"]
            annoy_ext.extra_compile_args += ["-DANNOYLIB_MULTITHREADED_BUILD"]
        elif compiler == 'msvc':
            annoy_ext.extra_compile_args += ["/std:c++14"]

        ## Set the minimum MacOS version
        if compiler == 'unix' and platform.system() == "Darwin":
            extra_compile_args += ['-mmacosx-version-min=10.12']
            extra_link_args    += ["-stdlib=libc++",
                                   "-mmacosx-version-min=10.12"]

        ## Poličar: "We don't want the compiler to optimize for system
        ## architecture if we're building packages to be distributed by
        ## conda-forge, but if the package is being built locally, this is
        ## desired."  Basically if we know what system we're on, we can do some
        ## further optimization.
        if not ("AZURE_BUILD" in os.environ or "CONDA_BUILD" in os.environ):
            if platform.machine() == 'ppc64le':
                extra_compile_args += ['-mpcu=native']
            if platform.machine() == 'x86_64':
                extra_compile_args += ['-march=native']

        ## Check for omp library.  Add compiler flags if it's found.
        if has_c_library("omp"):
            print(f"Found openMP.  Compiling with openmp flags...")
            if platform.system() == "Darwin" and compiler == 'unix':
                extra_compile_args += ["-Xpreprocessor", "-fopenmp"]
                extra_link_args    += ['-lomp']
            elif compiler == 'unix':
                extra_compile_args += ['-fopenmp']
                extra_link_args    += ['-fopenmp']
            elif compiler == 'msvc':
                extra_compile_args += ['/openmp']
                extra_link_args    += ['/openmp']
        else:
            warn_str =  f"You appear to be using a compiler that does not"
            warn_str += f" support openMP, meaning that this library will not"
            warn_str += f" be able to run on multiple cores. Please install or"
            warn_str += f" enable openMP to use multiple cores."
            warnings.warn(warn_str)

        ## Add other arguments that might be defined.
        for extension in extensions:
            extension.extra_compile_args += extra_compile_args
            extension.extra_link_args    += extra_link_args

        ## Add the numpy and system include and library directories
        for extension in extensions:
            extension.include_dirs.extend(get_include_dirs())
            extension.include_dirs.append(get_numpy_include())

            extension.library_dirs.extend(get_library_dirs())

        ## Run the super method.
        super().build_extensions()

## Prepare the Annoy extension.
## This is adapted from the annoy setup.py
extra_compile_args = []
extra_link_args    = []

annoy_path = "EMBEDR/dependencies/annoy/"
annoy_hdrs = ["annoylib.h", "kissrandom.h", "mman.h"]
annoy_ext = Extension("EMBEDR.dependencies.annoy.annoylib",
                      [annoy_path + "annoymodule.cc"],
                      depends=[annoy_path + f for f in annoy_hdrs],
                      language="c++",
                      extra_compile_args=extra_compile_args,
                      extra_link_args=extra_link_args)

## Prepare the quad_tree extension.
quad_tree_ext = Extension("EMBEDR.quad_tree",
                          ["EMBEDR/quad_tree.pyx"],
                          language="c++")

## Prepare the _tsne extension.
_tSNE_ext = Extension("EMBEDR._tsne", ["EMBEDR/_tsne.pyx"], language="c++")

## All the extensions together
extensions = [quad_tree_ext, _tSNE_ext, annoy_ext]

## To do the Fourier convolutions we need either fftw3 or to use numpy...
if has_c_library("fftw3"):
    print(f"FFTW3 header files found. Using the FFTW implementation of FFT.")
    fftw3_ext = Extension("EMBEDR._matrix_mul.matrix_mul",
                          ["EMBEDR/_matrix_mul/matrix_mul_fftw3.pyx"],
                          libraries=["fftw3"],
                          language="c++",)
    extensions.append(fftw3_ext)
else:
    print(f"FFTW3 header files couldn't be found. Using numpy for FFT.")
    numpy_ext = Extension("EMBEDR._matrix_mul.matrix_mul",
                          ["EMBEDR/_matrix_mul/matrix_mul_numpy.pyx"],
                          language="c++",)
    extensions.append(numpy_ext)

## Try and cythonize anything applicable.
try:
    from Cython.Build import cythonize
    extensions = cythonize(extensions)
except ImportError:
    pass

# Read in version
__version__: str = ""  ## This line suppresses linting errors.
exec(open(os.path.join("EMBEDR", "version.py")).read())

setup(
    name="EMBEDR",
    description="Statistical Quality Assessment of Dim. Red. Algorithms",
    # long_description=readme(),
    version=__version__,
    license="BSD-3-Clause",

    author="Eric Johnson",
    author_email="eric.johnson643@gmail.com",
    url="https://github.com/ejohnson643/EMBEDR",
    project_urls={
        "Source": "https://github.com/ejohnson643/EMBEDR",
        "Issue Tracker": "https://github.com/ejohnson643/EMBEDR/issues",
    },

    packages=setuptools.find_packages(include=["EMBEDR", "EMBEDR.*"]),
    python_requires=">=3.6",
    install_requires=[
        "numpy>=1.16.6",
        "scikit-learn>=0.20",
        "scipy",
        "cython",
        "numba"
    ],
    extras_require={
        "pynndescent": "pynndescent~=0.5.0",
    },
    ext_modules=extensions,
    cmdclass={"build_ext": MyBuildExt},
)