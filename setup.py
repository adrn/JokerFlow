#!/usr/bin/env python

import os
import sys
import tensorflow as tf
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext


def customize_compiler_for_nvcc(self):
    self.src_extensions.append('.cu')
    default_compiler_so = self.compiler_so
    super = self._compile

    def _compile(obj, src, ext, cc_args, extra_postargs, pp_opts):
        if os.path.splitext(src)[1] == '.cu':
            self.set_executable('compiler_so', "nvcc")
            postargs = extra_postargs["nvcc"]
        else:
            postargs = extra_postargs["gcc"]

        super(obj, src, ext, cc_args, postargs, pp_opts)
        self.compiler_so = default_compiler_so

    self._compile = _compile


class custom_build_ext(build_ext):

    def build_extensions(self):
        customize_compiler_for_nvcc(self.compiler)
        build_ext.build_extensions(self)


compile_flags = tf.sysconfig.get_compile_flags()
link_flags = tf.sysconfig.get_link_flags()

compile_flags += ["-D GOOGLE_CUDA=1"]
gcc_flags = compile_flags + ["-std=c++11", "-O2", "-march=native", "-fPIC"]
nvcc_flags = compile_flags + ["-std=c++11", "-shared", "-Xcompiler", "-fPIC",
                              "-x", "cu",
                              "--expt-relaxed-constexpr"]
if sys.platform == "darwin":
    gcc_flags += ["-mmacosx-version-min=10.9"]


extensions = [
    Extension(
        "jokerflow.kepler_op",
        sources=["jokerflow/kepler_op.cc",
                 "jokerflow/kepler_op.cc.cu"],
        language="c++",
        include_dirs=["jokerflow"],
        extra_compile_args=dict(
            nvcc=nvcc_flags,
            gcc=gcc_flags,
        ),
        extra_link_args=link_flags,
    ),
]

setup(
    name="jokerflow",
    license="MIT",
    packages=["jokerflow"],
    ext_modules=extensions,
    cmdclass={"build_ext": custom_build_ext},
    zip_safe=False,
)
