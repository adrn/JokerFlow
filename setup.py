#!/usr/bin/env python

import tensorflow as tf
from setuptools import setup, Extension

compile_flags = tf.sysconfig.get_compile_flags()
compile_flags += ["-std=c++11", "-O2", "-mmacosx-version-min=10.9"]
link_flags = tf.sysconfig.get_link_flags()

extensions = [
    Extension(
        "jokerflow.kepler_op",
        sources=["jokerflow/kepler_op.cc"],
        language="c++",
        extra_compile_args=compile_flags,
        extra_link_args=link_flags,
    ),
]

setup(
    name="jokerflow",
    license="MIT",
    packages=["jokerflow"],
    ext_modules=extensions,
    zip_safe=True,
)
