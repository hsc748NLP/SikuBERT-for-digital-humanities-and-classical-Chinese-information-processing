#!/usr/bin/env python
from setuptools import setup
from os import path

this_directory = path.abspath(path.dirname(__file__))
setup(
    install_requires=[
        "six",
        "tqdm~=4.30.0",
        "torch>=1.4.0",
        "torchtext==0.4.0",
        "future",
        "configargparse",
        "tensorboard>=1.14",
        "pyonmttok==1.*;platform_system=='Linux'",
        "pyyaml",
        "sentencepiece",
        "MeCab"
    ],
)
