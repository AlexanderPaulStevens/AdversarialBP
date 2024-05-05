# flake8: noqa
# isort:skip
import logging.config
import os
import pathlib

import yaml

from .evaluation import Benchmark
from .generation_methods import AdversarialExamples

def get_logger(logger: str):
    return logging.getLogger(logger)