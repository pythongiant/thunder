"""Compatibility shim.

All metadata lives in ``pyproject.toml``. This file exists because some
toolchains and ``pip install -e . --no-build-isolation`` invocations expect a
``setup.py`` next to the package. Kept intentionally empty of configuration.
"""

from setuptools import setup

setup()
