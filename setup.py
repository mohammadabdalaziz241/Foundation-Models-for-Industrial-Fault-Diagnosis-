"""
Minimal setup.py — makes the 'src' package editable-installable.

After running `.venv/bin/pip install -e . --no-deps` once, all scripts can use
    from src.config import ...
    from src.trainer import ...
without any sys.path.insert() hacks.

This does NOT install any extra dependencies (--no-deps) and does NOT affect
existing checkpoints, splits, or result files.
"""

from setuptools import setup, find_packages

setup(
    name="rmfd",
    version="0.1.0",
    description="Rotating Machinery Fault Diagnosis — MSc dissertation",
    packages=find_packages(),   # finds src/ and src/models/
    python_requires=">=3.12",
)
