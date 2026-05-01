"""Setup configuration for DP-Benchmark-RAG package.

Package for benchmarking differentially private retrieval-augmented generation (DP-RAG)
methods against vanilla and voting-based baselines.
"""

from pathlib import Path

from setuptools import find_packages, setup

BASE_DIR = Path(__file__).resolve().parent
REQUIREMENTS_FILE = BASE_DIR / "requirements.txt"
INSTALL_REQUIRES = [
    line.strip()
    for line in REQUIREMENTS_FILE.read_text(encoding="utf-8").splitlines()
    if line.strip() and not line.strip().startswith("#")
]

setup(
    name="benchdp_rag",
    version="0.1.0",
    packages=find_packages(),
    install_requires=INSTALL_REQUIRES,
)
