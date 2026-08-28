#!/usr/bin/env python
"""Run the sealed Part-6 latency protocol over all four downstream datasets.

This is a thin, inference-only entry point. It reuses the authoritative
benchmark implementation unchanged while extending its dataset scope to CWRU,
JNU, HIT, and MaFaulDa and writing to a separate result directory.
"""
from pathlib import Path
import sys


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from scripts.methodology_v2 import benchmark_part6_latency as benchmark


benchmark.DATASETS = ("CWRU", "JNU", "HIT", "MAFAULDA")
benchmark.EXPECTED_LOGITS = {"CWRU": 3, "JNU": 4, "HIT": 3, "MAFAULDA": 10}
benchmark.OUT_DIR = Path(
    "results/methodology_v2/part6_compression/latency_four_domain"
).resolve()
benchmark.AUDIT_PATH = benchmark.OUT_DIR / "lightweight_protocol_audit.md"


if __name__ == "__main__":
    benchmark.main()
