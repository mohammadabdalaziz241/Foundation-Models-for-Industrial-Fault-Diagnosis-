#!/usr/bin/env python
"""Thin wrapper: regenerate the CWRU grouping re-check table/stats.

Usage: .venv/bin/python scripts/methodology_v2/run_cwru_grouping_recheck.py
Read-only with respect to raw data; writes only into
methodology_v2/part1_audit/.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.methodology_v2.grouping_recheck import main  # noqa: E402

if __name__ == "__main__":
    main()
