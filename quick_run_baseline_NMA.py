#!/usr/bin/env python3
"""
Quick run script for NMA (Neighborhood Mean Aggregation) baseline on e_coli_core.
"""

import sys
from baseline.NMA_main import main


def run():
    # Forward CLI args so e.g. --train 1 --train_epochs 5 works
    if len(sys.argv) <= 1:
        sys.argv = ["quick_run_baseline_NMA.py", "--CBM", "e_coli_core", "--use_cpu", "1"]
    else:
        sys.argv = [sys.argv[0], "--CBM", "e_coli_core", "--use_cpu", "1"] + sys.argv[1:]
    main()


if __name__ == "__main__":
    run()
