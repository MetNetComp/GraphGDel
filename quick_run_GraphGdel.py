#!/usr/bin/env python3
"""
Quick run script for GraphGdel on e_coli_core model.
This script provides a simple way to test the GraphGdel framework.
"""

import os
import sys
from GraphGdel_main import main  

def run():
    # Default: eval only. To pre-train and save a matching checkpoint, run:
    #   python quick_run_GraphGdel.py --train 1 [--train_epochs 50]
    if len(sys.argv) <= 1:
        sys.argv = ['quick_run_GraphGdel.py', '--CBM', 'e_coli_core', '--use_cpu', '1']
    else:
        sys.argv = [sys.argv[0], '--CBM', 'e_coli_core', '--use_cpu', '1'] + sys.argv[1:]
    main()  # Call the main function from GraphGdel_main.py

if __name__ == "__main__":
    run()
