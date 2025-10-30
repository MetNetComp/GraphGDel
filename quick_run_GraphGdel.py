#!/usr/bin/env python3
"""
Quick run script for GraphGdel on e_coli_core model.
This script provides a simple way to test the GraphGdel framework.
"""

import os
import sys
from GraphGdel_main import main  

def run():
    sys.argv = ['quick_run.py', '--CBM', 'e_coli_core', '--use_cpu', '1']  # Simulate command-line arguments
    main()  # Call the main function from GraphGdel_main.py

if __name__ == "__main__":
    run()
