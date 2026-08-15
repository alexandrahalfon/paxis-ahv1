#!/usr/bin/env python3
"""
Wrapper script for backward compatibility.
Calls the main CLI from src/cli/main.py
"""
import sys
from src.cli.main import main

if __name__ == "__main__":
    main()
