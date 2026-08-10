#!/usr/bin/env python3
"""Launcher for guardrailer_security/train_phase3.py from project root."""
import sys
import os
from pathlib import Path

orig_cwd = Path.cwd()
security_dir = Path(__file__).resolve().parent.parent / "guardrailer_security"

sys.path.insert(0, str(security_dir))

path_flags = {"--data", "--output-dir", "--feedback-output", "--save-training-data"}
resolved = []
skip_next = False
for i, arg in enumerate(sys.argv[1:], 1):
    if skip_next:
        skip_next = False
        if not Path(arg).is_absolute():
            candidate = orig_cwd / arg
            if candidate.exists():
                arg = str(candidate)
        resolved.append(arg)
        continue
    if arg in path_flags:
        resolved.append(arg)
        skip_next = True
    else:
        resolved.append(arg)

sys.argv = [sys.argv[0]] + resolved
os.chdir(str(security_dir))

from train_phase3 import main

if __name__ == "__main__":
    main()
