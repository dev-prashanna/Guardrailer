#!/usr/bin/env python3
"""Download Guardrailer output files from a Kaggle notebook."""

import subprocess
import sys
import os

NOTEBOOK_SLUG = "prashannadeveloper/guardrailer-enhanced-ingest"  # update if different
OUTPUT_DIR = os.path.expanduser("~/Downloads/guardrailer_output")

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"Downloading output from notebook: {NOTEBOOK_SLUG}")
    print(f"Target directory: {OUTPUT_DIR}")
    
    result = subprocess.run(
        ["kaggle", "kernels", "output", NOTEBOOK_SLUG, "-p", OUTPUT_DIR],
        capture_output=True, text=True
    )
    
    if result.returncode == 0:
        print("Download complete!")
        for f in os.listdir(OUTPUT_DIR):
            size = os.path.getsize(os.path.join(OUTPUT_DIR, f)) / 1e6
            print(f"  {f}: {size:.1f} MB")
    else:
        print(f"Error: {result.stderr}")
        sys.exit(1)

if __name__ == "__main__":
    main()
