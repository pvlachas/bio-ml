"""
exec(open("tutorial.py").read())
"""

# Entrypoint to run all demos if you like:
#   python main.py
#
# Or run them individually:
#   python train_score_matching.py
#   python train_sliced_score_matching.py
#   python train_denoising_score_matching.py

import subprocess
import sys

def run(script):
    print(f"\n=== Running: {script} ===\n")
    subprocess.run([sys.executable, script], check=True)

if __name__ == "__main__":
    run("train_score_matching.py")
    run("train_sliced_score_matching.py")
    run("train_denoising_score_matching.py")
