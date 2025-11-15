"""
exec(open("tutorial.py").read())
"""

# Entrypoint to run all demos if you like:
#   python main.py
#
# Or run them individually:
#   python 1_score_matching.py
#   python 2_sliced_score_matching.py
#   python 3_denoising_score_matching.py

import subprocess
import sys

def run(script):
    print(f"\n=== Running: {script} ===\n")
    subprocess.run([sys.executable, script], check=True)

if __name__ == "__main__":
    run("1_score_matching.py")
    run("2_sliced_score_matching.py")
    run("3_denoising_score_matching.py")
