#!/usr/bin/env python3
"""Deterministically regenerate every results file from the current code.

Usage (from the Program directory):
    python3 run_all_experiments.py

Every generator is self-seeded (the stochastic eco-dynamics use fixed RNG
seeds), so the .txt outputs are reproducible run-to-run. All outputs are
consolidated into the results/ folder, which is therefore the single source of
truth for the pre-generated results.
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")

# (script, output file, capture_stdout_to_file?)
JOBS = [
    ("run_experiments.py",          "results_v2.txt",                  False),
    ("parameter_sweep.py",          "sweep_results.txt",               False),
    ("robustness_analysis.py",      "robustness_results.txt",          False),
    ("robustness_geometry.py",      "robustness_geometry_results.txt", False),
    ("motor_rule_test.py",          "motor_rule_results.txt",          True),
    ("life_history_analysis.py",    "lifehistory_results.txt",         False),
    ("age_structured_analysis.py",  "age_structured_results.txt",      False),
    ("eco_analysis.py",             "eco_results.txt",                 False),
]


def main():
    os.makedirs(RESULTS, exist_ok=True)
    env = dict(os.environ, PYTHONHASHSEED="0", PYTHONDONTWRITEBYTECODE="1")
    for script, out, capture in JOBS:
        print("  running %-28s -> results/%s" % (script, out))
        if capture:
            r = subprocess.run([sys.executable, script], cwd=HERE, env=env,
                               capture_output=True, text=True)
            if r.returncode == 0:
                with open(os.path.join(RESULTS, out), "w") as f:
                    f.write(r.stdout)
        else:
            r = subprocess.run([sys.executable, script], cwd=HERE, env=env)
            produced = os.path.join(HERE, out)
            if r.returncode == 0 and os.path.exists(produced):
                os.replace(produced, os.path.join(RESULTS, out))
        if r.returncode != 0:
            sys.exit("  ! %s exited with code %d" % (script, r.returncode))
    print("All experiment outputs regenerated into results/ (deterministic).")


if __name__ == "__main__":
    main()
