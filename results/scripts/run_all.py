#!/usr/bin/env python3
"""
run_all_analyses.py — Orchestrator for FJSSP/AGV log analysis.

Usage:
    python run_all_analyses.py <folder_with_csvs> [--out <custom_output_dir>]
    
Example:
    python run_all_analyses.py cap_control_unbound
    (Will output all figures to cap_control_unbound/figs)
"""

import argparse
import os
import sys
import subprocess
from pathlib import Path

def run_script(script_name: str, args_list: list):
    """Executes a target Python script via subprocess."""
    # Resolve the script path relative to this master script's location
    script_path = Path(__file__).resolve().parent / script_name
    
    if not script_path.exists():
        print(f"⚠️  Warning: Could not find script '{script_name}' in {script_path.parent}. Skipping.")
        return

    cmd = [sys.executable, str(script_path)] + args_list
    print(f"\n{'='*70}\n🚀 Running {script_name}...\n{'='*70}")
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ {script_name} exited with error code {e.returncode}.")

def main():
    parser = argparse.ArgumentParser(description="Run all analysis scripts on a directory of CSV logs.")
    parser.add_argument("folder", type=str, help="Folder containing the CSV files (e.g., cap_control_unbound)")
    parser.add_argument("--out", type=str, default=None, help="Output directory for all figures. Defaults to <folder>/figs")
    args = parser.parse_args()

    input_dir = Path(args.folder)
    if not input_dir.exists() or not input_dir.is_dir():
        sys.exit(f"❌ Error: Input directory '{input_dir}' does not exist.")

    # Default output directory is <input_folder>/figs
    out_dir = Path(args.out) if args.out else input_dir / "figs"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📁 Target data folder: {input_dir.absolute()}")
    print(f"📁 Output figure folder: {out_dir.absolute()}")

    # Dictionary mapping each script to its required CSV and specific argument syntax
    tasks = [
        {
            "script": "analyze_agv.py",
            "csv": "agv_performance.csv",
            "args": lambda csv, out: [str(csv), "--out", str(out)]
        },
        {
            "script": "plot_generated.py",
            "csv": "results.csv",
            "args": lambda csv, out: [str(csv), "--out", str(out)]
        },
        {
            "script": "throughput.py",
            "csv": "throughput.csv",
            "args": lambda csv, out: [str(csv), "-o", str(out)]
        },
        {
            "script": "diagnose_utilization.py",
            "csv": "machine_utilization.csv",
            "args": lambda csv, out: [str(csv), "--out", str(out)]
        },
        {
            "script": "find_lambda_plateau.py",
            "csv": "results.csv",
            # This script expects a file path for output, not a directory
            "args": lambda csv, out: ["--results", str(csv), "--out", str(out / "lambda_plateau.png")]
        }
    ]

    for task in tasks:
        csv_path = input_dir / task["csv"]
        if csv_path.exists():
            script_args = task["args"](csv_path, out_dir)
            run_script(task["script"], script_args)
        else:
            print(f"\n⚠️  Skipping {task['script']}: Required file '{task['csv']}' not found in {input_dir}.")

    print(f"\n✅ All analyses complete. Results are saved in: {out_dir}")

if __name__ == "__main__":
    main()