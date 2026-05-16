"""
Master Training and Evaluation Runner
=====================================

Runs all KGE models (TransE, ComplEx, TriModel) in sequence:
1. Train model
2. Wait for completion
3. Run evaluation
4. Move to next model

This ensures safe sequential execution without conflicts.
"""

import os
import subprocess
import sys
from typing import Tuple, List
from pathlib import Path
import time


# Model configurations: (training_script, evaluation_script, model_name)
MODELS: List[Tuple[str, str, str]] = [
    ("TransE/TransE_Torch.py", "TransE/TransE_Torch_evaluation.py", "TransE"),
    ("ComplEx/ComplEx_Torch.py", "ComplEx/ComplEx_Torch_evaluation.py", "ComplEx"),
    ("TriModel/TriModel_Torch.py", "TriModel/TriModel_Torch_evaluation.py", "TriModel"),
]


def run_command(command: List[str], description: str) -> bool:
    """
    Run a shell command and wait for completion.
    
    Args:
        command: Command to run as list of strings
        description: Human-readable description of what's running
        
    Returns:
        True if successful (exit code 0), False otherwise
    """
    print("\n" + "=" * 70)
    print(f"Starting: {description}")
    print("=" * 70)
    print(f"Command: {' '.join(command)}\n")
    
    try:
        result = subprocess.run(
            command,
            check=False,
            cwd=os.getcwd()
        )
        
        if result.returncode == 0:
            print(f"\n✓ SUCCESS: {description}")
            return True
        else:
            print(f"\n✗ FAILED: {description} (exit code: {result.returncode})")
            return False
            
    except Exception as e:
        print(f"\n✗ ERROR running {description}: {e}")
        return False


def main():
    print("\n" + "=" * 70)
    print("MASTER TRAINING AND EVALUATION RUNNER")
    print("=" * 70)
    print(f"Working directory: {os.getcwd()}\n")
    
    # Verify all scripts exist before starting
    print("Verifying scripts exist...")
    missing_scripts = []
    for train_script, eval_script, model_name in MODELS:
        if not os.path.exists(train_script):
            missing_scripts.append(train_script)
        if not os.path.exists(eval_script):
            missing_scripts.append(eval_script)
    
    if missing_scripts:
        print("\n✗ ERROR: Missing scripts:")
        for script in missing_scripts:
            print(f"  - {script}")
        sys.exit(1)
    
    print("✓ All scripts found\n")
    
    # Track results
    results = []
    total_start = time.time()
    
    # Run each model's training + evaluation
    for train_script, eval_script, model_name in MODELS:
        model_start = time.time()
        
        print(f"\n{'#' * 70}")
        print(f"# PROCESSING MODEL: {model_name}")
        print(f"{'#' * 70}")
        
        # Run training
        train_success = run_command(
            [sys.executable, train_script],
            f"{model_name} Training"
        )
        
        if not train_success:
            print(f"\n⚠ WARNING: {model_name} training failed. Skipping evaluation.")
            results.append((model_name, "TRAIN_FAILED"))
            continue
        
        # Run evaluation
        eval_success = run_command(
            [sys.executable, eval_script],
            f"{model_name} Evaluation"
        )
        
        if eval_success:
            results.append((model_name, "SUCCESS"))
        else:
            results.append((model_name, "EVAL_FAILED"))
        
        elapsed = time.time() - model_start
        print(f"\n{model_name} completed in {elapsed:.1f} seconds")
    
    # Summary report
    total_elapsed = time.time() - total_start
    print("\n" + "=" * 70)
    print("SUMMARY REPORT")
    print("=" * 70)
    
    for model_name, status in results:
        symbol = "✓" if status == "SUCCESS" else "✗"
        print(f"{symbol} {model_name:15} : {status}")
    
    print(f"\nTotal time: {total_elapsed / 60:.1f} minutes ({total_elapsed:.0f} seconds)")
    
    # Determine overall success
    all_success = all(status == "SUCCESS" for _, status in results)
    
    if all_success:
        print("\n✓ ALL MODELS COMPLETED SUCCESSFULLY")
        return 0
    else:
        print("\n✗ SOME MODELS FAILED (see above)")
        return 1


if __name__ == "__main__":
    sys.exit(main())
