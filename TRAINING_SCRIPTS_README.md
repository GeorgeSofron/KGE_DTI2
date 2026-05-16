# Training and Evaluation Runner Scripts

Three options to run all training models sequentially with automatic evaluation:

## Option 1: Python Script (Recommended - Cross-Platform)

```bash
python run_all_training_and_evaluation.py
```

**Features:**
- Cross-platform (Windows, Mac, Linux)
- Clear status reporting
- Automatic error detection
- Summary report with timing
- Stops on first failure by default

**Output:**
- ✓ Shows status of each model (training + evaluation)
- ✓ Reports total execution time
- ✓ Exits with proper error codes

---

## Option 2: PowerShell Script (Windows)

```powershell
.\run_all_training_and_evaluation.ps1
```

**Options:**
```powershell
.\run_all_training_and_evaluation.ps1 -ContinueOnError
```

**Features:**
- Windows-native
- `-ContinueOnError` flag: Continue even if a model fails
- Real-time output streaming
- Colored console output (on compatible terminals)

---

## Option 3: Batch File (Windows - Simplest)

```cmd
run_all_training_and_evaluation.bat
```

**Features:**
- Simplest to run (just double-click or run from cmd)
- Delegates to Python script
- No external dependencies

---

## Workflow

Each script runs models in this sequence:

1. **TransE Training** → (waits for completion) → **TransE Evaluation**
2. **ComplEx Training** → (waits for completion) → **ComplEx Evaluation**
3. **TriModel Training** → (waits for completion) → **TriModel Evaluation**

---

## Expected Output

```
======================================================================
MASTER TRAINING AND EVALUATION RUNNER
======================================================================
Working directory: C:\Users\georg\Music\MSc\KGE_DTI2-1

Verifying scripts exist...
✓ All scripts found

######################################################################
# PROCESSING MODEL: TransE
######################################################################

======================================================================
Starting: TransE Training
======================================================================
Command: python TransE/TransE_Torch.py

[Training runs here...]

✓ SUCCESS: TransE Training

======================================================================
Starting: TransE Evaluation
======================================================================
Command: python TransE/TransE_Torch_evaluation.py

[Evaluation runs here...]

✓ SUCCESS: TransE Evaluation

TransE completed in 1234.5 seconds

[... ComplEx and TriModel follow ...]

======================================================================
SUMMARY REPORT
======================================================================
✓ TransE        : SUCCESS
✓ ComplEx       : SUCCESS
✓ TriModel      : SUCCESS

Total time: 45.2 minutes (2712 seconds)

✓ ALL MODELS COMPLETED SUCCESSFULLY
```

---

## Error Handling

**If a model's training fails:**
- Python script: Stops immediately and reports failure
- PowerShell script: Stops immediately (use `-ContinueOnError` to skip failed models)
- Batch file: Delegates to Python script behavior

---

## Tips

1. **Monitor Progress**: Leave the terminal open to watch real-time status
2. **Long Runtime**: These scripts can run for hours. Consider:
   - Running in a `tmux`/`screen` session on Linux
   - Using `nohup` to persist if SSH disconnects
   - Redirecting output to a log file: `python run_all_training_and_evaluation.py > run.log 2>&1`

3. **Check Outputs**: After completion, verify results in:
   - `outputs_transe/`
   - `outputs_complex/`
   - `outputs_trimodel/`
   - `outputs_dti_evaluation_fixed/` (evaluation results)

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Scripts not found | Verify you're in the correct directory: `C:\Users\georg\Music\MSc\KGE_DTI2-1` |
| Python not found | Install Python or check PATH environment variable |
| Permission denied (PowerShell) | Run: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser` |
| Model training fails | Check individual model files for issues |
| Evaluation fails | Verify training outputs exist in corresponding `outputs_*` directory |

---

## Customization

To modify which models run or their order, edit the `MODELS` list in:
- **Python**: `run_all_training_and_evaluation.py` (around line 20)
- **PowerShell**: `run_all_training_and_evaluation.ps1` (around line 16)
