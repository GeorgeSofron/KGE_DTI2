# Master Training and Evaluation Runner (PowerShell)
# ================================================
# Runs all KGE models (TransE, ComplEx, TriModel) sequentially

param(
    [switch]$ContinueOnError = $false
)

$ErrorActionPreference = "Continue"

Write-Host ""
Write-Host ("=" * 70)
Write-Host "MASTER TRAINING AND EVALUATION RUNNER (PowerShell)"
Write-Host ("=" * 70)
Write-Host "Working directory: $(Get-Location)"
Write-Host ""

# Define models
$models = @(
    @{
        Name = "TransE"
        TrainScript = "TransE/TransE_Torch.py"
        EvalScript = "TransE/TransE_Torch_evaluation.py"
    },
    @{
        Name = "ComplEx"
        TrainScript = "ComplEx/ComplEx_Torch.py"
        EvalScript = "ComplEx/ComplEx_Torch_evaluation.py"
    },
    @{
        Name = "TriModel"
        TrainScript = "TriModel/TriModel_Torch.py"
        EvalScript = "TriModel/TriModel_Torch_evaluation.py"
    }
)

# Verify scripts exist
Write-Host "Verifying scripts exist..."
$missingScripts = @()

foreach ($model in $models) {
    if (-not (Test-Path $model.TrainScript)) {
        $missingScripts += $model.TrainScript
    }
    if (-not (Test-Path $model.EvalScript)) {
        $missingScripts += $model.EvalScript
    }
}

if ($missingScripts.Count -gt 0) {
    Write-Host ""
    Write-Host "✗ ERROR: Missing scripts:"
    foreach ($script in $missingScripts) {
        Write-Host "  - $script"
    }
    exit 1
}

Write-Host "✓ All scripts found"
Write-Host ""

$results = @()
$totalStart = Get-Date

# Run each model
foreach ($model in $models) {
    $modelStart = Get-Date
    
    Write-Host ""
    Write-Host ("#" * 70)
    Write-Host "# PROCESSING MODEL: $($model.Name)"
    Write-Host ("#" * 70)
    Write-Host ""
    
    # Run training
    Write-Host ("=" * 70)
    Write-Host "Starting: $($model.Name) Training"
    Write-Host ("=" * 70)
    Write-Host "Command: python $($model.TrainScript)"
    Write-Host ""
    
    $trainSuccess = $true
    try {
        & python $model.TrainScript
        if ($LASTEXITCODE -ne 0) {
            Write-Host ""
            Write-Host "✗ FAILED: $($model.Name) Training (exit code: $LASTEXITCODE)"
            $trainSuccess = $false
        } else {
            Write-Host ""
            Write-Host "✓ SUCCESS: $($model.Name) Training"
        }
    } catch {
        Write-Host ""
        Write-Host "✗ ERROR running $($model.Name) Training: $_"
        $trainSuccess = $false
    }
    
    if (-not $trainSuccess) {
        if ($ContinueOnError) {
            Write-Host "⚠ WARNING: $($model.Name) training failed. Skipping evaluation."
            $results += @{
                Name = $model.Name
                Status = "TRAIN_FAILED"
            }
        } else {
            Write-Host "✗ Stopping due to training failure."
            exit 1
        }
        continue
    }
    
    # Run evaluation
    Write-Host ""
    Write-Host ("=" * 70)
    Write-Host "Starting: $($model.Name) Evaluation"
    Write-Host ("=" * 70)
    Write-Host "Command: python $($model.EvalScript)"
    Write-Host ""
    
    $evalSuccess = $true
    try {
        & python $model.EvalScript
        if ($LASTEXITCODE -ne 0) {
            Write-Host ""
            Write-Host "✗ FAILED: $($model.Name) Evaluation (exit code: $LASTEXITCODE)"
            $evalSuccess = $false
        } else {
            Write-Host ""
            Write-Host "✓ SUCCESS: $($model.Name) Evaluation"
        }
    } catch {
        Write-Host ""
        Write-Host "✗ ERROR running $($model.Name) Evaluation: $_"
        $evalSuccess = $false
    }
    
    $status = if ($evalSuccess) { "SUCCESS" } else { "EVAL_FAILED" }
    $results += @{
        Name = $model.Name
        Status = $status
    }
    
    $elapsed = (Get-Date) - $modelStart
    Write-Host ""
    Write-Host "$($model.Name) completed in $([int]$elapsed.TotalSeconds) seconds"
}

# Summary
$totalElapsed = (Get-Date) - $totalStart

Write-Host ""
Write-Host ("=" * 70)
Write-Host "SUMMARY REPORT"
Write-Host ("=" * 70)

foreach ($result in $results) {
    $symbol = if ($result.Status -eq "SUCCESS") { "✓" } else { "✗" }
    Write-Host "$symbol $($result.Name.PadRight(15)) : $($result.Status)"
}

Write-Host ""
Write-Host "Total time: $([int]$totalElapsed.TotalMinutes) minutes ($([int]$totalElapsed.TotalSeconds) seconds)"
Write-Host ""

$allSuccess = $results | Where-Object { $_.Status -ne "SUCCESS" } | Measure-Object | Select-Object -ExpandProperty Count

if ($allSuccess -eq 0) {
    Write-Host "✓ ALL MODELS COMPLETED SUCCESSFULLY"
    exit 0
} else {
    Write-Host "✗ SOME MODELS FAILED (see above)"
    exit 1
}
