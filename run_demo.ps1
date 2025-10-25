# MotivateMe AI - Demo Launcher
Write-Host "================================" -ForegroundColor Cyan
Write-Host " MOTIVATEME AI - Quick Demo" -ForegroundColor Cyan
Write-Host "================================" -ForegroundColor Cyan
Write-Host ""

# Make sure we're in the scaffold directory
$scriptPath = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptPath

# Set Python path
$env:PYTHONPATH = "$PWD;$env:PYTHONPATH"

Write-Host "[1/2] Training reward model..." -ForegroundColor Yellow
python src\reward_model.py

if ($LASTEXITCODE -eq 0) {
    Write-Host "[OK] Training complete!" -ForegroundColor Green
} else {
    Write-Host "[WARN] Training had issues (this may be okay)" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "[2/2] Launching demo app..." -ForegroundColor Yellow
Write-Host ""
Write-Host "Opening browser at http://localhost:8501" -ForegroundColor Green
Write-Host "Press CTRL+C to stop" -ForegroundColor Gray
Write-Host ""

streamlit run src\demo_app_enhanced.py --server.headless true
