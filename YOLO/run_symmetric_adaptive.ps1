$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $python)) {
    throw "Python environment not found: $python"
}

$model = Join-Path $projectRoot "outputs\symmetric_hybrid_v2\model.json"
if (-not (Test-Path -LiteralPath $model)) {
    & $python (Join-Path $projectRoot "train_symmetric_probability_model.py")
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

$url = "http://127.0.0.1:8767"
Start-Process $url
& $python (Join-Path $projectRoot "symmetric_adaptive_server.py") --host 127.0.0.1 --port 8767
