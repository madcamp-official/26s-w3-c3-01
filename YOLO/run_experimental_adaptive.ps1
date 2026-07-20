$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $python)) {
    throw "Python environment not found: $python"
}

$url = "http://127.0.0.1:8766"
Start-Process $url
& $python (Join-Path $projectRoot "experimental_adaptive_server.py") --host 127.0.0.1 --port 8766
