param(
    [string]$SshHost = "ec2-user@15.164.62.191",
    [string]$SshKey = "$env:USERPROFILE\.ssh\billiard_ec2",
    [string]$RemoteDatabaseHost = "billiard-db.cx0e6meogl9d.ap-northeast-2.rds.amazonaws.com",
    [int]$TunnelPort = 15432,
    [int]$ServerPort = 8765
)

$ErrorActionPreference = "Stop"
$scriptDirectory = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path $scriptDirectory ".venv\Scripts\python.exe"
$server = Join-Path $scriptDirectory "local_probability_server.py"

if (-not (Test-Path -LiteralPath $SshKey)) {
    throw "SSH private key not found: $SshKey"
}
if (-not (Test-Path -LiteralPath $python)) {
    throw "Python virtual environment not found: $python"
}

$tunnelProcess = $null
$listener = Get-NetTCPConnection -LocalPort $TunnelPort -State Listen -ErrorAction SilentlyContinue
if (-not $listener) {
    $forward = "${TunnelPort}:${RemoteDatabaseHost}:5432"
    $arguments = @(
        "-N",
        "-L", $forward,
        "-i", $SshKey,
        "-o", "BatchMode=yes",
        "-o", "ExitOnForwardFailure=yes",
        "-o", "ServerAliveInterval=30",
        "-o", "ServerAliveCountMax=3",
        $SshHost
    )
    $tunnelProcess = Start-Process -FilePath "ssh.exe" -ArgumentList $arguments -WindowStyle Hidden -PassThru

    for ($attempt = 0; $attempt -lt 20; $attempt++) {
        Start-Sleep -Milliseconds 250
        $listener = Get-NetTCPConnection -LocalPort $TunnelPort -State Listen -ErrorAction SilentlyContinue
        if ($listener) {
            break
        }
        if ($tunnelProcess.HasExited) {
            throw "SSH tunnel exited before opening local port $TunnelPort"
        }
    }
    if (-not $listener) {
        Stop-Process -Id $tunnelProcess.Id -ErrorAction SilentlyContinue
        throw "SSH tunnel did not open local port $TunnelPort"
    }
}

Write-Host "SSH tunnel ready: 127.0.0.1:$TunnelPort -> RDS"
Write-Host "CueCast UI: http://127.0.0.1:$ServerPort"

try {
    & $python $server --host 127.0.0.1 --port $ServerPort
}
finally {
    if ($tunnelProcess -and -not $tunnelProcess.HasExited) {
        Stop-Process -Id $tunnelProcess.Id
    }
}
