# Start the AdaKernel OpenAI-compatible server for Qwen3.8 9B (used by OpenCode as provider "adakernel").
#   powershell -ExecutionPolicy Bypass -File C:\Users\OmerC\kernel-sanders\scripts\serve.ps1 [-Ctx 32768] [-Port 8090]
param(
    [int]$Ctx = 32768,
    [int]$Port = 8090,
    [string]$Model = ""
)
$root = Split-Path -Parent $PSScriptRoot
Push-Location $root
try {
    $extra = @()
    if ($Model) { $extra += @("--model", $Model) }
    python -m adakernel.serve --ctx $Ctx --port $Port @extra
} finally {
    Pop-Location
}
