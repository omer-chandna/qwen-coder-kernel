# Start the AdaKernel OpenAI-compatible server for Qwen3.8 9B (used by OpenCode as provider "adakernel").
#   powershell -ExecutionPolicy Bypass -File scripts\serve.ps1 [-Ctx 40960] [-Port 8090] [-PfCols 512] [-Model <gguf>] [-Kv8]
# -Ctx must match "limit.context" in OpenCode's config. 40960 with 512-token prompt chunks peaks at ~7.7 GB on an
# 8 GB card; -Ctx 32768 -PfCols 1024 reads long prompts ~15% faster and leaves ~0.8 GB more free.
# -Kv8 stores the attention cache as int8 (opt-in, docs/QWEN38_9B.md "8-bit KV cache"): -Ctx 65536 -Kv8 fits with
# room to spare, -Ctx 81920 -Kv8 is the most that fits (~7.6 GB in use).
param(
    [int]$Ctx = 40960,
    [int]$Port = 8090,
    [int]$PfCols = 512,
    [string]$Model = "",
    [switch]$Kv8
)
$root = Split-Path -Parent $PSScriptRoot
Push-Location $root
try {
    $extra = @()
    if ($Model) { $extra += @("--model", $Model) }
    if ($PfCols) { $extra += @("--pf-cols", $PfCols) }
    if ($Kv8) { $extra += @("--kv8") }
    python -m adakernel.serve --ctx $Ctx --port $Port @extra
} finally {
    Pop-Location
}
