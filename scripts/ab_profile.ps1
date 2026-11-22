# Interleaved A/B of benchmarks\chunk_profile.py between two checkouts (each with its own build\adakernel.dll):
#   powershell -ExecutionPolicy Bypass -File scripts\ab_profile.ps1 -A ..\qck-main -B . [-Rounds 2] [-Out experiments\results\ab]
# Runs A, B, A, B, ... so thermal drift affects both sides alike; compare the best (minimum) of each side.
param(
    [Parameter(Mandatory = $true)][string]$A,
    [Parameter(Mandatory = $true)][string]$B,
    [int]$Rounds = 2,
    [string]$Out = "experiments\results\ab",
    [string]$Extra = ""
)
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$OutDir = Join-Path $Root $Out
New-Item -ItemType Directory -Force $OutDir | Out-Null
for ($r = 0; $r -lt $Rounds; ++$r) {
    foreach ($side in @(@("A", $A), @("B", $B))) {
        $dir = Resolve-Path (Join-Path $Root $side[1])
        Push-Location $dir
        Write-Output "== $($side[0]) round $r ($dir)"
        $json = Join-Path $OutDir "$($side[0])_$r.json"
        if ($Extra) { python -m benchmarks.chunk_profile --output $json $Extra.Split(" ") }
        else { python -m benchmarks.chunk_profile --output $json }
        Pop-Location
    }
}
