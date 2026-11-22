# Build and run a standalone kernel bench on random data (no model load):
#   attn: benchmarks\attn_bench.cu, attention v1 vs v2 (~150 MB of VRAM)
#   gemm: benchmarks\gemm_bench.cu, prefill int8 GEMMs at the model's shapes (~1 GB of VRAM)
#   powershell -ExecutionPolicy Bypass -File scripts\cuda_bench.ps1 -Bench attn|gemm [-Cuda <toolkit root>] [-Case <n>]
param(
    [ValidateSet("attn", "gemm")][string]$Bench = "attn",
    [string]$Cuda = "",
    [int]$Case = -1
)
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$cands = @($env:AK_CUDA_ROOT, $env:CUDA_PATH) | Where-Object { $_ -and (Test-Path (Join-Path $_ "bin\nvcc.exe")) }
if (Test-Path "$Root\build\cuda_bin.txt") {  # the toolkit scripts\build.ps1 used
    foreach ($b in Get-Content "$Root\build\cuda_bin.txt") {
        $r = Split-Path -Parent $b.Trim([char]0xFEFF)
        if (Test-Path (Join-Path $r "bin\nvcc.exe")) { $cands = @($r) + $cands }
    }
}
if ($Cuda) { $cands = @($Cuda) + $cands }
$CudaRoot = @($cands)[0]
if (-not $CudaRoot) { throw "CUDA toolkit not found; pass -Cuda <root>." }
$vcvars = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
New-Item -ItemType Directory -Force "$Root\build-dev" | Out-Null
$exe = "$Root\build-dev\${Bench}_bench.exe"
$src = "$Root\benchmarks\${Bench}_bench.cu"
$cmd = "`"$vcvars`" >nul && `"$CudaRoot\bin\nvcc.exe`" -arch=sm_89 -O3 -std=c++17 -I`"$Root\include`" -I`"$Root\native\cuda`" -I`"$CudaRoot\include`" -Xcompiler /EHsc -diag-suppress 550,177 -o `"$exe`" `"$src`" -L`"$CudaRoot\lib`" -L`"$CudaRoot\lib\x64`" -lcublas"
cmd /c $cmd
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
$env:PATH = "$CudaRoot\bin;" + $env:PATH
& $exe $Case
exit $LASTEXITCODE
