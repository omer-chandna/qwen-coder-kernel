# Build and run benchmarks\attn_bench.cu: attention kernels v1 vs v2 on random data (no model load, ~150 MB VRAM).
#   powershell -ExecutionPolicy Bypass -File scripts\attn_bench.ps1 [-Cuda <toolkit root>] [-Case <n>]
param(
    [string]$Cuda = "",
    [int]$Case = -1
)
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$cands = @($Cuda, $env:AK_CUDA_ROOT, $env:CUDA_PATH) | Where-Object { $_ -and (Test-Path (Join-Path $_ "bin\nvcc.exe")) }
if (Test-Path "$Root\build\cuda_bin.txt") {
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
$exe = "$Root\build-dev\attn_bench.exe"
$cmd = "`"$vcvars`" >nul && `"$CudaRoot\bin\nvcc.exe`" -arch=sm_89 -O3 -std=c++17 -I`"$Root\include`" -I`"$Root\native\cuda`" -I`"$CudaRoot\include`" -Xcompiler /EHsc -diag-suppress 550,177 -o `"$exe`" `"$Root\benchmarks\attn_bench.cu`" -L`"$CudaRoot\lib`" -L`"$CudaRoot\lib\x64`" -lcublas"
cmd /c $cmd
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
$env:PATH = "$CudaRoot\bin;" + $env:PATH
& $exe $Case
exit $LASTEXITCODE
