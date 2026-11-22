# Build build/adakernel.dll.
#   powershell -ExecutionPolicy Bypass -File scripts\build.ps1 [-Cuda <toolkit root>] [-CMake <cmake.exe>] [-Ninja <ninja.exe>]
# Needs CUDA 12.8+ (sm_89), VS 2022 Build Tools, CMake 3.24+ and Ninja. The toolkit's bin folder is recorded in
# build/cuda_bin.txt so the Python loader finds cudart / cublas at run time.
param(
    [string]$Cuda = "",
    [string]$CMake = "",
    [string]$Ninja = "",
    [string]$BuildDir = "build"   # e.g. build-dev while a server holds build\adakernel.dll (load it via ADAKERNEL_DLL)
)
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$cudaCandidates = @(
    $Cuda,
    $env:AK_CUDA_ROOT,
    $env:CUDA_PATH,
    "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.5",
    "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4",
    "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0",
    "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"
) | Where-Object { $_ -and (Test-Path (Join-Path $_ "bin\nvcc.exe")) }
if (-not $cudaCandidates) { throw "CUDA 12.8+ toolkit not found; pass -Cuda <root> (nvcc 11.x cannot target sm_89)." }
$CudaRoot = @($cudaCandidates)[0]
$env:CUDA_PATH = $CudaRoot
$env:PATH = "$CudaRoot\bin;" + $env:PATH

$vcvars = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
if (-not (Test-Path $vcvars)) { throw "VS 2022 Build Tools vcvars64.bat not found." }
if (-not $CMake) { $CMake = (Get-Command cmake -ErrorAction SilentlyContinue).Source }
if (-not $Ninja) { $Ninja = (Get-Command ninja -ErrorAction SilentlyContinue).Source }
if (-not $CMake -or -not $Ninja) { throw "cmake / ninja not found; pip install cmake ninja, or pass -CMake and -Ninja." }

$cmd = "`"$vcvars`" && `"$CMake`" -S `"$Root`" -B `"$Root\$BuildDir`" -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_COMPILER=`"$CudaRoot\bin\nvcc.exe`" -DCMAKE_CUDA_ARCHITECTURES=89 -DCMAKE_MAKE_PROGRAM=`"$Ninja`" && `"$CMake`" --build `"$Root\$BuildDir`""
cmd /c $cmd
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
$bins = @("$CudaRoot\bin", "$CudaRoot\bin\x64") | Where-Object { Test-Path $_ }
Set-Content -Path "$Root\$BuildDir\cuda_bin.txt" -Value $bins -Encoding utf8
Write-Output "Built $Root\$BuildDir\adakernel.dll with $CudaRoot"
