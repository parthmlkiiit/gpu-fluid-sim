# Windows build without Visual Studio / nvcc.
# The kernels are compiled at runtime by NVRTC, so the program itself is plain
# C++: MinGW g++ links it against nvcuda.dll (driver API, via cuda.lib) and the
# NVRTC DLL from NVIDIA's pip wheel. Needs: an NVIDIA driver, Python + pip,
# MinGW-w64 g++ (e.g. MSYS2 ucrt64).
#
#   powershell -ExecutionPolicy Bypass -File scripts\build_windows.ps1

param(
  [string]$Tools = "tools\cuda"
)

$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)

$packages = @{
  "cuda_runtime" = "nvidia-cuda-runtime-cu12"   # cuda.h + cuda.lib
  "nvrtc"        = "nvidia-cuda-nvrtc-cu12"     # nvrtc.h + nvrtc DLLs
}
$folders = @{ "cuda_runtime" = "cuda_runtime"; "nvrtc" = "cuda_nvrtc" }

foreach ($name in $packages.Keys) {
  $dir = Join-Path $Tools $name
  if (-not (Test-Path "$dir\nvidia\$($folders[$name])")) {
    Write-Output "installing $($packages[$name]) into $dir"
    python -m pip install --quiet --no-deps --target $dir $packages[$name]
    if ($LASTEXITCODE -ne 0) { throw "pip install $($packages[$name]) failed" }
  }
}

$runtime = "$Tools\cuda_runtime\nvidia\cuda_runtime"
$nvrtc = "$Tools\nvrtc\nvidia\cuda_nvrtc"
$nvrtcDll = Get-ChildItem "$nvrtc\bin\nvrtc64_*.dll" |
            Where-Object { $_.Name -notmatch "alt" } | Select-Object -First 1

New-Item -ItemType Directory -Force bin | Out-Null
$sources = Get-ChildItem src\*.cc | ForEach-Object { $_.FullName }

Write-Output "compiling $($sources.Count) files with g++"
# -static keeps libstdc++/winpthread inside the exe; cuda.lib is the import
# library for nvcuda.dll (linking the System32 DLL directly records the
# wrong name, nvcuda_loader.dll).
g++ -std=c++17 -O2 -Wall -Wextra -ffp-contract=off -static -I. `
  -isystem third_party -isystem "$runtime\include" -isystem "$nvrtc\include" `
  $sources $nvrtcDll.FullName "$runtime\lib\x64\cuda.lib" -o bin\gpu_fluid.exe
if ($LASTEXITCODE -ne 0) { throw "build failed" }

Write-Output "built bin\gpu_fluid.exe"
Write-Output "before running, put the NVRTC DLL folder on PATH:"
Write-Output "  `$env:PATH = `"$((Resolve-Path "$nvrtc\bin").Path);`$env:PATH`""
