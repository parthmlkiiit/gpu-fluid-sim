# Windows version of run.sh: builds with scripts\build_windows.ps1, then runs
# the same experiments through run.sh in Git Bash (so both platforms use one
# list of experiments).
#
#   powershell -ExecutionPolicy Bypass -File scripts\run_windows.ps1

$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)

& "$PSScriptRoot\build_windows.ps1"

# ffmpeg for the video comes from the imageio-ffmpeg wheel.
if (-not (Test-Path tools\ffmpeg\imageio_ffmpeg)) {
  python -m pip install --quiet --target tools\ffmpeg imageio-ffmpeg
}
$ffmpeg = Get-ChildItem tools\ffmpeg\imageio_ffmpeg\binaries\ffmpeg*.exe |
          Select-Object -First 1

$nvrtcBin = (Resolve-Path tools\cuda\nvrtc\nvidia\cuda_nvrtc\bin).Path
$env:PATH = "$nvrtcBin;$env:PATH"
$env:BIN = "bin/gpu_fluid.exe"
$env:FFMPEG = $ffmpeg.FullName.Replace("\", "/")
$env:PYTHON = "python"
$env:SKIP_BUILD = "1"

# Plain "bash" can resolve to WSL; use the bash that ships with Git for Windows.
$gitBash = Join-Path (Split-Path (Split-Path (Get-Command git).Source)) "bin\bash.exe"
& $gitBash run.sh
if ($LASTEXITCODE -ne 0) { throw "run.sh failed" }
