# Download the mrt2_small model (1.1 GB; Google's Magenta RealTime 2 weights,
# CC-BY 4.0) to where mrt2 looks for it by default:
#   %USERPROFILE%\models\magenta-rt-v2\checkpoints\mrt2_small.safetensors
# Usage (PowerShell): .\scripts\download_weights.ps1 [-Dir <directory>]
param([string]$Dir = (Join-Path $HOME "models\magenta-rt-v2\checkpoints"))
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue" # the progress bar slows downloads down a lot

$Url = "https://huggingface.co/google/magenta-realtime-2/resolve/main/checkpoints/mrt2_small.safetensors"
$Sha256 = "5DD1CBC7C606C512C21DE0BCB04D4818BF0A3B873D7CBB9D1556D67D3B034DE3"
$File = Join-Path $Dir "mrt2_small.safetensors"

if ((Test-Path $File) -and ((Get-FileHash $File -Algorithm SHA256).Hash -eq $Sha256)) {
    Write-Host "already there: $File"
    exit 0
}
New-Item -ItemType Directory -Force -Path $Dir | Out-Null
Write-Host "downloading mrt2_small (1.1 GB) to $File (this takes a while)"
Invoke-WebRequest -Uri $Url -OutFile "$File.part"
Write-Host "checking the file's fingerprint (sha256)..."
if ((Get-FileHash "$File.part" -Algorithm SHA256).Hash -ne $Sha256) {
    Write-Error "the download is damaged (wrong sha256); delete $File.part and try again"
}
Move-Item -Force "$File.part" $File
Write-Host "done: $File"
