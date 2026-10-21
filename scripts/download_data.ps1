# Downloads the binary version of CIFAR-10 (~163 MB) into data\cifar-10-batches-bin.
# CIFAR-10: Alex Krizhevsky, "Learning Multiple Layers of Features from Tiny Images", 2009.
$ErrorActionPreference = "Stop"

$Url = "https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz"
$Md5 = "c32a1d4ab5d03f1284b67883e8d87530"

$Root = Split-Path -Parent $PSScriptRoot
$DataDir = Join-Path $Root "data"
$Target = Join-Path $DataDir "cifar-10-batches-bin"
$Archive = Join-Path $DataDir "cifar-10-binary.tar.gz"

if (Test-Path (Join-Path $Target "test_batch.bin")) {
    Write-Host "CIFAR-10 is already in $Target"
    exit 0
}

New-Item -ItemType Directory -Force -Path $DataDir | Out-Null
Write-Host "Downloading $Url"
$ProgressPreference = "SilentlyContinue"  # makes Invoke-WebRequest much faster
Invoke-WebRequest -Uri $Url -OutFile $Archive

$Actual = (Get-FileHash -Algorithm MD5 $Archive).Hash.ToLower()
if ($Actual -ne $Md5) {
    throw "Checksum mismatch (expected $Md5, got $Actual); delete $Archive and retry"
}

Write-Host "Extracting..."
tar -xzf $Archive -C $DataDir   # tar ships with Windows 10 (1803) and later
Remove-Item $Archive
Write-Host "Done: $Target"
