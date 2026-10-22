# Downloads the binary version of CIFAR-10 (~163 MB) into data\cifar-10-batches-bin,
# or with `cifar100` the binary version of CIFAR-100 (~161 MB) into data\cifar-100-binary.
# Alex Krizhevsky, "Learning Multiple Layers of Features from Tiny Images", 2009.
param([ValidateSet("cifar10", "cifar100")][string]$Dataset = "cifar10")
$ErrorActionPreference = "Stop"

if ($Dataset -eq "cifar100") {
    $Name = "CIFAR-100"
    $Url = "https://www.cs.toronto.edu/~kriz/cifar-100-binary.tar.gz"
    $Md5 = "03b5dce01913d631647c71ecec9e9cb8"
    $Folder = "cifar-100-binary"
    $Marker = "test.bin"
} else {
    $Name = "CIFAR-10"
    $Url = "https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz"
    $Md5 = "c32a1d4ab5d03f1284b67883e8d87530"
    $Folder = "cifar-10-batches-bin"
    $Marker = "test_batch.bin"
}

$Root = Split-Path -Parent $PSScriptRoot
$DataDir = Join-Path $Root "data"
$Target = Join-Path $DataDir $Folder
$Archive = Join-Path $DataDir ($Url.Split("/")[-1])

if (Test-Path (Join-Path $Target $Marker)) {
    Write-Host "$Name is already in $Target"
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
