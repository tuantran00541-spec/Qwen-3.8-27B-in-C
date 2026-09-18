param(
    [ValidateSet("PTQ1_0", "PQ2_0", "Both")]
    [string]$Quant = "PTQ1_0"
)

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$ModelDir = Join-Path $Root "models\bonsai2"
$BinDir = Join-Path $Root "tools\prism-llama"
New-Item -ItemType Directory -Force -Path $ModelDir,$BinDir | Out-Null

$ModelRevision = "6ed5e12bf84b7a63069882c91dd9e9218647d17b"
$ModelRepo = "prism-ml/Ternary-Bonsai-2-27B-gguf"
$ReleaseTag = "prism-b10683-d8f26ee"
$CpuAsset = "llama-$ReleaseTag-bin-win-cpu-x64.zip"
$CpuUrl = "https://github.com/PrismML-Eng/llama.cpp/releases/download/$ReleaseTag/$CpuAsset"

$Models = @{
    "PTQ1_0" = @{
        File = "Ternary-Bonsai-2-27B-PTQ1_0.gguf"
        Sha256 = "53107f530aa52eb00912263ab1ee29bd199261c87cd7b4ad4ca1318c1fe33ee3"
        Bytes = [int64]5946648928
    }
    "PQ2_0" = @{
        File = "Ternary-Bonsai-2-27B-PQ2_0.gguf"
        Sha256 = "3907dc1658db1f78a9826bf8d5bcb8dc65db0d466388937af57f2294fae62ec1"
        Bytes = [int64]7206168928
    }
}

$curl = Get-Command curl.exe -ErrorAction SilentlyContinue
if (-not $curl) {
    throw "curl.exe is required for resumable low-RAM downloads."
}

function Get-Sha256Lower([string]$Path) {
    return (Get-FileHash -Algorithm SHA256 -LiteralPath $Path).Hash.ToLowerInvariant()
}

function Download-VerifiedFile {
    param(
        [string]$Url,
        [string]$Destination,
        [string]$ExpectedSha,
        [int64]$ExpectedBytes
    )

    if (Test-Path -LiteralPath $Destination -PathType Leaf) {
        $item = Get-Item -LiteralPath $Destination
        if ($item.Length -eq $ExpectedBytes) {
            Write-Host "Verifying existing $($item.Name) ..."
            if ((Get-Sha256Lower $Destination) -eq $ExpectedSha) {
                Write-Host "[OK] $($item.Name) is already present and verified." -ForegroundColor Green
                return
            }
        }
        $bad = "$Destination.bad.$([DateTime]::UtcNow.ToString('yyyyMMddHHmmss'))"
        Move-Item -LiteralPath $Destination -Destination $bad -Force
        Write-Warning "Existing file failed integrity validation and was moved to $bad"
    }

    $part = "$Destination.part"
    if (Test-Path -LiteralPath $part -PathType Leaf) {
        Write-Host "Resuming $([IO.Path]::GetFileName($Destination)) from $((Get-Item -LiteralPath $part).Length) bytes ..."
    } else {
        Write-Host "Downloading $([IO.Path]::GetFileName($Destination)) ..."
    }

    $curlArgs = @(
        "--fail",
        "--location",
        "--retry", "10",
        "--retry-delay", "3",
        "--retry-all-errors",
        "--continue-at", "-",
        "--output", $part,
        $Url
    )
    & $curl.Source @curlArgs
    if ($LASTEXITCODE -ne 0) {
        throw "curl download failed rc=$LASTEXITCODE. Partial file kept at $part"
    }

    $actualBytes = (Get-Item -LiteralPath $part).Length
    if ($actualBytes -ne $ExpectedBytes) {
        throw "Downloaded file size mismatch: actual=$actualBytes expected=$ExpectedBytes path=$part"
    }

    Write-Host "Verifying SHA256 for $([IO.Path]::GetFileName($Destination)) ..."
    $actualSha = Get-Sha256Lower $part
    if ($actualSha -ne $ExpectedSha) {
        $bad = "$part.bad.$([DateTime]::UtcNow.ToString('yyyyMMddHHmmss'))"
        Move-Item -LiteralPath $part -Destination $bad -Force
        throw "SHA256 mismatch actual=$actualSha expected=$ExpectedSha; quarantined at $bad"
    }

    Move-Item -LiteralPath $part -Destination $Destination -Force
    Write-Host "[OK] Verified $([IO.Path]::GetFileName($Destination))" -ForegroundColor Green
}

function Install-PrismCpuBinary {
    $stamp = Join-Path $BinDir ".llama_release"
    $bench = Join-Path $BinDir "llama-bench.exe"
    if ((Test-Path -LiteralPath $bench -PathType Leaf) -and (Test-Path -LiteralPath $stamp -PathType Leaf)) {
        if ((Get-Content -Raw -LiteralPath $stamp).Trim() -eq $ReleaseTag) {
            Write-Host "[OK] Prism llama.cpp CPU binary $ReleaseTag already installed." -ForegroundColor Green
            return
        }
    }

    if (Test-Path -LiteralPath $BinDir) {
        Remove-Item -LiteralPath $BinDir -Recurse -Force
    }
    New-Item -ItemType Directory -Force -Path $BinDir | Out-Null

    $zip = Join-Path $env:TEMP $CpuAsset
    if (Test-Path -LiteralPath $zip) {
        Remove-Item -LiteralPath $zip -Force
    }
    Write-Host "Downloading pinned Prism llama.cpp CPU binary $ReleaseTag ..."
    $curlArgs = @(
        "--fail",
        "--location",
        "--retry", "5",
        "--retry-delay", "2",
        "--retry-all-errors",
        "--output", $zip,
        $CpuUrl
    )
    & $curl.Source @curlArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Prism llama.cpp binary download failed rc=$LASTEXITCODE"
    }

    Expand-Archive -LiteralPath $zip -DestinationPath $BinDir -Force
    Remove-Item -LiteralPath $zip -Force

    if (-not (Test-Path -LiteralPath $bench -PathType Leaf)) {
        $nested = Get-ChildItem -LiteralPath $BinDir -Filter "llama-bench.exe" -File -Recurse | Select-Object -First 1
        if (-not $nested) {
            throw "Pinned Prism archive did not contain llama-bench.exe"
        }
        $nestedRoot = $nested.Directory.FullName
        Get-ChildItem -LiteralPath $nestedRoot -Force | ForEach-Object {
            Move-Item -LiteralPath $_.FullName -Destination $BinDir -Force
        }
    }

    Set-Content -LiteralPath $stamp -Value $ReleaseTag -NoNewline -Encoding utf8
    Write-Host "[OK] Installed Prism llama.cpp CPU binary $ReleaseTag" -ForegroundColor Green
}

Install-PrismCpuBinary

$targets = if ($Quant -eq "Both") { @("PTQ1_0", "PQ2_0") } else { @($Quant) }
foreach ($q in $targets) {
    $spec = $Models[$q]
    $dest = Join-Path $ModelDir $spec.File
    $url = "https://huggingface.co/$ModelRepo/resolve/$ModelRevision/$($spec.File)?download=true"
    Download-VerifiedFile -Url $url -Destination $dest -ExpectedSha $spec.Sha256 -ExpectedBytes $spec.Bytes
}

Write-Host ""
Write-Host "Bonsai 2 reference setup ready."
Write-Host "  Prism binary: $ReleaseTag"
Write-Host "  Model revision: $ModelRevision"
Write-Host "  Quant(s): $($targets -join ', ')"
Write-Host "Run:"
Write-Host "  .\scripts\bench_bonsai2.ps1 -Quant $Quant"
Write-Host "BONSAI2_REFERENCE_SETUP_PASS"
