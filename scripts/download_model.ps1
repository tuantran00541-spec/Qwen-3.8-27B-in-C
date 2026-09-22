param(
    [ValidateSet("Q6", "Bonsai2")]
    [string]$Runtime = "Q6",
    [string]$ModelDir = ""
)

$ErrorActionPreference = 'Stop'
$Root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
if ([string]::IsNullOrWhiteSpace($ModelDir)) {
    $ModelDir = Join-Path $Root 'models'
} elseif (-not [System.IO.Path]::IsPathRooted($ModelDir)) {
    $ModelDir = Join-Path $Root $ModelDir
}
New-Item -ItemType Directory -Force -Path $ModelDir | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $ModelDir 'qwen-official') | Out-Null

$curl = Get-Command curl.exe -ErrorAction SilentlyContinue
if (-not $curl) {
    throw 'curl.exe was not found. Windows 10/11 normally includes it.'
}

function Get-Sha256Lower([string]$Path) {
    return (Get-FileHash -Algorithm SHA256 -Path $Path).Hash.ToLowerInvariant()
}

function Move-BadDownload([string]$Path, [string]$Label) {
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        return $null
    }
    $bad = "$Path.bad.$([DateTime]::UtcNow.ToString('yyyyMMddHHmmss'))"
    Move-Item -LiteralPath $Path -Destination $bad -Force
    Write-Warning "$Label failed integrity validation; moved it to $bad"
    return $bad
}

if ($Runtime -eq 'Bonsai2') {
    $BonsaiDir = Join-Path $ModelDir 'bonsai2'
    New-Item -ItemType Directory -Force -Path $BonsaiDir | Out-Null
    $ModelName = 'Ternary-Bonsai-2-27B-PTQ1_0.gguf'
    $ModelPath = Join-Path $BonsaiDir $ModelName
    $ExpectedSha = '53107f530aa52eb00912263ab1ee29bd199261c87cd7b4ad4ca1318c1fe33ee3'
    $ModelUrl = 'https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf/resolve/6ed5e12bf84b7a63069882c91dd9e9218647d17b/Ternary-Bonsai-2-27B-PTQ1_0.gguf?download=true'
} else {
    $ModelName = 'Qwen3.8-27B-Q6_K_L.gguf'
    $ModelPath = Join-Path $ModelDir $ModelName
    $ExpectedSha = 'a487690b9f17de581857c4ae484dab50800335bb9eb978a4fb02c0465629dc0a'
    $ModelUrl = 'https://huggingface.co/bartowski/Qwen3.8-27B-GGUF/resolve/main/Qwen3.8-27B-Q6_K_L.gguf?download=true'
}
$ModelPart = "$ModelPath.part"

if (Test-Path -LiteralPath $ModelPath -PathType Leaf) {
    Write-Host 'Existing GGUF found; verifying SHA256...'
    $actual = Get-Sha256Lower $ModelPath
    if ($actual -eq $ExpectedSha) {
        Write-Host 'Pinned GGUF already present and valid.'
    } else {
        Move-BadDownload $ModelPath 'Existing GGUF' | Out-Null
    }
}

if (-not (Test-Path -LiteralPath $ModelPath -PathType Leaf)) {
    if (Test-Path -LiteralPath $ModelPart -PathType Leaf) {
        $size = (Get-Item -LiteralPath $ModelPart).Length
        Write-Host "Resuming GGUF download from existing partial file ($size bytes)."
    } else {
        Write-Host 'Starting resumable GGUF download.'
    }

    & $curl.Source `
        --fail `
        --location `
        --retry 10 `
        --retry-delay 3 `
        --retry-all-errors `
        --continue-at - `
        --output $ModelPart `
        $ModelUrl
    if ($LASTEXITCODE -ne 0) {
        throw "curl GGUF download failed rc=$LASTEXITCODE. The .part file was kept for resume."
    }

    Write-Host 'Download finished; verifying pinned GGUF SHA256...'
    $actual = Get-Sha256Lower $ModelPart
    if ($actual -ne $ExpectedSha) {
        $bad = Move-BadDownload $ModelPart 'Downloaded GGUF'
        throw "GGUF SHA mismatch actual=$actual expected=$ExpectedSha. Corrupt completed download was quarantined at: $bad. Rerun the script to start a clean download."
    }
    Move-Item -LiteralPath $ModelPart -Destination $ModelPath -Force
}

$TokenizerPath = Join-Path $ModelDir 'qwen-official\tokenizer.json'
$TokenizerPart = "$TokenizerPath.part"
$TokenizerSha = '0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3'
$TokenizerUrl = 'https://huggingface.co/Qwen/Qwen3.8-27B/resolve/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0/tokenizer.json?download=true'

if (Test-Path -LiteralPath $TokenizerPath -PathType Leaf) {
    Write-Host 'Existing pinned tokenizer found; verifying SHA256...'
    $actualTokenizer = Get-Sha256Lower $TokenizerPath
    if ($actualTokenizer -eq $TokenizerSha) {
        Write-Host 'Pinned tokenizer already present and valid.'
    } else {
        Move-BadDownload $TokenizerPath 'Existing tokenizer' | Out-Null
    }
}

if (-not (Test-Path -LiteralPath $TokenizerPath -PathType Leaf)) {
    if (Test-Path -LiteralPath $TokenizerPart -PathType Leaf) {
        $size = (Get-Item -LiteralPath $TokenizerPart).Length
        Write-Host "Resuming tokenizer download from existing partial file ($size bytes)."
    } else {
        Write-Host 'Starting resumable pinned tokenizer download.'
    }

    & $curl.Source `
        --fail `
        --location `
        --retry 5 `
        --retry-delay 2 `
        --retry-all-errors `
        --continue-at - `
        --output $TokenizerPart `
        $TokenizerUrl
    if ($LASTEXITCODE -ne 0) {
        throw "curl tokenizer download failed rc=$LASTEXITCODE. The .part file was kept for resume."
    }

    $actualTokenizer = Get-Sha256Lower $TokenizerPart
    if ($actualTokenizer -ne $TokenizerSha) {
        $bad = Move-BadDownload $TokenizerPart 'Downloaded tokenizer'
        throw "Tokenizer SHA mismatch actual=$actualTokenizer expected=$TokenizerSha. Corrupt completed download was quarantined at: $bad. Rerun the script to start a clean download."
    }
    Move-Item -LiteralPath $TokenizerPart -Destination $TokenizerPath -Force
}

Write-Host "Runtime: $Runtime"
Write-Host "GGUF: $ModelPath"
Write-Host "Tokenizer: $TokenizerPath"
Write-Host 'QWEN38_NATIVE_WINDOWS_DOWNLOAD_PASS'
