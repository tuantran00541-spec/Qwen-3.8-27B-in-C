param(
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

$ModelName = 'Qwen3.8-27B-Q6_K_L.gguf'
$ModelPath = Join-Path $ModelDir $ModelName
$ModelPart = "$ModelPath.part"
$ExpectedSha = 'a487690b9f17de581857c4ae484dab50800335bb9eb978a4fb02c0465629dc0a'
$ModelUrl = 'https://huggingface.co/bartowski/Qwen3.8-27B-GGUF/resolve/main/Qwen3.8-27B-Q6_K_L.gguf?download=true'

function Get-Sha256Lower([string]$Path) {
    return (Get-FileHash -Algorithm SHA256 -Path $Path).Hash.ToLowerInvariant()
}

if (Test-Path $ModelPath) {
    Write-Host 'Existing GGUF found; verifying SHA256...'
    $actual = Get-Sha256Lower $ModelPath
    if ($actual -eq $ExpectedSha) {
        Write-Host 'Pinned GGUF already present and valid.'
    } else {
        $bad = "$ModelPath.bad.$([DateTime]::UtcNow.ToString('yyyyMMddHHmmss'))"
        Move-Item -Force $ModelPath $bad
        Write-Warning "Existing GGUF hash was wrong; moved it to $bad"
    }
}

if (-not (Test-Path $ModelPath)) {
    if (Test-Path $ModelPart) {
        $size = (Get-Item $ModelPart).Length
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
        throw "GGUF SHA mismatch actual=$actual expected=$ExpectedSha. Partial file was kept: $ModelPart"
    }
    Move-Item -Force $ModelPart $ModelPath
}

$TokenizerPath = Join-Path $ModelDir 'qwen-official\tokenizer.json'
$TokenizerUrl = 'https://huggingface.co/Qwen/Qwen3.8-27B/resolve/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0/tokenizer.json?download=true'
if (-not (Test-Path $TokenizerPath)) {
    $TokenizerPart = "$TokenizerPath.part"
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
        throw "curl tokenizer download failed rc=$LASTEXITCODE"
    }
    Move-Item -Force $TokenizerPart $TokenizerPath
}

Write-Host "GGUF: $ModelPath"
Write-Host "Tokenizer: $TokenizerPath"
Write-Host 'QWEN38_NATIVE_WINDOWS_DOWNLOAD_PASS'
