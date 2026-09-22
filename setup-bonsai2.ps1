param(
    [switch]$NoInstallTools
)

$ErrorActionPreference = 'Stop'
$Root = $PSScriptRoot
Set-Location $Root

function Test-Python([string]$Exe, [string[]]$Prefix = @()) {
    try {
        & $Exe @Prefix -c 'import struct,sys; raise SystemExit(0 if sys.version_info >= (3,10) and struct.calcsize("P") == 8 else 1)' *> $null
        return $LASTEXITCODE -eq 0
    } catch {
        return $false
    }
}

function Resolve-Python {
    $py = Get-Command py.exe -ErrorAction SilentlyContinue
    if ($py) {
        foreach ($v in @('3.12','3.11','3.10')) {
            if (Test-Python $py.Source @("-$v")) {
                return @{ Exe = $py.Source; Prefix = @("-$v") }
            }
        }
    }
    $python = Get-Command python.exe -ErrorAction SilentlyContinue
    if ($python -and (Test-Python $python.Source)) {
        return @{ Exe = $python.Source; Prefix = @() }
    }
    return $null
}

function Install-Winget([string]$Id) {
    $winget = Get-Command winget.exe -ErrorAction SilentlyContinue
    if (-not $winget) {
        throw "Missing required tool and winget is unavailable. Install $Id manually."
    }
    & $winget.Source install -e --id $Id --accept-package-agreements --accept-source-agreements
    if ($LASTEXITCODE -ne 0) { throw "winget failed for $Id rc=$LASTEXITCODE" }
}

$python = Resolve-Python
if (-not $python) {
    if ($NoInstallTools) { throw '64-bit Python >=3.10 is required.' }
    Install-Winget 'Python.Python.3.12'
    $env:Path += ";$env:LOCALAPPDATA\Programs\Python\Launcher;$env:LOCALAPPDATA\Programs\Python\Python312"
    $python = Resolve-Python
    if (-not $python) {
        throw 'Python was installed but is not visible yet. Reopen PowerShell and rerun setup-bonsai2.ps1.'
    }
}

$clang = Get-Command clang.exe -ErrorAction SilentlyContinue
if (-not $clang) {
    if ($NoInstallTools) { throw 'LLVM/clang is required.' }
    Install-Winget 'LLVM.LLVM'
    $env:Path += ';C:\Program Files\LLVM\bin'
    $clang = Get-Command clang.exe -ErrorAction SilentlyContinue
    if (-not $clang) {
        throw 'LLVM was installed but clang is not visible yet. Reopen PowerShell and rerun setup-bonsai2.ps1.'
    }
}

$venv = Join-Path $Root '.venv'
$venvPython = Join-Path $venv 'Scripts\python.exe'
if (-not (Test-Path -LiteralPath $venvPython -PathType Leaf)) {
    & $python.Exe @($python.Prefix) -m venv $venv
    if ($LASTEXITCODE -ne 0) { throw "venv creation failed rc=$LASTEXITCODE" }
}

& $venvPython -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw "pip upgrade failed rc=$LASTEXITCODE" }
& $venvPython -m pip install tokenizers numpy
if ($LASTEXITCODE -ne 0) { throw "Python dependency install failed rc=$LASTEXITCODE" }

Write-Host 'Building native Windows runtime including Bonsai 2 PTQ1 kernels...'
& (Join-Path $Root 'scripts\build_win32.ps1')
if ($LASTEXITCODE -ne 0) { throw "native build failed rc=$LASTEXITCODE" }

$env:PYTHONPATH = Join-Path $Root 'qwen38'
$env:QWEN38_EXPF_COMPAT_LIB = Join-Path $Root 'build\win32\qwen_glibc_expf_compat.dll'

Write-Host 'Running Bonsai 2 native Windows sanity...'
& $venvPython -u (Join-Path $Root 'runtime\win32_bonsai2.py') sanity --build-dir (Join-Path $Root 'build\win32')
if ($LASTEXITCODE -ne 0) { throw "Bonsai Windows sanity failed rc=$LASTEXITCODE" }

Write-Host 'Downloading the pinned ~5.95 GB Ternary Bonsai 2 PTQ1 model and tokenizer...'
& (Join-Path $Root 'scripts\download_model.ps1') -Runtime Bonsai2
if ($LASTEXITCODE -ne 0) { throw "Bonsai model download failed rc=$LASTEXITCODE" }

Write-Host ''
Write-Host 'Bonsai 2 native Windows setup is ready.'
Write-Host 'First run will create and cache the K3 trunk once.'
Write-Host 'Try:'
Write-Host '  .\run-bonsai2.ps1 "Hello" -MaxNewTokens 8'
Write-Host 'Or:'
Write-Host '  .\chat-bonsai2.ps1 -MaxNewTokens 32'
Write-Host 'QWEN38_BONSAI2_NATIVE_WINDOWS_SETUP_PASS'
