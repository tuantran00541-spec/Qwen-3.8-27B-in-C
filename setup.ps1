param(
    [switch]$NoInstallTools,
    [switch]$SkipPrepare
)

$ErrorActionPreference = 'Stop'
$Root = $PSScriptRoot
Set-Location $Root

function Resolve-Python {
    $py = Get-Command py.exe -ErrorAction SilentlyContinue
    if ($py) {
        return @{ Exe = $py.Source; Prefix = @('-3') }
    }
    $python = Get-Command python.exe -ErrorAction SilentlyContinue
    if ($python) {
        return @{ Exe = $python.Source; Prefix = @() }
    }
    $candidates = @(
        (Join-Path $env:LOCALAPPDATA 'Programs\Python\Launcher\py.exe'),
        (Join-Path $env:LOCALAPPDATA 'Programs\Python\Python312\python.exe')
    )
    foreach ($candidate in $candidates) {
        if (Test-Path $candidate) {
            if ($candidate.EndsWith('py.exe')) {
                return @{ Exe = $candidate; Prefix = @('-3') }
            }
            return @{ Exe = $candidate; Prefix = @() }
        }
    }
    return $null
}

function Install-WingetPackage([string]$Id) {
    $winget = Get-Command winget.exe -ErrorAction SilentlyContinue
    if (-not $winget) {
        throw "Missing required tool and winget is unavailable. Install package '$Id' manually."
    }
    Write-Host "Installing $Id with winget..."
    & $winget.Source install -e --id $Id --accept-package-agreements --accept-source-agreements
    if ($LASTEXITCODE -ne 0) {
        throw "winget failed to install $Id rc=$LASTEXITCODE"
    }
}

$python = Resolve-Python
if (-not $python) {
    if ($NoInstallTools) {
        throw 'Python 3 was not found. Install it with: winget install -e --id Python.Python.3.12'
    }
    Install-WingetPackage 'Python.Python.3.12'
    $env:Path += ";$env:LOCALAPPDATA\Programs\Python\Launcher;$env:LOCALAPPDATA\Programs\Python\Python312"
    $python = Resolve-Python
    if (-not $python) {
        throw 'Python was installed but is not visible yet. Close PowerShell, reopen it, then run setup.ps1 again.'
    }
}

$clang = Get-Command clang.exe -ErrorAction SilentlyContinue
if (-not $clang) {
    if ($NoInstallTools) {
        throw 'LLVM/clang was not found. Install it with: winget install -e --id LLVM.LLVM'
    }
    Install-WingetPackage 'LLVM.LLVM'
    $env:Path += ';C:\Program Files\LLVM\bin'
    $clang = Get-Command clang.exe -ErrorAction SilentlyContinue
    if (-not $clang) {
        throw 'LLVM was installed but clang is not visible yet. Close PowerShell, reopen it, then run setup.ps1 again.'
    }
}

$curl = Get-Command curl.exe -ErrorAction SilentlyContinue
if (-not $curl) {
    throw 'curl.exe is required for low-RAM resumable model download.'
}

$venv = Join-Path $Root '.venv'
$venvPython = Join-Path $venv 'Scripts\python.exe'
if (-not (Test-Path $venvPython)) {
    $pyArgs = @()
    $pyArgs += $python.Prefix
    $pyArgs += @('-m', 'venv', $venv)
    & $python.Exe @pyArgs
    if ($LASTEXITCODE -ne 0) { throw "Python venv creation failed rc=$LASTEXITCODE" }
}

& $venvPython -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw "pip upgrade failed rc=$LASTEXITCODE" }
& $venvPython -m pip install tokenizers
if ($LASTEXITCODE -ne 0) { throw "tokenizers install failed rc=$LASTEXITCODE" }

Write-Host 'Building the exact native Windows DLL bundle...'
& (Join-Path $Root 'scripts\build_win32.ps1')
if ($LASTEXITCODE -ne 0) { throw "Win32 DLL build failed rc=$LASTEXITCODE" }

$env:PYTHONPATH = Join-Path $Root 'qwen38'
$env:QWEN38_EXPF_COMPAT_LIB = Join-Path $Root 'build\win32\qwen_glibc_expf_compat.dll'

Write-Host 'Running native Windows DLL/direct-I/O sanity...'
& $venvPython -u (Join-Path $Root 'runtime\win32_generate.py') sanity --build-dir (Join-Path $Root 'build\win32')
if ($LASTEXITCODE -ne 0) { throw "native Windows sanity failed rc=$LASTEXITCODE" }

Write-Host 'Downloading the pinned GGUF with low-RAM resumable curl...'
& (Join-Path $Root 'scripts\download_model.ps1')
if ($LASTEXITCODE -ne 0) { throw "model download failed rc=$LASTEXITCODE" }

$work = Join-Path $Root 'work'
$k3 = Join-Path $work 'k3'
New-Item -ItemType Directory -Force -Path $work,$k3 | Out-Null
$model = Join-Path $Root 'models\Qwen3.8-27B-Q6_K_L.gguf'
$inventory = Join-Path $work 'inventory.json'
$needInventory = $true
if (Test-Path $inventory) {
    try {
        $inv = Get-Content -Raw $inventory | ConvertFrom-Json
        if ($inv.status -eq 'PASS' -and $inv.sha256 -eq 'a487690b9f17de581857c4ae484dab50800335bb9eb978a4fb02c0465629dc0a') {
            $needInventory = $false
            Write-Host 'Reusing validated GGUF inventory.'
        }
    } catch {
        $needInventory = $true
    }
}
if ($needInventory) {
    Write-Host 'Validating the real 64-layer GGUF contract...'
    & $venvPython -u (Join-Path $Root 'qwen38\qwen35_gguf_decoder_contract.py') real --model $model --output $inventory
    if ($LASTEXITCODE -ne 0) { throw "GGUF contract validation failed rc=$LASTEXITCODE" }
}

if (-not $SkipPrepare) {
    Write-Host 'Preparing the execution-ordered K3 trunk (one-time SSD copy)...'
    & $venvPython -u (Join-Path $Root 'runtime\win32_generate.py') prepare --model $model --work-dir $k3 --build-dir (Join-Path $Root 'build\win32')
    if ($LASTEXITCODE -ne 0) { throw "K3 prepare failed rc=$LASTEXITCODE" }
}

Write-Host ''
Write-Host 'Native Windows setup is ready.'
Write-Host 'Run a prompt:'
Write-Host '  .\run.ps1 "Explain why the sky is blue."'
Write-Host 'Or open the simple chat shell:'
Write-Host '  .\chat.ps1'
Write-Host 'QWEN38_NATIVE_WINDOWS_SETUP_PASS'
