param(
    [switch]$NoInstallTools,
    [switch]$SkipPrepare
)

$ErrorActionPreference = 'Stop'
$Root = $PSScriptRoot
Set-Location $Root

function Test-PythonCandidate {
    param(
        [string]$Exe,
        [string[]]$Prefix = @()
    )
    if (-not (Test-Path -LiteralPath $Exe -PathType Leaf)) {
        return $false
    }
    $probeArgs = @()
    $probeArgs += $Prefix
    $probeArgs += @(
        '-c',
        'import struct,sys; raise SystemExit(0 if sys.version_info >= (3,10) and struct.calcsize("P") == 8 else 1)'
    )
    & $Exe @probeArgs *> $null
    return $LASTEXITCODE -eq 0
}

function Resolve-Python {
    $candidates = @()
    $py = Get-Command py.exe -ErrorAction SilentlyContinue
    if ($py) {
        foreach ($version in @('3.12', '3.11', '3.10')) {
            $candidates += @{ Exe = $py.Source; Prefix = @("-$version") }
        }
    }

    $python = Get-Command python.exe -ErrorAction SilentlyContinue
    if ($python) {
        $candidates += @{ Exe = $python.Source; Prefix = @() }
    }

    $launcher = Join-Path $env:LOCALAPPDATA 'Programs\Python\Launcher\py.exe'
    if (Test-Path -LiteralPath $launcher -PathType Leaf) {
        foreach ($version in @('3.12', '3.11', '3.10')) {
            $candidates += @{ Exe = $launcher; Prefix = @("-$version") }
        }
    }

    $python312 = Join-Path $env:LOCALAPPDATA 'Programs\Python\Python312\python.exe'
    if (Test-Path -LiteralPath $python312 -PathType Leaf) {
        $candidates += @{ Exe = $python312; Prefix = @() }
    }

    foreach ($candidate in $candidates) {
        if (Test-PythonCandidate -Exe $candidate.Exe -Prefix $candidate.Prefix) {
            return $candidate
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

try {
    $drive = (Get-Item -LiteralPath $Root).PSDrive
    if ($drive -and $null -ne $drive.Free -and [int64]$drive.Free -lt 60GB) {
        Write-Warning "The repository drive has less than 60 GB free. The ~24 GB GGUF plus the ~21 GB K3 trunk need substantial SSD headroom."
    }
} catch {
    Write-Verbose "Could not query repository drive free space: $_"
}

$python = Resolve-Python
if (-not $python) {
    if ($NoInstallTools) {
        throw 'A 64-bit Python >= 3.10 was not found. Install Python 3.12 with: winget install -e --id Python.Python.3.12'
    }
    Install-WingetPackage 'Python.Python.3.12'
    $env:Path += ";$env:LOCALAPPDATA\Programs\Python\Launcher;$env:LOCALAPPDATA\Programs\Python\Python312"
    $python = Resolve-Python
    if (-not $python) {
        throw 'Python 3.12 was installed but a compatible 64-bit interpreter is not visible yet. Close PowerShell, reopen it, then run setup.ps1 again.'
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
if (Test-Path -LiteralPath $venvPython -PathType Leaf) {
    if (-not (Test-PythonCandidate -Exe $venvPython)) {
        Write-Warning 'Existing .venv uses an unsupported Python build; recreating it with a compatible 64-bit Python.'
        Remove-Item -LiteralPath $venv -Recurse -Force
    }
}
if (-not (Test-Path -LiteralPath $venvPython -PathType Leaf)) {
    $pyArgs = @()
    $pyArgs += $python.Prefix
    $pyArgs += @('-m', 'venv', $venv)
    & $python.Exe @pyArgs
    if ($LASTEXITCODE -ne 0) { throw "Python venv creation failed rc=$LASTEXITCODE" }
}
if (-not (Test-PythonCandidate -Exe $venvPython)) {
    throw 'The runtime virtual environment is not 64-bit Python >= 3.10.'
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

Write-Host 'Downloading the pinned GGUF and tokenizer with low-RAM resumable curl...'
& (Join-Path $Root 'scripts\download_model.ps1')
if ($LASTEXITCODE -ne 0) { throw "model/tokenizer download failed rc=$LASTEXITCODE" }

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
