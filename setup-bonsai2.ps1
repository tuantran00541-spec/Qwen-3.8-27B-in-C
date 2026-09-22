param(
    [switch]$NoInstallTools,
    [string]$PythonExe = ""
)

$ErrorActionPreference = 'Stop'
$Root = $PSScriptRoot
Set-Location $Root

function Get-PythonInfo {
    param(
        [string]$Exe,
        [string[]]$Prefix = @()
    )

    if ([string]::IsNullOrWhiteSpace($Exe)) {
        return $null
    }

    try {
        $probeArgs = @()
        $probeArgs += $Prefix
        $probeArgs += @(
            '-c',
            'import struct,sys; print("QWEN38_PYTHON_OK|%d|%d|%d|%s" % (sys.version_info[0], sys.version_info[1], struct.calcsize("P"), sys.executable))'
        )
        $lines = @(& $Exe @probeArgs 2>&1)
        $rc = $LASTEXITCODE
        if ($rc -ne 0) {
            return $null
        }

        $marker = $lines | Where-Object {
            $_.ToString().StartsWith('QWEN38_PYTHON_OK|')
        } | Select-Object -Last 1

        if (-not $marker) {
            return $null
        }

        $parts = $marker.ToString().Split('|', 5)
        if ($parts.Count -ne 5) {
            return $null
        }

        $major = [int]$parts[1]
        $minor = [int]$parts[2]
        $pointerBytes = [int]$parts[3]
        $reportedExe = $parts[4]

        if (($major -lt 3) -or (($major -eq 3) -and ($minor -lt 10))) {
            return $null
        }
        if ($pointerBytes -ne 8) {
            return $null
        }

        return @{
            Exe = $Exe
            Prefix = @($Prefix)
            ReportedExe = $reportedExe
            Version = "$major.$minor"
            PointerBytes = $pointerBytes
        }
    } catch {
        return $null
    }
}

function Resolve-Python {
    param(
        [string]$ExplicitExe = ""
    )

    $candidates = @()

    if (-not [string]::IsNullOrWhiteSpace($ExplicitExe)) {
        $candidates += @{
            Exe = $ExplicitExe
            Prefix = @()
            Label = "explicit: $ExplicitExe"
        }
    }

    $python = Get-Command python -CommandType Application -ErrorAction SilentlyContinue
    if ($python) {
        $candidates += @{
            Exe = $python.Source
            Prefix = @()
            Label = $python.Source
        }
    }

    $pythonExeCmd = Get-Command python.exe -CommandType Application -ErrorAction SilentlyContinue
    if ($pythonExeCmd) {
        $candidates += @{
            Exe = $pythonExeCmd.Source
            Prefix = @()
            Label = $pythonExeCmd.Source
        }
    }

    $py = Get-Command py.exe -CommandType Application -ErrorAction SilentlyContinue
    if ($py) {
        foreach ($v in @('3.12','3.11','3.10')) {
            $candidates += @{
                Exe = $py.Source
                Prefix = @("-$v")
                Label = "py.exe -$v"
            }
        }
    }

    foreach ($known in @(
        (Join-Path $env:LOCALAPPDATA 'Programs\Python\Python312\python.exe'),
        (Join-Path $env:LOCALAPPDATA 'Programs\Python\Python311\python.exe'),
        (Join-Path $env:LOCALAPPDATA 'Programs\Python\Python310\python.exe'),
        'C:\Program Files\Python312\python.exe',
        'C:\Program Files\Python311\python.exe',
        'C:\Program Files\Python310\python.exe'
    )) {
        if (Test-Path -LiteralPath $known -PathType Leaf) {
            $candidates += @{
                Exe = $known
                Prefix = @()
                Label = $known
            }
        }
    }

    $seen = @{}
    foreach ($candidate in $candidates) {
        $key = "$($candidate.Exe)|$($candidate.Prefix -join ' ')"
        if ($seen.ContainsKey($key)) {
            continue
        }
        $seen[$key] = $true

        $info = Get-PythonInfo -Exe $candidate.Exe -Prefix $candidate.Prefix
        if ($info) {
            Write-Host "Using Python: $($candidate.Label)"
            Write-Host "  reported executable: $($info.ReportedExe)"
            Write-Host "  version: $($info.Version), pointer bytes: $($info.PointerBytes)"
            return @{
                Exe = $candidate.Exe
                Prefix = @($candidate.Prefix)
                ReportedExe = $info.ReportedExe
            }
        }
    }

    return $null
}

function Resolve-Clang {
    $clang = Get-Command clang.exe -CommandType Application -ErrorAction SilentlyContinue
    if ($clang) {
        return $clang.Source
    }

    foreach ($known in @(
        'C:\Program Files\LLVM\bin\clang.exe',
        'C:\Program Files (x86)\LLVM\bin\clang.exe'
    )) {
        if (Test-Path -LiteralPath $known -PathType Leaf) {
            $dir = Split-Path -Parent $known
            if (($env:Path -split ';') -notcontains $dir) {
                $env:Path = "$dir;$env:Path"
            }
            return $known
        }
    }
    return $null
}

function Install-WingetPackage([string]$Id) {
    $winget = Get-Command winget.exe -CommandType Application -ErrorAction SilentlyContinue
    if (-not $winget) {
        throw "Missing required tool and winget is unavailable. Install $Id manually."
    }

    Write-Host "Installing/checking $Id with winget..."
    & $winget.Source install -e --id $Id --accept-package-agreements --accept-source-agreements
    $rc = $LASTEXITCODE
    if ($rc -ne 0) {
        Write-Warning "winget returned rc=$rc for $Id. Re-checking whether the tool is already installed before failing."
    }
    return $rc
}

$python = $null
if (-not [string]::IsNullOrWhiteSpace($PythonExe)) {
    if (-not (Test-Path -LiteralPath $PythonExe -PathType Leaf)) {
        throw "Explicit PythonExe does not exist: $PythonExe"
    }
    $resolvedPython = (Resolve-Path -LiteralPath $PythonExe).Path
    $python = @{
        Exe = $resolvedPython
        Prefix = @()
        ReportedExe = $resolvedPython
    }
    Write-Host "Using explicit Python without installer probe: $resolvedPython"
} else {
    $python = Resolve-Python
}

if (-not $python) {
    if ($NoInstallTools) {
        throw 'A 64-bit Python >=3.10 was not found.'
    }

    Install-WingetPackage 'Python.Python.3.12' | Out-Null

    $pythonDir = Join-Path $env:LOCALAPPDATA 'Programs\Python\Python312'
    $pythonScripts = Join-Path $pythonDir 'Scripts'
    $launcherDir = Join-Path $env:LOCALAPPDATA 'Programs\Python\Launcher'
    $env:Path = "$pythonDir;$pythonScripts;$launcherDir;$env:Path"

    $python = Resolve-Python
    if (-not $python) {
        throw 'Python 3.12 is not usable after the winget check. Expected a 64-bit Python >=3.10. Run: python --version'
    }
}

$clangPath = Resolve-Clang
if (-not $clangPath) {
    if ($NoInstallTools) {
        throw 'LLVM/clang was not found.'
    }

    Install-WingetPackage 'LLVM.LLVM' | Out-Null
    $llvmDir = 'C:\Program Files\LLVM\bin'
    if (Test-Path -LiteralPath $llvmDir -PathType Container) {
        $env:Path = "$llvmDir;$env:Path"
    }

    $clangPath = Resolve-Clang
    if (-not $clangPath) {
        throw 'LLVM/clang is not usable after the winget check. Expected clang.exe under C:\Program Files\LLVM\bin.'
    }
}
Write-Host "Using clang: $clangPath"

$venv = Join-Path $Root '.venv'
$venvPython = Join-Path $venv 'Scripts\python.exe'

if (-not (Test-Path -LiteralPath $venvPython -PathType Leaf)) {
    $venvArgs = @()
    $venvArgs += @($python.Prefix)
    $venvArgs += @('-m', 'venv', $venv)
    Write-Host "Creating virtual environment with: $($python.Exe)"
    & $python.Exe @venvArgs
    if ($LASTEXITCODE -ne 0) {
        throw "venv creation failed rc=$LASTEXITCODE"
    }
}

if (-not (Test-Path -LiteralPath $venvPython -PathType Leaf)) {
    throw "Virtual environment Python was not created: $venvPython"
}

Write-Host "Using venv Python: $venvPython"
& $venvPython --version
if ($LASTEXITCODE -ne 0) {
    throw "venv Python failed to execute rc=$LASTEXITCODE"
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
