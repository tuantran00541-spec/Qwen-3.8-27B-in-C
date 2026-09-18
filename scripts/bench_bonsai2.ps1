param(
    [ValidateSet("PTQ1_0", "PQ2_0", "Both")]
    [string]$Quant = "Both",
    [int[]]$Threads = @(),
    [int]$PromptTokens = 512,
    [int]$GenTokens = 128,
    [int]$Repetitions = 2
)

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Bin = Join-Path $Root "tools\prism-llama\llama-bench.exe"
$ModelDir = Join-Path $Root "models\bonsai2"
$OutDir = Join-Path $Root "work\bonsai2"
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

if (-not (Test-Path -LiteralPath $Bin -PathType Leaf)) {
    throw "Prism llama-bench.exe not found. Run .\scripts\setup_bonsai2.ps1 first."
}
if ($PromptTokens -lt 1 -or $GenTokens -lt 1 -or $Repetitions -lt 1) {
    throw "PromptTokens, GenTokens and Repetitions must all be >= 1."
}

$Models = @{
    "PTQ1_0" = "Ternary-Bonsai-2-27B-PTQ1_0.gguf"
    "PQ2_0" = "Ternary-Bonsai-2-27B-PQ2_0.gguf"
}

$logical = [Environment]::ProcessorCount
if ($Threads.Count -eq 0) {
    $candidates = @(4, 6, 8, 10, 12) | Where-Object { $_ -le $logical }
    if ($candidates.Count -eq 0) {
        $candidates = @([Math]::Max(1, $logical))
    } elseif ($candidates[-1] -ne $logical -and $logical -le 16) {
        $candidates += $logical
    }
    $Threads = @($candidates | Sort-Object -Unique)
}
foreach ($t in $Threads) {
    if ($t -lt 1) { throw "Thread counts must be >= 1." }
}

$targets = if ($Quant -eq "Both") { @("PTQ1_0", "PQ2_0") } else { @($Quant) }

$cpu = Get-CimInstance Win32_Processor | Select-Object -First 1
$cs = Get-CimInstance Win32_ComputerSystem
$os = Get-CimInstance Win32_OperatingSystem
$hardware = [ordered]@{
    cpu = $cpu.Name
    logical_processors = [int]$cpu.NumberOfLogicalProcessors
    cores = [int]$cpu.NumberOfCores
    ram_gib = [Math]::Round([double]$cs.TotalPhysicalMemory / 1GB, 3)
    windows = $os.Caption
    windows_version = $os.Version
}
$hardware | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $OutDir "hardware.json") -Encoding utf8

$results = @()
foreach ($q in $targets) {
    $model = Join-Path $ModelDir $Models[$q]
    if (-not (Test-Path -LiteralPath $model -PathType Leaf)) {
        throw "$q model missing at $model. Run .\scripts\setup_bonsai2.ps1 -Quant $Quant first."
    }

    foreach ($t in $Threads) {
        $tag = "$($q.ToLowerInvariant())-t$t"
        $stdout = Join-Path $OutDir "$tag.stdout.txt"
        $stderr = Join-Path $OutDir "$tag.stderr.txt"
        Remove-Item -LiteralPath $stdout,$stderr -Force -ErrorAction SilentlyContinue

        $args = @(
            "-m", ('"' + $model + '"'),
            "-ngl", "0",
            "-fa", "1",
            "-t", "$t",
            "-p", "$PromptTokens",
            "-n", "$GenTokens",
            "-r", "$Repetitions"
        )
        $argString = $args -join " "

        Write-Host ""
        Write-Host "=== Bonsai 2 $q / threads=$t ===" -ForegroundColor Cyan
        Write-Host "$Bin $argString"

        $started = Get-Date
        $p = Start-Process -FilePath $Bin -ArgumentList $argString -NoNewWindow -PassThru -RedirectStandardOutput $stdout -RedirectStandardError $stderr
        [int64]$peakWorkingSet = 0
        while (-not $p.HasExited) {
            Start-Sleep -Milliseconds 250
            try {
                $p.Refresh()
                if ($p.WorkingSet64 -gt $peakWorkingSet) {
                    $peakWorkingSet = $p.WorkingSet64
                }
            } catch {}
        }
        $p.WaitForExit()
        $ended = Get-Date

        if (Test-Path -LiteralPath $stdout) {
            Get-Content -LiteralPath $stdout | Write-Host
        }
        if (Test-Path -LiteralPath $stderr) {
            $errText = Get-Content -Raw -LiteralPath $stderr
            if (-not [string]::IsNullOrWhiteSpace($errText)) {
                Write-Host $errText
            }
        }

        $entry = [ordered]@{
            quant = $q
            threads = $t
            prompt_tokens = $PromptTokens
            generation_tokens = $GenTokens
            repetitions = $Repetitions
            exit_code = $p.ExitCode
            wall_seconds = [Math]::Round(($ended - $started).TotalSeconds, 3)
            peak_working_set_gib = [Math]::Round([double]$peakWorkingSet / 1GB, 3)
            model_bytes = (Get-Item -LiteralPath $model).Length
            stdout = [IO.Path]::GetFileName($stdout)
            stderr = [IO.Path]::GetFileName($stderr)
        }
        $results += [pscustomobject]$entry

        if ($p.ExitCode -ne 0) {
            $results | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $OutDir "benchmark-results.json") -Encoding utf8
            throw "llama-bench failed for $q threads=$t rc=$($p.ExitCode)"
        }
    }
}

$results | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $OutDir "benchmark-results.json") -Encoding utf8
Write-Host ""
Write-Host "Results: $(Join-Path $OutDir 'benchmark-results.json')"
Write-Host "BONSAI2_REFERENCE_BENCH_PASS"
