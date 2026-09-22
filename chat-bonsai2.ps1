param(
    [int]$MaxNewTokens = 32,
    [int]$Threads = 4,
    [ValidateSet('Low','Medium','Full')]
    [string]$MemoryMode = 'Medium',
    [switch]$LowRam
)

$ErrorActionPreference = 'Stop'
$Root = $PSScriptRoot
$Python = Join-Path $Root '.venv\Scripts\python.exe'
$Model = Join-Path $Root 'models\bonsai2\Ternary-Bonsai-2-27B-PTQ1_0.gguf'
$Tokenizer = Join-Path $Root 'models\qwen-official\tokenizer.json'
$Build = Join-Path $Root 'build\win32'

$Required = @(
    $Python, $Model, $Tokenizer,
    (Join-Path $Build 'qwen_glibc_expf_compat.dll'),
    (Join-Path $Build 'qwen_bonsai2_quant.dll'),
    (Join-Path $Build 'qwen_bonsai2_gdn_state.dll')
)
$Missing = @($Required | Where-Object { -not (Test-Path -LiteralPath $_ -PathType Leaf) })
if ($Missing.Count -gt 0) {
    throw "Bonsai setup is incomplete. Run .\setup-bonsai2.ps1 first. Missing count=$($Missing.Count)"
}

$env:PYTHONPATH = Join-Path $Root 'qwen38'
$env:QWEN38_EXPF_COMPAT_LIB = Join-Path $Build 'qwen_glibc_expf_compat.dll'

$argsList = @(
    '-u', (Join-Path $Root 'runtime\win32_bonsai2.py'), 'chat',
    '--model', $Model,
    '--tokenizer-json', $Tokenizer,
    '--work-dir', (Join-Path $Root 'work\bonsai2-k3'),
    '--build-dir', $Build,
    '--max-new-tokens', "$MaxNewTokens",
    '--threads', "$Threads"
)
# Legacy -LowRam stays supported; otherwise default to the balanced Medium profile.
$effectiveMode = if ($LowRam) { 'Low' } else { $MemoryMode }
$argsList += @('--memory-mode', $effectiveMode.ToLowerInvariant())
Write-Host "Bonsai 2 memory mode: $effectiveMode"

& $Python @argsList
if ($LASTEXITCODE -ne 0) { throw "Bonsai chat failed rc=$LASTEXITCODE" }
