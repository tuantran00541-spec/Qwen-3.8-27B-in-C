param(
    [Parameter(Position=0)]
    [string]$Prompt = "",
    [int]$MaxNewTokens = 32,
    [int]$Threads = 4,
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
if ($MaxNewTokens -lt 1) { throw 'MaxNewTokens must be >= 1.' }
if ($Threads -lt 1 -or $Threads -gt 48) { throw 'Threads must be in [1,48].' }
if ([string]::IsNullOrWhiteSpace($Prompt)) { $Prompt = Read-Host 'You' }
if ([string]::IsNullOrWhiteSpace($Prompt)) { throw 'Prompt is empty.' }

$env:PYTHONPATH = Join-Path $Root 'qwen38'
$env:QWEN38_EXPF_COMPAT_LIB = Join-Path $Build 'qwen_glibc_expf_compat.dll'

$argsList = @(
    '-u', (Join-Path $Root 'runtime\win32_bonsai2.py'), 'run',
    '--model', $Model,
    '--tokenizer-json', $Tokenizer,
    '--work-dir', (Join-Path $Root 'work\bonsai2-k3'),
    '--build-dir', $Build,
    '--output', (Join-Path $Root 'work\bonsai2-generation.json'),
    '--max-new-tokens', "$MaxNewTokens",
    '--threads', "$Threads",
    '--prompt', $Prompt
)
if (-not $LowRam) { $argsList += '--resident-decoder' }

& $Python @argsList
if ($LASTEXITCODE -ne 0) { throw "Bonsai generation failed rc=$LASTEXITCODE" }
