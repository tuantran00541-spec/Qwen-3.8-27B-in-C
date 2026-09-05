param(
    [int]$MaxNewTokens = 4
)

$ErrorActionPreference = 'Stop'
$Root = $PSScriptRoot
$Python = Join-Path $Root '.venv\Scripts\python.exe'
if (-not (Test-Path $Python)) {
    throw 'Runtime is not set up yet. Run .\setup.ps1 first.'
}

$env:PYTHONPATH = Join-Path $Root 'qwen38'
$env:QWEN38_EXPF_COMPAT_LIB = Join-Path $Root 'build\win32\qwen_glibc_expf_compat.dll'

& $Python -u (Join-Path $Root 'runtime\win32_generate.py') chat `
    --model (Join-Path $Root 'models\Qwen3.8-27B-Q6_K_L.gguf') `
    --inventory (Join-Path $Root 'work\inventory.json') `
    --tokenizer-json (Join-Path $Root 'models\qwen-official\tokenizer.json') `
    --work-dir (Join-Path $Root 'work\k3') `
    --build-dir (Join-Path $Root 'build\win32') `
    --max-new-tokens $MaxNewTokens
if ($LASTEXITCODE -ne 0) {
    throw "Qwen chat launcher failed rc=$LASTEXITCODE"
}
