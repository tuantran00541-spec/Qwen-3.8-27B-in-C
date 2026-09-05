param(
    [string]$OutDir = ""
)

$ErrorActionPreference = 'Stop'
$Root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$Src = Join-Path $Root 'qwen38'
if ([string]::IsNullOrWhiteSpace($OutDir)) {
    $Out = Join-Path $Root 'build\win32'
} elseif ([System.IO.Path]::IsPathRooted($OutDir)) {
    $Out = $OutDir
} else {
    $Out = Join-Path $Root $OutDir
}
New-Item -ItemType Directory -Force -Path $Out | Out-Null

$clang = Get-Command clang.exe -ErrorAction SilentlyContinue
if (-not $clang) {
    $clang = Get-Command clang -ErrorAction SilentlyContinue
}
if (-not $clang) {
    throw "clang was not found. Install LLVM first: winget install -e --id LLVM.LLVM"
}

function Build-Dll {
    param(
        [string]$Name,
        [string[]]$Sources,
        [string[]]$Exports,
        [string[]]$ExtraFlags = @()
    )
    $args = @(
        '-O3', '-std=c11', '-shared',
        '-fno-fast-math', '-fno-associative-math', '-ffp-contract=off',
        '-fno-vectorize', '-fno-slp-vectorize'
    )
    $args += $ExtraFlags
    $args += $Sources
    foreach ($symbol in $Exports) {
        $args += '-Xlinker'
        $args += "/EXPORT:$symbol"
    }
    $target = Join-Path $Out $Name
    $args += @('-o', $target)
    Write-Host "Building $Name"
    & $clang.Source @args
    if ($LASTEXITCODE -ne 0) {
        throw "clang failed for $Name rc=$LASTEXITCODE"
    }
}

$compatObj = Join-Path $Out 'qwen38_glibc_expf_compat.obj'
$compatCompileArgs = @(
    '-O3', '-std=c11', '-c', '-mfma',
    '-fno-fast-math', '-fno-associative-math', '-ffp-contract=off',
    '-fno-vectorize', '-fno-slp-vectorize',
    (Join-Path $Src 'qwen38_glibc_expf_compat.c'),
    '-o', $compatObj
)
& $clang.Source @compatCompileArgs
if ($LASTEXITCODE -ne 0) {
    throw "clang failed for glibc-compatible expf object rc=$LASTEXITCODE"
}

$redirectHeader = Join-Path $Src 'qwen38_glibc_expf_redirect.h'
$expfRedirect = @('-include', $redirectHeader)

Build-Dll 'qwen_glibc_expf_compat.dll' @($compatObj) @('qwen38_glibc_expf_compat')
Build-Dll 'qwen_quant_base.dll' @((Join-Path $Src 'gguf_quant_matvec_many_bridge.c')) @(
    'qwen_quantize_q8_k_scalar','qwen_quantize_q8_0_scalar',
    'qwen_matvec_q6_k_q8_k_scalar','qwen_matvec_q8_0_q8_0_scalar',
    'qwen_matvec_many_q6_k_q8_k_bridge','qwen_matvec_many_q8_0_q8_0_bridge') @('-mavx2')
Build-Dll 'qwen_q6_portable.dll' @((Join-Path $Src 'q6_persistent_pool_portable_exact.c')) @(
    'qwen_q6_pool_create','qwen_q6_pool_destroy','qwen_q6_pool_matvec_many',
    'qwen_q6_pool_calls','qwen_q6_pool_threads',
    'qwen_matvec_many_q6_k_q8_k_bridge','qwen_matvec_many_q8_0_q8_0_bridge') @('-mavx2')
Build-Dll 'qwen_gdn_state.dll' @((Join-Path $Src 'gdn_state_ar.c'),$compatObj) @('qwen_gdn_ar_step_f32') $expfRedirect
Build-Dll 'qwen_gdn_state_batch.dll' @((Join-Path $Src 'gdn_state_ar.c'),(Join-Path $Src 'gdn_state_batch_exact.c'),$compatObj) @('qwen_gdn_ar_batch_f32') $expfRedirect
Build-Dll 'qwen_f32.dll' @((Join-Path $Src 'f32_fsum_matvec.c')) @('qwen_matvec_f32_fsum_exact')
Build-Dll 'qwen_attention_core.dll' @((Join-Path $Src 'attention_core_exact.c'),$compatObj) @('qwen_attention_core_f32_exact') $expfRedirect
Build-Dll 'qwen_gdn_conv_silu.dll' @((Join-Path $Src 'gdn_conv_silu_exact.c'),$compatObj) @('qwen_gdn_conv_silu_many_f32_exact') $expfRedirect
Build-Dll 'qwen_gdn_output_gate.dll' @((Join-Path $Src 'gdn_output_gate_exact.c'),$compatObj) @('qwen_gdn_output_rmsnorm_gate_f32_exact') $expfRedirect
Build-Dll 'qwen_swiglu.dll' @((Join-Path $Src 'swiglu_exact.c'),$compatObj) @('qwen_swiglu_many_f32_exact') $expfRedirect
Build-Dll 'qwen_rmsnorm.dll' @((Join-Path $Src 'rmsnorm_exact.c')) @('qwen38_rmsnorm_exact_f32','qwen38_rmsnorm_heads_exact_f32')
Build-Dll 'qwen_rmsnorm_many.dll' @((Join-Path $Src 'rmsnorm_exact.c'),(Join-Path $Src 'rmsnorm_many_exact.c')) @('qwen38_rmsnorm_many_exact_f32')
Build-Dll 'qwen_rmsnorm_heads_many.dll' @((Join-Path $Src 'rmsnorm_exact.c'),(Join-Path $Src 'rmsnorm_heads_many_exact.c')) @('qwen38_rmsnorm_heads_many_exact_f32')
Build-Dll 'qwen_residual_add.dll' @((Join-Path $Src 'residual_add_exact.c')) @('qwen38_residual_add_many_exact_f32')
Build-Dll 'qwen_attention_gate.dll' @((Join-Path $Src 'attention_gate_exact.c'),$compatObj) @('qwen38_attention_gate_exact_f32') $expfRedirect
Build-Dll 'qwen_gdn_repeat_scale.dll' @((Join-Path $Src 'gdn_repeat_scale_exact.c')) @('qwen38_gdn_repeat_scale_many_exact_f32')
Build-Dll 'qwen_win32_direct_io.dll' @((Join-Path $Src 'win32_direct_io.c')) @(
    'qwen_win32_direct_open_utf8','qwen_win32_direct_close',
    'qwen_win32_direct_alignment','qwen_win32_direct_read',
    'qwen_win32_direct_no_buffering','qwen_win32_direct_overlapped',
    'qwen_win32_buffer_create','qwen_win32_buffer_ptr','qwen_win32_buffer_destroy')

Write-Host "DLL bundle: $Out"
Write-Host 'QWEN38_NATIVE_WINDOWS_DLL_BUILD_PASS'
