# 本机 PowerShell 入口；额外参数转交 train_event_model.py。
$ErrorActionPreference = "Stop"
$runtime = Join-Path $env:USERPROFILE ".cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
if ((Test-Path -LiteralPath $runtime) -and (Test-Path -LiteralPath (Join-Path $PSScriptRoot ".model_deps\torch"))) {
    $python = $runtime
    $env:PYTHONPATH = (Join-Path $PSScriptRoot ".model_deps") + [IO.Path]::PathSeparator + $env:PYTHONPATH
} else {
    $python = "python"
}
if ($args.Count -gt 0 -and $args[0] -eq "--epoch-study") {
    $studyArgs = @($args | Select-Object -Skip 1)
    & $python -X utf8 (Join-Path $PSScriptRoot "release_epoch_study.py") @studyArgs
} else {
    & $python -X utf8 (Join-Path $PSScriptRoot "train_event_model.py") @args
}
exit $LASTEXITCODE

