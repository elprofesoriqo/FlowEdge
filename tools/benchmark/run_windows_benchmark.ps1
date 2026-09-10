param(
    [Parameter(Mandatory = $true)]
    [string]$Executable,

    [Parameter(Mandatory = $true)]
    [string]$Output,

    [Parameter(Mandatory = $true)]
    [ValidateRange(1, [int]::MaxValue)]
    [int]$Runs,

    [Parameter(Mandatory = $true)]
    [string]$RuntimeDirectory,

    [string]$Filter = ""
)

$benchmarkArguments = @(
    "--benchmark_repetitions=$Runs"
    "--benchmark_report_aggregates_only=true"
    "--benchmark_out=$Output"
    "--benchmark_out_format=json"
)
if ($Filter) {
    $benchmarkArguments += "--benchmark_filter=$Filter"
}

$quotedArguments = $benchmarkArguments | ForEach-Object { '"' + $_.Replace('"', '\"') + '"' }
$command = 'set "PATH=' + $RuntimeDirectory + ';%PATH%" && "' + $Executable + '" ' + `
    ($quotedArguments -join ' ')
$process = Start-Process -FilePath "$env:SystemRoot\System32\cmd.exe" `
    -ArgumentList @('/d', '/s', '/c', '"' + $command + '"') -Wait -PassThru `
    -WindowStyle Hidden -UseNewEnvironment
exit $process.ExitCode
