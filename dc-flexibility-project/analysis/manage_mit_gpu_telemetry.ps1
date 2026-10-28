param(
    [Parameter(Mandatory=$true)]
    [ValidateSet('start','status','stop','resume')][string]$Action,
    [string]$Root = 'D:\mit-supercloud-gpu-summary-20261028',
    [string]$Python = 'D:\anaconda\python.exe',
    [ValidateRange(1,16)][int]$Concurrency = 16,
    [int]$Limit = 0
)

$ErrorActionPreference = 'Stop'
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'
$scriptPath = Join-Path $PSScriptRoot 'stream_mit_gpu_telemetry.py'
$progressPath = Join-Path $Root 'progress.json'
$runPath = Join-Path $Root 'run.json'
$stopPath = Join-Path $Root 'STOP'

if ($Action -eq 'status') {
    if (Test-Path -LiteralPath $progressPath) { Get-Content -LiteralPath $progressPath -Encoding UTF8 }
    if (Test-Path -LiteralPath $runPath) {
        $runRecord = Get-Content -LiteralPath $runPath -Raw -Encoding UTF8 | ConvertFrom-Json
        Get-Process -Id $runRecord.pid -ErrorAction SilentlyContinue | Select-Object Id,StartTime,CPU,WorkingSet64
    }
    exit
}
if ($Action -eq 'stop') {
    New-Item -ItemType File -Force -Path $stopPath | Out-Null
    Write-Output 'STOP created. Waiting for scheduled objects to commit before exit.'
    exit
}
if (Test-Path -LiteralPath $runPath) {
    $runRecord = Get-Content -LiteralPath $runPath -Raw -Encoding UTF8 | ConvertFrom-Json
    $existingProcess = Get-CimInstance Win32_Process -Filter "ProcessId=$($runRecord.pid)"
    if ($existingProcess -and $existingProcess.CommandLine.Contains($scriptPath)) {
        throw "Process $($runRecord.pid) is still running. Wait for it to exit."
    }
}
if ($Action -eq 'resume' -and (Test-Path -LiteralPath $stopPath)) {
    Remove-Item -LiteralPath $stopPath
}
$logDirectory = Join-Path $Root 'logs'
New-Item -ItemType Directory -Force -Path $logDirectory | Out-Null
$stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$stdoutPath = Join-Path $logDirectory "run-$stamp.stdout.log"
$stderrPath = Join-Path $logDirectory "run-$stamp.stderr.log"
$arguments = @('-u',('"'+$scriptPath+'"'),'run','--root',('"'+$Root+'"'),'--concurrency',"$Concurrency")
if ($Limit -gt 0) { $arguments += @('--limit',"$Limit") }
$taskProcess = Start-Process -FilePath $Python -ArgumentList $arguments -WindowStyle Hidden `
    -RedirectStandardOutput $stdoutPath -RedirectStandardError $stderrPath -PassThru
$record = @{ pid=$taskProcess.Id; stdout=$stdoutPath; stderr=$stderrPath; root=$Root; started=(Get-Date -Format o) }
$record | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $Root 'launch.json') -Encoding UTF8
$record | ConvertTo-Json
