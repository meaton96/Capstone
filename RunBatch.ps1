##############################################################################
#  RunBatchParallel.ps1
#
#  Launches one Unity process per PDR rule simultaneously, then waits for all
#  to finish and merges their individual CSVs into a single results.csv.
#
#  Usage:
#    .\RunBatch.ps1 `
#        -ExePath     ".\capstone.exe" `
#        -BatchConfig ".\BatchConfigs\BatchConfigs.json" `
#        -ResultsDir  ".\Results" `
#        -Repeats     3 `
#        -Timescale   100 `
#        -LogLevel    Low
#
#  Each worker writes:  Results\results_<RULE>.csv
#  Final merged output: Results\results.csv
##############################################################################

param(
    [Parameter(Mandatory)][string] $ExePath,
    [Parameter(Mandatory)][string] $BatchConfig,
    [string]  $ResultsDir = ".\Results",
    [int]     $Repeats = 1,
    [float]   $Timescale = 100,
    [string]  $LogLevel = "Low"
)

$Rules = @(
    "SPT_SMPT",
    "SPT_SRWT",
    "LPT_MMUR",
    "LPT_SMPT",
    "SRT_SRWT",
    "SRT_SMPT",
    "LRT_MMUR",
    "SDT_SRWT"
)

# ── Resolve absolute paths so child processes can find files ──────────────
$ExePath = Resolve-Path $ExePath
$BatchConfig = Resolve-Path $BatchConfig
if (-not (Test-Path $ResultsDir)) { New-Item -ItemType Directory -Path $ResultsDir | Out-Null }
$ResultsDir = Resolve-Path $ResultsDir

Write-Host "[Launcher] Starting $($Rules.Count) workers simultaneously..." -ForegroundColor Cyan
Write-Host "[Launcher] Exe:     $ExePath"
Write-Host "[Launcher] Config:  $BatchConfig"
Write-Host "[Launcher] Results: $ResultsDir"
Write-Host ""

$jobs = @()

foreach ($rule in $Rules) {
    $suffix = "_$rule"
    $logFile = Join-Path $ResultsDir "worker_${rule}.log"

    $args = @(
        "-batchmode", "-nographics",
        "-batchconfig", "`"$BatchConfig`"",
        "-rules", $rule,
        "-outputsuffix", $suffix,
        "-repeats", $Repeats,
        "-timescale", $Timescale,
        "-loglevel", $LogLevel,
        "-logFile", "`"$logFile`""   # Unity's own log redirect
    )

    Write-Host "[Launcher] Spawning worker: $rule"
    $proc = Start-Process -FilePath $ExePath `
        -ArgumentList $args `
        -PassThru `
        -WindowStyle Hidden
    $jobs += [PSCustomObject]@{ Rule = $rule; Process = $proc; Log = $logFile }
}

Write-Host ""
Write-Host "[Launcher] All $($jobs.Count) workers launched. Waiting for completion..." -ForegroundColor Yellow

# ── Poll and report progress ──────────────────────────────────────────────
$startTime = Get-Date
$lastReport = $startTime

while ($true) {
    $running = @($jobs | Where-Object { -not $_.Process.HasExited })
    $finished = @($jobs | Where-Object { $_.Process.HasExited })

    $now = Get-Date
    if (($now - $lastReport).TotalSeconds -ge 30) {
        $elapsed = ($now - $startTime).TotalMinutes
        Write-Host "[Launcher] $($finished.Count)/$($jobs.Count) done  ($([math]::Round($elapsed,1)) min elapsed)" `
            -ForegroundColor Cyan
        foreach ($j in $running) {
            Write-Host "  - Still running: $($j.Rule)"
        }
        $lastReport = $now
    }

    if ($running.Count -eq 0) { break }
    Start-Sleep -Seconds 5
}

$totalMin = [math]::Round(((Get-Date) - $startTime).TotalMinutes, 1)
Write-Host ""
Write-Host "[Launcher] All workers finished in ${totalMin} min." -ForegroundColor Green

# ── Check exit codes ──────────────────────────────────────────────────────
$failed = @($jobs | Where-Object { $_.Process.ExitCode -ne 0 })
if ($failed.Count -gt 0) {
    Write-Host "[Launcher] WARNING: $($failed.Count) worker(s) exited with errors:" -ForegroundColor Red
    foreach ($j in $failed) {
        Write-Host "  - $($j.Rule)  (exit code $($j.Process.ExitCode))  log: $($j.Log)" -ForegroundColor Red
    }
}

# ── Merge CSVs ────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "[Launcher] Merging CSV files..." -ForegroundColor Cyan

$mergedPath = Join-Path $ResultsDir "results.csv"
$headerWritten = $false
$rowsTotal = 0

foreach ($rule in $Rules) {
    $csv = Join-Path $ResultsDir "results_${rule}.csv"
    if (-not (Test-Path $csv)) {
        Write-Host "  [WARN] Missing: $csv" -ForegroundColor Yellow
        continue
    }

    $lines = Get-Content $csv
    if ($lines.Count -lt 2) { continue }   # header only — no data

    if (-not $headerWritten) {
        $lines[0] | Out-File -FilePath $mergedPath -Encoding utf8
        $headerWritten = $true
    }

    # Skip header row of each subsequent file
    $dataLines = $lines | Select-Object -Skip 1
    $dataLines | Out-File -FilePath $mergedPath -Encoding utf8 -Append
    $rowsTotal += $dataLines.Count
    Write-Host "  Merged $($dataLines.Count) rows from results_${rule}.csv"
}

Write-Host ""
Write-Host "[Launcher] Done. $rowsTotal total rows written to:" -ForegroundColor Green
Write-Host "  $mergedPath" -ForegroundColor Green