# Runs the V4 optimizer for each airfoil basis ONE AT A TIME, low priority,
# single-threaded BLAS, with a RAM watchdog (kills the run above $MaxGB).
param([string[]]$Airfoils = @("naca2412","e817","hq2512","eh2010","mh32","sd7032"),
      [double]$MaxGB = 5.0, [int]$Starts = 3)
$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
$env:PYTHONUTF8 = "1"; $env:FOIL_CASE = "zeeko_bullet750"; $env:V4_STARTS = "$Starts"
$env:OMP_NUM_THREADS = "1"; $env:OPENBLAS_NUM_THREADS = "1"; $env:MKL_NUM_THREADS = "1"
foreach ($af in $Airfoils) {
  $env:FOIL_AIRFOIL = $af
  $log = Join-Path $root "outputs\log_zeeko_$af.txt"
  $p = Start-Process -FilePath (Join-Path $root "venv\Scripts\python.exe") `
        -ArgumentList "src\optAlgoV4.py" -WorkingDirectory $root -NoNewWindow -PassThru `
        -RedirectStandardOutput $log -RedirectStandardError "$log.err"
  $p.PriorityClass = "BelowNormal"
  $peak = 0
  while (-not $p.HasExited) {
    Start-Sleep -Seconds 3
    # venv\Scripts\python.exe is a launcher: the real interpreter is its CHILD.
    $ids = @($p.Id) + @(Get-CimInstance Win32_Process -Filter "ParentProcessId=$($p.Id)" |
                        ForEach-Object { $_.ProcessId })
    $ws = (Get-Process -Id $ids -ErrorAction SilentlyContinue |
           Measure-Object WorkingSet64 -Sum).Sum / 1GB
    if ($ws -gt $peak) { $peak = $ws }
    if ($ws -gt $MaxGB) { Stop-Process -Id $ids -Force -ErrorAction SilentlyContinue; "KILLED: RAM {0:N1} GB > $MaxGB" -f $ws | Add-Content $log; break }
  }
  "{0}  peak RAM {1:N2} GB  exit {2}" -f $af, $peak, $p.ExitCode
}
"ALL DONE"
