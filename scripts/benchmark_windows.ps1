#requires -Version 5.1
<#
Run from Windows PowerShell. Uses synthetic inputs only and uploads nothing.
First run installs missing Rust/MSVC build tools through winget.
Results: benchmark-results/<timestamp>/ and a matching ZIP beside it.
#>
[CmdletBinding()]
param(
    [ValidateRange(1, 20)][int]$Repeats = 3,
    [ValidateRange(64, 100000)][int]$Blocks = 2048,
    [ValidateRange(1, 64)][int[]]$Workers = @(1, 2, 4, 8, 16, 32, 64),
    [switch]$SkipTests,
    [switch]$SkipSetup
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$repo = Split-Path -Parent $PSScriptRoot
$toolchain = '1.98.0-x86_64-pc-windows-msvc'
$stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$resultDir = Join-Path $repo "benchmark-results\$stamp"
$null = New-Item -ItemType Directory -Path $resultDir -Force
$utf8 = New-Object System.Text.UTF8Encoding($false)
$runs = New-Object System.Collections.Generic.List[object]
$originalLocation = Get-Location
$savedFlags = $env:RUSTFLAGS
$savedEncodedFlags = $env:CARGO_ENCODED_RUSTFLAGS
$savedTarget = $env:CARGO_BUILD_TARGET
$savedTargetDir = $env:CARGO_TARGET_DIR
$failed = $false

function Write-Json($Value, [string]$Path) {
    [System.IO.File]::WriteAllText($Path, (ConvertTo-Json -InputObject $Value -Depth 20), $utf8)
}

function Refresh-Path {
    $env:Path = "$env:USERPROFILE\.cargo\bin;" +
        [Environment]::GetEnvironmentVariable('Path', 'Machine') + ';' +
        [Environment]::GetEnvironmentVariable('Path', 'User')
}

function Install-Package([string]$Id, [string]$Override = '') {
    if ($SkipSetup) { throw "Missing prerequisite: $Id. Re-run without -SkipSetup to install it." }
    if (-not (Get-Command winget -ErrorAction SilentlyContinue)) {
        throw 'Install App Installer from Microsoft Store to enable winget, then run this script again.'
    }
    Write-Host "Installing $Id. Accept the installer elevation prompt if shown."
    $installArgs = @('install', '--id', $Id, '--exact', '--source', 'winget',
        '--accept-package-agreements', '--accept-source-agreements')
    if ($Override) { $installArgs += @('--override', $Override) }
    & winget @installArgs
    if ($LASTEXITCODE -eq 3010) { throw 'Installation needs a Windows restart. Restart and run this script again.' }
    if ($LASTEXITCODE -ne 0) { throw "Installation of $Id failed (exit $LASTEXITCODE)." }
    Refresh-Path
}

# Start-Process avoids treating native stderr progress as PowerShell errors.
function Invoke-Logged([string]$Program, [string[]]$Arguments, [string]$Name, [switch]$ShowProgress) {
    $stdout = Join-Path $resultDir "$Name.stdout.log"
    $stderr = Join-Path $resultDir "$Name.stderr.log"
    # Arguments here never contain embedded quotes or trailing backslashes.
    $quoted = @($Arguments | ForEach-Object { '"' + $_ + '"' })
    $watch = [Diagnostics.Stopwatch]::StartNew()
    $process = Start-Process -FilePath $Program -ArgumentList $quoted -NoNewWindow -PassThru `
        -RedirectStandardOutput $stdout -RedirectStandardError $stderr
    # Retain the process handle so Windows PowerShell can read its exit code
    # after it exits. Wait for this command, not unrelated descendants.
    $null = $process.Handle
    $nextReport = 10.0
    if ($ShowProgress) {
        Write-Host "Process $($process.Id). Logs: $stdout and $stderr"
    }
    while (-not $process.WaitForExit(1000)) {
        if ($ShowProgress -and $watch.Elapsed.TotalSeconds -ge $nextReport) {
            Write-Host ("Still waiting for {0} ({1:N0}s). Latest log lines:" -f $Name, $watch.Elapsed.TotalSeconds)
            Get-Content $stderr -Tail 5 | Write-Host
            Get-Content $stdout -Tail 5 | Write-Host
            $nextReport = $watch.Elapsed.TotalSeconds + 10.0
        }
    }
    $process.WaitForExit()
    $watch.Stop()
    if ($process.ExitCode -ne 0) {
        Get-Content $stderr -Tail 35 | Write-Host
        Get-Content $stdout -Tail 20 | Write-Host
        throw "$Name failed (exit $($process.ExitCode)). See $resultDir"
    }
    if ($ShowProgress) {
        Get-Content $stderr -Tail 5 | Write-Host
        Get-Content $stdout -Tail 5 | Write-Host
    }
    return [pscustomobject]@{ Stdout = $stdout; WallSeconds = $watch.Elapsed.TotalSeconds }
}

function Median([double[]]$Values) {
    $ordered = @($Values | Sort-Object)
    $middle = [int][Math]::Floor($ordered.Count / 2)
    if ($ordered.Count % 2) { return $ordered[$middle] }
    return ($ordered[$middle - 1] + $ordered[$middle]) / 2
}

try {
    Set-Location $repo
    if ($env:OS -ne 'Windows_NT') { throw 'This runner requires 64-bit Windows.' }
    if (-not [Environment]::Is64BitOperatingSystem) { throw '64-bit Windows is required.' }
    Refresh-Path
    $vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
    $compiler = @()
    if (Test-Path $vswhere) {
        $compiler = @(& $vswhere -products '*' -latest -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath)
    }
    if (-not $compiler.Count) {
        Install-Package 'Microsoft.VisualStudio.2022.BuildTools' '--passive --wait --norestart --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended'
    }
    if (-not (Get-Command rustup -ErrorAction SilentlyContinue)) { Install-Package 'Rustlang.Rustup' }
    Write-Host "Preparing Rust $toolchain..."
    & rustup toolchain install $toolchain --profile minimal
    if ($LASTEXITCODE -ne 0) { throw 'Rust toolchain installation failed.' }
    $cargo = (Get-Command cargo).Source

    # Portable baseline: runtime SIMD dispatch remains enabled. Changes apply
    # only to this process and are restored on exit.
    $env:RUSTFLAGS = $null
    $env:CARGO_ENCODED_RUSTFLAGS = $null
    $env:CARGO_BUILD_TARGET = $null
    $env:CARGO_TARGET_DIR = Join-Path $repo 'target\windows-scaling'
    $cpus = @(Get-CimInstance Win32_Processor)
    $system = Get-CimInstance Win32_ComputerSystem
    $os = Get-CimInstance Win32_OperatingSystem
    $gitCommit = 'unknown'
    if (Get-Command git -ErrorAction SilentlyContinue) {
        $gitCommit = (& git rev-parse HEAD | Out-String).Trim()
        & git status --porcelain | Out-File (Join-Path $resultDir 'working-tree.txt') -Encoding utf8
    }
    $rustVersion = (& rustc "+$toolchain" -vV | Out-String).Trim()
    Get-FileHash -Algorithm SHA256 -Path (Join-Path $repo 'Cargo.lock'), $PSCommandPath |
        Select-Object @{Name='File'; Expression={Split-Path -Leaf $_.Path}}, Hash |
        ConvertTo-Json | Out-File (Join-Path $resultDir 'runner-hashes.json') -Encoding utf8
    Write-Json ([ordered]@{
        started = (Get-Date).ToString('o'); commit = $gitCommit; rust = $rustVersion
        cpu = @($cpus | Select-Object Name, NumberOfCores, NumberOfLogicalProcessors)
        ram_bytes = $system.TotalPhysicalMemory; windows = $os.Caption; windows_version = $os.Version
        workers = $Workers; repeats = $Repeats; conditioning_sites = 10000; variables = 7
        blocks = $Blocks; block_edge_m = 2; points_per_block = 8; window = 64
        memory_budget_bytes = 17179869184; cache_pages = 131072; tests_skipped = [bool]$SkipTests
        timing = 'seconds measures estimation/simulation, excluding conditioning/store construction; separate build and process times retained'
        scope = 'Synthetic data fitting in RAM. This does not measure beyond-RAM performance or complete statistical acceptance.'
    }) (Join-Path $resultDir 'machine.json')
    & powercfg /getactivescheme | Out-File (Join-Path $resultDir 'power-plan.txt') -Encoding utf8

    if (-not $SkipTests) {
        Write-Host 'Running focused correctness checks. First compilation can take several minutes...'
        $null = Invoke-Logged $cargo @("+$toolchain", 'test', '--locked', '--release',
            '-p', 'terrustrial-core', '--features', 'sqlite', '--lib',
            '--test', 'packed_spatial', '--test', 'parallel_gaussian',
            '--test', 'solve_residuals', '--test', 'spatial_hierarchy') 'correctness' -ShowProgress
        Write-Host 'Correctness checks passed.'
    }
    Write-Host 'Building the optimized benchmark...'
    $null = Invoke-Logged $cargo @("+$toolchain", 'build', '--locked', '--release',
        '-p', 'terrustrial-core', '--features', 'sqlite', '--example', 'storage_comparison') 'build' -ShowProgress
    $binary = Join-Path $env:CARGO_TARGET_DIR 'release\examples\storage_comparison.exe'
    if (-not (Test-Path $binary)) { throw "Benchmark executable missing: $binary" }

    $expectedHashes = @{}
    foreach ($mode in @('estimate', 'simulate')) {
        # Discarded preflight runs warm the executable and check backend parity.
        Write-Host "Warming up $mode..."
        foreach ($backend in @('resident', 'packed')) {
            $warm = Invoke-Logged $binary @($backend, $mode, '10000', "$Blocks", '1', '131072', '64') "warm-$mode-$backend"
            $warmResult = Get-Content $warm.Stdout -Raw | ConvertFrom-Json
            if ($expectedHashes.ContainsKey($mode) -and $expectedHashes[$mode] -ne $warmResult.hash) {
                throw "$mode resident/packed warmup results differ. No performance acceptance."
            }
            $expectedHashes[$mode] = $warmResult.hash
        }
        for ($repeat = 1; $repeat -le $Repeats; $repeat++) {
            # Reverse sweep and backend order each repeat to limit order bias.
            $sweep = @($Workers | Sort-Object -Unique)
            $backends = @('resident', 'packed')
            if ($repeat % 2 -eq 0) { [Array]::Reverse($sweep); [Array]::Reverse($backends) }
            foreach ($worker in $sweep) {
                foreach ($backend in $backends) {
                    $name = "$mode-$backend-w$worker-r$repeat"
                    Write-Host "$name ... " -NoNewline
                    $run = Invoke-Logged $binary @($backend, $mode, '10000', "$Blocks", "$worker", '131072', '64') $name
                    $record = Get-Content $run.Stdout -Raw | ConvertFrom-Json
                    if ($record.seconds -le 0 -or $record.hash -ne $expectedHashes[$mode]) {
                        throw "$name has an invalid time or different output hash. No performance acceptance."
                    }
                    $record | Add-Member -NotePropertyName repeat -NotePropertyValue $repeat
                    $record | Add-Member -NotePropertyName process_wall_seconds -NotePropertyValue $run.WallSeconds
                    $runs.Add($record)
                    Write-Json @($runs.ToArray()) (Join-Path $resultDir 'runs.json')
                    Write-Host ('{0:N3}s, {1:N0} blocks/s, hash OK' -f $record.seconds, ($Blocks / $record.seconds))
                }
            }
        }
    }
    $summary = @(
        foreach ($mode in @('estimate', 'simulate')) {
            foreach ($backend in @('resident', 'packed')) {
                $baseline = @($runs | Where-Object { $_.mode -eq $mode -and $_.backend -eq $backend -and $_.workers -eq 1 })
                foreach ($worker in @($Workers | Sort-Object -Unique)) {
                    $cases = @($runs | Where-Object { $_.mode -eq $mode -and $_.backend -eq $backend -and $_.workers -eq $worker })
                    $seconds = Median @($cases | ForEach-Object { $_.seconds })
                    $speedup = $null
                    if ($baseline.Count) { $speedup = [Math]::Round((Median @($baseline | ForEach-Object { $_.seconds })) / $seconds, 2) }
                    [pscustomobject]@{
                        Mode = $mode; Backend = $backend; Workers = $worker
                        Seconds = [Math]::Round($seconds, 4); BlocksPerSecond = [Math]::Round($Blocks / $seconds)
                        Speedup = $speedup
                    }
                }
            }
        }
    )
    $summary | Format-Table -AutoSize | Out-Host
    $summary | Export-Csv (Join-Path $resultDir 'summary.csv') -NoTypeInformation
    Write-Json $summary (Join-Path $resultDir 'summary.json')
    Write-Host 'All measured outputs match across worker counts and storage backends.'
} catch {
    $failed = $true
    [System.IO.File]::WriteAllText((Join-Path $resultDir 'failure.txt'), ($_ | Out-String), $utf8)
    Write-Host "FAILED: $_" -ForegroundColor Red
} finally {
    $env:RUSTFLAGS = $savedFlags
    $env:CARGO_ENCODED_RUSTFLAGS = $savedEncodedFlags
    $env:CARGO_BUILD_TARGET = $savedTarget
    $env:CARGO_TARGET_DIR = $savedTargetDir
    Set-Location $originalLocation
    $zip = "$resultDir.zip"
    Compress-Archive -Path (Join-Path $resultDir '*') -DestinationPath $zip -Force
    Write-Host "Results saved locally: $zip"
    Write-Host 'Nothing was uploaded.'
}
if ($failed) { exit 1 }