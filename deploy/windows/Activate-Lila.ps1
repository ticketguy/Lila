[CmdletBinding()]
param(
    [string]$Distribution = 'Ubuntu-24.04',
    [switch]$Deactivate
)

$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$project = Join-Path $repoRoot 'windows\Lila.Desktop\Lila.Desktop.csproj'
$installRoot = Join-Path $env:LOCALAPPDATA 'LilaDesktop\VoiceInput\app'
$markerPath = Join-Path $installRoot '.lila-voice-input-install.json'
$startupKey = 'HKCU:\Software\Microsoft\Windows\CurrentVersion\Run'
$startupName = 'LilaDesktop'
. (Join-Path $PSScriptRoot 'DeploymentTransaction.ps1')

function Get-ExpectedStartupValue {
    param([string]$Executable)
    return '"' + $Executable + '" --startup'
}

if ($Deactivate) {
    if (-not (Test-Path -LiteralPath $markerPath -PathType Leaf)) {
        Write-Output 'No Lila portable voice-input installation marker found; nothing to remove.'
        exit 0
    }
    $marker = Get-Content -LiteralPath $markerPath -Raw | ConvertFrom-Json
    $recordedPath = [IO.Path]::GetFullPath([string]$marker.installPath)
    $expectedPath = [IO.Path]::GetFullPath($installRoot)
    if (-not $recordedPath.Equals($expectedPath, [StringComparison]::OrdinalIgnoreCase)) {
        throw 'The install marker does not match the expected per-user application directory; refusing removal.'
    }
    $exe = Join-Path $installRoot 'Lila.Desktop.exe'
    if (Test-Path -LiteralPath $startupKey) {
        $current = (Get-ItemProperty -LiteralPath $startupKey -Name $startupName -ErrorAction SilentlyContinue).$startupName
        if ($current -and $current -ceq (Get-ExpectedStartupValue $exe)) {
            Remove-ItemProperty -LiteralPath $startupKey -Name $startupName
            Write-Output 'Removed the matching current-user startup entry.'
        }
    }
    Remove-Item -LiteralPath $installRoot -Recurse -Force
    Write-Output 'Removed the marked application deployment. Settings, credentials, models, recordings and transcripts were preserved.'
    exit 0
}

if (-not $env:LOCALAPPDATA) { throw 'LOCALAPPDATA is unavailable for this user.' }
if (-not (Test-Path -LiteralPath $project -PathType Leaf)) {
    throw "Desktop project not found at '$project'. Run this script from a complete repository clone."
}
$dotnet = Get-Command dotnet -ErrorAction SilentlyContinue
if (-not $dotnet) { throw '.NET 10 SDK is required. Install the user-scoped .NET 10 SDK and retry.' }
$sdks = & dotnet --list-sdks 2>&1
if ($LASTEXITCODE -ne 0 -or -not ($sdks | Where-Object { $_ -match '^10\.' })) {
    throw '.NET 10 SDK was not found. Install it for the current user and retry.'
}
$wsl = Get-Command wsl.exe -ErrorAction SilentlyContinue
if (-not $wsl) { throw 'WSL is required. Install/update WSL and create the configured Linux distribution first.' }
$runningList = & wsl.exe --list --quiet 2>&1 | ForEach-Object { ($_ -replace "`0", '').Trim() }
if ($LASTEXITCODE -ne 0 -or -not ($runningList | Where-Object { $_ -eq $Distribution })) {
    throw "WSL distribution '$Distribution' was not found. Install it or pass -Distribution with its exact name."
}

$parent = Split-Path -Parent $installRoot
New-Item -ItemType Directory -Path $parent -Force | Out-Null
$stage = Join-Path $parent ('.publish-' + [guid]::NewGuid().ToString('N'))
$backup = Join-Path $parent ('.previous-' + [guid]::NewGuid().ToString('N'))
try {
    & dotnet publish $project -c Release -r win-x64 --self-contained true `
        -p:PublishSingleFile=false -p:PublishTrimmed=false -o $stage
    if ($LASTEXITCODE -ne 0) { throw 'The self-contained win-x64 desktop publish failed.' }
    $exe = Join-Path $stage 'Lila.Desktop.exe'
    if (-not (Test-Path -LiteralPath $exe -PathType Leaf)) {
        throw 'Publish completed without Lila.Desktop.exe; the output is invalid.'
    }
    $marker = [ordered]@{
        schema = 1
        installPath = [IO.Path]::GetFullPath($installRoot)
        executable = 'Lila.Desktop.exe'
        installedUtc = [DateTime]::UtcNow.ToString('o')
    }
    $marker | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $stage '.lila-voice-input-install.json') -Encoding utf8

    if (Test-Path -LiteralPath $installRoot) {
        if (-not (Test-Path -LiteralPath $markerPath -PathType Leaf)) {
            throw "'$installRoot' already exists without this installer's marker; it was left untouched."
        }
    }
    Invoke-LilaDeploymentSwap -Stage $stage -InstallRoot $installRoot -Backup $backup
} finally {
    if (Test-Path -LiteralPath $stage) {
        Remove-Item -LiteralPath $stage -Recurse -Force -ErrorAction SilentlyContinue
    }
}

Write-Output "Installed Lila Desktop to '$installRoot'."
Write-Output 'Windows startup registration was not changed. Enable it explicitly in Lila Settings if desired.'
Write-Output 'Run the Linux activation script from the WSL checkout to install the user service.'
Write-Output "To remove this deployment only: .\deploy\windows\Activate-Lila.ps1 -Deactivate"
