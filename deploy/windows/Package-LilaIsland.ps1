param(
    [string]$Runtime = "win-x64",
    [switch]$SingleFile,
    [switch]$SelfContained = $true,
    [string]$OutputDir
)

$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$projectPath = Join-Path $repoRoot "windows\Lila.Desktop\Lila.Desktop.csproj"

if ([string]::IsNullOrWhiteSpace($OutputDir)) {
    $OutputDir = Join-Path $repoRoot "dist\LilaIsland-$Runtime"
}

New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

$publishArgs = @(
    "publish", $projectPath,
    "-c", "Release",
    "-r", $Runtime,
    "-o", $OutputDir,
    "--self-contained", ($SelfContained.IsPresent ? "true" : "false")
)

if ($SingleFile) {
    $publishArgs += "-p:PublishSingleFile=true"
    $publishArgs += "-p:IncludeNativeLibrariesForSelfExtract=true"
}

Write-Host "Packaging Lila Island ($Runtime) -> $OutputDir"
& dotnet @publishArgs
if ($LASTEXITCODE -ne 0) {
    throw "dotnet publish failed with exit code $LASTEXITCODE"
}

$zipPath = "$OutputDir.zip"
if (Test-Path $zipPath) {
    Remove-Item -Force $zipPath
}
Compress-Archive -Path (Join-Path $OutputDir "*") -DestinationPath $zipPath
Write-Host "Created standalone package: $zipPath"
