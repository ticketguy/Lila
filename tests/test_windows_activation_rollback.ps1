$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot '..\deploy\windows\DeploymentTransaction.ps1')

function Assert-True {
    param([bool]$Condition, [string]$Message)
    if (-not $Condition) { throw $Message }
}

$testRoot = Join-Path ([IO.Path]::GetTempPath()) ('lila-rollback-test-' + [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $testRoot | Out-Null
try {
    # A failed publish move must restore the previous working deployment.
    $install = Join-Path $testRoot 'successful-rollback-install'
    $stage = Join-Path $testRoot 'successful-rollback-stage'
    $backup = Join-Path $testRoot 'successful-rollback-backup'
    New-Item -ItemType Directory -Path $install, $stage | Out-Null
    Set-Content -LiteralPath (Join-Path $install 'version.txt') -Value 'old-working-deployment'
    Set-Content -LiteralPath (Join-Path $stage 'version.txt') -Value 'new-deployment'
    $moves = [pscustomobject]@{ Count = 0 }
    $move = {
        param($Source, $Destination)
        $moves.Count++
        if ($moves.Count -eq 2) { throw 'simulated deployment move failure' }
        Move-Item -LiteralPath $Source -Destination $Destination -ErrorAction Stop
    }.GetNewClosure()
    $errorText = ''
    try {
        Invoke-LilaDeploymentSwap -Stage $stage -InstallRoot $install -Backup $backup -MoveOperation $move
    } catch {
        $errorText = $_.Exception.Message
    }
    Assert-True ($errorText -match 'restored successfully') 'Expected the rollback to be reported as successful.'
    Assert-True ((Get-Content -LiteralPath (Join-Path $install 'version.txt') -Raw).Trim() -eq 'old-working-deployment') 'Previous deployment was not restored.'
    Assert-True (-not (Test-Path -LiteralPath $backup)) 'Backup should have been moved back after successful rollback.'

    # If both deployment and rollback moves fail, leave the only old copy intact.
    $install = Join-Path $testRoot 'failed-rollback-install'
    $stage = Join-Path $testRoot 'failed-rollback-stage'
    $backup = Join-Path $testRoot 'failed-rollback-backup'
    New-Item -ItemType Directory -Path $install, $stage | Out-Null
    Set-Content -LiteralPath (Join-Path $install 'version.txt') -Value 'old-working-deployment'
    Set-Content -LiteralPath (Join-Path $stage 'version.txt') -Value 'new-deployment'
    $moves = [pscustomobject]@{ Count = 0 }
    $move = {
        param($Source, $Destination)
        $moves.Count++
        if ($moves.Count -ge 2) { throw "simulated move failure $($moves.Count)" }
        Move-Item -LiteralPath $Source -Destination $Destination -ErrorAction Stop
    }.GetNewClosure()
    $errorText = ''
    try {
        Invoke-LilaDeploymentSwap -Stage $stage -InstallRoot $install -Backup $backup -MoveOperation $move
    } catch {
        $errorText = $_.Exception.Message
    }
    Assert-True ($errorText -match 'Rollback also failed') 'Expected rollback failure to be reported.'
    Assert-True ($errorText.Contains($backup)) 'Error must report the preserved backup path.'
    Assert-True ((Get-Content -LiteralPath (Join-Path $backup 'version.txt') -Raw).Trim() -eq 'old-working-deployment') 'Previous deployment backup was not preserved.'
    Assert-True (-not (Test-Path -LiteralPath $install)) 'A partial deployment should not shadow the preserved backup.'
} finally {
    Remove-Item -LiteralPath $testRoot -Recurse -Force -ErrorAction SilentlyContinue
}

Write-Output 'Windows activation rollback tests passed (2).'
