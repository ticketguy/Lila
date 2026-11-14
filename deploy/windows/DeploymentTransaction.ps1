function Invoke-LilaDeploymentSwap {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)][string]$Stage,
        [Parameter(Mandatory)][string]$InstallRoot,
        [Parameter(Mandatory)][string]$Backup,
        [scriptblock]$MoveOperation = {
            param($Source, $Destination)
            Move-Item -LiteralPath $Source -Destination $Destination -ErrorAction Stop
        },
        [scriptblock]$RemoveOperation = {
            param($Path)
            Remove-Item -LiteralPath $Path -Recurse -Force -ErrorAction Stop
        }
    )

    $hadPreviousDeployment = Test-Path -LiteralPath $InstallRoot
    if ($hadPreviousDeployment) {
        & $MoveOperation $InstallRoot $Backup
    }

    try {
        & $MoveOperation $Stage $InstallRoot
    } catch {
        $deploymentFailure = $_
        if ($hadPreviousDeployment -and (Test-Path -LiteralPath $Backup)) {
            try {
                if (Test-Path -LiteralPath $InstallRoot) {
                    & $RemoveOperation $InstallRoot
                }
                & $MoveOperation $Backup $InstallRoot
            } catch {
                $rollbackFailure = $_
                throw "Deployment failed: $($deploymentFailure.Exception.Message) Rollback also failed: $($rollbackFailure.Exception.Message) The previous deployment is preserved at '$Backup'. Restore it manually before retrying."
            }
            throw "Deployment failed: $($deploymentFailure.Exception.Message) The previous deployment was restored successfully."
        }

        if (Test-Path -LiteralPath $InstallRoot) {
            try {
                & $RemoveOperation $InstallRoot
            } catch {
                throw "Deployment failed: $($deploymentFailure.Exception.Message) Partial output could not be removed and remains at '$InstallRoot'."
            }
        }
        throw "Deployment failed: $($deploymentFailure.Exception.Message)"
    }

    if (Test-Path -LiteralPath $Backup) {
        try {
            & $RemoveOperation $Backup
        } catch {
            Write-Warning "The new deployment is installed, but the previous deployment backup could not be removed. It remains at '$Backup'."
        }
    }
}
