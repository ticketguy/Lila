[CmdletBinding()]
param(
    [string]$Distribution = 'Ubuntu-24.04',
    [switch]$SkipDesktopPublish,
    [switch]$Launch,
    [switch]$RotateApiToken
)

$ErrorActionPreference = 'Stop'
$repoRoot = (Get-Item -LiteralPath (Join-Path $PSScriptRoot '..\..')).FullName
$wsl = Get-Command wsl.exe -ErrorAction SilentlyContinue
if (-not $wsl) { throw 'WSL is required. Install/update WSL and create the configured Linux distribution first.' }

function Get-LinuxRepositoryRoot {
    param([string]$WindowsPath, [string]$Distro)

    foreach ($prefix in @("\\wsl.localhost\$Distro\", "\\wsl$\$Distro\")) {
        if ($WindowsPath.StartsWith($prefix, [StringComparison]::OrdinalIgnoreCase)) {
            $relative = $WindowsPath.Substring($prefix.Length).Replace('\', '/')
            return '/' + $relative
        }
    }

    $normalized = $WindowsPath.Replace('\', '/')
    $converted = & $wsl.Source --distribution $Distro --exec wslpath -u $normalized
    if ($LASTEXITCODE -ne 0 -or -not $converted) {
        throw 'Could not map the repository path into the selected WSL distribution.'
    }
    return ([string]$converted).Trim()
}

function Get-Sha256Pin {
    param([Security.Cryptography.X509Certificates.X509Certificate2]$Certificate)
    $sha = [Security.Cryptography.SHA256]::Create()
    try {
        return ([BitConverter]::ToString($sha.ComputeHash($Certificate.RawData))).Replace('-', '').ToLowerInvariant()
    } finally {
        $sha.Dispose()
    }
}

function Set-LilaUserPin {
    param([string]$Name, [string]$Expected)
    $current = [Environment]::GetEnvironmentVariable($Name, 'User')
    $currentPin = ($current -replace '[^0-9A-Fa-f]', '').ToLowerInvariant()
    if (-not [string]::IsNullOrWhiteSpace($current) -and $currentPin -ne $Expected) {
        throw "Existing Windows $Name differs from the provisioned certificate; refusing to replace it."
    }
    [Environment]::SetEnvironmentVariable($Name, $Expected, 'User')
    [Environment]::SetEnvironmentVariable($Name, $Expected, 'Process')
}

if (-not $env:LOCALAPPDATA) { throw 'LOCALAPPDATA is unavailable for this user.' }
$linuxRepoRoot = Get-LinuxRepositoryRoot -WindowsPath $repoRoot -Distro $Distribution
$linuxProvision = "$linuxRepoRoot/deploy/linux/provision-mtls.sh"
$linuxActivation = "$linuxRepoRoot/deploy/linux/activate-voice-input.sh"
$linuxTokenRotation = "$linuxRepoRoot/deploy/linux/rotate-api-token.py"

& $wsl.Source --distribution $Distribution --exec bash $linuxActivation --check-only
if ($LASTEXITCODE -ne 0) { throw 'WSL setup prerequisites are incomplete; no certificates or service settings were changed.' }

if (-not $SkipDesktopPublish) {
    $activation = Join-Path $PSScriptRoot 'Activate-Lila.ps1'
    & $activation -Distribution $Distribution
    if (-not $?) { throw 'Windows desktop deployment failed.' }
}

$manifestOutput = & $wsl.Source --distribution $Distribution --exec bash $linuxProvision
if ($LASTEXITCODE -ne 0) { throw 'WSL mTLS provisioning failed; the service was not restarted.' }
try {
    $manifest = ([string]($manifestOutput | Select-Object -Last 1)) | ConvertFrom-Json
} catch {
    throw 'WSL mTLS setup returned an invalid result; the service was not restarted.'
}
foreach ($property in @('client_pfx', 'client_sha256', 'server_sha256', 'service_env')) {
    if (-not $manifest.$property) { throw "WSL mTLS setup omitted '$property'; the service was not restarted." }
}
if ($manifest.client_sha256 -notmatch '^[0-9a-f]{64}$' -or
    $manifest.server_sha256 -notmatch '^[0-9a-f]{64}$') {
    throw 'WSL mTLS setup returned an invalid certificate fingerprint.'
}

$clientPin = [string]$manifest.client_sha256
$serverPin = [string]$manifest.server_sha256
$pfxLinuxPath = [string]$manifest.client_pfx
$pfxWindowsPath = $null
try {
    $pfxWindowsPath = & $wsl.Source --distribution $Distribution --exec wslpath -w $pfxLinuxPath
    if ($LASTEXITCODE -ne 0 -or -not $pfxWindowsPath) {
        throw 'Could not map the temporary Island certificate into Windows.'
    }
    $pfxWindowsPath = ([string]$pfxWindowsPath).Trim()

    $storedCertificate = Get-ChildItem Cert:\CurrentUser\My |
        Where-Object { (Get-Sha256Pin $_) -eq $clientPin } |
        Select-Object -First 1
    if ($storedCertificate -and -not $storedCertificate.HasPrivateKey) {
        throw 'The matching Island certificate exists in Windows without its private key; refusing replacement.'
    }
    if (-not $storedCertificate) {
        Import-PfxCertificate -FilePath $pfxWindowsPath `
            -CertStoreLocation Cert:\CurrentUser\My `
            -Exportable:$false | Out-Null
        $storedCertificate = Get-ChildItem Cert:\CurrentUser\My |
            Where-Object { (Get-Sha256Pin $_) -eq $clientPin } |
            Select-Object -First 1
    }
    if (-not $storedCertificate -or -not $storedCertificate.HasPrivateKey) {
        throw 'Windows did not install the Island client certificate with its private key.'
    }

    Set-LilaUserPin -Name 'LILA_TLS_CLIENT_CERT_SHA256' -Expected $clientPin
    Set-LilaUserPin -Name 'LILA_TLS_SERVER_CERT_SHA256' -Expected $serverPin

    if ($RotateApiToken) {
        & $wsl.Source --distribution $Distribution --exec python3 $linuxTokenRotation
        if ($LASTEXITCODE -ne 0) { throw 'Lila API credential rotation failed.' }
    }

    $tokenReader = "$linuxRepoRoot/deploy/linux/read-service-token.py"
    $apiToken = & $wsl.Source --distribution $Distribution --exec python3 $tokenReader
    if ($LASTEXITCODE -ne 0 -or -not $apiToken) {
        throw 'Could not securely read the configured Lila API credential from WSL.'
    }

    if (-not ('LilaSetup.NativeCredentialStore' -as [type])) {
        Add-Type -TypeDefinition @'
using System;
using System.Runtime.InteropServices;
using System.Text;

namespace LilaSetup
{
    public static class NativeCredentialStore
    {
        private const uint GenericCredential = 1;
        private const uint PersistLocalMachine = 2;
        private const string Target = "Lila.Desktop/127.0.0.1:8765";

        [StructLayout(LayoutKind.Sequential, CharSet = CharSet.Unicode)]
        private struct NativeCredential
        {
            public uint Flags;
            public uint Type;
            public string TargetName;
            public string Comment;
            public System.Runtime.InteropServices.ComTypes.FILETIME LastWritten;
            public uint CredentialBlobSize;
            public IntPtr CredentialBlob;
            public uint Persist;
            public uint AttributeCount;
            public IntPtr Attributes;
            public string TargetAlias;
            public string UserName;
        }

        [DllImport("Advapi32.dll", EntryPoint = "CredWriteW", CharSet = CharSet.Unicode, SetLastError = true)]
        [return: MarshalAs(UnmanagedType.Bool)]
        private static extern bool CredWrite(ref NativeCredential credential, uint flags);

        [DllImport("user32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
        private static extern IntPtr SendMessageTimeout(
            IntPtr window, uint message, UIntPtr wParam, string lParam,
            uint flags, uint timeout, out UIntPtr result);

        public static void WriteToken(string token)
        {
            if (String.IsNullOrWhiteSpace(token))
                throw new ArgumentException("The Lila API credential is empty.");
            byte[] bytes = Encoding.UTF8.GetBytes(token.Trim());
            if (bytes.Length > 5120)
                throw new ArgumentOutOfRangeException("token", "The Lila API credential is too long.");
            GCHandle handle = GCHandle.Alloc(bytes, GCHandleType.Pinned);
            try
            {
                NativeCredential credential = new NativeCredential();
                credential.Type = GenericCredential;
                credential.TargetName = Target;
                credential.CredentialBlobSize = (uint)bytes.Length;
                credential.CredentialBlob = handle.AddrOfPinnedObject();
                credential.Persist = PersistLocalMachine;
                credential.UserName = Environment.UserName;
                if (!CredWrite(ref credential, 0))
                    throw new InvalidOperationException("Windows Credential Manager write failed (" +
                        Marshal.GetLastWin32Error().ToString() + ").");
            }
            finally
            {
                handle.Free();
                Array.Clear(bytes, 0, bytes.Length);
            }
        }

        public static void BroadcastEnvironmentChange()
        {
            UIntPtr result;
            SendMessageTimeout(new IntPtr(0xffff), 0x001A, UIntPtr.Zero,
                "Environment", 0x0002, 5000, out result);
        }
    }
}
'@
    }
    [LilaSetup.NativeCredentialStore]::BroadcastEnvironmentChange()
    [LilaSetup.NativeCredentialStore]::WriteToken([string]$apiToken)
    $apiToken = $null

    & $wsl.Source --distribution $Distribution --exec bash $linuxActivation --adopt-existing
    if ($LASTEXITCODE -ne 0) {
        throw 'Lila service activation failed. The TLS configuration is installed; inspect the WSL setup prerequisites and retry.'
    }
} finally {
    & $wsl.Source --distribution $Distribution --exec rm -- $pfxLinuxPath 2>$null | Out-Null
}

if ($Launch) {
    $executable = Join-Path $env:LOCALAPPDATA 'LilaDesktop\VoiceInput\app\Lila.Desktop.exe'
    if (-not (Test-Path -LiteralPath $executable -PathType Leaf)) {
        throw 'The Lila desktop executable is not installed; rerun without -SkipDesktopPublish.'
    }
    Start-Process -FilePath $executable
}

Write-Output 'Lila setup completed. The per-user mTLS client, certificate pins, API credential, and WSL service are configured.'
if ($RotateApiToken) { Write-Output 'The API credential was rotated and synchronized to Windows Credential Manager.' }
Write-Output 'Windows startup remains opt-in; use Lila Settings if you want it to start at sign-in.'
