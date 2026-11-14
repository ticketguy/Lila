using System.Diagnostics;
using System.Text.Json;

namespace Lila.Desktop.Core.Sentinel;

public sealed record WindowsTelemetryFact(
    string Source,
    string Metric,
    object Value,
    bool ReadOnly = true);

public sealed record WindowsCoverageGap(
    string Source,
    string Reason,
    string PermissionRequired);

public sealed record WindowsSentinelSnapshot(
    IReadOnlyList<WindowsTelemetryFact> Facts,
    IReadOnlyList<WindowsCoverageGap> Gaps,
    string OsVersion,
    bool AutonomousRemediationAttempted = false,
    bool ScreenCaptureUsed = false);

/// <summary>
/// M8: Opt-in, strictly read-only Windows host telemetry collector (Defender &amp; Firewall).
/// Never performs remediation, actuation, or screen capture.
/// </summary>
public sealed class WindowsSentinelCollector
{
    public delegate Task<(int ExitCode, string StdOut, string StdErr)> CommandRunner(
        string fileName, string arguments, CancellationToken ct);

    private readonly CommandRunner _runner;

    public WindowsSentinelCollector(CommandRunner? runner = null)
    {
        _runner = runner ?? RunReadOnlyProcessAsync;
    }

    public async Task<WindowsSentinelSnapshot> CollectAsync(
        bool optedIn,
        CancellationToken cancellationToken = default)
    {
        var osVersion = Environment.OSVersion.VersionString;
        if (!optedIn)
        {
            return new WindowsSentinelSnapshot(
                [],
                [
                    new WindowsCoverageGap("defender", "Sentinel Windows host collection is not opted in.", "owner_opt_in"),
                    new WindowsCoverageGap("firewall", "Sentinel Windows host collection is not opted in.", "owner_opt_in")
                ],
                osVersion);
        }

        var facts = new List<WindowsTelemetryFact>();
        var gaps = new List<WindowsCoverageGap>();

        // Read-only Defender status query
        const string psDefender = "try { $s = Get-MpComputerStatus -ErrorAction Stop; @{ realtime = [bool]$s.RealTimeProtectionEnabled; antivirus = [bool]$s.AntivirusEnabled } | ConvertTo-Json -Compress } catch { exit 2 }";
        var (defExit, defOut, _) = await _runner("powershell.exe",
            $"-NoProfile -NonInteractive -Command \"{psDefender}\"", cancellationToken).ConfigureAwait(false);
        if (defExit == 0 && !string.IsNullOrWhiteSpace(defOut))
        {
            try
            {
                using var doc = JsonDocument.Parse(defOut);
                if (doc.RootElement.TryGetProperty("realtime", out var rt))
                    facts.Add(new WindowsTelemetryFact("defender", "defender_realtime_enabled", rt.GetBoolean()));
                if (doc.RootElement.TryGetProperty("antivirus", out var av))
                    facts.Add(new WindowsTelemetryFact("defender", "defender_antivirus_enabled", av.GetBoolean()));
            }
            catch (JsonException)
            {
                gaps.Add(new WindowsCoverageGap("defender", "Malformed Defender status payload.", "read_only_defender_status"));
            }
        }
        else
        {
            gaps.Add(new WindowsCoverageGap("defender", "Unable to query Windows Defender status read-only.", "read_only_defender_status"));
        }

        // Read-only Firewall profile status query
        const string psFirewall = "try { $p = Get-NetFirewallProfile -ErrorAction Stop | ForEach-Object { @{ name = $_.Name; enabled = [bool]$_.Enabled } }; $p | ConvertTo-Json -Compress } catch { exit 2 }";
        var (fwExit, fwOut, _) = await _runner("powershell.exe",
            $"-NoProfile -NonInteractive -Command \"{psFirewall}\"", cancellationToken).ConfigureAwait(false);
        if (fwExit == 0 && !string.IsNullOrWhiteSpace(fwOut))
        {
            try
            {
                using var doc = JsonDocument.Parse(fwOut);
                if (doc.RootElement.ValueKind == JsonValueKind.Array)
                {
                    var allEnabled = doc.RootElement.EnumerateArray()
                        .All(el => el.TryGetProperty("enabled", out var en) && en.GetBoolean());
                    facts.Add(new WindowsTelemetryFact("firewall", "firewall_enabled", allEnabled));
                }
                else if (doc.RootElement.ValueKind == JsonValueKind.Object &&
                         doc.RootElement.TryGetProperty("enabled", out var en))
                {
                    facts.Add(new WindowsTelemetryFact("firewall", "firewall_enabled", en.GetBoolean()));
                }
            }
            catch (JsonException)
            {
                gaps.Add(new WindowsCoverageGap("firewall", "Malformed Firewall profile payload.", "read_only_firewall_status"));
            }
        }
        else
        {
            gaps.Add(new WindowsCoverageGap("firewall", "Unable to query Windows Firewall profiles read-only.", "read_only_firewall_status"));
        }

        return new WindowsSentinelSnapshot(facts, gaps, osVersion);
    }

    private static async Task<(int ExitCode, string StdOut, string StdErr)> RunReadOnlyProcessAsync(
        string fileName, string arguments, CancellationToken ct)
    {
        using var proc = new Process
        {
            StartInfo = new ProcessStartInfo
            {
                FileName = fileName,
                Arguments = arguments,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                UseShellExecute = false,
                CreateNoWindow = true,
            }
        };
        proc.Start();
        var stdout = await proc.StandardOutput.ReadToEndAsync(ct).ConfigureAwait(false);
        var stderr = await proc.StandardError.ReadToEndAsync(ct).ConfigureAwait(false);
        await proc.WaitForExitAsync(ct).ConfigureAwait(false);
        return (proc.ExitCode, stdout.Trim(), stderr.Trim());
    }
}
