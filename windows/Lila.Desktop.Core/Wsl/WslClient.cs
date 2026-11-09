using Lila.Desktop.Core.Api;

namespace Lila.Desktop.Core.Wsl;

public enum LilaAvailability
{
    WslUnavailable,
    ServiceInactive,
    AuthenticationFailure,
    ApiConnectionFailure,
    Starting,
    Ready
}

public sealed record AvailabilityResult(LilaAvailability State, string Message,
    LilaHealth? Health = null);

public sealed class WslClient(ICommandRunner commands, LilaApiClient api,
    string distribution = "Ubuntu-24.04")
{
    private const string WslExe = "wsl.exe";
    private static readonly TimeSpan CommandTimeout = TimeSpan.FromSeconds(12);

    public string Distribution { get; } = distribution;

    public async Task<AvailabilityResult> CheckAsync(CancellationToken cancellationToken = default)
    {
        CommandResult running;
        try
        {
            running = await commands.RunAsync(WslExe, ["--list", "--running", "--quiet"],
                CommandTimeout, cancellationToken).ConfigureAwait(false);
        }
        catch (Exception ex) when (ex is InvalidOperationException or OperationCanceledException)
        {
            return new AvailabilityResult(LilaAvailability.WslUnavailable,
                "WSL could not be queried.");
        }

        if (running.ExitCode != 0 || !ContainsDistribution(running.StandardOutput, Distribution))
            return new AvailabilityResult(LilaAvailability.WslUnavailable,
                $"WSL distribution {Distribution} is not running.");

        CommandResult service;
        try
        {
            service = await commands.RunAsync(WslExe,
                ["--distribution", Distribution, "--exec", "systemctl", "--user", "is-active", "lila.service"],
                CommandTimeout, cancellationToken).ConfigureAwait(false);
        }
        catch (Exception ex) when (ex is InvalidOperationException or OperationCanceledException)
        {
            return new AvailabilityResult(LilaAvailability.ServiceInactive,
                "WSL is running, but the Lila user service could not be queried.");
        }

        if (service.ExitCode != 0 || !service.StandardOutput.Trim().Equals("active", StringComparison.OrdinalIgnoreCase))
            return new AvailabilityResult(LilaAvailability.ServiceInactive,
                "WSL is running; the Lila service is inactive.");

        try
        {
            var health = await api.GetHealthAsync(cancellationToken).ConfigureAwait(false);
            if (!health.Ready)
                return new AvailabilityResult(LilaAvailability.Starting,
                    "Lila is loading or not ready.", health);
            return new AvailabilityResult(LilaAvailability.Ready, "Lila is ready.", health);
        }
        catch (LilaApiException ex) when (ex.Kind == LilaApiFailureKind.Authentication)
        {
            return new AvailabilityResult(LilaAvailability.AuthenticationFailure,
                "Lila rejected the saved credential.");
        }
        catch (LilaApiException)
        {
            return new AvailabilityResult(LilaAvailability.ApiConnectionFailure,
                "The Lila API could not be reached.");
        }
    }

    public async Task<AvailabilityResult> StartLilaAsync(CancellationToken cancellationToken = default)
    {
        CommandResult manager;
        try
        {
            manager = await commands.RunAsync(WslExe,
                ["--distribution", Distribution, "--exec", "systemctl", "--user", "show-environment"],
                CommandTimeout, cancellationToken).ConfigureAwait(false);
        }
        catch (Exception ex) when (ex is InvalidOperationException or OperationCanceledException)
        {
            return new AvailabilityResult(LilaAvailability.ServiceInactive,
                "WSL is running, but the systemd user manager could not be reached.");
        }
        if (manager.ExitCode != 0)
            return new AvailabilityResult(LilaAvailability.ServiceInactive,
                "WSL is running, but the systemd user manager is unavailable for this Linux user session.");

        CommandResult start;
        try
        {
            start = await commands.RunAsync(WslExe,
                ["--distribution", Distribution, "--exec", "systemctl", "--user", "start", "lila.service"],
                TimeSpan.FromSeconds(90), cancellationToken).ConfigureAwait(false);
        }
        catch (Exception ex) when (ex is InvalidOperationException or OperationCanceledException)
        {
            return new AvailabilityResult(LilaAvailability.WslUnavailable,
                "Could not start Lila through the WSL user service manager.");
        }

        if (start.ExitCode != 0)
            return new AvailabilityResult(LilaAvailability.ServiceInactive,
                "WSL started, but systemd could not start Lila. Check the user service and manager.");

        using var pollTimeout = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
        pollTimeout.CancelAfter(TimeSpan.FromSeconds(90));
        while (!pollTimeout.IsCancellationRequested)
        {
            var status = await CheckAsync(pollTimeout.Token).ConfigureAwait(false);
            if (status.State is LilaAvailability.Ready or LilaAvailability.AuthenticationFailure)
                return status;
            try
            {
                await Task.Delay(TimeSpan.FromSeconds(1), pollTimeout.Token).ConfigureAwait(false);
            }
            catch (OperationCanceledException) when (!cancellationToken.IsCancellationRequested)
            {
                break;
            }
        }
        return new AvailabilityResult(LilaAvailability.Starting,
            "Lila started but did not become ready before the 90-second timeout.");
    }

    public async Task<AvailabilityResult> StopLilaAsync(CancellationToken cancellationToken = default)
    {
        try
        {
            var result = await commands.RunAsync(WslExe,
                ["--distribution", Distribution, "--exec", "systemctl", "--user", "stop", "lila.service"],
                TimeSpan.FromSeconds(20), cancellationToken).ConfigureAwait(false);
            if (result.ExitCode != 0)
                return new AvailabilityResult(LilaAvailability.ServiceInactive,
                    "systemd could not stop Lila.");
            return new AvailabilityResult(LilaAvailability.ServiceInactive, "Lila stopped.");
        }
        catch (Exception ex) when (ex is InvalidOperationException or OperationCanceledException)
        {
            return new AvailabilityResult(LilaAvailability.WslUnavailable,
                "Could not reach the WSL service manager to stop Lila.");
        }
    }

    private static bool ContainsDistribution(string output, string distribution)
    {
        var entries = output.Replace("\0", string.Empty, StringComparison.Ordinal)
            .Replace("\uFEFF", string.Empty, StringComparison.Ordinal)
            .Split(['\r', '\n'], StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
        return entries.Any(entry => entry.Equals(distribution, StringComparison.OrdinalIgnoreCase));
    }
}
