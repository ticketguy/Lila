namespace Lila.Desktop.Core.Wsl;

public interface IWslKeepAlive : IAsyncDisposable
{
    bool IsRunning { get; }
    Task StartAsync(string distribution, CancellationToken cancellationToken = default);
    Task StopAsync(CancellationToken cancellationToken = default);
}

/// <summary>Starts the selected distro and supervises the tray-owned keepalive.</summary>
public sealed class WslLifecycleSupervisor(ICommandRunner commands,
    IWslKeepAlive keepAlive, string distribution = "Ubuntu-24.04") : IAsyncDisposable
{
    private readonly SemaphoreSlim _gate = new(1, 1);
    private const string WslExe = "wsl.exe";

    public string Distribution { get; } = distribution;
    public bool KeepAliveRunning => keepAlive.IsRunning;

    public async Task EnsureRunningAsync(CancellationToken cancellationToken = default)
    {
        await _gate.WaitAsync(cancellationToken).ConfigureAwait(false);
        try
        {
            var running = await commands.RunAsync(WslExe,
                ["--list", "--running", "--quiet"], TimeSpan.FromSeconds(12), cancellationToken)
                .ConfigureAwait(false);
            if (running.ExitCode != 0)
                throw new InvalidOperationException("WSL could not list running distributions.");

            if (!ContainsDistribution(running.StandardOutput, Distribution))
            {
                var start = await commands.RunAsync(WslExe,
                    ["--distribution", Distribution, "--exec", "/bin/true"],
                    TimeSpan.FromSeconds(35), cancellationToken).ConfigureAwait(false);
                if (start.ExitCode != 0)
                    throw new InvalidOperationException($"Could not start WSL distribution {Distribution}.");
            }

            if (!keepAlive.IsRunning)
                await keepAlive.StartAsync(Distribution, cancellationToken).ConfigureAwait(false);
            if (!keepAlive.IsRunning)
                throw new InvalidOperationException("The managed WSL keepalive exited during startup.");
        }
        finally
        {
            _gate.Release();
        }
    }

    public async Task StopAsync(CancellationToken cancellationToken = default)
    {
        await _gate.WaitAsync(cancellationToken).ConfigureAwait(false);
        try
        {
            await keepAlive.StopAsync(cancellationToken).ConfigureAwait(false);
        }
        finally
        {
            _gate.Release();
        }
    }

    private static bool ContainsDistribution(string output, string distribution) =>
        output.Replace("\0", string.Empty, StringComparison.Ordinal)
            .Replace("\uFEFF", string.Empty, StringComparison.Ordinal)
            .Split(['\r', '\n'], StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)
            .Any(entry => entry.Equals(distribution, StringComparison.OrdinalIgnoreCase));

    public async ValueTask DisposeAsync()
    {
        await StopAsync().ConfigureAwait(false);
        await keepAlive.DisposeAsync().ConfigureAwait(false);
        _gate.Dispose();
    }
}
