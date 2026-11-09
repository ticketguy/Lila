using System.Diagnostics;
using Lila.Desktop.Core.Wsl;

namespace Lila.Desktop.Wsl;

/// <summary>
/// Owns one WSL client process whose shell waits for a shutdown line on stdin.
/// The Windows Process handle is monitored and stdin EOF is the graceful stop.
/// </summary>
public sealed class ManagedWslKeepAlive : IWslKeepAlive
{
    private readonly SemaphoreSlim _gate = new(1, 1);
    private Process? _process;

    public bool IsRunning
    {
        get
        {
            try { return _process is { HasExited: false }; }
            catch (InvalidOperationException) { return false; }
        }
    }

    public async Task StartAsync(string distribution, CancellationToken cancellationToken = default)
    {
        await _gate.WaitAsync(cancellationToken).ConfigureAwait(false);
        try
        {
            if (IsRunning)
                return;
            _process?.Dispose();
            var start = new ProcessStartInfo
            {
                FileName = "wsl.exe",
                UseShellExecute = false,
                CreateNoWindow = true,
                RedirectStandardInput = true,
                RedirectStandardOutput = true,
                RedirectStandardError = true
            };
            foreach (var argument in new[]
                     {
                         "--distribution", distribution, "--exec", "/bin/sh", "-c",
                         "IFS= read -r _ || :"
                     })
                start.ArgumentList.Add(argument);

            var process = new Process { StartInfo = start, EnableRaisingEvents = true };
            try
            {
                if (!process.Start())
                    throw new InvalidOperationException("The managed WSL process did not start.");
                _ = process.StandardOutput.ReadToEndAsync(cancellationToken);
                _ = process.StandardError.ReadToEndAsync(cancellationToken);
                _process = process;
                await Task.Delay(150, cancellationToken).ConfigureAwait(false);
                if (process.HasExited)
                {
                    _process = null;
                    var code = process.ExitCode;
                    process.Dispose();
                    throw new InvalidOperationException(
                        $"The managed WSL process exited during startup ({code}).");
                }
            }
            catch
            {
                try
                {
                    if (!process.HasExited)
                        process.Kill(entireProcessTree: true);
                }
                catch (InvalidOperationException) { }
                process.Dispose();
                throw;
            }
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
            var process = _process;
            _process = null;
            if (process is null)
                return;
            try
            {
                if (!process.HasExited)
                {
                    await process.StandardInput.WriteLineAsync("stop".AsMemory(), cancellationToken)
                        .ConfigureAwait(false);
                    await process.StandardInput.FlushAsync(cancellationToken).ConfigureAwait(false);
                    process.StandardInput.Close();
                    using var timeout = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
                    timeout.CancelAfter(TimeSpan.FromSeconds(8));
                    await process.WaitForExitAsync(timeout.Token).ConfigureAwait(false);
                }
            }
            catch (OperationCanceledException)
            {
                try { process.Kill(entireProcessTree: true); }
                catch (InvalidOperationException) { }
                await process.WaitForExitAsync(CancellationToken.None).ConfigureAwait(false);
                if (cancellationToken.IsCancellationRequested)
                    throw;
            }
            finally
            {
                process.Dispose();
            }
        }
        finally
        {
            _gate.Release();
        }
    }

    public async ValueTask DisposeAsync()
    {
        await StopAsync().ConfigureAwait(false);
        _gate.Dispose();
    }
}
