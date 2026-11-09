using System.Diagnostics;
using System.Text;

namespace Lila.Desktop.Core.Wsl;

public sealed record CommandResult(int ExitCode, string StandardOutput, string StandardError);

public interface ICommandRunner
{
    Task<CommandResult> RunAsync(string executable, IReadOnlyList<string> arguments,
        TimeSpan timeout, CancellationToken cancellationToken = default);
}

public sealed class SystemCommandRunner : ICommandRunner
{
    private const int MaxOutputChars = 64 * 1024;

    public async Task<CommandResult> RunAsync(string executable, IReadOnlyList<string> arguments,
        TimeSpan timeout, CancellationToken cancellationToken = default)
    {
        using var process = new Process
        {
            StartInfo = new ProcessStartInfo
            {
                FileName = executable,
                UseShellExecute = false,
                CreateNoWindow = true,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                // WSL emits UTF-16 NUL-interleaved text for --list, while the
                // Linux command's output is UTF-8. UTF-8 plus NUL removal in
                // WslClient handles both without corrupting systemctl output.
                StandardOutputEncoding = Encoding.UTF8,
                StandardErrorEncoding = Encoding.UTF8
            }
        };
        foreach (var argument in arguments)
            process.StartInfo.ArgumentList.Add(argument);

        try
        {
            if (!process.Start())
                throw new InvalidOperationException("Unable to launch the requested system command.");
        }
        catch (Exception ex) when (ex is System.ComponentModel.Win32Exception or InvalidOperationException)
        {
            throw new InvalidOperationException("The WSL command could not be started.", ex);
        }

        using var timeoutSource = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
        timeoutSource.CancelAfter(timeout);
        try
        {
            var stdoutTask = ReadBoundedAsync(process.StandardOutput, timeoutSource.Token);
            var stderrTask = ReadBoundedAsync(process.StandardError, timeoutSource.Token);
            await process.WaitForExitAsync(timeoutSource.Token).ConfigureAwait(false);
            return new CommandResult(process.ExitCode,
                await stdoutTask.ConfigureAwait(false), await stderrTask.ConfigureAwait(false));
        }
        catch (OperationCanceledException)
        {
            try { process.Kill(entireProcessTree: true); } catch (InvalidOperationException) { }
            throw;
        }
    }

    private static async Task<string> ReadBoundedAsync(StreamReader reader, CancellationToken token)
    {
        var buffer = new char[4096];
        var result = new StringBuilder();
        while (result.Length < MaxOutputChars)
        {
            var count = await reader.ReadAsync(buffer.AsMemory(0,
                Math.Min(buffer.Length, MaxOutputChars - result.Length)), token).ConfigureAwait(false);
            if (count == 0)
                break;
            result.Append(buffer, 0, count);
        }
        return result.ToString();
    }
}
