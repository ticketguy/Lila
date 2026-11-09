using System.Diagnostics;
using Lila.Desktop.Core.Api;
using Lila.Desktop.Core.Keyboard;
using Lila.Desktop.Core.Voice;

namespace Lila.Desktop.Keyboard;

/// <summary>Writes timing and state metadata only; never request bodies or responses.</summary>
public sealed class InteractionDiagnostics : IDisposable
{
    private readonly object _sync = new();
    private readonly long _origin = Stopwatch.GetTimestamp();
    private readonly StreamWriter _writer;

    public InteractionDiagnostics()
    {
        var directory = System.IO.Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
            "LilaDesktop", "diagnostics");
        Directory.CreateDirectory(directory);
        var path = System.IO.Path.Combine(directory, $"keyboard-{DateTime.Now:yyyyMMdd-HHmmss}-{Guid.NewGuid():N}.log");
        _writer = new StreamWriter(new FileStream(path, FileMode.CreateNew, FileAccess.Write,
            FileShare.Read)) { AutoFlush = true };
        DiagnosticPath = path;
        Write("diagnostic_started");
    }

    public string DiagnosticPath { get; }

    public void RecordKey(RawKeyEvent key, VoiceInteractionState state)
    {
        var timestamp = Elapsed(key.Timestamp);
        var down = key.KeyDownTimestamp is long downTicks ? Elapsed(downTicks) : "unknown";
        var held = key.HeldDuration?.TotalMilliseconds.ToString("F3", System.Globalization.CultureInfo.InvariantCulture)
                   ?? "n/a";
        Write($"key t_ms={timestamp} vk=0x{key.VirtualKey:X2} scan=0x{key.ScanCode:X2} " +
              $"make_break={(key.IsKeyUp ? "break" : "make")} repeat={key.IsRepeat.ToString().ToLowerInvariant()} " +
              $"keydown_ms={down} held_ms={held} state={state}");
    }

    public void RecordState(VoiceInteractionSnapshot snapshot) =>
        Write($"voice_state state={snapshot.State} has_error={(!string.IsNullOrEmpty(snapshot.Error)).ToString().ToLowerInvariant()}");

    public void RecordRequest(ApiRequestDiagnostic request, VoiceInteractionState state) =>
        Write($"api t_ms={Elapsed(Stopwatch.GetTimestamp())} method={request.Method} route={request.Route} " +
              $"http={(request.StatusCode?.ToString() ?? "none")} outcome={request.Outcome} " +
              $"duration_ms={request.Duration.TotalMilliseconds:F3} state={state}");

    private string Elapsed(long timestamp) =>
        Stopwatch.GetElapsedTime(_origin, timestamp).TotalMilliseconds.ToString("F3",
            System.Globalization.CultureInfo.InvariantCulture);

    private void Write(string line)
    {
        lock (_sync)
            _writer.WriteLine(line);
    }

    public void Dispose()
    {
        Write("diagnostic_stopped");
        _writer.Dispose();
    }
}
