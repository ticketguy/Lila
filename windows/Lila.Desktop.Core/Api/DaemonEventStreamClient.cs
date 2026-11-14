using System.Net.Http.Headers;
using System.Runtime.CompilerServices;
using System.Text;
using System.Text.Json;

namespace Lila.Desktop.Core.Api;

/// <summary>
/// M6: Resilient cursor-tracked SSE / replay client for Lila Daemon's
/// versioned event stream (/events and /events/stream).
/// </summary>
public sealed class DaemonEventStreamClient(LilaApiClient api)
{
    private readonly object _sync = new();
    private long _lastCursor;
    private readonly Queue<long> _seenOrder = new();
    private readonly HashSet<long> _seenIds = [];
    private const int MaxDedupWindow = 1024;

    public long LastCursor
    {
        get { lock (_sync) return _lastCursor; }
    }

    public event Action<DaemonEventDto>? EventReceived;
    public event Action? GapDetected;

    public bool TryAcceptEvent(DaemonEventDto ev)
    {
        lock (_sync)
        {
            if (_seenIds.Contains(ev.EventId))
                return false;
            _seenIds.Add(ev.EventId);
            _seenOrder.Enqueue(ev.EventId);
            while (_seenOrder.Count > MaxDedupWindow)
            {
                var old = _seenOrder.Dequeue();
                _seenIds.Remove(old);
            }
            if (ev.Cursor > _lastCursor)
                _lastCursor = ev.Cursor;
        }
        EventReceived?.Invoke(ev);
        return true;
    }

    public async Task<int> PollOnceAsync(CancellationToken cancellationToken = default)
    {
        var batch = await api.PollEventsAsync(LastCursor, 100, cancellationToken).ConfigureAwait(false);
        if (batch.GapDetected)
            GapDetected?.Invoke();

        var accepted = 0;
        foreach (var ev in batch.Events)
        {
            if (TryAcceptEvent(ev))
                accepted++;
        }
        lock (_sync)
        {
            if (batch.NextCursor > _lastCursor)
                _lastCursor = batch.NextCursor;
        }
        return accepted;
    }

    public async Task<int> ConsumeSseStreamOnceAsync(CancellationToken cancellationToken = default)
    {
        var accepted = 0;
        await foreach (var ev in api.StreamEventsAsync(LastCursor, cancellationToken).ConfigureAwait(false))
        {
            if (TryAcceptEvent(ev))
                accepted++;
        }
        return accepted;
    }

    public static DaemonEventDto? ParseSseDataLine(string dataJson, JsonSerializerOptions options)
    {
        if (string.IsNullOrWhiteSpace(dataJson))
            return null;
        return JsonSerializer.Deserialize<DaemonEventDto>(dataJson, options);
    }
}
