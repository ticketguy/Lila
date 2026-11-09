using Lila.Desktop.Core.Api;
using Lila.Desktop.Core.Keyboard;

namespace Lila.Desktop.Core.Voice;

public enum VoiceInteractionState
{
    Idle,
    Starting,
    Recording,
    Processing,
    Cancelling,
    Response
}

public enum PushToTalkMode
{
    HoldToTalk,
    ToggleToTalk
}

public sealed record VoiceInteractionSnapshot(VoiceInteractionState State,
    string? SessionId = null, string? Transcript = null, string? Response = null,
    string? Error = null);

public sealed class VoiceSessionController(IVoiceSessionApi api,
    TimeSpan? missedReleaseSafetyLimit = null, TimeSpan? responseDisplayDuration = null)
{
    private readonly object _sync = new();
    private readonly TimeSpan _safetyLimit = missedReleaseSafetyLimit ?? TimeSpan.FromSeconds(115);
    private readonly TimeSpan _responseDuration = responseDisplayDuration ?? TimeSpan.FromSeconds(1.5);
    private VoiceInteractionSnapshot _snapshot = new(VoiceInteractionState.Idle);
    private bool _keyHeld;
    private bool _releaseWhenStarted;
    private bool _toggleStopWhenStarted;
    private bool _cancelWhenStarted;
    private RawKeyIdentity? _activeShortcutKey;
    private PushToTalkMode _sessionMode = PushToTalkMode.HoldToTalk;
    private CancellationTokenSource? _watchdog;

    public event Action<VoiceInteractionSnapshot>? StateChanged;

    public VoiceInteractionSnapshot Snapshot
    {
        get { lock (_sync) return _snapshot; }
    }

    public PushToTalkMode Mode { get; set; } = PushToTalkMode.HoldToTalk;

    public void Handle(InterpretedKeyEvent input)
    {
        switch (input.Kind)
        {
            case ShortcutEventKind.ShortcutDown:
                if (!input.RawEvent.IsRepeat)
                    HandleShortcutDown(input.RawEvent.Identity);
                break;
            case ShortcutEventKind.ShortcutUp:
                HandleShortcutUp(input.RawEvent.Identity);
                break;
            case ShortcutEventKind.EscapeDown:
                CancelActive();
                break;
        }
    }

    public void CancelActive()
    {
        string? sessionId = null;
        VoiceInteractionSnapshot? changed = null;
        lock (_sync)
        {
            if (_snapshot.State is VoiceInteractionState.Starting)
            {
                _cancelWhenStarted = true;
                _releaseWhenStarted = false;
                _toggleStopWhenStarted = false;
                changed = SetStateLocked(VoiceInteractionState.Cancelling,
                    error: "Cancelling as soon as the service confirms the session.");
            }
            else if (_snapshot.State is VoiceInteractionState.Recording)
            {
                sessionId = _snapshot.SessionId;
                CancelWatchdogLocked();
                changed = SetStateLocked(VoiceInteractionState.Cancelling,
                    sessionId: sessionId);
            }
        }
        Publish(changed);
        if (sessionId is not null)
            _ = CancelRequestAsync(sessionId);
    }

    public void ToggleFromUi()
    {
        string? stopSession = null;
        var start = false;
        VoiceInteractionSnapshot? changed = null;
        lock (_sync)
        {
            if (_snapshot.State is VoiceInteractionState.Idle or VoiceInteractionState.Response)
            {
                _sessionMode = PushToTalkMode.HoldToTalk;
                _keyHeld = true;
                _releaseWhenStarted = false;
                _toggleStopWhenStarted = false;
                _cancelWhenStarted = false;
                changed = SetStateLocked(VoiceInteractionState.Starting);
                start = true;
            }
            else if (_snapshot.State == VoiceInteractionState.Starting)
            {
                if (_sessionMode == PushToTalkMode.ToggleToTalk)
                    _toggleStopWhenStarted = true;
                else
                    _releaseWhenStarted = true;
            }
            else if (_snapshot.State == VoiceInteractionState.Recording)
            {
                stopSession = _snapshot.SessionId;
                CancelWatchdogLocked();
                changed = SetStateLocked(VoiceInteractionState.Processing,
                    sessionId: stopSession);
            }
        }
        Publish(changed);
        if (start)
            _ = StartRequestAsync();
        else if (stopSession is not null)
            _ = StopRequestAsync(stopSession);
    }

    public async Task CancelAndWaitAsync(TimeSpan timeout)
    {
        CancelActive();
        var deadline = DateTime.UtcNow + timeout;
        while (DateTime.UtcNow < deadline)
        {
            if (Snapshot.State is VoiceInteractionState.Idle or VoiceInteractionState.Response)
                return;
            await Task.Delay(50).ConfigureAwait(false);
        }
    }

    public void OnServiceUnavailable()
    {
        CancelActive();
        lock (_sync)
        {
            _keyHeld = false;
            if (_snapshot.State == VoiceInteractionState.Starting)
                _cancelWhenStarted = true;
        }
    }

    private void HandleShortcutDown(RawKeyIdentity identity)
    {
        string? stopSession = null;
        var start = false;
        VoiceInteractionSnapshot? changed = null;
        lock (_sync)
        {
            if (_snapshot.State == VoiceInteractionState.Idle)
            {
                _activeShortcutKey = identity;
                _sessionMode = Mode;
                _keyHeld = true;
                _releaseWhenStarted = false;
                _cancelWhenStarted = false;
                changed = SetStateLocked(VoiceInteractionState.Starting);
                start = true;
            }
            else if (_snapshot.State == VoiceInteractionState.Response)
            {
                _activeShortcutKey = identity;
                _sessionMode = Mode;
                _snapshot = new VoiceInteractionSnapshot(VoiceInteractionState.Idle);
                _keyHeld = true;
                changed = SetStateLocked(VoiceInteractionState.Starting);
                start = true;
            }
            else if (_snapshot.State == VoiceInteractionState.Recording &&
                     _sessionMode == PushToTalkMode.ToggleToTalk)
            {
                _activeShortcutKey = identity;
                _keyHeld = true;
                stopSession = _snapshot.SessionId;
                CancelWatchdogLocked();
                changed = SetStateLocked(VoiceInteractionState.Processing,
                    sessionId: stopSession);
            }
            else if (_sessionMode == PushToTalkMode.ToggleToTalk &&
                     _snapshot.State == VoiceInteractionState.Starting)
            {
                _activeShortcutKey = identity;
                _keyHeld = true;
                _toggleStopWhenStarted = true;
            }
        }
        Publish(changed);
        if (start)
            _ = StartRequestAsync();
        else if (stopSession is not null)
            _ = StopRequestAsync(stopSession);
    }

    private void HandleShortcutUp(RawKeyIdentity identity)
    {
        string? stopSession = null;
        VoiceInteractionSnapshot? changed = null;
        lock (_sync)
        {
            if (!_keyHeld || _activeShortcutKey != identity)
                return;
            _keyHeld = false;
            _activeShortcutKey = null;
            if (_sessionMode == PushToTalkMode.ToggleToTalk)
                return;

            if (_snapshot.State == VoiceInteractionState.Starting)
                _releaseWhenStarted = true;
            else if (_snapshot.State == VoiceInteractionState.Recording)
            {
                stopSession = _snapshot.SessionId;
                CancelWatchdogLocked();
                changed = SetStateLocked(VoiceInteractionState.Processing,
                    sessionId: stopSession);
            }
        }
        Publish(changed);
        if (stopSession is not null)
            _ = StopRequestAsync(stopSession);
    }

    private async Task StartRequestAsync()
    {
        VoiceStartResult started;
        try
        {
            started = await api.StartVoiceAsync().ConfigureAwait(false);
        }
        catch (Exception ex)
        {
            // A timed-out start may have succeeded remotely. Query state once;
            // never issue a second start automatically.
            try
            {
                var remote = await api.GetVoiceStatusAsync().ConfigureAwait(false);
                if (!string.IsNullOrWhiteSpace(remote.SessionId) &&
                    remote.State is "recording" or "transcribing")
                {
                    started = new VoiceStartResult(remote.State, remote.SessionId,
                        remote.ElapsedSeconds, remote.Error);
                }
                else
                {
                    SetIdleWithError("Could not confirm the voice session start. " + SafeMessage(ex));
                    return;
                }
            }
            catch (Exception statusError)
            {
                SetIdleWithError("Voice start outcome is unknown. Do not retry until status is checked. " +
                                 SafeMessage(statusError));
                return;
            }
        }

        string? cancelId = null;
        string? stopId = null;
        VoiceInteractionSnapshot? changed;
        lock (_sync)
        {
            if (string.IsNullOrWhiteSpace(started.SessionId))
            {
                changed = SetStateLocked(VoiceInteractionState.Idle,
                    error: "Lila returned no voice session ID.");
            }
            else if (_cancelWhenStarted || _snapshot.State == VoiceInteractionState.Cancelling)
            {
                cancelId = started.SessionId;
                changed = SetStateLocked(VoiceInteractionState.Cancelling,
                    sessionId: cancelId);
            }
            else if (_sessionMode == PushToTalkMode.HoldToTalk &&
                     (_releaseWhenStarted || !_keyHeld))
            {
                stopId = started.SessionId;
                changed = SetStateLocked(VoiceInteractionState.Processing,
                    sessionId: stopId);
                CancelWatchdogLocked();
            }
            else if (_sessionMode == PushToTalkMode.ToggleToTalk && _toggleStopWhenStarted)
            {
                stopId = started.SessionId;
                changed = SetStateLocked(VoiceInteractionState.Processing,
                    sessionId: stopId);
            }
            else
            {
                changed = SetStateLocked(VoiceInteractionState.Recording,
                    sessionId: started.SessionId);
                StartWatchdogLocked();
            }
        }
        Publish(changed);
        if (cancelId is not null)
            await CancelRequestAsync(cancelId).ConfigureAwait(false);
        else if (stopId is not null)
            await StopRequestAsync(stopId).ConfigureAwait(false);
    }

    private async Task StopRequestAsync(string sessionId)
    {
        try
        {
            var result = await api.StopVoiceAsync(sessionId).ConfigureAwait(false);
            VoiceInteractionSnapshot changed;
            lock (_sync)
            {
                if (_snapshot.State == VoiceInteractionState.Cancelling)
                    return;
                changed = SetStateLocked(VoiceInteractionState.Response, sessionId,
                    result.Transcript, result.Response);
            }
            Publish(changed);
            await Task.Delay(_responseDuration).ConfigureAwait(false);
            lock (_sync)
            {
                if (_snapshot.State == VoiceInteractionState.Response &&
                    _snapshot.SessionId == sessionId)
                    changed = SetStateLocked(VoiceInteractionState.Idle);
                else
                    changed = null!;
            }
            Publish(changed);
        }
        catch (Exception ex)
        {
            var prefix = ex is LilaApiException { StatusCode: not null }
                ? "Voice processing failed. "
                : "Voice stop outcome is uncertain; it was not retried. ";
            SetIdleWithError(prefix + SafeMessage(ex));
        }
    }

    private async Task CancelRequestAsync(string sessionId)
    {
        try
        {
            await api.CancelVoiceAsync(sessionId).ConfigureAwait(false);
            SetIdleWithError(null);
        }
        catch (Exception ex)
        {
            SetIdleWithError("Cancellation could not be confirmed. The service enforces its recording limit. " +
                             SafeMessage(ex));
        }
    }

    private void StartWatchdogLocked()
    {
        CancelWatchdogLocked();
        _watchdog = new CancellationTokenSource();
        var token = _watchdog.Token;
        _ = Task.Run(async () =>
        {
            try
            {
                await Task.Delay(_safetyLimit, token).ConfigureAwait(false);
                SafetyStopRecording();
            }
            catch (OperationCanceledException) { }
        });
    }

    private void SafetyStopRecording()
    {
        string? sessionId = null;
        VoiceInteractionSnapshot? changed = null;
        lock (_sync)
        {
            if (_snapshot.State == VoiceInteractionState.Recording)
            {
                _keyHeld = false;
                _activeShortcutKey = null;
                sessionId = _snapshot.SessionId;
                changed = SetStateLocked(VoiceInteractionState.Processing, sessionId,
                    error: "Recording stopped by the maximum hold safety limit.");
                CancelWatchdogLocked();
            }
        }
        Publish(changed);
        if (sessionId is not null)
            _ = StopRequestAsync(sessionId);
    }

    private void CancelWatchdogLocked()
    {
        _watchdog?.Cancel();
        _watchdog?.Dispose();
        _watchdog = null;
    }

    private VoiceInteractionSnapshot SetStateLocked(VoiceInteractionState state,
        string? sessionId = null, string? transcript = null, string? response = null,
        string? error = null)
    {
        _snapshot = new VoiceInteractionSnapshot(state, sessionId, transcript, response, error);
        return _snapshot;
    }

    private void SetIdleWithError(string? error)
    {
        VoiceInteractionSnapshot changed;
        lock (_sync)
        {
            _keyHeld = false;
            _activeShortcutKey = null;
            _releaseWhenStarted = false;
            _toggleStopWhenStarted = false;
            _cancelWhenStarted = false;
            CancelWatchdogLocked();
            changed = SetStateLocked(VoiceInteractionState.Idle, error: error);
        }
        Publish(changed);
    }

    private void Publish(VoiceInteractionSnapshot? snapshot)
    {
        if (snapshot is not null)
            StateChanged?.Invoke(snapshot);
    }

    private static string SafeMessage(Exception exception) => exception is LilaApiException apiError
        ? apiError.Message : "Check Lila's connection and service status.";
}

public interface IVoiceSessionApi
{
    Task<VoiceStartResult> StartVoiceAsync(CancellationToken cancellationToken = default);
    Task<VoiceStopResult> StopVoiceAsync(string sessionId, CancellationToken cancellationToken = default);
    Task<VoiceCancelResult> CancelVoiceAsync(string sessionId, CancellationToken cancellationToken = default);
    Task<VoiceSessionStatus> GetVoiceStatusAsync(CancellationToken cancellationToken = default);
}
