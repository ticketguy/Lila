using Lila.Desktop.Core.Api;
using Lila.Desktop.Core.Voice;

namespace Lila.Desktop.Core.Keyboard;

public enum VoiceShortcutSource
{
    HoldToTalk,
    FnPAndPauseToggle,
    StartButton
}

/// <summary>
/// Coordinates two independent global bindings. Right Ctrl is armed only while
/// physically held alone; Pause toggles on key-down and ignores its key-up.
/// Unrelated key identities are never retained or exposed by this class.
/// </summary>
public sealed class GlobalVoiceShortcutController : IDisposable
{
    private readonly object _sync = new();
    private readonly VoiceSessionController _voice;
    private readonly TimeSpan _armingDelay;
    private readonly TimeSpan _toggleDebounce;
    private readonly ShortcutBinding _holdBinding;
    private readonly ShortcutBinding _toggleBinding;
    private readonly bool _holdEnabled;
    private readonly bool _toggleEnabled;
    private VoiceShortcutSource? _activeSource;
    private RawKeyIdentity? _rightControlIdentity;
    private InterpretedKeyEvent? _rightControlDown;
    private CancellationTokenSource? _arming;
    private long _lastToggleTimestamp;
    private bool _armingSuppressed;
    private bool _rightControlStarted;
    private bool _muted;
    private bool _ready;
    private bool _disposed;

    public GlobalVoiceShortcutController(VoiceSessionController voice,
        ushort holdVirtualKey = KeyEventInterpreter.RightControlVirtualKey,
        ushort toggleVirtualKey = KeyEventInterpreter.PauseVirtualKey,
        bool holdEnabled = true, bool toggleEnabled = true,
        TimeSpan? armingDelay = null, TimeSpan? toggleDebounce = null)
        : this(voice, new ShortcutBinding(holdVirtualKey), new ShortcutBinding(toggleVirtualKey),
            holdEnabled, toggleEnabled, armingDelay, toggleDebounce)
    {
    }

    public GlobalVoiceShortcutController(VoiceSessionController voice,
        ShortcutBinding holdBinding, ShortcutBinding toggleBinding,
        bool holdEnabled = true, bool toggleEnabled = true,
        TimeSpan? armingDelay = null, TimeSpan? toggleDebounce = null)
    {
        _voice = voice;
        if (holdEnabled)
            holdBinding.Validate(ShortcutActionMode.Hold);
        if (toggleEnabled)
            toggleBinding.Validate(ShortcutActionMode.Toggle);
        if (holdEnabled && toggleEnabled && ShortcutBinding.Conflicts(holdBinding, toggleBinding))
            throw new ArgumentException("Enabled hold and toggle shortcuts must be different.");
        _holdBinding = holdBinding;
        _toggleBinding = toggleBinding;
        _holdEnabled = holdEnabled;
        _toggleEnabled = toggleEnabled;
        _armingDelay = armingDelay ?? TimeSpan.FromMilliseconds(180);
        _toggleDebounce = toggleDebounce ?? TimeSpan.FromMilliseconds(220);
        _voice.StateChanged += OnVoiceStateChanged;
    }

    public VoiceShortcutSource? ActiveSource
    {
        get { lock (_sync) return _activeSource; }
        private set => _activeSource = value;
    }
    public TimeSpan ArmingDelay => _armingDelay;
    public event Action? MutedShortcutAttempted;
    public bool IsMuted
    {
        get { lock (_sync) return _muted; }
    }

    public void SetMuted(bool muted)
    {
        lock (_sync)
        {
            if (_muted == muted)
                return;
            _muted = muted;
            if (muted)
                ResetArmingLocked();
        }
        if (muted)
            _voice.CancelActive();
    }

    public void SetReady(bool ready)
    {
        lock (_sync)
        {
            _ready = ready;
            if (!ready)
                ResetArmingLocked();
        }
        if (!ready)
            _voice.OnServiceUnavailable();
    }

    public void ApplyRuntimeState(bool serviceReady, VoiceSessionStatus? remoteStatus)
    {
        _voice.ObserveRemoteStatus(remoteStatus);
        SetReady(serviceReady && !_voice.IsRemoteVoiceDisabled);
    }

    public void Handle(InterpretedKeyEvent input)
    {
        lock (_sync)
        {
            if (_disposed)
                return;
            if (input.Kind == ShortcutEventKind.EscapeDown)
            {
                ResetArmingLocked();
                _voice.CancelActive();
                return;
            }
            if (!_ready || _muted)
            {
                if (_muted && input.Kind == ShortcutEventKind.ShortcutDown && !input.RawEvent.IsRepeat)
                {
                    var pressedKey = input.RawEvent.VirtualKey;
                    if ((_holdEnabled && pressedKey == _holdBinding.VirtualKey) ||
                        (_toggleEnabled && pressedKey == _toggleBinding.VirtualKey))
                    {
                        Task.Run(() => MutedShortcutAttempted?.Invoke());
                    }
                }
                return;
            }

            var key = input.RawEvent.VirtualKey;
            if (_holdEnabled && key == _holdBinding.VirtualKey)
            {
                HandleRightControlLocked(input);
                return;
            }
            if (_toggleEnabled && key == _toggleBinding.VirtualKey)
                HandlePauseLocked(input);
        }
    }

    /// <summary>Signals only that a non-modifier went down; the key is not retained.</summary>
    public void NotifyNonModifierKeyDown(ushort virtualKey)
    {
        lock (_sync)
        {
            if (_rightControlIdentity is null || _rightControlStarted || virtualKey == _holdBinding.VirtualKey)
                return;
            _armingSuppressed = true;
            _arming?.Cancel();
        }
    }

    public void NotifyDeviceRemoved(nint deviceHandle)
    {
        InterpretedKeyEvent? release = null;
        lock (_sync)
        {
            if (_rightControlIdentity is not { } identity || identity.DeviceHandle != deviceHandle)
                return;
            if (_rightControlStarted && _rightControlDown is not null)
                release = MakeRelease(_rightControlDown.RawEvent);
            ResetArmingLocked();
        }
        if (release is not null)
            _voice.Handle(release);
    }

    public void ToggleFromStartButton()
    {
        lock (_sync)
        {
            if (!_ready || _muted || _disposed)
                return;
            if (ActiveSource is null)
            {
                if (!_voice.CanStartNewSession)
                    return;
                ActiveSource = VoiceShortcutSource.StartButton;
            }
        }
        _voice.ToggleFromUi();
    }

    public void Reset()
    {
        lock (_sync)
            ResetArmingLocked();
    }

    private void HandleRightControlLocked(InterpretedKeyEvent input)
    {
        if (input.Kind == ShortcutEventKind.ShortcutDown)
        {
            if (input.RawEvent.IsRepeat || _rightControlIdentity is not null || ActiveSource is not null ||
                !_voice.CanStartNewSession)
                return;
            _rightControlIdentity = input.RawEvent.Identity;
            _rightControlDown = input;
            _armingSuppressed = false;
            _rightControlStarted = false;
            _arming = new CancellationTokenSource();
            _ = ArmRightControlAsync(input, _arming.Token);
            return;
        }

        if (input.Kind != ShortcutEventKind.ShortcutUp ||
            _rightControlIdentity != input.RawEvent.Identity)
            return;

        var shouldRelease = _rightControlStarted;
        ResetArmingLocked();
        if (shouldRelease && ActiveSource == VoiceShortcutSource.HoldToTalk)
            _voice.Handle(input);
    }

    private void HandlePauseLocked(InterpretedKeyEvent input)
    {
        if (input.Kind == ShortcutEventKind.ShortcutUp)
        {
            if (ActiveSource == VoiceShortcutSource.FnPAndPauseToggle)
                _voice.Handle(input); // Release clears the physical identity; it never ends toggle mode.
            return;
        }
        if (input.Kind != ShortcutEventKind.ShortcutDown || input.RawEvent.IsRepeat)
            return;

        var now = input.RawEvent.Timestamp == 0
            ? System.Diagnostics.Stopwatch.GetTimestamp() : input.RawEvent.Timestamp;
        if (_lastToggleTimestamp != 0 &&
            System.Diagnostics.Stopwatch.GetElapsedTime(_lastToggleTimestamp, now) < _toggleDebounce)
            return;
        _lastToggleTimestamp = now;

        if (ActiveSource is null)
        {
            if (!_voice.CanStartNewSession)
                return;
            ActiveSource = VoiceShortcutSource.FnPAndPauseToggle;
            _voice.Mode = PushToTalkMode.ToggleToTalk;
            _voice.Handle(input);
        }
        else if (ActiveSource == VoiceShortcutSource.FnPAndPauseToggle)
        {
            _voice.Mode = PushToTalkMode.ToggleToTalk;
            _voice.Handle(input);
        }
    }

    private async Task ArmRightControlAsync(InterpretedKeyEvent down, CancellationToken token)
    {
        try
        {
            await Task.Delay(_armingDelay, token).ConfigureAwait(false);
            lock (_sync)
            {
                if (_disposed || !_ready || token.IsCancellationRequested ||
                    _armingSuppressed || _rightControlIdentity != down.RawEvent.Identity ||
                    ActiveSource is not null || !_voice.CanStartNewSession)
                    return;
                _rightControlStarted = true;
                ActiveSource = VoiceShortcutSource.HoldToTalk;
                _voice.Mode = PushToTalkMode.HoldToTalk;
                _voice.Handle(down);
            }
        }
        catch (OperationCanceledException) { }
    }

    private void OnVoiceStateChanged(VoiceInteractionSnapshot state)
    {
        if (state.State is not (VoiceInteractionState.Idle or VoiceInteractionState.Response))
            return;
        lock (_sync)
        {
            if (state.State == VoiceInteractionState.Idle || ActiveSource is not null)
                ActiveSource = null;
            if (!_rightControlStarted)
                ResetArmingLocked();
        }
    }

    private void ResetArmingLocked()
    {
        _arming?.Cancel();
        _arming?.Dispose();
        _arming = null;
        _rightControlIdentity = null;
        _rightControlDown = null;
        _armingSuppressed = false;
        _rightControlStarted = false;
    }

    private static InterpretedKeyEvent MakeRelease(RawKeyEvent down) =>
        new(ShortcutEventKind.ShortcutUp, down with
        {
            IsKeyUp = true,
            IsRepeat = false,
            Timestamp = System.Diagnostics.Stopwatch.GetTimestamp()
        });

    public void Dispose()
    {
        lock (_sync)
        {
            if (_disposed)
                return;
            _disposed = true;
            ResetArmingLocked();
        }
        _voice.StateChanged -= OnVoiceStateChanged;
    }
}
