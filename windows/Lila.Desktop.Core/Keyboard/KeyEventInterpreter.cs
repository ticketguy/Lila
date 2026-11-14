namespace Lila.Desktop.Core.Keyboard;

/// <summary>
/// Tracks only configured primary keys, Escape and the modifier state needed to
/// recognize configured chords. Other keys are discarded by the caller.
/// </summary>
public sealed class KeyEventInterpreter(ushort shortcutVirtualKey)
{
    public const ushort EscapeVirtualKey = 0x1B;
    public const ushort PauseVirtualKey = 0x13;
    public const ushort RightControlVirtualKey = 0xA3;

    private readonly Dictionary<RawKeyIdentity, long> _down = [];
    private readonly HashSet<RawKeyIdentity> _heldModifiers = [];
    private readonly HashSet<RawKeyIdentity> _matchedDown = [];
    private readonly Dictionary<ushort, ShortcutModifiers> _shortcuts = new()
    {
        [shortcutVirtualKey] = ShortcutModifiers.None
    };

    public ushort ShortcutVirtualKey { get; private set; } = shortcutVirtualKey;
    public IReadOnlyDictionary<ushort, ShortcutModifiers> Shortcuts => _shortcuts;
    public IReadOnlySet<ushort> ShortcutVirtualKeys => _shortcuts.Keys.ToHashSet();

    public void SetShortcut(ushort virtualKey) =>
        SetShortcuts([new ShortcutBinding(virtualKey)]);

    public void SetShortcuts(IEnumerable<ushort> virtualKeys) =>
        SetShortcuts(virtualKeys.Select(key => new ShortcutBinding(key)));

    public void SetShortcuts(IEnumerable<ShortcutBinding> bindings)
    {
        var values = bindings.ToArray();
        foreach (var binding in values)
        {
            if (binding.VirtualKey == EscapeVirtualKey)
                throw new ArgumentException("Escape is reserved for cancellation.", nameof(bindings));
        }
        if (values.Select(binding => binding.VirtualKey).Distinct().Count() != values.Length)
            throw new ArgumentException("Each configured primary key must be unique.", nameof(bindings));
        _shortcuts.Clear();
        foreach (var binding in values)
            _shortcuts[binding.VirtualKey] = binding.Modifiers;
        ShortcutVirtualKey = values.FirstOrDefault().VirtualKey;
        ResetHeldKeys();
    }

    public InterpretedKeyEvent? Interpret(ushort virtualKey, ushort scanCode,
        ushort flags, bool isKeyUp, nint deviceHandle = default, long? timestamp = null)
    {
        var eventTimestamp = timestamp ?? System.Diagnostics.Stopwatch.GetTimestamp();
        var identity = new RawKeyIdentity(deviceHandle, virtualKey, scanCode,
            (ushort)(flags & 0x06));
        var modifier = ModifierFor(virtualKey);
        var isEscape = virtualKey == EscapeVirtualKey;
        var isConfigured = _shortcuts.TryGetValue(virtualKey, out var requiredModifiers);
        var trackIdentity = isEscape || isConfigured || modifier != ShortcutModifiers.None;
        long keyDownTimestamp = 0;
        var alreadyDown = trackIdentity && _down.TryGetValue(identity, out keyDownTimestamp);

        if (isKeyUp)
        {
            _down.Remove(identity);
            if (modifier != ShortcutModifiers.None)
                _heldModifiers.Remove(identity);
        }
        else
        {
            if (!alreadyDown)
                _down[identity] = eventTimestamp;
            if (modifier != ShortcutModifiers.None)
                _heldModifiers.Add(identity);
        }

        var wasMatched = isKeyUp && _matchedDown.Remove(identity);
        var currentModifiers = GetModifiers();
        // A modifier used as the primary key does not count as its own chord modifier.
        if (modifier != ShortcutModifiers.None)
            currentModifiers &= ~modifier;
        var matches = isConfigured && currentModifiers == requiredModifiers;
        if (!isEscape && !wasMatched && !matches)
            return null;
        if (!isKeyUp && matches)
            _matchedDown.Add(identity);

        var raw = new RawKeyEvent(virtualKey, scanCode, flags, isKeyUp,
            IsRepeat: !isKeyUp && alreadyDown,
            Timestamp: eventTimestamp,
            KeyDownTimestamp: alreadyDown ? keyDownTimestamp : isKeyUp ? null : eventTimestamp,
            DeviceHandle: deviceHandle,
            Modifiers: currentModifiers);

        var kind = isEscape
            ? isKeyUp ? ShortcutEventKind.EscapeUp
                : alreadyDown ? ShortcutEventKind.EscapeRepeat : ShortcutEventKind.EscapeDown
            : isKeyUp ? ShortcutEventKind.ShortcutUp
                : alreadyDown ? ShortcutEventKind.ShortcutRepeat : ShortcutEventKind.ShortcutDown;
        return new InterpretedKeyEvent(kind, raw);
    }

    public void ResetDevice(nint deviceHandle)
    {
        foreach (var identity in _down.Keys.Where(key => key.DeviceHandle == deviceHandle).ToArray())
            _down.Remove(identity);
        _heldModifiers.RemoveWhere(key => key.DeviceHandle == deviceHandle);
        _matchedDown.RemoveWhere(key => key.DeviceHandle == deviceHandle);
    }

    /// <summary>Use after focus/device loss so a lost break cannot wedge the key.</summary>
    public void ResetHeldKeys()
    {
        _down.Clear();
        _heldModifiers.Clear();
        _matchedDown.Clear();
    }

    private ShortcutModifiers GetModifiers()
    {
        var result = ShortcutModifiers.None;
        foreach (var identity in _heldModifiers)
            result |= ModifierFor(identity.VirtualKey);
        return result;
    }

    private static ShortcutModifiers ModifierFor(ushort key) => key switch
    {
        0x10 or 0xA0 or 0xA1 => ShortcutModifiers.Shift,
        0x11 or 0xA2 or 0xA3 => ShortcutModifiers.Control,
        0x12 or 0xA4 or 0xA5 => ShortcutModifiers.Alt,
        _ => ShortcutModifiers.None
    };
}
