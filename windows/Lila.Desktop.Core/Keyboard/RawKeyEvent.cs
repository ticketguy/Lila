namespace Lila.Desktop.Core.Keyboard;

public sealed record RawKeyEvent(
    ushort VirtualKey,
    ushort ScanCode,
    ushort Flags,
    bool IsKeyUp,
    bool IsRepeat,
    long Timestamp = 0,
    long? KeyDownTimestamp = null,
    nint DeviceHandle = default,
    ShortcutModifiers Modifiers = ShortcutModifiers.None)
{
    public bool IsExtended => (Flags & 0x02) != 0;
    public bool IsE1 => (Flags & 0x04) != 0;
    public string Direction => IsKeyUp ? "UP" : "DOWN";
    public RawKeyIdentity Identity => new(DeviceHandle, VirtualKey, ScanCode,
        (ushort)(Flags & 0x06));
    public TimeSpan? HeldDuration => IsKeyUp && KeyDownTimestamp is long start && Timestamp >= start
        ? System.Diagnostics.Stopwatch.GetElapsedTime(start, Timestamp)
        : null;
}

public readonly record struct RawKeyIdentity(
    nint DeviceHandle,
    ushort VirtualKey,
    ushort ScanCode,
    ushort ExtendedFlags);

public enum ShortcutEventKind
{
    None,
    ShortcutDown,
    ShortcutRepeat,
    ShortcutUp,
    EscapeDown,
    EscapeRepeat,
    EscapeUp
}

public sealed record InterpretedKeyEvent(
    ShortcutEventKind Kind,
    RawKeyEvent RawEvent);
