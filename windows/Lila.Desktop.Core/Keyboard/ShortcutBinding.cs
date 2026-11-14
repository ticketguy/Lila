namespace Lila.Desktop.Core.Keyboard;

[Flags]
public enum ShortcutModifiers : byte
{
    None = 0,
    Control = 1,
    Alt = 2,
    Shift = 4
}

public enum ShortcutActionMode
{
    Hold,
    Toggle
}

/// <summary>A conservative global shortcut binding represented by Win32 VK values.</summary>
public readonly record struct ShortcutBinding(ushort VirtualKey, ShortcutModifiers Modifiers = ShortcutModifiers.None)
{
    public string DisplayName
    {
        get
        {
            var prefix = string.Join("+", new[]
            {
                Modifiers.HasFlag(ShortcutModifiers.Control) ? "Ctrl" : null,
                Modifiers.HasFlag(ShortcutModifiers.Alt) ? "Alt" : null,
                Modifiers.HasFlag(ShortcutModifiers.Shift) ? "Shift" : null
            }.Where(value => value is not null));
            var key = VirtualKey switch
            {
                KeyEventInterpreter.RightControlVirtualKey => "Right Ctrl",
                KeyEventInterpreter.PauseVirtualKey => "Pause (Fn + P on ThinkPad)",
                >= 0x70 and <= 0x7B => $"F{VirtualKey - 0x6F}",
                _ => $"VK 0x{VirtualKey:X2}"
            };
            return prefix.Length == 0 ? key : $"{prefix}+{key}";
        }
    }

    public void Validate(ShortcutActionMode mode)
    {
        const ShortcutModifiers allModifiers = ShortcutModifiers.Control |
            ShortcutModifiers.Alt | ShortcutModifiers.Shift;
        if ((Modifiers & ~allModifiers) != 0)
            throw new ArgumentException("Only Ctrl, Alt and Shift modifiers are supported.");
        if (VirtualKey is 0 or KeyEventInterpreter.EscapeVirtualKey ||
            VirtualKey is >= 0x30 and <= 0x5A || VirtualKey is >= 0x60 and <= 0x6F ||
            (VirtualKey is >= 0xA0 and <= 0xA5 && VirtualKey != KeyEventInterpreter.RightControlVirtualKey) ||
            VirtualKey is >= 0x5B and <= 0x5C)
            throw new ArgumentException("This key is reserved, ordinary typing, or unreliable as a global shortcut.");

        if (VirtualKey == KeyEventInterpreter.RightControlVirtualKey)
        {
            if (Modifiers != ShortcutModifiers.None || mode != ShortcutActionMode.Hold)
                throw new ArgumentException("Right Ctrl is supported only as standalone hold-to-talk.");
            return;
        }

        var pauseToggle = VirtualKey == KeyEventInterpreter.PauseVirtualKey &&
                          Modifiers == ShortcutModifiers.None && mode == ShortcutActionMode.Toggle;
        if (pauseToggle)
            return;

        throw new ArgumentException(
            "Supported bindings are Right Ctrl (hold) and Pause (toggle). Other combinations are blocked.");
    }

    public static bool Conflicts(ShortcutBinding hold, ShortcutBinding toggle) =>
        hold.VirtualKey == toggle.VirtualKey;
}
