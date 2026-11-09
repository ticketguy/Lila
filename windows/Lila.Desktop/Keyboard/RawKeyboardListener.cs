using System.Runtime.InteropServices;
using Lila.Desktop.Core.Keyboard;

namespace Lila.Desktop.Keyboard;

public sealed class RawKeyboardListener : NativeWindow, IDisposable
{
    private const int WmInput = 0x00FF;
    private const int WmInputDeviceChange = 0x00FE;
    private const uint RidInput = 0x10000003;
    private const uint RidevRemove = 0x00000001;
    private const uint RidevInputSink = 0x00000100;
    private const uint RidevDeviceNotify = 0x00002000;
    private const ushort GenericDesktopPage = 0x01;
    private const ushort KeyboardUsage = 0x06;
    private readonly KeyEventInterpreter _interpreter;
    private bool _disposed;

    public RawKeyboardListener(ushort shortcutVirtualKey)
        : this([shortcutVirtualKey]) { }

    public RawKeyboardListener(IEnumerable<ushort> shortcutVirtualKeys)
        : this(shortcutVirtualKeys.Select(key => new ShortcutBinding(key))) { }

    public RawKeyboardListener(IEnumerable<ShortcutBinding> bindings)
    {
        var values = bindings.ToArray();
        _interpreter = new KeyEventInterpreter(values.FirstOrDefault().VirtualKey);
        _interpreter.SetShortcuts(values);
        CreateHandle(new CreateParams { Caption = "LilaRawKeyboardInput" });
        try
        {
            Register(RidevInputSink, Handle);
        }
        catch
        {
            DestroyHandle();
            throw;
        }
    }

    public event Action<InterpretedKeyEvent>? RelevantKeyEvent;
    public event Action<RawKeyEvent>? DiagnosticKeyEvent;
    public event Action<ushort>? NonModifierKeyDown;
    public event Action<nint>? KeyboardDeviceRemoved;

    public ushort ShortcutVirtualKey
    {
        get => _interpreter.ShortcutVirtualKey;
        set => _interpreter.SetShortcut(value);
    }

    public void SetShortcuts(IEnumerable<ushort> virtualKeys) => _interpreter.SetShortcuts(virtualKeys);

    public void SetShortcuts(IEnumerable<ShortcutBinding> bindings) => _interpreter.SetShortcuts(bindings);

    protected override void WndProc(ref Message message)
    {
        if (message.Msg == WmInput)
            ProcessRawInput(message.LParam);
        else if (message.Msg == WmInputDeviceChange && message.WParam.ToInt32() == 2)
        {
            _interpreter.ResetDevice(message.LParam);
            KeyboardDeviceRemoved?.Invoke(message.LParam);
        }
        base.WndProc(ref message);
    }

    private void ProcessRawInput(IntPtr rawInputHandle)
    {
        uint size = 0;
        var headerSize = (uint)Marshal.SizeOf<RawInputHeader>();
        if (GetRawInputData(rawInputHandle, RidInput, IntPtr.Zero, ref size, headerSize) == uint.MaxValue ||
            size < headerSize + Marshal.SizeOf<RawKeyboard>())
            return;

        var buffer = Marshal.AllocHGlobal((int)size);
        try
        {
            if (GetRawInputData(rawInputHandle, RidInput, buffer, ref size, headerSize) == uint.MaxValue)
                return;
            var header = Marshal.PtrToStructure<RawInputHeader>(buffer);
            if (header.Type != 1) // RIM_TYPEKEYBOARD
                return;
            var key = Marshal.PtrToStructure<RawKeyboard>(IntPtr.Add(buffer, (int)headerSize));
            var virtualKey = key.VirtualKey == 0x11
                ? (ushort)((key.Flags & 0x02) != 0 ? 0xA3 : 0xA2)
                : key.VirtualKey;
            var isUp = (key.Flags & 0x01) != 0;
            if (!isUp && !IsModifier(virtualKey))
                NonModifierKeyDown?.Invoke(virtualKey);

            var interpreted = _interpreter.Interpret(virtualKey, key.MakeCode,
                key.Flags, isUp, header.Device);
            if (interpreted is null)
                return;

            DiagnosticKeyEvent?.Invoke(interpreted.RawEvent);
            RelevantKeyEvent?.Invoke(interpreted);
        }
        finally
        {
            Marshal.FreeHGlobal(buffer);
        }
    }

    private static void Register(uint flags, IntPtr target)
    {
        var device = new RawInputDevice
        {
            UsagePage = GenericDesktopPage,
            Usage = KeyboardUsage,
            Flags = flags == RidevRemove ? RidevRemove : flags | RidevDeviceNotify,
            Target = target
        };
        if (!RegisterRawInputDevices([device], 1, (uint)Marshal.SizeOf<RawInputDevice>()))
            throw new InvalidOperationException(
                $"Windows Raw Input registration failed ({Marshal.GetLastWin32Error()}).");
    }

    private static bool IsModifier(ushort key) => key is
        0x10 or 0xA0 or 0xA1 or // Shift
        0x11 or 0xA2 or 0xA3 or // Ctrl
        0x12 or 0xA4 or 0xA5 or // Alt
        0x5B or 0x5C;           // Windows keys

    public void Dispose()
    {
        if (_disposed)
            return;
        _disposed = true;
        try { Register(RidevRemove, IntPtr.Zero); }
        catch (InvalidOperationException) { }
        finally { DestroyHandle(); }
    }

    [StructLayout(LayoutKind.Sequential)]
    private struct RawInputDevice
    {
        public ushort UsagePage;
        public ushort Usage;
        public uint Flags;
        public IntPtr Target;
    }

    [StructLayout(LayoutKind.Sequential)]
    private struct RawInputHeader
    {
        public uint Type;
        public uint Size;
        public IntPtr Device;
        public IntPtr WParam;
    }

    [StructLayout(LayoutKind.Sequential)]
    private struct RawKeyboard
    {
        public ushort MakeCode;
        public ushort Flags;
        public ushort Reserved;
        public ushort VirtualKey;
        public uint Message;
        public uint ExtraInformation;
    }

    [DllImport("user32.dll", SetLastError = true)]
    private static extern bool RegisterRawInputDevices(RawInputDevice[] devices,
        uint numberOfDevices, uint structureSize);

    [DllImport("user32.dll", SetLastError = true)]
    private static extern uint GetRawInputData(IntPtr rawInput, uint command,
        IntPtr data, ref uint size, uint headerSize);
}
