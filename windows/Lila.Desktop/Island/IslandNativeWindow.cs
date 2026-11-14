using System.Diagnostics;
using System.Drawing;
using System.Runtime.InteropServices;
using System.Windows.Forms;

namespace Lila.Desktop.Island;

internal static class IslandNativeWindow
{
    private const int DwmwaUseImmersiveDarkMode = 20;
    private const int DwmwaWindowCornerPreference = 33;
    private const int DwmwaSystemBackdropType = 38;
    private const int DwmwcpRound = 2;
    private const int DwmSystemBackdropTransient = 3;
    private const uint MonitorDefaultToNearest = 2;
    private const int WcaAccentPolicy = 19;
    private const int AccentEnableBlurBehind = 3;
    private const int AccentEnableAcrylicBlurBehind = 4;
    private const uint SwpNoMove = 0x0002;
    private const uint SwpNoSize = 0x0001;
    private const uint SwpNoActivate = 0x0010;

    public static string LastBackdropDiagnostic { get; private set; } = "uninitialized";

    public static bool ApplyGlassBackdrop(IntPtr handle)
    {
        var osVersion = Environment.OSVersion.Version;
        var isWin11OrLater = osVersion.Major >= 10 && osVersion.Build >= 22000;
        var hrDark = SetDwmAttribute(handle, DwmwaUseImmersiveDarkMode, 1);
        var hrCorner = isWin11OrLater ? SetDwmAttribute(handle, DwmwaWindowCornerPreference, DwmwcpRound) : -1;
        var hrBackdrop = isWin11OrLater ? SetDwmAttribute(handle, DwmwaSystemBackdropType, DwmSystemBackdropTransient) : -1;

        var acrylicApplied = false;
        if (!isWin11OrLater && osVersion.Major >= 10 && osVersion.Build >= 17134)
        {
            // Windows 10 (e.g. 19045): AccentEnableBlurBehind (3) gives fast frosted blur
            // without Win10's AccentEnableAcrylicBlurBehind (4) DWM drag/animation stall.
            acrylicApplied = ApplyWin10Accent(handle, AccentEnableBlurBehind, 0x99121620);
        }

        var tier = isWin11OrLater && hrBackdrop == 0
            ? "Tier2-DwmTransientBackdrop"
            : acrylicApplied
                ? "Tier2-Win10AcrylicBackdrop"
                : "Tier1-FlatTranslucentSurface";
        LastBackdropDiagnostic =
            $"OS={osVersion} Tier={tier} hrDark=0x{hrDark:X8} hrCorner=0x{hrCorner:X8} hrBackdrop=0x{hrBackdrop:X8} win10Acrylic={acrylicApplied}";
        Debug.WriteLine("[Lila.Island.Glass] " + LastBackdropDiagnostic);
        return isWin11OrLater && hrBackdrop == 0 || acrylicApplied;
    }

    public static bool ApplyWin10Accent(IntPtr handle, int accentState, uint abgrTint = 0x99121620)
    {
        if (handle == IntPtr.Zero)
            return false;
        try
        {
            var accent = new AccentPolicy
            {
                AccentState = accentState,
                AccentFlags = 2,
                GradientColor = unchecked((int)abgrTint),
                AnimationId = 0
            };
            var size = Marshal.SizeOf(accent);
            var ptr = Marshal.AllocHGlobal(size);
            try
            {
                Marshal.StructureToPtr(accent, ptr, false);
                var data = new WindowCompositionAttributeData
                {
                    Attribute = WcaAccentPolicy,
                    Data = ptr,
                    SizeOfData = size
                };
                return SetWindowCompositionAttribute(handle, ref data);
            }
            finally
            {
                Marshal.FreeHGlobal(ptr);
            }
        }
        catch
        {
            return false;
        }
    }

    public static void PlaceWindowDirectlyBehind(IntPtr behindHandle, IntPtr frontHandle)
    {
        if (behindHandle == IntPtr.Zero || frontHandle == IntPtr.Zero)
            return;
        SetWindowPos(behindHandle, frontHandle, 0, 0, 0, 0, SwpNoMove | SwpNoSize | SwpNoActivate);
    }

    public static bool IsFullscreenApplicationActive(IntPtr islandHandle)
    {
        var foreground = GetForegroundWindow();
        if (foreground == IntPtr.Zero || foreground == islandHandle)
            return false;

        _ = GetWindowThreadProcessId(foreground, out var processId);
        if (processId == Environment.ProcessId || !GetWindowRect(foreground, out var nativeBounds))
            return false;

        var bounds = Rectangle.FromLTRB(nativeBounds.Left, nativeBounds.Top,
            nativeBounds.Right, nativeBounds.Bottom);
        if (bounds.Width <= 0 || bounds.Height <= 0)
            return false;
        var monitor = MonitorFromWindow(foreground, MonitorDefaultToNearest);
        if (monitor == IntPtr.Zero)
            return false;
        var screen = Screen.FromHandle(foreground).Bounds;
        return Math.Abs(bounds.Left - screen.Left) <= 2 &&
               Math.Abs(bounds.Top - screen.Top) <= 2 &&
               Math.Abs(bounds.Right - screen.Right) <= 2 &&
               Math.Abs(bounds.Bottom - screen.Bottom) <= 2;
    }

    private static int SetDwmAttribute(IntPtr handle, int attribute, int value)
    {
        try
        {
            return DwmSetWindowAttribute(handle, attribute, ref value, sizeof(int));
        }
        catch (DllNotFoundException)
        {
            return -2;
        }
        catch (EntryPointNotFoundException)
        {
            return -3;
        }
    }

    [StructLayout(LayoutKind.Sequential)]
    private struct NativeRect
    {
        public int Left;
        public int Top;
        public int Right;
        public int Bottom;
    }

    [StructLayout(LayoutKind.Sequential)]
    private struct AccentPolicy
    {
        public int AccentState;
        public int AccentFlags;
        public int GradientColor;
        public int AnimationId;
    }

    [StructLayout(LayoutKind.Sequential)]
    private struct WindowCompositionAttributeData
    {
        public int Attribute;
        public IntPtr Data;
        public int SizeOfData;
    }

    [DllImport("dwmapi.dll", PreserveSig = true)]
    private static extern int DwmSetWindowAttribute(IntPtr hwnd, int attribute,
        ref int value, int valueSize);

    [DllImport("user32.dll")]
    private static extern bool SetWindowCompositionAttribute(IntPtr hwnd,
        ref WindowCompositionAttributeData data);

    [DllImport("user32.dll", SetLastError = true)]
    private static extern bool SetWindowPos(IntPtr hWnd, IntPtr hWndInsertAfter,
        int x, int y, int cx, int cy, uint uFlags);

    [DllImport("user32.dll")]
    private static extern IntPtr GetForegroundWindow();

    [DllImport("user32.dll")]
    private static extern bool GetWindowRect(IntPtr window, out NativeRect bounds);

    [DllImport("user32.dll")]
    private static extern uint GetWindowThreadProcessId(IntPtr window, out uint processId);

    [DllImport("user32.dll")]
    private static extern IntPtr MonitorFromWindow(IntPtr window, uint flags);
}

/// <summary>
/// Non-activating, click-through DWM Acrylic/Blur backdrop window paired directly behind
/// Lila Island so Windows 10 (build 19045+) and Windows 11 get true iOS-style frosted glass
/// blur without affecting GDI+ child control alpha rendering.
/// </summary>
internal sealed class IslandBackdropForm : Form
{
    private const int WsExTransparent = 0x00000020;
    private const int WsExToolWindow = 0x00000080;
    private const int WsExNoActivate = 0x08000000;
    private const int AccentBlurBehind = 3;
    private const int AccentAcrylicBlurBehind = 4;

    private readonly Form _ownerForm;
    private bool _dragging;

    public bool IsGlassActive { get; private set; }

    public IslandBackdropForm(Form ownerForm)
    {
        _ownerForm = ownerForm;
        FormBorderStyle = FormBorderStyle.None;
        StartPosition = FormStartPosition.Manual;
        ShowInTaskbar = false;
        TopMost = ownerForm.TopMost;
        Enabled = false;
        BackColor = Color.Black;
        Bounds = ownerForm.Bounds;
    }

    protected override bool ShowWithoutActivation => true;

    protected override CreateParams CreateParams
    {
        get
        {
            var cp = base.CreateParams;
            cp.ExStyle |= WsExTransparent | WsExToolWindow | WsExNoActivate;
            return cp;
        }
    }

    protected override void OnHandleCreated(EventArgs e)
    {
        base.OnHandleCreated(e);
        IsGlassActive = IslandNativeWindow.ApplyGlassBackdrop(Handle);
    }

    protected override void OnPaintBackground(PaintEventArgs e)
    {
        // Black/zero-alpha client area lets DWM Acrylic blur shine through cleanly.
        e.Graphics.Clear(Color.Black);
    }

    public void SetDragging(bool dragging)
    {
        if (_dragging == dragging || IsDisposed || !IsHandleCreated)
            return;
        _dragging = dragging;
        // Switch from Acrylic (4) to lightweight BlurBehind (3) while dragging to avoid Win10 19045 drag stutter.
        var os = Environment.OSVersion.Version;
        if (os.Major >= 10 && os.Build < 22000)
        {
            IslandNativeWindow.ApplyWin10Accent(Handle,
                dragging ? AccentBlurBehind : AccentAcrylicBlurBehind,
                dragging ? 0xBB141820u : 0x99121620u);
        }
    }

    public void SyncToOwner(Rectangle bounds, Region? ownerRegion, bool ownerVisible)
    {
        if (IsDisposed)
            return;
        if (!ownerVisible)
        {
            if (Visible)
                Hide();
            return;
        }
        if (!Visible)
        {
            Show();
            if (IsHandleCreated)
                IsGlassActive = IslandNativeWindow.ApplyGlassBackdrop(Handle);
        }
        if (Bounds != bounds)
            Bounds = bounds;
        if (ownerRegion is not null)
        {
            var old = Region;
            Region = ownerRegion.Clone();
            old?.Dispose();
        }
        if (IsHandleCreated && _ownerForm.IsHandleCreated)
            IslandNativeWindow.PlaceWindowDirectlyBehind(Handle, _ownerForm.Handle);
    }
}
