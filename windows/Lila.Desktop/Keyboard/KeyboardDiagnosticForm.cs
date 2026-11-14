using System.Drawing.Drawing2D;
using Lila.Desktop.Core.Keyboard;
using Lila.Desktop.Core.Voice;
using Lila.Desktop.Island;

namespace Lila.Desktop.Keyboard;

public sealed class KeyboardDiagnosticForm : Form
{
    private static readonly Color SurfaceBg = Color.FromArgb(20, 24, 32);
    private static readonly Color CardBg = Color.FromArgb(26, 31, 42);
    private static readonly Color TextPrimary = Color.FromArgb(245, 247, 250);
    private static readonly Color TextMuted = Color.FromArgb(167, 176, 192);
    private static readonly Color AccentBlue = Color.FromArgb(124, 156, 255);
    private static readonly Color AccentPurple = Color.FromArgb(176, 108, 255);
    private static readonly Color MakeColor = Color.FromArgb(124, 196, 255);
    private static readonly Color BreakColor = Color.FromArgb(255, 184, 107);

    private readonly ListView _events = new()
    {
        Dock = DockStyle.Fill,
        View = View.Details,
        FullRowSelect = true,
        OwnerDraw = true,
        BorderStyle = BorderStyle.None,
        BackColor = CardBg,
        ForeColor = TextPrimary,
        HeaderStyle = ColumnHeaderStyle.Nonclickable
    };
    private readonly RawKeyboardListener _listener;
    private readonly InteractionDiagnostics _diagnostics;
    private readonly Func<VoiceInteractionState> _currentState;
    private readonly Label _statePill = new();
    private bool _dragging;
    private Point _dragStartPointer;
    private Point _dragStartLocation;

    public KeyboardDiagnosticForm(RawKeyboardListener listener, InteractionDiagnostics diagnostics,
        Func<VoiceInteractionState> currentState)
    {
        _listener = listener;
        _diagnostics = diagnostics;
        _currentState = currentState;

        Text = "Lila Keyboard Diagnostic";
        FormBorderStyle = FormBorderStyle.None;
        StartPosition = FormStartPosition.CenterParent;
        ShowInTaskbar = false;
        TopMost = true;
        AutoScaleMode = AutoScaleMode.Dpi;
        ClientSize = new Size(920, 420);
        BackColor = SurfaceBg;
        ForeColor = TextPrimary;
        Opacity = 0.94;
        KeyPreview = true;
        SetStyle(ControlStyles.OptimizedDoubleBuffer | ControlStyles.AllPaintingInWmPaint |
                 ControlStyles.UserPaint | ControlStyles.ResizeRedraw, true);

        BuildUi();

        _listener.DiagnosticKeyEvent += OnDiagnosticKey;
        FormClosed += (_, _) => _listener.DiagnosticKeyEvent -= OnDiagnosticKey;
        KeyDown += (_, e) =>
        {
            if (e.KeyCode == Keys.Escape)
                Close();
        };
    }

    protected override void OnHandleCreated(EventArgs e)
    {
        base.OnHandleCreated(e);
        IslandNativeWindow.ApplyGlassBackdrop(Handle);
        UpdateWindowRegion();
    }

    protected override void OnResize(EventArgs e)
    {
        base.OnResize(e);
        UpdateWindowRegion();
    }

    protected override void OnPaint(PaintEventArgs e)
    {
        base.OnPaint(e);
        var bounds = ClientRectangle;
        if (bounds.Width < 4 || bounds.Height < 4)
            return;
        e.Graphics.SmoothingMode = SmoothingMode.AntiAlias;
        using var path = RoundedRect(new RectangleF(0.5f, 0.5f, bounds.Width - 1, bounds.Height - 1), 16);
        using var border = new LinearGradientBrush(bounds,
            Color.FromArgb(140, 124, 156, 255),
            Color.FromArgb(70, 176, 108, 255), 35f);
        using var pen = new Pen(border, 1.3f);
        e.Graphics.DrawPath(pen, path);
    }

    private void BuildUi()
    {
        var root = new TableLayoutPanel
        {
            Dock = DockStyle.Fill,
            BackColor = SurfaceBg,
            ColumnCount = 1,
            RowCount = 4,
            Padding = new Padding(14, 12, 14, 12)
        };
        root.RowStyles.Add(new RowStyle(SizeType.Absolute, 42));
        root.RowStyles.Add(new RowStyle(SizeType.Absolute, 28));
        root.RowStyles.Add(new RowStyle(SizeType.Percent, 100));
        root.RowStyles.Add(new RowStyle(SizeType.Absolute, 42));

        var header = new Panel { Dock = DockStyle.Fill, BackColor = SurfaceBg };
        header.MouseDown += OnHeaderMouseDown;
        header.MouseMove += OnHeaderMouseMove;
        header.MouseUp += OnHeaderMouseUp;

        var title = new Label
        {
            Text = "Keyboard Diagnostic",
            AutoSize = true,
            Location = new Point(4, 8),
            ForeColor = TextPrimary,
            BackColor = SurfaceBg,
            Font = new Font("Segoe UI Semibold", 11.5f, FontStyle.Bold)
        };
        title.MouseDown += OnHeaderMouseDown;
        title.MouseMove += OnHeaderMouseMove;
        title.MouseUp += OnHeaderMouseUp;

        _statePill.AutoSize = true;
        _statePill.BackColor = Color.FromArgb(30, 38, 56);
        _statePill.ForeColor = AccentBlue;
        _statePill.Font = new Font("Segoe UI Semibold", 8.25f, FontStyle.Bold);
        _statePill.Padding = new Padding(8, 3, 8, 3);
        _statePill.Text = $"State: {_currentState()}";
        _statePill.Location = new Point(195, 9);

        var closeBtn = new Button
        {
            Text = "✕",
            FlatStyle = FlatStyle.Flat,
            BackColor = Color.FromArgb(34, 40, 54),
            ForeColor = TextPrimary,
            Font = new Font("Segoe UI", 9.5f, FontStyle.Bold),
            Cursor = Cursors.Hand
        };
        closeBtn.FlatAppearance.BorderSize = 1;
        closeBtn.FlatAppearance.BorderColor = Color.FromArgb(52, 255, 255, 255);
        closeBtn.Click += (_, _) => Close();

        header.SizeChanged += (_, _) =>
        {
            var scale = (IsHandleCreated ? DeviceDpi : 96) / 96f;
            var btnSize = (int)Math.Round(30 * scale);
            closeBtn.Bounds = new Rectangle(header.ClientSize.Width - btnSize, (int)Math.Round(4 * scale), btnSize, btnSize);
        };

        header.Controls.Add(title);
        header.Controls.Add(_statePill);
        header.Controls.Add(closeBtn);

        var note = new Label
        {
            Dock = DockStyle.Fill,
            ForeColor = TextMuted,
            BackColor = SurfaceBg,
            Font = new Font("Segoe UI", 8.5f),
            TextAlign = ContentAlignment.MiddleLeft,
            Text = "Only configured shortcuts and Escape are recorded. Right Ctrl uses hold-to-talk; Pause (Fn + P) toggles."
        };

        var listHost = new Panel
        {
            Dock = DockStyle.Fill,
            BackColor = CardBg,
            Padding = new Padding(1)
        };
        listHost.Paint += (_, e) =>
        {
            e.Graphics.SmoothingMode = SmoothingMode.AntiAlias;
            using var path = RoundedRect(new RectangleF(0.5f, 0.5f, listHost.Width - 1, listHost.Height - 1), 10);
            using var pen = new Pen(Color.FromArgb(42, 255, 255, 255), 1f);
            e.Graphics.DrawPath(pen, path);
        };
        ConfigureListView();
        listHost.Controls.Add(_events);

        var footer = new Panel { Dock = DockStyle.Fill, BackColor = SurfaceBg, Padding = new Padding(0, 8, 0, 0) };
        var clear = StylePillButton("Clear view", primary: false);
        clear.Click += (_, _) => _events.Items.Clear();

        var done = StylePillButton("Close", primary: true);
        done.Click += (_, _) => Close();

        footer.SizeChanged += (_, _) =>
        {
            var scale = (IsHandleCreated ? DeviceDpi : 96) / 96f;
            var btnH = (int)Math.Round(30 * scale);
            var clearW = (int)Math.Round(104 * scale);
            var doneW = (int)Math.Round(92 * scale);
            var top = Math.Max(0, (footer.ClientSize.Height - btnH) / 2);
            clear.Bounds = new Rectangle(0, top, clearW, btnH);
            done.Bounds = new Rectangle(footer.ClientSize.Width - doneW, top, doneW, btnH);
        };

        footer.Controls.Add(clear);
        footer.Controls.Add(done);

        root.Controls.Add(header, 0, 0);
        root.Controls.Add(note, 0, 1);
        root.Controls.Add(listHost, 0, 2);
        root.Controls.Add(footer, 0, 3);
        Controls.Add(root);
    }

    private void ConfigureListView()
    {
        _events.Font = new Font("Consolas", 8.75f);
        _events.Columns.Add("Monotonic ms", 115);
        _events.Columns.Add("Virtual key", 95);
        _events.Columns.Add("Raw scan", 90);
        _events.Columns.Add("Make/break", 110);
        _events.Columns.Add("Repeat", 70);
        _events.Columns.Add("Key-down ms", 115);
        _events.Columns.Add("Hold ms", 100);
        _events.Columns.Add("Voice state", 160);

        _events.DrawColumnHeader += (_, e) =>
        {
            using var bg = new SolidBrush(Color.FromArgb(30, 35, 48));
            e.Graphics.FillRectangle(bg, e.Bounds);
            using var linePen = new Pen(Color.FromArgb(45, 255, 255, 255));
            e.Graphics.DrawLine(linePen, e.Bounds.Left, e.Bounds.Bottom - 1, e.Bounds.Right, e.Bounds.Bottom - 1);
            TextRenderer.DrawText(e.Graphics, e.Header?.Text ?? string.Empty,
                new Font("Segoe UI Semibold", 8.25f, FontStyle.Bold),
                e.Bounds, AccentBlue, TextFormatFlags.Left | TextFormatFlags.VerticalCenter | TextFormatFlags.EndEllipsis);
        };

        _events.DrawItem += (_, e) =>
        {
            var isEven = e.ItemIndex % 2 == 0;
            var bgColor = e.Item.Selected
                ? Color.FromArgb(48, 60, 92)
                : isEven ? CardBg : Color.FromArgb(23, 27, 37);
            using var bg = new SolidBrush(bgColor);
            e.Graphics.FillRectangle(bg, e.Bounds);
        };

        _events.DrawSubItem += (_, e) =>
        {
            var textColor = TextPrimary;
            if (e.ColumnIndex == 3)
                textColor = e.SubItem?.Text.Contains("UP") == true ? BreakColor : MakeColor;
            else if (e.ColumnIndex == 7)
                textColor = AccentPurple;

            TextRenderer.DrawText(e.Graphics, e.SubItem?.Text ?? string.Empty, _events.Font,
                e.Bounds, textColor, TextFormatFlags.Left | TextFormatFlags.VerticalCenter | TextFormatFlags.EndEllipsis);
        };
    }

    private static Button StylePillButton(string text, bool primary)
    {
        var btn = new Button
        {
            Text = text,
            FlatStyle = FlatStyle.Flat,
            BackColor = primary ? AccentBlue : Color.FromArgb(34, 40, 54),
            ForeColor = TextPrimary,
            Font = new Font("Segoe UI Semibold", 8.75f, FontStyle.Bold),
            Cursor = Cursors.Hand
        };
        btn.FlatAppearance.BorderSize = primary ? 0 : 1;
        btn.FlatAppearance.BorderColor = Color.FromArgb(56, 255, 255, 255);
        return btn;
    }

    private void OnHeaderMouseDown(object? sender, MouseEventArgs e)
    {
        if (e.Button != MouseButtons.Left)
            return;
        _dragging = true;
        _dragStartPointer = Cursor.Position;
        _dragStartLocation = Location;
    }

    private void OnHeaderMouseMove(object? sender, MouseEventArgs e)
    {
        if (!_dragging || Control.MouseButtons != MouseButtons.Left)
            return;
        var cur = Cursor.Position;
        Location = new Point(
            _dragStartLocation.X + (cur.X - _dragStartPointer.X),
            _dragStartLocation.Y + (cur.Y - _dragStartPointer.Y));
    }

    private void OnHeaderMouseUp(object? sender, MouseEventArgs e) => _dragging = false;

    private void UpdateWindowRegion()
    {
        if (ClientSize.Width <= 0 || ClientSize.Height <= 0)
            return;
        using var path = RoundedRect(new RectangleF(0, 0, ClientSize.Width, ClientSize.Height), 16);
        var old = Region;
        Region = new Region(path);
        old?.Dispose();
    }

    private static GraphicsPath RoundedRect(RectangleF bounds, float radius)
    {
        var path = new GraphicsPath();
        var d = Math.Min(radius * 2, Math.Min(bounds.Width, bounds.Height));
        var arc = new RectangleF(bounds.X, bounds.Y, d, d);
        path.AddArc(arc, 180, 90);
        arc.X = bounds.Right - d;
        path.AddArc(arc, 270, 90);
        arc.Y = bounds.Bottom - d;
        path.AddArc(arc, 0, 90);
        arc.X = bounds.X;
        path.AddArc(arc, 90, 90);
        path.CloseFigure();
        return path;
    }

    private void OnDiagnosticKey(RawKeyEvent input)
    {
        if (IsDisposed || !IsHandleCreated)
            return;
        if (InvokeRequired)
        {
            BeginInvoke(new Action(() => OnDiagnosticKey(input)));
            return;
        }

        var state = _currentState();
        _statePill.Text = $"State: {state}";
        _diagnostics.RecordKey(input, state);
        var item = new ListViewItem(System.Diagnostics.Stopwatch.GetElapsedTime(0, input.Timestamp)
            .TotalMilliseconds.ToString("F3", System.Globalization.CultureInfo.InvariantCulture));
        item.SubItems.Add($"0x{input.VirtualKey:X2}");
        item.SubItems.Add($"0x{input.ScanCode:X2}");
        item.SubItems.Add(input.IsKeyUp ? "BREAK / UP" : "MAKE / DOWN");
        item.SubItems.Add(input.IsRepeat ? "yes" : "no");
        item.SubItems.Add(input.KeyDownTimestamp is long down
            ? System.Diagnostics.Stopwatch.GetElapsedTime(0, down).TotalMilliseconds.ToString("F3") : "n/a");
        item.SubItems.Add(input.HeldDuration?.TotalMilliseconds.ToString("F3") ?? "n/a");
        item.SubItems.Add(state.ToString());
        _events.Items.Add(item);
        if (_events.Items.Count > 500)
            _events.Items.RemoveAt(0);
        item.EnsureVisible();
    }
}
