using System.Drawing.Drawing2D;
using Lila.Desktop.Core.Api;

namespace Lila.Desktop.Island;

/// <summary>
/// Non-activating Reply Peek bubble (#peek in HTML reference): max-width 280px,
/// shown beside collapsed or tucked Island for 6s; clicking opens/expands Lila.
/// </summary>
internal sealed class IslandPeekPopup : Form
{
    private readonly System.Windows.Forms.Timer _hideTimer = new() { Interval = 6_000 };
    private readonly Label _label = new()
    {
        AutoSize = false,
        ForeColor = Color.FromArgb(245, 247, 250),
        BackColor = Color.Transparent,
        Font = new Font("Segoe UI", 9.5f),
        Cursor = Cursors.Hand
    };

    public event Action? Clicked;

    protected override bool ShowWithoutActivation => true;

    protected override CreateParams CreateParams
    {
        get
        {
            var cp = base.CreateParams;
            cp.ExStyle |= 0x08000000 | 0x00000080; // WS_EX_NOACTIVATE | WS_EX_TOOLWINDOW
            return cp;
        }
    }

    public IslandPeekPopup()
    {
        FormBorderStyle = FormBorderStyle.None;
        StartPosition = FormStartPosition.Manual;
        ShowInTaskbar = false;
        TopMost = true;
        BackColor = Color.FromArgb(20, 24, 32);
        DoubleBuffered = true;
        Cursor = Cursors.Hand;

        _label.Click += (_, _) =>
        {
            HidePopup();
            Clicked?.Invoke();
        };
        Click += (_, _) =>
        {
            HidePopup();
            Clicked?.Invoke();
        };
        Controls.Add(_label);

        _hideTimer.Tick += (_, _) => HidePopup();
    }

    public void ShowPeek(string text, Rectangle anchorBounds, Screen screen)
    {
        if (IsDisposed || string.IsNullOrWhiteSpace(text))
            return;

        var scale = (DeviceDpi > 0 ? DeviceDpi : 96) / 96d;
        var padH = (int)Math.Round(13 * scale);
        var padV = (int)Math.Round(10 * scale);
        var maxW = (int)Math.Round(280 * scale);
        var textMaxW = maxW - padH * 2;

        _label.Font = new Font("Segoe UI", 9.5f);
        var measured = TextRenderer.MeasureText(text, _label.Font, new Size(textMaxW, (int)Math.Round(120 * scale)),
            TextFormatFlags.WordBreak | TextFormatFlags.TextBoxControl);
        var w = Math.Clamp(measured.Width + padH * 2 + 4, (int)Math.Round(120 * scale), maxW);
        var h = Math.Clamp(measured.Height + padV * 2 + 2, (int)Math.Round(38 * scale), (int)Math.Round(120 * scale));

        _label.Text = text;
        _label.Bounds = new Rectangle(padH, padV, w - padH * 2, h - padV * 2);

        var gap = (int)Math.Round(10 * scale);
        var workArea = screen.WorkingArea;
        var dockedRight = anchorBounds.Left + anchorBounds.Width / 2 >= workArea.Left + workArea.Width / 2;
        var x = dockedRight ? anchorBounds.Left - w - gap : anchorBounds.Right + gap;
        var y = anchorBounds.Top + (anchorBounds.Height - h) / 2;
        x = Math.Clamp(x, workArea.Left + 4, workArea.Right - w - 4);
        y = Math.Clamp(y, workArea.Top + 4, workArea.Bottom - h - 4);

        Bounds = new Rectangle(x, y, w, h);
        UpdateRoundedRegion((float)(14 * scale));
        if (!Visible)
            Show();
        Invalidate();
        _hideTimer.Stop();
        _hideTimer.Start();
    }

    public void HidePopup()
    {
        _hideTimer.Stop();
        if (!IsDisposed && Visible)
            Hide();
    }

    protected override void OnPaint(PaintEventArgs e)
    {
        base.OnPaint(e);
        e.Graphics.SmoothingMode = SmoothingMode.AntiAlias;
        var scale = (float)((DeviceDpi > 0 ? DeviceDpi : 96) / 96d);
        using var path = RoundedRect(new RectangleF(1, 1, Width - 3, Height - 3), 14f * scale);
        using var fill = new LinearGradientBrush(ClientRectangle,
            Color.FromArgb(245, 24, 29, 40), Color.FromArgb(245, 18, 22, 30), 35f);
        e.Graphics.FillPath(fill, path);
        using var pen = new Pen(Color.FromArgb(65, 255, 255, 255), 1.4f);
        e.Graphics.DrawPath(pen, path);
    }

    private void UpdateRoundedRegion(float radius)
    {
        if (Width <= 0 || Height <= 0)
            return;
        using var path = RoundedRect(new RectangleF(0, 0, Width, Height), radius);
        var old = Region;
        Region = new Region(path);
        old?.Dispose();
    }

    private static GraphicsPath RoundedRect(RectangleF bounds, float radius)
    {
        var path = new GraphicsPath();
        var d = Math.Min(radius * 2, Math.Min(bounds.Width, bounds.Height));
        path.AddArc(bounds.X, bounds.Y, d, d, 180, 90);
        path.AddArc(bounds.Right - d, bounds.Y, d, d, 270, 90);
        path.AddArc(bounds.Right - d, bounds.Bottom - d, d, d, 0, 90);
        path.AddArc(bounds.X, bounds.Bottom - d, d, d, 90, 90);
        path.CloseFigure();
        return path;
    }

    protected override void Dispose(bool disposing)
    {
        if (disposing)
            _hideTimer.Dispose();
        base.Dispose(disposing);
    }
}

/// <summary>
/// Floating Tucked Approval Toast (#toast in HTML reference): 300px wide card shown
/// beside #tab when an approval arrives while tucked. Includes .tbar progress bar,
/// countdown, Reject, Approve (.primary), and Open Lila buttons.
/// </summary>
public sealed class IslandApprovalToastPopup : Form
{
    private readonly Label _title = new()
    {
        Text = "Lila needs your approval",
        AutoSize = true,
        ForeColor = Color.FromArgb(255, 208, 137),
        BackColor = Color.Transparent,
        Font = new Font("Segoe UI", 9.5f, FontStyle.Bold)
    };
    private readonly Label _detail = new()
    {
        AutoSize = false,
        ForeColor = Color.FromArgb(245, 247, 250),
        BackColor = Color.Transparent,
        Font = new Font("Segoe UI", 8.75f)
    };
    private readonly Panel _progressBar = new() { Height = 3, BackColor = Color.FromArgb(30, 255, 255, 255) };
    private readonly Label _countdown = new()
    {
        AutoSize = true,
        ForeColor = Color.FromArgb(167, 176, 192),
        BackColor = Color.Transparent,
        Font = new Font("Segoe UI", 8f)
    };
    private readonly Button _rejectBtn = new() { Text = "Reject" };
    private readonly Button _approveBtn = new() { Text = "Approve" };
    private readonly Button _speakBtn = new() { Text = "Speak" };
    private readonly Button _detailsBtn = new() { Text = "Details" };
    private readonly Button _openBtn = new() { Text = "Open" };
    private readonly System.Windows.Forms.Timer _autoTuckTimer = new() { Interval = 7000 };
    private float _progressFraction = 1f;
    private bool _detailsExpanded;
    private string? _lastInvocationId;
    private string? _autoTuckedInvocationId;
    private ApprovalRequestView? _lastRequest;
    private double _lastRemaining;
    private double _lastTotal;
    private Rectangle _lastTabBounds;
    private Screen? _lastScreen;
    private bool _lastSpokenAvailable;

    public event Action? RejectClicked;
    public event Action? ApproveClicked;
    public event Action? SpeakApproveClicked;
    public event Action? OpenLilaClicked;
    public event Action? AutoTucked;

    public bool DetailsExpanded => _detailsExpanded;
    public bool TestShowWithoutActivation => ShowWithoutActivation;
    public int TestExStyle => CreateParams.ExStyle;

    public void PerformTestAction(string action)
    {
        switch (action)
        {
            case "approve": _approveBtn.PerformClick(); break;
            case "reject": _rejectBtn.PerformClick(); break;
            case "speak": _speakBtn.PerformClick(); break;
            case "details": _detailsBtn.PerformClick(); break;
            case "open": _openBtn.PerformClick(); break;
        }
    }

    public bool IsAutoTuckedFor(string? invocationId) =>
        !string.IsNullOrEmpty(invocationId) &&
        string.Equals(_autoTuckedInvocationId, invocationId, StringComparison.Ordinal);

    protected override bool ShowWithoutActivation => true;

    protected override CreateParams CreateParams
    {
        get
        {
            var cp = base.CreateParams;
            cp.ExStyle |= 0x08000000 | 0x00000080; // WS_EX_NOACTIVATE | WS_EX_TOOLWINDOW
            return cp;
        }
    }

    public IslandApprovalToastPopup(int autoTuckMs = 7000)
    {
        _autoTuckTimer.Interval = Math.Max(20, autoTuckMs);
        FormBorderStyle = FormBorderStyle.None;
        StartPosition = FormStartPosition.Manual;
        ShowInTaskbar = false;
        TopMost = true;
        BackColor = Color.FromArgb(20, 24, 32);
        DoubleBuffered = true;

        StyleToastButton(_rejectBtn, primary: false);
        StyleToastButton(_approveBtn, primary: true);
        StyleToastButton(_speakBtn, primary: false);
        StyleToastButton(_detailsBtn, primary: false);
        StyleToastButton(_openBtn, primary: false);

        _rejectBtn.Click += (_, _) => RejectClicked?.Invoke();
        _approveBtn.Click += (_, _) => ApproveClicked?.Invoke();
        _speakBtn.Click += (_, _) => SpeakApproveClicked?.Invoke();
        _openBtn.Click += (_, _) => OpenLilaClicked?.Invoke();
        _detailsBtn.Click += (_, _) =>
        {
            _detailsExpanded = !_detailsExpanded;
            _detailsBtn.Text = _detailsExpanded ? "Less" : "Details";
            if (_detailsExpanded)
                _autoTuckTimer.Stop();
            if (_lastRequest is not null && _lastScreen is not null)
                ShowOrUpdateToast(_lastRequest, _lastRemaining, _lastTotal, _lastTabBounds, _lastScreen,
                    _lastSpokenAvailable, forceShow: true);
        };
        _autoTuckTimer.Tick += (_, _) =>
        {
            _autoTuckTimer.Stop();
            if (_detailsExpanded)
                return;
            _autoTuckedInvocationId = _lastInvocationId;
            if (!IsDisposed && Visible)
                Hide();
            AutoTucked?.Invoke();
        };

        _progressBar.Paint += (_, e) =>
        {
            var fillW = (int)Math.Round(_progressBar.Width * Math.Clamp(_progressFraction, 0f, 1f));
            if (fillW > 0)
            {
                using var b = new SolidBrush(Color.FromArgb(255, 184, 107));
                e.Graphics.FillRectangle(b, 0, 0, fillW, _progressBar.Height);
            }
        };

        Controls.Add(_title);
        Controls.Add(_detail);
        Controls.Add(_progressBar);
        Controls.Add(_countdown);
        Controls.Add(_rejectBtn);
        Controls.Add(_approveBtn);
        Controls.Add(_speakBtn);
        Controls.Add(_detailsBtn);
        Controls.Add(_openBtn);
    }

    protected override void Dispose(bool disposing)
    {
        if (disposing)
            _autoTuckTimer.Dispose();
        base.Dispose(disposing);
    }

    public void ShowOrUpdateToast(ApprovalRequestView request, double remainingSeconds, double totalSeconds,
        Rectangle tabBounds, Screen screen, bool spokenAvailable = false, bool forceShow = false)
    {
        if (IsDisposed)
            return;

        if (!string.Equals(_lastInvocationId, request.InvocationId, StringComparison.Ordinal))
        {
            _lastInvocationId = request.InvocationId;
            _autoTuckedInvocationId = null;
            _detailsExpanded = false;
            _detailsBtn.Text = "Details";
            _autoTuckTimer.Stop();
            _autoTuckTimer.Start();
        }
        else if (forceShow)
        {
            _autoTuckedInvocationId = null;
            if (!_detailsExpanded)
            {
                _autoTuckTimer.Stop();
                _autoTuckTimer.Start();
            }
        }
        else if (IsAutoTuckedFor(request.InvocationId))
        {
            return;
        }

        _lastRequest = request;
        _lastRemaining = remainingSeconds;
        _lastTotal = totalSeconds;
        _lastTabBounds = tabBounds;
        _lastScreen = screen;
        _lastSpokenAvailable = spokenAvailable;

        var scale = (DeviceDpi > 0 ? DeviceDpi : 96) / 96d;
        int S(int v) => (int)Math.Round(v * scale);

        var w = S(320);
        var pad = S(12);
        var innerW = w - pad * 2;

        _title.Location = new Point(pad, pad);
        var action = request.DisplayAction ?? request.CapabilityId;
        var target = request.Target is { } t ? t.GetRawText() : (request.TargetKind ?? "target");
        if (_detailsExpanded)
        {
            var values = request.DisplayValues.ValueKind == System.Text.Json.JsonValueKind.Undefined
                ? "{}"
                : request.DisplayValues.GetRawText();
            _detail.Text = $"{action}\r\nTarget: {target}\r\nValues: {values}";
        }
        else
        {
            if (target.Length > 86)
                target = target[..83] + "…";
            _detail.Text = $"{action}\r\n{target}";
        }
        var detailHeight = _detailsExpanded ? S(84) : S(40);
        _detail.Bounds = new Rectangle(pad, pad + S(22), innerW, detailHeight);

        _progressFraction = totalSeconds > 0
            ? (float)Math.Clamp(remainingSeconds / totalSeconds, 0d, 1d)
            : 0f;
        var progressY = pad + S(26) + detailHeight;
        _progressBar.Bounds = new Rectangle(pad, progressY, innerW, Math.Max(2, S(3)));
        _progressBar.Invalidate();

        _countdown.Text = $"Rejects automatically in {Math.Max(0, (int)Math.Ceiling(remainingSeconds))} s";
        _countdown.Location = new Point(pad, progressY + S(7));

        var btnY = progressY + S(30);
        var btnGap = S(5);
        var btnH = S(28);
        var showSpeak = spokenAvailable && request.ApprovalReady;
        _speakBtn.Visible = showSpeak;
        _speakBtn.Enabled = showSpeak;
        var topCount = showSpeak ? 3 : 2;
        var topW = (innerW - btnGap * (topCount - 1)) / topCount;
        _rejectBtn.Bounds = new Rectangle(pad, btnY, topW, btnH);
        _approveBtn.Bounds = new Rectangle(pad + topW + btnGap, btnY, topW, btnH);
        if (showSpeak)
            _speakBtn.Bounds = new Rectangle(pad + (topW + btnGap) * 2, btnY, innerW - (topW + btnGap) * 2, btnH);
        _approveBtn.Enabled = request.ApprovalReady;
        _rejectBtn.Enabled = request.ApprovalReady;

        var row2Y = btnY + btnH + S(5);
        var row2W = (innerW - btnGap) / 2;
        _detailsBtn.Bounds = new Rectangle(pad, row2Y, row2W, S(24));
        _openBtn.Bounds = new Rectangle(pad + row2W + btnGap, row2Y, innerW - row2W - btnGap, S(24));

        var h = row2Y + S(24) + pad;
        var workArea = screen.WorkingArea;
        var dockedRight = tabBounds.Left + tabBounds.Width / 2 >= workArea.Left + workArea.Width / 2;
        var x = dockedRight ? tabBounds.Left - w - S(10) : tabBounds.Right + S(10);
        var y = Math.Clamp(tabBounds.Top - S(20), workArea.Top + 4, workArea.Bottom - h - 4);
        x = Math.Clamp(x, workArea.Left + 4, workArea.Right - w - 4);

        if (Bounds != new Rectangle(x, y, w, h))
        {
            Bounds = new Rectangle(x, y, w, h);
            using var path = RoundedRect(new RectangleF(0, 0, w, h), (float)(14 * scale));
            var old = Region;
            Region = new Region(path);
            old?.Dispose();
        }
        if (!Visible)
            Show();
        Invalidate();
    }

    public void HideToast()
    {
        _autoTuckTimer.Stop();
        if (!IsDisposed && Visible)
            Hide();
    }

    private static void StyleToastButton(Button btn, bool primary)
    {
        btn.FlatStyle = FlatStyle.Flat;
        btn.FlatAppearance.BorderSize = 1;
        btn.FlatAppearance.BorderColor = primary
            ? Color.FromArgb(150, 140, 255)
            : Color.FromArgb(60, 255, 255, 255);
        btn.BackColor = primary
            ? Color.FromArgb(132, 122, 255)
            : Color.FromArgb(36, 41, 54);
        btn.ForeColor = Color.White;
        btn.Font = new Font("Segoe UI", 8.25f, primary ? FontStyle.Bold : FontStyle.Regular);
        btn.Cursor = Cursors.Hand;
    }

    protected override void OnPaint(PaintEventArgs e)
    {
        base.OnPaint(e);
        e.Graphics.SmoothingMode = SmoothingMode.AntiAlias;
        var scale = (float)((DeviceDpi > 0 ? DeviceDpi : 96) / 96d);
        using var path = RoundedRect(new RectangleF(1, 1, Width - 3, Height - 3), 14f * scale);
        using var fill = new LinearGradientBrush(ClientRectangle,
            Color.FromArgb(248, 32, 29, 34), Color.FromArgb(248, 20, 24, 32), 45f);
        e.Graphics.FillPath(fill, path);
        using var pen = new Pen(Color.FromArgb(165, 255, 184, 107), 1.5f);
        e.Graphics.DrawPath(pen, path);
    }

    private static GraphicsPath RoundedRect(RectangleF bounds, float radius)
    {
        var path = new GraphicsPath();
        var d = Math.Min(radius * 2, Math.Min(bounds.Width, bounds.Height));
        path.AddArc(bounds.X, bounds.Y, d, d, 180, 90);
        path.AddArc(bounds.Right - d, bounds.Y, d, d, 270, 90);
        path.AddArc(bounds.Right - d, bounds.Bottom - d, d, d, 0, 90);
        path.AddArc(bounds.X, bounds.Bottom - d, d, d, 90, 90);
        path.CloseFigure();
        return path;
    }
}
