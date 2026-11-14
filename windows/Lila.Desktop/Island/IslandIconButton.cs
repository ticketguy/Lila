using System.ComponentModel;
using System.Drawing.Drawing2D;

namespace Lila.Desktop.Island;

public enum IslandIcon
{
    Settings,
    MoreHorizontal,
    ArrowLeft,
    ChevronLeft,
    ChevronRight,
    Microphone,
    Send,
    Close,
    ScreenHighlight
}

public sealed class IslandIconButton : Button
{
    private readonly System.Windows.Forms.Timer _tabAnimTimer = new() { Interval = 45 };
    private bool _isMuted;
    private bool _isPrimary;
    private bool _edgeTab;
    private bool _dockedRight;
    private bool _isDimmed;
    private bool _hovered;
    private bool _pressed;
    private float _phase;
    private IslandVisualState _activityState;
    private IslandIcon _icon;

    [Browsable(false)]
    [DesignerSerializationVisibility(DesignerSerializationVisibility.Hidden)]
    public IslandIcon Icon
    {
        get => _icon;
        set
        {
            if (_icon == value)
                return;
            _icon = value;
            Invalidate();
        }
    }

    [Browsable(false)]
    [DesignerSerializationVisibility(DesignerSerializationVisibility.Hidden)]
    public bool IsMuted
    {
        get => _isMuted;
        set
        {
            if (_isMuted == value)
                return;
            _isMuted = value;
            Invalidate();
        }
    }

    [Browsable(false)]
    [DesignerSerializationVisibility(DesignerSerializationVisibility.Hidden)]
    public bool IsPrimary
    {
        get => _isPrimary;
        set
        {
            if (_isPrimary == value)
                return;
            _isPrimary = value;
            Invalidate();
        }
    }

    [Browsable(false)]
    [DesignerSerializationVisibility(DesignerSerializationVisibility.Hidden)]
    public bool IsEdgeTab
    {
        get => _edgeTab;
        set
        {
            if (_edgeTab == value)
                return;
            _edgeTab = value;
            UpdateTabAnimation();
            Invalidate();
        }
    }

    [Browsable(false)]
    [DesignerSerializationVisibility(DesignerSerializationVisibility.Hidden)]
    public bool IsDockedRight
    {
        get => _dockedRight;
        set
        {
            if (_dockedRight == value)
                return;
            _dockedRight = value;
            Invalidate();
        }
    }

    [Browsable(false)]
    [DesignerSerializationVisibility(DesignerSerializationVisibility.Hidden)]
    public bool IsDimmed
    {
        get => _isDimmed;
        set
        {
            if (_isDimmed == value)
                return;
            _isDimmed = value;
            Invalidate();
        }
    }

    [Browsable(false)]
    [DesignerSerializationVisibility(DesignerSerializationVisibility.Hidden)]
    public IslandVisualState ActivityState
    {
        get => _activityState;
        set
        {
            if (_activityState == value)
                return;
            _activityState = value;
            UpdateTabAnimation();
            Invalidate();
        }
    }

    public IslandIconButton()
    {
        FlatStyle = FlatStyle.Flat;
        FlatAppearance.BorderSize = 0;
        UseVisualStyleBackColor = false;
        BackColor = Color.FromArgb(20, 24, 32);
        Text = string.Empty;
        TabStop = true;
        Cursor = Cursors.Hand;
        SetStyle(ControlStyles.UserPaint | ControlStyles.AllPaintingInWmPaint |
                 ControlStyles.OptimizedDoubleBuffer | ControlStyles.ResizeRedraw |
                 ControlStyles.Opaque, true);
        _tabAnimTimer.Tick += (_, _) =>
        {
            _phase = (_phase + 0.09f) % 1f;
            Invalidate();
        };
    }

    protected override void OnPaintBackground(PaintEventArgs pevent)
    {
        // Handled explicitly in OnPaint to prevent WinForms WM_PRINTCLIENT ghosting
    }

    protected override void OnSizeChanged(EventArgs e)
    {
        base.OnSizeChanged(e);
        if (_edgeTab || Width < 4 || Height < 4)
        {
            var old = Region;
            Region = null;
            old?.Dispose();
            return;
        }
        using var path = new GraphicsPath();
        path.AddEllipse(0, 0, Width, Height);
        var prev = Region;
        Region = new Region(path);
        prev?.Dispose();
    }

    private void UpdateTabAnimation()
    {
        if (_edgeTab && _activityState is IslandVisualState.Listening or IslandVisualState.Speaking)
            _tabAnimTimer.Start();
        else
            _tabAnimTimer.Stop();
    }

    protected override void OnMouseEnter(EventArgs e)
    {
        base.OnMouseEnter(e);
        _hovered = true;
        Invalidate();
    }

    protected override void OnMouseLeave(EventArgs e)
    {
        base.OnMouseLeave(e);
        _hovered = false;
        Invalidate();
    }

    protected override void OnMouseDown(MouseEventArgs e)
    {
        base.OnMouseDown(e);
        _pressed = e.Button == MouseButtons.Left;
        Invalidate();
    }

    protected override void OnMouseUp(MouseEventArgs e)
    {
        base.OnMouseUp(e);
        _pressed = false;
        Invalidate();
    }

    protected override void OnEnabledChanged(EventArgs e)
    {
        base.OnEnabledChanged(e);
        Invalidate();
    }

    protected override void OnPaint(PaintEventArgs e)
    {
        var graphics = e.Graphics;
        var surfaceBg = Parent?.BackColor is { A: 255 } pBg ? pBg : Color.FromArgb(20, 24, 32);
        graphics.Clear(surfaceBg);
        graphics.SmoothingMode = SmoothingMode.AntiAlias;
        graphics.PixelOffsetMode = PixelOffsetMode.HighQuality;

        var scale = Math.Max(0.65f, Math.Min(ClientSize.Width, ClientSize.Height) / 24f);
        var background = GetBackgroundColor();
        using (var shape = CreateButtonPath())
        {
            if (Enabled && !_edgeTab && (_isPrimary ||
                            _activityState is IslandVisualState.Listening or IslandVisualState.Speaking or IslandVisualState.Thinking))
            {
                using var brush = new LinearGradientBrush(
                    ClientRectangle, Color.FromArgb(124, 156, 255),
                    Color.FromArgb(176, 108, 255), 35f);
                graphics.FillPath(brush, shape);
            }
            else
            {
                using var brush = new SolidBrush(background);
                graphics.FillPath(brush, shape);
            }

            var borderColor = _edgeTab
                ? _activityState switch
                {
                    IslandVisualState.Attention => Color.FromArgb(180, 255, 184, 107),
                    IslandVisualState.Listening => Color.FromArgb(180, 124, 156, 255),
                    IslandVisualState.Speaking or IslandVisualState.Thinking => Color.FromArgb(180, 176, 108, 255),
                    _ => Color.FromArgb(60, 255, 255, 255)
                }
                : _activityState switch
                {
                    IslandVisualState.Attention => Color.FromArgb(180, 255, 184, 107),
                    IslandVisualState.Muted or IslandVisualState.Error => Color.FromArgb(170, 255, 93, 93),
                    _ when _isPrimary => Color.Transparent,
                    _ => Color.FromArgb(38, 255, 255, 255)
                };
            if (borderColor != Color.Transparent)
            {
                using var border = new Pen(borderColor, Math.Max(1f, scale * 0.7f));
                graphics.DrawPath(border, shape);
            }
        }

        var alpha = _edgeTab && _isDimmed && !_hovered ? 115 : (Enabled ? 245 : 105);
        var iconColor = !_edgeTab && (_isMuted || _activityState is IslandVisualState.Muted)
            ? Color.FromArgb(alpha, 255, 142, 142)
            : Color.FromArgb(alpha, 255, 255, 255);

        if (_edgeTab)
        {
            DrawEdgeTabContents(graphics, iconColor);
            return;
        }

        var iconSize = 16.5f * scale;
        graphics.TranslateTransform((ClientSize.Width - iconSize) / 2f,
            (ClientSize.Height - iconSize) / 2f);
        graphics.ScaleTransform(iconSize / 24f, iconSize / 24f);
        using var iconPen = new Pen(iconColor, 2.0f)
        {
            StartCap = LineCap.Round,
            EndCap = LineCap.Round,
            LineJoin = LineJoin.Round
        };
        DrawIcon(graphics, iconPen, iconColor);
        if (_isMuted && Icon == IslandIcon.Microphone)
        {
            using var mutePen = new Pen(Color.FromArgb(255, 255, 93, 93), 2.1f)
            {
                StartCap = LineCap.Round,
                EndCap = LineCap.Round
            };
            graphics.DrawLine(mutePen, 4, 4, 20, 20);
        }
        graphics.ResetTransform();
    }

    private void DrawEdgeTabContents(Graphics graphics, Color iconColor)
    {
        var w = ClientSize.Width;
        var h = ClientSize.Height;
        var dpiScale = Math.Max(1f, h / 56f);

        // 1. Muted red dot at top (7x7 logical, top: 7px) matching HTML #tab.muted::before
        if (_isMuted || _activityState == IslandVisualState.Muted)
        {
            var dotSize = 7f * dpiScale;
            var dotRect = new RectangleF((w - dotSize) / 2f, 7f * dpiScale, dotSize, dotSize);
            using var glowBrush = new SolidBrush(Color.FromArgb(100, 255, 93, 93));
            graphics.FillEllipse(glowBrush, RectangleF.Inflate(dotRect, 2.5f * dpiScale, 2.5f * dpiScale));
            using var dotBrush = new SolidBrush(Color.FromArgb(255, 93, 93));
            graphics.FillEllipse(dotBrush, dotRect);
        }

        // 2. Center chevron (< or >)
        var chevronSize = Math.Min(13f * dpiScale, w * 0.62f);
        var stateSaved = graphics.Save();
        graphics.TranslateTransform((w - chevronSize) / 2f, (h - chevronSize) / 2f);
        graphics.ScaleTransform(chevronSize / 24f, chevronSize / 24f);
        using (var pen = new Pen(iconColor, 2.2f)
               { StartCap = LineCap.Round, EndCap = LineCap.Round, LineJoin = LineJoin.Round })
        {
            DrawIcon(graphics, pen, iconColor);
        }
        graphics.Restore(stateSaved);

        // 3. 3-bar mini equalizer at bottom (bottom: 6px) when Listening or Speaking
        if (_activityState is IslandVisualState.Listening or IslandVisualState.Speaking)
        {
            var barColor = _activityState == IslandVisualState.Speaking
                ? Color.FromArgb(207, 163, 255)
                : Color.FromArgb(165, 188, 255);
            using var barBrush = new SolidBrush(barColor);
            var barW = 2f * dpiScale;
            var gap = 2f * dpiScale;
            var totalW = 3 * barW + 2 * gap;
            var startX = (w - totalW) / 2f;
            var bottomY = h - 6f * dpiScale;
            for (var i = 0; i < 3; i++)
            {
                var wave = 0.5f + 0.5f * MathF.Sin((_phase + i * 0.22f) * MathF.Tau);
                var barH = (3f + 6f * wave) * dpiScale;
                var x = startX + i * (barW + gap);
                graphics.FillRectangle(barBrush, x, bottomY - barH, barW, barH);
            }
        }
    }

    private Color GetBackgroundColor()
    {
        if (_edgeTab)
        {
            var tabAlpha = _isDimmed && !_hovered ? 115 : 225;
            return Color.FromArgb(tabAlpha, 20, 24, 32);
        }
        if (!Enabled)
            return Color.FromArgb(18, 255, 255, 255);
        if (_isMuted || _activityState is IslandVisualState.Muted or IslandVisualState.Error)
            return Color.FromArgb(46, 255, 93, 93);
        if (_activityState == IslandVisualState.Attention)
            return Color.FromArgb(50, 255, 184, 107);
        if (_activityState is IslandVisualState.Listening or IslandVisualState.Speaking or IslandVisualState.Thinking || _isPrimary)
            return Color.FromArgb(127, 119, 255);
        if (_pressed)
            return Color.FromArgb(65, 255, 255, 255);
        if (_hovered)
            return Color.FromArgb(40, 255, 255, 255);
        return Color.FromArgb(20, 255, 255, 255);
    }

    private GraphicsPath CreateButtonPath()
    {
        var path = new GraphicsPath();
        var bounds = RectangleF.Inflate(ClientRectangle, -1.0f, -1.0f);
        if (_edgeTab)
        {
            var radius = Math.Min(14f * Math.Max(1f, ClientSize.Height / 56f), bounds.Width - 2f);
            if (!_dockedRight)
            {
                path.AddLine(bounds.Left, bounds.Top, bounds.Right - radius, bounds.Top);
                path.AddArc(bounds.Right - radius * 2, bounds.Top,
                    radius * 2, radius * 2, 270, 90);
                path.AddLine(bounds.Right, bounds.Top + radius, bounds.Right, bounds.Bottom - radius);
                path.AddArc(bounds.Right - radius * 2, bounds.Bottom - radius * 2,
                    radius * 2, radius * 2, 0, 90);
                path.AddLine(bounds.Right - radius, bounds.Bottom, bounds.Left, bounds.Bottom);
                path.AddLine(bounds.Left, bounds.Bottom, bounds.Left, bounds.Top);
            }
            else
            {
                path.AddLine(bounds.Left + radius, bounds.Top, bounds.Right, bounds.Top);
                path.AddLine(bounds.Right, bounds.Top, bounds.Right, bounds.Bottom);
                path.AddLine(bounds.Right, bounds.Bottom, bounds.Left + radius, bounds.Bottom);
                path.AddArc(bounds.Left, bounds.Bottom - radius * 2,
                    radius * 2, radius * 2, 90, 90);
                path.AddLine(bounds.Left, bounds.Bottom - radius, bounds.Left, bounds.Top + radius);
                path.AddArc(bounds.Left, bounds.Top, radius * 2, radius * 2, 180, 90);
            }
            path.CloseFigure();
            return path;
        }

        path.AddEllipse(bounds);
        return path;
    }

    private void DrawIcon(Graphics graphics, Pen pen, Color fillColor)
    {
        switch (Icon)
        {
            case IslandIcon.Settings:
                DrawGear(graphics, pen);
                break;
            case IslandIcon.MoreHorizontal:
                using (var dotBrush = new SolidBrush(fillColor))
                {
                    graphics.FillEllipse(dotBrush, 4.2f, 10.2f, 3.6f, 3.6f);
                    graphics.FillEllipse(dotBrush, 10.2f, 10.2f, 3.6f, 3.6f);
                    graphics.FillEllipse(dotBrush, 16.2f, 10.2f, 3.6f, 3.6f);
                }
                break;
            case IslandIcon.ArrowLeft:
                graphics.DrawLine(pen, 19, 12, 5, 12);
                graphics.DrawLines(pen, [new PointF(11, 6), new PointF(5, 12), new PointF(11, 18)]);
                break;
            case IslandIcon.ChevronLeft:
                graphics.DrawLines(pen, [new PointF(15, 5), new PointF(8, 12), new PointF(15, 19)]);
                break;
            case IslandIcon.ChevronRight:
                graphics.DrawLines(pen, [new PointF(9, 5), new PointF(16, 12), new PointF(9, 19)]);
                break;
            case IslandIcon.Microphone:
                using (var microphone = RoundedPath(new RectangleF(9, 3, 6, 11), 3))
                    graphics.DrawPath(pen, microphone);
                graphics.DrawArc(pen, 5, 6, 14, 13, 0, 180);
                graphics.DrawLine(pen, 12, 19, 12, 22);
                break;
            case IslandIcon.Send:
                graphics.DrawLine(pen, 12, 19, 12, 5);
                graphics.DrawLines(pen, [new PointF(6, 11), new PointF(12, 5), new PointF(18, 11)]);
                break;
            case IslandIcon.Close:
                graphics.DrawLine(pen, 6, 6, 18, 18);
                graphics.DrawLine(pen, 18, 6, 6, 18);
                break;
            case IslandIcon.ScreenHighlight:
                // 4 corner selection brackets + center sparkle/focus dot
                graphics.DrawLines(pen, [new PointF(4, 9), new PointF(4, 4), new PointF(9, 4)]);
                graphics.DrawLines(pen, [new PointF(15, 4), new PointF(20, 4), new PointF(20, 9)]);
                graphics.DrawLines(pen, [new PointF(20, 15), new PointF(20, 20), new PointF(15, 20)]);
                graphics.DrawLines(pen, [new PointF(9, 20), new PointF(4, 20), new PointF(4, 15)]);
                using (var dot = new SolidBrush(fillColor))
                    graphics.FillEllipse(dot, 10.2f, 10.2f, 3.6f, 3.6f);
                break;
        }
    }

    private static void DrawGear(Graphics graphics, Pen pen)
    {
        graphics.DrawEllipse(pen, 8.5f, 8.5f, 7f, 7f);
        for (var index = 0; index < 8; index++)
        {
            var angle = index * MathF.PI / 4f;
            var inner = new PointF(12f + MathF.Cos(angle) * 6.2f, 12f + MathF.Sin(angle) * 6.2f);
            var outer = new PointF(12f + MathF.Cos(angle) * 9.2f, 12f + MathF.Sin(angle) * 9.2f);
            graphics.DrawLine(pen, inner, outer);
        }
    }

    private static GraphicsPath RoundedPath(RectangleF bounds, float radius)
    {
        var path = new GraphicsPath();
        var diameter = radius * 2;
        path.AddArc(bounds.X, bounds.Y, diameter, diameter, 180, 90);
        path.AddArc(bounds.Right - diameter, bounds.Y, diameter, diameter, 270, 90);
        path.AddArc(bounds.Right - diameter, bounds.Bottom - diameter, diameter, diameter, 0, 90);
        path.AddArc(bounds.X, bounds.Bottom - diameter, diameter, diameter, 90, 90);
        path.CloseFigure();
        return path;
    }

    protected override void Dispose(bool disposing)
    {
        if (disposing)
            _tabAnimTimer.Dispose();
        base.Dispose(disposing);
    }
}
