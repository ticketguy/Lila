using System.ComponentModel;
using System.Drawing.Drawing2D;

namespace Lila.Desktop.Island;

/// <summary>
/// 5-bar voice equalizer matching HTML .bars (2.5px bar width, 2px gap,
/// #A5BCFF when Listening, #CFA3FF when Speaking).
/// </summary>
public sealed class IslandEqualizerControl : Control
{
    private readonly System.Windows.Forms.Timer _timer = new() { Interval = 36 };
    private IslandVisualState _state;
    private float _phase;

    [Browsable(false)]
    [DesignerSerializationVisibility(DesignerSerializationVisibility.Hidden)]
    public IslandVisualState State
    {
        get => _state;
        set
        {
            if (_state == value)
                return;
            _state = value;
            var active = _state is IslandVisualState.Listening or IslandVisualState.Speaking;
            Visible = active;
            if (active)
                _timer.Start();
            else
                _timer.Stop();
            Invalidate();
        }
    }

    public IslandEqualizerControl()
    {
        SetStyle(
            ControlStyles.UserPaint |
            ControlStyles.AllPaintingInWmPaint |
            ControlStyles.OptimizedDoubleBuffer |
            ControlStyles.ResizeRedraw |
            ControlStyles.Opaque,
            true);
        BackColor = Color.FromArgb(20, 24, 32);
        Size = new Size(22, 14);
        Visible = false;
        TabStop = false;
        _timer.Tick += (_, _) =>
        {
            _phase = (_phase + 0.085f) % 1f;
            Invalidate();
        };
    }

    protected override void OnPaintBackground(PaintEventArgs pevent)
    {
    }

    protected override void OnPaint(PaintEventArgs e)
    {
        var g = e.Graphics;
        var surfaceBg = Parent?.BackColor is { A: 255 } pBg ? pBg : Color.FromArgb(20, 24, 32);
        g.Clear(surfaceBg);
        if (_state is not (IslandVisualState.Listening or IslandVisualState.Speaking))
            return;

        g.SmoothingMode = SmoothingMode.AntiAlias;
        var scale = Math.Max(1f, ClientSize.Height / 14f);
        var barWidth = 2.5f * scale;
        var gap = 2.0f * scale;
        var totalWidth = 5 * barWidth + 4 * gap;
        var startX = (ClientSize.Width - totalWidth) / 2f;
        var centerY = ClientSize.Height / 2f;

        var color = _state == IslandVisualState.Speaking
            ? Color.FromArgb(207, 163, 255)
            : Color.FromArgb(165, 188, 255);
        using var brush = new SolidBrush(color);

        for (var i = 0; i < 5; i++)
        {
            var wave = 0.5f + 0.5f * MathF.Sin((_phase + i * 0.14f) * MathF.Tau);
            var h = (3.2f + 9.8f * wave) * scale;
            var x = startX + i * (barWidth + gap);
            var y = centerY - h / 2f;
            using var path = new GraphicsPath();
            var r = Math.Min(barWidth, h);
            path.AddArc(x, y, r, r, 180, 180);
            path.AddArc(x, y + h - r, r, r, 0, 180);
            path.CloseFigure();
            g.FillPath(brush, path);
        }
    }

    protected override void Dispose(bool disposing)
    {
        if (disposing)
            _timer.Dispose();
        base.Dispose(disposing);
    }
}
