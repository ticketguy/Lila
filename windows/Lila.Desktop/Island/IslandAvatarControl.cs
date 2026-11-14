using System.ComponentModel;
using System.Drawing.Drawing2D;

namespace Lila.Desktop.Island;

public enum IslandVisualState
{
    Idle,
    Listening,
    Speaking,
    Attention,
    Thinking,
    Interrupted,
    Muted,
    Error,
    Disconnected
}

public sealed class IslandAvatarControl : Control
{
    private readonly System.Windows.Forms.Timer _pulse = new() { Interval = 32 };
    private IslandVisualState _state;
    private bool _isMuted;
    private float _phase;
    private long _flashStartedTimestamp;
    private bool _flashingMute;

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
            UpdateTimerState();
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

    public IslandAvatarControl()
    {
        SetStyle(
            ControlStyles.UserPaint |
            ControlStyles.AllPaintingInWmPaint |
            ControlStyles.OptimizedDoubleBuffer |
            ControlStyles.ResizeRedraw |
            ControlStyles.Opaque,
            true);
        BackColor = Color.FromArgb(20, 24, 32);
        AccessibleName = "Lila activity";
        Size = new Size(40, 40);
        _pulse.Tick += (_, _) =>
        {
            _phase = (_phase + 0.028f) % 1f;
            if (_flashingMute &&
                System.Diagnostics.Stopwatch.GetElapsedTime(_flashStartedTimestamp).TotalMilliseconds >= 360)
            {
                _flashingMute = false;
                UpdateTimerState();
            }
            Invalidate();
        };
    }

    protected override void OnPaintBackground(PaintEventArgs pevent)
    {
        // Cleared explicitly in OnPaint
    }

    protected override void OnSizeChanged(EventArgs e)
    {
        base.OnSizeChanged(e);
        if (Width < 4 || Height < 4)
            return;
        using var path = new GraphicsPath();
        path.AddEllipse(0, 0, Width, Height);
        var prev = Region;
        Region = new Region(path);
        prev?.Dispose();
    }

    public void FlashMuteBadge()
    {
        _flashingMute = true;
        _flashStartedTimestamp = System.Diagnostics.Stopwatch.GetTimestamp();
        UpdateTimerState();
        Invalidate();
    }

    private void UpdateTimerState()
    {
        var needsAnimation = _flashingMute ||
            _state is IslandVisualState.Listening or IslandVisualState.Speaking or
                IslandVisualState.Attention or IslandVisualState.Thinking;
        if (needsAnimation)
            _pulse.Start();
        else
            _pulse.Stop();
    }

    protected override void OnPaint(PaintEventArgs e)
    {
        var graphics = e.Graphics;
        var surfaceBg = Parent?.BackColor is { A: 255 } pBg ? pBg : Color.FromArgb(20, 24, 32);
        graphics.Clear(surfaceBg);
        graphics.SmoothingMode = SmoothingMode.AntiAlias;
        graphics.PixelOffsetMode = PixelOffsetMode.HighQuality;
        var scale = Math.Min(ClientSize.Width, ClientSize.Height) / 48f;
        if (scale <= 0)
            return;

        var origin = new PointF((ClientSize.Width - 48 * scale) / 2f,
            (ClientSize.Height - 48 * scale) / 2f);
        var accent = AccentColor();

        // Outer state glow matching HTML .av::after
        if (_state is IslandVisualState.Listening or IslandVisualState.Speaking or IslandVisualState.Attention)
        {
            var pulse = 0.45f + 0.40f * (0.5f + 0.5f * MathF.Sin(_phase * MathF.Tau));
            using var outerGlow = new Pen(Color.FromArgb((int)(pulse * 95), accent), 4.2f * scale);
            graphics.DrawEllipse(outerGlow, origin.X + 1.5f * scale, origin.Y + 1.5f * scale,
                45f * scale, 45f * scale);
        }

        var saved = graphics.Save();
        graphics.TranslateTransform(origin.X, origin.Y);
        graphics.ScaleTransform(scale, scale);

        // Inset avatar circle (3..45 in 48x48 coordinate space so outer rings & mute badge never clip)
        using (var clip = new GraphicsPath())
        {
            clip.AddEllipse(3, 3, 42, 42);
            graphics.SetClip(clip);
        }
        using (var clipCircle = new GraphicsPath())
        {
            clipCircle.AddEllipse(3, 3, 42, 42);
            using var background = new PathGradientBrush(clipCircle)
            {
                CenterPoint = new PointF(18, 15),
                CenterColor = Color.FromArgb(43, 58, 102),
                SurroundColors = Enumerable.Repeat(
                    Color.FromArgb(23, 27, 44), clipCircle.PointCount).ToArray()
            };
            graphics.FillEllipse(background, 3, 3, 42, 42);
        }

        // Portrait artwork matching HTML SVG avatar
        using (var hairBase = new GraphicsPath())
        {
            hairBase.AddBezier(24, 6, 14, 6, 9, 14, 9, 24);
            hairBase.AddBezier(9, 24, 8.5f, 33, 8.5f, 40, 11.5f, 45);
            hairBase.AddLine(11.5f, 45, 36.5f, 45);
            hairBase.AddBezier(36.5f, 45, 39.5f, 40, 39.5f, 33, 39, 24);
            hairBase.AddBezier(39, 24, 39, 14, 34, 6, 24, 6);
            using var hairBrush = new LinearGradientBrush(
                new RectangleF(8, 6, 32, 40), Color.FromArgb(111, 134, 232),
                Color.FromArgb(122, 79, 214), 45f);
            graphics.FillPath(hairBrush, hairBase);
        }

        using (var body = new SolidBrush(Color.FromArgb(35, 42, 68)))
        using (var shoulders = new GraphicsPath())
        {
            shoulders.AddBezier(12, 48, 12, 41, 17, 38, 24, 38);
            shoulders.AddBezier(24, 38, 31, 38, 36, 41, 36, 48);
            shoulders.CloseFigure();
            graphics.FillPath(body, shoulders);
        }
        using (var neck = new SolidBrush(Color.FromArgb(227, 184, 166)))
        {
            var neckShape = new[] { new PointF(20.8f, 32.5f), new PointF(27.2f, 32.5f),
                new PointF(27.2f, 39.5f), new PointF(24, 41.5f), new PointF(20.8f, 39.5f) };
            graphics.FillPolygon(neck, neckShape);
        }
        using (var face = new SolidBrush(Color.FromArgb(244, 215, 200)))
            graphics.FillEllipse(face, 14.6f, 12.8f, 18.8f, 22.4f);

        using (var hair = new GraphicsPath())
        {
            hair.AddBezier(14, 24, 13.5f, 13, 20, 8.8f, 26.5f, 9.8f);
            hair.AddBezier(26.5f, 9.8f, 32, 10.8f, 35, 15.8f, 34, 24);
            hair.AddBezier(34, 24, 32.2f, 19, 28.5f, 16.2f, 23.5f, 16.6f);
            hair.AddBezier(23.5f, 16.6f, 19.5f, 16.9f, 16, 20.2f, 14, 24);
            hair.CloseFigure();
            using var hairBrush = new LinearGradientBrush(
                new RectangleF(13, 9, 22, 16), Color.FromArgb(157, 180, 255),
                Color.FromArgb(176, 108, 255), 35f);
            graphics.FillPath(hairBrush, hair);
        }
        using (var sideHair = new GraphicsPath())
        using (var hairBrush = new LinearGradientBrush(
                   new RectangleF(8, 20, 32, 26), Color.FromArgb(157, 180, 255),
                   Color.FromArgb(176, 108, 255), 20f))
        {
            sideHair.AddBezier(9f, 24, 8.5f, 33, 8.5f, 40, 11.5f, 45);
            sideHair.AddLine(11.5f, 45, 16, 45);
            sideHair.AddBezier(16, 45, 14, 39, 13.5f, 32, 14.5f, 25);
            sideHair.CloseFigure();
            graphics.FillPath(hairBrush, sideHair);

            sideHair.Reset();
            sideHair.AddBezier(39f, 24, 39.5f, 33, 39.5f, 40, 36.5f, 45);
            sideHair.AddLine(36.5f, 45, 32, 45);
            sideHair.AddBezier(32, 45, 34, 39, 34.5f, 32, 33.5f, 25);
            sideHair.CloseFigure();
            graphics.FillPath(hairBrush, sideHair);
        }
        using (var features = new Pen(Color.FromArgb(106, 65, 86), 1.1f)
               { StartCap = LineCap.Round, EndCap = LineCap.Round })
        using (var mouth = new SolidBrush(Color.FromArgb(212, 106, 134)))
        using (var mouthOutline = new Pen(Color.FromArgb(212, 106, 134), 0.6f))
        {
            graphics.DrawArc(features, 19.2f, 24, 3.8f, 2, 20, 140);
            graphics.DrawArc(features, 25, 24, 3.8f, 2, 20, 140);
            using var smile = new GraphicsPath();
            smile.AddBezier(21.6f, 30.2f, 23.2f, 32.0f, 24.8f, 32.0f, 26.4f, 30.2f);
            smile.AddBezier(26.4f, 30.2f, 24.8f, 31.0f, 23.2f, 31.0f, 21.6f, 30.2f);
            smile.CloseFigure();
            graphics.FillPath(mouth, smile);
            graphics.DrawPath(mouthOutline, smile);
        }

        graphics.Restore(saved);

        var circle = new RectangleF(origin.X + 3f * scale, origin.Y + 3f * scale,
            42f * scale, 42f * scale);

        if (_state == IslandVisualState.Thinking)
        {
            // Conic-style multi-segment spinning ring (#7C9CFF -> #B06CFF -> #5EE0A0)
            using var baseRing = new Pen(Color.FromArgb(46, 255, 255, 255), 1.4f * scale);
            graphics.DrawEllipse(baseRing, circle);

            var ringRect = RectangleF.Inflate(circle, 1.5f * scale, 1.5f * scale);
            var baseAngle = _phase * 360f;
            ReadOnlySpan<(Color Color, float Sweep)> segments =
            [
                (Color.FromArgb(235, 124, 156, 255), 95f),
                (Color.FromArgb(235, 176, 108, 255), 95f),
                (Color.FromArgb(215, 94, 224, 160), 75f)
            ];
            var cursor = baseAngle;
            foreach (var (segColor, sweep) in segments)
            {
                using var segPen = new Pen(segColor, 2.2f * scale)
                {
                    StartCap = LineCap.Round,
                    EndCap = LineCap.Round
                };
                graphics.DrawArc(segPen, ringRect, cursor, sweep);
                cursor += sweep - 8f;
            }
        }
        else if (_state is IslandVisualState.Listening or IslandVisualState.Speaking or IslandVisualState.Attention)
        {
            using var activeRing = new Pen(accent, 2.0f * scale);
            graphics.DrawEllipse(activeRing, RectangleF.Inflate(circle, 1.0f * scale, 1.0f * scale));
        }
        else
        {
            using var idleRing = new Pen(Color.FromArgb(48, 255, 255, 255), 1.4f * scale);
            graphics.DrawEllipse(idleRing, circle);
        }

        // Mute badge (.mbadge: 14x14 #FF5D5D circle at bottom-right with white minus bar & .flash scale(1.8))
        if (_isMuted || _state == IslandVisualState.Muted)
        {
            var flashFactor = 1f;
            if (_flashingMute)
            {
                var elapsedMs = (float)System.Diagnostics.Stopwatch.GetElapsedTime(_flashStartedTimestamp).TotalMilliseconds;
                var t = Math.Clamp(elapsedMs / 360f, 0f, 1f);
                flashFactor = 1f + 0.8f * MathF.Sin(t * MathF.PI);
            }
            var baseBadgeSize = 15.5f * scale;
            var badgeSize = baseBadgeSize * flashFactor;
            var badgeCenter = new PointF(origin.X + 39.5f * scale, origin.Y + 39.5f * scale);
            var badgeRect = new RectangleF(
                badgeCenter.X - badgeSize / 2f,
                badgeCenter.Y - badgeSize / 2f,
                badgeSize,
                badgeSize);

            if (_flashingMute)
            {
                using var flashGlow = new SolidBrush(Color.FromArgb(110, 255, 93, 93));
                graphics.FillEllipse(flashGlow, RectangleF.Inflate(badgeRect, 3.5f * scale, 3.5f * scale));
            }
            using (var badgeFill = new SolidBrush(Color.FromArgb(255, 93, 93)))
                graphics.FillEllipse(badgeFill, badgeRect);
            using (var badgeBorder = new Pen(Color.FromArgb(190, 18, 22, 30), 1.6f * scale))
                graphics.DrawEllipse(badgeBorder, badgeRect);

            var barHalfW = 3.2f * scale * flashFactor;
            using var barPen = new Pen(Color.White, Math.Max(1.6f, 1.8f * scale * flashFactor))
            {
                StartCap = LineCap.Round,
                EndCap = LineCap.Round
            };
            graphics.DrawLine(barPen, badgeCenter.X - barHalfW, badgeCenter.Y, badgeCenter.X + barHalfW, badgeCenter.Y);
        }
    }

    private Color AccentColor() => _state switch
    {
        IslandVisualState.Listening => Color.FromArgb(124, 156, 255),
        IslandVisualState.Speaking => Color.FromArgb(176, 108, 255),
        IslandVisualState.Attention => Color.FromArgb(255, 184, 107),
        IslandVisualState.Thinking => Color.FromArgb(176, 108, 255),
        IslandVisualState.Interrupted => Color.FromArgb(255, 184, 107),
        IslandVisualState.Muted => Color.FromArgb(255, 93, 93),
        IslandVisualState.Error => Color.FromArgb(255, 138, 128),
        IslandVisualState.Disconnected => Color.FromArgb(125, 132, 145),
        _ => Color.FromArgb(94, 224, 160)
    };

    protected override void Dispose(bool disposing)
    {
        if (disposing)
            _pulse.Dispose();
        base.Dispose(disposing);
    }
}
