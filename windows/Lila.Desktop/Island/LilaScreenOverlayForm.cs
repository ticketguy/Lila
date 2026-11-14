using System.Drawing.Drawing2D;
using System.Drawing.Imaging;
using System.Drawing.Text;
using System.Runtime.InteropServices;
using Lila.Desktop.Core.Island;

namespace Lila.Desktop.Island;

/// <summary>
/// Reusable bottom-center "Lila Screen" Dynamic Island overlay:
/// - Per-pixel alpha layered window (no jagged edges, no focus stealing).
/// - Click-through on empty space and live captions (WM_NCHITTEST -> HTTRANSPARENT).
/// - Clickable bottom pill (active voice/state pill or idle "N results · click to open" chip)
///   and up to 3 stacked dismissible result cards (Research, Image, Video, Task, Private).
/// - Hidden with zero timers running when idle and no overlay cards are visible.
/// </summary>
public sealed class LilaScreenOverlayForm : Form
{
    private const int WsExLayered = 0x00080000;
    private const int WsExToolWindow = 0x00000080;
    private const int WsExNoActivate = 0x08000000;
    private const int WmNcHitTest = 0x0084;
    private const int HtTransparent = -1;
    private const int HtClient = 1;
    private const byte AcSrcOver = 0x00;
    private const byte AcSrcAlpha = 0x01;
    private const int UlwAlpha = 0x02;

    private static readonly Color Indigo = Color.FromArgb(124, 156, 255);
    private static readonly Color Violet = Color.FromArgb(176, 108, 255);
    private static readonly Color SpeakingPink = Color.FromArgb(192, 132, 252);
    private static readonly Color Amber = Color.FromArgb(255, 184, 107);
    private static readonly Color Mint = Color.FromArgb(94, 224, 160);
    private static readonly Color Coral = Color.FromArgb(255, 93, 93);
    private static readonly Color VideoPink = Color.FromArgb(255, 143, 177);

    private readonly LilaScreenStore _store;
    private readonly System.Windows.Forms.Timer _animTimer = new() { Interval = 40 };
    private readonly List<ScreenCaptionLine> _captions = [];
    private readonly List<(string Id, Rectangle CardRect, Rectangle DismissRect)> _cardHits = [];
    private Rectangle _pillHitRect = Rectangle.Empty;

    private IslandVisualState _visualState = IslandVisualState.Disconnected;
    private string _statusTitle = string.Empty;
    private string _statusSub = string.Empty;
    private bool _voiceActive;
    private bool _suppressWordsForFullscreen;
    private float _animPhase;
    private long _captionsFadeAtTicks;
    private Screen? _targetScreen;

    public event Action? PillClicked;
    public event Action<string>? CardClicked;

    public LilaScreenOverlayForm(LilaScreenStore store)
    {
        _store = store ?? throw new ArgumentNullException(nameof(store));
        FormBorderStyle = FormBorderStyle.None;
        StartPosition = FormStartPosition.Manual;
        ShowInTaskbar = false;
        TopMost = true;
        AutoScaleMode = AutoScaleMode.None;
        Visible = false;

        _store.Changed += OnStoreChanged;
        _animTimer.Tick += (_, _) => OnAnimTick();
    }

    protected override bool ShowWithoutActivation => true;

    protected override CreateParams CreateParams
    {
        get
        {
            var cp = base.CreateParams;
            cp.ExStyle |= WsExLayered | WsExToolWindow | WsExNoActivate;
            return cp;
        }
    }

    public void SetTargetScreen(Screen? screen)
    {
        _targetScreen = screen ?? Screen.PrimaryScreen;
        if (Visible)
            RenderFrame();
    }

    public void SetFullscreenSuppressed(bool suppressed)
    {
        if (_suppressWordsForFullscreen == suppressed)
            return;
        _suppressWordsForFullscreen = suppressed;
        if (Visible)
            RenderFrame();
    }

    public void UpdateVoiceState(IslandVisualState state, string statusTitle, string statusSub, bool voiceActive)
    {
        _visualState = state;
        _statusTitle = statusTitle;
        _statusSub = statusSub;
        _voiceActive = voiceActive;
        RefreshVisibilityAndRender();
    }

    public void PushCaption(string speaker, string text, bool isFinal)
    {
        if (string.IsNullOrWhiteSpace(text))
            return;
        var clean = text.Trim();
        if (_captions.Count > 0 && _captions[^1].Speaker == speaker && !_captions[^1].IsFinal)
            _captions[^1] = new ScreenCaptionLine(speaker, clean, isFinal, DateTimeOffset.UtcNow);
        else
            _captions.Add(new ScreenCaptionLine(speaker, clean, isFinal, DateTimeOffset.UtcNow));

        while (_captions.Count > 6)
            _captions.RemoveAt(0);

        _captionsFadeAtTicks = isFinal ? Environment.TickCount64 + 4500 : 0;
        RefreshVisibilityAndRender();
    }

    public void ClearCaptions()
    {
        if (_captions.Count == 0)
            return;
        _captions.Clear();
        _captionsFadeAtTicks = 0;
        RefreshVisibilityAndRender();
    }

    private void OnStoreChanged()
    {
        if (IsDisposed)
            return;
        if (InvokeRequired)
        {
            BeginInvoke(new Action(OnStoreChanged));
            return;
        }
        RefreshVisibilityAndRender();
    }

    private bool IsPillVisible =>
        _voiceActive ||
        _visualState is IslandVisualState.Listening or IslandVisualState.Thinking or IslandVisualState.Speaking or
            IslandVisualState.Attention or IslandVisualState.Interrupted or IslandVisualState.Muted ||
        _store.VisibleOverlayCards.Count > 0;

    private bool NeedsAnimationTimer =>
        Visible &&
        (_visualState is IslandVisualState.Listening or IslandVisualState.Thinking or IslandVisualState.Speaking ||
         _captionsFadeAtTicks > 0 ||
         _store.VisibleOverlayCards.Any(c => c.Kind == ScreenItemKind.Task && c.TaskStage < ScreenTaskStage.Verified));

    private void RefreshVisibilityAndRender()
    {
        if (IsDisposed)
            return;
        var shouldShow = IsPillVisible || (_captions.Count > 0 && _store.CaptionsEnabled && !_suppressWordsForFullscreen);
        if (!shouldShow)
        {
            _animTimer.Stop();
            if (Visible)
                Hide();
            return;
        }

        if (!Visible)
            Show();

        RenderFrame();
        if (NeedsAnimationTimer)
            _animTimer.Start();
        else
            _animTimer.Stop();
    }

    private void OnAnimTick()
    {
        _animPhase = (_animPhase + 0.08f) % 1f;
        if (_captionsFadeAtTicks > 0 && Environment.TickCount64 >= _captionsFadeAtTicks)
        {
            _captions.Clear();
            _captionsFadeAtTicks = 0;
            RefreshVisibilityAndRender();
            return;
        }
        RenderFrame();
    }

    protected override void WndProc(ref Message m)
    {
        if (m.Msg == WmNcHitTest)
        {
            var pt = PointToClient(Cursor.Position);
            if (_pillHitRect.Contains(pt) || _cardHits.Any(hit => hit.CardRect.Contains(pt)))
            {
                m.Result = (IntPtr)HtClient;
                return;
            }
            m.Result = (IntPtr)HtTransparent;
            return;
        }
        base.WndProc(ref m);
    }

    protected override void OnMouseClick(MouseEventArgs e)
    {
        base.OnMouseClick(e);
        if (e.Button != MouseButtons.Left)
            return;

        foreach (var hit in _cardHits)
        {
            if (hit.DismissRect.Contains(e.Location))
            {
                _store.DismissFromOverlay(hit.Id);
                return;
            }
            if (hit.CardRect.Contains(e.Location))
            {
                CardClicked?.Invoke(hit.Id);
                return;
            }
        }

        if (_pillHitRect.Contains(e.Location))
            PillClicked?.Invoke();
    }

    private int Scale(int logical)
    {
        var dpi = IsHandleCreated ? DeviceDpi : 96;
        return (int)Math.Round(logical * (dpi / 96f));
    }

    private void RenderFrame()
    {
        if (IsDisposed || !IsHandleCreated)
            return;

        var screen = _targetScreen ?? Screen.PrimaryScreen ?? Screen.AllScreens.First();
        var work = screen.WorkingArea;
        var overlayW = Math.Min(Scale(580), Math.Max(Scale(320), work.Width - Scale(32)));
        var cards = _store.VisibleOverlayCards;

        // Budget height cleanly so we never cover too much of the user's screen
        var maxOverlayH = (int)(work.Height * 0.62);
        var padBottom = Scale(16);
        var gap = Scale(10);

        var isIdleChip = !_voiceActive &&
            _visualState is not (IslandVisualState.Listening or IslandVisualState.Thinking or
                IslandVisualState.Speaking or IslandVisualState.Attention or
                IslandVisualState.Interrupted or IslandVisualState.Muted);

        var pillH = Scale(isIdleChip ? 46 : 56);
        var pillW = isIdleChip
            ? Math.Min(overlayW, Scale(270))
            : Math.Min(overlayW, Scale(340));

        var captionBlockH = (_captions.Count > 0 && _store.CaptionsEnabled && !_suppressWordsForFullscreen)
            ? Math.Min(Scale(150), _captions.Count * Scale(28) + Scale(8))
            : 0;

        var cardHeights = cards.Select(MeasureCardHeight).ToArray();
        var totalCardsH = cardHeights.Sum() + (cards.Count > 0 ? (cards.Count - 1) * gap : 0);
        var totalH = Math.Min(maxOverlayH, Scale(28) + totalCardsH + (cards.Count > 0 ? gap : 0) +
                                           captionBlockH + (captionBlockH > 0 ? gap : 0) + pillH + padBottom);

        var overlayRect = new Rectangle(
            work.Left + (work.Width - overlayW) / 2,
            work.Bottom - totalH,
            overlayW,
            totalH);

        if (Bounds != overlayRect)
            Bounds = overlayRect;

        using var bmp = new Bitmap(Math.Max(1, overlayW), Math.Max(1, totalH), PixelFormat.Format32bppArgb);
        using (var g = Graphics.FromImage(bmp))
        {
            g.SmoothingMode = SmoothingMode.AntiAlias;
            g.TextRenderingHint = TextRenderingHint.AntiAliasGridFit;
            g.Clear(Color.Transparent);

            var y = totalH - padBottom - pillH;
            _pillHitRect = new Rectangle((overlayW - pillW) / 2, y, pillW, pillH);
            DrawPill(g, _pillHitRect, isIdleChip);

            if (captionBlockH > 0)
            {
                var capRect = new Rectangle(Scale(12), y - gap - captionBlockH, overlayW - Scale(24), captionBlockH);
                DrawCaptions(g, capRect);
                y = capRect.Top;
            }

            _cardHits.Clear();
            var cardW = Math.Min(Scale(520), overlayW - Scale(12));
            var cardX = (overlayW - cardW) / 2;
            for (var i = 0; i < cards.Count; i++)
            {
                var ch = cardHeights[i];
                var cardY = y - gap - ch;
                if (cardY < Scale(4))
                    break;
                var cardRect = new Rectangle(cardX, cardY, cardW, ch);
                var dismissRect = DrawCard(g, cardRect, cards[i]);
                _cardHits.Add((cards[i].Id, cardRect, dismissRect));
                y = cardY;
            }
        }

        PresentLayeredBitmap(bmp, overlayRect);
    }

    private int MeasureCardHeight(ScreenItem item)
    {
        if (item.IsPrivate || _store.ScreenSharingActive)
            return Scale(68);
        return item.Kind switch
        {
            ScreenItemKind.Task => Scale(154),
            ScreenItemKind.Image or ScreenItemKind.Video => Scale(106),
            _ => Scale(118)
        };
    }

    private void DrawPill(Graphics g, Rectangle rect, bool isIdleChip)
    {
        var accent = StateAccentColor();
        var radius = rect.Height / 2f;

        // Soft glow halo
        using (var glowBrush = new SolidBrush(Color.FromArgb(42, accent)))
        using (var glowPath = RoundedRect(RectangleF.Inflate(rect, Scale(6), Scale(4)), radius + Scale(4)))
        {
            g.FillPath(glowBrush, glowPath);
        }

        using var path = RoundedRect(rect, radius);
        using (var fill = new SolidBrush(Color.FromArgb(218, 20, 24, 32)))
            g.FillPath(fill, path);

        // Top glass specular sheen
        var sheenRect = new Rectangle(rect.X, rect.Y, rect.Width, rect.Height / 2);
        using (var sheen = new LinearGradientBrush(sheenRect,
                   Color.FromArgb(28, 255, 255, 255),
                   Color.FromArgb(0, 255, 255, 255),
                   LinearGradientMode.Vertical))
        {
            g.SetClip(path);
            g.FillRectangle(sheen, sheenRect);
            g.ResetClip();
        }

        using (var border = new LinearGradientBrush(rect, accent, Color.FromArgb(65, 255, 255, 255), 35f))
        using (var pen = new Pen(border, Math.Max(1.3f, Scale(1))))
        {
            g.DrawPath(pen, path);
        }

        // Avatar badge
        var avaSize = Scale(isIdleChip ? 30 : 36);
        var avaRect = new Rectangle(rect.X + Scale(8), rect.Y + (rect.Height - avaSize) / 2, avaSize, avaSize);
        using (var avaFill = new LinearGradientBrush(avaRect, Indigo, Violet, 45f))
            g.FillEllipse(avaFill, avaRect);
        using (var avaRing = new Pen(accent, Math.Max(1.5f, Scale(2))))
            g.DrawEllipse(avaRing, avaRect);

        // Equalizer bars when active
        var barsW = isIdleChip ? 0 : Scale(46);
        if (!isIdleChip)
        {
            var barCount = 7;
            var barW = Scale(3);
            var barGap = Scale(3);
            var startX = rect.Right - Scale(16) - (barCount * barW + (barCount - 1) * barGap);
            var maxBarH = Scale(20);
            var active = _visualState is IslandVisualState.Listening or IslandVisualState.Speaking;
            for (var i = 0; i < barCount; i++)
            {
                var level = active
                    ? 0.28f + 0.72f * (0.5f + 0.5f * MathF.Sin((_animPhase * MathF.Tau) + i * 0.85f))
                    : 0.22f;
                var bh = Math.Max(Scale(4), (int)(maxBarH * level));
                var bx = startX + i * (barW + barGap);
                var by = rect.Y + (rect.Height - bh) / 2;
                using var barBrush = new LinearGradientBrush(
                    new Rectangle(bx, by, barW, bh), Indigo, Violet, LinearGradientMode.Vertical);
                g.FillRectangle(barBrush, bx, by, barW, bh);
            }
        }

        // Title + Subtitle
        var textX = avaRect.Right + Scale(10);
        var textW = Math.Max(Scale(60), rect.Right - Scale(14) - barsW - textX);
        var title = isIdleChip
            ? "Lila"
            : string.IsNullOrWhiteSpace(_statusTitle) ? "Lila" : _statusTitle;
        var count = _store.VisibleOverlayCards.Count;
        var sub = isIdleChip
            ? $"{count} {(count == 1 ? "result" : "results")} · click to open"
            : _statusSub;

        using var titleFont = new Font("Segoe UI Semibold", 9.5f, FontStyle.Bold);
        using var subFont = new Font("Segoe UI", 8f);
        var titleRect = new Rectangle(textX, rect.Y + Scale(isIdleChip ? 6 : 8), textW, Scale(20));
        var subRect = new Rectangle(textX, rect.Y + Scale(isIdleChip ? 24 : 28), textW, Scale(18));
        TextRenderer.DrawText(g, title, titleFont, titleRect, Color.FromArgb(245, 247, 250),
            TextFormatFlags.Left | TextFormatFlags.EndEllipsis | TextFormatFlags.NoPrefix);
        if (!string.IsNullOrWhiteSpace(sub))
        {
            TextRenderer.DrawText(g, sub, subFont, subRect, Color.FromArgb(182, 190, 206),
                TextFormatFlags.Left | TextFormatFlags.EndEllipsis | TextFormatFlags.NoPrefix);
        }
    }

    private void DrawCaptions(Graphics g, Rectangle rect)
    {
        var lineH = Scale(26);
        var y = rect.Bottom - lineH;
        using var font = new Font("Segoe UI Semibold", 11.5f, FontStyle.Bold);
        using var partialFont = new Font("Segoe UI", 11.5f, FontStyle.Italic);
        using var tagFont = new Font("Segoe UI Semibold", 7.75f, FontStyle.Bold);

        for (var idx = _captions.Count - 1; idx >= 0 && y >= rect.Top; idx--)
        {
            var line = _captions[idx];
            var age = _captions.Count - 1 - idx;
            var alpha = age switch
            {
                0 => 255,
                1 => 185,
                2 => 122,
                3 => 72,
                _ => 36
            };

            var lineRect = new Rectangle(rect.X, y, rect.Width, lineH);
            var isMe = string.Equals(line.Speaker, "You", StringComparison.OrdinalIgnoreCase);
            var label = isMe ? "YOU  " : "LILA  ";
            var textFont = (isMe && !line.IsFinal) ? partialFont : font;
            var full = label + line.Text;

            // Subtle dark shadow for contrast over light windows
            var shadowRect = new Rectangle(lineRect.X, lineRect.Y + Scale(2), lineRect.Width, lineRect.Height);
            TextRenderer.DrawText(g, full, textFont, shadowRect, Color.FromArgb((int)(alpha * 0.78), 0, 0, 0),
                TextFormatFlags.HorizontalCenter | TextFormatFlags.VerticalCenter | TextFormatFlags.EndEllipsis | TextFormatFlags.NoPrefix);

            var textColor = isMe
                ? Color.FromArgb(line.IsFinal ? alpha : (int)(alpha * 0.72), 205, 214, 255)
                : Color.FromArgb(alpha, 255, 255, 255);
            TextRenderer.DrawText(g, full, textFont, lineRect, textColor,
                TextFormatFlags.HorizontalCenter | TextFormatFlags.VerticalCenter | TextFormatFlags.EndEllipsis | TextFormatFlags.NoPrefix);

            y -= lineH;
        }
    }

    private Rectangle DrawCard(Graphics g, Rectangle rect, ScreenItem item)
    {
        var hideContent = item.IsPrivate || _store.ScreenSharingActive;
        var (kindLabel, kindColor) = hideContent
            ? ("PRIVATE", Amber)
            : item.Kind switch
            {
                ScreenItemKind.Research => ("RESEARCH", Indigo),
                ScreenItemKind.Image => ("IMAGE", SpeakingPink),
                ScreenItemKind.Video => ("VIDEO", VideoPink),
                ScreenItemKind.Task => ("TASK", Mint),
                _ => ("RESULT", Indigo)
            };

        using var path = RoundedRect(rect, Scale(16));
        using (var fill = new SolidBrush(Color.FromArgb(222, 20, 24, 32)))
            g.FillPath(fill, path);

        var sheenRect = new Rectangle(rect.X, rect.Y, rect.Width, rect.Height / 3);
        using (var sheen = new LinearGradientBrush(sheenRect,
                   Color.FromArgb(20, 255, 255, 255),
                   Color.FromArgb(0, 255, 255, 255),
                   LinearGradientMode.Vertical))
        {
            g.SetClip(path);
            g.FillRectangle(sheen, sheenRect);
            g.ResetClip();
        }

        using (var borderPen = new Pen(Color.FromArgb(72, 255, 255, 255), Math.Max(1f, Scale(1))))
            g.DrawPath(borderPen, path);

        var pad = Scale(14);
        var dismissSize = Scale(22);
        var dismissRect = new Rectangle(rect.Right - pad - dismissSize, rect.Y + Scale(10), dismissSize, dismissSize);
        using (var dismissBg = new SolidBrush(Color.FromArgb(40, 255, 255, 255)))
            g.FillEllipse(dismissBg, dismissRect);
        using (var xFont = new Font("Segoe UI", 9.5f, FontStyle.Bold))
            TextRenderer.DrawText(g, "×", xFont, dismissRect, Color.FromArgb(220, 230, 245),
                TextFormatFlags.HorizontalCenter | TextFormatFlags.VerticalCenter | TextFormatFlags.NoPrefix);

        using var kindFont = new Font("Segoe UI Semibold", 7.75f, FontStyle.Bold);
        using var metaFont = new Font("Segoe UI", 8f);
        using var titleFont = new Font("Segoe UI Semibold", 9.5f, FontStyle.Bold);
        using var bodyFont = new Font("Segoe UI", 8.5f);

        var kindSize = TextRenderer.MeasureText(kindLabel, kindFont);
        var headerY = rect.Y + Scale(11);
        TextRenderer.DrawText(g, kindLabel, kindFont, new Point(rect.X + pad, headerY), kindColor, TextFormatFlags.NoPrefix);

        var provText = hideContent
            ? (_store.ScreenSharingActive ? "Hidden while screen sharing" : "Private result")
            : item.SourceLabel;
        var provRect = new Rectangle(rect.X + pad + kindSize.Width + Scale(6), headerY,
            Math.Max(Scale(40), dismissRect.Left - Scale(8) - (rect.X + pad + kindSize.Width + Scale(6))), Scale(18));
        TextRenderer.DrawText(g, provText, metaFont, provRect, Color.FromArgb(167, 176, 192),
            TextFormatFlags.Left | TextFormatFlags.EndEllipsis | TextFormatFlags.NoPrefix);

        var contentX = rect.X + pad;
        var contentW = rect.Width - pad * 2;
        var contentY = rect.Y + Scale(32);

        if (hideContent)
        {
            TextRenderer.DrawText(g, "Open Lila to view.", bodyFont,
                new Rectangle(contentX, contentY, contentW, Scale(24)),
                Color.FromArgb(215, 222, 235),
                TextFormatFlags.Left | TextFormatFlags.EndEllipsis | TextFormatFlags.NoPrefix);
            return dismissRect;
        }

        if (item.Kind == ScreenItemKind.Task)
        {
            TextRenderer.DrawText(g, item.Title, titleFont,
                new Rectangle(contentX, contentY, contentW, Scale(20)),
                Color.FromArgb(245, 247, 250),
                TextFormatFlags.Left | TextFormatFlags.EndEllipsis | TextFormatFlags.NoPrefix);

            string[] steps =
            [
                "Owner approved",
                "Authorised (Harness)",
                "Dispatched",
                "Completed",
                item.TaskStage == ScreenTaskStage.CompletedNotVerified ? "Verified: pending" : "Verified"
            ];
            var stepY = contentY + Scale(22);
            var stepH = Scale(16);
            var doneCount = (int)item.TaskStage;
            for (var s = 0; s < steps.Length; s++)
            {
                var isDone = s < doneCount;
                var isCur = s == doneCount;
                var dotRect = new Rectangle(contentX + Scale(2), stepY + Scale(4), Scale(8), Scale(8));
                if (isDone)
                {
                    using var dot = new SolidBrush(Mint);
                    g.FillEllipse(dot, dotRect);
                }
                else if (isCur)
                {
                    using var dot = new SolidBrush(Amber);
                    g.FillEllipse(dot, dotRect);
                }
                else
                {
                    using var ring = new Pen(Color.FromArgb(95, 255, 255, 255), 1.3f);
                    g.DrawEllipse(ring, dotRect);
                }
                TextRenderer.DrawText(g, steps[s], metaFont,
                    new Rectangle(contentX + Scale(16), stepY, contentW - Scale(16), stepH),
                    (isDone || isCur) ? Color.FromArgb(245, 247, 250) : Color.FromArgb(150, 160, 178),
                    TextFormatFlags.Left | TextFormatFlags.EndEllipsis | TextFormatFlags.NoPrefix);
                stepY += stepH;
            }
            TextRenderer.DrawText(g, item.TaskStatusText, metaFont,
                new Rectangle(contentX, rect.Bottom - Scale(24), contentW, Scale(18)),
                Color.FromArgb(167, 176, 192),
                TextFormatFlags.Left | TextFormatFlags.EndEllipsis | TextFormatFlags.NoPrefix);
            return dismissRect;
        }

        if (item.Kind is ScreenItemKind.Image or ScreenItemKind.Video)
        {
            var thumbRect = new Rectangle(contentX, contentY, Scale(108), Scale(60));
            using (var thumbPath = RoundedRect(thumbRect, Scale(10)))
            {
                var drewImage = false;
                if (!string.IsNullOrWhiteSpace(item.MediaPath) && File.Exists(item.MediaPath))
                {
                    try
                    {
                        using var fs = new FileStream(item.MediaPath, FileMode.Open, FileAccess.Read, FileShare.ReadWrite);
                        using var img = Image.FromStream(fs);
                        g.SetClip(thumbPath);
                        g.DrawImage(img, thumbRect);
                        g.ResetClip();
                        drewImage = true;
                    }
                    catch
                    {
                        g.ResetClip();
                    }
                }
                if (!drewImage)
                {
                    using var thumbBg = new LinearGradientBrush(thumbRect, Color.FromArgb(43, 58, 102), Violet, 45f);
                    g.FillPath(thumbBg, thumbPath);
                }
            }
            if (item.Kind == ScreenItemKind.Video)
            {
                // Play triangle
                var cx = thumbRect.Left + thumbRect.Width / 2;
                var cy = thumbRect.Top + thumbRect.Height / 2;
                var tri = new[]
                {
                    new Point(cx - Scale(5), cy - Scale(8)),
                    new Point(cx + Scale(9), cy),
                    new Point(cx - Scale(5), cy + Scale(8))
                };
                using var triBrush = new SolidBrush(Color.White);
                g.FillPolygon(triBrush, tri);
                if (!string.IsNullOrWhiteSpace(item.DurationLabel))
                {
                    TextRenderer.DrawText(g, item.DurationLabel, metaFont,
                        new Rectangle(thumbRect.Right - Scale(42), thumbRect.Bottom - Scale(18), Scale(38), Scale(16)),
                        Color.White, TextFormatFlags.Right | TextFormatFlags.NoPrefix);
                }
            }
            var tx = thumbRect.Right + Scale(12);
            var tw = Math.Max(Scale(80), rect.Right - pad - tx);
            TextRenderer.DrawText(g, item.Title, titleFont,
                new Rectangle(tx, contentY, tw, Scale(22)),
                Color.FromArgb(245, 247, 250),
                TextFormatFlags.Left | TextFormatFlags.EndEllipsis | TextFormatFlags.NoPrefix);
            var mediaNote = item.Kind == ScreenItemKind.Video
                ? "Poster only. Plays in Lila, never in the overlay."
                : "Open Lila for the full image.";
            TextRenderer.DrawText(g, mediaNote, bodyFont,
                new Rectangle(tx, contentY + Scale(24), tw, Scale(36)),
                Color.FromArgb(182, 190, 206),
                TextFormatFlags.Left | TextFormatFlags.WordBreak | TextFormatFlags.NoPrefix);
            return dismissRect;
        }

        // Research card
        TextRenderer.DrawText(g, item.Title, titleFont,
            new Rectangle(contentX, contentY, contentW, Scale(20)),
            Color.FromArgb(245, 247, 250),
            TextFormatFlags.Left | TextFormatFlags.EndEllipsis | TextFormatFlags.NoPrefix);
        TextRenderer.DrawText(g, item.Summary, bodyFont,
            new Rectangle(contentX, contentY + Scale(22), contentW, Scale(38)),
            Color.FromArgb(215, 222, 236),
            TextFormatFlags.Left | TextFormatFlags.WordBreak | TextFormatFlags.EndEllipsis | TextFormatFlags.NoPrefix);
        var sourceChip = item.Sources.Count > 0
            ? $"{item.Sources.Count} {(item.Sources.Count == 1 ? "source" : "sources")}"
            : "Unverified summary";
        TextRenderer.DrawText(g, sourceChip, metaFont,
            new Rectangle(contentX, rect.Bottom - Scale(24), contentW, Scale(18)),
            Color.FromArgb(167, 176, 192),
            TextFormatFlags.Left | TextFormatFlags.NoPrefix);

        return dismissRect;
    }

    private Color StateAccentColor() => _visualState switch
    {
        IslandVisualState.Listening => Indigo,
        IslandVisualState.Thinking => Violet,
        IslandVisualState.Speaking => SpeakingPink,
        IslandVisualState.Attention or IslandVisualState.Interrupted => Amber,
        IslandVisualState.Muted or IslandVisualState.Error => Coral,
        _ => Indigo
    };

    private static GraphicsPath RoundedRect(RectangleF bounds, float radius)
    {
        var path = new GraphicsPath();
        var d = Math.Min(radius * 2, Math.Min(bounds.Width, bounds.Height));
        if (d <= 0)
        {
            path.AddRectangle(bounds);
            return path;
        }
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

    private void PresentLayeredBitmap(Bitmap bitmap, Rectangle screenBounds)
    {
        var screenDc = GetDC(IntPtr.Zero);
        var memDc = CreateCompatibleDC(screenDc);
        var hBitmap = bitmap.GetHbitmap(Color.FromArgb(0));
        var oldBitmap = SelectObject(memDc, hBitmap);
        try
        {
            var size = new NativeSize(bitmap.Width, bitmap.Height);
            var srcPoint = new NativePoint(0, 0);
            var dstPoint = new NativePoint(screenBounds.Left, screenBounds.Top);
            var blend = new BlendFunction
            {
                BlendOp = AcSrcOver,
                BlendFlags = 0,
                SourceConstantAlpha = 255,
                AlphaFormat = AcSrcAlpha
            };
            UpdateLayeredWindow(Handle, screenDc, ref dstPoint, ref size, memDc, ref srcPoint, 0, ref blend, UlwAlpha);
        }
        finally
        {
            SelectObject(memDc, oldBitmap);
            DeleteObject(hBitmap);
            DeleteDC(memDc);
            ReleaseDC(IntPtr.Zero, screenDc);
        }
    }

    protected override void Dispose(bool disposing)
    {
        if (disposing)
        {
            _store.Changed -= OnStoreChanged;
            _animTimer.Dispose();
        }
        base.Dispose(disposing);
    }

    [StructLayout(LayoutKind.Sequential)]
    private struct NativePoint(int x, int y)
    {
        public int X = x;
        public int Y = y;
    }

    [StructLayout(LayoutKind.Sequential)]
    private struct NativeSize(int cx, int cy)
    {
        public int Width = cx;
        public int Height = cy;
    }

    [StructLayout(LayoutKind.Sequential, Pack = 1)]
    private struct BlendFunction
    {
        public byte BlendOp;
        public byte BlendFlags;
        public byte SourceConstantAlpha;
        public byte AlphaFormat;
    }

    [DllImport("user32.dll", SetLastError = true)]
    private static extern bool UpdateLayeredWindow(IntPtr hwnd, IntPtr hdcDst, ref NativePoint pptDst,
        ref NativeSize psize, IntPtr hdcSrc, ref NativePoint pptSrc, int crKey,
        ref BlendFunction pblend, int dwFlags);

    [DllImport("user32.dll")]
    private static extern IntPtr GetDC(IntPtr hWnd);

    [DllImport("user32.dll")]
    private static extern int ReleaseDC(IntPtr hWnd, IntPtr hDC);

    [DllImport("gdi32.dll")]
    private static extern IntPtr CreateCompatibleDC(IntPtr hdc);

    [DllImport("gdi32.dll")]
    private static extern bool DeleteDC(IntPtr hdc);

    [DllImport("gdi32.dll")]
    private static extern IntPtr SelectObject(IntPtr hdc, IntPtr hObject);

    [DllImport("gdi32.dll")]
    private static extern bool DeleteObject(IntPtr hObject);
}
