using System.Drawing.Drawing2D;
using System.Drawing.Imaging;
using System.Security.Cryptography;

namespace Lila.Desktop.Island;

/// <summary>
/// Transient in-memory representation of an explicit owner-selected screen region or full screen.
/// Never persisted to disk by default; zeroes its PNG buffer on Dispose (M7 privacy contract).
/// </summary>
public sealed class TransientScreenHighlight : IDisposable
{
    private byte[] _pngBytes;
    private bool _disposed;

    public TransientScreenHighlight(
        byte[] pngBytes,
        Rectangle screenBounds,
        bool isFullScreen,
        string? comment,
        bool sendImmediately,
        TimeSpan? ttl = null)
    {
        _pngBytes = pngBytes;
        Sha256Hex = Convert.ToHexString(SHA256.HashData(pngBytes)).ToLowerInvariant();
        ScreenBounds = screenBounds;
        IsFullScreen = isFullScreen;
        Comment = string.IsNullOrWhiteSpace(comment) ? null : comment.Trim();
        SendImmediately = sendImmediately;
        CapturedUtc = DateTimeOffset.UtcNow;
        ExpiresUtc = CapturedUtc.Add(ttl ?? TimeSpan.FromMinutes(5));
    }

    public string Sha256Hex { get; }
    public Rectangle ScreenBounds { get; }
    public bool IsFullScreen { get; }
    public string? Comment { get; set; }
    public bool SendImmediately { get; }
    public string? DaemonCaptureId { get; set; }
    public DateTimeOffset CapturedUtc { get; }
    public DateTimeOffset ExpiresUtc { get; }
    public bool IsExpired => _disposed || DateTimeOffset.UtcNow >= ExpiresUtc;
    public int ByteLength => _disposed ? 0 : _pngBytes.Length;
    public string ToBase64Png() => _disposed ? string.Empty : Convert.ToBase64String(_pngBytes);

    public string ScopeSummary => IsFullScreen
        ? $"Full screen ({ScreenBounds.Width}×{ScreenBounds.Height})"
        : $"Region {ScreenBounds.Width}×{ScreenBounds.Height} at ({ScreenBounds.X},{ScreenBounds.Y})";

    public string BuildPromptContext(string? userMessage)
    {
        var prompt = string.IsNullOrWhiteSpace(userMessage)
            ? (Comment ?? "Please inspect this highlighted screen area and help me with it.")
            : userMessage.Trim();
        var notePart = !string.IsNullOrWhiteSpace(Comment) &&
                       !string.Equals(Comment, prompt, StringComparison.Ordinal)
            ? $" | Highlight note: {Comment}"
            : string.Empty;
        return $"[Screen Highlight: {ScopeSummary}, sha256={Sha256Hex[..12]}, size={ByteLength}B{notePart}]\n{prompt}";
    }

    public void Dispose()
    {
        if (_disposed)
            return;
        _disposed = true;
        CryptographicOperations.ZeroMemory(_pngBytes);
        _pngBytes = [];
    }
}

/// <summary>
/// Interactive screen highlighter and commentor overlay (like Claude / Antigravity region commentor).
/// Allows the owner to drag-select any region of the screen or click "Full Screen",
/// write an inline comment/instruction pinned to that selection, and either ask Lila immediately
/// or attach it to the Lila Island composer.
/// </summary>
public sealed class ScreenHighlightOverlayForm : Form
{
    private readonly Screen _screen;
    private readonly Bitmap _snapshot;
    private readonly Panel _topBar = new();
    private readonly Label _topHint = new();
    private readonly Button _fullScreenBtn = new();
    private readonly Button _cancelTopBtn = new();

    private readonly Panel _commentCard = new();
    private readonly Label _scopeLabel = new();
    private readonly Panel _commentInputPill = new();
    private readonly TextBox _commentInput = new();
    private readonly Button _askLilaBtn = new();
    private readonly Button _attachBtn = new();
    private readonly Button _reselectBtn = new();

    private bool _isDragging;
    private Point _dragStart;
    private Rectangle _selectedClientRect = Rectangle.Empty;
    private bool _isFullScreenSelection;

    public TransientScreenHighlight? ResultHighlight { get; private set; }

    public ScreenHighlightOverlayForm(Screen screen, Bitmap snapshot)
    {
        _screen = screen;
        _snapshot = snapshot;

        FormBorderStyle = FormBorderStyle.None;
        StartPosition = FormStartPosition.Manual;
        Bounds = screen.Bounds;
        TopMost = true;
        ShowInTaskbar = false;
        DoubleBuffered = true;
        KeyPreview = true;
        Cursor = Cursors.Cross;
        BackColor = Color.Black;

        BuildTopToolbar();
        BuildCommentorCard();
    }

    public static Bitmap CaptureScreenSnapshot(Screen screen)
    {
        var bounds = screen.Bounds;
        var bitmap = new Bitmap(Math.Max(1, bounds.Width), Math.Max(1, bounds.Height), PixelFormat.Format32bppArgb);
        using var g = Graphics.FromImage(bitmap);
        g.CopyFromScreen(bounds.Location, Point.Empty, bounds.Size, CopyPixelOperation.SourceCopy);
        return bitmap;
    }

    private int Scale(int logical) => (int)Math.Round(logical * (DeviceDpi / 96f));

    private void BuildTopToolbar()
    {
        var barH = Scale(46);
        var radius = barH / 2;
        _topBar.BackColor = Color.FromArgb(24, 28, 38);
        _topBar.Cursor = Cursors.Default;
        _topBar.Paint += (_, e) =>
        {
            e.Graphics.SmoothingMode = SmoothingMode.AntiAlias;
            using var path = CreateRoundedPath(new RectangleF(0.5f, 0.5f, _topBar.Width - 1, _topBar.Height - 1), radius);
            var sheenRect = new Rectangle(0, 0, _topBar.Width, Math.Max(10, _topBar.Height / 2));
            using (var sheen = new LinearGradientBrush(
                       sheenRect,
                       Color.FromArgb(24, 255, 255, 255),
                       Color.FromArgb(0, 255, 255, 255),
                       LinearGradientMode.Vertical))
            {
                e.Graphics.SetClip(path);
                e.Graphics.FillRectangle(sheen, sheenRect);
                e.Graphics.ResetClip();
            }
            using var pen = new Pen(Color.FromArgb(120, 124, 156, 255), 1.3f);
            e.Graphics.DrawPath(pen, path);
        };
        _topBar.SizeChanged += (_, _) => ApplyRoundedRegion(_topBar, _topBar.Height / 2);

        _topHint.Text = "Drag to highlight any area for Lila";
        _topHint.AutoSize = true;
        _topHint.ForeColor = Color.FromArgb(245, 247, 250);
        _topHint.Font = new Font("Segoe UI Semibold", 9.25f, FontStyle.Bold);

        StylePillButton(_fullScreenBtn, "Full Screen", primary: true);
        _fullScreenBtn.Click += (_, _) => SelectFullScreen();

        StylePillButton(_cancelTopBtn, "Esc · Cancel", primary: false);
        _cancelTopBtn.Click += (_, _) => CancelAndClose();

        var hintSize = TextRenderer.MeasureText(_topHint.Text, _topHint.Font);
        var padX = Scale(18);
        var gap = Scale(12);
        var btnGap = Scale(8);
        var btnH = Scale(30);
        var fullBtnW = Math.Max(Scale(96), TextRenderer.MeasureText("Full Screen", _fullScreenBtn.Font).Width + Scale(22));
        var cancelBtnW = Math.Max(Scale(92), TextRenderer.MeasureText("Esc · Cancel", _cancelTopBtn.Font).Width + Scale(22));
        var btnY = (barH - btnH) / 2;

        _topHint.Location = new Point(padX, (barH - hintSize.Height) / 2);
        var fullX = padX + hintSize.Width + gap;
        _fullScreenBtn.Bounds = new Rectangle(fullX, btnY, fullBtnW, btnH);
        var cancelX = fullX + fullBtnW + btnGap;
        _cancelTopBtn.Bounds = new Rectangle(cancelX, btnY, cancelBtnW, btnH);
        _topBar.Size = new Size(cancelX + cancelBtnW + padX, barH);

        _topBar.Controls.Add(_topHint);
        _topBar.Controls.Add(_fullScreenBtn);
        _topBar.Controls.Add(_cancelTopBtn);
        Controls.Add(_topBar);
        PositionTopBar();
    }

    private void BuildCommentorCard()
    {
        var cardW = Scale(430);
        var pad = Scale(16);
        var radius = Scale(16);
        _commentCard.BackColor = Color.FromArgb(22, 26, 36);
        _commentCard.Visible = false;
        _commentCard.Cursor = Cursors.Default;
        _commentCard.Paint += (_, e) =>
        {
            e.Graphics.SmoothingMode = SmoothingMode.AntiAlias;
            using var path = CreateRoundedPath(
                new RectangleF(0.5f, 0.5f, _commentCard.Width - 1, _commentCard.Height - 1), radius);
            var sheenRect = new Rectangle(0, 0, _commentCard.Width, Math.Max(14, _commentCard.Height / 3));
            using (var sheen = new LinearGradientBrush(
                       sheenRect,
                       Color.FromArgb(22, 255, 255, 255),
                       Color.FromArgb(0, 255, 255, 255),
                       LinearGradientMode.Vertical))
            {
                e.Graphics.SetClip(path);
                e.Graphics.FillRectangle(sheen, sheenRect);
                e.Graphics.ResetClip();
            }
            using var border = new LinearGradientBrush(
                _commentCard.ClientRectangle,
                Color.FromArgb(200, 124, 156, 255),
                Color.FromArgb(200, 176, 108, 255),
                35f);
            using var pen = new Pen(border, 1.5f);
            e.Graphics.DrawPath(pen, path);
        };
        _commentCard.SizeChanged += (_, _) => ApplyRoundedRegion(_commentCard, radius);

        _scopeLabel.AutoSize = true;
        _scopeLabel.Location = new Point(pad, Scale(12));
        _scopeLabel.ForeColor = Color.FromArgb(165, 188, 255);
        _scopeLabel.Font = new Font("Segoe UI Semibold", 8.5f, FontStyle.Bold);
        _scopeLabel.Text = "Highlighted region";

        var pillY = Scale(38);
        var pillH = Scale(36);
        var pillW = cardW - pad * 2;
        _commentInputPill.Bounds = new Rectangle(pad, pillY, pillW, pillH);
        _commentInputPill.BackColor = Color.FromArgb(16, 19, 28);
        _commentInputPill.Paint += (_, e) =>
        {
            e.Graphics.SmoothingMode = SmoothingMode.AntiAlias;
            using var p = CreateRoundedPath(
                new RectangleF(0.5f, 0.5f, _commentInputPill.Width - 1, _commentInputPill.Height - 1), pillH / 2f);
            using var fill = new SolidBrush(Color.FromArgb(16, 19, 28));
            e.Graphics.FillPath(fill, p);
            var borderClr = _commentInput.Focused
                ? Color.FromArgb(190, 124, 156, 255)
                : Color.FromArgb(54, 255, 255, 255);
            using var pen = new Pen(borderClr, 1.2f);
            e.Graphics.DrawPath(pen, p);
        };
        _commentInputPill.SizeChanged += (_, _) => ApplyRoundedRegion(_commentInputPill, _commentInputPill.Height / 2);
        ApplyRoundedRegion(_commentInputPill, pillH / 2);

        _commentInput.BackColor = Color.FromArgb(16, 19, 28);
        _commentInput.ForeColor = Color.FromArgb(245, 247, 250);
        _commentInput.BorderStyle = BorderStyle.None;
        _commentInput.Font = new Font("Segoe UI", 9.5f);
        _commentInput.PlaceholderText = "Ask Lila to work on this, comment, or search…";
        var inputPadX = Scale(14);
        var textH = _commentInput.PreferredHeight;
        _commentInput.Bounds = new Rectangle(inputPadX, Math.Max(2, (pillH - textH) / 2), pillW - inputPadX * 2, textH);
        _commentInput.GotFocus += (_, _) => _commentInputPill.Invalidate();
        _commentInput.LostFocus += (_, _) => _commentInputPill.Invalidate();
        _commentInput.KeyDown += (_, e) =>
        {
            if (e.KeyCode == Keys.Enter && !e.Shift)
            {
                e.SuppressKeyPress = true;
                CompleteCapture(sendImmediately: true);
            }
            else if (e.KeyCode == Keys.Escape)
            {
                e.SuppressKeyPress = true;
                CancelAndClose();
            }
        };
        _commentInputPill.Controls.Add(_commentInput);

        var btnY = pillY + pillH + Scale(12);
        var btnH = Scale(30);
        StylePillButton(_reselectBtn, "Reselect", primary: false);
        _reselectBtn.Bounds = new Rectangle(pad, btnY, Scale(84), btnH);
        _reselectBtn.Click += (_, _) => ClearSelection();

        var askW = Scale(108);
        var attachW = Scale(116);
        var gap = Scale(8);
        StylePillButton(_askLilaBtn, "Ask Lila ↵", primary: true);
        _askLilaBtn.Bounds = new Rectangle(cardW - pad - askW, btnY, askW, btnH);
        _askLilaBtn.Click += (_, _) => CompleteCapture(sendImmediately: true);

        StylePillButton(_attachBtn, "Attach to chat", primary: false);
        _attachBtn.Bounds = new Rectangle(_askLilaBtn.Left - gap - attachW, btnY, attachW, btnH);
        _attachBtn.Click += (_, _) => CompleteCapture(sendImmediately: false);

        _commentCard.Size = new Size(cardW, btnY + btnH + Scale(14));
        _commentCard.Controls.Add(_scopeLabel);
        _commentCard.Controls.Add(_commentInputPill);
        _commentCard.Controls.Add(_reselectBtn);
        _commentCard.Controls.Add(_attachBtn);
        _commentCard.Controls.Add(_askLilaBtn);
        Controls.Add(_commentCard);
    }

    private void PositionTopBar()
    {
        _topBar.Location = new Point(Math.Max(12, (ClientSize.Width - _topBar.Width) / 2), Scale(18));
        ApplyRoundedRegion(_topBar, _topBar.Height / 2);
    }

    protected override void OnResize(EventArgs e)
    {
        base.OnResize(e);
        PositionTopBar();
    }

    protected override void OnKeyDown(KeyEventArgs e)
    {
        base.OnKeyDown(e);
        if (e.KeyCode == Keys.Escape)
        {
            e.Handled = true;
            CancelAndClose();
        }
    }

    protected override void OnMouseDown(MouseEventArgs e)
    {
        base.OnMouseDown(e);
        if (e.Button == MouseButtons.Right)
        {
            CancelAndClose();
            return;
        }
        if (e.Button != MouseButtons.Left)
            return;

        _isDragging = true;
        _isFullScreenSelection = false;
        _dragStart = e.Location;
        _selectedClientRect = new Rectangle(e.Location, Size.Empty);
        _commentCard.Visible = false;
        Invalidate();
    }

    protected override void OnMouseMove(MouseEventArgs e)
    {
        base.OnMouseMove(e);
        if (!_isDragging)
            return;
        _selectedClientRect = NormalizeRectangle(_dragStart, e.Location);
        Invalidate();
    }

    protected override void OnMouseUp(MouseEventArgs e)
    {
        base.OnMouseUp(e);
        if (!_isDragging || e.Button != MouseButtons.Left)
            return;
        _isDragging = false;
        _selectedClientRect = NormalizeRectangle(_dragStart, e.Location);
        if (_selectedClientRect.Width < 14 || _selectedClientRect.Height < 14)
        {
            _selectedClientRect = Rectangle.Empty;
            Invalidate();
            return;
        }
        ShowCommentorCardForSelection();
        Invalidate();
    }

    private void SelectFullScreen()
    {
        _isDragging = false;
        _isFullScreenSelection = true;
        _selectedClientRect = new Rectangle(0, 0, ClientSize.Width, ClientSize.Height);
        ShowCommentorCardForSelection();
        Invalidate();
    }

    private void ClearSelection()
    {
        _isDragging = false;
        _isFullScreenSelection = false;
        _selectedClientRect = Rectangle.Empty;
        _commentCard.Visible = false;
        Cursor = Cursors.Cross;
        Invalidate();
    }

    private void ShowCommentorCardForSelection()
    {
        _scopeLabel.Text = _isFullScreenSelection
            ? $"Full screen highlighted ({_selectedClientRect.Width}×{_selectedClientRect.Height}) · Transient"
            : $"Highlighted region ({_selectedClientRect.Width}×{_selectedClientRect.Height}) · Transient";

        var cardW = _commentCard.Width;
        var cardH = _commentCard.Height;
        var x = Math.Clamp(
            _selectedClientRect.Left + (_selectedClientRect.Width - cardW) / 2,
            12,
            Math.Max(12, ClientSize.Width - cardW - 12));

        int y;
        if (_isFullScreenSelection)
            y = Math.Max(Scale(84), (ClientSize.Height - cardH) / 3);
        else if (_selectedClientRect.Bottom + Scale(12) + cardH <= ClientSize.Height - Scale(12))
            y = _selectedClientRect.Bottom + Scale(12);
        else if (_selectedClientRect.Top - Scale(12) - cardH >= Scale(76))
            y = _selectedClientRect.Top - Scale(12) - cardH;
        else
            y = Math.Max(Scale(84), (ClientSize.Height - cardH) / 3);

        _commentCard.Location = new Point(x, y);
        ApplyRoundedRegion(_commentCard, Scale(16));
        _commentCard.Visible = true;
        _commentCard.BringToFront();
        _topBar.BringToFront();
        _commentInput.Focus();
    }

    private void CompleteCapture(bool sendImmediately)
    {
        if (_selectedClientRect.Width < 8 || _selectedClientRect.Height < 8)
            return;

        var cropRect = Rectangle.Intersect(_selectedClientRect, new Rectangle(0, 0, _snapshot.Width, _snapshot.Height));
        if (cropRect.Width <= 0 || cropRect.Height <= 0)
            return;

        using var cropped = _snapshot.Clone(cropRect, PixelFormat.Format32bppArgb);
        using var ms = new MemoryStream();
        cropped.Save(ms, ImageFormat.Png);
        var pngBytes = ms.ToArray();

        var desktopBounds = new Rectangle(
            _screen.Bounds.Left + cropRect.Left,
            _screen.Bounds.Top + cropRect.Top,
            cropRect.Width,
            cropRect.Height);

        ResultHighlight = new TransientScreenHighlight(
            pngBytes,
            desktopBounds,
            _isFullScreenSelection,
            _commentInput.Text,
            sendImmediately);

        DialogResult = DialogResult.OK;
        Close();
    }

    private void CancelAndClose()
    {
        ResultHighlight?.Dispose();
        ResultHighlight = null;
        DialogResult = DialogResult.Cancel;
        Close();
    }

    protected override void OnPaint(PaintEventArgs e)
    {
        var g = e.Graphics;
        g.SmoothingMode = SmoothingMode.AntiAlias;

        // 1. Draw the frozen screen capture
        g.DrawImage(_snapshot, ClientRectangle);

        // 2. Dim unselected areas
        using var dimBrush = new SolidBrush(Color.FromArgb(138, 10, 13, 20));
        if (_selectedClientRect.Width > 0 && _selectedClientRect.Height > 0)
        {
            using var dimRegion = new Region(ClientRectangle);
            dimRegion.Exclude(_selectedClientRect);
            g.FillRegion(dimBrush, dimRegion);

            // 3. Draw glowing selection frame + corner handles
            using var grad = new LinearGradientBrush(
                _selectedClientRect,
                Color.FromArgb(245, 124, 156, 255),
                Color.FromArgb(245, 176, 108, 255),
                35f);
            using var borderPen = new Pen(grad, 2.2f);
            g.DrawRectangle(borderPen, _selectedClientRect);

            DrawCornerHandles(g, _selectedClientRect);
            DrawDimensionBadge(g, _selectedClientRect);
        }
        else
        {
            g.FillRectangle(dimBrush, ClientRectangle);
        }
    }

    private static void DrawCornerHandles(Graphics g, Rectangle r)
    {
        using var handleBrush = new SolidBrush(Color.FromArgb(245, 247, 250));
        using var outlinePen = new Pen(Color.FromArgb(220, 124, 156, 255), 1.5f);
        ReadOnlySpan<Point> points =
        [
            new Point(r.Left, r.Top),
            new Point(r.Right, r.Top),
            new Point(r.Left, r.Bottom),
            new Point(r.Right, r.Bottom)
        ];
        foreach (var pt in points)
        {
            var box = new Rectangle(pt.X - 4, pt.Y - 4, 8, 8);
            g.FillEllipse(handleBrush, box);
            g.DrawEllipse(outlinePen, box);
        }
    }

    private static void DrawDimensionBadge(Graphics g, Rectangle r)
    {
        var text = $"{r.Width} × {r.Height}";
        using var font = new Font("Segoe UI Semibold", 8f, FontStyle.Bold);
        var size = g.MeasureString(text, font);
        var badgeRect = new RectangleF(
            r.Left,
            r.Top >= 26 ? r.Top - 24 : r.Top + 6,
            size.Width + 14,
            20);
        using var path = CreateRoundedPath(badgeRect, 6);
        using var fill = new SolidBrush(Color.FromArgb(225, 20, 24, 34));
        using var pen = new Pen(Color.FromArgb(120, 124, 156, 255), 1f);
        g.FillPath(fill, path);
        g.DrawPath(pen, path);
        using var textBrush = new SolidBrush(Color.FromArgb(245, 247, 250));
        g.DrawString(text, font, textBrush, badgeRect.X + 7, badgeRect.Y + 2);
    }

    private static Rectangle NormalizeRectangle(Point a, Point b) =>
        new(Math.Min(a.X, b.X), Math.Min(a.Y, b.Y), Math.Abs(a.X - b.X), Math.Abs(a.Y - b.Y));

    private static void StylePillButton(Button button, string text, bool primary)
    {
        button.Text = text;
        button.FlatStyle = FlatStyle.Flat;
        button.FlatAppearance.BorderSize = 1;
        button.FlatAppearance.BorderColor = primary
            ? Color.FromArgb(165, 148, 255)
            : Color.FromArgb(55, 255, 255, 255);
        button.BackColor = primary
            ? Color.FromArgb(126, 118, 252)
            : Color.FromArgb(36, 41, 54);
        button.ForeColor = Color.FromArgb(245, 247, 250);
        button.Font = new Font("Segoe UI Semibold", 8.5f, primary ? FontStyle.Bold : FontStyle.Regular);
        button.Cursor = Cursors.Hand;
    }

    private static void ApplyRoundedRegion(Control control, float radius)
    {
        if (control.Width <= 0 || control.Height <= 0)
            return;
        var prev = control.Region;
        using var path = CreateRoundedPath(new RectangleF(0, 0, control.Width, control.Height), radius);
        control.Region = new Region(path);
        prev?.Dispose();
    }

    private static GraphicsPath CreateRoundedPath(RectangleF bounds, float radius)
    {
        var path = new GraphicsPath();
        var d = Math.Min(radius * 2, Math.Min(bounds.Width, bounds.Height));
        if (d <= 0)
        {
            path.AddRectangle(bounds);
            return path;
        }
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
            _snapshot.Dispose();
        base.Dispose(disposing);
    }
}
