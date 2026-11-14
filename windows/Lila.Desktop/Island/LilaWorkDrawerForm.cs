using System.Diagnostics;
using System.Drawing.Drawing2D;
using Lila.Desktop.Core.Island;

namespace Lila.Desktop.Island;

public sealed record CapabilityCandidateView(
    string CapabilityId,
    string Version,
    string Status,
    bool TestsPassed,
    string Provenance,
    string TestSummary);

/// <summary>
/// Reusable "Lila's work" drawer displaying all typed ScreenItems (Research, Image, Video, Task)
/// with DPI-safe layout, real image preview, explicit click-to-play video launch (never autoplays),
/// and Screen Sharing privacy toggle.
/// </summary>
public sealed class LilaWorkDrawerForm : Form
{
    private static readonly Color SurfaceBg = Color.FromArgb(20, 24, 32);
    private static readonly Color CardBg = Color.FromArgb(28, 34, 48);
    private static readonly Color CardOpenBg = Color.FromArgb(35, 42, 60);
    private static readonly Color TextPrimary = Color.FromArgb(245, 247, 250);
    private static readonly Color TextMuted = Color.FromArgb(167, 176, 192);
    private static readonly Color AccentBlue = Color.FromArgb(124, 156, 255);
    private static readonly Color AccentAmber = Color.FromArgb(255, 184, 107);
    private static readonly Color AccentMint = Color.FromArgb(94, 224, 160);
    private static readonly Color AccentViolet = Color.FromArgb(192, 132, 252);

    private readonly LilaScreenStore _store;
    private readonly FlowLayoutPanel _list = new();
    private readonly Button _shareToggle = new();
    private readonly Label _shareWarning = new();
    private string? _selectedId;
    private Screen? _targetScreen;
    private int _catalogVersion;
    private IReadOnlyList<CapabilityCandidateView> _candidates = Array.Empty<CapabilityCandidateView>();

    public event Action? OpenMainIslandRequested;
    public event Action<string>? CancelTaskRequested;
    public event Action<string>? PromoteCapabilityRequested;
    public event Action<int>? RollbackCapabilityRequested;

    public LilaWorkDrawerForm(LilaScreenStore store)
    {
        _store = store ?? throw new ArgumentNullException(nameof(store));
        Text = "Lila’s work";
        FormBorderStyle = FormBorderStyle.None;
        StartPosition = FormStartPosition.Manual;
        ShowInTaskbar = false;
        TopMost = true;
        AutoScaleMode = AutoScaleMode.Dpi;
        BackColor = SurfaceBg;
        ForeColor = TextPrimary;
        Opacity = 0.96;
        KeyPreview = true;
        SetStyle(ControlStyles.OptimizedDoubleBuffer | ControlStyles.AllPaintingInWmPaint |
                 ControlStyles.UserPaint | ControlStyles.ResizeRedraw, true);

        BuildUi();
        _store.Changed += OnStoreChanged;
        KeyDown += (_, e) =>
        {
            if (e.KeyCode == Keys.Escape)
                Hide();
        };
    }

    public void OpenForItem(string? itemId, Screen? screen)
    {
        _targetScreen = screen ?? Screen.PrimaryScreen;
        if (!string.IsNullOrWhiteSpace(itemId))
            _selectedId = itemId;
        else if (_selectedId is null && _store.AllItems.Count > 0)
            _selectedId = _store.AllItems[0].Id;

        PositionOnScreen();
        RebuildList();
        if (!Visible)
            Show();
        BringToFront();
        Activate();
    }

    public void ToggleDrawer(Screen? screen)
    {
        if (Visible)
            Hide();
        else
            OpenForItem(null, screen);
    }

    protected override void OnHandleCreated(EventArgs e)
    {
        base.OnHandleCreated(e);
        IslandNativeWindow.ApplyGlassBackdrop(Handle);
        UpdateRegion();
    }

    protected override void OnResize(EventArgs e)
    {
        base.OnResize(e);
        UpdateRegion();
        ResizeCards();
    }

    protected override void OnPaint(PaintEventArgs e)
    {
        base.OnPaint(e);
        if (ClientSize.Width < 4 || ClientSize.Height < 4)
            return;
        e.Graphics.SmoothingMode = SmoothingMode.AntiAlias;
        var radius = Scale(16);
        using var path = RoundedRect(new RectangleF(0.5f, 0.5f, ClientSize.Width - 1, ClientSize.Height - 1), radius);
        var sheenRect = new Rectangle(0, 0, ClientSize.Width, Math.Max(20, ClientSize.Height / 4));
        using (var sheen = new LinearGradientBrush(sheenRect,
                   Color.FromArgb(20, 255, 255, 255),
                   Color.FromArgb(0, 255, 255, 255),
                   LinearGradientMode.Vertical))
        {
            e.Graphics.SetClip(path);
            e.Graphics.FillRectangle(sheen, sheenRect);
            e.Graphics.ResetClip();
        }
        using var border = new LinearGradientBrush(ClientRectangle,
            Color.FromArgb(95, 124, 156, 255),
            Color.FromArgb(45, 255, 255, 255), 35f);
        using var pen = new Pen(border, Math.Max(1.2f, Scale(1)));
        e.Graphics.DrawPath(pen, path);
    }

    private int Scale(int logical) => (int)Math.Round(logical * ((IsHandleCreated ? DeviceDpi : 96) / 96f));

    private void PositionOnScreen()
    {
        var screen = _targetScreen ?? Screen.PrimaryScreen ?? Screen.AllScreens.First();
        var work = screen.WorkingArea;
        var w = Math.Min(Scale(384), Math.Max(Scale(300), work.Width - Scale(24)));
        var h = Math.Min(Scale(580), Math.Max(Scale(340), work.Height - Scale(130)));
        Bounds = new Rectangle(work.Left + Scale(12), work.Top + Scale(72), w, h);
    }

    private void BuildUi()
    {
        var pad = Scale(14);
        var root = new TableLayoutPanel
        {
            Dock = DockStyle.Fill,
            BackColor = SurfaceBg,
            ColumnCount = 1,
            RowCount = 4,
            Padding = new Padding(pad, Scale(12), pad, Scale(12))
        };

        var titleFont = new Font("Segoe UI Semibold", 11.5f, FontStyle.Bold);
        var subFont = new Font("Segoe UI", 8.25f);
        var titleSize = TextRenderer.MeasureText("Lila’s work", titleFont);
        var subSize = TextRenderer.MeasureText("Recent results and tasks", subFont);
        var headerH = Math.Max(Scale(52), titleSize.Height + subSize.Height + Scale(8));
        var footerH = Scale(42);

        root.RowStyles.Add(new RowStyle(SizeType.Absolute, headerH));
        root.RowStyles.Add(new RowStyle(SizeType.AutoSize));
        root.RowStyles.Add(new RowStyle(SizeType.Percent, 100));
        root.RowStyles.Add(new RowStyle(SizeType.Absolute, footerH));

        var header = new Panel { Dock = DockStyle.Fill, BackColor = SurfaceBg };
        var title = new Label
        {
            Text = "Lila’s work",
            AutoSize = true,
            Location = new Point(0, 0),
            ForeColor = TextPrimary,
            BackColor = SurfaceBg,
            Font = titleFont
        };
        var sub = new Label
        {
            Text = "Recent results and tasks",
            AutoSize = true,
            Location = new Point(0, titleSize.Height + Scale(2)),
            ForeColor = TextMuted,
            BackColor = SurfaceBg,
            Font = subFont
        };

        _shareToggle.FlatStyle = FlatStyle.Flat;
        _shareToggle.FlatAppearance.BorderSize = 1;
        _shareToggle.Font = new Font("Segoe UI Semibold", 8f, FontStyle.Bold);
        _shareToggle.Cursor = Cursors.Hand;
        _shareToggle.Click += (_, _) => _store.SetScreenSharing(!_store.ScreenSharingActive);

        var closeBtn = new Button
        {
            Text = "‹",
            FlatStyle = FlatStyle.Flat,
            BackColor = CardBg,
            ForeColor = TextPrimary,
            Font = new Font("Segoe UI", 11f, FontStyle.Bold),
            Cursor = Cursors.Hand
        };
        closeBtn.FlatAppearance.BorderSize = 1;
        closeBtn.FlatAppearance.BorderColor = Color.FromArgb(46, 255, 255, 255);
        closeBtn.Click += (_, _) => Hide();

        void LayoutHeaderButtons()
        {
            var btnH = Scale(30);
            var closeW = Scale(30);
            var shareTextW = TextRenderer.MeasureText("Screen sharing", _shareToggle.Font).Width;
            var shareW = Math.Max(Scale(116), shareTextW + Scale(20));
            var btnTop = Scale(4);
            closeBtn.Bounds = new Rectangle(header.ClientSize.Width - closeW, btnTop, closeW, btnH);
            _shareToggle.Bounds = new Rectangle(closeBtn.Left - shareW - Scale(6), btnTop, shareW, btnH);
        }

        header.SizeChanged += (_, _) => LayoutHeaderButtons();
        header.Controls.Add(title);
        header.Controls.Add(sub);
        header.Controls.Add(_shareToggle);
        header.Controls.Add(closeBtn);

        _shareWarning.Text = "Screen sharing is on. Anything open here is visible to viewers.";
        _shareWarning.AutoSize = true;
        _shareWarning.ForeColor = AccentAmber;
        _shareWarning.BackColor = SurfaceBg;
        _shareWarning.Font = new Font("Segoe UI Semibold", 8.25f, FontStyle.Bold);
        _shareWarning.Padding = new Padding(0, Scale(2), 0, Scale(8));
        _shareWarning.Visible = false;

        _list.Dock = DockStyle.Fill;
        _list.FlowDirection = FlowDirection.TopDown;
        _list.WrapContents = false;
        _list.AutoScroll = true;
        _list.BackColor = SurfaceBg;
        _list.Margin = Padding.Empty;
        _list.Padding = new Padding(0, Scale(4), 0, Scale(4));

        var footer = new Panel { Dock = DockStyle.Fill, BackColor = SurfaceBg };
        var note = new Label
        {
            Text = "The overlay is a screen, not a chat.",
            AutoSize = true,
            ForeColor = TextMuted,
            BackColor = SurfaceBg,
            Font = new Font("Segoe UI", 8f)
        };
        var openLilaBtn = new Button
        {
            Text = "Open Lila",
            FlatStyle = FlatStyle.Flat,
            BackColor = AccentBlue,
            ForeColor = TextPrimary,
            Font = new Font("Segoe UI Semibold", 8.25f, FontStyle.Bold),
            Cursor = Cursors.Hand
        };
        openLilaBtn.FlatAppearance.BorderSize = 0;
        openLilaBtn.Click += (_, _) => OpenMainIslandRequested?.Invoke();

        var clearBtn = new Button
        {
            Text = "Clear",
            FlatStyle = FlatStyle.Flat,
            BackColor = CardBg,
            ForeColor = TextMuted,
            Font = new Font("Segoe UI", 8.25f),
            Cursor = Cursors.Hand
        };
        clearBtn.FlatAppearance.BorderSize = 1;
        clearBtn.FlatAppearance.BorderColor = Color.FromArgb(42, 255, 255, 255);
        clearBtn.Click += (_, _) => _store.ClearAll();

        footer.SizeChanged += (_, _) =>
        {
            var btnH = Scale(30);
            var openW = Math.Max(Scale(92), TextRenderer.MeasureText("Open Lila", openLilaBtn.Font).Width + Scale(22));
            var clearW = Math.Max(Scale(66), TextRenderer.MeasureText("Clear", clearBtn.Font).Width + Scale(20));
            var btnY = Math.Max(0, (footer.ClientSize.Height - btnH) / 2);
            openLilaBtn.Bounds = new Rectangle(footer.ClientSize.Width - openW, btnY, openW, btnH);
            clearBtn.Bounds = new Rectangle(openLilaBtn.Left - clearW - Scale(6), btnY, clearW, btnH);
            note.Location = new Point(0, Math.Max(0, (footer.ClientSize.Height - note.Height) / 2));
        };
        footer.Controls.Add(note);
        footer.Controls.Add(clearBtn);
        footer.Controls.Add(openLilaBtn);

        root.Controls.Add(header, 0, 0);
        root.Controls.Add(_shareWarning, 0, 1);
        root.Controls.Add(_list, 0, 2);
        root.Controls.Add(footer, 0, 3);
        Controls.Add(root);
        UpdateShareButtonStyle();
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
        UpdateShareButtonStyle();
        if (Visible)
            RebuildList();
    }

    private void UpdateShareButtonStyle()
    {
        var on = _store.ScreenSharingActive;
        _shareToggle.Text = on ? "Sharing: ON" : "Screen sharing";
        _shareToggle.BackColor = on ? Color.FromArgb(105, 92, 52) : CardBg;
        _shareToggle.ForeColor = on ? TextPrimary : TextMuted;
        _shareToggle.FlatAppearance.BorderColor = on ? AccentAmber : Color.FromArgb(52, 255, 255, 255);
        _shareWarning.Visible = on;
    }

    private void RebuildList()
    {
        _list.SuspendLayout();
        foreach (Control old in _list.Controls)
            old.Dispose();
        _list.Controls.Clear();

        if (_store.AllItems.Count == 0 && _candidates.Count == 0)
        {
            _list.Controls.Add(new Label
            {
                Text = "Nothing yet. Results and tasks from Lila will appear here.",
                AutoSize = true,
                ForeColor = TextMuted,
                BackColor = SurfaceBg,
                Font = new Font("Segoe UI", 8.75f),
                Margin = new Padding(2, Scale(8), 2, Scale(8))
            });
            _list.ResumeLayout(true);
            return;
        }

        var rowW = Math.Max(Scale(220), _list.ClientSize.Width - Scale(18));
        if (_candidates.Count > 0)
        {
            _list.Controls.Add(new Label
            {
                Text = $"CAPABILITY CANDIDATES · v{_catalogVersion}",
                AutoSize = true,
                ForeColor = AccentAmber,
                BackColor = SurfaceBg,
                Font = new Font("Segoe UI Semibold", 7.75f, FontStyle.Bold),
                Margin = new Padding(2, Scale(4), 2, Scale(6))
            });
            foreach (var candidate in _candidates)
                _list.Controls.Add(BuildCandidateCard(candidate, rowW));
        }
        foreach (var item in _store.AllItems)
        {
            var isOpen = item.Id == _selectedId;
            var card = new Panel
            {
                Width = rowW,
                AutoSize = true,
                MinimumSize = new Size(rowW, Scale(54)),
                MaximumSize = new Size(rowW, 0),
                BackColor = isOpen ? CardOpenBg : CardBg,
                Padding = new Padding(Scale(11), Scale(9), Scale(11), Scale(11)),
                Margin = new Padding(0, 0, 0, Scale(8)),
                Cursor = Cursors.Hand
            };
            card.Paint += (_, e) =>
            {
                e.Graphics.SmoothingMode = SmoothingMode.AntiAlias;
                using var p = RoundedRect(new RectangleF(0.5f, 0.5f, card.Width - 1, card.Height - 1), Scale(11));
                using var pen = new Pen(Color.FromArgb(isOpen ? 90 : 38, 255, 255, 255), 1f);
                e.Graphics.DrawPath(pen, p);
            };

            var stack = new FlowLayoutPanel
            {
                Dock = DockStyle.Top,
                AutoSize = true,
                FlowDirection = FlowDirection.TopDown,
                WrapContents = false,
                BackColor = card.BackColor,
                Width = rowW - Scale(22)
            };

            var kindText = item.IsPrivate ? "PRIVATE" : item.Kind.ToString().ToUpperInvariant();
            var kindColor = item.IsPrivate ? AccentAmber : item.Kind switch
            {
                ScreenItemKind.Research => AccentBlue,
                ScreenItemKind.Task => AccentMint,
                _ => AccentViolet
            };

            var headerLbl = new Label
            {
                Text = string.IsNullOrWhiteSpace(item.SourceLabel) ? kindText : $"{kindText} · {item.SourceLabel}",
                AutoSize = true,
                ForeColor = kindColor,
                BackColor = card.BackColor,
                Font = new Font("Segoe UI Semibold", 7.75f, FontStyle.Bold),
                MaximumSize = new Size(rowW - Scale(24), 0)
            };
            var titleLbl = new Label
            {
                Text = item.Title,
                AutoSize = true,
                ForeColor = TextPrimary,
                BackColor = card.BackColor,
                Font = new Font("Segoe UI Semibold", 9.25f, FontStyle.Bold),
                MaximumSize = new Size(rowW - Scale(24), 0),
                Margin = new Padding(0, Scale(2), 0, 0)
            };
            stack.Controls.Add(headerLbl);
            stack.Controls.Add(titleLbl);

            if (isOpen)
            {
                if (!string.IsNullOrWhiteSpace(item.Summary))
                {
                    stack.Controls.Add(new Label
                    {
                        Text = item.Summary,
                        AutoSize = true,
                        ForeColor = Color.FromArgb(220, 226, 238),
                        BackColor = card.BackColor,
                        Font = new Font("Segoe UI", 8.5f),
                        MaximumSize = new Size(rowW - Scale(24), 0),
                        Margin = new Padding(0, Scale(6), 0, 0)
                    });
                }

                if (item.Kind == ScreenItemKind.Image)
                {
                    AddImageDetail(stack, item, rowW - Scale(24), card.BackColor);
                }
                else if (item.Kind == ScreenItemKind.Video)
                {
                    AddVideoDetail(stack, item, rowW - Scale(24), card.BackColor);
                }
                else if (item.Kind == ScreenItemKind.Research && item.Sources.Count > 0)
                {
                    stack.Controls.Add(new Label
                    {
                        Text = "Sources:\n• " + string.Join("\n• ", item.Sources),
                        AutoSize = true,
                        ForeColor = TextMuted,
                        BackColor = card.BackColor,
                        Font = new Font("Segoe UI", 8f),
                        MaximumSize = new Size(rowW - Scale(24), 0),
                        Margin = new Padding(0, Scale(6), 0, 0)
                    });
                    stack.Controls.Add(new Label
                    {
                        Text = "Not verified. Check the sources before relying on it.",
                        AutoSize = true,
                        ForeColor = AccentAmber,
                        BackColor = card.BackColor,
                        Font = new Font("Segoe UI", 7.75f),
                        MaximumSize = new Size(rowW - Scale(24), 0),
                        Margin = new Padding(0, Scale(4), 0, 0)
                    });
                }
                else if (item.Kind == ScreenItemKind.Task)
                {
                    stack.Controls.Add(new Label
                    {
                        Text = item.TaskStatusText + (string.IsNullOrWhiteSpace(item.EvidenceSummary)
                            ? string.Empty
                            : "\n" + item.EvidenceSummary),
                        AutoSize = true,
                        ForeColor = item.TaskStage == ScreenTaskStage.Verified ? AccentMint : AccentAmber,
                        BackColor = card.BackColor,
                        Font = new Font("Segoe UI", 8f),
                        MaximumSize = new Size(rowW - Scale(24), 0),
                        Margin = new Padding(0, Scale(6), 0, 0)
                    });
                    if (item.TaskStage < ScreenTaskStage.CompletedNotVerified)
                    {
                        var cancelBtn = new Button
                        {
                            Text = "Cancel task",
                            AutoSize = true,
                            FlatStyle = FlatStyle.Flat,
                            ForeColor = AccentAmber,
                            BackColor = Color.FromArgb(46, 52, 70),
                            Font = new Font("Segoe UI Semibold", 8f, FontStyle.Bold),
                            Margin = new Padding(0, Scale(6), 0, 0),
                            Cursor = Cursors.Hand
                        };
                        cancelBtn.FlatAppearance.BorderColor = Color.FromArgb(95, 255, 184, 107);
                        cancelBtn.FlatAppearance.BorderSize = 1;
                        var taskId = item.Id;
                        cancelBtn.Click += (_, _) => CancelTaskRequested?.Invoke(taskId);
                        stack.Controls.Add(cancelBtn);
                    }
                }
            }

            void ToggleSelect(object? _, EventArgs __)
            {
                _selectedId = _selectedId == item.Id ? null : item.Id;
                RebuildList();
            }

            card.Click += ToggleSelect;
            stack.Click += ToggleSelect;
            foreach (Control child in stack.Controls)
            {
                if (child is not Button)
                    child.Click += ToggleSelect;
            }

            card.Controls.Add(stack);
            _list.Controls.Add(card);
        }
        _list.ResumeLayout(true);
    }

    public void SetCapabilityCandidates(int catalogVersion, IReadOnlyList<CapabilityCandidateView> candidates)
    {
        _catalogVersion = catalogVersion;
        _candidates = candidates ?? Array.Empty<CapabilityCandidateView>();
        if (!IsDisposed)
            RebuildList();
    }

    private Panel BuildCandidateCard(CapabilityCandidateView candidate, int rowW)
    {
        var card = new Panel
        {
            Width = rowW,
            AutoSize = true,
            MinimumSize = new Size(rowW, Scale(70)),
            MaximumSize = new Size(rowW, 0),
            BackColor = CardOpenBg,
            Padding = new Padding(Scale(11), Scale(9), Scale(11), Scale(11)),
            Margin = new Padding(0, 0, 0, Scale(8))
        };
        card.Paint += (_, e) =>
        {
            e.Graphics.SmoothingMode = SmoothingMode.AntiAlias;
            using var p = RoundedRect(new RectangleF(0.5f, 0.5f, card.Width - 1, card.Height - 1), Scale(11));
            using var pen = new Pen(Color.FromArgb(75, AccentAmber), 1f);
            e.Graphics.DrawPath(pen, p);
        };

        var stack = new FlowLayoutPanel
        {
            Dock = DockStyle.Top,
            AutoSize = true,
            FlowDirection = FlowDirection.TopDown,
            WrapContents = false,
            BackColor = card.BackColor,
            Width = rowW - Scale(22)
        };

        var statusLabel = candidate.TestsPassed ? "TESTS PASSED" : "TESTS FAILED";
        stack.Controls.Add(new Label
        {
            Text = $"CAPABILITY · v{candidate.Version} · {candidate.Status.ToUpperInvariant()} · {statusLabel}",
            AutoSize = true,
            ForeColor = candidate.TestsPassed ? Color.FromArgb(92, 200, 140) : AccentAmber,
            BackColor = card.BackColor,
            Font = new Font("Segoe UI Semibold", 7.25f, FontStyle.Bold),
            MaximumSize = new Size(rowW - Scale(24), 0)
        });
        stack.Controls.Add(new Label
        {
            Text = candidate.CapabilityId,
            AutoSize = true,
            ForeColor = TextPrimary,
            BackColor = card.BackColor,
            Font = new Font("Segoe UI Semibold", 9.25f, FontStyle.Bold),
            MaximumSize = new Size(rowW - Scale(24), 0),
            Margin = new Padding(0, Scale(2), 0, 0)
        });
        stack.Controls.Add(new Label
        {
            Text = $"Provenance: {candidate.Provenance}\n{candidate.TestSummary}",
            AutoSize = true,
            ForeColor = TextMuted,
            BackColor = card.BackColor,
            Font = new Font("Segoe UI", 8f),
            MaximumSize = new Size(rowW - Scale(24), 0),
            Margin = new Padding(0, Scale(4), 0, Scale(6))
        });

        var actions = new FlowLayoutPanel
        {
            AutoSize = true,
            FlowDirection = FlowDirection.LeftToRight,
            WrapContents = false,
            BackColor = card.BackColor,
            Margin = new Padding(0)
        };

        var canPromote = candidate.TestsPassed &&
            !string.Equals(candidate.Status, "promoted", StringComparison.OrdinalIgnoreCase);
        var promoteBtn = new Button
        {
            Name = $"PromoteBtn:{candidate.CapabilityId}",
            Text = "Promote",
            Enabled = canPromote,
            FlatStyle = FlatStyle.Flat,
            Size = new Size(Scale(84), Scale(26)),
            BackColor = canPromote ? AccentBlue : CardBg,
            ForeColor = TextPrimary,
            Font = new Font("Segoe UI Semibold", 8f, FontStyle.Bold),
            Cursor = canPromote ? Cursors.Hand : Cursors.Default,
            Margin = new Padding(0, 0, Scale(6), 0)
        };
        promoteBtn.FlatAppearance.BorderSize = 0;
        promoteBtn.Click += (_, _) => PromoteCapabilityRequested?.Invoke(candidate.CapabilityId);
        actions.Controls.Add(promoteBtn);

        var canRollback = _catalogVersion > 0;
        var rollbackTarget = Math.Max(0, _catalogVersion - 1);
        var rollbackBtn = new Button
        {
            Name = $"RollbackBtn:{candidate.CapabilityId}",
            Text = $"Rollback to v{rollbackTarget}",
            Enabled = canRollback,
            FlatStyle = FlatStyle.Flat,
            Size = new Size(Scale(116), Scale(26)),
            BackColor = CardBg,
            ForeColor = TextPrimary,
            Font = new Font("Segoe UI", 8f),
            Cursor = canRollback ? Cursors.Hand : Cursors.Default,
            Margin = new Padding(0)
        };
        rollbackBtn.FlatAppearance.BorderSize = 1;
        rollbackBtn.FlatAppearance.BorderColor = Color.FromArgb(60, 255, 255, 255);
        rollbackBtn.Click += (_, _) => RollbackCapabilityRequested?.Invoke(rollbackTarget);
        actions.Controls.Add(rollbackBtn);

        stack.Controls.Add(actions);
        card.Controls.Add(stack);
        return card;
    }

    private void AddImageDetail(FlowLayoutPanel stack, ScreenItem item, int maxW, Color bg)
    {
        if (!string.IsNullOrWhiteSpace(item.MediaPath) && File.Exists(item.MediaPath))
        {
            try
            {
                using var fs = new FileStream(item.MediaPath, FileMode.Open, FileAccess.Read, FileShare.ReadWrite);
                using var loaded = Image.FromStream(fs);
                var pic = new PictureBox
                {
                    Image = new Bitmap(loaded),
                    SizeMode = PictureBoxSizeMode.Zoom,
                    Width = maxW,
                    Height = Scale(160),
                    BackColor = Color.FromArgb(14, 17, 24),
                    Margin = new Padding(0, Scale(8), 0, 0)
                };
                stack.Controls.Add(pic);
            }
            catch
            {
                // Fall through to path note
            }
        }

        if (!string.IsNullOrWhiteSpace(item.MediaPath))
        {
            var openBtn = new Button
            {
                Text = "Open image",
                FlatStyle = FlatStyle.Flat,
                Size = new Size(Scale(106), Scale(28)),
                BackColor = AccentBlue,
                ForeColor = TextPrimary,
                Font = new Font("Segoe UI Semibold", 8.25f, FontStyle.Bold),
                Margin = new Padding(0, Scale(6), 0, 0),
                Cursor = Cursors.Hand
            };
            openBtn.FlatAppearance.BorderSize = 0;
            openBtn.Click += (_, _) => LaunchMediaFile(item.MediaPath);
            stack.Controls.Add(openBtn);
        }
        else
        {
            stack.Controls.Add(new Label
            {
                Text = "Generated/shared image preview.",
                AutoSize = true,
                ForeColor = AccentAmber,
                BackColor = bg,
                Font = new Font("Segoe UI", 8f),
                Margin = new Padding(0, Scale(6), 0, 0)
            });
        }
    }

    private void AddVideoDetail(FlowLayoutPanel stack, ScreenItem item, int maxW, Color bg)
    {
        stack.Controls.Add(new Label
        {
            Text = $"Duration: {item.DurationLabel ?? "clip"} · Plays only when you click Play (never autoplays).",
            AutoSize = true,
            ForeColor = AccentAmber,
            BackColor = bg,
            Font = new Font("Segoe UI", 8f),
            MaximumSize = new Size(maxW, 0),
            Margin = new Padding(0, Scale(6), 0, 0)
        });

        if (!string.IsNullOrWhiteSpace(item.MediaPath) && File.Exists(item.MediaPath))
        {
            var playBtn = new Button
            {
                Text = "▶ Play video",
                FlatStyle = FlatStyle.Flat,
                Size = new Size(Scale(108), Scale(28)),
                BackColor = AccentBlue,
                ForeColor = TextPrimary,
                Font = new Font("Segoe UI Semibold", 8.25f, FontStyle.Bold),
                Margin = new Padding(0, Scale(6), 0, 0),
                Cursor = Cursors.Hand
            };
            playBtn.FlatAppearance.BorderSize = 0;
            playBtn.Click += (_, _) => LaunchMediaFile(item.MediaPath);
            stack.Controls.Add(playBtn);
        }
    }

    private static void LaunchMediaFile(string path)
    {
        if (string.IsNullOrWhiteSpace(path) || !File.Exists(path))
            return;
        try
        {
            Process.Start(new ProcessStartInfo(path) { UseShellExecute = true });
        }
        catch
        {
            // Ignore shell launch errors cleanly
        }
    }

    private void ResizeCards()
    {
        var rowW = Math.Max(Scale(220), _list.ClientSize.Width - Scale(18));
        foreach (Control ctrl in _list.Controls)
        {
            if (ctrl is Panel p)
            {
                p.Width = rowW;
                p.MinimumSize = new Size(rowW, Scale(54));
                p.MaximumSize = new Size(rowW, 0);
            }
        }
    }

    private void UpdateRegion()
    {
        if (ClientSize.Width <= 0 || ClientSize.Height <= 0)
            return;
        using var path = RoundedRect(new RectangleF(0, 0, ClientSize.Width, ClientSize.Height), Scale(16));
        var old = Region;
        Region = new Region(path);
        old?.Dispose();
    }

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

    protected override void Dispose(bool disposing)
    {
        if (disposing)
            _store.Changed -= OnStoreChanged;
        base.Dispose(disposing);
    }
}
