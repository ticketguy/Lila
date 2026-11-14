using System.Drawing.Drawing2D;
using Lila.Desktop.Core.Api;
using Lila.Desktop.Core.Island;
using Lila.Desktop.Core.Keyboard;
using Lila.Desktop.Core.Lifecycle;
using Lila.Desktop.Core.Sentinel;
using Lila.Desktop.Core.Voice;
using Lila.Desktop.Core.Wsl;
using Lila.Desktop.Island;
using Lila.Desktop.Keyboard;
using Lila.Desktop.Security;
using Lila.Desktop.Settings;
using Lila.Desktop.Wsl;
using Microsoft.Win32;

namespace Lila.Desktop;

public sealed class MainForm : Form
{
    private readonly AppSettings _settings;
    private readonly WindowsCredentialStore _credentials = new();
    private readonly HttpClient _http;
    private readonly LilaApiClient _api;
    private DaemonEventStreamClient? _eventStream;
    private CancellationTokenSource? _eventStreamCts;
    private WslClient _wsl;
    private readonly VoiceSessionController _voice;
    private GlobalVoiceShortcutController _shortcuts;
    private RawKeyboardListener? _keyboard;
    private WslLifecycleSupervisor _lifecycle;
    private readonly NotifyIcon _tray;
    private readonly System.Windows.Forms.Timer _statusTimer = new() { Interval = 5_000 };
    private readonly System.Windows.Forms.Timer _fullscreenTimer = new() { Interval = 1_000 };
    private readonly System.Windows.Forms.Timer _transitionTimer = new() { Interval = 15 };
    private readonly System.Windows.Forms.Timer _retuckTimer = new() { Interval = 100 };
    private readonly System.Windows.Forms.Timer _approvalCountdownTimer = new() { Interval = 250 };
    private readonly System.Windows.Forms.Timer _washBreathTimer = new() { Interval = 120 };
    private IslandBackdropForm? _backdrop;

    private readonly Label _connection = new()
    {
        AutoSize = true,
        Text = "Checking Lila…",
        BackColor = Color.FromArgb(20, 24, 32)
    };
    private readonly Panel _connectionDot = new()
    {
        Size = new Size(7, 7),
        BackColor = Color.FromArgb(20, 24, 32)
    };
    private Color _connectionDotColor = Color.FromArgb(94, 224, 160);
    private readonly Label _shortcutSummary = new()
    {
        AutoSize = false,
        BackColor = Color.FromArgb(20, 24, 32)
    };

    // Approval card (#ap) controls — >= 200px detail height, .tbar, 250ms countdown, inline Owner Phrase
    private readonly Panel _approvalCard = new()
    {
        Dock = DockStyle.Top,
        Height = 296,
        Visible = false,
        BackColor = Color.FromArgb(36, 33, 39),
        Padding = new Padding(11, 10, 11, 10)
    };
    private readonly Label _approvalHeaderLabel = new()
    {
        Text = "Lila needs your approval",
        Dock = DockStyle.Top,
        Height = 20,
        ForeColor = Color.FromArgb(255, 208, 137),
        Font = new Font("Segoe UI", 9.5f, FontStyle.Bold),
        TextAlign = ContentAlignment.MiddleLeft
    };
    private readonly ComboBox _approvalPicker = new()
    {
        Dock = DockStyle.Top,
        DropDownStyle = ComboBoxStyle.DropDownList,
        Visible = false
    };
    private readonly Panel _approvalDetailsScroll = new()
    {
        Dock = DockStyle.Fill,
        AutoScroll = true,
        BackColor = Color.FromArgb(22, 25, 34),
        Padding = new Padding(8, 6, 8, 6)
    };
    private readonly Label _approvalDetails = new()
    {
        AutoSize = true,
        MaximumSize = new Size(310, 0),
        ForeColor = Color.FromArgb(245, 247, 250),
        BackColor = Color.FromArgb(22, 25, 34),
        Font = new Font("Segoe UI", 8.75f),
        Text = "No pending approvals."
    };
    private readonly Panel _approvalProgressBar = new()
    {
        Dock = DockStyle.Bottom,
        Height = 3,
        BackColor = Color.FromArgb(46, 42, 50)
    };
    private readonly Label _approvalCountdownLabel = new()
    {
        Dock = DockStyle.Bottom,
        Height = 18,
        ForeColor = Color.FromArgb(167, 176, 192),
        BackColor = Color.FromArgb(36, 33, 39),
        Font = new Font("Segoe UI", 8f),
        TextAlign = ContentAlignment.MiddleLeft,
        Text = string.Empty
    };
    private readonly Panel _inlinePhraseRow = new()
    {
        Dock = DockStyle.Bottom,
        Height = 30,
        BackColor = Color.FromArgb(36, 33, 39),
        Padding = new Padding(0, 3, 0, 3)
    };
    private readonly TextBox _inlineOwnerPhrase = new()
    {
        UseSystemPasswordChar = true,
        MaxLength = 512,
        PlaceholderText = "Owner Phrase (or click Voice)",
        AccessibleName = "Private Owner Phrase"
    };
    private readonly Button _voiceApprovalButton = new()
    {
        Text = "Voice",
        Enabled = false,
        AccessibleName = "Authenticate approval by voice"
    };
    private readonly Panel _approvalButtonsRow = new()
    {
        Dock = DockStyle.Bottom,
        Height = 34,
        BackColor = Color.FromArgb(36, 33, 39),
        Padding = new Padding(0, 4, 0, 0)
    };
    private readonly Button _rejectButton = new() { Text = "Reject", Enabled = false };
    private readonly Button _approveButton = new() { Text = "Approve", Enabled = false };

    // Main Island containers
    private readonly Panel _surface = new()
    {
        Dock = DockStyle.Fill,
        BackColor = Color.FromArgb(20, 24, 32),
        Padding = new Padding(1)
    };
    private readonly Panel _header = new()
    {
        Dock = DockStyle.Top,
        Height = 64,
        BackColor = Color.FromArgb(20, 24, 32)
    };
    private readonly Panel _body = new()
    {
        Dock = DockStyle.Fill,
        BackColor = Color.FromArgb(20, 24, 32),
        Padding = new Padding(12, 0, 12, 8)
    };
    private readonly Panel _composerRow = new()
    {
        Dock = DockStyle.Bottom,
        Height = 44,
        BackColor = Color.FromArgb(20, 24, 32),
        Padding = new Padding(0, 4, 0, 2)
    };
    private readonly Panel _inputPill = new()
    {
        BackColor = Color.FromArgb(16, 19, 26),
        Padding = new Padding(12, 6, 12, 6)
    };
    private readonly Panel _footer = new()
    {
        Dock = DockStyle.Bottom,
        Height = 36,
        BackColor = Color.FromArgb(20, 24, 32)
    };
    private readonly FlowLayoutPanel _conversation = new()
    {
        Dock = DockStyle.Fill,
        AutoScroll = true,
        FlowDirection = FlowDirection.TopDown,
        WrapContents = false,
        BackColor = Color.FromArgb(14, 17, 23),
        Padding = new Padding(10)
    };

    private readonly IslandAvatarControl _avatar = new();
    private readonly IslandEqualizerControl _headerEqualizer = new();
    private readonly IslandEqualizerControl _collapsedEqualizer = new();
    private readonly Label _title = new()
    {
        Text = "Lila",
        AutoSize = true,
        Font = new Font("Segoe UI Semibold", 11.25f, FontStyle.Bold),
        ForeColor = Color.FromArgb(245, 247, 250),
        BackColor = Color.FromArgb(20, 24, 32)
    };
    private readonly IslandIconButton _muteButton = new()
    {
        Icon = IslandIcon.Microphone,
        AccessibleName = "Mute microphone"
    };
    private readonly IslandIconButton _moreButton = new()
    {
        Icon = IslandIcon.MoreHorizontal,
        AccessibleName = "More options"
    };
    private readonly IslandIconButton _tuckButton = new()
    {
        Icon = IslandIcon.ChevronRight,
        AccessibleName = "Tuck Lila to screen edge"
    };
    private readonly IslandIconButton _edgeTab = new()
    {
        Icon = IslandIcon.ChevronLeft,
        IsEdgeTab = true,
        Visible = false,
        AccessibleName = "Restore Lila Island"
    };
    private readonly ContextMenuStrip _overflowMenu = new();
    private readonly ToolStripMenuItem _overflowHighlightItem;
    private readonly ToolStripMenuItem _overflowWorkDrawerItem;
    private readonly ToolStripMenuItem _overflowScreenShareItem;
    private readonly ToolStripMenuItem _overflowCaptionsItem;
    private readonly ToolStripMenuItem _overflowSettingsItem;
    private readonly ToolStripMenuItem _overflowSpokenEnrollItem;
    private readonly ToolStripMenuItem _overflowRetryItem;
    private string? _spokenEnrollSessionId;
    private readonly ToolStripMenuItem _overflowStopItem;
    private readonly ToolStripMenuItem _overflowDiagnosticItem;
    private readonly SettingsSheetControl _settingsSheet;
    private readonly IslandPeekPopup _peekPopup = new();
    private readonly IslandApprovalToastPopup _toastPopup = new();
    private readonly LilaScreenStore _screenStore = new();
    private readonly LilaScreenOverlayForm _screenOverlay;
    private readonly LilaWorkDrawerForm _workDrawer;
    private readonly ToolTip _toolTip = new();

    private ApprovalRequestView? _shownApproval;
    private double _approvalRemainingSeconds;
    private double _approvalInitialSeconds = 60.0;
    private long _approvalObservedTimestamp;
    private string? _lastApprovalId;
    private string? _lastObservedApprovalStatus;
    private bool _approvalBusy;
    private bool _approvalStateUnavailable;
    private string? _ownerPttSessionId;
    private ApprovalRequestView? _ownerPttRequest;

    private readonly TextBox _textInput = new() { BorderStyle = BorderStyle.None };
    private readonly Button _startButton = new() { Text = "Retry Lila", AutoSize = true, Visible = false };
    private readonly Button _stopButton = new() { Text = "Stop Lila", AutoSize = true, Enabled = false, Visible = false };
    private readonly Button _cancelButton = new() { Text = "Cancel", AutoSize = false, Enabled = false, Visible = false };
    private readonly IslandIconButton _highlightButton = new()
    {
        Icon = IslandIcon.ScreenHighlight,
        AccessibleName = "Highlight screen region or full screen for Lila"
    };
    private readonly IslandIconButton _voiceToggleButton = new()
    {
        Icon = IslandIcon.Microphone,
        AccessibleName = "Hold to talk, or click to keep voice on"
    };
    private readonly IslandIconButton _sendButton = new()
    {
        Icon = IslandIcon.Send,
        IsPrimary = true,
        AccessibleName = "Send message to Lila"
    };
    private readonly Panel _highlightBanner = new()
    {
        Dock = DockStyle.Bottom,
        Height = 28,
        Visible = false,
        BackColor = Color.FromArgb(24, 30, 44),
        Padding = new Padding(10, 4, 8, 4)
    };
    private readonly Label _highlightBannerLabel = new()
    {
        AutoSize = false,
        ForeColor = Color.FromArgb(165, 188, 255),
        Font = new Font("Segoe UI Semibold", 8f, FontStyle.Bold),
        TextAlign = ContentAlignment.MiddleLeft
    };
    private readonly Button _discardHighlightButton = new()
    {
        Text = "Discard",
        Size = new Size(58, 20),
        AccessibleName = "Discard attached screen highlight"
    };
    private TransientScreenHighlight? _attachedHighlight;
    private bool _highlightOverlayOpen;

    private readonly SemaphoreSlim _connectionGate = new(1, 1);
    private readonly HashSet<string> _knownApprovalIds = new(StringComparer.Ordinal);
    private bool _textBusy;
    private bool _serviceManuallyStopped;
    private bool _isReady;
    private bool _expanded;
    private bool _tucked;
    private bool _tuckMode;
    private IslandDockEdge _tuckedEdge = IslandDockEdge.Right;
    private long _lastTuckedTicks;
    private long _lastEdgeAutoRestoreTicks;
    private string? _liveTranscriptText;
    private long? _hoverOutsideSinceTicks;
    private bool _settingsOpen;
    private bool _fullscreenActive;
    private bool _tuckedForFullscreen;
    private bool _approvalSnapshotLoaded;
    private bool _dragging;
    private bool _dragMoved;
    private Control? _dragCaptureControl;
    private long _manualRestoreUntilTicks;
    private Rectangle _dragStartBounds;
    private Point _dragStartPointer;
    private Rectangle _restoreBounds;
    private string? _restoreMonitorDeviceName;
    private Rectangle _transitionStart;
    private Rectangle _transitionTarget;
    private long _transitionStarted;
    private VoiceSessionStatus? _remoteVoiceStatus;
    private string? _voiceStatusError;
    private IslandVisualState _lastReportedVoiceVisual = IslandVisualState.Disconnected;
    private float _washPhase;
    private long _micDownTimestamp;
    private bool _micHoldActive;
    private int _conversationMessageCount;
    private bool _diagnosticOpen;
    private InteractionDiagnostics? _interactionDiagnostics;
    private KeyboardDiagnosticForm? _diagnosticForm;
    private bool _exitRequested;
    private readonly bool _startMinimized;

    public MainForm(AppSettings settings, bool startMinimized = false)
    {
        SuspendLayout();
        _settings = settings;
        _startMinimized = startMinimized;
        _expanded = settings.IslandExpanded;
        var handler = LilaMutualTlsHandler.CreateFromEnvironment();
        _http = new HttpClient(handler) { Timeout = Timeout.InfiniteTimeSpan };
        _api = new LilaApiClient(_http, _credentials);
        _wsl = new WslClient(new SystemCommandRunner(), _api, settings.Distribution);
        _lifecycle = new WslLifecycleSupervisor(new SystemCommandRunner(),
            new ManagedWslKeepAlive(), settings.Distribution);
        _voice = new VoiceSessionController(_api);
        _voice.StateChanged += OnVoiceStateChanged;
        _shortcuts = CreateShortcutController();

        _settingsSheet = new SettingsSheetControl(_settings, _credentials)
        {
            Dock = DockStyle.Fill,
            Visible = false
        };
        _settingsSheet.BackRequested += HideSettingsSheet;
        _settingsSheet.SettingsSaved += async previousDistribution => await OnSettingsSheetSavedAsync(previousDistribution);

        _screenOverlay = new LilaScreenOverlayForm(_screenStore);
        _workDrawer = new LilaWorkDrawerForm(_screenStore);
        _screenOverlay.PillClicked += () => ShowWindow();
        _screenOverlay.CardClicked += itemId =>
        {
            _workDrawer.OpenForItem(itemId, Screen.FromRectangle(Bounds));
            _ = RefreshCapabilityCandidatesAsync();
        };
        _workDrawer.OpenMainIslandRequested += () => ShowWindow();
        _workDrawer.CancelTaskRequested += async taskId =>
        {
            try
            {
                await _api.CancelTaskAsync(taskId);
            }
            catch { /* Daemon returns 404 if already finished/ephemeral. */ }
        };
        _workDrawer.PromoteCapabilityRequested += async capabilityId =>
        {
            try
            {
                await _api.PromoteCapabilityCandidateAsync(capabilityId, ownerApproved: true);
                await RefreshCapabilityCandidatesAsync();
            }
            catch { /* Surface remains responsive if daemon is offline. */ }
        };
        _workDrawer.RollbackCapabilityRequested += async targetVersion =>
        {
            try
            {
                await _api.RollbackCapabilityCatalogAsync(targetVersion);
                await RefreshCapabilityCandidatesAsync();
            }
            catch { /* Surface remains responsive if daemon is offline. */ }
        };

        _overflowHighlightItem = new ToolStripMenuItem("Highlight screen / comment", null,
            async (_, _) => await StartScreenHighlightFlowAsync());
        _overflowWorkDrawerItem = new ToolStripMenuItem("Lila’s work drawer", null,
            (_, _) =>
            {
                _workDrawer.ToggleDrawer(Screen.FromRectangle(Bounds));
                if (_workDrawer.Visible)
                    _ = RefreshCapabilityCandidatesAsync();
            });
        _overflowScreenShareItem = new ToolStripMenuItem("Screen sharing privacy", null, (_, _) =>
        {
            var next = !_screenStore.ScreenSharingActive;
            _screenStore.SetScreenSharing(next);
            _overflowScreenShareItem!.Checked = next;
        })
        {
            CheckOnClick = false
        };
        _overflowCaptionsItem = new ToolStripMenuItem("Live captions", null, (_, _) =>
        {
            var next = !_screenStore.CaptionsEnabled;
            _screenStore.SetCaptionsEnabled(next);
            _overflowCaptionsItem!.Checked = next;
            if (!next)
                _screenOverlay.ClearCaptions();
        })
        {
            Checked = true,
            CheckOnClick = false
        };
        _overflowSettingsItem = new ToolStripMenuItem("Settings", null, (_, _) => ShowSettings());
        _overflowSpokenEnrollItem = new ToolStripMenuItem("Enroll voice approval", null, async (_, _) => await ToggleSpokenEnrollmentAsync());
        _overflowRetryItem = new ToolStripMenuItem("Retry connection", null, async (_, _) => await StartLilaAsync());
        _overflowStopItem = new ToolStripMenuItem("Stop Lila", null, async (_, _) => await StopLilaAsync());
        _overflowDiagnosticItem = new ToolStripMenuItem("Keyboard diagnostic", null, (_, _) => ShowDiagnostic());
        StyleOverflowMenu();

        Text = "Lila";
        StartPosition = FormStartPosition.Manual;
        FormBorderStyle = FormBorderStyle.None;
        ShowInTaskbar = false;
        TopMost = true;
        AutoScaleDimensions = new SizeF(96f, 96f);
        AutoScaleMode = AutoScaleMode.Dpi;
        BackColor = Color.FromArgb(20, 24, 32);
        Size = new Size(64, 64);
        MinimumSize = new Size(22, 56);
        Icon = SystemIcons.Application;
        if (_startMinimized)
        {
            WindowState = FormWindowState.Minimized;
            ShowInTaskbar = false;
        }

        try
        {
            _keyboard = new RawKeyboardListener(ConfiguredBindings());
            _keyboard.RelevantKeyEvent += OnRelevantKey;
            _keyboard.NonModifierKeyDown += _shortcuts.NotifyNonModifierKeyDown;
            _keyboard.KeyboardDeviceRemoved += _shortcuts.NotifyDeviceRemoved;
        }
        catch (Exception ex)
        {
            _isReady = false;
            _connection.Text = "Keyboard input unavailable: " + ex.Message;
            _keyboard = null!;
        }

        BuildLayout();

        // Seed initial Lila greeting matching lila-island-preview.html #log
        AppendConversation("Lila", "Hi, I’m here when you need me.", suppressNotifications: true);

        // Set placement BEFORE the window ever shows to prevent any (0,0) top-left flash or clipping
        RestoreSavedPlacement();
        ApplyExpandedLayout(_expanded, animate: false, save: false);

        _startButton.Click += async (_, _) => await StartLilaAsync();
        _stopButton.Click += async (_, _) => await StopLilaAsync();
        _cancelButton.Click += (_, _) => _voice.CancelActive();
        _highlightButton.Click += async (_, _) => await StartScreenHighlightFlowAsync();
        _discardHighlightButton.Click += (_, _) => DiscardAttachedHighlight();
        WireMicButtonHoldAndTap();
        _muteButton.Click += (_, _) =>
        {
            _shortcuts.SetMuted(!_shortcuts.IsMuted);
            RefreshVoiceIndicator(_voice.Snapshot);
        };
        _moreButton.Click += (_, _) =>
        {
            _overflowRetryItem.Enabled = !_isReady;
            _overflowStopItem.Enabled = _isReady;
            _overflowMenu.Show(_moreButton, new Point(0, _moreButton.Height + ScaleLogical(4)));
        };
        _tuckButton.Click += (_, _) =>
        {
            _tuckMode = true;
            TuckIsland();
        };
        _edgeTab.Click += (_, _) =>
        {
            _manualRestoreUntilTicks = Environment.TickCount64 + 4000;
            ShowWindow();
        };
        _edgeTab.MouseEnter += (_, _) =>
        {
            if (!_fullscreenActive &&
                Control.MouseButtons == MouseButtons.None &&
                Environment.TickCount64 - _lastTuckedTicks >= 250)
            {
                _lastEdgeAutoRestoreTicks = Environment.TickCount64;
                RestoreIsland(preserveTuckMode: true);
            }
        };
        _surface.MouseDown += OnHeaderMouseDown;
        _surface.MouseMove += OnHeaderMouseMove;
        _surface.MouseUp += OnHeaderMouseUp;
        _surface.MouseCaptureChanged += OnHeaderMouseCaptureChanged;
        _header.MouseDown += OnHeaderMouseDown;
        _header.MouseMove += OnHeaderMouseMove;
        _header.MouseUp += OnHeaderMouseUp;
        _header.MouseCaptureChanged += OnHeaderMouseCaptureChanged;
        _avatar.MouseDown += OnHeaderMouseDown;
        _avatar.MouseMove += OnHeaderMouseMove;
        _avatar.MouseUp += OnHeaderMouseUp;
        _avatar.MouseCaptureChanged += OnHeaderMouseCaptureChanged;
        _connection.MouseDown += OnHeaderMouseDown;
        _connection.MouseMove += OnHeaderMouseMove;
        _connection.MouseUp += OnHeaderMouseUp;
        _connection.MouseCaptureChanged += OnHeaderMouseCaptureChanged;
        _connectionDot.MouseDown += OnHeaderMouseDown;
        _connectionDot.MouseMove += OnHeaderMouseMove;
        _connectionDot.MouseUp += OnHeaderMouseUp;
        _connectionDot.MouseCaptureChanged += OnHeaderMouseCaptureChanged;
        _title.MouseDown += OnHeaderMouseDown;
        _title.MouseMove += OnHeaderMouseMove;
        _title.MouseUp += OnHeaderMouseUp;
        _title.MouseCaptureChanged += OnHeaderMouseCaptureChanged;

        _retuckTimer.Tick += (_, _) =>
        {
            if (!_tuckMode || _tucked || _expanded || _dragging || _overflowMenu.Visible)
            {
                _retuckTimer.Stop();
                _hoverOutsideSinceTicks = null;
                return;
            }
            if (_transitionTimer.Enabled)
            {
                _hoverOutsideSinceTicks = null;
                return;
            }

            var hoverZone = Bounds;
            hoverZone.Inflate(ScaleLogical(8), ScaleLogical(8));
            if (hoverZone.Contains(Cursor.Position))
            {
                _hoverOutsideSinceTicks = null;
                return;
            }

            var now = Environment.TickCount64;
            _hoverOutsideSinceTicks ??= now;
            if (now - _hoverOutsideSinceTicks.Value >= 320)
            {
                _retuckTimer.Stop();
                _hoverOutsideSinceTicks = null;
                TuckIsland();
            }
        };
        _approvalCountdownTimer.Tick += (_, _) => TickApprovalCountdown();
        _washBreathTimer.Tick += (_, _) =>
        {
            _washPhase = (_washPhase + 0.03f) % 1f;
            _surface.Invalidate();
        };
        _fullscreenTimer.Tick += (_, _) => CheckFullscreenState();
        _transitionTimer.Tick += (_, _) => AdvanceWindowTransition();
        _conversation.SizeChanged += (_, _) => LayoutConversationBubbles();
        _sendButton.Click += async (_, _) => await SendTextAsync();
        _approveButton.Click += async (_, _) => await SubmitApprovalAsync("approve");
        _rejectButton.Click += async (_, _) => await SubmitApprovalAsync("reject");
        _voiceApprovalButton.Click += async (_, _) => await SubmitSpokenApprovalAsync();

        _peekPopup.Clicked += () =>
        {
            if (_shownApproval is not null && (!_expanded || _tucked))
            {
                var elapsed = System.Diagnostics.Stopwatch.GetElapsedTime(_approvalObservedTimestamp).TotalSeconds;
                var current = Math.Max(0.0, _approvalRemainingSeconds - elapsed);
                _toastPopup.ShowOrUpdateToast(_shownApproval, current, _approvalInitialSeconds,
                    Bounds, Screen.FromRectangle(Bounds), _voiceApprovalButton.Enabled, forceShow: true);
                return;
            }
            ShowWindow();
        };
        _toastPopup.OpenLilaClicked += () => ShowWindow();
        _toastPopup.ApproveClicked += async () => await SubmitApprovalAsync("approve");
        _toastPopup.RejectClicked += async () => await SubmitApprovalAsync("reject");
        _toastPopup.SpeakApproveClicked += async () => await SubmitSpokenApprovalAsync();
        _toastPopup.AutoTucked += () =>
        {
            var count = Math.Max(1, _approvalPicker.Items.Count);
            if (_tray is not null)
                _tray.Text = $"Lila — {count} pending";
            _toolTip.SetToolTip(_surface, $"{count} pending — click to review");
            _peekPopup.ShowPeek($"{count} pending", Bounds, Screen.FromRectangle(Bounds));
        };

        _approvalPicker.SelectedIndexChanged += (_, _) =>
        {
            if (_approvalPicker.SelectedItem is ApprovalChoice choice)
            {
                _shownApproval = choice.Request;
                _lastApprovalId = choice.Request.InvocationId;
                ResetApprovalCountdown(choice.Request.ExpiresInSeconds);
                ShowApprovalDetails(choice.Request, _approvalPicker.Items.Count);
            }
        };
        _textInput.GotFocus += (_, _) => _inputPill.Invalidate();
        _textInput.LostFocus += (_, _) => _inputPill.Invalidate();
        _textInput.KeyDown += async (_, e) =>
        {
            if (e.KeyCode == Keys.Enter && !e.Shift)
            {
                e.SuppressKeyPress = true;
                await SendTextAsync();
            }
        };
        _statusTimer.Tick += async (_, _) =>
        {
            await RefreshStatusAsync();
            await RefreshApprovalsAsync();
            StartEventStreamLoop();
            _ = RunOptedInSentinelAsync();
        };
        _statusTimer.Start();
        _fullscreenTimer.Start();
        EnableDoubleBuffering(_surface);
        EnableDoubleBuffering(_conversation);
        EnableDoubleBuffering(_header);
        EnableDoubleBuffering(_inputPill);
        SystemEvents.DisplaySettingsChanged += OnDisplaySettingsChanged;
        Shown += async (_, _) =>
        {
            RestoreSavedPlacement();
            ApplyExpandedLayout(_expanded, animate: false, save: false);
            if (_startMinimized)
                BeginInvoke(new Action(Hide));
            await RefreshStatusAsync();
            await RefreshApprovalsAsync();
            StartEventStreamLoop();
            _ = RunOptedInSentinelAsync();
        };

        var menu = new ContextMenuStrip();
        menu.Items.Add("Show Lila", null, (_, _) => ShowWindow());
        menu.Items.Add("Highlight screen / comment", null, async (_, _) => await StartScreenHighlightFlowAsync());
        menu.Items.Add("Start Lila", null, async (_, _) => await StartLilaAsync());
        menu.Items.Add("Stop Lila", null, async (_, _) => await StopLilaAsync());
        menu.Items.Add("Settings", null, (_, _) => ShowSettings());
        menu.Items.Add("Tuck Island", null, (_, _) =>
        {
            _tuckMode = true;
            TuckIsland();
        });
        var muteVoiceItem = new ToolStripMenuItem("Mute microphone input");
        muteVoiceItem.Click += (_, _) =>
        {
            _shortcuts.SetMuted(!_shortcuts.IsMuted);
            RefreshVoiceIndicator(_voice.Snapshot);
        };
        menu.Items.Add(muteVoiceItem);
        menu.Opening += (_, _) =>
            muteVoiceItem.Text = _shortcuts.IsMuted
                ? "Unmute microphone input" : "Mute microphone input";
        menu.Items.Add("Keyboard diagnostic", null, (_, _) => ShowDiagnostic());
        menu.Items.Add("Exit", null, (_, _) => ExitApplication());
        _tray = new NotifyIcon
        {
            Icon = SystemIcons.Application,
            Text = "Lila — checking service",
            ContextMenuStrip = menu,
            Visible = true
        };
        _tray.DoubleClick += (_, _) => ShowWindow();
        _tray.BalloonTipClicked += (_, _) => ShowWindow();
        ResumeLayout(performLayout: true);
    }

    private void StyleOverflowMenu()
    {
        _overflowMenu.BackColor = Color.FromArgb(24, 29, 40);
        _overflowMenu.ForeColor = Color.FromArgb(245, 247, 250);
        _overflowMenu.ShowImageMargin = false;
        _overflowMenu.Font = new Font("Segoe UI", 9f);
        _overflowMenu.Renderer = new DarkMenuRenderer();
        _overflowMenu.Items.AddRange([
            _overflowHighlightItem,
            _overflowWorkDrawerItem,
            _overflowScreenShareItem,
            _overflowCaptionsItem,
            new ToolStripSeparator(),
            _overflowSettingsItem,
            _overflowSpokenEnrollItem,
            new ToolStripSeparator(),
            _overflowRetryItem,
            _overflowStopItem,
            new ToolStripSeparator(),
            _overflowDiagnosticItem
        ]);
        foreach (ToolStripItem item in _overflowMenu.Items)
            item.ForeColor = Color.FromArgb(245, 247, 250);
    }

    private sealed class DarkMenuRenderer : ToolStripProfessionalRenderer
    {
        public DarkMenuRenderer() : base(new DarkColorTable()) { }

        protected override void OnRenderItemText(ToolStripItemTextRenderEventArgs e)
        {
            e.TextColor = e.Item.Enabled
                ? Color.FromArgb(245, 247, 250)
                : Color.FromArgb(115, 125, 145);
            base.OnRenderItemText(e);
        }
    }

    private sealed class DarkColorTable : ProfessionalColorTable
    {
        public override Color ToolStripDropDownBackground => Color.FromArgb(24, 29, 40);
        public override Color MenuBorder => Color.FromArgb(65, 255, 255, 255);
        public override Color MenuItemBorder => Color.FromArgb(90, 124, 156, 255);
        public override Color MenuItemSelected => Color.FromArgb(45, 124, 156, 255);
        public override Color SeparatorDark => Color.FromArgb(40, 255, 255, 255);
        public override Color SeparatorLight => Color.Transparent;
    }

    private void WireMicButtonHoldAndTap()
    {
        // HTML reference #mic: hold >= 300ms = HoldToTalk (release to send); tap < 300ms = toggle continuous voice
        _voiceToggleButton.MouseDown += (_, e) =>
        {
            if (e.Button != MouseButtons.Left)
                return;
            if (_shortcuts.IsMuted)
            {
                OnMutedVoiceAttempt();
                return;
            }
            if (!_isReady)
                return;
            if (_shownApproval is { ApprovalReady: true } pendingApproval &&
                _voiceApprovalButton.Enabled && !_approvalBusy)
            {
                _ = StartOwnerPttAsync(pendingApproval);
                return;
            }
            _micDownTimestamp = System.Diagnostics.Stopwatch.GetTimestamp();
            _micHoldActive = false;
            var state = _voice.Snapshot.State;
            if (state is VoiceInteractionState.Idle or VoiceInteractionState.Response)
            {
                _shortcuts.ToggleFromStartButton();
                _micHoldActive = true;
            }
        };
        _voiceToggleButton.MouseUp += (_, e) =>
        {
            if (e.Button != MouseButtons.Left || _shortcuts.IsMuted || !_isReady)
                return;
            if (_ownerPttSessionId is not null)
            {
                _ = StopOwnerPttAsync();
                return;
            }
            var heldMs = System.Diagnostics.Stopwatch.GetElapsedTime(_micDownTimestamp).TotalMilliseconds;
            if (_micHoldActive)
            {
                // If held >= 300ms, releasing submits immediately (Hold-to-talk); if < 300ms, keep voice on (Toggle-to-talk)!
                if (heldMs >= 300 &&
                    _voice.Snapshot.State is VoiceInteractionState.Starting or VoiceInteractionState.Recording)
                {
                    _shortcuts.ToggleFromStartButton();
                }
                _micHoldActive = false;
            }
            else if (_voice.Snapshot.State is VoiceInteractionState.Starting or VoiceInteractionState.Recording)
            {
                _shortcuts.ToggleFromStartButton();
            }
        };
    }

    private void OnMutedVoiceAttempt()
    {
        if (InvokeRequired)
        {
            BeginInvoke(new Action(OnMutedVoiceAttempt));
            return;
        }
        _avatar.FlashMuteBadge();
        ShowPeekMessage("Mic is off. Turn it on to talk.");
    }

    private ShortcutBinding[] ConfiguredBindings()
    {
        var keys = new List<ShortcutBinding>();
        if (_settings.HoldToTalkEnabled)
            keys.Add(new ShortcutBinding(_settings.HoldToTalkVirtualKey, _settings.HoldToTalkModifiers));
        if (_settings.ToggleToTalkEnabled)
            keys.Add(new ShortcutBinding(_settings.ToggleToTalkVirtualKey, _settings.ToggleToTalkModifiers));
        return keys.DistinctBy(binding => binding.VirtualKey).ToArray();
    }

    private GlobalVoiceShortcutController CreateShortcutController()
    {
        var controller = new GlobalVoiceShortcutController(_voice,
            new ShortcutBinding(_settings.HoldToTalkVirtualKey, _settings.HoldToTalkModifiers),
            new ShortcutBinding(_settings.ToggleToTalkVirtualKey, _settings.ToggleToTalkModifiers),
            _settings.HoldToTalkEnabled, _settings.ToggleToTalkEnabled,
            TimeSpan.FromMilliseconds(Math.Clamp(_settings.RightControlArmingDelayMilliseconds, 100, 600)));
        controller.MutedShortcutAttempted += OnMutedVoiceAttempt;
        return controller;
    }

    private static string ShortcutLabel(bool enabled, ShortcutBinding binding) =>
        enabled ? binding.DisplayName : "disabled";

    private void BuildLayout()
    {
        _surface.Paint += (_, e) => PaintIslandSurface(e.Graphics, _surface.ClientRectangle);
        _surface.SizeChanged += (_, _) => UpdateIslandRegion();
        _header.Resize += (_, _) => LayoutHeader();
        _composerRow.Resize += (_, _) => LayoutComposerRow();
        _inputPill.Paint += (_, e) => PaintInputPill(e.Graphics, _inputPill.ClientRectangle);
        _connectionDot.Paint += (_, e) =>
        {
            e.Graphics.Clear(Color.FromArgb(20, 24, 32));
            e.Graphics.SmoothingMode = SmoothingMode.AntiAlias;
            using var brush = new SolidBrush(_connectionDotColor);
            e.Graphics.FillEllipse(brush, 0, 0, _connectionDot.Width - 1, _connectionDot.Height - 1);
        };

        _avatar.State = IslandVisualState.Disconnected;
        _avatar.Cursor = Cursors.Hand;
        _title.Cursor = Cursors.SizeAll;
        _connection.Cursor = Cursors.SizeAll;

        _connection.ForeColor = Color.FromArgb(167, 176, 192);
        _connection.Font = new Font("Segoe UI", 8.25f);
        _shortcutSummary.Text = ShortcutSummary();
        _shortcutSummary.ForeColor = Color.FromArgb(198, 167, 176, 192);
        _shortcutSummary.Font = new Font("Segoe UI", 7.75f);
        _shortcutSummary.AutoEllipsis = true;
        _shortcutSummary.UseMnemonic = false;

        _approvalPicker.DisplayMember = nameof(ApprovalChoice.Label);
        _approvalPicker.BackColor = Color.FromArgb(32, 36, 46);
        _approvalPicker.ForeColor = Color.White;
        _approvalPicker.FlatStyle = FlatStyle.Flat;

        StyleButton(_cancelButton);
        StyleButton(_approveButton, primary: true);
        StyleButton(_rejectButton);
        StyleButton(_voiceApprovalButton);

        _inlineOwnerPhrase.BackColor = Color.FromArgb(28, 32, 42);
        _inlineOwnerPhrase.ForeColor = Color.White;
        _inlineOwnerPhrase.BorderStyle = BorderStyle.FixedSingle;
        _inlineOwnerPhrase.Font = new Font("Segoe UI", 8.5f);

        _muteButton.AccessibleName = "Mute microphone input";
        _moreButton.AccessibleName = "More options";
        _tuckButton.AccessibleDescription = "Move Lila to a small accessible tab at the screen edge";
        _voiceToggleButton.AccessibleName = "Hold to talk, or click to keep voice on";
        _sendButton.AccessibleName = "Send message to Lila";
        _cancelButton.Text = "Cancel";
        _cancelButton.AccessibleName = "Cancel current voice capture";

        _textInput.PlaceholderText = "Message Lila";
        _textInput.BackColor = Color.FromArgb(20, 24, 34);
        _textInput.ForeColor = Color.FromArgb(245, 247, 250);
        _textInput.BorderStyle = BorderStyle.None;
        _textInput.Font = new Font("Segoe UI", 9.5f);

        _edgeTab.Dock = DockStyle.Fill;
        _muteButton.Size = new Size(32, 32);
        _moreButton.Size = new Size(32, 32);
        _tuckButton.Size = new Size(32, 32);
        _highlightButton.Size = new Size(34, 34);
        _voiceToggleButton.Size = new Size(34, 34);
        _sendButton.Size = new Size(34, 34);

        StyleButton(_discardHighlightButton);
        _discardHighlightButton.Font = new Font("Segoe UI", 7.5f);
        _highlightBanner.Resize += (_, _) =>
        {
            _discardHighlightButton.Location = new Point(
                _highlightBanner.ClientSize.Width - _discardHighlightButton.Width - ScaleLogical(6),
                Math.Max(2, (_highlightBanner.ClientSize.Height - _discardHighlightButton.Height) / 2));
            _highlightBannerLabel.Bounds = new Rectangle(
                ScaleLogical(8),
                ScaleLogical(2),
                Math.Max(60, _discardHighlightButton.Left - ScaleLogical(14)),
                _highlightBanner.ClientSize.Height - ScaleLogical(4));
        };
        _highlightBanner.Controls.Add(_highlightBannerLabel);
        _highlightBanner.Controls.Add(_discardHighlightButton);

        _toolTip.SetToolTip(_muteButton, "Mute microphone");
        _toolTip.SetToolTip(_moreButton, "More (Highlight screen, Settings, Retry, Stop, Diagnostic)");
        _toolTip.SetToolTip(_tuckButton, "Tuck to screen edge");
        _toolTip.SetToolTip(_highlightButton, "Highlight part of screen or full screen to comment / ask Lila");
        _toolTip.SetToolTip(_voiceToggleButton, "Hold to talk, or click to keep voice on");
        _toolTip.SetToolTip(_sendButton, "Send");

        _header.Controls.Add(_avatar);
        _header.Controls.Add(_collapsedEqualizer);
        _header.Controls.Add(_title);
        _header.Controls.Add(_connectionDot);
        _header.Controls.Add(_connection);
        _header.Controls.Add(_headerEqualizer);
        _header.Controls.Add(_muteButton);
        _header.Controls.Add(_moreButton);
        _header.Controls.Add(_tuckButton);

        // Approval card (#ap) layout with >= 200px detail space, .tbar, countdown, inline phrase, equal Reject/Approve
        _approvalCard.Paint += (_, e) => PaintApprovalBorder(e.Graphics, _approvalCard.ClientRectangle);
        _approvalCard.SizeChanged += (_, _) =>
        {
            UpdateApprovalRegion();
            LayoutApprovalControls();
        };
        _approvalProgressBar.Paint += (_, e) =>
        {
            var fraction = _approvalInitialSeconds > 0
                ? Math.Clamp(_approvalRemainingSeconds / _approvalInitialSeconds, 0d, 1d)
                : 0d;
            var fillW = (int)Math.Round(_approvalProgressBar.Width * fraction);
            if (fillW > 0)
            {
                using var b = new SolidBrush(Color.FromArgb(255, 184, 107));
                e.Graphics.FillRectangle(b, 0, 0, fillW, _approvalProgressBar.Height);
            }
        };

        _approvalDetailsScroll.Controls.Add(_approvalDetails);
        _inlinePhraseRow.Controls.Add(_inlineOwnerPhrase);
        _inlinePhraseRow.Controls.Add(_voiceApprovalButton);
        _approvalButtonsRow.Controls.Add(_rejectButton);
        _approvalButtonsRow.Controls.Add(_approveButton);

        _approvalCard.Controls.Add(_approvalDetailsScroll);
        _approvalCard.Controls.Add(_approvalProgressBar);
        _approvalCard.Controls.Add(_approvalCountdownLabel);
        _approvalCard.Controls.Add(_inlinePhraseRow);
        _approvalCard.Controls.Add(_approvalButtonsRow);
        _approvalCard.Controls.Add(_approvalPicker);
        _approvalCard.Controls.Add(_approvalHeaderLabel);

        // Separate composer controls: pill input (#msg) + circular highlight + circular mic (#mic) + circular send (#send)
        _inputPill.Controls.Add(_textInput);
        _composerRow.Controls.Add(_inputPill);
        _composerRow.Controls.Add(_highlightButton);
        _composerRow.Controls.Add(_voiceToggleButton);
        _composerRow.Controls.Add(_sendButton);

        // Footer (.keys) below composer row
        _cancelButton.Size = new Size(56, 22);
        _footer.Resize += (_, _) =>
        {
            _cancelButton.Location = new Point(_footer.ClientSize.Width - _cancelButton.Width, ScaleLogical(4));
            _shortcutSummary.Bounds = new Rectangle(
                0,
                ScaleLogical(2),
                _cancelButton.Visible
                    ? Math.Max(60, _footer.ClientSize.Width - _cancelButton.Width - ScaleLogical(6))
                    : _footer.ClientSize.Width,
                _footer.ClientSize.Height - ScaleLogical(2));
        };
        _footer.Controls.Add(_shortcutSummary);
        _footer.Controls.Add(_cancelButton);

        _conversation.SizeChanged += (_, _) => UpdateConversationRegion();

        // WinForms bottom docking order: last added DockStyle.Bottom sits above earlier DockStyle.Bottom
        // So adding _footer first, _composerRow second, and _highlightBanner third places _highlightBanner above _composerRow and _composerRow above _footer (.keys)!
        _body.Controls.Add(_conversation);
        _body.Controls.Add(_highlightBanner);
        _body.Controls.Add(_composerRow);
        _body.Controls.Add(_footer);
        _body.Controls.Add(_approvalCard);

        _surface.Controls.Add(_settingsSheet);
        _surface.Controls.Add(_body);
        _surface.Controls.Add(_header);
        _surface.Controls.Add(_edgeTab);
        Controls.Add(_surface);

        _body.Visible = _expanded;
        LayoutHeader();
        LayoutComposerRow();
        LayoutApprovalControls();
        UpdateIslandRegion();
    }

    private void LayoutComposerRow()
    {
        _composerRow.Height = ScaleLogical(42);
        _footer.Height = ScaleLogical(36);
        _highlightBanner.Height = ScaleLogical(28);
        if (_composerRow.ClientSize.Width <= 0)
            return;
        var h = ScaleLogical(34);
        var btnSize = ScaleLogical(34);
        var gap = ScaleLogical(6);
        var top = Math.Max(0, (_composerRow.ClientSize.Height - h) / 2);
        var sendX = _composerRow.ClientSize.Width - btnSize;
        var micX = sendX - gap - btnSize;
        var highlightX = micX - gap - btnSize;
        var pillW = Math.Max(ScaleLogical(80), highlightX - gap);

        _inputPill.Bounds = new Rectangle(0, top, pillW, h);
        _highlightButton.Bounds = new Rectangle(highlightX, top, btnSize, btnSize);
        _voiceToggleButton.Bounds = new Rectangle(micX, top, btnSize, btnSize);
        _sendButton.Bounds = new Rectangle(sendX, top, btnSize, btnSize);

        var textH = _textInput.PreferredHeight;
        var padH = ScaleLogical(13);
        _textInput.Bounds = new Rectangle(
            padH,
            Math.Max(2, (_inputPill.ClientSize.Height - textH) / 2),
            Math.Max(40, _inputPill.ClientSize.Width - padH * 2),
            textH);
    }

    private void LayoutApprovalControls()
    {
        var innerW = Math.Max(120, _approvalCard.ClientSize.Width - _approvalCard.Padding.Horizontal);
        _approvalDetails.MaximumSize = new Size(Math.Max(100, innerW - ScaleLogical(22)), 0);

        var voiceBtnW = ScaleLogical(68);
        var gap = ScaleLogical(6);
        _inlineOwnerPhrase.Bounds = new Rectangle(
            0,
            ScaleLogical(3),
            Math.Max(80, innerW - voiceBtnW - gap),
            ScaleLogical(24));
        _voiceApprovalButton.Bounds = new Rectangle(
            innerW - voiceBtnW,
            ScaleLogical(2),
            voiceBtnW,
            ScaleLogical(26));

        var btnW = (innerW - gap) / 2;
        var btnH = ScaleLogical(29);
        _rejectButton.Bounds = new Rectangle(0, ScaleLogical(4), btnW, btnH);
        _approveButton.Bounds = new Rectangle(btnW + gap, ScaleLogical(4), innerW - btnW - gap, btnH);
    }

    private string ShortcutSummary()
    {
        var holdName = ShortcutLabel(_settings.HoldToTalkEnabled,
            new ShortcutBinding(_settings.HoldToTalkVirtualKey, _settings.HoldToTalkModifiers));
        var toggleName = ShortcutLabel(_settings.ToggleToTalkEnabled,
            new ShortcutBinding(_settings.ToggleToTalkVirtualKey, _settings.ToggleToTalkModifiers));
        return $"Hold {holdName} to talk, release to send. {toggleName} keeps voice on. Voice and text may be retained.";
    }

    private static void StyleButton(Button button, bool primary = false)
    {
        button.FlatStyle = FlatStyle.Flat;
        button.FlatAppearance.BorderSize = 1;
        button.FlatAppearance.BorderColor = primary
            ? Color.FromArgb(148, 138, 255)
            : Color.FromArgb(46, 255, 255, 255);
        button.BackColor = primary
            ? Color.FromArgb(132, 122, 255)
            : Color.FromArgb(36, 40, 52);
        button.ForeColor = Color.FromArgb(245, 247, 250);
        button.UseVisualStyleBackColor = false;
        button.Font = new Font("Segoe UI", 8.5f, primary ? FontStyle.Bold : FontStyle.Regular);
        button.Cursor = Cursors.Hand;
    }

    private void PaintIslandSurface(Graphics graphics, Rectangle bounds)
    {
        if (bounds.Width < 4 || bounds.Height < 4)
            return;
        graphics.SmoothingMode = SmoothingMode.AntiAlias;
        var radius = ScaleLogical(_tucked ? 14 : _expanded ? 16 : 18);
        using var path = RoundedRectangle(new RectangleF(1, 1, bounds.Width - 3, bounds.Height - 3), radius);

        // Tier 1 base dark glass tint (#141820)
        using (var baseFill = new SolidBrush(Color.FromArgb(20, 24, 32)))
            graphics.FillPath(baseFill, path);

        // Directional state-reactive radial/linear wash (#wash in HTML) with gentle breathing
        var visual = _avatar.State;
        var breath = 0.72f + 0.28f * (0.5f + 0.5f * MathF.Sin(_washPhase * MathF.Tau));
        var (washStart, washEnd, borderStart, borderEnd) = visual switch
        {
            IslandVisualState.Listening => (
                Color.FromArgb((int)(56 * breath), 124, 156, 255),
                Color.FromArgb(10, 124, 156, 255),
                Color.FromArgb(168, 124, 156, 255),
                Color.FromArgb(72, 124, 156, 255)),
            IslandVisualState.Speaking or IslandVisualState.Thinking => (
                Color.FromArgb((int)(56 * breath), 176, 108, 255),
                Color.FromArgb(10, 124, 156, 255),
                Color.FromArgb(168, 176, 108, 255),
                Color.FromArgb(72, 124, 156, 255)),
            IslandVisualState.Attention or IslandVisualState.Interrupted => (
                Color.FromArgb((int)(54 * breath), 255, 184, 107),
                Color.FromArgb(8, 255, 184, 107),
                Color.FromArgb(172, 255, 184, 107),
                Color.FromArgb(68, 255, 184, 107)),
            IslandVisualState.Muted or IslandVisualState.Error => (
                Color.FromArgb(30, 255, 93, 93),
                Color.FromArgb(6, 255, 93, 93),
                Color.FromArgb(115, 255, 93, 93),
                Color.FromArgb(46, 255, 255, 255)),
            _ => (
                Color.FromArgb(22, 124, 156, 255),
                Color.FromArgb(8, 176, 108, 255),
                Color.FromArgb(58, 255, 255, 255),
                Color.FromArgb(34, 255, 255, 255))
        };

        using (var wash = new LinearGradientBrush(bounds, washStart, washEnd, 35f))
            graphics.FillPath(wash, path);

        // iOS/iPhone frosted glass top specular sheen + crisp inner glass rim
        var sheenHeight = Math.Max(16, (int)(bounds.Height * 0.36f));
        var sheenRect = new Rectangle(bounds.X, bounds.Y, bounds.Width, sheenHeight);
        using (var sheen = new LinearGradientBrush(
                   sheenRect,
                   Color.FromArgb(26, 255, 255, 255),
                   Color.FromArgb(0, 255, 255, 255),
                   LinearGradientMode.Vertical))
        {
            graphics.SetClip(path);
            graphics.FillRectangle(sheen, sheenRect);
            graphics.ResetClip();
        }

        if (bounds.Width > 8 && bounds.Height > 8)
        {
            using var innerPath = RoundedRectangle(
                new RectangleF(2.2f, 2.2f, bounds.Width - 5.4f, bounds.Height - 5.4f),
                Math.Max(4, radius - 1));
            using var innerRim = new LinearGradientBrush(
                bounds,
                Color.FromArgb(44, 255, 255, 255),
                Color.FromArgb(10, 255, 255, 255),
                LinearGradientMode.Vertical);
            using var innerPen = new Pen(innerRim, 1f);
            graphics.DrawPath(innerPen, innerPath);
        }

        using var border = new LinearGradientBrush(bounds, borderStart, borderEnd, 35f);
        using var pen = new Pen(border, Math.Max(1.3f, DeviceDpi / 64f));
        graphics.DrawPath(pen, path);
    }

    private void PaintInputPill(Graphics graphics, Rectangle bounds)
    {
        if (bounds.Width < 4 || bounds.Height < 4)
            return;
        graphics.SmoothingMode = SmoothingMode.AntiAlias;
        using var path = RoundedRectangle(
            new RectangleF(0.5f, 0.5f, bounds.Width - 1, bounds.Height - 1), ScaleLogical(17));
        using var fill = new SolidBrush(Color.FromArgb(20, 24, 34));
        var borderColor = _textInput.Focused
            ? Color.FromArgb(185, 124, 156, 255)
            : Color.FromArgb(52, 255, 255, 255);
        using var border = new Pen(borderColor, Math.Max(1f, ScaleLogical(1)));
        graphics.FillPath(fill, path);
        graphics.DrawPath(border, path);
    }

    private void PaintApprovalBorder(Graphics graphics, Rectangle bounds)
    {
        if (bounds.Width < 4 || bounds.Height < 4)
            return;
        graphics.SmoothingMode = SmoothingMode.AntiAlias;
        using var path = RoundedRectangle(
            new RectangleF(0.5f, 0.5f, bounds.Width - 1, bounds.Height - 1), ScaleLogical(12));
        using var border = new Pen(Color.FromArgb(128, 255, 184, 107), Math.Max(1f, ScaleLogical(1)));
        graphics.DrawPath(border, path);
    }

    private void UpdateConversationRegion()
    {
        if (_conversation.ClientSize.Width <= 0 || _conversation.ClientSize.Height <= 0)
            return;
        using var path = RoundedRectangle(
            new RectangleF(0, 0, _conversation.Width, _conversation.Height),
            ScaleLogical(12));
        var prev = _conversation.Region;
        _conversation.Region = new Region(path);
        prev?.Dispose();
    }

    private void UpdateApprovalRegion()
    {
        if (_approvalCard.ClientSize.Width <= 0 || _approvalCard.ClientSize.Height <= 0)
            return;
        using var path = RoundedRectangle(
            new RectangleF(0, 0, _approvalCard.ClientSize.Width, _approvalCard.ClientSize.Height),
            ScaleLogical(12));
        var previous = _approvalCard.Region;
        _approvalCard.Region = new Region(path);
        previous?.Dispose();
    }

    private void UpdateIslandRegion()
    {
        if (_surface.ClientSize.Width <= 0 || _surface.ClientSize.Height <= 0)
            return;
        var radius = ScaleLogical(_tucked ? 14 : _expanded ? 16 : 18);
        using var path = RoundedRectangle(
            new RectangleF(0, 0, _surface.ClientSize.Width, _surface.ClientSize.Height), radius);
        var oldSurfaceRegion = _surface.Region;
        _surface.Region = new Region(path);
        oldSurfaceRegion?.Dispose();

        var oldFormRegion = Region;
        Region = new Region(path);
        oldFormRegion?.Dispose();

        _backdrop?.SyncToOwner(Bounds, Region, Visible && Opacity > 0.05);
        _surface.Invalidate();
    }

    private static GraphicsPath RoundedRectangle(RectangleF bounds, float radius)
    {
        var path = new GraphicsPath();
        var diameter = Math.Min(radius * 2, Math.Min(bounds.Width, bounds.Height));
        if (diameter <= 0)
        {
            path.AddRectangle(bounds);
            return path;
        }
        var arc = new RectangleF(bounds.X, bounds.Y, diameter, diameter);
        path.AddArc(arc, 180, 90);
        arc.X = bounds.Right - diameter;
        path.AddArc(arc, 270, 90);
        arc.Y = bounds.Bottom - diameter;
        path.AddArc(arc, 0, 90);
        arc.X = bounds.X;
        path.AddArc(arc, 90, 90);
        path.CloseFigure();
        return path;
    }

    private int ScaleLogical(int value) =>
        (int)Math.Round(value * (DeviceDpi > 0 ? DeviceDpi : 96) / 96d);

    private Size IslandSize(Screen screen, bool expanded)
    {
        var area = screen.WorkingArea;
        var logical = expanded
            ? new Size(ScaleLogical(380), ScaleLogical(560))
            : new Size(ScaleLogical(64), ScaleLogical(64));
        return new Size(Math.Min(logical.Width, area.Width), Math.Min(logical.Height, area.Height));
    }

    private Size EdgeTabSize() => new(
        Math.Max(1, ScaleLogical(22)),
        Math.Max(1, ScaleLogical(56)));

    private void RestoreSavedPlacement()
    {
        var screens = Screen.AllScreens;
        if (screens.Length == 0)
            return;
        var fallback = Screen.PrimaryScreen ?? screens[0];
        var screen = screens.FirstOrDefault(candidate =>
                          candidate.DeviceName.Equals(_settings.IslandMonitorDeviceName,
                              StringComparison.OrdinalIgnoreCase))
                      ?? fallback;
        var size = IslandSize(screen, _expanded);
        var hasSavedPosition = _settings.IslandLeft != int.MinValue &&
                               _settings.IslandTop != int.MinValue;
        var location = hasSavedPosition
            ? new Point(_settings.IslandLeft, _settings.IslandTop)
            : new Point(screen.WorkingArea.Right - size.Width - ScaleLogical(24),
                screen.WorkingArea.Top + Math.Max(ScaleLogical(48), screen.WorkingArea.Height / 3));
        Bounds = IslandPlacement.ClampToWorkArea(
            new Rectangle(location, size), screen.WorkingArea);
        _restoreBounds = Bounds;
        _restoreMonitorDeviceName = screen.DeviceName;
        UpdateIslandRegion();
    }

    private void LayoutHeader()
    {
        var headerHeight = ScaleLogical(64);
        _header.Height = headerHeight;
        var avatarSize = ScaleLogical(_expanded ? 40 : 36);
        _avatar.Size = new Size(avatarSize, avatarSize);
        _avatar.Location = _expanded
            ? new Point(ScaleLogical(12), (headerHeight - avatarSize) / 2)
            : new Point((_header.ClientSize.Width - avatarSize) / 2,
                Math.Max(ScaleLogical(6), (_header.ClientSize.Height - avatarSize) / 2 - ScaleLogical(2)));

        var eqW = ScaleLogical(22);
        var eqH = ScaleLogical(14);
        _collapsedEqualizer.Size = new Size(eqW, eqH);
        _collapsedEqualizer.Location = new Point(
            (_header.ClientSize.Width - eqW) / 2,
            _header.ClientSize.Height - eqH - ScaleLogical(4));

        _title.Visible = _expanded;
        _connection.Visible = _expanded;
        _connectionDot.Visible = _expanded;
        _muteButton.Visible = _expanded;
        _moreButton.Visible = _expanded;
        _tuckButton.Visible = _expanded;

        if (!_expanded)
        {
            _headerEqualizer.Visible = false;
            return;
        }
        _collapsedEqualizer.Visible = false;

        var left = ScaleLogical(62);
        _title.Location = new Point(left, ScaleLogical(12));
        var dotSize = Math.Max(6, ScaleLogical(7));
        _connectionDot.Size = new Size(dotSize, dotSize);
        _connectionDot.Location = new Point(left + ScaleLogical(1), ScaleLogical(38));
        _connection.Location = new Point(left + dotSize + ScaleLogical(6), ScaleLogical(34));

        var btnSize = ScaleLogical(32);
        var btnGap = ScaleLogical(6);
        var right = _header.ClientSize.Width - ScaleLogical(12);
        _tuckButton.Size = new Size(btnSize, btnSize);
        _tuckButton.Location = new Point(right - btnSize, (headerHeight - btnSize) / 2);
        right = _tuckButton.Left - btnGap;

        _moreButton.Size = new Size(btnSize, btnSize);
        _moreButton.Location = new Point(right - btnSize, (headerHeight - btnSize) / 2);
        right = _moreButton.Left - btnGap;

        _muteButton.Size = new Size(btnSize, btnSize);
        _muteButton.Location = new Point(right - btnSize, (headerHeight - btnSize) / 2);
        right = _muteButton.Left - ScaleLogical(8);

        _headerEqualizer.Size = new Size(eqW, eqH);
        _headerEqualizer.Location = new Point(right - eqW, (headerHeight - eqH) / 2);

        UpdateDockingIcons();
        _muteButton.AccessibleName = _shortcuts.IsMuted
            ? "Unmute microphone input"
            : "Mute microphone input";
    }

    private void UpdateDockingIcons()
    {
        if (_tucked || Screen.AllScreens.Length == 0)
            return;
        var workArea = Screen.FromRectangle(Bounds).WorkingArea;
        var dockedRight = Bounds.Left + Bounds.Width / 2 >= workArea.Left + workArea.Width / 2;
        _tuckButton.Icon = dockedRight ? IslandIcon.ChevronRight : IslandIcon.ChevronLeft;
    }

    private void ApplyExpandedLayout(bool expanded, bool animate, bool save)
    {
        if (_tucked)
            RestoreIsland(allowFullscreenOverride: true);

        _expanded = expanded;
        if (expanded)
        {
            _tuckMode = false;
            _retuckTimer.Stop();
            _peekPopup.HidePopup();
            _toastPopup.HideToast();
        }
        else if (_settingsOpen)
        {
            _settingsOpen = false;
            _settingsSheet.Visible = false;
        }

        _header.Visible = !_settingsOpen;
        _body.Visible = _expanded && !_settingsOpen;
        _settingsSheet.Visible = _expanded && _settingsOpen;
        _edgeTab.Visible = false;
        LayoutHeader();
        LayoutComposerRow();
        LayoutApprovalControls();

        if (Screen.AllScreens.Length == 0)
            return;
        var screen = Screen.FromRectangle(Bounds);
        var workArea = screen.WorkingArea;
        var size = IslandSize(screen, _expanded);
        var snapDistance = ScaleLogical(28);
        var nearLeft = Math.Abs(Bounds.Left - workArea.Left) <= snapDistance;
        var nearRight = Math.Abs(workArea.Right - Bounds.Right) <= snapDistance;
        var left = nearLeft
            ? workArea.Left
            : nearRight
                ? workArea.Right - size.Width
                : Bounds.Left;
        var target = IslandPlacement.ClampToWorkArea(
            new Rectangle(left, Bounds.Top, size.Width, size.Height), workArea);
        AnimateBounds(target, animate);
        _restoreBounds = target;
        _restoreMonitorDeviceName = screen.DeviceName;
        UpdateIslandRegion();
        RefreshVoiceIndicator(_voice.Snapshot);
        if (save)
            SavePlacement();
    }

    private void ToggleExpanded() => ApplyExpandedLayout(!_expanded, animate: true, save: true);

    private void ScheduleAutoRetuckIfNeeded()
    {
        if (_tuckMode && !_tucked && !_expanded && !_dragging)
        {
            _retuckTimer.Stop();
            _retuckTimer.Start();
        }
    }

    private void AnimateBounds(Rectangle target, bool animate)
    {
        _transitionTimer.Stop();
        _transitionTarget = target;
        if (!animate || !Visible || Bounds == target)
        {
            Bounds = target;
            UpdateIslandRegion();
            return;
        }
        _backdrop?.SyncToOwner(Bounds, Region, ownerVisible: false);
        _transitionStart = Bounds;
        _transitionStarted = System.Diagnostics.Stopwatch.GetTimestamp();
        _transitionTimer.Start();
    }

    private void AdvanceWindowTransition()
    {
        const double durationMilliseconds = 160;
        var elapsed = System.Diagnostics.Stopwatch.GetElapsedTime(_transitionStarted).TotalMilliseconds;
        var progress = Math.Clamp(elapsed / durationMilliseconds, 0, 1);
        var eased = 1 - Math.Pow(1 - progress, 3);
        var previousSize = Size;
        Bounds = new Rectangle(
            Interpolate(_transitionStart.Left, _transitionTarget.Left, eased),
            Interpolate(_transitionStart.Top, _transitionTarget.Top, eased),
            Interpolate(_transitionStart.Width, _transitionTarget.Width, eased),
            Interpolate(_transitionStart.Height, _transitionTarget.Height, eased));
        if (Size != previousSize)
            UpdateIslandRegion();
        if (progress < 1)
            return;
        _transitionTimer.Stop();
        Bounds = _transitionTarget;
        UpdateIslandRegion();
    }

    private static int Interpolate(int from, int to, double progress) =>
        (int)Math.Round(from + (to - from) * progress);

    private void OnHeaderMouseDown(object? sender, MouseEventArgs e)
    {
        if (e.Button != MouseButtons.Left || _tucked)
            return;
        _retuckTimer.Stop();
        _hoverOutsideSinceTicks = null;
        _dragging = true;
        _dragMoved = false;
        _dragStartBounds = Bounds;
        _dragStartPointer = Cursor.Position;
        _transitionTimer.Stop();
        _backdrop?.SetDragging(true);
        _backdrop?.SyncToOwner(Bounds, Region, ownerVisible: false);
        _dragCaptureControl = sender as Control ?? _header;
        _dragCaptureControl.Capture = true;
    }

    private void OnHeaderMouseMove(object? sender, MouseEventArgs e)
    {
        if (!_dragging)
            return;
        if (Control.MouseButtons != MouseButtons.Left)
        {
            _dragging = false;
            _backdrop?.SetDragging(false);
            if (_dragCaptureControl is not null)
            {
                _dragCaptureControl.Capture = false;
                _dragCaptureControl = null;
            }
            return;
        }
        var pointer = Cursor.Position;
        var dx = pointer.X - _dragStartPointer.X;
        var dy = pointer.Y - _dragStartPointer.Y;
        if (Math.Abs(dx) + Math.Abs(dy) > ScaleLogical(3))
            _dragMoved = true;
        var screen = Screen.FromPoint(pointer);
        var target = new Rectangle(_dragStartBounds.X + dx, _dragStartBounds.Y + dy,
            _dragStartBounds.Width, _dragStartBounds.Height);
        Bounds = IslandPlacement.ClampToWorkArea(target, screen.WorkingArea);
        UpdateDockingIcons();
    }

    private void OnHeaderMouseUp(object? sender, MouseEventArgs e)
    {
        if (!_dragging)
        {
            if (e.Button == MouseButtons.Left && Environment.TickCount64 - _lastEdgeAutoRestoreTicks <= 800)
                ShowWindow();
            return;
        }
        if (e.Button != MouseButtons.Left)
            return;
        _dragging = false;
        _backdrop?.SetDragging(false);
        if (_dragCaptureControl is not null)
        {
            _dragCaptureControl.Capture = false;
            _dragCaptureControl = null;
        }
        _header.Capture = false;
        if (!_dragMoved)
        {
            _manualRestoreUntilTicks = Environment.TickCount64 + 4000;
            Opacity = ActiveGlassOpacity;
            TopMost = true;
            BringToFront();
            if (!_expanded)
                ShowWindow();
            else
                ToggleExpanded();
            return;
        }

        _tuckMode = false;
        var screen = Screen.FromPoint(Cursor.Position);
        Bounds = IslandPlacement.SnapToEdges(Bounds, screen.WorkingArea, ScaleLogical(24));
        _backdrop?.SyncToOwner(Bounds, Region, Visible && Opacity > 0.05);
        UpdateDockingIcons();
        _restoreBounds = Bounds;
        _restoreMonitorDeviceName = screen.DeviceName;
        SavePlacement();
    }

    private void OnHeaderMouseCaptureChanged(object? sender, EventArgs e)
    {
        if (!_dragging)
            return;
        if (Control.MouseButtons == MouseButtons.Left)
            return;
        _dragging = false;
        _backdrop?.SetDragging(false);
        _dragCaptureControl = null;
    }

    private void TuckIsland(bool dueToFullscreen = false)
    {
        if (_tucked)
            return;
        _transitionTimer.Stop();
        _retuckTimer.Stop();
        _hoverOutsideSinceTicks = null;
        _lastTuckedTicks = Environment.TickCount64;
        _restoreBounds = Bounds;
        var screen = Screen.FromRectangle(Bounds);
        _restoreMonitorDeviceName = screen.DeviceName;
        var workArea = screen.WorkingArea;
        var leftDistance = Math.Abs(Bounds.Left - workArea.Left);
        var rightDistance = Math.Abs(workArea.Right - Bounds.Right);
        var edge = leftDistance <= rightDistance ? IslandDockEdge.Left : IslandDockEdge.Right;
        _tuckedEdge = edge;
        var tabBounds = IslandPlacement.GetTuckedBounds(_restoreBounds, workArea,
            edge, EdgeTabSize());
        _tucked = true;
        _tuckedForFullscreen = dueToFullscreen;
        _header.Visible = false;
        _body.Visible = false;
        _settingsSheet.Visible = false;
        _edgeTab.Visible = true;
        _edgeTab.IsDockedRight = edge == IslandDockEdge.Right;
        _edgeTab.Icon = edge == IslandDockEdge.Left
            ? IslandIcon.ChevronRight : IslandIcon.ChevronLeft;
        _edgeTab.IsMuted = _shortcuts.IsMuted;
        _edgeTab.IsDimmed = _fullscreenActive;
        _edgeTab.ActivityState = _avatar.State;
        Bounds = tabBounds;
        UpdateIslandRegion();
        UpdateSatellitePopups();
        if (!dueToFullscreen)
            SavePlacement();
    }

    private void RestoreIsland(bool allowFullscreenOverride = false, bool preserveTuckMode = false)
    {
        if (!_fullscreenActive)
            _tuckedForFullscreen = false;
        if (!_tucked || (_fullscreenActive && !allowFullscreenOverride))
            return;
        if (!preserveTuckMode)
            _tuckMode = false;

        var screen = Screen.AllScreens.FirstOrDefault(candidate =>
                          candidate.DeviceName.Equals(_restoreMonitorDeviceName,
                              StringComparison.OrdinalIgnoreCase))
                      ?? Screen.PrimaryScreen
                      ?? Screen.AllScreens.First();
        var showExpanded = preserveTuckMode ? false : _expanded;
        var targetSize = IslandSize(screen, showExpanded);
        var peekX = _tuckedEdge == IslandDockEdge.Left
            ? screen.WorkingArea.Left
            : screen.WorkingArea.Right - targetSize.Width;
        var baseRect = preserveTuckMode
            ? new Rectangle(peekX, Bounds.Top, targetSize.Width, targetSize.Height)
            : new Rectangle(_restoreBounds.Location, targetSize);
        var restored = IslandPlacement.ClampToWorkArea(baseRect, screen.WorkingArea);
        _tucked = false;
        _expanded = showExpanded;
        _header.Visible = !(showExpanded && _settingsOpen);
        _body.Visible = showExpanded && !_settingsOpen;
        _settingsSheet.Visible = showExpanded && _settingsOpen;
        _edgeTab.Visible = false;
        LayoutHeader();
        AnimateBounds(restored, animate: true);
        if (preserveTuckMode)
        {
            _hoverOutsideSinceTicks = null;
            _retuckTimer.Start();
        }
        else
        {
            _retuckTimer.Stop();
            _restoreBounds = restored;
        }
        _restoreMonitorDeviceName = screen.DeviceName;
        UpdateIslandRegion();
        UpdateSatellitePopups();
        if (!preserveTuckMode)
            SavePlacement();
    }

    private void SavePlacement()
    {
        var bounds = _tucked ? _restoreBounds : Bounds;
        var screen = Screen.AllScreens.FirstOrDefault(candidate =>
                         candidate.DeviceName.Equals(_restoreMonitorDeviceName,
                             StringComparison.OrdinalIgnoreCase))
                     ?? Screen.FromRectangle(bounds);
        _settings.IslandLeft = bounds.Left;
        _settings.IslandTop = bounds.Top;
        _settings.IslandMonitorDeviceName = screen.DeviceName;
        _settings.IslandExpanded = _expanded;
        try
        {
            _settings.Save();
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
        {
            _connection.Text = "Position could not be saved.";
            NotifyUser("Lila settings", "Island position was not saved. " + SafeUiError(ex),
                ToolTipIcon.Warning);
        }
    }

    private void CheckFullscreenState()
    {
        if (IsDisposed || !IsHandleCreated)
            return;
        var fullscreen = IslandNativeWindow.IsFullscreenApplicationActive(Handle);
        if (fullscreen == _fullscreenActive)
            return;
        _fullscreenActive = fullscreen;
        _edgeTab.IsDimmed = fullscreen;
        _screenOverlay.SetFullscreenSuppressed(fullscreen);
        if (fullscreen)
        {
            if (!_tucked && !_expanded && Environment.TickCount64 >= _manualRestoreUntilTicks)
                TuckIsland(dueToFullscreen: true);
        }
        else if (_tuckedForFullscreen)
        {
            RestoreIsland();
        }
    }

    private void OnDisplaySettingsChanged(object? sender, EventArgs e)
    {
        if (IsDisposed || !IsHandleCreated)
            return;
        try
        {
            BeginInvoke(new Action(RecoverPlacementAfterDisplayChange));
        }
        catch (InvalidOperationException)
        {
        }
    }

    private void RecoverPlacementAfterDisplayChange()
    {
        if (_tucked)
        {
            var screen = Screen.AllScreens.FirstOrDefault(candidate =>
                              candidate.DeviceName.Equals(_restoreMonitorDeviceName,
                                  StringComparison.OrdinalIgnoreCase))
                          ?? Screen.PrimaryScreen
                          ?? Screen.AllScreens.First();
            _restoreBounds = IslandPlacement.ClampToWorkArea(_restoreBounds, screen.WorkingArea);
            _restoreMonitorDeviceName = screen.DeviceName;
            Bounds = IslandPlacement.GetTuckedBounds(_restoreBounds, screen.WorkingArea,
                _restoreBounds.Left + _restoreBounds.Width / 2 <
                screen.WorkingArea.Left + screen.WorkingArea.Width / 2
                    ? IslandDockEdge.Left : IslandDockEdge.Right,
                EdgeTabSize());
        }
        else
        {
            var screen = Screen.FromRectangle(Bounds);
            Bounds = IslandPlacement.ClampToWorkArea(
                new Rectangle(Bounds.Location, IslandSize(screen, _expanded)), screen.WorkingArea);
            _restoreBounds = Bounds;
            _restoreMonitorDeviceName = screen.DeviceName;
            SavePlacement();
        }
        LayoutHeader();
        LayoutComposerRow();
        LayoutApprovalControls();
        UpdateIslandRegion();
        UpdateSatellitePopups();
    }

    private double ActiveGlassOpacity => _backdrop?.IsGlassActive == true ? 0.87 : 0.94;

    protected override void OnHandleCreated(EventArgs e)
    {
        base.OnHandleCreated(e);
        if (_backdrop is null && !_startMinimized)
        {
            _backdrop = new IslandBackdropForm(this);
        }
        IslandNativeWindow.ApplyGlassBackdrop(Handle);
        Opacity = ActiveGlassOpacity;
    }

    protected override void OnVisibleChanged(EventArgs e)
    {
        base.OnVisibleChanged(e);
        if (Visible && _backdrop is null)
        {
            _backdrop = new IslandBackdropForm(this);
        }
        _backdrop?.SyncToOwner(Bounds, Region, Visible && Opacity > 0.05);
        if (Visible)
            Opacity = ActiveGlassOpacity;
    }

    protected override void OnDpiChanged(DpiChangedEventArgs e)
    {
        base.OnDpiChanged(e);
        if (IsDisposed || !IsHandleCreated)
            return;
        BeginInvoke(new Action(RecoverPlacementAfterDisplayChange));
    }

    private async Task RefreshStatusAsync()
    {
        if (IsDisposed || !await _connectionGate.WaitAsync(0))
            return;
        try
        {
            await _lifecycle.EnsureRunningAsync();
            var result = await _wsl.CheckAsync();
            if (!_serviceManuallyStopped &&
                result.State is not (LilaAvailability.Ready or LilaAvailability.AuthenticationFailure))
            {
                _shortcuts.SetReady(false);
                _isReady = false;
                _voiceToggleButton.Enabled = false;
                result = await _wsl.StartLilaAsync();
            }
            ApplyAvailability(result);
            if (result.State == LilaAvailability.Ready)
                await RefreshVoiceStatusAsync();
            else
            {
                _remoteVoiceStatus = null;
                _voice.ObserveRemoteStatus(null);
                _voiceStatusError = null;
                RefreshVoiceIndicator(_voice.Snapshot);
            }
        }
        catch (Exception ex)
        {
            var wasReady = _isReady;
            _remoteVoiceStatus = null;
            _shortcuts.ApplyRuntimeState(false, null);
            _isReady = false;
            _connection.Text = "Disconnected — " + SafeUiError(ex);
            _connectionDotColor = Color.FromArgb(255, 93, 93);
            _connectionDot.Invalidate();
            _startButton.Enabled = true;
            _stopButton.Enabled = false;
            _tray.Text = "Lila — disconnected";
            _voiceStatusError = SafeUiError(ex);
            RefreshVoiceIndicator(_voice.Snapshot);
            if (wasReady)
                NotifyUser("Lila disconnected", "The connection to Lila was lost.", ToolTipIcon.Warning);
        }
        finally
        {
            _connectionGate.Release();
        }
    }

    private void ApplyAvailability(AvailabilityResult result)
    {
        var wasReady = _isReady;
        var ready = result.State == LilaAvailability.Ready;
        _isReady = ready;
        if (!ready)
            _remoteVoiceStatus = null;
        _shortcuts.ApplyRuntimeState(ready, _remoteVoiceStatus);
        _voiceToggleButton.Enabled = ready && _voice.Snapshot.State is
            (VoiceInteractionState.Idle or VoiceInteractionState.Response or
             VoiceInteractionState.Starting or VoiceInteractionState.Recording);
        _connection.Text = result.State switch
        {
            LilaAvailability.WslUnavailable => "WSL unavailable",
            LilaAvailability.ServiceInactive => "Lila service inactive",
            LilaAvailability.AuthenticationFailure => "Authentication failed",
            LilaAvailability.ApiConnectionFailure => "API unavailable",
            LilaAvailability.Starting => "Starting…",
            _ => "Connected"
        };
        _connectionDotColor = ready
            ? Color.FromArgb(94, 224, 160)
            : result.State == LilaAvailability.Starting
                ? Color.FromArgb(255, 184, 107)
                : Color.FromArgb(255, 93, 93);
        _connectionDot.Invalidate();
        if (ready && _keyboard is null)
            _connection.Text = "Connected · keyboard unavailable";
        _startButton.Enabled = !ready;
        _stopButton.Enabled = ready || result.State == LilaAvailability.Starting;
        _tray.Text = ready ? "Lila — connected" : "Lila — disconnected";
        RefreshVoiceIndicator(_voice.Snapshot);
        if (ready != wasReady)
            NotifyUser(ready ? "Lila connected" : "Lila disconnected",
                ready ? "Lila is ready." : "Lila is unavailable.",
                ready ? ToolTipIcon.Info : ToolTipIcon.Warning);
    }

    private bool RemoteVoiceSessionIsActive() =>
        _remoteVoiceStatus?.State is "recording" or "transcribing" or "cancelling" or "timed_out";

    private async Task RefreshVoiceStatusAsync()
    {
        try
        {
            _remoteVoiceStatus = await _api.GetVoiceStatusAsync();
            _voiceStatusError = null;
            _shortcuts.ApplyRuntimeState(_isReady, _remoteVoiceStatus);
        }
        catch (Exception ex)
        {
            _remoteVoiceStatus = null;
            _voice.ObserveRemoteStatus(null);
            _voiceStatusError = SafeUiError(ex);
        }
        RefreshVoiceIndicator(_voice.Snapshot);
    }

    private void StartEventStreamLoop()
    {
        if (_eventStreamCts is not null)
            return;
        _eventStreamCts = new CancellationTokenSource();
        _eventStream = new DaemonEventStreamClient(_api);
        _eventStream.EventReceived += ev =>
        {
            if (IsDisposed || !IsHandleCreated)
                return;
            try
            {
                BeginInvoke(() => HandleDaemonEvent(ev));
            }
            catch (InvalidOperationException) { }
        };
        _eventStream.GapDetected += () =>
        {
            if (IsDisposed || !IsHandleCreated)
                return;
            try
            {
                BeginInvoke(async () => await RefreshApprovalsAsync());
            }
            catch (InvalidOperationException) { }
        };

        var ct = _eventStreamCts.Token;
        _ = Task.Run(async () =>
        {
            while (!ct.IsCancellationRequested)
            {
                try
                {
                    await _eventStream.ConsumeSseStreamOnceAsync(ct).ConfigureAwait(false);
                }
                catch (OperationCanceledException)
                {
                    break;
                }
                catch
                {
                    try { await Task.Delay(TimeSpan.FromSeconds(2), ct).ConfigureAwait(false); }
                    catch (OperationCanceledException) { break; }
                }
            }
        }, ct);
    }

    private void HandleDaemonEvent(DaemonEventDto ev)
    {
        if (IsDisposed)
            return;
        if (ev.EventType.StartsWith("approval.", StringComparison.Ordinal))
        {
            _ = RefreshApprovalsAsync();
            return;
        }
        if (ev.EventType.StartsWith("task.", StringComparison.Ordinal))
        {
            var kind = ev.Payload.TryGetProperty("kind", out var kProp) ? kProp.GetString() ?? "" : "";
            if (string.Equals(kind, "chat", StringComparison.OrdinalIgnoreCase))
                return;
            // Only surface a result/outcome card when the task has finished, never when first submitted.
            if (ev.EventType is not ("task.completed" or "task.failed" or "task.uncertain"))
                return;
            var taskId = ev.Payload.TryGetProperty("task_id", out var tid) ? tid.GetString() ?? $"task-{ev.EventId}" : $"task-{ev.EventId}";
            var desc = ev.Payload.TryGetProperty("description", out var d) ? d.GetString() ?? ev.EventType : ev.EventType;
            var status = ev.Payload.TryGetProperty("status", out var st) ? st.GetString() ?? "" : "";
            var item = new ScreenItem
            {
                Id = taskId,
                Kind = ev.EventType == "task.completed" ? ScreenItemKind.Research : ScreenItemKind.Task,
                Title = desc,
                SourceLabel = ev.EventType == "task.completed" ? "Result ready" : "Task status",
                TaskStage = ev.EventType == "task.completed"
                    ? ScreenTaskStage.CompletedNotVerified
                    : ScreenTaskStage.Dispatched,
                Summary = string.IsNullOrEmpty(status) ? ev.EventType : $"Status: {status}"
            };
            if (IsHandleCreated)
                BeginInvoke(() => _screenStore.Upsert(item));
            _ = AttachLatestTraceEvidenceAsync(taskId, desc, status);
            return;
        }
        if (ev.EventType == "voice.transcript")
        {
            var text = ev.Payload.TryGetProperty("accumulated", out var acc)
                ? acc.GetString() ?? ""
                : (ev.Payload.TryGetProperty("text", out var txt) ? txt.GetString() ?? "" : "");
            var isFinal = ev.Payload.TryGetProperty("is_final", out var fin) &&
                          fin.ValueKind == System.Text.Json.JsonValueKind.True;
            _liveTranscriptText = isFinal ? null : (string.IsNullOrWhiteSpace(text) ? _liveTranscriptText : text);
            if (!string.IsNullOrWhiteSpace(text))
            {
                _screenOverlay.PushCaption("You", text, isFinal: isFinal);
                if (_tucked || !_expanded)
                    ShowPeekMessage(text);
            }
            RefreshVoiceIndicator(_voice.Snapshot);
            return;
        }
        if (ev.EventType == "notification.posted")
        {
            var title = ev.Payload.TryGetProperty("title", out var t) ? t.GetString() ?? "Lila" : "Lila";
            var summary = ev.Payload.TryGetProperty("summary", out var s) ? s.GetString() ?? "" : "";
            var notif = new ScreenItem
            {
                Id = $"notif-{ev.EventId}",
                Kind = ScreenItemKind.Research,
                Title = title,
                Summary = summary
            };
            if (IsHandleCreated)
                BeginInvoke(() => _screenStore.Upsert(notif));
        }
    }

    private async Task AttachLatestTraceEvidenceAsync(string taskId, string desc, string status)
    {
        try
        {
            var traces = await _api.GetTracesAsync(limit: 5).ConfigureAwait(false);
            if (traces.Traces.Count == 0 || IsDisposed || !IsHandleCreated)
                return;
            var latest = traces.Traces[0];
            var verStatus = latest.TryGetProperty("verification_status", out var vs) ? vs.GetString() : null;
            var exitCode = latest.TryGetProperty("exit_code", out var ec) && ec.ValueKind == System.Text.Json.JsonValueKind.Number
                ? ec.GetInt32().ToString()
                : null;
            if (string.IsNullOrEmpty(verStatus) && string.IsNullOrEmpty(exitCode))
                return;
            var evidence = $"Verification: {verStatus ?? "unknown"}" +
                           (exitCode is not null ? $" · Exit: {exitCode}" : "");
            BeginInvoke(() => _screenStore.Upsert(new ScreenItem
            {
                Id = taskId,
                Kind = string.Equals(status, "completed", StringComparison.OrdinalIgnoreCase)
                    ? ScreenItemKind.Research
                    : ScreenItemKind.Task,
                Title = desc,
                SourceLabel = string.Equals(status, "completed", StringComparison.OrdinalIgnoreCase)
                    ? "Result ready"
                    : "Task · states from Harness",
                TaskStage = string.Equals(verStatus, "passed", StringComparison.OrdinalIgnoreCase)
                    ? ScreenTaskStage.Verified
                    : ScreenTaskStage.CompletedNotVerified,
                Summary = string.IsNullOrEmpty(status) ? "Trace attached" : $"Status: {status}",
                EvidenceSummary = evidence
            }));
        }
        catch { /* Best-effort evidence enrichment; never block UI event loop. */ }
    }

    private async Task RunOptedInSentinelAsync()
    {
        if (!_settings.SentinelOptIn || IsDisposed)
            return;
        try
        {
            var collector = new WindowsSentinelCollector();
            var snap = await collector.CollectAsync(optedIn: true).ConfigureAwait(false);
            var hostFacts = snap.Facts
                .Select(f => new SentinelHostFactDto(f.Source, f.Metric, f.Value, f.ReadOnly))
                .ToArray();
            var triage = await _api.PostSentinelTriageAsync(hostFacts).ConfigureAwait(false);
            if (!IsDisposed && (triage.HighestTier == "warning" || triage.HighestTier == "critical"))
            {
                BeginInvoke(() => _screenStore.Upsert(new ScreenItem
                {
                    Id = "sentinel-triage",
                    Kind = ScreenItemKind.Research,
                    Title = $"Sentinel ({triage.HighestTier.ToUpperInvariant()})",
                    SourceLabel = "Read-only host telemetry",
                    Summary = $"{triage.Indicators.Count} security indicator(s) detected."
                }));
            }
        }
        catch { /* Non-fatal; coverage gaps remain explicit on daemon side. */ }
    }

    private async Task RefreshApprovalsAsync()
    {
        if (IsDisposed || _approvalBusy)
            return;
        try
        {
            var list = await _api.GetPendingApprovalsAsync();
            var ownerStatus = await _api.GetOwnerStatusAsync();
            var typedAvailable = ownerStatus.Providers.Any(provider =>
                provider.ProviderId == "typed_owner_phrase" && provider.Available && provider.Enrolled);
            var spokenAvailable = ownerStatus.Providers.Any(provider =>
                provider.ProviderId == "spoken_owner_phrase" && provider.Available && provider.Enrolled);
            _approvalStateUnavailable = false;
            var currentIds = list.Requests.Select(item => item.InvocationId).ToHashSet(StringComparer.Ordinal);
            var newCount = _approvalSnapshotLoaded
                ? list.Requests.Count(item => !_knownApprovalIds.Contains(item.InvocationId))
                : 0;
            _knownApprovalIds.Clear();
            foreach (var id in currentIds)
                _knownApprovalIds.Add(id);
            _approvalSnapshotLoaded = true;
            if (newCount > 0)
            {
                NotifyUser("Lila approval required",
                    $"{newCount} pending Harness approval request(s) require your explicit decision.",
                    ToolTipIcon.Warning);
            }

            if (list.Requests.Count == 0)
            {
                _shownApproval = null;
                _approvalCountdownTimer.Stop();
                _approvalPicker.Items.Clear();
                _approvalPicker.Visible = false;
                _approveButton.Enabled = _rejectButton.Enabled = false;
                _voiceApprovalButton.Enabled = false;
                if (_lastApprovalId is not null)
                {
                    var status = await _api.GetApprovalStatusAsync(_lastApprovalId);
                    if (!string.Equals(status.Status, _lastObservedApprovalStatus, StringComparison.Ordinal))
                    {
                        _lastObservedApprovalStatus = status.Status;
                        NotifyUser("Harness approval update",
                            $"Harness reports request state: {status.Status}.",
                            status.Status is "denied" or "expired" ? ToolTipIcon.Warning : ToolTipIcon.Info);
                    }
                }
                _approvalCard.Visible = false;
                _toastPopup.HideToast();
                RefreshVoiceIndicator(_voice.Snapshot);
                return;
            }

            var selectedId = _shownApproval?.InvocationId;
            _approvalPicker.Items.Clear();
            foreach (var request in list.Requests)
            {
                var action = request.DisplayAction ?? request.CapabilityId;
                var choice = new ApprovalChoice($"{action} ({request.InvocationId[..Math.Min(8, request.InvocationId.Length)]})", request);
                _approvalPicker.Items.Add(choice);
            }
            var selectedIndex = list.Requests
                .Select((item, idx) => (item, idx))
                .FirstOrDefault(pair => pair.item.InvocationId == selectedId).idx;
            _approvalPicker.SelectedIndex = selectedIndex;
            _approvalPicker.Visible = list.Requests.Count > 1;
            _approvalCard.Height = ScaleLogical(list.Requests.Count > 1 ? 318 : 292);

            _shownApproval = list.Requests[selectedIndex];
            _lastApprovalId = _shownApproval.InvocationId;
            _lastObservedApprovalStatus = _shownApproval.Status;
            ResetApprovalCountdown(_shownApproval.ExpiresInSeconds);
            _approvalCard.Visible = true;
            _approveButton.Enabled = _rejectButton.Enabled =
                typedAvailable && _shownApproval.ApprovalReady;
            _voiceApprovalButton.Enabled = spokenAvailable && _shownApproval.ApprovalReady;
            ShowApprovalDetails(_shownApproval, list.Requests.Count);
            if (!typedAvailable)
            {
                _approvalDetails.Text += "\r\nOwner authentication is not enrolled locally. Enroll with python -m src.oia.cli init in WSL.";
            }
            else if (!spokenAvailable)
            {
                _approvalDetails.Text += "\r\nVoice approval not enrolled yet: open Menu (⋯) -> Enroll voice approval.";
            }
            UpdateSatellitePopups();
            RefreshVoiceIndicator(_voice.Snapshot);
        }
        catch (Exception ex)
        {
            _shownApproval = null;
            _approvalStateUnavailable = true;
            _approvalCountdownTimer.Stop();
            _approvalPicker.Items.Clear();
            _approvalPicker.Visible = false;
            _approvalCard.Visible = true;
            _approvalDetails.Text = "Approval or owner-authentication state unavailable. No decision is assumed. " +
                                    SafeUiError(ex);
            _approveButton.Enabled = _rejectButton.Enabled = false;
            _voiceApprovalButton.Enabled = false;
            _toastPopup.HideToast();
            RefreshVoiceIndicator(_voice.Snapshot);
        }
    }

    private void ResetApprovalCountdown(double expiresInSeconds)
    {
        _approvalRemainingSeconds = Math.Max(0, expiresInSeconds);
        if (_approvalInitialSeconds < _approvalRemainingSeconds || _approvalInitialSeconds <= 0)
            _approvalInitialSeconds = Math.Max(60.0, _approvalRemainingSeconds);
        _approvalObservedTimestamp = System.Diagnostics.Stopwatch.GetTimestamp();
        UpdateApprovalCountdownUi();
        _approvalCountdownTimer.Start();
    }

    private void TickApprovalCountdown()
    {
        if (_shownApproval is null)
        {
            _approvalCountdownTimer.Stop();
            return;
        }
        var elapsed = System.Diagnostics.Stopwatch.GetElapsedTime(_approvalObservedTimestamp).TotalSeconds;
        var current = Math.Max(0.0, _approvalRemainingSeconds - elapsed);
        _approvalCountdownLabel.Text = $"Rejects automatically in {Math.Max(0, (int)Math.Ceiling(current))} s";
        _approvalProgressBar.Invalidate();
        if (!_expanded || _tucked)
        {
            _toastPopup.ShowOrUpdateToast(_shownApproval, current, _approvalInitialSeconds,
                Bounds, Screen.FromRectangle(Bounds), _voiceApprovalButton.Enabled);
        }
    }

    private void UpdateApprovalCountdownUi()
    {
        _approvalCountdownLabel.Text =
            $"Rejects automatically in {Math.Max(0, (int)Math.Ceiling(_approvalRemainingSeconds))} s";
        _approvalProgressBar.Invalidate();
    }

    private void ShowApprovalDetails(ApprovalRequestView item, int pendingCount)
    {
        var target = item.Target is { } targetJson ? targetJson.GetRawText() : "Not resolved";
        var values = item.DisplayValues.ValueKind == System.Text.Json.JsonValueKind.Undefined
            ? "{}" : item.DisplayValues.GetRawText();
        _approvalHeaderLabel.Text = pendingCount > 1
            ? $"Lila needs your approval ({pendingCount} pending)"
            : "Lila needs your approval";
        _approvalDetails.Text =
            $"{item.DisplayAction ?? "Unreviewed operation"} — {item.CapabilityId} ({item.ContractVersion})\r\n" +
            $"Target ({item.TargetKind ?? "unresolved"}): {target}\r\n" +
            $"Reviewed values: {values}\r\n" +
            $"Policy: {item.PolicyReason} · Risk: {item.Risk ?? "unspecified"}\r\n" +
            $"Detail: {item.PolicyDetail}\r\n" +
            (item.ApprovalReady
                ? "Verified display metadata confirmed by Harness."
                : $"Approval disabled: {item.DisplayBlockReason ?? "operation cannot be safely reviewed"}.");
        if (!item.ApprovalReady)
            _approveButton.Enabled = _rejectButton.Enabled = _voiceApprovalButton.Enabled = false;
    }

    private async Task SubmitApprovalAsync(string decision)
    {
        var request = _shownApproval;
        if (_approvalBusy || request is null)
            return;

        var ownerPhrase = _inlineOwnerPhrase.Text;
        OwnerPhrasePrompt? phrasePrompt = null;
        if (string.IsNullOrWhiteSpace(ownerPhrase))
        {
            phrasePrompt = new OwnerPhrasePrompt();
            var promptResult = phrasePrompt.ShowDialog(this);
            if (promptResult != DialogResult.OK || string.IsNullOrWhiteSpace(phrasePrompt.Phrase))
            {
                phrasePrompt.ClearPhrase();
                phrasePrompt.Dispose();
                return;
            }
            ownerPhrase = phrasePrompt.Phrase;
        }

        _approvalBusy = true;
        _approveButton.Enabled = _rejectButton.Enabled = false;
        _voiceApprovalButton.Enabled = false;
        _approvalDetails.Text += "\r\nSubmitting explicit " + decision.ToUpperInvariant() + " request…";
        try
        {
            _ = await _api.SubmitOwnerDecisionAsync(request, decision, ownerPhrase);
            _lastApprovalId = request.InvocationId;
            var status = await _api.GetApprovalStatusAsync(request.InvocationId);
            _lastObservedApprovalStatus = status.Status;
            _approvalDetails.Text = $"Harness reports {status.Status}: {status.ReasonCode}.";
            AppendConversation("Harness evidence",
                $"Approval decision ({decision}): Harness reports {status.Status} ({status.ReasonCode}).");
            NotifyUser("Harness approval update",
                $"Harness reports request state: {status.Status}.",
                status.Status is "denied" or "expired" ? ToolTipIcon.Warning : ToolTipIcon.Info);
        }
        catch (LilaApiException ex) when (ex.Kind == LilaApiFailureKind.Service)
        {
            _approvalDetails.Text = "Owner decision could not be authenticated or the request is no longer valid. No approval is assumed. " + SafeUiError(ex);
        }
        catch (Exception ex)
        {
            _approvalDetails.Text = "Decision outcome unavailable. Reloading authoritative state is required. " + SafeUiError(ex);
        }
        finally
        {
            ownerPhrase = string.Empty;
            _inlineOwnerPhrase.Clear();
            phrasePrompt?.ClearPhrase();
            phrasePrompt?.Dispose();
            _approvalBusy = false;
            await RefreshApprovalsAsync();
        }
    }

    private async Task SubmitSpokenApprovalAsync()
    {
        var request = _shownApproval;
        if (_approvalBusy || request is null)
            return;
        _approvalBusy = true;
        _approveButton.Enabled = _rejectButton.Enabled = _voiceApprovalButton.Enabled = false;
        string? sessionId = null;
        try
        {
            var started = await _api.StartOwnerVoiceAsync(request);
            sessionId = started.SessionId;
            var prompt = $"Speak your exact Owner Phrase, then say either approve or reject, then repeat these six digits: {started.VoiceChallengeCode}.\r\n\r\nThe recording uses Lila's existing local transcription path and is kept out of conversation. Press OK after speaking.";
            if (MessageBox.Show(this, prompt, "Voice owner authentication",
                    MessageBoxButtons.OKCancel, MessageBoxIcon.Information) != DialogResult.OK)
            {
                await _api.CancelOwnerVoiceAsync(request, sessionId);
                return;
            }
            var result = await _api.StopOwnerVoiceAsync(request, sessionId);
            sessionId = null;
            _lastApprovalId = request.InvocationId;
            _lastObservedApprovalStatus = result.Status;
            _approvalDetails.Text = $"Harness reports {result.Status}: {result.ReasonCode}.";
            NotifyUser("Harness approval update",
                $"Harness reports request state: {result.Status}.",
                result.Status is "denied" or "expired" ? ToolTipIcon.Warning : ToolTipIcon.Info);
        }
        catch (Exception ex)
        {
            _approvalDetails.Text = "Spoken authentication failed or could not be verified. No approval is assumed. " + SafeUiError(ex);
            if (sessionId is not null)
            {
                try { await _api.CancelOwnerVoiceAsync(request, sessionId); }
                catch (Exception cancelError)
                {
                    _approvalDetails.Text += "\r\nVoice cancellation could not be confirmed. " +
                                             SafeUiError(cancelError);
                }
            }
        }
        finally
        {
            _approvalBusy = false;
            await RefreshApprovalsAsync();
        }
    }

    private async Task StartOwnerPttAsync(ApprovalRequestView request)
    {
        if (_approvalBusy || _ownerPttSessionId is not null)
            return;
        try
        {
            var started = await _api.StartOwnerVoiceAsync(request);
            _ownerPttRequest = request;
            _ownerPttSessionId = started.SessionId;
            _voiceToggleButton.ActivityState = IslandVisualState.Listening;
            _toolTip.SetToolTip(_voiceToggleButton,
                $"Private OIA-01 voice approval — say phrase, decision, and {started.VoiceChallengeCode}");
        }
        catch (Exception ex)
        {
            _ownerPttRequest = null;
            _ownerPttSessionId = null;
            _approvalDetails.Text = "Private OIA-01 voice start failed: " + SafeUiError(ex);
        }
    }

    private async Task StopOwnerPttAsync()
    {
        var request = _ownerPttRequest;
        var sessionId = _ownerPttSessionId;
        _ownerPttRequest = null;
        _ownerPttSessionId = null;
        if (request is null || string.IsNullOrEmpty(sessionId))
            return;
        _approvalBusy = true;
        try
        {
            var result = await _api.StopOwnerVoiceAsync(request, sessionId);
            _lastApprovalId = request.InvocationId;
            _lastObservedApprovalStatus = result.Status;
            _approvalDetails.Text = $"Harness reports {result.Status}: {result.ReasonCode}.";
            NotifyUser("Harness approval update",
                $"Harness reports request state: {result.Status}.",
                result.Status is "denied" or "expired" ? ToolTipIcon.Warning : ToolTipIcon.Info);
        }
        catch (Exception ex)
        {
            _approvalDetails.Text = "Private OIA-01 voice approval failed: " + SafeUiError(ex);
        }
        finally
        {
            _approvalBusy = false;
            RefreshVoiceIndicator(_voice.Snapshot);
            await RefreshApprovalsAsync();
        }
    }

    private sealed class OwnerPhrasePrompt : Form
    {
        private readonly TextBox _phrase = new()
        {
            UseSystemPasswordChar = true,
            Dock = DockStyle.Top,
            MaxLength = 512,
            AccessibleName = "Private Owner Phrase"
        };

        public string Phrase => _phrase.Text;

        public OwnerPhrasePrompt()
        {
            Text = "Verify owner decision";
            FormBorderStyle = FormBorderStyle.FixedDialog;
            StartPosition = FormStartPosition.CenterParent;
            MinimizeBox = false;
            MaximizeBox = false;
            ShowInTaskbar = false;
            BackColor = Color.FromArgb(24, 29, 40);
            ForeColor = Color.FromArgb(245, 247, 250);
            Width = 400;
            Height = 156;
            Padding = new Padding(12);
            var label = new Label
            {
                Text = "Enter your Owner Phrase to authenticate this exact decision.",
                Dock = DockStyle.Top,
                Height = 36,
                ForeColor = Color.FromArgb(245, 247, 250)
            };
            _phrase.BackColor = Color.FromArgb(36, 41, 54);
            _phrase.ForeColor = Color.White;
            var buttons = new FlowLayoutPanel
            {
                Dock = DockStyle.Bottom,
                FlowDirection = FlowDirection.RightToLeft,
                Height = 40
            };
            var submit = new Button { Text = "Authenticate", DialogResult = DialogResult.OK, AutoSize = true };
            var cancel = new Button { Text = "Cancel", DialogResult = DialogResult.Cancel, AutoSize = true };
            StyleButton(submit, primary: true);
            StyleButton(cancel, primary: false);
            buttons.Controls.Add(submit);
            buttons.Controls.Add(cancel);
            Controls.Add(_phrase);
            Controls.Add(label);
            Controls.Add(buttons);
            AcceptButton = submit;
            CancelButton = cancel;
        }

        public void ClearPhrase() => _phrase.Clear();
    }

    private sealed record ApprovalChoice(string Label, ApprovalRequestView Request);

    private async Task StartLilaAsync()
    {
        _serviceManuallyStopped = false;
        _startButton.Enabled = false;
        _connection.Text = "Retrying Lila connection…";
        await RefreshStatusAsync();
    }

    private async Task StopLilaAsync()
    {
        if (MessageBox.Show(this, "Stop the Lila service in WSL?", "Stop Lila",
                MessageBoxButtons.YesNo, MessageBoxIcon.Question) != DialogResult.Yes)
            return;
        _voice.OnServiceUnavailable();
        _shortcuts.ApplyRuntimeState(false, null);
        _isReady = false;
        var result = await _wsl.StopLilaAsync();
        _serviceManuallyStopped = true;
        _connection.Text = result.Message;
        _startButton.Enabled = true;
        _stopButton.Enabled = false;
        _voiceToggleButton.Enabled = false;
        _tray.Text = "Lila — stopped";
    }

    private async Task StartScreenHighlightFlowAsync()
    {
        if (_highlightOverlayOpen || IsDisposed)
            return;
        _highlightOverlayOpen = true;
        var wasVisible = Visible;
        try
        {
            var screen = Screen.AllScreens.Length > 0
                ? Screen.FromRectangle(Bounds)
                : Screen.PrimaryScreen;
            if (screen is null)
                return;

            // Briefly hide Lila Island so the screen capture is unobstructed
            _peekPopup.HidePopup();
            _toastPopup.HideToast();
            if (wasVisible)
            {
                Opacity = 0;
                _backdrop?.SyncToOwner(Bounds, Region, ownerVisible: false);
                await Task.Delay(45);
            }

            // M7: request short-lived daemon-signed owner gesture token before capture
            PerceptionGestureResult? gesture = null;
            try { gesture = await _api.IssuePerceptionGestureAsync("region"); }
            catch { /* Highlight still works offline as a transient local UI overlay. */ }

            using var snapshot = ScreenHighlightOverlayForm.CaptureScreenSnapshot(screen);
            if (wasVisible)
            {
                Opacity = ActiveGlassOpacity;
                _backdrop?.SyncToOwner(Bounds, Region, ownerVisible: true);
            }

            using var overlay = new ScreenHighlightOverlayForm(screen, snapshot);
            var dialogResult = overlay.ShowDialog(this);
            if (dialogResult != DialogResult.OK || overlay.ResultHighlight is null)
                return;

            if (gesture is not null)
            {
                try
                {
                    var hl = overlay.ResultHighlight;
                    var cap = await _api.CapturePerceptionAsync(
                        gesture.OwnerGestureToken,
                        hl.IsFullScreen ? "fullscreen" : "region",
                        hl.ScopeSummary,
                        hl.ToBase64Png(),
                        hl.ScreenBounds.Width,
                        hl.ScreenBounds.Height);
                    hl.DaemonCaptureId = cap.CaptureId;
                    await _api.ObservePerceptionAsync(cap.CaptureId);
                }
                catch { /* Keep transient local highlight working even if daemon perception endpoint is offline. */ }
            }

            ShowWindow();
            if (!_expanded)
                ApplyExpandedLayout(expanded: true, animate: true, save: true);

            AttachScreenHighlight(overlay.ResultHighlight);
            if (!string.IsNullOrWhiteSpace(overlay.ResultHighlight.Comment) &&
                string.IsNullOrWhiteSpace(_textInput.Text))
            {
                _textInput.Text = overlay.ResultHighlight.Comment;
            }

            if (overlay.ResultHighlight.SendImmediately)
            {
                await SendTextAsync();
            }
            else
            {
                _textInput.Focus();
            }
        }
        catch (Exception ex)
        {
            AppendConversation("Connection", "Screen highlight failed: " + SafeUiError(ex));
        }
        finally
        {
            if (wasVisible && !IsDisposed)
            {
                Opacity = ActiveGlassOpacity;
                _backdrop?.SyncToOwner(Bounds, Region, ownerVisible: Visible);
            }
            _highlightOverlayOpen = false;
        }
    }

    private void AttachScreenHighlight(TransientScreenHighlight highlight)
    {
        _attachedHighlight?.Dispose();
        _attachedHighlight = highlight;
        _highlightBannerLabel.Text = $"Highlight: {highlight.ScopeSummary} · Transient";
        _highlightBanner.Visible = true;
        _highlightButton.ActivityState = IslandVisualState.Listening;
    }

    private void DiscardAttachedHighlight()
    {
        if (_attachedHighlight?.DaemonCaptureId is { } cid)
            _ = _api.DiscardPerceptionAsync(cid);
        _attachedHighlight?.Dispose();
        _attachedHighlight = null;
        _highlightBanner.Visible = false;
        _highlightButton.ActivityState = IslandVisualState.Idle;
    }

    private async Task SendTextAsync()
    {
        var text = _textInput.Text;
        var highlight = _attachedHighlight;
        if (highlight is { IsExpired: true })
        {
            DiscardAttachedHighlight();
            highlight = null;
        }

        if (_textBusy || (string.IsNullOrWhiteSpace(text) && highlight is null))
            return;

        _textBusy = true;
        _sendButton.Enabled = false;
        var displayMessage = highlight is not null
            ? $"[{highlight.ScopeSummary}] {(string.IsNullOrWhiteSpace(text) ? (highlight.Comment ?? "Inspect highlighted screen area") : text)}"
            : text;
        var payloadText = highlight is not null
            ? highlight.BuildPromptContext(text)
            : text;

        AppendConversation("You", displayMessage);
        _textInput.Clear();
        DiscardAttachedHighlight();
        RefreshVoiceIndicator(_voice.Snapshot);
        try
        {
            var result = await _api.SendTextAsync(payloadText);
            AppendConversation("Lila", result.Response);
            AppendToolEvidence(result.ToolResults, result.ToolResultsOmitted);
        }
        catch (Exception ex)
        {
            AppendConversation("Connection", SafeUiError(ex));
            if (ex is LilaApiException { Kind: LilaApiFailureKind.Connection or LilaApiFailureKind.Timeout })
            {
                _shortcuts.ApplyRuntimeState(false, null);
                _isReady = false;
                _voiceToggleButton.Enabled = false;
                await RefreshStatusAsync();
            }
        }
        finally
        {
            _textBusy = false;
            _sendButton.Enabled = true;
            RefreshVoiceIndicator(_voice.Snapshot);
        }
    }

    private void OnRelevantKey(InterpretedKeyEvent input)
    {
        if (_diagnosticOpen)
            return;
        if (input.RawEvent.VirtualKey == KeyEventInterpreter.RightControlVirtualKey &&
            _isReady && !_shortcuts.IsMuted)
        {
            if (!input.RawEvent.IsKeyUp && !input.RawEvent.IsRepeat &&
                _ownerPttSessionId is null &&
                _shownApproval is { ApprovalReady: true } pendingApproval &&
                _voiceApprovalButton.Enabled && !_approvalBusy)
            {
                BeginInvoke(() => _ = StartOwnerPttAsync(pendingApproval));
                return;
            }
            if (input.RawEvent.IsKeyUp && _ownerPttSessionId is not null)
            {
                BeginInvoke(() => _ = StopOwnerPttAsync());
                return;
            }
        }
        _shortcuts.Handle(input);
    }

    private void OnVoiceStateChanged(VoiceInteractionSnapshot state)
    {
        _interactionDiagnostics?.RecordState(state);
        if (IsDisposed || !IsHandleCreated)
            return;
        if (InvokeRequired)
        {
            BeginInvoke(new Action(() => OnVoiceStateChanged(state)));
            return;
        }

        RefreshVoiceIndicator(state);

        if (state.State == VoiceInteractionState.Response)
        {
            if (!string.IsNullOrWhiteSpace(state.Transcript))
                AppendConversation("You (voice)", state.Transcript);
            if (!string.IsNullOrWhiteSpace(state.Response))
                AppendConversation("Lila", state.Response);
            AppendToolEvidence(state.ToolResults, state.ToolResultsOmitted);
        }
        if (state.Error is not null)
        {
            _avatar.AccessibleDescription = "Lila voice error: " + state.Error;
            _toolTip.SetToolTip(_avatar, "Voice error: " + state.Error);
            if (_tucked || !_expanded)
                ShowPeekMessage("Voice error: " + state.Error);
            if (_tucked)
                NotifyUser("Lila voice error", state.Error, ToolTipIcon.Warning);
        }
    }

    private void ShowPeekMessage(string text)
    {
        if (IsDisposed || string.IsNullOrWhiteSpace(text))
            return;
        var screen = Screen.FromRectangle(Bounds);
        _peekPopup.ShowPeek(text, Bounds, screen);
    }

    private void UpdateSatellitePopups()
    {
        if (IsDisposed)
            return;
        if ((!_expanded || _tucked) && _shownApproval is not null)
        {
            var elapsed = System.Diagnostics.Stopwatch.GetElapsedTime(_approvalObservedTimestamp).TotalSeconds;
            var current = Math.Max(0.0, _approvalRemainingSeconds - elapsed);
            _toastPopup.ShowOrUpdateToast(_shownApproval, current, _approvalInitialSeconds,
                Bounds, Screen.FromRectangle(Bounds), _voiceApprovalButton.Enabled);
        }
        else
        {
            _toastPopup.HideToast();
        }
    }

    private static void EnableDoubleBuffering(Control control)
    {
        try
        {
            typeof(Control).GetProperty("DoubleBuffered",
                System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic)
                ?.SetValue(control, true, null);
        }
        catch
        {
        }
    }

    private static Size MeasureBubbleText(string text, Font font, int maxTextWidth)
    {
        var normalized = text.Replace("\r\n", "\n").Replace("\n", Environment.NewLine);
        var measured = TextRenderer.MeasureText(
            string.IsNullOrEmpty(normalized) ? " " : normalized,
            font,
            new Size(Math.Max(60, maxTextWidth), int.MaxValue),
            TextFormatFlags.WordBreak | TextFormatFlags.TextBoxControl);
        return new Size(Math.Min(maxTextWidth, Math.Max(24, measured.Width + 4)), Math.Max(font.Height + 2, measured.Height + 2));
    }

    private void AppendConversation(string speaker, string text, bool suppressNotifications = false)
    {
        var userMessage = speaker.StartsWith("You", StringComparison.Ordinal);
        var warning = speaker is "Connection" or "Voice error";
        var color = warning
            ? Color.FromArgb(97, 49, 53)
            : userMessage
                ? Color.FromArgb(132, 126, 245)
                : Color.FromArgb(30, 35, 46);
        var padX = ScaleLogical(11);
        var padY = ScaleLogical(8);
        var font = new Font("Segoe UI", 9.5f);
        var maxBubbleW = ScaleLogical(260);
        var maxTextW = Math.Max(60, maxBubbleW - padX * 2);
        var normalizedText = text.Replace("\r\n", "\n").Replace("\n", Environment.NewLine);
        var textBox = new TextBox
        {
            Text = normalizedText,
            ReadOnly = true,
            Multiline = true,
            WordWrap = true,
            BorderStyle = BorderStyle.None,
            ShortcutsEnabled = true,
            TabStop = false,
            Cursor = Cursors.IBeam,
            ForeColor = Color.FromArgb(245, 247, 250),
            BackColor = color,
            Font = font
        };
        var copyMenu = new ContextMenuStrip();
        copyMenu.Items.Add("Copy message", null, (_, _) =>
        {
            var selected = string.IsNullOrEmpty(textBox.SelectedText) ? textBox.Text : textBox.SelectedText;
            if (!string.IsNullOrEmpty(selected))
                Clipboard.SetText(selected);
        });
        textBox.ContextMenuStrip = copyMenu;

        var textSize = MeasureBubbleText(normalizedText, font, maxTextW);
        textBox.Location = new Point(padX, padY);
        textBox.Size = textSize;

        var bubble = new Panel
        {
            BackColor = color,
            Size = new Size(textSize.Width + padX * 2, textSize.Height + padY * 2),
            ContextMenuStrip = copyMenu
        };
        bubble.Controls.Add(textBox);
        UpdateBubbleRegion(bubble);
        var row = new Panel
        {
            BackColor = Color.Transparent,
            Margin = new Padding(0, 3, 0, 3),
            Height = bubble.Height + ScaleLogical(4),
            Tag = new BubbleLayout(bubble, textBox, userMessage, padX, padY)
        };
        row.Controls.Add(bubble);
        _conversation.SuspendLayout();
        _conversation.Controls.Add(row);
        _conversationMessageCount++;
        if (_conversationMessageCount > 160)
        {
            var old = _conversation.Controls[0];
            _conversation.Controls.RemoveAt(0);
            old.Dispose();
            _conversationMessageCount--;
        }
        LayoutConversationBubbles();
        _conversation.ResumeLayout(true);
        _conversation.ScrollControlIntoView(row);
        if (!suppressNotifications && (speaker == "You" || speaker == "Lila"))
            _screenOverlay.PushCaption(speaker, text, isFinal: true);
        if (!suppressNotifications)
        {
            if (speaker == "Lila" && (_tucked || !_expanded))
            {
                if (Visible)
                    ShowPeekMessage(text);
                else
                    NotifyUser("Lila replied", "A new Lila response is available in the Island.", ToolTipIcon.Info);
            }
            else if (warning && (_tucked || !_expanded))
            {
                if (Visible)
                    ShowPeekMessage(text);
                else
                    NotifyUser("Lila needs attention", text, ToolTipIcon.Warning);
            }
        }
    }

    private sealed record BubbleLayout(Panel Bubble, TextBox Text, bool RightAligned, int PadX, int PadY);

    private void LayoutConversationBubbles()
    {
        if (_conversation.ClientSize.Width <= 0)
            return;
        var rowWidth = Math.Max(ScaleLogical(96),
            _conversation.ClientSize.Width - _conversation.Padding.Horizontal - ScaleLogical(8));
        foreach (var row in _conversation.Controls.OfType<Panel>())
        {
            row.Width = rowWidth;
            if (row.Tag is not BubbleLayout bubble)
                continue;
            var maximumWidth = Math.Max(ScaleLogical(120), rowWidth * 4 / 5);
            var maxTextW = Math.Max(60, maximumWidth - bubble.PadX * 2);
            var textSize = MeasureBubbleText(bubble.Text.Text, bubble.Text.Font, maxTextW);
            var bubbleSize = new Size(textSize.Width + bubble.PadX * 2, textSize.Height + bubble.PadY * 2);
            if (bubble.Text.Size != textSize)
                bubble.Text.Size = textSize;
            var sizeChanged = bubble.Bubble.Size != bubbleSize;
            if (sizeChanged)
                bubble.Bubble.Size = bubbleSize;
            bubble.Bubble.Left = bubble.RightAligned
                ? Math.Max(0, row.ClientSize.Width - bubble.Bubble.Width)
                : 0;
            row.Height = bubble.Bubble.Height + ScaleLogical(4);
            if (sizeChanged)
                UpdateBubbleRegion(bubble.Bubble);
        }
    }

    private void UpdateBubbleRegion(Panel bubble)
    {
        if (bubble.Width <= 0 || bubble.Height <= 0)
            return;
        using var path = RoundedRectangle(
            new RectangleF(0, 0, bubble.Width, bubble.Height), ScaleLogical(12));
        var previous = bubble.Region;
        bubble.Region = new Region(path);
        previous?.Dispose();
    }

    private void AppendToolEvidence(IReadOnlyList<ToolResultView> results, int omitted)
    {
        foreach (var result in results)
        {
            var label = string.IsNullOrWhiteSpace(result.Label) ? result.Tool : result.Label;
            var status = string.IsNullOrWhiteSpace(result.Status) ? "status not reported" : result.Status;
            var verification = string.IsNullOrWhiteSpace(result.VerificationStatus)
                ? "not reported" : result.VerificationStatus;
            var evidence = $"{label}: {status}; verification: {verification}.";
            AppendConversation("Harness evidence", evidence);

            if (TryApplyToolUiEffects(result, evidence))
                continue;

            var stage = string.Equals(verification, "verified", StringComparison.OrdinalIgnoreCase)
                ? ScreenTaskStage.Verified
                : result.Success
                    ? ScreenTaskStage.CompletedNotVerified
                    : ScreenTaskStage.Dispatched;
            _screenStore.Upsert(new ScreenItem
            {
                Id = $"tool-{result.Tool}-{Environment.TickCount64}",
                Kind = ScreenItemKind.Task,
                Title = label,
                SourceLabel = "Task · states from Harness",
                TaskStage = stage,
                EvidenceSummary = evidence,
                Summary = string.IsNullOrWhiteSpace(result.Output) ? evidence : result.Output
            });
            if (_tucked && Visible)
                ShowPeekMessage(evidence);
        }
        if (omitted > 0)
            AppendConversation("Harness evidence",
                $"{omitted} additional tool result(s) were omitted by the API response.");
    }

    private bool TryApplyToolUiEffects(ToolResultView result, string evidence)
    {
        if (result.Data is not System.Text.Json.JsonElement data ||
            data.ValueKind != System.Text.Json.JsonValueKind.Object)
            return false;

        if (data.TryGetProperty("ui_action", out var actionProp) &&
            actionProp.ValueKind == System.Text.Json.JsonValueKind.String)
        {
            ApplyAssistantUiAction(actionProp.GetString() ?? string.Empty);
            return true;
        }

        if (data.TryGetProperty("screen_item", out var itemProp) &&
            itemProp.ValueKind == System.Text.Json.JsonValueKind.Object)
        {
            var kindStr = itemProp.TryGetProperty("kind", out var k) ? k.GetString() : "research";
            var kind = kindStr?.ToLowerInvariant() switch
            {
                "image" => ScreenItemKind.Image,
                "video" => ScreenItemKind.Video,
                "task" => ScreenItemKind.Task,
                _ => ScreenItemKind.Research
            };
            var sources = new List<string>();
            if (itemProp.TryGetProperty("sources", out var srcArr) &&
                srcArr.ValueKind == System.Text.Json.JsonValueKind.Array)
            {
                foreach (var s in srcArr.EnumerateArray())
                {
                    var text = s.GetString();
                    if (!string.IsNullOrWhiteSpace(text))
                        sources.Add(text);
                }
            }
            _screenStore.Upsert(new ScreenItem
            {
                Id = $"screen-{Environment.TickCount64}-{Guid.NewGuid().ToString("N")[..4]}",
                Kind = kind,
                Title = itemProp.TryGetProperty("title", out var t) ? (t.GetString() ?? result.Tool) : result.Tool,
                Summary = itemProp.TryGetProperty("summary", out var sum) ? (sum.GetString() ?? string.Empty) : string.Empty,
                SourceLabel = kind == ScreenItemKind.Research
                    ? $"Found by Lila · {sources.Count} {(sources.Count == 1 ? "source" : "sources")}"
                    : "From Lila",
                Sources = sources,
                MediaPath = itemProp.TryGetProperty("media_path", out var mp) ? mp.GetString() : null,
                DurationLabel = itemProp.TryGetProperty("duration", out var dur) ? dur.GetString() : null,
                IsPrivate = itemProp.TryGetProperty("is_private", out var priv) && priv.ValueKind == System.Text.Json.JsonValueKind.True,
                EvidenceSummary = evidence
            });
            return true;
        }
        return false;
    }

    private void ApplyAssistantUiAction(string action)
    {
        switch (action.Trim().ToLowerInvariant())
        {
            case "tuck":
                _tuckMode = true;
                TuckIsland();
                break;
            case "restore":
                _tuckMode = false;
                RestoreIsland(allowFullscreenOverride: true);
                break;
            case "expand":
                if (_tucked)
                    RestoreIsland(allowFullscreenOverride: true);
                ApplyExpandedLayout(expanded: true, animate: true, save: true);
                break;
            case "collapse":
                ApplyExpandedLayout(expanded: false, animate: true, save: true);
                break;
            case "mute_mic":
                _shortcuts.SetMuted(true);
                RefreshVoiceIndicator(_voice.Snapshot);
                break;
            case "unmute_mic":
                _shortcuts.SetMuted(false);
                RefreshVoiceIndicator(_voice.Snapshot);
                break;
            case "open_drawer":
                _workDrawer.OpenForItem(null, Screen.FromRectangle(Bounds));
                _ = RefreshCapabilityCandidatesAsync();
                break;
            case "close_drawer":
                _workDrawer.Hide();
                break;
            case "screen_sharing_on":
                _screenStore.SetScreenSharing(true);
                _overflowScreenShareItem.Checked = true;
                break;
            case "screen_sharing_off":
                _screenStore.SetScreenSharing(false);
                _overflowScreenShareItem.Checked = false;
                break;
            case "captions_on":
                _screenStore.SetCaptionsEnabled(true);
                _overflowCaptionsItem.Checked = true;
                break;
            case "captions_off":
                _screenStore.SetCaptionsEnabled(false);
                _overflowCaptionsItem.Checked = false;
                _screenOverlay.ClearCaptions();
                break;
            case "clear_screen":
                _screenStore.ClearAll();
                break;
            case "highlight_screen":
                _ = StartScreenHighlightFlowAsync();
                break;
        }
    }

    private void RefreshVoiceIndicator(VoiceInteractionSnapshot state)
    {
        var visual = !_isReady
            ? IslandVisualState.Disconnected
            : _shortcuts.IsMuted
                ? IslandVisualState.Muted
                : _shownApproval is not null
                    ? IslandVisualState.Attention
                    : IslandVisualState.Idle;
        var text = _isReady ? "Connected" : "Voice unavailable";
        var detail = state.Error;

        if (_shortcuts.IsMuted)
            text = "Mic off";
        else if (_approvalStateUnavailable)
        {
            text = "Approval state unavailable";
            detail = "No approval decision is assumed; reconnect to reload authoritative Harness state.";
            visual = IslandVisualState.Error;
        }
        else if (state.Error is not null)
        {
            text = "Voice error";
            visual = IslandVisualState.Error;
        }
        else if (state.State != VoiceInteractionState.Idle)
        {
            var isToggle = _shortcuts.ActiveSource == VoiceShortcutSource.FnPAndPauseToggle ||
                           _voice.Mode == PushToTalkMode.ToggleToTalk;
            (text, visual) = state.State switch
            {
                VoiceInteractionState.Starting => (
                    isToggle ? "Voice on: say what you need" : "Listening: release to send",
                    IslandVisualState.Listening),
                VoiceInteractionState.Recording => (
                    isToggle ? "Voice on: say what you need" : "Listening: release to send",
                    IslandVisualState.Listening),
                VoiceInteractionState.Processing => ("Thinking…", IslandVisualState.Thinking),
                VoiceInteractionState.Cancelling => ("Cancelling voice", IslandVisualState.Interrupted),
                VoiceInteractionState.Response => ("Speaking…", IslandVisualState.Speaking),
                _ => ("Voice state unavailable", IslandVisualState.Error)
            };
        }
        else if (_shownApproval is not null)
        {
            text = "Approval needed";
            visual = IslandVisualState.Attention;
        }
        else if (_voiceStatusError is not null)
        {
            text = "Voice state unavailable";
            detail = _voiceStatusError;
            visual = IslandVisualState.Error;
        }
        else if (_remoteVoiceStatus?.State == "recording")
        {
            text = "Listening: release to send";
            visual = IslandVisualState.Listening;
        }
        else if (_remoteVoiceStatus?.State == "transcribing")
        {
            text = "Thinking…";
            visual = IslandVisualState.Thinking;
        }
        else if (_remoteVoiceStatus?.State == "cancelling")
        {
            text = "Cancelling voice";
            visual = IslandVisualState.Interrupted;
        }
        else if (_remoteVoiceStatus?.State == "timed_out")
        {
            text = "Voice timed out";
            detail = _remoteVoiceStatus.Error;
            visual = IslandVisualState.Interrupted;
        }
        else if (_remoteVoiceStatus?.State == "failed")
        {
            text = "Voice error";
            detail = _remoteVoiceStatus.Error;
            visual = IslandVisualState.Error;
        }
        else if (_remoteVoiceStatus?.State == "disabled")
        {
            text = "Voice unavailable";
            visual = IslandVisualState.Disconnected;
        }

        if (_textBusy && _isReady && !_shortcuts.IsMuted &&
            !_approvalStateUnavailable && state.Error is null &&
            (state.State is VoiceInteractionState.Idle or VoiceInteractionState.Response) &&
            !RemoteVoiceSessionIsActive())
        {
            text = "Thinking…";
            visual = IslandVisualState.Thinking;
        }

        if (visual != IslandVisualState.Listening)
            _liveTranscriptText = null;

        // Update header status text & dot color when ready so header reflects authoritative state
        if (_isReady)
        {
            _connection.Text = visual == IslandVisualState.Listening && !string.IsNullOrWhiteSpace(_liveTranscriptText)
                ? _liveTranscriptText
                : (_keyboard is null && visual == IslandVisualState.Idle)
                    ? "Connected · keyboard unavailable"
                    : text;
            _connectionDotColor = visual switch
            {
                IslandVisualState.Listening => Color.FromArgb(124, 156, 255),
                IslandVisualState.Speaking or IslandVisualState.Thinking => Color.FromArgb(176, 108, 255),
                IslandVisualState.Attention or IslandVisualState.Interrupted => Color.FromArgb(255, 184, 107),
                IslandVisualState.Muted or IslandVisualState.Error => Color.FromArgb(255, 93, 93),
                _ => Color.FromArgb(94, 224, 160)
            };
            _connectionDot.Invalidate();
        }

        _lastReportedVoiceVisual = visual;

        _avatar.IsMuted = _shortcuts.IsMuted;
        _avatar.State = visual;
        _avatar.AccessibleDescription = string.IsNullOrWhiteSpace(detail)
            ? "Lila status: " + text
            : $"Lila status: {text}. {detail}";
        _toolTip.SetToolTip(_avatar, string.IsNullOrWhiteSpace(detail)
            ? text : $"{text}: {detail}");

        if (_expanded)
        {
            _headerEqualizer.State = visual;
            _collapsedEqualizer.State = IslandVisualState.Idle;
        }
        else
        {
            _headerEqualizer.State = IslandVisualState.Idle;
            _collapsedEqualizer.State = visual;
        }

        if (visual is IslandVisualState.Listening or IslandVisualState.Speaking or
            IslandVisualState.Thinking or IslandVisualState.Attention)
            _washBreathTimer.Start();
        else
            _washBreathTimer.Stop();
        _surface.Invalidate();

        _muteButton.IsMuted = _shortcuts.IsMuted;
        _muteButton.AccessibleName = _shortcuts.IsMuted
            ? "Unmute microphone input" : "Mute microphone input";
        var localCanControl = state.State is VoiceInteractionState.Idle or
            VoiceInteractionState.Response or VoiceInteractionState.Starting or
            VoiceInteractionState.Recording;
        var localCaptureActive = state.State is
            VoiceInteractionState.Starting or VoiceInteractionState.Recording;
        var remoteOwnedElsewhere = RemoteVoiceSessionIsActive() && !localCaptureActive;
        _voiceToggleButton.Enabled = _isReady &&
                                     localCanControl && !remoteOwnedElsewhere;
        _voiceToggleButton.IsMuted = _shortcuts.IsMuted;
        _voiceToggleButton.ActivityState = visual;
        var capturing = state.State is VoiceInteractionState.Recording or VoiceInteractionState.Starting;
        _voiceToggleButton.AccessibleName = capturing
            ? "Stop and submit voice capture" : "Hold to talk, or click to keep voice on";
        _toolTip.SetToolTip(_voiceToggleButton, capturing
            ? "Stop and submit voice capture" : "Hold to talk, or click to keep voice on");
        _cancelButton.Enabled = capturing;
        _cancelButton.Visible = capturing;
        _edgeTab.IsMuted = _shortcuts.IsMuted;
        _edgeTab.IsDimmed = _fullscreenActive;
        _edgeTab.ActivityState = visual;

        var activeVoice = capturing || visual is IslandVisualState.Listening or
            IslandVisualState.Thinking or IslandVisualState.Speaking or
            IslandVisualState.Attention or IslandVisualState.Interrupted;
        _screenOverlay.SetTargetScreen(Screen.FromRectangle(Bounds));
        _screenOverlay.UpdateVoiceState(visual, text, detail ?? string.Empty, activeVoice);
    }

    private void NotifyUser(string title, string message, ToolTipIcon icon)
    {
        if (_exitRequested || IsDisposed || !_tray.Visible)
            return;
        _tray.BalloonTipTitle = title;
        _tray.BalloonTipText = message.Length > 240 ? message[..240] : message;
        _tray.BalloonTipIcon = icon;
        _tray.ShowBalloonTip(4_000);
    }

    private void ShowSettings()
    {
        if (_tucked)
            RestoreIsland(allowFullscreenOverride: true);
        if (!_expanded)
            ApplyExpandedLayout(expanded: true, animate: true, save: true);

        _settingsSheet.ReloadFromSettings();
        _settingsOpen = true;
        _header.Visible = false;
        _body.Visible = false;
        _settingsSheet.Visible = true;
        _settingsSheet.BringToFront();
        _surface.Invalidate();
    }

    private void HideSettingsSheet()
    {
        _settingsOpen = false;
        _settingsSheet.Visible = false;
        _header.Visible = true;
        _body.Visible = _expanded;
        LayoutHeader();
        _surface.Invalidate();
    }

    private async Task OnSettingsSheetSavedAsync(string previousDistribution)
    {
        var wasMuted = _shortcuts.IsMuted;
        _shortcutSummary.Text = ShortcutSummary();
        _shortcuts.MutedShortcutAttempted -= OnMutedVoiceAttempt;
        _shortcuts.SetReady(false);
        _shortcuts.Dispose();
        if (_keyboard is not null)
        {
            _keyboard.NonModifierKeyDown -= _shortcuts.NotifyNonModifierKeyDown;
            _keyboard.KeyboardDeviceRemoved -= _shortcuts.NotifyDeviceRemoved;
            _keyboard.SetShortcuts(ConfiguredBindings());
        }
        _shortcuts = CreateShortcutController();
        _shortcuts.SetMuted(wasMuted);
        if (_keyboard is not null)
        {
            _keyboard.NonModifierKeyDown += _shortcuts.NotifyNonModifierKeyDown;
            _keyboard.KeyboardDeviceRemoved += _shortcuts.NotifyDeviceRemoved;
        }
        if (!previousDistribution.Equals(_settings.Distribution, StringComparison.OrdinalIgnoreCase))
        {
            await _lifecycle.StopAsync();
            await _lifecycle.DisposeAsync();
            _wsl = new WslClient(new SystemCommandRunner(), _api, _settings.Distribution);
            _lifecycle = new WslLifecycleSupervisor(new SystemCommandRunner(),
                new ManagedWslKeepAlive(), _settings.Distribution);
        }
        _serviceManuallyStopped = false;
        HideSettingsSheet();
        await RefreshStatusAsync();
    }

    private void ShowDiagnostic()
    {
        if (_keyboard is null)
        {
            MessageBox.Show(this, "Raw keyboard input is unavailable.", "Keyboard diagnostic",
                MessageBoxButtons.OK, MessageBoxIcon.Warning);
            return;
        }
        if (_diagnosticForm is { IsDisposed: false })
        {
            _diagnosticForm.Show();
            _diagnosticForm.BringToFront();
            _diagnosticForm.Activate();
            return;
        }

        _diagnosticOpen = true;
        var diagnostics = new InteractionDiagnostics();
        _interactionDiagnostics = diagnostics;
        void OnRequest(ApiRequestDiagnostic request) =>
            diagnostics.RecordRequest(request, _voice.Snapshot.State);
        _api.RequestCompleted += OnRequest;

        var form = new KeyboardDiagnosticForm(_keyboard, diagnostics, () => _voice.Snapshot.State);
        form.FormClosed += (_, _) =>
        {
            _api.RequestCompleted -= OnRequest;
            diagnostics.Dispose();
            _interactionDiagnostics = null;
            _diagnosticOpen = false;
            _diagnosticForm = null;
        };
        _diagnosticForm = form;
        form.Show(this);
    }

    public void OpenDiagnosticMode() => ShowDiagnostic();

    private void ShowWindow()
    {
        _manualRestoreUntilTicks = Environment.TickCount64 + 4000;
        ShowInTaskbar = false;
        Show();
        WindowState = FormWindowState.Normal;
        Opacity = ActiveGlassOpacity;
        TopMost = true;
        BringToFront();
        _tuckMode = false;
        if (_tucked)
            RestoreIsland(allowFullscreenOverride: true);
        if (!_expanded)
            ApplyExpandedLayout(expanded: true, animate: true, save: true);
        Activate();
        if (!_settingsOpen)
            _textInput.Focus();
    }

    private async void ExitApplication()
    {
        if (_exitRequested)
            return;
        _exitRequested = true;
        _eventStreamCts?.Cancel();
        _fullscreenTimer.Stop();
        _transitionTimer.Stop();
        _retuckTimer.Stop();
        _approvalCountdownTimer.Stop();
        _washBreathTimer.Stop();
        _peekPopup.HidePopup();
        _toastPopup.HideToast();
        _shortcuts.ApplyRuntimeState(false, null);
        await _voice.CancelAndWaitAsync(TimeSpan.FromSeconds(25));
        try { await _wsl.StopLilaAsync(); }
        catch { /* Release the managed WSL process even if systemd is unreachable. */ }
        try { await _lifecycle.DisposeAsync(); }
        catch { /* The Windows process handle is closed during process shutdown. */ }
        Close();
    }

    protected override void OnFormClosing(FormClosingEventArgs e)
    {
        if (!_exitRequested && e.CloseReason == CloseReason.UserClosing)
        {
            e.Cancel = true;
            ShowInTaskbar = false;
            Hide();
            return;
        }

        _statusTimer.Stop();
        _eventStreamCts?.Cancel();
        _eventStreamCts?.Dispose();
        _eventStreamCts = null;
        _fullscreenTimer.Stop();
        _transitionTimer.Stop();
        _retuckTimer.Stop();
        _approvalCountdownTimer.Stop();
        _washBreathTimer.Stop();
        SystemEvents.DisplaySettingsChanged -= OnDisplaySettingsChanged;
        SavePlacement();
        _backdrop?.Close();
        _backdrop?.Dispose();
        _backdrop = null;
        _attachedHighlight?.Dispose();
        _peekPopup.Dispose();
        _toastPopup.Dispose();
        _screenOverlay.Dispose();
        _workDrawer.Dispose();
        _overflowMenu.Dispose();
        _tray.Visible = false;
        _tray.Dispose();
        _keyboard?.Dispose();
        _shortcuts.Dispose();
        _http.Dispose();
        base.OnFormClosing(e);
    }

    private async Task ToggleSpokenEnrollmentAsync()
    {
        if (IsDisposed)
            return;
        try
        {
            if (_spokenEnrollSessionId is null)
            {
                var started = await _api.StartSpokenEnrollmentAsync();
                _spokenEnrollSessionId = started.SessionId;
                _overflowSpokenEnrollItem.Text = "Finish voice enrollment";
                NotifyUser("Spoken approval enrollment",
                    "Speak your enrolled Private Owner Phrase now, then click 'Finish voice enrollment' in the menu.",
                    ToolTipIcon.Info);
            }
            else
            {
                var sessionId = _spokenEnrollSessionId;
                _spokenEnrollSessionId = null;
                _overflowSpokenEnrollItem.Text = "Enroll voice approval";
                await _api.StopSpokenEnrollmentAsync(sessionId);
                await RefreshApprovalsAsync();
                NotifyUser("Spoken approval enrolled",
                    "Your spoken owner passphrase was verified. Voice approval is now enabled.",
                    ToolTipIcon.Info);
            }
        }
        catch (Exception ex)
        {
            _spokenEnrollSessionId = null;
            _overflowSpokenEnrollItem.Text = "Enroll voice approval";
            NotifyUser("Spoken enrollment failed", SafeUiError(ex), ToolTipIcon.Warning);
        }
    }

    private async Task RefreshCapabilityCandidatesAsync()
    {
        try
        {
            var result = await _api.GetCapabilityCandidatesAsync();
            var catalogVersion = result.CatalogVersion.ValueKind == System.Text.Json.JsonValueKind.Number &&
                result.CatalogVersion.TryGetInt32(out var parsedVersion)
                ? parsedVersion
                : (result.CatalogVersion.ValueKind == System.Text.Json.JsonValueKind.Object &&
                   result.CatalogVersion.TryGetProperty("version", out var verEl) &&
                   verEl.TryGetInt32(out var objVersion)
                    ? objVersion
                    : 0);
            var views = new List<CapabilityCandidateView>(result.Candidates.Count);
            foreach (var item in result.Candidates)
            {
                var id = item.TryGetProperty("capability_id", out var idEl) ? idEl.GetString() ?? "" : "";
                if (string.IsNullOrWhiteSpace(id))
                    continue;
                var ver = item.TryGetProperty("version", out var vEl) ? vEl.GetString() ?? "0.0.0" : "0.0.0";
                var status = item.TryGetProperty("status", out var sEl) ? sEl.GetString() ?? "staged" : "staged";
                var passed = item.TryGetProperty("tests_passed", out var tpEl) && tpEl.ValueKind == System.Text.Json.JsonValueKind.True;
                var provenance = item.TryGetProperty("provenance", out var pEl) ? pEl.GetString() ?? "sandbox" : "sandbox";
                var summary = item.TryGetProperty("test_summary", out var tsEl) ? tsEl.GetString() ?? "" : "";
                views.Add(new CapabilityCandidateView(id, ver, status, passed, provenance, summary));
            }
            if (!IsDisposed)
                BeginInvoke(() => _workDrawer.SetCapabilityCandidates(catalogVersion, views));
        }
        catch
        {
            // Ignore transient daemon disconnect when refreshing drawer candidates.
        }
    }

    private static string SafeUiError(Exception exception) => exception switch
    {
        LilaApiException apiError => apiError.Message,
        _ => "The operation failed. Check WSL, Lila service status, and Settings."
    };
}
