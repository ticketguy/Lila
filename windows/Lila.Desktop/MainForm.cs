using Lila.Desktop.Core.Api;
using Lila.Desktop.Core.Keyboard;
using Lila.Desktop.Core.Lifecycle;
using Lila.Desktop.Core.Voice;
using Lila.Desktop.Core.Wsl;
using Lila.Desktop.Keyboard;
using Lila.Desktop.Security;
using Lila.Desktop.Settings;
using Lila.Desktop.Wsl;

namespace Lila.Desktop;

public sealed class MainForm : Form
{
    private readonly AppSettings _settings;
    private readonly WindowsCredentialStore _credentials = new();
    private readonly HttpClient _http;
    private readonly LilaApiClient _api;
    private WslClient _wsl;
    private readonly VoiceSessionController _voice;
    private GlobalVoiceShortcutController _shortcuts;
    private RawKeyboardListener? _keyboard;
    private WslLifecycleSupervisor _lifecycle;
    private readonly NotifyIcon _tray;
    private readonly System.Windows.Forms.Timer _statusTimer = new() { Interval = 5_000 };
    private readonly Label _connection = new() { AutoSize = true, Text = "Checking Lila…" };
    private readonly Label _voiceIndicator = new() { AutoSize = true, Text = "● Voice idle", ForeColor = Color.DimGray };
    private readonly Label _shortcutSummary = new() { AutoSize = true };
    private readonly RichTextBox _conversation = new() { ReadOnly = true, Dock = DockStyle.Fill };
    private readonly TextBox _textInput = new() { Dock = DockStyle.Fill };
    private readonly Button _startButton = new() { Text = "Retry Lila", AutoSize = true };
    private readonly Button _stopButton = new() { Text = "Stop Lila", AutoSize = true, Enabled = false };
    private readonly Button _cancelButton = new() { Text = "Cancel voice", AutoSize = true, Enabled = false };
    private readonly Button _voiceToggleButton = new() { Text = "Start voice", AutoSize = true };
    private readonly Button _sendButton = new() { Text = "Send", AutoSize = true };
    private readonly SemaphoreSlim _connectionGate = new(1, 1);
    private bool _textBusy;
    private bool _serviceManuallyStopped;
    private bool _isReady;
    private bool _diagnosticOpen;
    private InteractionDiagnostics? _interactionDiagnostics;
    private bool _exitRequested;
    private readonly bool _startMinimized;

    public MainForm(AppSettings settings, bool startMinimized = false)
    {
        _settings = settings;
        _startMinimized = startMinimized;
        var handler = new HttpClientHandler { AllowAutoRedirect = false };
        _http = new HttpClient(handler) { Timeout = Timeout.InfiniteTimeSpan };
        _api = new LilaApiClient(_http, _credentials);
        _wsl = new WslClient(new SystemCommandRunner(), _api, settings.Distribution);
        _lifecycle = new WslLifecycleSupervisor(new SystemCommandRunner(),
            new ManagedWslKeepAlive(), settings.Distribution);
        _voice = new VoiceSessionController(_api);
        _voice.StateChanged += OnVoiceStateChanged;
        _shortcuts = CreateShortcutController();

        Text = "Lila";
        StartPosition = FormStartPosition.CenterScreen;
        MinimumSize = new Size(440, 380);
        Size = new Size(560, 470);
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
        _startButton.Click += async (_, _) => await StartLilaAsync();
        _stopButton.Click += async (_, _) => await StopLilaAsync();
        _cancelButton.Click += (_, _) => _voice.CancelActive();
        _voiceToggleButton.Click += (_, _) => _shortcuts.ToggleFromStartButton();
        _sendButton.Click += async (_, _) => await SendTextAsync();
        _textInput.KeyDown += async (_, e) =>
        {
            if (e.KeyCode == Keys.Enter && !e.Shift)
            {
                e.SuppressKeyPress = true;
                await SendTextAsync();
            }
        };
        _statusTimer.Tick += async (_, _) => await RefreshStatusAsync();
        _statusTimer.Start();
        Shown += async (_, _) =>
        {
            if (_startMinimized)
                BeginInvoke(new Action(Hide));
            await RefreshStatusAsync();
        };

        var menu = new ContextMenuStrip();
        menu.Items.Add("Show Lila", null, (_, _) => ShowWindow());
        menu.Items.Add("Start Lila", null, async (_, _) => await StartLilaAsync());
        menu.Items.Add("Stop Lila", null, async (_, _) => await StopLilaAsync());
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

    private GlobalVoiceShortcutController CreateShortcutController() =>
        new(_voice,
            new ShortcutBinding(_settings.HoldToTalkVirtualKey, _settings.HoldToTalkModifiers),
            new ShortcutBinding(_settings.ToggleToTalkVirtualKey, _settings.ToggleToTalkModifiers),
            _settings.HoldToTalkEnabled, _settings.ToggleToTalkEnabled,
            TimeSpan.FromMilliseconds(Math.Clamp(_settings.RightControlArmingDelayMilliseconds, 100, 600)));

    private static string ShortcutLabel(bool enabled, ShortcutBinding binding) =>
        enabled ? binding.DisplayName : "disabled";

    private void BuildLayout()
    {
        var root = new TableLayoutPanel
        {
            Dock = DockStyle.Fill,
            Padding = new Padding(10),
            ColumnCount = 1,
            RowCount = 4
        };
        root.RowStyles.Add(new RowStyle(SizeType.Absolute, 38));
        root.RowStyles.Add(new RowStyle(SizeType.Percent, 100));
        root.RowStyles.Add(new RowStyle(SizeType.Absolute, 38));
        root.RowStyles.Add(new RowStyle(SizeType.Absolute, 40));

        var top = new FlowLayoutPanel { Dock = DockStyle.Fill, WrapContents = false };
        top.Controls.Add(_connection);
        top.Controls.Add(_startButton);
        top.Controls.Add(_stopButton);
        var settingsButton = new Button { Text = "Settings", AutoSize = true };
        settingsButton.Click += (_, _) => ShowSettings();
        top.Controls.Add(settingsButton);
        var diagnostic = new Button { Text = "Keyboard diagnostic", AutoSize = true };
        diagnostic.Click += (_, _) => ShowDiagnostic();
        top.Controls.Add(diagnostic);

        var input = new TableLayoutPanel { Dock = DockStyle.Fill, ColumnCount = 2 };
        input.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 100));
        input.ColumnStyles.Add(new ColumnStyle(SizeType.AutoSize));
        input.Controls.Add(_textInput, 0, 0);
        input.Controls.Add(_sendButton, 1, 0);

        var footer = new FlowLayoutPanel { Dock = DockStyle.Fill, WrapContents = false };
        footer.Controls.Add(_voiceIndicator);
        footer.Controls.Add(_voiceToggleButton);
        footer.Controls.Add(_cancelButton);
        _shortcutSummary.Text = ShortcutSummary();
        _shortcutSummary.Padding = new Padding(8, 6, 0, 0);
        footer.Controls.Add(_shortcutSummary);
        footer.Controls.Add(new Label { Text = "Voice/text may be retained in WSL", AutoSize = true, Padding = new Padding(8, 6, 0, 0) });

        root.Controls.Add(top, 0, 0);
        root.Controls.Add(_conversation, 0, 1);
        root.Controls.Add(input, 0, 2);
        root.Controls.Add(footer, 0, 3);
        Controls.Add(root);
    }

    private string ShortcutSummary() =>
        $"Hold: {ShortcutLabel(_settings.HoldToTalkEnabled, new ShortcutBinding(_settings.HoldToTalkVirtualKey, _settings.HoldToTalkModifiers))}  •  " +
        $"Toggle: {ShortcutLabel(_settings.ToggleToTalkEnabled, new ShortcutBinding(_settings.ToggleToTalkVirtualKey, _settings.ToggleToTalkModifiers))}";

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
        }
        catch (Exception ex)
        {
            _shortcuts.SetReady(false);
            _connection.Text = "Disconnected — " + SafeUiError(ex);
            _startButton.Enabled = true;
            _stopButton.Enabled = false;
            _tray.Text = "Lila — disconnected";
        }
        finally
        {
            _connectionGate.Release();
        }
    }

    private void ApplyAvailability(AvailabilityResult result)
    {
        var ready = result.State == LilaAvailability.Ready;
        _isReady = ready;
        _shortcuts.SetReady(ready);
        _voiceToggleButton.Enabled = ready && _voice.Snapshot.State is
            (VoiceInteractionState.Idle or VoiceInteractionState.Response or
             VoiceInteractionState.Starting or VoiceInteractionState.Recording);
        _connection.Text = result.State switch
        {
            LilaAvailability.WslUnavailable => "Disconnected — WSL unavailable",
            LilaAvailability.ServiceInactive => "Disconnected — Lila service inactive",
            LilaAvailability.AuthenticationFailure => "Authentication failed — open Settings",
            LilaAvailability.ApiConnectionFailure => "Disconnected — Lila API unavailable",
            LilaAvailability.Starting => "Lila is starting…",
            _ => "Lila connected"
        };
        if (ready && _keyboard is null)
            _connection.Text = "Lila connected — global keyboard input unavailable";
        _startButton.Enabled = !ready;
        _stopButton.Enabled = ready || result.State == LilaAvailability.Starting;
        _tray.Text = ready ? "Lila — connected" : "Lila — disconnected";
    }

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
        _shortcuts.SetReady(false);
        _isReady = false;
        var result = await _wsl.StopLilaAsync();
        _serviceManuallyStopped = true;
        _connection.Text = result.Message;
        _startButton.Enabled = true;
        _stopButton.Enabled = false;
        _voiceToggleButton.Enabled = false;
        _tray.Text = "Lila — stopped";
    }

    private async Task SendTextAsync()
    {
        var text = _textInput.Text;
        if (_textBusy || string.IsNullOrWhiteSpace(text))
            return;
        _textBusy = true;
        _sendButton.Enabled = false;
        AppendConversation("You", text);
        _textInput.Clear();
        try
        {
            var result = await _api.SendTextAsync(text);
            AppendConversation("Lila", result.Response);
        }
        catch (Exception ex)
        {
            AppendConversation("Connection", SafeUiError(ex));
            if (ex is LilaApiException { Kind: LilaApiFailureKind.Connection or LilaApiFailureKind.Timeout })
            {
                _shortcuts.SetReady(false);
                _isReady = false;
                _voiceToggleButton.Enabled = false;
                await RefreshStatusAsync();
            }
        }
        finally
        {
            _textBusy = false;
            _sendButton.Enabled = true;
        }
    }

    private void OnRelevantKey(InterpretedKeyEvent input)
    {
        if (_diagnosticOpen)
            return;
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

        var sourceLabel = _shortcuts.ActiveSource switch
        {
            VoiceShortcutSource.HoldToTalk => "Right Ctrl hold",
            VoiceShortcutSource.FnPAndPauseToggle => "Fn + P toggle",
            VoiceShortcutSource.StartButton => "Start voice button",
            _ => string.Empty
        };
        _voiceIndicator.Text = state.State switch
        {
            VoiceInteractionState.Starting => $"● Starting microphone… {sourceLabel}",
            VoiceInteractionState.Recording => $"● Recording — {sourceLabel}",
            VoiceInteractionState.Processing => "● Processing speech…",
            VoiceInteractionState.Cancelling => "● Cancelling…",
            VoiceInteractionState.Response => "● Response received",
            _ when state.Error is not null => "● Voice error",
            _ => "● Voice idle"
        };
        _voiceIndicator.ForeColor = state.State switch
        {
            VoiceInteractionState.Recording => Color.Firebrick,
            VoiceInteractionState.Starting or VoiceInteractionState.Processing or
                VoiceInteractionState.Cancelling => Color.DarkOrange,
            _ => Color.DimGray
        };
        _cancelButton.Enabled = state.State is VoiceInteractionState.Starting or VoiceInteractionState.Recording;
        _voiceToggleButton.Enabled = _isReady && (state.State is VoiceInteractionState.Idle or
            VoiceInteractionState.Starting or VoiceInteractionState.Recording or VoiceInteractionState.Response);
        _voiceToggleButton.Text = state.State is VoiceInteractionState.Recording or VoiceInteractionState.Starting
            ? "Stop & submit" : "Start voice";

        if (state.State == VoiceInteractionState.Response)
        {
            if (!string.IsNullOrWhiteSpace(state.Transcript))
                AppendConversation("You (voice)", state.Transcript);
            if (!string.IsNullOrWhiteSpace(state.Response))
                AppendConversation("Lila", state.Response);
        }
        if (state.Error is not null)
            _connection.Text = state.Error;
    }

    private void AppendConversation(string speaker, string text)
    {
        _conversation.AppendText($"{speaker}: {text}{Environment.NewLine}{Environment.NewLine}");
        if (_conversation.TextLength > 200_000)
        {
            _conversation.Select(0, _conversation.TextLength - 150_000);
            _conversation.SelectedText = "[Earlier messages omitted from this window]\r\n\r\n";
        }
        _conversation.SelectionStart = _conversation.TextLength;
        _conversation.ScrollToCaret();
    }

    private async void ShowSettings()
    {
        var previousDistribution = _settings.Distribution;
        using var form = new SettingsForm(_settings, _credentials);
        if (form.ShowDialog(this) == DialogResult.OK)
        {
            _shortcutSummary.Text = ShortcutSummary();
            _shortcuts.SetReady(false);
            _shortcuts.Dispose();
            if (_keyboard is not null)
            {
                _keyboard.NonModifierKeyDown -= _shortcuts.NotifyNonModifierKeyDown;
                _keyboard.KeyboardDeviceRemoved -= _shortcuts.NotifyDeviceRemoved;
                _keyboard.SetShortcuts(ConfiguredBindings());
            }
            _shortcuts = CreateShortcutController();
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
            await RefreshStatusAsync();
        }
    }

    private void ShowDiagnostic()
    {
        if (_keyboard is null)
        {
            MessageBox.Show(this, "Raw keyboard input is unavailable.", "Keyboard diagnostic",
                MessageBoxButtons.OK, MessageBoxIcon.Warning);
            return;
        }
        _diagnosticOpen = true;
        using var diagnostics = new InteractionDiagnostics();
        _interactionDiagnostics = diagnostics;
        void OnRequest(ApiRequestDiagnostic request) =>
            diagnostics.RecordRequest(request, _voice.Snapshot.State);
        _api.RequestCompleted += OnRequest;
        try
        {
            using var form = new KeyboardDiagnosticForm(_keyboard, diagnostics,
                () => _voice.Snapshot.State);
            form.ShowDialog(this);
        }
        finally
        {
            _api.RequestCompleted -= OnRequest;
            _interactionDiagnostics = null;
            _diagnosticOpen = false;
        }
    }

    public void OpenDiagnosticMode() => ShowDiagnostic();

    private void ShowWindow()
    {
        ShowInTaskbar = true;
        Show();
        WindowState = FormWindowState.Normal;
        Activate();
    }

    private async void ExitApplication()
    {
        if (_exitRequested)
            return;
        _exitRequested = true;
        _shortcuts.SetReady(false);
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
        _tray.Visible = false;
        _tray.Dispose();
        _keyboard?.Dispose();
        _shortcuts.Dispose();
        _http.Dispose();
        base.OnFormClosing(e);
    }

    private static string SafeUiError(Exception exception) => exception switch
    {
        LilaApiException apiError => apiError.Message,
        _ => "The operation failed. Check WSL, Lila service status, and Settings."
    };
}
