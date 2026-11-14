using System.Drawing.Drawing2D;
using Lila.Desktop.Core.Keyboard;
using Lila.Desktop.Island;
using Lila.Desktop.Security;
using Lila.Desktop.Startup;

namespace Lila.Desktop.Settings;

/// <summary>
/// In-panel glass sheet inside the 380x560 Lila Island panel with a ← Back button.
/// Restyles the existing Connection, Voice Shortcuts, and Authentication settings
/// without opening a separate modal window.
/// </summary>
public sealed class SettingsSheetControl : UserControl
{
    private static readonly Color CardColor = Color.FromArgb(28, 33, 45);
    private static readonly Color TextColor = Color.FromArgb(245, 247, 250);
    private static readonly Color MutedColor = Color.FromArgb(167, 176, 192);
    private static readonly Color AccentColor = Color.FromArgb(124, 156, 255);

    private readonly AppSettings _settings;
    private readonly WindowsCredentialStore _credentials;
    private readonly TextBox _distro = new();
    private readonly CheckBox _autoStart = new() { Text = "Start Lila at Windows sign-in", AutoSize = true };
    private readonly CheckBox _sentinelOptIn = new() { Text = "Sentinel read-only host telemetry", AutoSize = true };
    private readonly CheckBox _holdEnabled = new() { Text = "Enable", AutoSize = true };
    private readonly ComboBox _holdShortcut = new();
    private readonly NumericUpDown _armingDelay = new() { Minimum = 100, Maximum = 600, Increment = 10, Width = 80 };
    private readonly CheckBox _toggleEnabled = new() { Text = "Enable", AutoSize = true };
    private readonly ComboBox _toggleShortcut = new();
    private readonly TextBox _token = new();
    private readonly Label _credentialStatus = new();
    private readonly Label _statusBanner = new() { AutoSize = true, ForeColor = Color.FromArgb(94, 224, 160) };
    private readonly FlowLayoutPanel _content;
    private readonly List<Panel> _sections = [];

    public event Action? BackRequested;
    public event Action<string>? SettingsSaved;

    public SettingsSheetControl(AppSettings settings, WindowsCredentialStore credentials)
    {
        SuspendLayout();
        _settings = settings;
        _credentials = credentials;
        SetStyle(ControlStyles.UserPaint | ControlStyles.AllPaintingInWmPaint |
                 ControlStyles.OptimizedDoubleBuffer | ControlStyles.ResizeRedraw |
                 ControlStyles.Opaque, true);
        BackColor = Color.FromArgb(20, 24, 32);
        ForeColor = TextColor;
        Font = new Font("Segoe UI", 9f);

        StyleTextBox(_distro);
        StyleTextBox(_token);
        _token.UseSystemPasswordChar = true;
        _token.PlaceholderText = "Paste Lila API token";
        StyleComboBox(_holdShortcut);
        StyleComboBox(_toggleShortcut);
        StyleNumeric(_armingDelay);
        StyleCheckBox(_autoStart);
        StyleCheckBox(_sentinelOptIn);
        StyleCheckBox(_holdEnabled);
        StyleCheckBox(_toggleEnabled);

        _credentialStatus.AutoSize = false;
        _credentialStatus.TextAlign = ContentAlignment.MiddleLeft;
        _credentialStatus.ForeColor = MutedColor;
        _credentialStatus.BackColor = CardColor;

        var root = new TableLayoutPanel
        {
            Dock = DockStyle.Fill,
            BackColor = Color.FromArgb(20, 24, 32),
            ColumnCount = 1,
            RowCount = 3,
            Padding = new Padding(12, 10, 12, 12)
        };
        root.RowStyles.Add(new RowStyle(SizeType.Absolute, 48));
        root.RowStyles.Add(new RowStyle(SizeType.Percent, 100));
        root.RowStyles.Add(new RowStyle(SizeType.Absolute, 44));

        var header = BuildHeader();
        _content = new FlowLayoutPanel
        {
            Dock = DockStyle.Fill,
            FlowDirection = FlowDirection.TopDown,
            WrapContents = false,
            AutoScroll = true,
            BackColor = Color.FromArgb(20, 24, 32),
            Padding = new Padding(0, 2, 0, 4),
            Margin = Padding.Empty
        };
        _content.Controls.Add(BuildSection("CONNECTION", 3,
            ("WSL distro", _distro),
            ("Startup", Wrap(_autoStart)),
            ("Sentinel", Wrap(_sentinelOptIn))));
        _content.Controls.Add(BuildSection("VOICE SHORTCUTS", 3,
            ("Hold to talk", BuildShortcutRow(_holdEnabled, _holdShortcut)),
            ("Arming delay", BuildArmingDelayRow()),
            ("Toggle voice", BuildShortcutRow(_toggleEnabled, _toggleShortcut))));
        _content.Controls.Add(BuildSection("AUTHENTICATION", 2,
            ("Status", _credentialStatus),
            ("API token", BuildTokenRow())));
        _content.ClientSizeChanged += (_, _) => ResizeSections();

        var footer = BuildFooter();
        root.Controls.Add(header, 0, 0);
        root.Controls.Add(_content, 0, 1);
        root.Controls.Add(footer, 0, 2);
        Controls.Add(root);

        ReloadFromSettings();
        ResumeLayout(performLayout: true);
    }

    public void ReloadFromSettings()
    {
        _distro.Text = _settings.Distribution;
        _autoStart.Checked = _settings.AutoStart;
        _sentinelOptIn.Checked = _settings.SentinelOptIn;
        _holdEnabled.Checked = _settings.HoldToTalkEnabled;
        _toggleEnabled.Checked = _settings.ToggleToTalkEnabled;
        _armingDelay.Value = Math.Clamp(_settings.RightControlArmingDelayMilliseconds, 100, 600);
        ConfigureShortcut(_holdShortcut,
            new ShortcutBinding(_settings.HoldToTalkVirtualKey, _settings.HoldToTalkModifiers),
            ShortcutActionMode.Hold);
        ConfigureShortcut(_toggleShortcut,
            new ShortcutBinding(_settings.ToggleToTalkVirtualKey, _settings.ToggleToTalkModifiers),
            ShortcutActionMode.Toggle);
        _credentialStatus.Text = _credentials.ReadToken() is null
            ? "No token stored"
            : "Saved in Credential Manager";
        _statusBanner.Text = string.Empty;
        ResizeSections();
    }

    private Control BuildHeader()
    {
        var header = new Panel { Dock = DockStyle.Fill, BackColor = Color.FromArgb(20, 24, 32) };
        var backBtn = new IslandIconButton
        {
            Icon = IslandIcon.ArrowLeft,
            Size = new Size(32, 32),
            Location = new Point(2, 6),
            AccessibleName = "Back to Lila Island"
        };
        backBtn.Click += (_, _) => BackRequested?.Invoke();

        var title = new Label
        {
            Text = "Settings",
            AutoSize = true,
            Location = new Point(42, 6),
            ForeColor = TextColor,
            BackColor = Color.FromArgb(20, 24, 32),
            Font = new Font("Segoe UI", 12f, FontStyle.Bold)
        };
        var subtitle = new Label
        {
            Text = "Connection, shortcuts & credentials",
            AutoSize = true,
            Location = new Point(43, 27),
            ForeColor = MutedColor,
            BackColor = Color.FromArgb(20, 24, 32),
            Font = new Font("Segoe UI", 8f)
        };
        header.Controls.Add(backBtn);
        header.Controls.Add(title);
        header.Controls.Add(subtitle);
        return header;
    }

    private Control BuildSection(string title, int rowCount, params (string Label, Control Value)[] rows)
    {
        var section = new Panel
        {
            Height = 42 + rowCount * 36,
            Width = 330,
            Margin = new Padding(0, 0, 0, 8),
            Padding = new Padding(11, 8, 11, 8),
            BackColor = CardColor
        };
        section.Paint += (_, e) =>
        {
            e.Graphics.SmoothingMode = SmoothingMode.AntiAlias;
            using var path = CreateRoundedPath(
                new RectangleF(0.5f, 0.5f, section.Width - 1, section.Height - 1), 11);
            var sheenRect = new Rectangle(0, 0, section.Width, Math.Max(14, section.Height / 3));
            using (var sheen = new LinearGradientBrush(
                       sheenRect,
                       Color.FromArgb(18, 255, 255, 255),
                       Color.FromArgb(0, 255, 255, 255),
                       LinearGradientMode.Vertical))
            {
                e.Graphics.SetClip(path);
                e.Graphics.FillRectangle(sheen, sheenRect);
                e.Graphics.ResetClip();
            }
            using var pen = new Pen(Color.FromArgb(44, 255, 255, 255), 1f);
            e.Graphics.DrawPath(pen, path);
        };
        section.SizeChanged += (_, _) =>
        {
            if (section.Width <= 0 || section.Height <= 0)
                return;
            var old = section.Region;
            using var path = CreateRoundedPath(new RectangleF(0, 0, section.Width, section.Height), 11);
            section.Region = new Region(path);
            old?.Dispose();
        };

        var heading = new Label
        {
            Text = title,
            Dock = DockStyle.Top,
            Height = 20,
            ForeColor = AccentColor,
            Font = new Font("Segoe UI", 7.75f, FontStyle.Bold),
            TextAlign = ContentAlignment.MiddleLeft
        };
        var grid = new TableLayoutPanel
        {
            Dock = DockStyle.Fill,
            BackColor = CardColor,
            ColumnCount = 2,
            RowCount = rowCount,
            Margin = Padding.Empty
        };
        grid.ColumnStyles.Add(new ColumnStyle(SizeType.Absolute, 96));
        grid.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 100));
        for (var i = 0; i < rowCount; i++)
        {
            grid.RowStyles.Add(new RowStyle(SizeType.Absolute, 34));
            var (labelText, value) = rows[i];
            var lbl = new Label
            {
                Text = labelText,
                Dock = DockStyle.Fill,
                ForeColor = MutedColor,
                TextAlign = ContentAlignment.MiddleLeft,
                Font = new Font("Segoe UI", 8.5f)
            };
            value.Dock = DockStyle.Fill;
            value.Margin = new Padding(0, 3, 0, 3);
            grid.Controls.Add(lbl, 0, i);
            grid.Controls.Add(value, 1, i);
        }
        section.Controls.Add(grid);
        section.Controls.Add(heading);
        _sections.Add(section);
        return section;
    }

    private static Control BuildShortcutRow(CheckBox enabled, ComboBox key)
    {
        var panel = new TableLayoutPanel
        {
            Dock = DockStyle.Fill,
            ColumnCount = 2,
            RowCount = 1,
            BackColor = CardColor,
            Margin = Padding.Empty
        };
        panel.ColumnStyles.Add(new ColumnStyle(SizeType.AutoSize));
        panel.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 100));
        panel.RowStyles.Add(new RowStyle(SizeType.Percent, 100));
        enabled.Anchor = AnchorStyles.Left;
        enabled.Margin = new Padding(0, 0, 8, 0);
        key.Anchor = AnchorStyles.Left | AnchorStyles.Right;
        key.Margin = Padding.Empty;
        panel.Controls.Add(enabled, 0, 0);
        panel.Controls.Add(key, 1, 0);
        return panel;
    }

    private Control BuildArmingDelayRow()
    {
        var panel = new TableLayoutPanel
        {
            Dock = DockStyle.Fill,
            ColumnCount = 2,
            RowCount = 1,
            BackColor = CardColor,
            Margin = Padding.Empty
        };
        panel.ColumnStyles.Add(new ColumnStyle(SizeType.AutoSize));
        panel.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 100));
        panel.RowStyles.Add(new RowStyle(SizeType.Percent, 100));
        _armingDelay.Anchor = AnchorStyles.Left;
        _armingDelay.Margin = Padding.Empty;
        panel.Controls.Add(_armingDelay, 0, 0);
        panel.Controls.Add(new Label
        {
            Text = "ms",
            Dock = DockStyle.Fill,
            TextAlign = ContentAlignment.MiddleLeft,
            ForeColor = MutedColor,
            BackColor = CardColor,
            Margin = new Padding(6, 0, 0, 0)
        }, 1, 0);
        return panel;
    }

    private Control BuildTokenRow()
    {
        var panel = new TableLayoutPanel
        {
            Dock = DockStyle.Fill,
            ColumnCount = 2,
            RowCount = 1,
            BackColor = CardColor,
            Margin = Padding.Empty
        };
        panel.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 100));
        panel.ColumnStyles.Add(new ColumnStyle(SizeType.AutoSize));
        panel.RowStyles.Add(new RowStyle(SizeType.Percent, 100));
        _token.Anchor = AnchorStyles.Left | AnchorStyles.Right;
        _token.Margin = new Padding(0, 0, 6, 0);
        panel.Controls.Add(_token, 0, 0);
        var saveBtn = CreateButton("Save token", primary: false);
        saveBtn.MinimumSize = new Size(76, 26);
        saveBtn.Height = 26;
        saveBtn.Anchor = AnchorStyles.Right;
        saveBtn.Margin = Padding.Empty;
        saveBtn.Click += (_, _) => SaveCredential();
        panel.Controls.Add(saveBtn, 1, 0);
        return panel;
    }

    private Control BuildFooter()
    {
        var footer = new TableLayoutPanel
        {
            Dock = DockStyle.Fill,
            BackColor = Color.FromArgb(20, 24, 32),
            ColumnCount = 3,
            RowCount = 1,
            Margin = Padding.Empty
        };
        footer.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 100));
        footer.ColumnStyles.Add(new ColumnStyle(SizeType.AutoSize));
        footer.ColumnStyles.Add(new ColumnStyle(SizeType.AutoSize));
        footer.RowStyles.Add(new RowStyle(SizeType.Percent, 100));

        _statusBanner.Anchor = AnchorStyles.Left;
        var cancel = CreateButton("Back");
        cancel.Anchor = AnchorStyles.Right;
        cancel.Margin = new Padding(6, 6, 0, 6);
        cancel.Click += (_, _) => BackRequested?.Invoke();

        var save = CreateButton("Save settings", primary: true);
        save.Anchor = AnchorStyles.Right;
        save.Margin = new Padding(8, 6, 0, 6);
        save.Click += (_, _) => SaveAllSettings();

        footer.Controls.Add(_statusBanner, 0, 0);
        footer.Controls.Add(cancel, 1, 0);
        footer.Controls.Add(save, 2, 0);
        return footer;
    }

    private void SaveCredential()
    {
        try
        {
            _credentials.WriteToken(_token.Text);
            _token.Clear();
            _credentialStatus.Text = "Saved in Credential Manager";
            _statusBanner.ForeColor = Color.FromArgb(94, 224, 160);
            _statusBanner.Text = "Token saved.";
        }
        catch (Exception ex)
        {
            _statusBanner.ForeColor = Color.FromArgb(255, 138, 128);
            _statusBanner.Text = ex.Message;
        }
    }

    private void SaveAllSettings()
    {
        var previousDistro = _settings.Distribution;
        var distro = _distro.Text.Trim();
        if (distro.Length is < 1 or > 64)
        {
            _statusBanner.ForeColor = Color.FromArgb(255, 138, 128);
            _statusBanner.Text = "Enter a valid WSL distro.";
            return;
        }
        var hold = ((ShortcutChoice)_holdShortcut.SelectedItem!).Binding;
        var toggle = ((ShortcutChoice)_toggleShortcut.SelectedItem!).Binding;
        try
        {
            if (_holdEnabled.Checked)
                hold.Validate(ShortcutActionMode.Hold);
            if (_toggleEnabled.Checked)
                toggle.Validate(ShortcutActionMode.Toggle);
        }
        catch (ArgumentException ex)
        {
            _statusBanner.ForeColor = Color.FromArgb(255, 138, 128);
            _statusBanner.Text = ex.Message;
            return;
        }
        if (_holdEnabled.Checked && _toggleEnabled.Checked && ShortcutBinding.Conflicts(hold, toggle))
        {
            _statusBanner.ForeColor = Color.FromArgb(255, 138, 128);
            _statusBanner.Text = "Shortcuts conflict.";
            return;
        }

        _settings.Distribution = distro;
        _settings.HoldToTalkVirtualKey = hold.VirtualKey;
        _settings.HoldToTalkModifiers = hold.Modifiers;
        _settings.ToggleToTalkVirtualKey = toggle.VirtualKey;
        _settings.ToggleToTalkModifiers = toggle.Modifiers;
        _settings.HoldToTalkEnabled = _holdEnabled.Checked;
        _settings.ToggleToTalkEnabled = _toggleEnabled.Checked;
        _settings.RightControlArmingDelayMilliseconds = (int)_armingDelay.Value;
        _settings.AutoStart = _autoStart.Checked;
        _settings.SentinelOptIn = _sentinelOptIn.Checked;
        try
        {
            StartupRegistration.SetEnabled(_settings.AutoStart);
            _settings.Save();
        }
        catch (Exception ex)
        {
            _statusBanner.ForeColor = Color.FromArgb(255, 138, 128);
            _statusBanner.Text = "Save failed: " + ex.Message;
            return;
        }
        _statusBanner.ForeColor = Color.FromArgb(94, 224, 160);
        _statusBanner.Text = "Saved";
        SettingsSaved?.Invoke(previousDistro);
    }

    private void ResizeSections()
    {
        var width = Math.Max(200, _content.ClientSize.Width - _content.Padding.Horizontal - 6);
        foreach (var section in _sections)
            section.Width = width;
    }

    private static Control Wrap(Control control)
    {
        var panel = new Panel { Dock = DockStyle.Fill, BackColor = CardColor };
        control.Location = new Point(0, 5);
        panel.Controls.Add(control);
        return panel;
    }

    private static void ConfigureShortcut(ComboBox combo, ShortcutBinding selected, ShortcutActionMode mode)
    {
        combo.DropDownStyle = ComboBoxStyle.DropDownList;
        combo.Items.Clear();
        var choice = mode == ShortcutActionMode.Hold
            ? new ShortcutChoice(new ShortcutBinding(KeyEventInterpreter.RightControlVirtualKey))
            : new ShortcutChoice(new ShortcutBinding(KeyEventInterpreter.PauseVirtualKey));
        combo.Items.Add(choice);
        combo.SelectedIndex = 0;
    }

    private static void StyleTextBox(TextBox textBox)
    {
        textBox.BackColor = Color.FromArgb(38, 43, 56);
        textBox.ForeColor = TextColor;
        textBox.BorderStyle = BorderStyle.FixedSingle;
        textBox.Font = new Font("Segoe UI", 8.75f);
    }

    private static void StyleComboBox(ComboBox combo)
    {
        combo.BackColor = Color.FromArgb(38, 43, 56);
        combo.ForeColor = TextColor;
        combo.FlatStyle = FlatStyle.Flat;
        combo.Font = new Font("Segoe UI", 8.5f);
    }

    private static void StyleNumeric(NumericUpDown numeric)
    {
        numeric.BackColor = Color.FromArgb(38, 43, 56);
        numeric.ForeColor = TextColor;
        numeric.BorderStyle = BorderStyle.FixedSingle;
        numeric.TextAlign = HorizontalAlignment.Center;
    }

    private static void StyleCheckBox(CheckBox checkBox)
    {
        checkBox.ForeColor = TextColor;
        checkBox.BackColor = CardColor;
        checkBox.Font = new Font("Segoe UI", 8.5f);
    }

    private static Button CreateButton(string text, bool primary = false)
    {
        var button = new Button
        {
            Text = text,
            AutoSize = true,
            AutoSizeMode = AutoSizeMode.GrowAndShrink,
            MinimumSize = new Size(72, 28),
            FlatStyle = FlatStyle.Flat,
            BackColor = primary ? Color.FromArgb(124, 120, 248) : Color.FromArgb(40, 45, 58),
            ForeColor = TextColor,
            Font = new Font("Segoe UI", 8.5f, primary ? FontStyle.Bold : FontStyle.Regular),
            Cursor = Cursors.Hand,
            Height = 28,
            Padding = new Padding(8, 0, 8, 0)
        };
        button.FlatAppearance.BorderSize = 1;
        button.FlatAppearance.BorderColor = primary
            ? Color.FromArgb(158, 148, 255)
            : Color.FromArgb(68, 76, 94);
        return button;
    }

    private static GraphicsPath CreateRoundedPath(RectangleF bounds, float radius)
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

    private sealed record ShortcutChoice(ShortcutBinding Binding)
    {
        public override string ToString() => Binding.DisplayName;
    }
}
