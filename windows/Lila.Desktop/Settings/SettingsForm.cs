using Lila.Desktop.Core.Keyboard;
using Lila.Desktop.Island;
using Lila.Desktop.Security;
using Lila.Desktop.Startup;
using System.Drawing.Drawing2D;

namespace Lila.Desktop.Settings;

public sealed class SettingsForm : Form
{
    private static readonly Color SurfaceColor = Color.FromArgb(20, 24, 32);
    private static readonly Color CardColor = Color.FromArgb(31, 35, 46);
    private static readonly Color TextColor = Color.FromArgb(242, 243, 249);
    private static readonly Color MutedColor = Color.FromArgb(160, 168, 186);
    private static readonly Color AccentColor = Color.FromArgb(124, 156, 255);

    private readonly AppSettings _settings;
    private readonly WindowsCredentialStore _credentials;
    private readonly TextBox _distro = new();
    private readonly CheckBox _autoStart = new() { Text = "Start Lila at my Windows sign-in", AutoSize = true };
    private readonly CheckBox _holdEnabled = new() { Text = "Enable", AutoSize = true };
    private readonly ComboBox _holdShortcut = new();
    private readonly NumericUpDown _armingDelay = new() { Minimum = 100, Maximum = 600, Increment = 10, Width = 90 };
    private readonly CheckBox _toggleEnabled = new() { Text = "Enable", AutoSize = true };
    private readonly ComboBox _toggleShortcut = new();
    private readonly TextBox _token = new();
    private readonly Label _credentialStatus = new();
    private readonly List<Panel> _sections = [];

    public SettingsForm(AppSettings settings, WindowsCredentialStore credentials)
    {
        SuspendLayout();
        _settings = settings;
        _credentials = credentials;
        Text = "Lila Settings";
        FormBorderStyle = FormBorderStyle.None;
        ShowInTaskbar = false;
        MaximizeBox = false;
        MinimizeBox = false;
        StartPosition = FormStartPosition.CenterParent;
        AutoScaleDimensions = new SizeF(96f, 96f);
        AutoScaleMode = AutoScaleMode.Dpi;
        BackColor = SurfaceColor;
        ForeColor = TextColor;
        Font = new Font("Segoe UI", 9f);
        ClientSize = new Size(520, 610);
        MinimumSize = new Size(480, 560);

        _distro.Text = settings.Distribution;
        _autoStart.Checked = settings.AutoStart;
        _holdEnabled.Checked = settings.HoldToTalkEnabled;
        _toggleEnabled.Checked = settings.ToggleToTalkEnabled;
        _armingDelay.Value = Math.Clamp(settings.RightControlArmingDelayMilliseconds, 100, 600);
        ConfigureShortcut(_holdShortcut,
            new ShortcutBinding(settings.HoldToTalkVirtualKey, settings.HoldToTalkModifiers),
            ShortcutActionMode.Hold);
        ConfigureShortcut(_toggleShortcut,
            new ShortcutBinding(settings.ToggleToTalkVirtualKey, settings.ToggleToTalkModifiers),
            ShortcutActionMode.Toggle);
        _token.UseSystemPasswordChar = true;
        _token.PlaceholderText = "Paste existing Lila API token";
        _credentialStatus.Text = _credentials.ReadToken() is null
            ? "No credential saved"
            : "Credential saved in Windows Credential Manager";
        _credentialStatus.AutoSize = false;
        _credentialStatus.TextAlign = ContentAlignment.MiddleLeft;
        _credentialStatus.ForeColor = MutedColor;

        StyleTextBox(_distro);
        StyleTextBox(_token);
        StyleComboBox(_holdShortcut);
        StyleComboBox(_toggleShortcut);
        StyleNumeric(_armingDelay);
        StyleCheckBox(_autoStart);
        StyleCheckBox(_holdEnabled);
        StyleCheckBox(_toggleEnabled);

        var root = new TableLayoutPanel
        {
            Dock = DockStyle.Fill,
            BackColor = SurfaceColor,
            ColumnCount = 1,
            RowCount = 3,
            Padding = new Padding(18, 12, 18, 14)
        };
        root.RowStyles.Add(new RowStyle(SizeType.Absolute, 58));
        root.RowStyles.Add(new RowStyle(SizeType.Percent, 100));
        root.RowStyles.Add(new RowStyle(SizeType.Absolute, 54));

        var header = BuildHeader();
        var content = new FlowLayoutPanel
        {
            Dock = DockStyle.Fill,
            FlowDirection = FlowDirection.TopDown,
            WrapContents = false,
            AutoScroll = true,
            BackColor = SurfaceColor,
            Padding = new Padding(0, 2, 0, 4),
            Margin = Padding.Empty
        };
        content.Controls.Add(BuildSection("CONNECTION", 2,
            ("WSL distribution", _distro),
            ("Windows startup", Wrap(_autoStart))));
        content.Controls.Add(BuildSection("VOICE SHORTCUTS", 3,
            ("Hold to talk", BuildShortcutPanel(_holdEnabled, _holdShortcut)),
            ("Arming delay", BuildArmingDelayPanel()),
            ("Toggle to talk", BuildShortcutPanel(_toggleEnabled, _toggleShortcut))));
        content.Controls.Add(BuildSection("AUTHENTICATION", 2,
            ("Credential status", _credentialStatus),
            ("API token", BuildCredentialPanel())));
        content.ClientSizeChanged += (_, _) => ResizeSections(content);
        ResizeSections(content);

        var footer = BuildFooter();
        root.Controls.Add(header, 0, 0);
        root.Controls.Add(content, 0, 1);
        root.Controls.Add(footer, 0, 2);
        Controls.Add(root);
        Region = CreateRoundedRegion(ClientRectangle, 16);
        SizeChanged += (_, _) =>
        {
            var oldRegion = Region;
            Region = CreateRoundedRegion(ClientRectangle, 16);
            oldRegion?.Dispose();
            ResizeSections(content);
        };
        AcceptButton = FindButton(footer, "Save");
        CancelButton = FindButton(footer, "Cancel");
        Shown += (_, _) =>
        {
            var workArea = Screen.FromControl(this).WorkingArea;
            MinimumSize = new Size(Math.Min(MinimumSize.Width, workArea.Width),
                Math.Min(MinimumSize.Height, workArea.Height));
            Size = new Size(Math.Min(Width, workArea.Width), Math.Min(Height, workArea.Height));
            Location = new Point(Math.Clamp(Left, workArea.Left, workArea.Right - Width),
                Math.Clamp(Top, workArea.Top, workArea.Bottom - Height));
            ResizeSections(content);
        };
        ResumeLayout(performLayout: true);
    }

    private Control BuildHeader()
    {
        var header = new Panel { Dock = DockStyle.Fill, BackColor = SurfaceColor };
        var gear = new IslandIconButton
        {
            Icon = IslandIcon.Settings,
            Size = new Size(36, 36),
            Location = new Point(0, 8),
            Enabled = false,
            TabStop = false,
            AccessibleName = "Settings"
        };
        var title = new Label
        {
            Text = "Settings",
            AutoSize = true,
            Location = new Point(46, 8),
            ForeColor = TextColor,
            Font = new Font("Segoe UI", 15f, FontStyle.Bold)
        };
        var subtitle = new Label
        {
            Text = "Lila Island",
            AutoSize = true,
            Location = new Point(48, 34),
            ForeColor = MutedColor,
            Font = new Font("Segoe UI", 8.5f)
        };
        var close = new IslandIconButton
        {
            Icon = IslandIcon.Close,
            Size = new Size(32, 32),
            Anchor = AnchorStyles.Top | AnchorStyles.Right,
            AccessibleName = "Close settings",
            Location = new Point(header.ClientSize.Width - 34, 8)
        };
        close.Click += (_, _) =>
        {
            DialogResult = DialogResult.Cancel;
            Close();
        };
        header.Resize += (_, _) => close.Left = header.ClientSize.Width - close.Width;
        header.Controls.Add(gear);
        header.Controls.Add(title);
        header.Controls.Add(subtitle);
        header.Controls.Add(close);
        header.MouseDown += BeginWindowDrag;
        title.MouseDown += BeginWindowDrag;
        subtitle.MouseDown += BeginWindowDrag;
        return header;
    }

    private Control BuildSection(string title, int rowCount,
        params (string Label, Control Value)[] rows)
    {
        var section = new Panel
        {
            Height = 50 + rowCount * 40,
            Width = 460,
            Margin = new Padding(0, 0, 0, 10),
            Padding = new Padding(14, 10, 14, 10),
            BackColor = CardColor
        };
        section.Paint += (_, e) =>
        {
            e.Graphics.SmoothingMode = SmoothingMode.AntiAlias;
            using var path = CreateRoundedPath(
                new RectangleF(0.5f, 0.5f, section.Width - 1, section.Height - 1), 12);
            using var pen = new Pen(Color.FromArgb(34, 255, 255, 255), 1f);
            e.Graphics.DrawPath(pen, path);
        };
        section.SizeChanged += (_, _) =>
        {
            var oldRegion = section.Region;
            using var path = CreateRoundedPath(
                new RectangleF(0, 0, section.Width, section.Height), 12);
            section.Region = new Region(path);
            oldRegion?.Dispose();
        };

        var heading = new Label
        {
            Text = title,
            Dock = DockStyle.Top,
            Height = 24,
            ForeColor = AccentColor,
            Font = new Font("Segoe UI", 8f, FontStyle.Bold),
            TextAlign = ContentAlignment.MiddleLeft
        };
        var grid = new TableLayoutPanel
        {
            Dock = DockStyle.Fill,
            BackColor = Color.Transparent,
            ColumnCount = 2,
            RowCount = rowCount,
            Padding = new Padding(0, 2, 0, 0),
            Margin = Padding.Empty
        };
        grid.ColumnStyles.Add(new ColumnStyle(SizeType.Absolute, 132));
        grid.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 100));
        for (var index = 0; index < rowCount; index++)
        {
            grid.RowStyles.Add(new RowStyle(SizeType.Absolute, 38));
            var (labelText, value) = rows[index];
            var label = new Label
            {
                Text = labelText,
                Dock = DockStyle.Fill,
                ForeColor = MutedColor,
                TextAlign = ContentAlignment.MiddleLeft,
                Margin = new Padding(0, 0, 8, 0)
            };
            value.Dock = DockStyle.Fill;
            value.Margin = new Padding(0, 3, 0, 3);
            grid.Controls.Add(label, 0, index);
            grid.Controls.Add(value, 1, index);
        }
        section.Controls.Add(grid);
        section.Controls.Add(heading);
        _sections.Add(section);
        return section;
    }

    private Control BuildShortcutPanel(CheckBox enabled, ComboBox key)
    {
        var panel = new TableLayoutPanel
        {
            Dock = DockStyle.Fill,
            ColumnCount = 2,
            RowCount = 1,
            BackColor = Color.Transparent,
            Margin = Padding.Empty
        };
        panel.ColumnStyles.Add(new ColumnStyle(SizeType.AutoSize));
        panel.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 100));
        panel.RowStyles.Add(new RowStyle(SizeType.Percent, 100));
        enabled.Anchor = AnchorStyles.Left;
        enabled.Margin = new Padding(0, 0, 10, 0);
        key.Anchor = AnchorStyles.Left | AnchorStyles.Right;
        key.Margin = Padding.Empty;
        panel.Controls.Add(enabled, 0, 0);
        panel.Controls.Add(key, 1, 0);
        return panel;
    }

    private Control BuildArmingDelayPanel()
    {
        var panel = new TableLayoutPanel
        {
            Dock = DockStyle.Fill,
            ColumnCount = 2,
            RowCount = 1,
            BackColor = Color.Transparent,
            Margin = Padding.Empty
        };
        panel.ColumnStyles.Add(new ColumnStyle(SizeType.AutoSize));
        panel.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 100));
        panel.RowStyles.Add(new RowStyle(SizeType.Percent, 100));
        _armingDelay.Width = 90;
        _armingDelay.Anchor = AnchorStyles.Left;
        _armingDelay.Margin = Padding.Empty;
        panel.Controls.Add(_armingDelay, 0, 0);
        panel.Controls.Add(new Label
        {
            Text = "milliseconds",
            Dock = DockStyle.Fill,
            TextAlign = ContentAlignment.MiddleLeft,
            ForeColor = MutedColor,
            Margin = new Padding(8, 0, 0, 0)
        }, 1, 0);
        return panel;
    }

    private Control BuildCredentialPanel()
    {
        var panel = new TableLayoutPanel
        {
            Dock = DockStyle.Fill,
            ColumnCount = 2,
            RowCount = 1,
            BackColor = Color.Transparent,
            Margin = Padding.Empty
        };
        panel.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 100));
        panel.ColumnStyles.Add(new ColumnStyle(SizeType.AutoSize));
        panel.RowStyles.Add(new RowStyle(SizeType.Percent, 100));
        _token.Anchor = AnchorStyles.Left | AnchorStyles.Right;
        _token.Margin = new Padding(0, 0, 8, 0);
        panel.Controls.Add(_token, 0, 0);
        var save = CreateButton("Save token");
        save.Anchor = AnchorStyles.Right;
        save.Margin = Padding.Empty;
        save.Click += (_, _) => SaveCredential();
        panel.Controls.Add(save, 1, 0);
        return panel;
    }

    private Control BuildFooter()
    {
        var footer = new TableLayoutPanel
        {
            Dock = DockStyle.Fill,
            BackColor = SurfaceColor,
            ColumnCount = 3,
            RowCount = 1,
            Margin = Padding.Empty
        };
        footer.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 100));
        footer.ColumnStyles.Add(new ColumnStyle(SizeType.AutoSize));
        footer.ColumnStyles.Add(new ColumnStyle(SizeType.AutoSize));
        footer.RowStyles.Add(new RowStyle(SizeType.Percent, 100));
        var cancel = CreateButton("Cancel");
        cancel.DialogResult = DialogResult.Cancel;
        cancel.Anchor = AnchorStyles.Right;
        cancel.Margin = new Padding(10, 6, 0, 6);
        var save = CreateButton("Save", primary: true);
        save.Anchor = AnchorStyles.Right;
        save.Margin = new Padding(10, 6, 0, 6);
        save.Click += (_, _) => SaveSettings();
        footer.Controls.Add(cancel, 1, 0);
        footer.Controls.Add(save, 2, 0);
        return footer;
    }

    private static Button FindButton(Control parent, string text) =>
        parent.Controls.OfType<Button>().First(button => button.Text == text);

    private static Control Wrap(Control control)
    {
        var panel = new Panel { Dock = DockStyle.Fill, BackColor = Color.Transparent };
        control.Location = new Point(0, 7);
        panel.Controls.Add(control);
        return panel;
    }

    private void ResizeSections(FlowLayoutPanel content)
    {
        var width = Math.Max(1, content.ClientSize.Width - content.Padding.Horizontal
            - SystemInformation.VerticalScrollBarWidth - content.Margin.Horizontal);
        foreach (var section in _sections)
            section.Width = width;
    }

    private static void ConfigureShortcut(ComboBox combo, ShortcutBinding selected,
        ShortcutActionMode mode)
    {
        combo.DropDownStyle = ComboBoxStyle.DropDownList;
        var choices = SupportedChoices(mode);
        combo.Items.AddRange(choices.Cast<object>().ToArray());
        var index = Array.FindIndex(choices, choice => choice.Binding == selected);
        combo.SelectedIndex = index < 0 ? 0 : index;
    }

    private static ShortcutChoice[] SupportedChoices(ShortcutActionMode mode)
    {
        var choices = new List<ShortcutChoice>();
        if (mode == ShortcutActionMode.Hold)
            choices.Add(new ShortcutChoice(new ShortcutBinding(KeyEventInterpreter.RightControlVirtualKey)));
        else
            choices.Add(new ShortcutChoice(new ShortcutBinding(KeyEventInterpreter.PauseVirtualKey)));
        return choices.ToArray();
    }

    private static void StyleTextBox(TextBox textBox)
    {
        textBox.BackColor = Color.FromArgb(40, 45, 58);
        textBox.ForeColor = TextColor;
        textBox.BorderStyle = BorderStyle.FixedSingle;
        textBox.Font = new Font("Segoe UI", 9f);
    }

    private static void StyleComboBox(ComboBox combo)
    {
        combo.BackColor = Color.FromArgb(40, 45, 58);
        combo.ForeColor = TextColor;
        combo.FlatStyle = FlatStyle.Flat;
        combo.Font = new Font("Segoe UI", 9f);
    }

    private static void StyleNumeric(NumericUpDown numeric)
    {
        numeric.BackColor = Color.FromArgb(40, 45, 58);
        numeric.ForeColor = TextColor;
        numeric.BorderStyle = BorderStyle.FixedSingle;
        numeric.TextAlign = HorizontalAlignment.Center;
    }

    private static void StyleCheckBox(CheckBox checkBox)
    {
        checkBox.ForeColor = TextColor;
        checkBox.BackColor = Color.Transparent;
        checkBox.Margin = new Padding(0, 5, 10, 0);
    }

    private static Button CreateButton(string text, bool primary = false)
    {
        var button = new Button
        {
            Text = text,
            AutoSize = true,
            AutoSizeMode = AutoSizeMode.GrowAndShrink,
            MinimumSize = new Size(82, 30),
            FlatStyle = FlatStyle.Flat,
            BackColor = primary ? Color.FromArgb(112, 112, 242) : Color.FromArgb(43, 48, 61),
            ForeColor = TextColor,
            Font = new Font("Segoe UI", 9f, primary ? FontStyle.Bold : FontStyle.Regular),
            Cursor = Cursors.Hand,
            Height = 30,
            Padding = new Padding(8, 0, 8, 0)
        };
        button.FlatAppearance.BorderSize = 1;
        button.FlatAppearance.BorderColor = primary
            ? Color.FromArgb(153, 144, 255)
            : Color.FromArgb(70, 78, 96);
        button.FlatAppearance.MouseOverBackColor = primary
            ? Color.FromArgb(132, 127, 255)
            : Color.FromArgb(57, 64, 80);
        return button;
    }

    private void BeginWindowDrag(object? sender, MouseEventArgs e)
    {
        if (e.Button != MouseButtons.Left)
            return;
        ReleaseCapture();
        SendMessage(Handle, 0xA1, (IntPtr)0x2, IntPtr.Zero);
    }

    private void SaveCredential()
    {
        try
        {
            _credentials.WriteToken(_token.Text);
            _token.Clear();
            _credentialStatus.Text = "Credential saved in Windows Credential Manager";
        }
        catch (Exception ex)
        {
            MessageBox.Show(this, ex.Message, "Credential error", MessageBoxButtons.OK,
                MessageBoxIcon.Error);
        }
    }

    private void SaveSettings()
    {
        var distribution = _distro.Text.Trim();
        if (distribution.Length is < 1 or > 64)
        {
            MessageBox.Show(this, "Enter a valid WSL distribution name.", "Settings",
                MessageBoxButtons.OK, MessageBoxIcon.Warning);
            DialogResult = DialogResult.None;
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
            MessageBox.Show(this, ex.Message, "Unsupported shortcut", MessageBoxButtons.OK,
                MessageBoxIcon.Warning);
            DialogResult = DialogResult.None;
            return;
        }
        if (_holdEnabled.Checked && _toggleEnabled.Checked && ShortcutBinding.Conflicts(hold, toggle))
        {
            MessageBox.Show(this, "Choose different shortcuts for the two enabled modes.", "Settings",
                MessageBoxButtons.OK, MessageBoxIcon.Warning);
            DialogResult = DialogResult.None;
            return;
        }

        _settings.Distribution = distribution;
        _settings.HoldToTalkVirtualKey = hold.VirtualKey;
        _settings.HoldToTalkModifiers = hold.Modifiers;
        _settings.ToggleToTalkVirtualKey = toggle.VirtualKey;
        _settings.ToggleToTalkModifiers = toggle.Modifiers;
        _settings.HoldToTalkEnabled = _holdEnabled.Checked;
        _settings.ToggleToTalkEnabled = _toggleEnabled.Checked;
        _settings.RightControlArmingDelayMilliseconds = (int)_armingDelay.Value;
        _settings.AutoStart = _autoStart.Checked;
        try
        {
            StartupRegistration.SetEnabled(_settings.AutoStart);
            _settings.Save();
        }
        catch (Exception ex)
        {
            MessageBox.Show(this, "Could not save settings: " + ex.Message, "Settings",
                MessageBoxButtons.OK, MessageBoxIcon.Error);
            DialogResult = DialogResult.None;
            return;
        }
        DialogResult = DialogResult.OK;
        Close();
    }

    private static GraphicsPath CreateRoundedPath(RectangleF bounds, float radius)
    {
        var path = new GraphicsPath();
        var diameter = radius * 2;
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

    private static Region CreateRoundedRegion(Rectangle bounds, float radius)
    {
        using var path = CreateRoundedPath(bounds, radius);
        return new Region(path);
    }

    [System.Runtime.InteropServices.DllImport("user32.dll")]
    private static extern bool ReleaseCapture();

    [System.Runtime.InteropServices.DllImport("user32.dll")]
    private static extern IntPtr SendMessage(IntPtr handle, int message, IntPtr wParam, IntPtr lParam);

    private sealed record ShortcutChoice(ShortcutBinding Binding)
    {
        public override string ToString() => Binding.DisplayName;
    }
}
