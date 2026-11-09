using Lila.Desktop.Core.Keyboard;
using Lila.Desktop.Security;
using Lila.Desktop.Startup;

namespace Lila.Desktop.Settings;

public sealed class SettingsForm : Form
{
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

    public SettingsForm(AppSettings settings, WindowsCredentialStore credentials)
    {
        _settings = settings;
        _credentials = credentials;
        Text = "Lila Settings";
        FormBorderStyle = FormBorderStyle.FixedDialog;
        MaximizeBox = false;
        MinimizeBox = false;
        StartPosition = FormStartPosition.CenterParent;
        ClientSize = new Size(520, 380);

        var table = new TableLayoutPanel
        {
            Dock = DockStyle.Fill,
            Padding = new Padding(14),
            ColumnCount = 2,
            RowCount = 8,
            AutoSize = false
        };
        table.ColumnStyles.Add(new ColumnStyle(SizeType.Absolute, 135));
        table.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 100));
        for (var i = 0; i < 8; i++)
            table.RowStyles.Add(new RowStyle(SizeType.Absolute, i is 2 or 4 ? 48 : 38));

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
        _credentialStatus.AutoSize = true;

        AddRow(table, 0, "WSL distribution", _distro);
        AddRow(table, 1, "Windows startup", _autoStart);
        AddRow(table, 2, "Hold-to-talk key", BuildShortcutPanel(_holdEnabled, _holdShortcut));
        AddRow(table, 3, "Arming delay (ms)", _armingDelay);
        AddRow(table, 4, "Toggle-to-talk key", BuildShortcutPanel(_toggleEnabled, _toggleShortcut));
        AddRow(table, 5, "API credential", BuildCredentialPanel());
        AddRow(table, 6, "", _credentialStatus);

        var buttons = new FlowLayoutPanel
        {
            Dock = DockStyle.Fill,
            FlowDirection = FlowDirection.RightToLeft
        };
        var save = new Button { Text = "Save", DialogResult = DialogResult.OK, AutoSize = true };
        var cancel = new Button { Text = "Cancel", DialogResult = DialogResult.Cancel, AutoSize = true };
        save.Click += (_, _) => SaveSettings();
        buttons.Controls.Add(save);
        buttons.Controls.Add(cancel);
        table.Controls.Add(buttons, 0, 7);
        table.SetColumnSpan(buttons, 2);
        Controls.Add(table);
        AcceptButton = save;
        CancelButton = cancel;
    }

    private static void ConfigureShortcut(ComboBox combo, ShortcutBinding selected,
        ShortcutActionMode mode)
    {
        combo.DropDownStyle = ComboBoxStyle.DropDownList;
        var choices = SupportedChoices(mode);
        combo.Items.AddRange(choices.Cast<object>().ToArray());
        var index = Array.FindIndex(choices, choice => choice.Binding == selected);
        if (index < 0)
            index = 0;
        combo.SelectedIndex = index;
        combo.Width = 210;
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

    private static Control BuildShortcutPanel(CheckBox enabled, ComboBox key)
    {
        var panel = new FlowLayoutPanel { Dock = DockStyle.Fill, WrapContents = false };
        panel.Controls.Add(enabled);
        panel.Controls.Add(key);
        return panel;
    }

    private Control BuildCredentialPanel()
    {
        var panel = new TableLayoutPanel { Dock = DockStyle.Fill, ColumnCount = 2 };
        panel.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 100));
        panel.ColumnStyles.Add(new ColumnStyle(SizeType.AutoSize));
        panel.Controls.Add(_token, 0, 0);
        var save = new Button { Text = "Save token", AutoSize = true };
        save.Click += (_, _) => SaveCredential();
        panel.Controls.Add(save, 1, 0);
        return panel;
    }

    private static void AddRow(TableLayoutPanel table, int row, string title, Control value)
    {
        var label = new Label { Text = title, Dock = DockStyle.Fill, TextAlign = ContentAlignment.MiddleLeft };
        value.Dock = DockStyle.Fill;
        table.Controls.Add(label, 0, row);
        table.Controls.Add(value, 1, row);
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

    private sealed record ShortcutChoice(ShortcutBinding Binding)
    {
        public override string ToString() => Binding.DisplayName;
    }
}
