using System.Text.Json;
using Lila.Desktop.Core.Keyboard;

namespace Lila.Desktop.Settings;

public sealed class AppSettings
{
    public string Distribution { get; set; } = "Ubuntu-24.04";
    public ushort HoldToTalkVirtualKey { get; set; } = 0xA3; // Right Ctrl
    public ShortcutModifiers HoldToTalkModifiers { get; set; } = ShortcutModifiers.None;
    public ushort ToggleToTalkVirtualKey { get; set; } = 0x13; // ThinkPad Fn+P / Pause
    public ShortcutModifiers ToggleToTalkModifiers { get; set; } = ShortcutModifiers.None;
    public bool HoldToTalkEnabled { get; set; } = true;
    public bool ToggleToTalkEnabled { get; set; } = true;
    public int RightControlArmingDelayMilliseconds { get; set; } = 180;
    public bool AutoStart { get; set; } = false;

    public static string SettingsPath => Path.Combine(
        Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
        "LilaDesktop", "settings.json");

    public static AppSettings Load()
    {
        try
        {
            if (!File.Exists(SettingsPath))
                return new AppSettings();
            var settings = JsonSerializer.Deserialize<AppSettings>(File.ReadAllText(SettingsPath))
                           ?? new AppSettings();
            try
            {
                if (settings.HoldToTalkEnabled)
                    new ShortcutBinding(settings.HoldToTalkVirtualKey, settings.HoldToTalkModifiers)
                        .Validate(ShortcutActionMode.Hold);
                if (settings.ToggleToTalkEnabled)
                    new ShortcutBinding(settings.ToggleToTalkVirtualKey, settings.ToggleToTalkModifiers)
                        .Validate(ShortcutActionMode.Toggle);
                if (settings.HoldToTalkEnabled && settings.ToggleToTalkEnabled &&
                    ShortcutBinding.Conflicts(
                        new ShortcutBinding(settings.HoldToTalkVirtualKey, settings.HoldToTalkModifiers),
                        new ShortcutBinding(settings.ToggleToTalkVirtualKey, settings.ToggleToTalkModifiers)))
                    throw new ArgumentException("Hold and toggle shortcuts conflict.");
                return settings;
            }
            catch (ArgumentException)
            {
                settings.HoldToTalkVirtualKey = KeyEventInterpreter.RightControlVirtualKey;
                settings.HoldToTalkModifiers = ShortcutModifiers.None;
                settings.ToggleToTalkVirtualKey = KeyEventInterpreter.PauseVirtualKey;
                settings.ToggleToTalkModifiers = ShortcutModifiers.None;
                settings.HoldToTalkEnabled = true;
                settings.ToggleToTalkEnabled = true;
                return settings;
            }
        }
        catch
        {
            return new AppSettings();
        }
    }

    public void Save()
    {
        var directory = Path.GetDirectoryName(SettingsPath)!;
        Directory.CreateDirectory(directory);
        var temporary = SettingsPath + ".tmp";
        File.WriteAllText(temporary, JsonSerializer.Serialize(this,
            new JsonSerializerOptions { WriteIndented = true }));
        File.Move(temporary, SettingsPath, overwrite: true);
    }
}
