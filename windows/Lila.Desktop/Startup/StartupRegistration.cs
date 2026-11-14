using Microsoft.Win32;
using Lila.Desktop.Core.Lifecycle;

namespace Lila.Desktop.Startup;

internal static class StartupRegistration
{
    private const string RunSubKey = @"Software\Microsoft\Windows\CurrentVersion\Run";
    private const string ValueName = "LilaDesktop";

    public static void SetEnabled(bool enabled)
    {
        using var key = Registry.CurrentUser.CreateSubKey(RunSubKey, writable: true)
            ?? throw new InvalidOperationException("Could not open the current-user Startup registry key.");
        var value = StartupRegistrationCommand.GetValue(enabled, Application.ExecutablePath);
        if (value is not null)
            key.SetValue(ValueName, value,
                RegistryValueKind.String);
        else
            key.DeleteValue(ValueName, throwOnMissingValue: false);
    }
}
