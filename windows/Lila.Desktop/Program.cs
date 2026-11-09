using Lila.Desktop.Settings;
using Lila.Desktop.Core.Lifecycle;
using Lila.Desktop.Startup;
using System.Security.Principal;

namespace Lila.Desktop;

internal static class Program
{
    [STAThread]
    private static void Main(string[] args)
    {
        ApplicationConfiguration.Initialize();
        var settings = AppSettings.Load();
        var userIdentity = WindowsIdentity.GetCurrent().User?.Value ?? Environment.UserName;
        using var singleInstance = SingleInstanceGuard.Acquire(userIdentity);
        if (!singleInstance.IsFirstInstance)
            return;
        var startup = args.Any(argument => argument.Equals("--startup", StringComparison.OrdinalIgnoreCase));
        try { StartupRegistration.SetEnabled(settings.AutoStart); }
        catch (Exception ex)
        {
            MessageBox.Show($"Could not update Lila's current-user startup setting: {ex.Message}",
                "Lila startup setting", MessageBoxButtons.OK, MessageBoxIcon.Warning);
        }
        using var form = new MainForm(settings, startup);
        if (args.Any(argument => argument.Equals("--diagnostic", StringComparison.OrdinalIgnoreCase)))
            form.Shown += (_, _) => form.OpenDiagnosticMode();
        Application.Run(form);
    }
}
