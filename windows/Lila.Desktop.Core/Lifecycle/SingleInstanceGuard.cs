namespace Lila.Desktop.Core.Lifecycle;

/// <summary>Per-user global named mutex protecting one tray process across sessions.</summary>
public sealed class SingleInstanceGuard : IDisposable
{
    private readonly Mutex _mutex;

    private SingleInstanceGuard(Mutex mutex, bool ownsMutex)
    {
        _mutex = mutex;
        IsFirstInstance = ownsMutex;
    }

    public bool IsFirstInstance { get; }

    public static SingleInstanceGuard Acquire(string userIdentity)
    {
        var safeIdentity = string.Concat(userIdentity.Select(c => char.IsLetterOrDigit(c) ? c : '_'));
        var scope = OperatingSystem.IsWindows() ? "Global\\" : "";
        var mutex = new Mutex(initiallyOwned: true, $"{scope}LilaDesktop_{safeIdentity}", out var created);
        return new SingleInstanceGuard(mutex, created);
    }

    public void Dispose()
    {
        if (IsFirstInstance)
        {
            try { _mutex.ReleaseMutex(); }
            catch (ApplicationException) { }
        }
        _mutex.Dispose();
    }
}

public static class StartupRegistrationCommand
{
    public static string? GetValue(bool enabled, string executablePath) => enabled
        ? $"\"{executablePath}\" --startup"
        : null;
}
