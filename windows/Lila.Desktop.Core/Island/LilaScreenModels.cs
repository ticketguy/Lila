namespace Lila.Desktop.Core.Island;

/// <summary>
/// Typed result kinds rendered by the Lila Screen overlay and Lila's Work drawer.
/// Plain-text and typed-field presentation only; never renders untrusted HTML/markup.
/// </summary>
public enum ScreenItemKind
{
    Research,
    Image,
    Video,
    Task
}

/// <summary>
/// Five-stage Harness task lifecycle shown on Task cards.
/// Deliberately stops at CompletedNotVerified (4) until explicit verification lands (5).
/// </summary>
public enum ScreenTaskStage
{
    WaitingForApproval = 0,
    OwnerApproved = 1,
    AuthorisedHarness = 2,
    Dispatched = 3,
    CompletedNotVerified = 4,
    Verified = 5
}

/// <summary>
/// Typed, plain-text item handed to Lila Screen by Core, Harness, or a tool adapter.
/// Sensitivity (IsPrivate) must be supplied upstream; the UI never guesses privacy.
/// </summary>
public sealed record ScreenItem
{
    public string Id { get; init; } = Guid.NewGuid().ToString("N")[..12];
    public ScreenItemKind Kind { get; init; } = ScreenItemKind.Research;
    public string Title { get; init; } = string.Empty;
    public string Summary { get; init; } = string.Empty;
    public string SourceLabel { get; init; } = string.Empty;
    public IReadOnlyList<string> Sources { get; init; } = Array.Empty<string>();
    public string? MediaPath { get; init; }
    public string? DurationLabel { get; init; }
    public ScreenTaskStage TaskStage { get; init; } = ScreenTaskStage.WaitingForApproval;
    public string? EvidenceSummary { get; init; }
    public bool IsPrivate { get; init; }
    public DateTimeOffset CreatedUtc { get; init; } = DateTimeOffset.UtcNow;

    public string TaskStatusText => TaskStage switch
    {
        ScreenTaskStage.WaitingForApproval => "Waiting for your approval",
        ScreenTaskStage.OwnerApproved => "In progress · Owner approved",
        ScreenTaskStage.AuthorisedHarness => "In progress · Authorised (Harness)",
        ScreenTaskStage.Dispatched => "In progress · Dispatched",
        ScreenTaskStage.CompletedNotVerified => "Completed, not yet verified",
        ScreenTaskStage.Verified => "Completed and verified",
        _ => "In progress"
    };
}

/// <summary>
/// Authoritative caption line displayed above the bottom-center pill.
/// </summary>
public sealed record ScreenCaptionLine(
    string Speaker,
    string Text,
    bool IsFinal,
    DateTimeOffset UpdatedUtc);

/// <summary>
/// Pure, UI-agnostic state store for Lila Screen results, card cap (max 3),
/// and privacy toggles (Screen Sharing redaction & Captions on/off).
/// </summary>
public sealed class LilaScreenStore
{
    public const int MaxVisibleCards = 3;
    public const int MaxDrawerItems = 40;

    private readonly List<ScreenItem> _items = [];
    private readonly HashSet<string> _dismissedOverlayIds = [];

    public event Action? Changed;

    public bool ScreenSharingActive { get; private set; }
    public bool CaptionsEnabled { get; private set; } = true;

    public IReadOnlyList<ScreenItem> AllItems => _items;

    public IReadOnlyList<ScreenItem> VisibleOverlayCards =>
        _items.Where(item => !_dismissedOverlayIds.Contains(item.Id))
              .Take(MaxVisibleCards)
              .ToArray();

    public void SetScreenSharing(bool active)
    {
        if (ScreenSharingActive == active)
            return;
        ScreenSharingActive = active;
        Changed?.Invoke();
    }

    public void SetCaptionsEnabled(bool enabled)
    {
        if (CaptionsEnabled == enabled)
            return;
        CaptionsEnabled = enabled;
        Changed?.Invoke();
    }

    public ScreenItem Upsert(ScreenItem item)
    {
        ArgumentNullException.ThrowIfNull(item);
        var index = _items.FindIndex(existing => existing.Id == item.Id);
        if (index >= 0)
        {
            _items[index] = item;
        }
        else
        {
            _items.Insert(0, item);
            if (_items.Count > MaxDrawerItems)
                _items.RemoveRange(MaxDrawerItems, _items.Count - MaxDrawerItems);
        }
        _dismissedOverlayIds.Remove(item.Id);
        Changed?.Invoke();
        return item;
    }

    public bool DismissFromOverlay(string id)
    {
        if (string.IsNullOrWhiteSpace(id))
            return false;
        if (!_dismissedOverlayIds.Add(id))
            return false;
        Changed?.Invoke();
        return true;
    }

    public void ClearAll()
    {
        _items.Clear();
        _dismissedOverlayIds.Clear();
        Changed?.Invoke();
    }
}
