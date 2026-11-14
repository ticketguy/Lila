using Lila.Desktop.Core.Island;
using Xunit;

namespace Lila.Desktop.Tests;

public sealed class LilaScreenStoreTests
{
    [Fact]
    public void CapsVisibleOverlayCardsAtThreeWhileKeepingOlderItemsInDrawer()
    {
        var store = new LilaScreenStore();
        for (var i = 1; i <= 5; i++)
        {
            store.Upsert(new ScreenItem
            {
                Id = $"item-{i}",
                Kind = ScreenItemKind.Research,
                Title = $"Research {i}",
                Summary = "Plain text summary"
            });
        }

        Assert.Equal(5, store.AllItems.Count);
        Assert.Equal(3, store.VisibleOverlayCards.Count);
        Assert.Equal(new[] { "item-5", "item-4", "item-3" },
            store.VisibleOverlayCards.Select(c => c.Id).ToArray());
    }

    [Fact]
    public void DismissingOverlayCardKeepsItemInDrawerAndPromotesNextCard()
    {
        var store = new LilaScreenStore();
        for (var i = 1; i <= 4; i++)
        {
            store.Upsert(new ScreenItem
            {
                Id = $"item-{i}",
                Kind = ScreenItemKind.Task,
                Title = $"Task {i}",
                TaskStage = ScreenTaskStage.CompletedNotVerified
            });
        }

        Assert.True(store.DismissFromOverlay("item-4"));
        Assert.Equal(4, store.AllItems.Count);
        Assert.Equal(new[] { "item-3", "item-2", "item-1" },
            store.VisibleOverlayCards.Select(c => c.Id).ToArray());
    }

    [Fact]
    public void TaskStatusStopsAtCompletedNotVerifiedBeforeVerified()
    {
        var unverified = new ScreenItem
        {
            Kind = ScreenItemKind.Task,
            Title = "Run check",
            TaskStage = ScreenTaskStage.CompletedNotVerified
        };
        var verified = unverified with { TaskStage = ScreenTaskStage.Verified };

        Assert.Equal("Completed, not yet verified", unverified.TaskStatusText);
        Assert.Equal("Completed and verified", verified.TaskStatusText);
    }

    [Fact]
    public void PrivacyAndCaptionsTogglesUpdateState()
    {
        var store = new LilaScreenStore();
        Assert.False(store.ScreenSharingActive);
        Assert.True(store.CaptionsEnabled);

        store.SetScreenSharing(true);
        store.SetCaptionsEnabled(false);

        Assert.True(store.ScreenSharingActive);
        Assert.False(store.CaptionsEnabled);
    }
}
