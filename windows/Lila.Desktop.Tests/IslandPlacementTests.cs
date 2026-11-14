using System.Drawing;
using Lila.Desktop.Core.Island;
using Xunit;

namespace Lila.Desktop.Tests;

public sealed class IslandPlacementTests
{
    private static readonly Rectangle LeftMonitor = new(0, 0, 1920, 1080);
    private static readonly Rectangle RightMonitor = new(1920, 0, 2560, 1440);

    [Theory]
    [InlineData(6, 90, 0)]
    [InlineData(1798, 90, 1820)]
    [InlineData(1700, 90, 1700)]
    public void DraggingNearAnEdgeSnapsWithinTheTargetMonitor(int x, int y, int expectedX)
    {
        var placed = IslandPlacement.SnapToEdges(new Rectangle(x, y, 100, 80),
            LeftMonitor, snapDistance: 24);

        Assert.Equal(expectedX, placed.X);
        Assert.Equal(y, placed.Y);
        Assert.True(LeftMonitor.Contains(placed));
    }

    [Fact]
    public void RestoreClampsAnOldPositionFullyIntoTheSelectedMonitor()
    {
        var placed = IslandPlacement.ClampToWorkArea(
            new Rectangle(-900, 1700, 500, 700), RightMonitor);

        Assert.Equal(new Rectangle(1920, 740, 500, 700), placed);
        Assert.True(RightMonitor.Contains(placed));
    }

    [Theory]
    [InlineData(IslandDockEdge.Left, 1920)]
    [InlineData(IslandDockEdge.Right, 4458)]
    public void TuckedIndicatorRemainsAccessibleAtEitherMonitorEdge(
        IslandDockEdge edge, int expectedX)
    {
        var tab = IslandPlacement.GetTuckedBounds(new Rectangle(2100, 600, 380, 560),
            RightMonitor, edge, new Size(22, 56));

        Assert.Equal(expectedX, tab.X);
        Assert.Equal(22, tab.Width);
        Assert.Equal(56, tab.Height);
        Assert.True(RightMonitor.Contains(tab));
    }
}
