using System.Drawing;

namespace Lila.Desktop.Core.Island;

public enum IslandDockEdge
{
    Left,
    Right
}

public static class IslandPlacement
{
    public static Rectangle ClampToWorkArea(Rectangle bounds, Rectangle workArea)
    {
        if (workArea.Width <= 0 || workArea.Height <= 0)
            throw new ArgumentOutOfRangeException(nameof(workArea));

        var width = Math.Min(bounds.Width, workArea.Width);
        var height = Math.Min(bounds.Height, workArea.Height);
        var left = Math.Clamp(bounds.Left, workArea.Left, workArea.Right - width);
        var top = Math.Clamp(bounds.Top, workArea.Top, workArea.Bottom - height);
        return new Rectangle(left, top, width, height);
    }

    public static Rectangle SnapToEdges(Rectangle bounds, Rectangle workArea,
        int snapDistance = 24)
    {
        if (snapDistance < 0)
            throw new ArgumentOutOfRangeException(nameof(snapDistance));

        bounds = ClampToWorkArea(bounds, workArea);
        var left = Math.Abs(bounds.Left - workArea.Left) <= snapDistance
            ? workArea.Left
            : Math.Abs(workArea.Right - bounds.Right) <= snapDistance
                ? workArea.Right - bounds.Width
                : bounds.Left;
        return new Rectangle(left, bounds.Top, bounds.Width, bounds.Height);
    }

    public static Rectangle GetTuckedBounds(Rectangle previousBounds,
        Rectangle workArea, IslandDockEdge edge, Size indicatorSize)
    {
        if (indicatorSize.Width <= 0 || indicatorSize.Height <= 0)
            throw new ArgumentOutOfRangeException(nameof(indicatorSize));

        var width = Math.Min(indicatorSize.Width, workArea.Width);
        var height = Math.Min(indicatorSize.Height, workArea.Height);
        var left = edge == IslandDockEdge.Left ? workArea.Left : workArea.Right - width;
        var top = Math.Clamp(previousBounds.Top + (previousBounds.Height - height) / 2,
            workArea.Top, workArea.Bottom - height);
        return new Rectangle(left, top, width, height);
    }
}
