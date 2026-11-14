using Lila.Desktop.Core.Keyboard;
using Xunit;

namespace Lila.Desktop.Tests;

public sealed class ShortcutSafetyTests
{
    [Theory]
    [InlineData(0xA3, ShortcutModifiers.None, ShortcutActionMode.Hold)]
    [InlineData(0x13, ShortcutModifiers.None, ShortcutActionMode.Toggle)]
    public void AllowsConservativeSingleKeyBindings(ushort key, ShortcutModifiers modifiers,
        ShortcutActionMode mode) =>
        new ShortcutBinding(key, modifiers).Validate(mode);

    [Theory]
    [InlineData(0x41, ShortcutModifiers.None, ShortcutActionMode.Hold)] // A
    [InlineData(0x43, ShortcutModifiers.Control, ShortcutActionMode.Toggle)] // Ctrl+C
    [InlineData(0x7B, ShortcutModifiers.Control | ShortcutModifiers.Alt, ShortcutActionMode.Toggle)]
    [InlineData(0x13, ShortcutModifiers.None, ShortcutActionMode.Hold)] // Fn+P is a pulse
    [InlineData(0xA3, ShortcutModifiers.None, ShortcutActionMode.Toggle)]
    [InlineData(0x7B, (ShortcutModifiers)8, ShortcutActionMode.Hold)]
    [InlineData(0x91, ShortcutModifiers.None, ShortcutActionMode.Hold)]
    [InlineData(0x91, ShortcutModifiers.Control | ShortcutModifiers.Alt, ShortcutActionMode.Toggle)]
    public void RejectsTypingReservedOrModeIncompatibleBindings(ushort key,
        ShortcutModifiers modifiers, ShortcutActionMode mode) =>
        Assert.Throws<ArgumentException>(() => new ShortcutBinding(key, modifiers).Validate(mode));

    [Fact]
    public void IdenticalPrimaryKeysConflictEvenWhenModifiersDiffer() =>
        Assert.True(ShortcutBinding.Conflicts(
            new ShortcutBinding(0x7B, ShortcutModifiers.Alt),
            new ShortcutBinding(0x7B, ShortcutModifiers.Control | ShortcutModifiers.Alt)));

}
