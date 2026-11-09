# Global shortcut guide

## Defaults

- **Right Ctrl** (`VK_RCONTROL`, `0xA3`): hold-to-talk. A 180 ms arming delay
  filters quick modifier use. A non-modifier key-down during that interval
  suppresses activation, preserving ordinary Ctrl shortcuts. Hold duration is
  tied to the matching physical device/key release; a watchdog bounds a missed
  release.
- **Pause / Fn+P** (`VK_PAUSE`, `0x13`): toggle-to-talk. First fresh key-down
  starts; key-up does nothing; the second fresh key-down submits. Repeats and
  very short duplicate pulses are ignored.
- **Escape**: cancel either active session; cancellation does not submit audio.

Bindings can be enabled and configured independently. The same primary key
cannot be assigned to both modes, even with different modifiers.

## Conservative supported set

The Settings dropdown allows Right Ctrl alone for hold mode and Pause alone
for toggle mode (recommended for ThinkPad Fn+P). Scroll Lock is deliberately
blocked because activating it changes the operating system's Scroll Lock
state.

All other keys and combinations are rejected, including ordinary letters and
digits, function keys commonly used by IDEs/debuggers (F9–F12), Escape,
Windows-key chords, Ctrl+C/V/S, Ctrl+Shift+Esc, Ctrl+Alt+Delete, and
modifier-only bindings other than Right Ctrl hold. The policy avoids common
typing/application shortcuts and Windows-reserved/security combinations. It
does not intercept keyboard input; a supported shortcut may still be seen by
the foreground application.

Modifier matching is exact. Extra held modifiers prevent activation. Raw Input
does not expose a uniform Fn key: laptop firmware may map Fn combinations to a
Pause key, another virtual key, or no event. Use the keyboard diagnostic to
inspect only supported shortcut events. If no stable event appears, that key
cannot be configured by this app. Do not advertise an unobserved Fn combination
as supported.
