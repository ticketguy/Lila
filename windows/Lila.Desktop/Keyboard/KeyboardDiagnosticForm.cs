using Lila.Desktop.Core.Keyboard;
using Lila.Desktop.Core.Voice;

namespace Lila.Desktop.Keyboard;

public sealed class KeyboardDiagnosticForm : Form
{
    private readonly ListView _events = new() { Dock = DockStyle.Fill, View = View.Details, FullRowSelect = true };
    private readonly RawKeyboardListener _listener;
    private readonly InteractionDiagnostics _diagnostics;
    private readonly Func<VoiceInteractionState> _currentState;

    public KeyboardDiagnosticForm(RawKeyboardListener listener, InteractionDiagnostics diagnostics,
        Func<VoiceInteractionState> currentState)
    {
        _listener = listener;
        _diagnostics = diagnostics;
        _currentState = currentState;
        Text = "Lila Keyboard Diagnostic";
        StartPosition = FormStartPosition.CenterParent;
        ClientSize = new Size(950, 360);
        _events.Columns.Add("Monotonic ms", 110);
        _events.Columns.Add("Virtual key", 90);
        _events.Columns.Add("Raw scan", 90);
        _events.Columns.Add("Make/break", 90);
        _events.Columns.Add("Repeat", 65);
        _events.Columns.Add("Key-down ms", 110);
        _events.Columns.Add("Hold ms", 95);
        _events.Columns.Add("Voice state", 105);
        var note = new Label
        {
            Dock = DockStyle.Top,
            Height = 42,
            Text = "Only configured shortcuts and Escape are recorded. Right Ctrl uses hold-to-talk; Pause (Fn + P) toggles."
        };
        var clear = new Button { Text = "Clear view", Dock = DockStyle.Bottom, Height = 32 };
        clear.Click += (_, _) => _events.Items.Clear();
        _listener.DiagnosticKeyEvent += OnDiagnosticKey;
        FormClosed += (_, _) => _listener.DiagnosticKeyEvent -= OnDiagnosticKey;
        Controls.Add(_events);
        Controls.Add(note);
        Controls.Add(clear);
    }

    private void OnDiagnosticKey(RawKeyEvent input)
    {
        if (IsDisposed || !IsHandleCreated)
            return;
        if (InvokeRequired)
        {
            BeginInvoke(new Action(() => OnDiagnosticKey(input)));
            return;
        }

        var state = _currentState();
        _diagnostics.RecordKey(input, state);
        var item = new ListViewItem(System.Diagnostics.Stopwatch.GetElapsedTime(0, input.Timestamp)
            .TotalMilliseconds.ToString("F3", System.Globalization.CultureInfo.InvariantCulture));
        item.SubItems.Add($"0x{input.VirtualKey:X2}");
        item.SubItems.Add($"0x{input.ScanCode:X2}");
        item.SubItems.Add(input.IsKeyUp ? "BREAK / UP" : "MAKE / DOWN");
        item.SubItems.Add(input.IsRepeat ? "yes" : "no");
        item.SubItems.Add(input.KeyDownTimestamp is long down
            ? System.Diagnostics.Stopwatch.GetElapsedTime(0, down).TotalMilliseconds.ToString("F3") : "n/a");
        item.SubItems.Add(input.HeldDuration?.TotalMilliseconds.ToString("F3") ?? "n/a");
        item.SubItems.Add(state.ToString());
        _events.Items.Add(item);
        if (_events.Items.Count > 500)
            _events.Items.RemoveAt(0);
        item.EnsureVisible();
    }
}
