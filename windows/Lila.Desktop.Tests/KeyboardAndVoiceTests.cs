using Lila.Desktop.Core.Api;
using Lila.Desktop.Core.Keyboard;
using Lila.Desktop.Core.Voice;
using Xunit;

namespace Lila.Desktop.Tests;

public sealed class KeyboardAndVoiceTests
{
    [Fact]
    public void InterpreterReportsPauseDownRepeatAndUpOnly()
    {
        var interpreter = new KeyEventInterpreter(KeyEventInterpreter.PauseVirtualKey);
        Assert.Null(interpreter.Interpret(0x41, 0x1E, 0, false)); // ordinary A is discarded
        var down = interpreter.Interpret(0x13, 0x45, 0x04, false)!;
        var repeat = interpreter.Interpret(0x13, 0x45, 0x04, false)!;
        var up = interpreter.Interpret(0x13, 0x45, 0x05, true)!;
        Assert.Equal(ShortcutEventKind.ShortcutDown, down.Kind);
        Assert.Equal(ShortcutEventKind.ShortcutRepeat, repeat.Kind);
        Assert.True(repeat.RawEvent.IsRepeat);
        Assert.Equal(ShortcutEventKind.ShortcutUp, up.Kind);
        Assert.Equal((ushort)0x45, up.RawEvent.ScanCode);
        Assert.True(down.RawEvent.IsE1);
    }

    [Theory]
    [InlineData(1)]
    [InlineData(3)]
    [InlineData(5)]
    [InlineData(10)]
    public void InterpreterMeasuresPhysicalHoldWithMonotonicTimestamps(int seconds)
    {
        var interpreter = new KeyEventInterpreter(KeyEventInterpreter.PauseVirtualKey);
        var start = System.Diagnostics.Stopwatch.GetTimestamp();
        interpreter.Interpret(0x13, 0x1D, 0, false, (nint)7, start);
        var up = interpreter.Interpret(0x13, 0x1D, 1, true, (nint)7,
            start + seconds * System.Diagnostics.Stopwatch.Frequency)!;
        Assert.Equal(TimeSpan.FromSeconds(seconds), up.RawEvent.HeldDuration!.Value);
    }

    [Fact]
    public void EscapeIsInterpretedOnlyAsCancellationKey()
    {
        var interpreter = new KeyEventInterpreter(0x13);
        var escape = interpreter.Interpret(KeyEventInterpreter.EscapeVirtualKey, 0x01, 0, false)!;
        Assert.Equal(ShortcutEventKind.EscapeDown, escape.Kind);
        Assert.Null(interpreter.Interpret(0x41, 0x1E, 0, false));
    }

    [Fact]
    public async Task FastPressReleaseWaitsForStartThenStopsSameSession()
    {
        var api = new FakeVoiceApi { StartGate = new TaskCompletionSource<VoiceStartResult>(
            TaskCreationOptions.RunContinuationsAsynchronously) };
        var controller = new VoiceSessionController(api, responseDisplayDuration: TimeSpan.Zero);
        var states = Observe(controller);

        controller.Handle(Key(ShortcutEventKind.ShortcutDown));
        controller.Handle(Key(ShortcutEventKind.ShortcutUp));
        Assert.Equal(VoiceInteractionState.Starting, controller.Snapshot.State);
        Assert.Equal(0, api.StopCount);

        api.StartGate.SetResult(StartResult);
        await WaitForAsync(() => controller.Snapshot.State == VoiceInteractionState.Idle);

        Assert.Equal(new[] { "start", "stop:session-1" }, api.Calls);
        Assert.Contains(VoiceInteractionState.Starting, states);
        Assert.Contains(VoiceInteractionState.Processing, states);
        Assert.Contains(VoiceInteractionState.Response, states);
    }

    [Fact]
    public async Task RepeatKeyDownDoesNotStartAnotherSession()
    {
        var api = new FakeVoiceApi();
        var interpreter = new KeyEventInterpreter(0x13);
        var controller = new VoiceSessionController(api, responseDisplayDuration: TimeSpan.Zero);
        controller.Handle(interpreter.Interpret(0x13, 0x45, 0, false)!);
        controller.Handle(interpreter.Interpret(0x13, 0x45, 0, false)!);
        await WaitForAsync(() => controller.Snapshot.State == VoiceInteractionState.Recording);
        Assert.Equal(1, api.StartCount);
        controller.Handle(interpreter.Interpret(0x13, 0x45, 1, true)!);
        await WaitForAsync(() => controller.Snapshot.State == VoiceInteractionState.Idle);
    }

    [Fact]
    public async Task NewPressWhileRecordingDoesNotOverlapSessions()
    {
        var api = new FakeVoiceApi();
        var controller = new VoiceSessionController(api);
        controller.Handle(Key(ShortcutEventKind.ShortcutDown));
        await WaitForAsync(() => controller.Snapshot.State == VoiceInteractionState.Recording);
        controller.Handle(Key(ShortcutEventKind.ShortcutUp));
        await WaitForAsync(() => controller.Snapshot.State == VoiceInteractionState.Idle);
        controller.Handle(Key(ShortcutEventKind.ShortcutDown));
        await WaitForAsync(() => controller.Snapshot.State == VoiceInteractionState.Recording);
        Assert.Equal(2, api.StartCount);
        controller.Handle(Key(ShortcutEventKind.EscapeDown));
        await WaitForAsync(() => controller.Snapshot.State == VoiceInteractionState.Idle);
    }

    [Fact]
    public async Task EscapeDuringStartingCancelsAsSoonAsSessionIdArrives()
    {
        var api = new FakeVoiceApi { StartGate = new TaskCompletionSource<VoiceStartResult>(
            TaskCreationOptions.RunContinuationsAsynchronously) };
        var controller = new VoiceSessionController(api);
        controller.Handle(Key(ShortcutEventKind.ShortcutDown));
        controller.Handle(Key(ShortcutEventKind.EscapeDown));
        Assert.Equal(VoiceInteractionState.Cancelling, controller.Snapshot.State);

        api.StartGate.SetResult(StartResult);
        await WaitForAsync(() => controller.Snapshot.State == VoiceInteractionState.Idle);
        Assert.Equal(1, api.CancelCount);
        Assert.Equal(0, api.StopCount);
    }

    [Fact]
    public async Task EscapeDuringRecordingCancelsWithoutStoppingOrSubmitting()
    {
        var api = new FakeVoiceApi();
        var controller = new VoiceSessionController(api);
        controller.Handle(Key(ShortcutEventKind.ShortcutDown));
        await WaitForAsync(() => controller.Snapshot.State == VoiceInteractionState.Recording);
        controller.Handle(Key(ShortcutEventKind.EscapeDown));
        await WaitForAsync(() => controller.Snapshot.State == VoiceInteractionState.Idle);

        Assert.Equal(1, api.CancelCount);
        Assert.Equal(0, api.StopCount);
    }

    [Fact]
    public async Task ToggleModeStopsOnSecondPressNotRelease()
    {
        var api = new FakeVoiceApi();
        var controller = new VoiceSessionController(api, responseDisplayDuration: TimeSpan.Zero)
        {
            Mode = PushToTalkMode.ToggleToTalk
        };
        controller.Handle(Key(ShortcutEventKind.ShortcutDown));
        await WaitForAsync(() => controller.Snapshot.State == VoiceInteractionState.Recording);
        controller.Handle(Key(ShortcutEventKind.ShortcutUp));
        Assert.Equal(0, api.StopCount);
        controller.Handle(Key(ShortcutEventKind.ShortcutDown));
        await WaitForAsync(() => controller.Snapshot.State == VoiceInteractionState.Idle);
        Assert.Equal(1, api.StopCount);
    }

    [Fact]
    public async Task ToggleModeKeyUpDuringPendingStartDoesNotStopSession()
    {
        var api = new FakeVoiceApi { StartGate = new TaskCompletionSource<VoiceStartResult>(
            TaskCreationOptions.RunContinuationsAsynchronously) };
        var controller = new VoiceSessionController(api)
        {
            Mode = PushToTalkMode.ToggleToTalk
        };
        var interpreter = new KeyEventInterpreter(0x13);
        controller.Handle(interpreter.Interpret(0x13, 0x1D, 0, false)!);
        controller.Handle(interpreter.Interpret(0x13, 0x1D, 1, true)!);
        api.StartGate.SetResult(StartResult);
        await WaitForAsync(() => controller.Snapshot.State == VoiceInteractionState.Recording);
        Assert.Equal(0, api.StopCount);
        controller.CancelActive();
        await WaitForAsync(() => controller.Snapshot.State == VoiceInteractionState.Idle);
    }

    [Fact]
    public async Task KeyUpFromDifferentRawDeviceCannotStopActiveHold()
    {
        var api = new FakeVoiceApi();
        var controller = new VoiceSessionController(api);
        controller.Handle(new InterpretedKeyEvent(ShortcutEventKind.ShortcutDown,
            new RawKeyEvent(0x13, 0x1D, 0, false, false, DeviceHandle: (nint)1)));
        await WaitForAsync(() => controller.Snapshot.State == VoiceInteractionState.Recording);
        controller.Handle(new InterpretedKeyEvent(ShortcutEventKind.ShortcutUp,
            new RawKeyEvent(0x13, 0x1D, 1, true, false, DeviceHandle: (nint)2)));
        Assert.Equal(VoiceInteractionState.Recording, controller.Snapshot.State);
        Assert.Equal(0, api.StopCount);
        controller.Handle(new InterpretedKeyEvent(ShortcutEventKind.ShortcutUp,
            new RawKeyEvent(0x13, 0x1D, 1, true, false, DeviceHandle: (nint)1)));
        await WaitForAsync(() => controller.Snapshot.State == VoiceInteractionState.Idle);
        Assert.Equal(1, api.StopCount);
    }

    [Fact]
    public async Task UiToggleStartsThenStopsOnSecondClick()
    {
        var api = new FakeVoiceApi();
        var controller = new VoiceSessionController(api);
        controller.ToggleFromUi();
        await WaitForAsync(() => controller.Snapshot.State == VoiceInteractionState.Recording);
        controller.ToggleFromUi();
        await WaitForAsync(() => controller.Snapshot.State == VoiceInteractionState.Idle);
        Assert.Equal(1, api.StartCount);
        Assert.Equal(1, api.StopCount);
    }

    [Fact]
    public async Task ServerVoiceErrorIsShownWithoutAnUncertainOutcomeMessage()
    {
        var api = new FakeVoiceApi
        {
            StopError = new LilaApiException(LilaApiFailureKind.Service,
                "Recording is empty or has an invalid audio format.",
                System.Net.HttpStatusCode.BadGateway)
        };
        var controller = new VoiceSessionController(api);
        controller.Handle(Key(ShortcutEventKind.ShortcutDown));
        await WaitForAsync(() => controller.Snapshot.State == VoiceInteractionState.Recording);
        controller.Handle(Key(ShortcutEventKind.ShortcutUp));
        await WaitForAsync(() => controller.Snapshot.State == VoiceInteractionState.Idle &&
                                 controller.Snapshot.Error is not null);

        Assert.Contains("Recording is empty", controller.Snapshot.Error!);
        Assert.DoesNotContain("uncertain", controller.Snapshot.Error!);
    }

    [Fact]
    public async Task MissedKeyUpTriggersBoundedAutomaticStop()
    {
        var api = new FakeVoiceApi();
        var controller = new VoiceSessionController(api,
            missedReleaseSafetyLimit: TimeSpan.FromMilliseconds(50),
            responseDisplayDuration: TimeSpan.Zero);
        controller.Handle(Key(ShortcutEventKind.ShortcutDown));
        await WaitForAsync(() => controller.Snapshot.State == VoiceInteractionState.Idle);
        Assert.Equal(1, api.StopCount);
    }

    private static List<VoiceInteractionState> Observe(VoiceSessionController controller)
    {
        var states = new List<VoiceInteractionState>();
        controller.StateChanged += snapshot =>
        {
            lock (states) states.Add(snapshot.State);
        };
        return states;
    }

    private static InterpretedKeyEvent Key(ShortcutEventKind kind) =>
        new(kind, new RawKeyEvent(0x13, 0x45, 0, kind is ShortcutEventKind.ShortcutUp, false));

    private static async Task WaitForAsync(Func<bool> condition)
    {
        var limit = DateTime.UtcNow.AddSeconds(3);
        while (!condition() && DateTime.UtcNow < limit)
            await Task.Delay(10);
        Assert.True(condition(), "Timed out waiting for the expected state.");
    }

    private static readonly VoiceStartResult StartResult = new("recording", "session-1", 0, null);

    private sealed class FakeVoiceApi : IVoiceSessionApi
    {
        private readonly object _lock = new();
        public TaskCompletionSource<VoiceStartResult>? StartGate { get; init; }
        public Exception? StopError { get; init; }
        public List<string> Calls { get; } = [];
        public int StartCount { get; private set; }
        public int StopCount { get; private set; }
        public int CancelCount { get; private set; }

        public Task<VoiceStartResult> StartVoiceAsync(CancellationToken cancellationToken = default)
        {
            lock (_lock)
            {
                StartCount++;
                Calls.Add("start");
            }
            return StartGate?.Task ?? Task.FromResult(StartResult);
        }

        public Task<VoiceStopResult> StopVoiceAsync(string sessionId,
            CancellationToken cancellationToken = default)
        {
            lock (_lock)
            {
                StopCount++;
                Calls.Add("stop:" + sessionId);
            }
            if (StopError is not null)
                return Task.FromException<VoiceStopResult>(StopError);
            return Task.FromResult(new VoiceStopResult(sessionId, "raw transcript", "response",
                new VoiceSessionStatus("completed", sessionId, null, null)));
        }

        public Task<VoiceCancelResult> CancelVoiceAsync(string sessionId,
            CancellationToken cancellationToken = default)
        {
            lock (_lock)
            {
                CancelCount++;
                Calls.Add("cancel:" + sessionId);
            }
            return Task.FromResult(new VoiceCancelResult("cancelled", sessionId, null));
        }

        public Task<VoiceSessionStatus> GetVoiceStatusAsync(CancellationToken cancellationToken = default) =>
            Task.FromResult(new VoiceSessionStatus("idle", null, null, null));
    }
}
