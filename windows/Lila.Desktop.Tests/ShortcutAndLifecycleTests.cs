using Lila.Desktop.Core.Api;
using Lila.Desktop.Core.Keyboard;
using Lila.Desktop.Core.Lifecycle;
using Lila.Desktop.Core.Voice;
using Lila.Desktop.Core.Wsl;
using Xunit;

namespace Lila.Desktop.Tests;

public sealed class ShortcutAndLifecycleTests
{
    [Fact]
    public void StartupRegistrationIsPerExecutableAndExplicitlyStartsMinimized()
    {
        var command = StartupRegistrationCommand.GetValue(true,
            @"D:\Lila Portable\Lila.Desktop.exe");
        Assert.Equal("\"D:\\Lila Portable\\Lila.Desktop.exe\" --startup", command);
        Assert.Null(StartupRegistrationCommand.GetValue(false,
            @"D:\Lila Portable\Lila.Desktop.exe"));
    }

    [Fact]
    public void NamedMutexAllowsOnlyOneInstanceForSameUserAcrossProcesses()
    {
        using var first = SingleInstanceGuard.Acquire("S-1-5-21-test-user");
        using var second = SingleInstanceGuard.Acquire("S-1-5-21-test-user");
        Assert.True(first.IsFirstInstance);
        Assert.False(second.IsFirstInstance);
    }

    [Fact]
    public async Task RightControlArmsThenPhysicalReleaseStopsOnce()
    {
        var api = new FakeVoiceApi();
        var voice = new VoiceSessionController(api, responseDisplayDuration: TimeSpan.Zero);
        using var gate = NewGate(voice, armingDelay: TimeSpan.FromMilliseconds(35));
        var keys = Interpreter();
        gate.Handle(keys.Interpret(0xA3, 0x1D, 0x02, false, (nint)1)!);
        await WaitForAsync(() => voice.Snapshot.State == VoiceInteractionState.Recording);
        gate.Handle(keys.Interpret(0xA3, 0x1D, 0x03, true, (nint)1)!);
        await WaitForAsync(() => voice.Snapshot.State == VoiceInteractionState.Idle);
        Assert.Equal(1, api.StartCount);
        Assert.Equal(1, api.StopCount);
    }

    [Fact]
    public async Task ShortcutsRemainUnavailableUntilAuthenticatedReadyState()
    {
        var api = new FakeVoiceApi();
        var voice = new VoiceSessionController(api);
        using var gate = new GlobalVoiceShortcutController(voice,
            armingDelay: TimeSpan.FromMilliseconds(20), toggleDebounce: TimeSpan.Zero);
        var keys = Interpreter();
        gate.Handle(keys.Interpret(0x13, 0x45, 0x04, false, (nint)2)!);
        await Task.Delay(40);
        Assert.Equal(0, api.StartCount);

        gate.SetReady(true);
        gate.Handle(keys.Interpret(0x13, 0x45, 0x05, true, (nint)2)!);
        gate.Handle(keys.Interpret(0x13, 0x45, 0x04, false, (nint)2)!);
        await WaitForAsync(() => voice.Snapshot.State == VoiceInteractionState.Recording);
        Assert.Equal(1, api.StartCount);
        gate.SetReady(false);
        await WaitForAsync(() => voice.Snapshot.State == VoiceInteractionState.Idle);
        Assert.Equal(1, api.CancelCount);
    }

    [Fact]
    public async Task RightControlQuickTapAndCtrlShortcutDoNotStartVoice()
    {
        var api = new FakeVoiceApi();
        var voice = new VoiceSessionController(api);
        using var gate = NewGate(voice, armingDelay: TimeSpan.FromMilliseconds(60));
        var keys = Interpreter();

        gate.Handle(keys.Interpret(0xA3, 0x1D, 0x02, false, (nint)1)!);
        gate.Handle(keys.Interpret(0xA3, 0x1D, 0x03, true, (nint)1)!);
        await Task.Delay(90);
        Assert.Equal(0, api.StartCount);

        gate.Handle(keys.Interpret(0xA3, 0x1D, 0x02, false, (nint)1)!);
        gate.NotifyNonModifierKeyDown(0x43); // Ctrl+C: only the key category is used.
        await Task.Delay(90);
        gate.Handle(keys.Interpret(0xA3, 0x1D, 0x03, true, (nint)1)!);
        Assert.Equal(0, api.StartCount);
    }

    [Fact]
    public async Task MutingCancelsActiveCaptureAndBlocksBothVoiceShortcuts()
    {
        var api = new FakeVoiceApi();
        var voice = new VoiceSessionController(api);
        using var gate = NewGate(voice, armingDelay: TimeSpan.FromMilliseconds(20),
            toggleDebounce: TimeSpan.Zero);
        var keys = Interpreter();

        gate.SetMuted(true);
        gate.Handle(keys.Interpret(0x13, 0x45, 0x04, false, (nint)2)!);
        gate.Handle(keys.Interpret(0x13, 0x45, 0x05, true, (nint)2)!);
        gate.Handle(keys.Interpret(0xA3, 0x1D, 0x02, false, (nint)1)!);
        await Task.Delay(50);
        gate.Handle(keys.Interpret(0xA3, 0x1D, 0x03, true, (nint)1)!);
        Assert.Equal(0, api.StartCount);

        gate.SetMuted(false);
        gate.Handle(keys.Interpret(0x13, 0x45, 0x04, false, (nint)2)!);
        await WaitForAsync(() => voice.Snapshot.State == VoiceInteractionState.Recording);
        gate.SetMuted(true);
        await WaitForAsync(() => voice.Snapshot.State == VoiceInteractionState.Idle);

        Assert.Equal(1, api.StartCount);
        Assert.Equal(1, api.CancelCount);
    }

    [Fact]
    public async Task PauseToggleIgnoresKeyUpAndStopsOnlyOnSecondPress()
    {
        var api = new FakeVoiceApi();
        var voice = new VoiceSessionController(api, responseDisplayDuration: TimeSpan.Zero);
        using var gate = NewGate(voice, toggleDebounce: TimeSpan.Zero);
        var keys = Interpreter();

        gate.Handle(keys.Interpret(0x13, 0x45, 0x04, false, (nint)2)!);
        gate.Handle(keys.Interpret(0x13, 0x45, 0x05, true, (nint)2)!);
        await WaitForAsync(() => voice.Snapshot.State == VoiceInteractionState.Recording);
        Assert.Equal(0, api.StopCount);

        gate.Handle(keys.Interpret(0x13, 0x45, 0x04, false, (nint)2)!);
        gate.Handle(keys.Interpret(0x13, 0x45, 0x05, true, (nint)2)!);
        await WaitForAsync(() => voice.Snapshot.State == VoiceInteractionState.Idle);
        Assert.Equal(1, api.StartCount);
        Assert.Equal(1, api.StopCount);
    }

    [Fact]
    public async Task EscapeCancelsRightControlWithoutSubmitting()
    {
        var api = new FakeVoiceApi();
        var voice = new VoiceSessionController(api, responseDisplayDuration: TimeSpan.Zero);
        using var gate = NewGate(voice, armingDelay: TimeSpan.FromMilliseconds(20));
        var keys = Interpreter();
        gate.Handle(keys.Interpret(0xA3, 0x1D, 0x02, false, (nint)1)!);
        await WaitForAsync(() => voice.Snapshot.State == VoiceInteractionState.Recording);
        gate.Handle(keys.Interpret(KeyEventInterpreter.EscapeVirtualKey, 0x01, 0, false, (nint)1)!);
        await WaitForAsync(() => voice.Snapshot.State == VoiceInteractionState.Idle);
        Assert.Equal(1, api.CancelCount);
        Assert.Equal(0, api.StopCount);
    }

    [Fact]
    public async Task RepeatAndOtherShortcutDoNotDuplicateOrEndToggleSession()
    {
        var api = new FakeVoiceApi();
        var voice = new VoiceSessionController(api, responseDisplayDuration: TimeSpan.Zero);
        using var gate = NewGate(voice, toggleDebounce: TimeSpan.Zero);
        var keys = Interpreter();
        gate.Handle(keys.Interpret(0x13, 0x45, 0x04, false, (nint)2)!);
        gate.Handle(keys.Interpret(0x13, 0x45, 0x04, false, (nint)2)!); // repeat
        gate.Handle(keys.Interpret(0x13, 0x45, 0x05, true, (nint)2)!);
        await WaitForAsync(() => voice.Snapshot.State == VoiceInteractionState.Recording);
        gate.Handle(keys.Interpret(0xA3, 0x1D, 0x02, false, (nint)1)!);
        gate.Handle(keys.Interpret(0xA3, 0x1D, 0x03, true, (nint)1)!);
        Assert.Equal(VoiceInteractionState.Recording, voice.Snapshot.State);
        Assert.Equal(1, api.StartCount);
        Assert.Equal(0, api.StopCount);
    }

    [Fact]
    public async Task TogglePressDuringPendingStartStopsThatSessionExactlyOnce()
    {
        var api = new FakeVoiceApi
        {
            StartGate = new TaskCompletionSource<VoiceStartResult>(TaskCreationOptions.RunContinuationsAsynchronously)
        };
        var voice = new VoiceSessionController(api, responseDisplayDuration: TimeSpan.Zero);
        using var gate = NewGate(voice, toggleDebounce: TimeSpan.Zero);
        var keys = Interpreter();
        gate.Handle(keys.Interpret(0x13, 0x45, 0x04, false, (nint)2)!);
        gate.Handle(keys.Interpret(0x13, 0x45, 0x05, true, (nint)2)!);
        gate.Handle(keys.Interpret(0x13, 0x45, 0x04, false, (nint)2)!);
        gate.Handle(keys.Interpret(0x13, 0x45, 0x05, true, (nint)2)!);
        api.StartGate.SetResult(new VoiceStartResult("recording", "race-session", 0, null));
        await WaitForAsync(() => voice.Snapshot.State == VoiceInteractionState.Idle);
        Assert.Equal(1, api.StartCount);
        Assert.Equal(1, api.StopCount);
    }

    [Fact]
    public async Task EscapeCancelsToggleAndDeviceRemovalReleasesHold()
    {
        var api = new FakeVoiceApi();
        var voice = new VoiceSessionController(api, responseDisplayDuration: TimeSpan.Zero);
        using var gate = NewGate(voice, armingDelay: TimeSpan.FromMilliseconds(20), toggleDebounce: TimeSpan.Zero);
        var keys = Interpreter();
        gate.Handle(keys.Interpret(0x13, 0x45, 0x04, false, (nint)2)!);
        await WaitForAsync(() => voice.Snapshot.State == VoiceInteractionState.Recording);
        gate.Handle(keys.Interpret(KeyEventInterpreter.EscapeVirtualKey, 0x01, 0, false, (nint)2)!);
        await WaitForAsync(() => voice.Snapshot.State == VoiceInteractionState.Idle);
        Assert.Equal(1, api.CancelCount);
        Assert.Equal(0, api.StopCount);

        gate.Handle(keys.Interpret(0xA3, 0x1D, 0x02, false, (nint)1)!);
        await WaitForAsync(() => voice.Snapshot.State == VoiceInteractionState.Recording);
        gate.NotifyDeviceRemoved((nint)1);
        await WaitForAsync(() => voice.Snapshot.State == VoiceInteractionState.Idle);
        Assert.Equal(1, api.StopCount);
    }

    [Fact]
    public async Task RightControlSubmitsOnceAndNeverCancelsWhenRecordingIsCachedRemotely()
    {
        var api = new FakeVoiceApi();
        var voice = new VoiceSessionController(api, responseDisplayDuration: TimeSpan.Zero);
        using var gate = NewGate(voice, armingDelay: TimeSpan.FromMilliseconds(25));
        var keys = Interpreter();

        // Physical-style RawInput key-down on Right Ctrl (0xA3, scan 0x1D, E0 flag 0x02)
        gate.Handle(keys.Interpret(0xA3, 0x1D, 0x02, false, (nint)1)!);
        await WaitForAsync(() => voice.Snapshot.State == VoiceInteractionState.Recording);

        // Simulate RefreshVoiceStatusAsync + ApplyAvailability polling while recording is cached remotely
        var cachedRemote = new VoiceSessionStatus("recording", "test-session", 1.2, null);
        gate.ApplyRuntimeState(serviceReady: true, cachedRemote);
        gate.ApplyRuntimeState(serviceReady: true, cachedRemote);
        Assert.Equal(VoiceInteractionState.Recording, voice.Snapshot.State);
        Assert.Equal(0, api.CancelCount);

        // Physical-style RawInput key-up on Right Ctrl submits once with zero cancels
        gate.Handle(keys.Interpret(0xA3, 0x1D, 0x03, true, (nint)1)!);
        await WaitForAsync(() => voice.Snapshot.State == VoiceInteractionState.Idle);
        Assert.Equal(1, api.StartCount);
        Assert.Equal(1, api.StopCount);
        Assert.Equal(0, api.CancelCount);
    }

    [Fact]
    public async Task FnPauseToggleSubmitsOnceAndNeverCancelsWhenRecordingIsCachedRemotely()
    {
        var api = new FakeVoiceApi();
        var voice = new VoiceSessionController(api, responseDisplayDuration: TimeSpan.Zero);
        using var gate = NewGate(voice, toggleDebounce: TimeSpan.Zero);
        var keys = Interpreter();

        // First Fn+P / Pause pulse (down + immediate up) starts ToggleToTalk recording
        gate.Handle(keys.Interpret(0x13, 0x45, 0x04, false, (nint)2)!);
        gate.Handle(keys.Interpret(0x13, 0x45, 0x05, true, (nint)2)!);
        await WaitForAsync(() => voice.Snapshot.State == VoiceInteractionState.Recording);

        // Simulate RefreshVoiceStatusAsync + ApplyAvailability polling while recording is cached remotely
        var cachedRemote = new VoiceSessionStatus("recording", "test-session", 2.4, null);
        gate.ApplyRuntimeState(serviceReady: true, cachedRemote);
        gate.ApplyRuntimeState(serviceReady: true, cachedRemote);
        Assert.Equal(VoiceInteractionState.Recording, voice.Snapshot.State);
        Assert.Equal(0, api.CancelCount);

        // Second Fn+P / Pause pulse stops and submits the recording once with zero cancels
        gate.Handle(keys.Interpret(0x13, 0x45, 0x04, false, (nint)2)!);
        gate.Handle(keys.Interpret(0x13, 0x45, 0x05, true, (nint)2)!);
        await WaitForAsync(() => voice.Snapshot.State == VoiceInteractionState.Idle);
        Assert.Equal(1, api.StartCount);
        Assert.Equal(1, api.StopCount);
        Assert.Equal(0, api.CancelCount);
    }

    [Fact]
    public async Task UnownedRemoteRecordingBlocksNewStartWithoutCallingCancel()
    {
        var api = new FakeVoiceApi();
        var voice = new VoiceSessionController(api, responseDisplayDuration: TimeSpan.Zero);
        using var gate = NewGate(voice, armingDelay: TimeSpan.FromMilliseconds(20), toggleDebounce: TimeSpan.Zero);
        var keys = Interpreter();

        // An external session is active remotely while local controller is Idle
        gate.ApplyRuntimeState(serviceReady: true, new VoiceSessionStatus("recording", "foreign-session", 0.5, null));
        gate.Handle(keys.Interpret(0x13, 0x45, 0x04, false, (nint)2)!);
        gate.Handle(keys.Interpret(0x13, 0x45, 0x05, true, (nint)2)!);
        gate.Handle(keys.Interpret(0xA3, 0x1D, 0x02, false, (nint)1)!);
        await Task.Delay(40);
        gate.Handle(keys.Interpret(0xA3, 0x1D, 0x03, true, (nint)1)!);
        Assert.Equal(0, api.StartCount);
        Assert.Equal(0, api.CancelCount);

        // Once remote status returns to idle, shortcuts start and submit normally
        gate.ApplyRuntimeState(serviceReady: true, new VoiceSessionStatus("idle", null, null, null));
        gate.Handle(keys.Interpret(0x13, 0x45, 0x04, false, (nint)2)!);
        gate.Handle(keys.Interpret(0x13, 0x45, 0x05, true, (nint)2)!);
        await WaitForAsync(() => voice.Snapshot.State == VoiceInteractionState.Recording);
        gate.Handle(keys.Interpret(0x13, 0x45, 0x04, false, (nint)2)!);
        gate.Handle(keys.Interpret(0x13, 0x45, 0x05, true, (nint)2)!);
        await WaitForAsync(() => voice.Snapshot.State == VoiceInteractionState.Idle);
        Assert.Equal(1, api.StartCount);
        Assert.Equal(1, api.StopCount);
        Assert.Equal(0, api.CancelCount);
    }

    [Fact]
    public async Task ManagedWslHelperRestartsAfterUnexpectedExitAndStopsCleanly()
    {
        var commands = new FakeCommandRunner(
            new CommandResult(0, "", ""), new CommandResult(0, "", ""),
            new CommandResult(0, "Ubuntu-24.04\n", ""));
        var keepAlive = new FakeKeepAlive();
        await using var supervisor = new WslLifecycleSupervisor(commands, keepAlive);

        await supervisor.EnsureRunningAsync();
        Assert.True(supervisor.KeepAliveRunning);
        Assert.Equal(1, keepAlive.StartCount);
        keepAlive.IsRunning = false;
        await supervisor.EnsureRunningAsync();
        Assert.Equal(2, keepAlive.StartCount);
        await supervisor.StopAsync();
        Assert.False(keepAlive.IsRunning);
        Assert.Equal(1, keepAlive.StopCount);
    }

    private static GlobalVoiceShortcutController NewGate(VoiceSessionController voice,
        TimeSpan? armingDelay = null, TimeSpan? toggleDebounce = null)
    {
        var gate = new GlobalVoiceShortcutController(voice,
            armingDelay: armingDelay ?? TimeSpan.FromMilliseconds(180),
            toggleDebounce: toggleDebounce ?? TimeSpan.FromMilliseconds(220));
        gate.SetReady(true);
        return gate;
    }

    private static KeyEventInterpreter Interpreter()
    {
        var keys = new KeyEventInterpreter(KeyEventInterpreter.RightControlVirtualKey);
        keys.SetShortcuts([KeyEventInterpreter.RightControlVirtualKey, KeyEventInterpreter.PauseVirtualKey]);
        return keys;
    }

    private static async Task WaitForAsync(Func<bool> condition)
    {
        var deadline = DateTime.UtcNow.AddSeconds(3);
        while (!condition() && DateTime.UtcNow < deadline)
            await Task.Delay(10);
        Assert.True(condition(), "Timed out waiting for the expected voice state.");
    }

    private sealed class FakeVoiceApi : IVoiceSessionApi
    {
        public TaskCompletionSource<VoiceStartResult>? StartGate { get; init; }
        public int StartCount { get; private set; }
        public int StopCount { get; private set; }
        public int CancelCount { get; private set; }

        public Task<VoiceStartResult> StartVoiceAsync(CancellationToken cancellationToken = default)
        {
            StartCount++;
            return StartGate?.Task ?? Task.FromResult(new VoiceStartResult("recording", "test-session", 0, null));
        }

        public Task<VoiceStopResult> StopVoiceAsync(string sessionId, CancellationToken cancellationToken = default)
        {
            StopCount++;
            return Task.FromResult(new VoiceStopResult(sessionId, "transcript", "response",
                new VoiceSessionStatus("completed", sessionId, null, null)));
        }

        public Task<VoiceCancelResult> CancelVoiceAsync(string sessionId, CancellationToken cancellationToken = default)
        {
            CancelCount++;
            return Task.FromResult(new VoiceCancelResult("cancelled", sessionId, null));
        }

        public Task<VoiceSessionStatus> GetVoiceStatusAsync(CancellationToken cancellationToken = default) =>
            Task.FromResult(new VoiceSessionStatus("idle", null, null, null));
    }

    private sealed class FakeKeepAlive : IWslKeepAlive
    {
        public bool IsRunning { get; set; }
        public int StartCount { get; private set; }
        public int StopCount { get; private set; }
        public Task StartAsync(string distribution, CancellationToken cancellationToken = default)
        {
            StartCount++;
            IsRunning = true;
            return Task.CompletedTask;
        }
        public Task StopAsync(CancellationToken cancellationToken = default)
        {
            if (IsRunning)
                StopCount++;
            IsRunning = false;
            return Task.CompletedTask;
        }
        public ValueTask DisposeAsync() => ValueTask.CompletedTask;
    }

    private sealed class FakeCommandRunner(params CommandResult[] results) : ICommandRunner
    {
        private readonly Queue<CommandResult> _results = new(results);
        public Task<CommandResult> RunAsync(string executable, IReadOnlyList<string> arguments,
            TimeSpan timeout, CancellationToken cancellationToken = default)
        {
            Assert.Equal("wsl.exe", executable);
            Assert.NotEmpty(arguments);
            return Task.FromResult(_results.Dequeue());
        }
    }
}
