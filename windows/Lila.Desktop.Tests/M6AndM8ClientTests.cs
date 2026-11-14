using System.Drawing;
using System.Net;
using System.Text;
using System.Text.Json;
using System.Windows.Forms;
using Lila.Desktop.Core.Api;
using Lila.Desktop.Core.Security;
using Lila.Desktop.Core.Sentinel;
using Lila.Desktop.Island;
using Xunit;

namespace Lila.Desktop.Tests;

public sealed class M6AndM8ClientTests
{
    private sealed class StubHandler(Func<HttpRequestMessage, HttpResponseMessage> responder) : HttpMessageHandler
    {
        protected override Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancellationToken)
            => Task.FromResult(responder(request));
    }

    private sealed class FixedCreds : ICredentialStore
    {
        public string? ReadToken() => new string('t', 40);
        public void WriteToken(string token) { }
        public void DeleteToken() { }
    }

    [Fact]
    public async Task EventStreamClient_AdvancesCursor_Deduplicates_AndSignalsGap()
    {
        var call = 0;
        using var http = new HttpClient(new StubHandler(_ =>
        {
            call++;
            var json = call == 1
                ? """{"schema_version":"1.0","since_cursor":0,"next_cursor":2,"gap_detected":false,"events":[{"schema_version":"1.0","event_id":1,"cursor":1,"event_type":"task.created","timestamp":1.0,"payload":{"task_id":"t1","description":"Hi","status":"queued"}},{"schema_version":"1.0","event_id":2,"cursor":2,"event_type":"task.updated","timestamp":2.0,"payload":{"task_id":"t1","status":"running"}}]}"""
                : """{"schema_version":"1.0","since_cursor":2,"next_cursor":4,"gap_detected":true,"events":[{"schema_version":"1.0","event_id":2,"cursor":2,"event_type":"task.updated","timestamp":2.0,"payload":{"task_id":"t1","status":"running"}},{"schema_version":"1.0","event_id":4,"cursor":4,"event_type":"task.completed","timestamp":4.0,"payload":{"task_id":"t1","status":"completed"}}]}""";
            return new HttpResponseMessage(HttpStatusCode.OK)
            {
                Content = new StringContent(json, Encoding.UTF8, "application/json")
            };
        }));

        var api = new LilaApiClient(http, new FixedCreds());
        var client = new DaemonEventStreamClient(api);
        var received = new List<long>();
        var gaps = 0;
        client.EventReceived += ev => received.Add(ev.EventId);
        client.GapDetected += () => gaps++;

        var first = await client.PollOnceAsync();
        Assert.Equal(2, first);
        Assert.Equal(2, client.LastCursor);
        Assert.Equal(0, gaps);

        // Event 2 replayed on second poll is deduplicated; gap_detected is surfaced.
        var second = await client.PollOnceAsync();
        Assert.Equal(1, second);
        Assert.Equal(4, client.LastCursor);
        Assert.Equal(1, gaps);
        Assert.Equal([1L, 2L, 4L], received);
    }

    [Fact]
    public async Task WindowsSentinelCollector_EnforcesOptInAndReadOnlyFacts()
    {
        var collector = new WindowsSentinelCollector((_, _, _) =>
            Task.FromResult((0, """{"realtime":true,"antivirus":true}""", "")));

        var notOpted = await collector.CollectAsync(optedIn: false);
        Assert.Empty(notOpted.Facts);
        Assert.Equal(2, notOpted.Gaps.Count);
        Assert.False(notOpted.AutonomousRemediationAttempted);
        Assert.False(notOpted.ScreenCaptureUsed);

        var optedCollector = new WindowsSentinelCollector((file, args, _) =>
        {
            if (args.Contains("Get-MpComputerStatus"))
                return Task.FromResult((0, """{"realtime":true,"antivirus":true}""", ""));
            return Task.FromResult((0, """[{"name":"Domain","enabled":true},{"name":"Private","enabled":false}]""", ""));
        });
        var snap = await optedCollector.CollectAsync(optedIn: true);
        Assert.Empty(snap.Gaps);
        Assert.Contains(snap.Facts, f => f.Metric == "defender_realtime_enabled" && f.ReadOnly && (bool)f.Value);
        Assert.Contains(snap.Facts, f => f.Metric == "firewall_enabled" && f.ReadOnly && !(bool)f.Value);
    }

    [Fact]
    public void DiscreetApprovalPopup_NonActivating_ApproveRejectDetailsAutoTuckAndRestore()
    {
        Exception? failure = null;
        var thread = new Thread(() =>
        {
            try
            {
                using var popup = new IslandApprovalToastPopup(autoTuckMs: 60);
                Assert.True(popup.TestShowWithoutActivation);
                Assert.NotEqual(0, popup.TestExStyle & 0x08000000); // WS_EX_NOACTIVATE
                Assert.NotEqual(0, popup.TestExStyle & 0x00000080); // WS_EX_TOOLWINDOW

                var req = new ApprovalRequestView(
                    InvocationId: "inv-toast-1",
                    Status: "pending",
                    CapabilityId: "file_write",
                    ContractVersion: "1.0",
                    OperationDigest: "digest-1",
                    ChallengeId: "chal-1",
                    VoiceChallengeCode: "123456",
                    KeyId: "key-1",
                    TargetKind: "file",
                    TargetResolverId: "path",
                    Target: JsonDocument.Parse("{\"path\":\"notes.txt\"}").RootElement,
                    Arguments: JsonDocument.Parse("{\"path\":\"notes.txt\"}").RootElement,
                    PolicyReason: "Requires approval",
                    PolicyDetail: "Write file",
                    Risk: "medium",
                    ExpiresInSeconds: 60,
                    ApprovalReady: true,
                    DisplayAction: "Write notes.txt",
                    DisplayValues: JsonDocument.Parse("{\"bytes\":42}").RootElement);

                var approved = 0;
                var rejected = 0;
                var spoke = 0;
                var autoTucked = 0;
                popup.ApproveClicked += () => approved++;
                popup.RejectClicked += () => rejected++;
                popup.SpeakApproveClicked += () => spoke++;
                popup.AutoTucked += () => autoTucked++;

                var screen = Screen.PrimaryScreen ?? Screen.AllScreens[0];
                popup.ShowOrUpdateToast(req, 55, 60, new Rectangle(100, 100, 64, 64), screen, spokenAvailable: true);
                Assert.True(popup.Visible);

                popup.PerformTestAction("approve");
                popup.PerformTestAction("reject");
                popup.PerformTestAction("speak");
                Assert.Equal(1, approved);
                Assert.Equal(1, rejected);
                Assert.Equal(1, spoke);

                // Expanding details pauses auto-tuck
                popup.PerformTestAction("details");
                Assert.True(popup.DetailsExpanded);
                var sw = System.Diagnostics.Stopwatch.StartNew();
                while (sw.ElapsedMilliseconds < 120)
                {
                    Application.DoEvents();
                    Thread.Sleep(10);
                }
                Assert.Equal(0, autoTucked);
                Assert.True(popup.Visible);

                // Collapsing details + forceShow re-arms auto-tuck -> hides into pending chip
                popup.PerformTestAction("details");
                Assert.False(popup.DetailsExpanded);
                popup.ShowOrUpdateToast(req, 50, 60, new Rectangle(100, 100, 64, 64), screen, spokenAvailable: true, forceShow: true);
                sw.Restart();
                while (autoTucked == 0 && sw.ElapsedMilliseconds < 1000)
                {
                    Application.DoEvents();
                    Thread.Sleep(15);
                }
                Assert.Equal(1, autoTucked);
                Assert.False(popup.Visible);
                Assert.True(popup.IsAutoTuckedFor("inv-toast-1"));

                // Non-forced update keeps it tucked; forceShow restores it
                popup.ShowOrUpdateToast(req, 45, 60, new Rectangle(100, 100, 64, 64), screen, spokenAvailable: true);
                Assert.False(popup.Visible);
                popup.ShowOrUpdateToast(req, 45, 60, new Rectangle(100, 100, 64, 64), screen, spokenAvailable: true, forceShow: true);
                Assert.True(popup.Visible);
                Assert.False(popup.IsAutoTuckedFor("inv-toast-1"));
            }
            catch (Exception ex)
            {
                failure = ex;
            }
        });
        thread.SetApartmentState(ApartmentState.STA);
        thread.Start();
        thread.Join();
        if (failure is not null)
            throw new Exception("STA WinForms test failed", failure);
    }

    [Fact]
    public async Task SseStream_ReconnectReplay_ApprovalNotification_AndCancelTask()
    {
        var seenSince = new List<string>();
        var cancelCalled = false;
        using var http = new HttpClient(new StubHandler(req =>
        {
            var path = req.RequestUri!.PathAndQuery;
            if (path.StartsWith("/events/stream", StringComparison.Ordinal))
            {
                seenSince.Add(path);
                var sse = seenSince.Count == 1
                    ? "id: 1\nevent: approval.requested\ndata: {\"schema_version\":\"1.0\",\"event_id\":1,\"cursor\":1,\"event_type\":\"approval.requested\",\"timestamp\":1.0,\"payload\":{\"invocation_id\":\"inv-1\"}}\n\n"
                    : "id: 2\nevent: task.cancelled\ndata: {\"schema_version\":\"1.0\",\"event_id\":2,\"cursor\":2,\"event_type\":\"task.cancelled\",\"timestamp\":2.0,\"payload\":{\"task_id\":\"t-1\",\"status\":\"cancelled\"}}\n\n";
                return new HttpResponseMessage(HttpStatusCode.OK)
                {
                    Content = new StringContent(sse, Encoding.UTF8, "text/event-stream")
                };
            }
            if (path == "/tasks/t-1/cancel" && req.Method == HttpMethod.Post)
            {
                cancelCalled = true;
                return new HttpResponseMessage(HttpStatusCode.OK)
                {
                    Content = new StringContent("""{"task_id":"t-1","status":"cancelled"}""", Encoding.UTF8, "application/json")
                };
            }
            return new HttpResponseMessage(HttpStatusCode.NotFound);
        }))
        {
            BaseAddress = new Uri("http://127.0.0.1:18790/")
        };

        var api = new LilaApiClient(http, new FixedCreds());
        var stream = new DaemonEventStreamClient(api);
        var events = new List<string>();
        stream.EventReceived += ev => events.Add(ev.EventType);

        await stream.ConsumeSseStreamOnceAsync();
        await stream.ConsumeSseStreamOnceAsync();

        Assert.Equal(["approval.requested", "task.cancelled"], events);
        Assert.Equal("/events/stream?since=0", seenSince[0]);
        Assert.Equal("/events/stream?since=1", seenSince[1]);

        var cancelled = await api.CancelTaskAsync("t-1", "owner_stopped");
        Assert.True(cancelCalled);
        Assert.Equal("cancelled", cancelled.Status);
    }

    [Fact]
    public async Task M7SyntheticCapture_GestureCaptureObserveDiscard_ThroughProductionClient()
    {
        var png1x1 = Convert.FromBase64String(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=");
        using var hl = new TransientScreenHighlight(png1x1, new Rectangle(10, 20, 1, 1), false, "Inspect", false);
        var seenSteps = new List<string>();

        using var http = new HttpClient(new StubHandler(req =>
        {
            var path = req.RequestUri!.AbsolutePath;
            seenSteps.Add(path);
            return path switch
            {
                "/perception/gesture" => new HttpResponseMessage(HttpStatusCode.OK)
                {
                    Content = new StringContent("""{"owner_gesture_token":"tok-synth-1","scope":"region"}""", Encoding.UTF8, "application/json")
                },
                "/perception/capture" => new HttpResponseMessage(HttpStatusCode.OK)
                {
                    Content = new StringContent("""{"capture_id":"cap-synth-1","scope":"region","image_digest":"abc","persisted":false,"stale":false,"ttl_remaining_s":299.0}""", Encoding.UTF8, "application/json")
                },
                "/perception/observe" => new HttpResponseMessage(HttpStatusCode.OK)
                {
                    Content = new StringContent("""{"capability":"vision.observe","capture":{"capture_id":"cap-synth-1"}}""", Encoding.UTF8, "application/json")
                },
                "/perception/discard" => new HttpResponseMessage(HttpStatusCode.OK)
                {
                    Content = new StringContent("""{"discarded":true}""", Encoding.UTF8, "application/json")
                },
                _ => new HttpResponseMessage(HttpStatusCode.NotFound)
            };
        }))
        {
            BaseAddress = new Uri("http://127.0.0.1:18790/")
        };

        var api = new LilaApiClient(http, new FixedCreds());
        var gest = await api.IssuePerceptionGestureAsync("region");
        var cap = await api.CapturePerceptionAsync(gest.OwnerGestureToken, "region", hl.ScopeSummary, hl.ToBase64Png(), 1, 1);
        Assert.False(cap.Persisted);
        var obs = await api.ObservePerceptionAsync(cap.CaptureId);
        Assert.Equal("vision.observe", obs.GetProperty("capability").GetString());
        await api.DiscardPerceptionAsync(cap.CaptureId);
        Assert.Equal(["/perception/gesture", "/perception/capture", "/perception/observe", "/perception/discard"], seenSteps);
    }

    [Fact]
    public void WorkDrawer_CapabilityCandidateReview_RendersAndFiresPromoteRollback()
    {
        Exception? failure = null;
        string? promotedId = null;
        int? rollbackVersion = null;

        var t = new Thread(() =>
        {
            try
            {
                var store = new Lila.Desktop.Core.Island.LilaScreenStore();
                using var drawer = new Lila.Desktop.Island.LilaWorkDrawerForm(store);
                drawer.Opacity = 0;
                drawer.PromoteCapabilityRequested += id => promotedId = id;
                drawer.RollbackCapabilityRequested += ver => rollbackVersion = ver;

                drawer.SetCapabilityCandidates(2, [
                    new Lila.Desktop.Island.CapabilityCandidateView(
                        "cap.alpha",
                        "0.2.0",
                        "staged",
                        true,
                        "owner-sandbox",
                        "4/4 candidate tests passed")
                ]);
                drawer.Show();
                System.Windows.Forms.Application.DoEvents();

                var promoteButtons = drawer.Controls.Find("PromoteBtn:cap.alpha", searchAllChildren: true);
                Assert.Single(promoteButtons);
                var promoteBtn = Assert.IsType<System.Windows.Forms.Button>(promoteButtons[0]);
                Assert.True(promoteBtn.Enabled);
                promoteBtn.PerformClick();

                var rollbackButtons = drawer.Controls.Find("RollbackBtn:cap.alpha", searchAllChildren: true);
                Assert.Single(rollbackButtons);
                var rollbackBtn = Assert.IsType<System.Windows.Forms.Button>(rollbackButtons[0]);
                Assert.True(rollbackBtn.Enabled);
                rollbackBtn.PerformClick();
            }
            catch (Exception ex)
            {
                failure = ex;
            }
        });
        t.SetApartmentState(ApartmentState.STA);
        t.Start();
        t.Join();

        Assert.Null(failure);
        Assert.Equal("cap.alpha", promotedId);
        Assert.Equal(1, rollbackVersion);
    }

    [Fact]
    public async Task OwnerSpokenEnrollment_StartAndStop_CallsDaemonEndpoints()
    {
        var seenPaths = new List<string>();
        using var http = new HttpClient(new StubHandler(req =>
        {
            var path = req.RequestUri!.AbsolutePath;
            seenPaths.Add(path);
            return path switch
            {
                "/owner/spoken/start" => new HttpResponseMessage(HttpStatusCode.OK)
                {
                    Content = new StringContent("""{"session_id":"spk-1","state":"listening"}""", Encoding.UTF8, "application/json")
                },
                "/owner/spoken/stop" => new HttpResponseMessage(HttpStatusCode.OK)
                {
                    Content = new StringContent("""{"enrolled":true,"available":true,"reason_code":"provider.enrolled"}""", Encoding.UTF8, "application/json")
                },
                _ => new HttpResponseMessage(HttpStatusCode.NotFound)
            };
        }))
        {
            BaseAddress = new Uri("http://127.0.0.1:18790/")
        };

        var api = new LilaApiClient(http, new FixedCreds());
        var start = await api.StartSpokenEnrollmentAsync();
        Assert.Equal("spk-1", start.SessionId);
        var stop = await api.StopSpokenEnrollmentAsync(start.SessionId);
        Assert.True(stop.GetProperty("enrolled").GetBoolean());
        Assert.True(stop.GetProperty("available").GetBoolean());
        Assert.Equal(["/owner/spoken/start", "/owner/spoken/stop"], seenPaths);
    }

    [Fact]
    public void WorkDrawer_PromoteAndRollback_RoundTrip_ThroughAuthenticatedApiAndPersistentCatalog()
    {
        var tempDir = Directory.CreateTempSubdirectory("lila-m10-e2e-");
        var catalogPath = Path.Combine(tempDir.FullName, "catalog.json");
        var catalogVersion = 1;
        var candidateStatus = "staged";
        void PersistCatalog()
        {
            var json = System.Text.Json.JsonSerializer.Serialize(new
            {
                catalog_version = catalogVersion,
                candidates = new[]
                {
                    new
                    {
                        capability_id = "cap.alpha",
                        version = "0.2.0",
                        status = candidateStatus,
                        tests_passed = true,
                        provenance = "owner-sandbox",
                        test_summary = "4/4 passed"
                    }
                }
            });
            File.WriteAllText(catalogPath, json);
        }
        PersistCatalog();

        var authedCalls = new List<string>();
        using var http = new HttpClient(new StubHandler(req =>
        {
            Assert.Equal("Bearer", req.Headers.Authorization?.Scheme);
            Assert.Equal(new string('t', 40), req.Headers.Authorization?.Parameter);
            var path = req.RequestUri!.AbsolutePath;
            authedCalls.Add($"{req.Method} {path}");

            if (req.Method == HttpMethod.Get && path == "/capabilities/candidates")
            {
                return new HttpResponseMessage(HttpStatusCode.OK)
                {
                    Content = new StringContent(File.ReadAllText(catalogPath), Encoding.UTF8, "application/json")
                };
            }
            if (req.Method == HttpMethod.Post && path == "/capabilities/candidates/cap.alpha/promote")
            {
                catalogVersion = 2;
                candidateStatus = "promoted";
                PersistCatalog();
                return new HttpResponseMessage(HttpStatusCode.OK)
                {
                    Content = new StringContent($$"""{"catalog_version":{{catalogVersion}},"status":"promoted"}""", Encoding.UTF8, "application/json")
                };
            }
            if (req.Method == HttpMethod.Post && path == "/capabilities/rollback")
            {
                catalogVersion = 1;
                candidateStatus = "staged";
                PersistCatalog();
                return new HttpResponseMessage(HttpStatusCode.OK)
                {
                    Content = new StringContent($$"""{"catalog_version":{{catalogVersion}},"rolled_back_to":1}""", Encoding.UTF8, "application/json")
                };
            }
            return new HttpResponseMessage(HttpStatusCode.NotFound);
        }))
        {
            BaseAddress = new Uri("http://127.0.0.1:18790/")
        };

        var api = new LilaApiClient(http, new FixedCreds());
        Exception? failure = null;
        try
        {
            var t = new Thread(() =>
            {
                try
                {
                    var store = new Lila.Desktop.Core.Island.LilaScreenStore();
                    using var drawer = new Lila.Desktop.Island.LilaWorkDrawerForm(store);
                    drawer.Opacity = 0;

                    void RefreshFromCatalog()
                    {
                        var res = api.GetCapabilityCandidatesAsync().GetAwaiter().GetResult();
                        var ver = res.CatalogVersion.GetInt32();
                        var list = res.Candidates.Select(c => new Lila.Desktop.Island.CapabilityCandidateView(
                            c.GetProperty("capability_id").GetString()!,
                            c.GetProperty("version").GetString()!,
                            c.GetProperty("status").GetString()!,
                            c.GetProperty("tests_passed").GetBoolean(),
                            c.GetProperty("provenance").GetString()!,
                            c.GetProperty("test_summary").GetString()!)).ToList();
                        drawer.SetCapabilityCandidates(ver, list);
                    }

                    drawer.PromoteCapabilityRequested += id =>
                    {
                        api.PromoteCapabilityCandidateAsync(id, ownerApproved: true).GetAwaiter().GetResult();
                        RefreshFromCatalog();
                    };
                    drawer.RollbackCapabilityRequested += targetVer =>
                    {
                        api.RollbackCapabilityCatalogAsync(targetVer).GetAwaiter().GetResult();
                        RefreshFromCatalog();
                    };

                    RefreshFromCatalog();
                    drawer.Show();
                    System.Windows.Forms.Application.DoEvents();

                    var promoteBtn = Assert.IsType<System.Windows.Forms.Button>(
                        Assert.Single(drawer.Controls.Find("PromoteBtn:cap.alpha", true)));
                    Assert.True(promoteBtn.Enabled);
                    promoteBtn.PerformClick();

                    // After promotion, persistent catalog has v2 + promoted and button is disabled.
                    Assert.Contains("\"catalog_version\":2", File.ReadAllText(catalogPath));
                    Assert.Contains("\"status\":\"promoted\"", File.ReadAllText(catalogPath));
                    var promoteBtnAfter = Assert.IsType<System.Windows.Forms.Button>(
                        Assert.Single(drawer.Controls.Find("PromoteBtn:cap.alpha", true)));
                    Assert.False(promoteBtnAfter.Enabled);

                    var rollbackBtn = Assert.IsType<System.Windows.Forms.Button>(
                        Assert.Single(drawer.Controls.Find("RollbackBtn:cap.alpha", true)));
                    rollbackBtn.PerformClick();

                    // After rollback, persistent catalog is restored to v1 + staged.
                    Assert.Contains("\"catalog_version\":1", File.ReadAllText(catalogPath));
                    Assert.Contains("\"status\":\"staged\"", File.ReadAllText(catalogPath));
                }
                catch (Exception ex)
                {
                    failure = ex;
                }
            });
            t.SetApartmentState(ApartmentState.STA);
            t.Start();
            t.Join();

            Assert.Null(failure);
            Assert.Equal([
                "GET /capabilities/candidates",
                "POST /capabilities/candidates/cap.alpha/promote",
                "GET /capabilities/candidates",
                "POST /capabilities/rollback",
                "GET /capabilities/candidates"
            ], authedCalls);
        }
        finally
        {
            tempDir.Delete(recursive: true);
        }
    }
}
