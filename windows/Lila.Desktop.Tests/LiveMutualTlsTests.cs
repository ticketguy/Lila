using System.Net;
using System.Text.Json;
using Lila.Desktop.Core.Api;
using Lila.Desktop.Core.Security;
using Lila.Desktop.Core.Voice;
using Lila.Desktop.Keyboard;
using Xunit;

namespace Lila.Desktop.Tests;

public sealed class RequiresLiveMutualTlsFactAttribute : FactAttribute
{
    public RequiresLiveMutualTlsFactAttribute()
    {
        if (Environment.GetEnvironmentVariable("LILA_RUN_LIVE_MTLS_TEST") != "1")
            Skip = "Requires the explicitly provisioned, short-lived Windows/WSL mTLS fixture.";
    }
}

public sealed class LiveMutualTlsTests
{
    [RequiresLiveMutualTlsFact]
    public async Task ProductionApiReloadsAuthoritativeLifecycleAndKeepsDiagnosticsPrivate()
    {
        var token = Required("LILA_LIVE_MTLS_API_TOKEN");
        var phrase = Required("LILA_LIVE_MTLS_OWNER_PHRASE");
        var clientPin = Required("LILA_LIVE_MTLS_ALLOWED_CLIENT_PIN");
        var serverPin = Required("LILA_LIVE_MTLS_SERVER_PIN");
        var port = int.Parse(Required("LILA_LIVE_MTLS_PORT"));
        string invocationId;
        var requestDiagnostics = new List<ApiRequestDiagnostic>();
        var errors = new List<string>();
        var diagnostics = new InteractionDiagnostics();
        var diagnosticPath = diagnostics.DiagnosticPath;
        void Capture(ApiRequestDiagnostic item)
        {
            requestDiagnostics.Add(item);
            diagnostics.RecordRequest(item, VoiceInteractionState.Idle);
        }
        try
        {
            using (var firstConnection = CreateProductionApiHttp(clientPin, serverPin, port))
            {
                var first = new LilaApiClient(firstConnection, new FixtureCredential(token));
                first.RequestCompleted += Capture;
                Assert.True((await first.GetHealthAsync()).Ready);
                var pending = Assert.Single((await first.GetPendingApprovalsAsync()).Requests);
                Assert.True(pending.ApprovalReady);
                invocationId = pending.InvocationId;
                var bearerOnly = await Assert.ThrowsAsync<LilaApiException>(
                    () => first.SubmitApprovalDecisionAsync(pending, "approve"));
                Assert.Equal(HttpStatusCode.Forbidden, bearerOnly.StatusCode);
                errors.Add(bearerOnly.ToString());
                Assert.Equal("pending", (await first.GetApprovalStatusAsync(invocationId)).Status);
            }
            // Dispose the transport and recreate the same client used by Island.
            using (var reconnectedHttp = CreateProductionApiHttp(clientPin, serverPin, port))
            {
                var reconnected = new LilaApiClient(reconnectedHttp, new FixtureCredential(token));
                reconnected.RequestCompleted += Capture;
                var pending = Assert.Single((await reconnected.GetPendingApprovalsAsync()).Requests);
                Assert.Equal(invocationId, pending.InvocationId);
                var rejected = await Assert.ThrowsAsync<LilaApiException>(() =>
                    reconnected.SubmitOwnerDecisionAsync(pending, "approve", phrase + " incorrect"));
                Assert.Equal(HttpStatusCode.Forbidden, rejected.StatusCode);
                errors.Add(rejected.ToString());
                Assert.Equal("pending", (await reconnected.GetApprovalStatusAsync(invocationId)).Status);
                Assert.Equal("approved", (await reconnected.SubmitOwnerDecisionAsync(pending, "approve", phrase)).Status);
                var delayed = await Assert.ThrowsAsync<LilaApiException>(() =>
                    reconnected.SubmitOwnerDecisionAsync(pending, "approve", phrase));
                errors.Add(delayed.ToString());
            }
            using var reloadedHttp = CreateProductionApiHttp(clientPin, serverPin, port);
            var reloaded = new LilaApiClient(reloadedHttp, new FixtureCredential(token));
            reloaded.RequestCompleted += Capture;
            Assert.Empty((await reloaded.GetPendingApprovalsAsync()).Requests);
            Assert.Equal("approved", (await reloaded.GetApprovalStatusAsync(invocationId)).Status);
            await FixtureCommand("create_pending");
            var stale = Assert.Single((await reloaded.GetPendingApprovalsAsync()).Requests);
            await FixtureCommand("stop_daemon");
            var offline = await Assert.ThrowsAsync<LilaApiException>(() => reloaded.GetPendingApprovalsAsync());
            Assert.Equal(LilaApiFailureKind.Connection, offline.Kind);
            errors.Add(offline.ToString());
            await FixtureCommand("restart_daemon");
            Assert.Empty((await reloaded.GetPendingApprovalsAsync()).Requests);
            var missing = await Assert.ThrowsAsync<LilaApiException>(() => reloaded.GetApprovalStatusAsync(stale.InvocationId));
            Assert.Equal(HttpStatusCode.NotFound, missing.StatusCode);
            errors.Add(missing.ToString());
            var staleDecision = await Assert.ThrowsAsync<LilaApiException>(() =>
                reloaded.SubmitOwnerDecisionAsync(stale, "approve", phrase));
            errors.Add(staleDecision.ToString());
            diagnostics.Dispose();
            var channels = JsonSerializer.Serialize(requestDiagnostics) + string.Join("\n", errors)
                           + await File.ReadAllTextAsync(diagnosticPath);
            foreach (var secret in new[] { token, phrase, Required("LILA_LIVE_MTLS_SECRET_SENTINEL") })
                Assert.False(channels.Contains(secret, StringComparison.Ordinal), "Sensitive fixture value reached Windows diagnostics.");
            Assert.False(channels.Contains("PRIVATE KEY", StringComparison.Ordinal), "Private key material reached Windows diagnostics.");
            await FixtureCommand("assert_evidence");
        }
        finally
        {
            if (File.Exists(diagnosticPath))
            {
                // Dispose before deleting a test-owned diagnostic file.
                try { diagnostics.Dispose(); } catch (ObjectDisposedException) { }
                File.Delete(diagnosticPath);
            }
        }
    }

    private static async Task FixtureCommand(string action)
    {
        var directory = Required("LILA_LIVE_MTLS_FIXTURE_DIRECTORY");
        var commandId = Guid.NewGuid().ToString("N");
        await File.WriteAllTextAsync(Path.Combine(directory, commandId + ".command"), action);
        var response = Path.Combine(directory, commandId + ".result");
        using var timeout = new CancellationTokenSource(TimeSpan.FromSeconds(15));
        while (!File.Exists(response))
            await Task.Delay(100, timeout.Token);
        Assert.Equal("ok", await File.ReadAllTextAsync(response));
    }

    private static HttpClient CreateProductionApiHttp(string clientPin, string serverPin, int port) =>
        new(new FixturePortHandler(port) { InnerHandler = CreateHandler(clientPin, serverPin) })
        { Timeout = Timeout.InfiniteTimeSpan };

    // Route to the test listener only; API payloads and production TLS validation stay intact.
    private sealed class FixturePortHandler(int port) : DelegatingHandler
    {
        protected override Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancellationToken)
        {
            request.RequestUri = new UriBuilder(request.RequestUri!) { Port = port }.Uri;
            return base.SendAsync(request, cancellationToken);
        }
    }

    private sealed class FixtureCredential(string token) : ICredentialStore
    {
        public string ReadToken() => token;
        public void WriteToken(string value) => throw new NotSupportedException();
        public void DeleteToken() => throw new NotSupportedException();
    }

    [RequiresLiveMutualTlsFact]
    public async Task ProductionCertificateSelectorConnectsToDaemonAndRejectsInvalidIdentities()
    {
        var token = Required("LILA_LIVE_MTLS_API_TOKEN");
        var serverPin = Required("LILA_LIVE_MTLS_SERVER_PIN");
        var port = int.Parse(Required("LILA_LIVE_MTLS_PORT"));

        using (var allowed = CreateHttpClient(Required("LILA_LIVE_MTLS_ALLOWED_CLIENT_PIN"), serverPin))
        {
            using var response = await GetHealth(allowed, port, token);
            Assert.Equal(HttpStatusCode.OK, response.StatusCode);
            var body = await response.Content.ReadAsStringAsync();
            using var document = JsonDocument.Parse(body);
            Assert.Equal("running", document.RootElement.GetProperty("status").GetString());
        }

        using (var unlisted = CreateHttpClient(Required("LILA_LIVE_MTLS_UNLISTED_CLIENT_PIN"), serverPin))
        {
            using var response = await GetHealth(unlisted, port, token);
            Assert.Equal(HttpStatusCode.Unauthorized, response.StatusCode);
        }

        using (var badBearer = CreateHttpClient(
                   Required("LILA_LIVE_MTLS_ALLOWED_CLIENT_PIN"), serverPin))
        {
            using var response = await GetHealth(badBearer, port, "wrong-test-bearer");
            Assert.Equal(HttpStatusCode.Unauthorized, response.StatusCode);
        }

        using (var missingBearer = CreateHttpClient(
                   Required("LILA_LIVE_MTLS_ALLOWED_CLIENT_PIN"), serverPin))
        {
            using var response = await GetHealth(missingBearer, port, null);
            Assert.Equal(HttpStatusCode.Unauthorized, response.StatusCode);
        }

        await AssertConnectionFailure(
            Required("LILA_LIVE_MTLS_UNTRUSTED_CLIENT_PIN"), serverPin, port, token);
        await AssertConnectionFailure(
            Required("LILA_LIVE_MTLS_ALLOWED_CLIENT_PIN"),
            Required("LILA_LIVE_MTLS_WRONG_SERVER_PIN"), port, token);

        Assert.Throws<InvalidOperationException>(() => CreateHandler(
            Required("LILA_LIVE_MTLS_EXPIRED_CLIENT_PIN"), serverPin));
        Assert.Throws<InvalidOperationException>(() => CreateHandler(
            new string('0', 64), serverPin));

        using var noCertificateHandler = new HttpClientHandler
        {
            ServerCertificateCustomValidationCallback = (_, certificate, _, _) =>
                certificate is not null &&
                LilaMutualTlsHandler.IsPinnedServerCertificate(certificate, serverPin)
        };
        using var noCertificateHttp = new HttpClient(noCertificateHandler)
        {
            Timeout = TimeSpan.FromSeconds(8)
        };
        await Assert.ThrowsAnyAsync<HttpRequestException>(
            () => GetHealth(noCertificateHttp, port, token));
    }

    private static async Task AssertConnectionFailure(
        string clientPin, string serverPin, int port, string token)
    {
        using var client = CreateHttpClient(clientPin, serverPin);
        await Assert.ThrowsAnyAsync<HttpRequestException>(() => GetHealth(client, port, token));
    }

    private static HttpClient CreateHttpClient(string clientPin, string serverPin) =>
        new(CreateHandler(clientPin, serverPin)) { Timeout = TimeSpan.FromSeconds(8) };

    private static async Task<HttpResponseMessage> GetHealth(HttpClient client, int port, string? token)
    {
        using var request = new HttpRequestMessage(HttpMethod.Get, $"https://127.0.0.1:{port}/health");
        if (token is not null)
            request.Headers.Authorization =
                new System.Net.Http.Headers.AuthenticationHeaderValue("Bearer", token);
        return await client.SendAsync(request);
    }

    private static HttpClientHandler CreateHandler(string clientPin, string serverPin) =>
        LilaMutualTlsHandler.CreateFromEnvironment(name => name switch
        {
            "LILA_TLS_CLIENT_CERT_SHA256" => clientPin,
            "LILA_TLS_SERVER_CERT_SHA256" => serverPin,
            _ => Environment.GetEnvironmentVariable(name)
        });

    private static string Required(string name) =>
        Environment.GetEnvironmentVariable(name)
        ?? throw new InvalidOperationException("Live mTLS test configuration is incomplete.");
}
