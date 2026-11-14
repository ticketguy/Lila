using System.Net;
using System.Net.Http.Json;
using System.Text;
using System.Text.Json;
using System.Security.Cryptography;
using System.Security.Cryptography.X509Certificates;
using Lila.Desktop.Core.Api;
using Lila.Desktop.Core.Security;
using Lila.Desktop.Core.Wsl;
using Xunit;

namespace Lila.Desktop.Tests;

public sealed class ApiAndWslTests
{
    [Fact]
    public void PinnedDaemonIdentityAcceptsOnlyCurrentCertificateWithExactSha256Pin()
    {
        using var rsa = RSA.Create(2048);
        var request = new CertificateRequest("CN=Lila test daemon", rsa,
            HashAlgorithmName.SHA256, RSASignaturePadding.Pkcs1);
        using var certificate = request.CreateSelfSigned(
            DateTimeOffset.UtcNow.AddMinutes(-1), DateTimeOffset.UtcNow.AddMinutes(5));
        var pin = Convert.ToHexString(SHA256.HashData(certificate.RawData));

        Assert.True(LilaMutualTlsHandler.IsPinnedServerCertificate(certificate, pin));
        Assert.False(LilaMutualTlsHandler.IsPinnedServerCertificate(certificate, new string('0', 64)));
        Assert.False(LilaMutualTlsHandler.IsPinnedServerCertificate(null, pin));

        using var expired = request.CreateSelfSigned(
            DateTimeOffset.UtcNow.AddMinutes(-10), DateTimeOffset.UtcNow.AddMinutes(-5));
        var expiredPin = Convert.ToHexString(SHA256.HashData(expired.RawData));
        Assert.False(LilaMutualTlsHandler.IsPinnedServerCertificate(expired, expiredPin));
    }

    [Fact]
    public void JsonContentHasUnknownLengthAndWouldBeChunkedByHttp11()
    {
        using var content = JsonContent.Create(new { text = "hey" });
        Assert.Null(content.Headers.ContentLength);
    }

    [Fact]
    public async Task MissingCredentialFailsBeforeNetworkRequest()
    {
        var handler = new StubHandler((_, _) => throw new Exception("network must not be used"));
        using var http = NewHttpClient(handler);
        var api = new LilaApiClient(http, new FakeCredentialStore(null));

        var error = await Assert.ThrowsAsync<LilaApiException>(() => api.GetHealthAsync());
        Assert.Equal(LilaApiFailureKind.Authentication, error.Kind);
        Assert.Equal(0, handler.RequestCount);
    }

    [Fact]
    public async Task BearerTokenIsSentOnlyInAuthorizationHeader()
    {
        const string secret = "test-only-token-not-for-logs";
        HttpRequestMessage? captured = null;
        var handler = new StubHandler((request, _) =>
        {
            captured = request;
            return Task.FromResult(JsonResponse("{\"status\":\"running\",\"ready\":true,\"voice_enabled\":true}"));
        });
        using var http = NewHttpClient(handler);
        var api = new LilaApiClient(http, new FakeCredentialStore(secret));

        var health = await api.GetHealthAsync();

        Assert.Equal("Bearer", captured!.Headers.Authorization!.Scheme);
        Assert.Equal(secret, captured.Headers.Authorization.Parameter);
        Assert.Equal(new Uri("https://127.0.0.1:8765/health"), captured.RequestUri);
        Assert.DoesNotContain(secret, captured.RequestUri!.ToString());
        Assert.True(health.VoiceEnabled);
    }

    [Fact]
    public async Task TextRequestUsesBoundedContentLengthJsonExpectedByTheDaemon()
    {
        long? contentLength = null;
        string? transferEncoding = null;
        string? contentType = null;
        string? body = null;
        string? method = null;
        string? path = null;
        var handler = new StubHandler(async (request, token) =>
        {
            method = request.Method.Method;
            path = request.RequestUri!.AbsolutePath;
            contentLength = request.Content!.Headers.ContentLength;
            transferEncoding = request.Headers.TransferEncoding.FirstOrDefault()?.Value;
            contentType = request.Content.Headers.ContentType?.MediaType;
            body = await request.Content.ReadAsStringAsync(token);
            return JsonResponse("{\"response\":\"ok\"}");
        });
        using var http = NewHttpClient(handler);
        var api = new LilaApiClient(http, new FakeCredentialStore("test-token"));

        await api.SendTextAsync("hey");

        Assert.Equal("POST", method);
        Assert.Equal("/chat", path);
        Assert.Equal("application/json", contentType);
        Assert.Null(transferEncoding);
        Assert.Equal("{\"text\":\"hey\"}", body);
        Assert.Equal(Encoding.UTF8.GetByteCount(body!), contentLength);
    }

    [Fact]
    public async Task IslandLoadsPendingApprovalFromAuthoritativeDaemonState()
    {
        var handler = new StubHandler((_, _) => Task.FromResult(JsonResponse(
            "{\"requests\":[{\"invocation_id\":\"inv-1\",\"status\":\"pending\",\"capability_id\":\"files.write\",\"contract_version\":\"1\",\"operation_digest\":\"" + new string('a', 64) + "\",\"challenge_id\":\"ch-1\",\"key_id\":\"owner-key\",\"target\":{\"path\":\"/tmp/out.txt\"},\"arguments\":{\"path\":\"/tmp/out.txt\",\"token\":\"[REDACTED]\"},\"policy_reason\":\"policy.approval_required\",\"policy_detail\":\"Review this operation\",\"risk\":\"high\",\"expires_in_seconds\":12.5}]}")));
        using var http = NewHttpClient(handler);
        var api = new LilaApiClient(http, new FakeCredentialStore("daemon-bearer"));

        var result = await api.GetPendingApprovalsAsync();

        Assert.Single(result.Requests);
        Assert.Equal("inv-1", result.Requests[0].InvocationId);
        Assert.Equal(12.5, result.Requests[0].ExpiresInSeconds);
        Assert.Equal("/approvals", handler.LastUri!.AbsolutePath);
    }

    [Fact]
    public async Task IslandDecisionSubmitsExactIdsWithoutOwnerProofAndDoesNotAssumeApproval()
    {
        string? body = null;
        var handler = new StubHandler(async (request, token) =>
        {
            body = await request.Content!.ReadAsStringAsync(token);
            return new HttpResponseMessage(HttpStatusCode.Forbidden)
            {
                Content = new StringContent("{\"error\":\"A separate owner decision proof is required.\",\"code\":\"approval.owner_authentication_required\"}", Encoding.UTF8, "application/json")
            };
        });
        using var http = NewHttpClient(handler);
        var api = new LilaApiClient(http, new FakeCredentialStore("daemon-bearer"));
        var view = new ApprovalRequestView("inv-2", "pending", "files.write", "1",
            new string('b', 64), "challenge-2", "123456", "owner-key", "file", "test.path", null,
            JsonDocument.Parse("{\"path\":\"/tmp/out.txt\"}").RootElement,
            "policy.approval_required", "Review", "high", 10);

        var error = await Assert.ThrowsAsync<LilaApiException>(() => api.SubmitApprovalDecisionAsync(view, "approve"));

        Assert.Equal(LilaApiFailureKind.Service, error.Kind);
        Assert.Equal(new Uri("https://127.0.0.1:8765/approvals/inv-2/decision"), handler.LastUri);
        using var payload = JsonDocument.Parse(body!);
        Assert.Equal("challenge-2", payload.RootElement.GetProperty("challenge_id").GetString());
        Assert.Equal(new string('b', 64), payload.RootElement.GetProperty("operation_digest").GetString());
        Assert.Equal("approve", payload.RootElement.GetProperty("decision").GetString());
        Assert.False(payload.RootElement.TryGetProperty("proof", out _));
    }

    [Fact]
    public async Task TypedOwnerDecisionUsesDedicatedOiaRouteAndDoesNotManufactureProof()
    {
        string? body = null;
        var handler = new StubHandler(async (request, token) =>
        {
            body = await request.Content!.ReadAsStringAsync(token);
            return new HttpResponseMessage(HttpStatusCode.OK)
            {
                Content = new StringContent("{\"invocation_id\":\"inv-2\",\"status\":\"approved\",\"reason_code\":\"approval.approved\"}", Encoding.UTF8, "application/json")
            };
        });
        using var http = NewHttpClient(handler);
        var api = new LilaApiClient(http, new FakeCredentialStore("daemon-bearer"));
        var view = new ApprovalRequestView("inv-2", "pending", "files.write", "1",
            new string('b', 64), "challenge-2", "123456", "owner-key", "file", "test.path", null,
            JsonDocument.Parse("{\"path\":\"/tmp/out.txt\"}").RootElement,
            "policy.approval_required", "Review", "high", 10);

        await api.SubmitOwnerDecisionAsync(view, "approve", "private phrase");

        Assert.Equal(new Uri("https://127.0.0.1:8765/owner/approvals/inv-2/decision"), handler.LastUri);
        using var payload = JsonDocument.Parse(body!);
        Assert.Equal("private phrase", payload.RootElement.GetProperty("phrase").GetString());
        Assert.Equal("approve", payload.RootElement.GetProperty("decision").GetString());
        Assert.False(payload.RootElement.TryGetProperty("proof", out _));
    }

    [Fact]
    public async Task VoiceStartSendsExactlyAnEmptyJsonObjectWithKnownLength()
    {
        string? body = null;
        long? contentLength = null;
        string? route = null;
        var handler = new StubHandler(async (request, token) =>
        {
            route = request.RequestUri!.AbsolutePath;
            body = await request.Content!.ReadAsStringAsync(token);
            contentLength = request.Content.Headers.ContentLength;
            return JsonResponse("{\"state\":\"recording\",\"session_id\":\"s1\"}");
        });
        using var http = NewHttpClient(handler);
        var api = new LilaApiClient(http, new FakeCredentialStore("test-token"));

        var started = await api.StartVoiceAsync();

        Assert.Equal("/voice/start", route);
        Assert.Equal("{}", body);
        Assert.Equal(2, contentLength);
        Assert.Equal("s1", started.SessionId);
    }

    [Fact]
    public async Task SnakeCaseVoiceStatusFieldsMapToTheWindowsModel()
    {
        var handler = new StubHandler((_, _) => Task.FromResult(JsonResponse(
            "{\"state\":\"recording\",\"session_id\":\"session-1\",\"elapsed_seconds\":1.25,\"error\":null}")));
        using var http = NewHttpClient(handler);
        var api = new LilaApiClient(http, new FakeCredentialStore("test-token"));

        var status = await api.GetVoiceStatusAsync();

        Assert.Equal("session-1", status.SessionId);
        Assert.Equal(1.25, status.ElapsedSeconds);
    }

    [Fact]
    public async Task VoiceCancelSendsSessionIdInTheDaemonFieldName()
    {
        string? body = null;
        var handler = new StubHandler(async (request, token) =>
        {
            body = await request.Content!.ReadAsStringAsync(token);
            return JsonResponse("{\"state\":\"cancelled\",\"session_id\":\"session-1\"}");
        });
        using var http = NewHttpClient(handler);
        var api = new LilaApiClient(http, new FakeCredentialStore("test-token"));

        await api.CancelVoiceAsync("session-1");

        Assert.Equal("{\"session_id\":\"session-1\"}", body);
    }

    [Fact]
    public async Task SafeServerValidationErrorIsDisplayedByTheApiClient()
    {
        var handler = new StubHandler((_, _) => Task.FromResult(new HttpResponseMessage(HttpStatusCode.BadRequest)
        {
            Content = new StringContent("{\"error\":\"Required field 'text' is missing.\",\"code\":\"missing_field\"}",
                Encoding.UTF8, "application/json")
        }));
        using var http = NewHttpClient(handler);
        var api = new LilaApiClient(http, new FakeCredentialStore("test-token"));

        var error = await Assert.ThrowsAsync<LilaApiException>(() => api.SendTextAsync("hey"));

        Assert.Equal(LilaApiFailureKind.InvalidRequest, error.Kind);
        Assert.Equal("Required field 'text' is missing.", error.Message);
    }

    [Fact]
    public async Task UnauthorizedResponseIsClassifiedAsAuthenticationFailure()
    {
        var handler = new StubHandler((_, _) => Task.FromResult(
            new HttpResponseMessage(HttpStatusCode.Unauthorized)));
        using var http = NewHttpClient(handler);
        var api = new LilaApiClient(http, new FakeCredentialStore("bad"));
        var error = await Assert.ThrowsAsync<LilaApiException>(() => api.GetHealthAsync());
        Assert.Equal(LilaApiFailureKind.Authentication, error.Kind);
    }

    [Fact]
    public async Task NetworkFailureIsClassifiedAsConnectionFailure()
    {
        var handler = new StubHandler((_, _) => throw new HttpRequestException("offline"));
        using var http = NewHttpClient(handler);
        var api = new LilaApiClient(http, new FakeCredentialStore("test-token"));
        var error = await Assert.ThrowsAsync<LilaApiException>(() => api.GetHealthAsync());
        Assert.Equal(LilaApiFailureKind.Connection, error.Kind);
    }

    [Fact]
    public async Task RequestTimeoutIsBoundedAndClassified()
    {
        var handler = new StubHandler(async (_, token) =>
        {
            await Task.Delay(Timeout.InfiniteTimeSpan, token);
            return new HttpResponseMessage(HttpStatusCode.OK);
        });
        using var http = NewHttpClient(handler);
        http.Timeout = TimeSpan.FromMilliseconds(40);
        var api = new LilaApiClient(http, new FakeCredentialStore("test-token"));

        var error = await Assert.ThrowsAsync<LilaApiException>(() => api.GetHealthAsync());
        Assert.Equal(LilaApiFailureKind.Timeout, error.Kind);
    }

    [Fact]
    public async Task ChatResponseIsParsedWithoutLoggingTheToken()
    {
        const string secret = "private-token";
        var handler = new StubHandler((_, _) => Task.FromResult(
            JsonResponse("{\"response\":\"hello from Lila\"}")));
        using var http = NewHttpClient(handler);
        var api = new LilaApiClient(http, new FakeCredentialStore(secret));

        var result = await api.SendTextAsync("hello");

        Assert.Equal("hello from Lila", result.Response);
        Assert.Equal("Bearer", handler.LastAuthorization!.Scheme);
        Assert.DoesNotContain(secret, handler.LastUri!.ToString());
    }

    [Fact]
    public async Task WslStoppedIsDistinguishedFromInactiveLilaService()
    {
        var noDistro = new QueueCommandRunner(new CommandResult(0, "", ""));
        var api = NewApi(JsonResponse("{\"status\":\"running\",\"ready\":true,\"voice_enabled\":true}"));
        var stopped = await new WslClient(noDistro, api).CheckAsync();
        Assert.Equal(LilaAvailability.WslUnavailable, stopped.State);
        Assert.Single(noDistro.Calls);

        var inactive = new QueueCommandRunner(
            new CommandResult(0, InterleaveNuls("Ubuntu-24.04") + "\r\n", ""),
            new CommandResult(3, "inactive\n", ""));
        var notStarted = await new WslClient(inactive, api).CheckAsync();
        Assert.Equal(LilaAvailability.ServiceInactive, notStarted.State);
    }

    [Fact]
    public async Task WslRecoveryStartsOnlyTheNamedUserUnitAndPollsHealth()
    {
        var runner = new QueueCommandRunner(
            new CommandResult(0, "", ""), // user systemd manager is available
            new CommandResult(0, "", ""), // explicit start command
            new CommandResult(0, "Ubuntu-24.04\r\n", ""),
            new CommandResult(0, "active\n", ""));
        var api = NewApi(JsonResponse("{\"status\":\"running\",\"ready\":true,\"voice_enabled\":true}"));
        var client = new WslClient(runner, api);

        var result = await client.StartLilaAsync();

        Assert.Equal(LilaAvailability.Ready, result.State);
        Assert.Equal("wsl.exe", runner.Calls[0].Executable);
        Assert.Equal(new[] { "--distribution", "Ubuntu-24.04", "--exec", "systemctl", "--user", "show-environment" },
            runner.Calls[0].Arguments);
        Assert.Equal(new[] { "--distribution", "Ubuntu-24.04", "--exec", "systemctl", "--user", "start", "lila.service" },
            runner.Calls[1].Arguments);
        Assert.DoesNotContain(runner.Calls.SelectMany(call => call.Arguments), value => value.Contains("token"));
    }

    [Fact]
    public async Task LilaStartFailsClearlyWhenSystemdUserManagerIsUnavailable()
    {
        var runner = new QueueCommandRunner(new CommandResult(1, "", "Failed to connect to bus"));
        var api = NewApi(JsonResponse("{\"status\":\"running\",\"ready\":true,\"voice_enabled\":true}"));
        var result = await new WslClient(runner, api).StartLilaAsync();
        Assert.Equal(LilaAvailability.ServiceInactive, result.State);
        Assert.Contains("user manager", result.Message, StringComparison.OrdinalIgnoreCase);
        Assert.Single(runner.Calls);
        Assert.Equal(new[] { "--distribution", "Ubuntu-24.04", "--exec", "systemctl", "--user", "show-environment" },
            runner.Calls[0].Arguments);
    }

    private static LilaApiClient NewApi(HttpResponseMessage response)
    {
        var handler = new StubHandler((_, _) => Task.FromResult(response));
        return new LilaApiClient(NewHttpClient(handler), new FakeCredentialStore("test-token"));
    }

    private static HttpClient NewHttpClient(StubHandler handler) =>
        new(handler) { Timeout = Timeout.InfiniteTimeSpan };

    private static HttpResponseMessage JsonResponse(string json) => new(HttpStatusCode.OK)
    {
        Content = new StringContent(json, Encoding.UTF8, "application/json")
    };

    private static string InterleaveNuls(string value) =>
        string.Concat(value.Select(character => character + "\0"));

    private sealed class FakeCredentialStore(string? token) : ICredentialStore
    {
        public string? ReadToken() => token;
        public void WriteToken(string value) => token = value;
        public void DeleteToken() => token = null;
    }

    private sealed class StubHandler(Func<HttpRequestMessage, CancellationToken,
        Task<HttpResponseMessage>> responder) : HttpMessageHandler
    {
        public int RequestCount { get; private set; }
        public System.Net.Http.Headers.AuthenticationHeaderValue? LastAuthorization { get; private set; }
        public Uri? LastUri { get; private set; }

        protected override async Task<HttpResponseMessage> SendAsync(HttpRequestMessage request,
            CancellationToken cancellationToken)
        {
            RequestCount++;
            LastAuthorization = request.Headers.Authorization;
            LastUri = request.RequestUri;
            return await responder(request, cancellationToken);
        }
    }

    private sealed class QueueCommandRunner(params CommandResult[] results) : ICommandRunner
    {
        private readonly Queue<CommandResult> _results = new(results);
        public List<RecordedCommand> Calls { get; } = [];

        public Task<CommandResult> RunAsync(string executable, IReadOnlyList<string> arguments,
            TimeSpan timeout, CancellationToken cancellationToken = default)
        {
            Calls.Add(new RecordedCommand(executable, arguments.ToArray()));
            if (_results.Count == 0)
                throw new InvalidOperationException("No queued command result.");
            return Task.FromResult(_results.Dequeue());
        }
    }

    private sealed record RecordedCommand(string Executable, string[] Arguments);
}
