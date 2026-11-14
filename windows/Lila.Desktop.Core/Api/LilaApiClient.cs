using System.Net;
using System.Net.Http.Json;
using System.Text.Json;
using System.Net.Http.Headers;
using System.Text;
using Lila.Desktop.Core.Security;
using Lila.Desktop.Core.Voice;

namespace Lila.Desktop.Core.Api;

public sealed class LilaApiClient(HttpClient http, ICredentialStore credentials) : IVoiceSessionApi
{
    public static readonly Uri BaseAddress = new("https://127.0.0.1:8765/");
    private const int MaxJsonRequestBytes = 1_000_000;
    private const int MaxErrorBodyBytes = 8_192;

    public event Action<ApiRequestDiagnostic>? RequestCompleted;

    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        PropertyNameCaseInsensitive = true,
        PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower
    };

    public async Task<LilaHealth> GetHealthAsync(CancellationToken cancellationToken = default)
    {
        using var response = await SendAsync(HttpMethod.Get, "health", null,
            TimeSpan.FromSeconds(4), cancellationToken).ConfigureAwait(false);
        return await ReadAsync<LilaHealth>(response, cancellationToken).ConfigureAwait(false);
    }

    public async Task<ChatResult> SendTextAsync(string text,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(text);
        if (Encoding.UTF8.GetByteCount(text) > 900_000)
            throw new ArgumentOutOfRangeException(nameof(text), "Text exceeds the client request limit.");

        using var response = await SendAsync(HttpMethod.Post, "chat", new { text },
            TimeSpan.FromSeconds(180), cancellationToken).ConfigureAwait(false);
        return await ReadAsync<ChatResult>(response, cancellationToken).ConfigureAwait(false);
    }

    public async Task<ApprovalListResult> GetPendingApprovalsAsync(
        CancellationToken cancellationToken = default)
    {
        using var response = await SendAsync(HttpMethod.Get, "approvals", null,
            TimeSpan.FromSeconds(5), cancellationToken).ConfigureAwait(false);
        return await ReadAsync<ApprovalListResult>(response, cancellationToken).ConfigureAwait(false);
    }

    public async Task<OwnerStatusResult> GetOwnerStatusAsync(CancellationToken cancellationToken = default)
    {
        using var response = await SendAsync(HttpMethod.Get, "owner/status", null,
            TimeSpan.FromSeconds(5), cancellationToken).ConfigureAwait(false);
        return await ReadAsync<OwnerStatusResult>(response, cancellationToken).ConfigureAwait(false);
    }

    public async Task<ApprovalDetailResult> GetApprovalAsync(string invocationId,
        CancellationToken cancellationToken = default)
    {
        ValidateApprovalId(invocationId);
        using var response = await SendAsync(HttpMethod.Get,
            $"approvals/{Uri.EscapeDataString(invocationId)}", null,
            TimeSpan.FromSeconds(5), cancellationToken).ConfigureAwait(false);
        return await ReadAsync<ApprovalDetailResult>(response, cancellationToken).ConfigureAwait(false);
    }

    public async Task<ApprovalStatusResult> GetApprovalStatusAsync(string invocationId,
        CancellationToken cancellationToken = default)
    {
        ValidateApprovalId(invocationId);
        using var response = await SendAsync(HttpMethod.Get,
            $"approvals/{Uri.EscapeDataString(invocationId)}/status", null,
            TimeSpan.FromSeconds(5), cancellationToken).ConfigureAwait(false);
        return await ReadAsync<ApprovalStatusResult>(response, cancellationToken).ConfigureAwait(false);
    }

    /// <summary>Attempt exact decision submission without manufacturing owner proof.</summary>
    public async Task<ApprovalStatusResult> SubmitApprovalDecisionAsync(
        ApprovalRequestView request, string decision, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(request);
        ValidateApprovalId(request.InvocationId);
        if (decision is not ("approve" or "reject"))
            throw new ArgumentOutOfRangeException(nameof(decision));
        var payload = new Dictionary<string, string>
        {
            ["challenge_id"] = request.ChallengeId,
            ["key_id"] = request.KeyId,
            ["operation_digest"] = request.OperationDigest,
            ["decision"] = decision
        };
        using var response = await SendAsync(HttpMethod.Post,
            $"approvals/{Uri.EscapeDataString(request.InvocationId)}/decision", payload,
            TimeSpan.FromSeconds(10), cancellationToken).ConfigureAwait(false);
        return await ReadAsync<ApprovalStatusResult>(response, cancellationToken).ConfigureAwait(false);
    }

    /// <summary>Send typed owner intent to the dedicated OIA path.</summary>
    public async Task<ApprovalStatusResult> SubmitOwnerDecisionAsync(
        ApprovalRequestView request, string decision, string ownerPhrase,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(request);
        ArgumentException.ThrowIfNullOrWhiteSpace(ownerPhrase);
        ValidateApprovalId(request.InvocationId);
        if (decision is not ("approve" or "reject"))
            throw new ArgumentOutOfRangeException(nameof(decision));
        var payload = new { phrase = ownerPhrase, decision };
        using var response = await SendAsync(HttpMethod.Post,
            $"owner/approvals/{Uri.EscapeDataString(request.InvocationId)}/decision", payload,
            TimeSpan.FromSeconds(15), cancellationToken).ConfigureAwait(false);
        return await ReadAsync<ApprovalStatusResult>(response, cancellationToken).ConfigureAwait(false);
    }

    public async Task<OwnerVoiceStartResult> StartOwnerVoiceAsync(
        ApprovalRequestView request, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(request);
        ValidateApprovalId(request.InvocationId);
        using var response = await SendAsync(HttpMethod.Post,
            $"owner/approvals/{Uri.EscapeDataString(request.InvocationId)}/voice/start", new { },
            TimeSpan.FromSeconds(10), cancellationToken).ConfigureAwait(false);
        return await ReadAsync<OwnerVoiceStartResult>(response, cancellationToken).ConfigureAwait(false);
    }

    public async Task<ApprovalStatusResult> StopOwnerVoiceAsync(
        ApprovalRequestView request, string sessionId, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(request);
        ArgumentException.ThrowIfNullOrWhiteSpace(sessionId);
        ValidateApprovalId(request.InvocationId);
        using var response = await SendAsync(HttpMethod.Post,
            $"owner/approvals/{Uri.EscapeDataString(request.InvocationId)}/voice/stop",
            new { session_id = sessionId }, TimeSpan.FromSeconds(300), cancellationToken).ConfigureAwait(false);
        return await ReadAsync<ApprovalStatusResult>(response, cancellationToken).ConfigureAwait(false);
    }

    public async Task CancelOwnerVoiceAsync(ApprovalRequestView request, string sessionId,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(request);
        ArgumentException.ThrowIfNullOrWhiteSpace(sessionId);
        ValidateApprovalId(request.InvocationId);
        using var response = await SendAsync(HttpMethod.Post,
            $"owner/approvals/{Uri.EscapeDataString(request.InvocationId)}/voice/cancel",
            new { session_id = sessionId }, TimeSpan.FromSeconds(10), cancellationToken).ConfigureAwait(false);
        _ = await ReadAsync<JsonElement>(response, cancellationToken).ConfigureAwait(false);
    }

    private static void ValidateApprovalId(string invocationId)
    {
        if (string.IsNullOrWhiteSpace(invocationId) || invocationId.Length > 128 ||
            !invocationId.All(character => char.IsAsciiLetterOrDigit(character) || character is '.' or '_' or '-'))
            throw new ArgumentException("Invalid approval identifier.", nameof(invocationId));
    }

    public async Task<VoiceStartResult> StartSpokenEnrollmentAsync(CancellationToken cancellationToken = default)
    {
        using var response = await SendAsync(HttpMethod.Post, "owner/spoken/start", new { },
            TimeSpan.FromSeconds(10), cancellationToken).ConfigureAwait(false);
        return await ReadAsync<VoiceStartResult>(response, cancellationToken).ConfigureAwait(false);
    }

    public async Task<JsonElement> StopSpokenEnrollmentAsync(string sessionId,
        CancellationToken cancellationToken = default)
    {
        using var response = await SendAsync(HttpMethod.Post, "owner/spoken/stop", new { session_id = sessionId },
            TimeSpan.FromSeconds(180), cancellationToken).ConfigureAwait(false);
        return await ReadAsync<JsonElement>(response, cancellationToken).ConfigureAwait(false);
    }

    public async Task CancelSpokenEnrollmentAsync(string sessionId, CancellationToken cancellationToken = default)
    {
        using var response = await SendAsync(HttpMethod.Post, "owner/spoken/cancel", new { session_id = sessionId },
            TimeSpan.FromSeconds(10), cancellationToken).ConfigureAwait(false);
        await ReadAsync<JsonElement>(response, cancellationToken).ConfigureAwait(false);
    }

    public async Task<VoiceStartResult> StartVoiceAsync(CancellationToken cancellationToken = default)
    {
        using var response = await SendAsync(HttpMethod.Post, "voice/start", new { },
            TimeSpan.FromSeconds(10), cancellationToken).ConfigureAwait(false);
        return await ReadAsync<VoiceStartResult>(response, cancellationToken).ConfigureAwait(false);
    }

    public async Task<VoiceStopResult> StopVoiceAsync(string sessionId,
        CancellationToken cancellationToken = default)
    {
        using var response = await SendAsync(HttpMethod.Post, "voice/stop", new { session_id = sessionId },
            TimeSpan.FromSeconds(300), cancellationToken).ConfigureAwait(false);
        return await ReadAsync<VoiceStopResult>(response, cancellationToken).ConfigureAwait(false);
    }

    public async Task<VoiceCancelResult> CancelVoiceAsync(string sessionId,
        CancellationToken cancellationToken = default)
    {
        using var response = await SendAsync(HttpMethod.Post, "voice/cancel", new { session_id = sessionId },
            TimeSpan.FromSeconds(10), cancellationToken).ConfigureAwait(false);
        return await ReadAsync<VoiceCancelResult>(response, cancellationToken).ConfigureAwait(false);
    }

    public async Task<VoiceSessionStatus> GetVoiceStatusAsync(CancellationToken cancellationToken = default)
    {
        using var response = await SendAsync(HttpMethod.Get, "voice/status", null,
            TimeSpan.FromSeconds(5), cancellationToken).ConfigureAwait(false);
        return await ReadAsync<VoiceSessionStatus>(response, cancellationToken).ConfigureAwait(false);
    }

    public async Task<EventBatchResult> PollEventsAsync(long sinceCursor = 0, int limit = 100,
        CancellationToken cancellationToken = default)
    {
        using var response = await SendAsync(HttpMethod.Get, $"events?since={sinceCursor}&limit={limit}", null,
            TimeSpan.FromSeconds(10), cancellationToken).ConfigureAwait(false);
        return await ReadAsync<EventBatchResult>(response, cancellationToken).ConfigureAwait(false);
    }

    public async Task<TaskListResult> GetTasksAsync(CancellationToken cancellationToken = default)
    {
        using var response = await SendAsync(HttpMethod.Get, "tasks", null,
            TimeSpan.FromSeconds(5), cancellationToken).ConfigureAwait(false);
        return await ReadAsync<TaskListResult>(response, cancellationToken).ConfigureAwait(false);
    }

    public async Task<DaemonTaskDto> CancelTaskAsync(string taskId, string reason = "owner_cancelled",
        CancellationToken cancellationToken = default)
    {
        using var response = await SendAsync(HttpMethod.Post, $"tasks/{Uri.EscapeDataString(taskId)}/cancel",
            new { reason }, TimeSpan.FromSeconds(10), cancellationToken).ConfigureAwait(false);
        return await ReadAsync<DaemonTaskDto>(response, cancellationToken).ConfigureAwait(false);
    }

    public async Task<TraceListResult> GetTracesAsync(int limit = 25, CancellationToken cancellationToken = default)
    {
        using var response = await SendAsync(HttpMethod.Get, $"traces?limit={limit}", null,
            TimeSpan.FromSeconds(5), cancellationToken).ConfigureAwait(false);
        return await ReadAsync<TraceListResult>(response, cancellationToken).ConfigureAwait(false);
    }

    public async Task<SentinelTriageResult> PostSentinelTriageAsync(
        IReadOnlyList<SentinelHostFactDto> hostFacts,
        IReadOnlyList<string>? sources = null,
        CancellationToken cancellationToken = default)
    {
        using var response = await SendAsync(HttpMethod.Post, "sentinel/triage", new
        {
            sources = sources ?? new[] { "system_resources", "defender", "firewall" },
            host_facts = hostFacts
        }, TimeSpan.FromSeconds(10), cancellationToken).ConfigureAwait(false);
        return await ReadAsync<SentinelTriageResult>(response, cancellationToken).ConfigureAwait(false);
    }

    public async Task<PerceptionGestureResult> IssuePerceptionGestureAsync(string scope = "region",
        CancellationToken cancellationToken = default)
    {
        using var response = await SendAsync(HttpMethod.Post, "perception/gesture", new { scope },
            TimeSpan.FromSeconds(5), cancellationToken).ConfigureAwait(false);
        return await ReadAsync<PerceptionGestureResult>(response, cancellationToken).ConfigureAwait(false);
    }

    public async Task<PerceptionCaptureResult> CapturePerceptionAsync(
        string ownerGestureToken,
        string scope,
        string targetLabel,
        string imageBase64,
        int width,
        int height,
        CancellationToken cancellationToken = default)
    {
        using var response = await SendAsync(HttpMethod.Post, "perception/capture", new
        {
            initiator = "owner_gesture",
            owner_gesture_token = ownerGestureToken,
            scope,
            target_label = targetLabel,
            image_base64 = imageBase64,
            width,
            height
        }, TimeSpan.FromSeconds(5), cancellationToken).ConfigureAwait(false);
        return await ReadAsync<PerceptionCaptureResult>(response, cancellationToken).ConfigureAwait(false);
    }

    public async Task<JsonElement> ObservePerceptionAsync(string captureId,
        CancellationToken cancellationToken = default)
    {
        using var response = await SendAsync(HttpMethod.Post, "perception/observe",
            new { capture_id = captureId }, TimeSpan.FromSeconds(5), cancellationToken).ConfigureAwait(false);
        return await ReadAsync<JsonElement>(response, cancellationToken).ConfigureAwait(false);
    }

    public async Task DiscardPerceptionAsync(string captureId,
        CancellationToken cancellationToken = default)
    {
        using var response = await SendAsync(HttpMethod.Post, "perception/discard",
            new { capture_id = captureId }, TimeSpan.FromSeconds(5), cancellationToken).ConfigureAwait(false);
        response.EnsureSuccessStatusCode();
    }

    public async Task<CapabilityCandidateListResult> GetCapabilityCandidatesAsync(
        CancellationToken cancellationToken = default)
    {
        using var response = await SendAsync(HttpMethod.Get, "capabilities/candidates", null,
            TimeSpan.FromSeconds(5), cancellationToken).ConfigureAwait(false);
        return await ReadAsync<CapabilityCandidateListResult>(response, cancellationToken).ConfigureAwait(false);
    }

    public async Task<JsonElement> PromoteCapabilityCandidateAsync(string capabilityId, bool ownerApproved = true,
        CancellationToken cancellationToken = default)
    {
        using var response = await SendAsync(HttpMethod.Post,
            $"capabilities/candidates/{Uri.EscapeDataString(capabilityId)}/promote",
            new { owner_approved = ownerApproved }, TimeSpan.FromSeconds(10), cancellationToken).ConfigureAwait(false);
        return await ReadAsync<JsonElement>(response, cancellationToken).ConfigureAwait(false);
    }

    public async Task<JsonElement> RollbackCapabilityCatalogAsync(int targetVersion,
        CancellationToken cancellationToken = default)
    {
        using var response = await SendAsync(HttpMethod.Post, "capabilities/rollback",
            new { target_version = targetVersion }, TimeSpan.FromSeconds(5), cancellationToken).ConfigureAwait(false);
        return await ReadAsync<JsonElement>(response, cancellationToken).ConfigureAwait(false);
    }

    public async IAsyncEnumerable<DaemonEventDto> StreamEventsAsync(long sinceCursor = 0,
        [System.Runtime.CompilerServices.EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        var token = credentials.ReadToken();
        if (string.IsNullOrWhiteSpace(token))
            throw new LilaApiException(LilaApiFailureKind.Authentication,
                "No Lila API credential is saved. Add it in Settings.");

        using var request = new HttpRequestMessage(HttpMethod.Get,
            new Uri(BaseAddress, $"events/stream?since={sinceCursor}"));
        request.Headers.Authorization = new System.Net.Http.Headers.AuthenticationHeaderValue("Bearer", token);
        request.Headers.TryAddWithoutValidation("Last-Event-ID", sinceCursor.ToString());

        using var response = await http.SendAsync(request, HttpCompletionOption.ResponseHeadersRead,
            cancellationToken).ConfigureAwait(false);
        if (response.StatusCode == HttpStatusCode.Unauthorized)
            throw new LilaApiException(LilaApiFailureKind.Authentication,
                "Lila rejected the saved API credential.", HttpStatusCode.Unauthorized);
        if (!response.IsSuccessStatusCode)
            throw new LilaApiException(LilaApiFailureKind.Service,
                $"Event stream returned {(int)response.StatusCode}.", response.StatusCode);

        await using var stream = await response.Content.ReadAsStreamAsync(cancellationToken).ConfigureAwait(false);
        using var reader = new StreamReader(stream, System.Text.Encoding.UTF8);
        while (!cancellationToken.IsCancellationRequested)
        {
            var line = await reader.ReadLineAsync(cancellationToken).ConfigureAwait(false);
            if (line is null)
                break;
            if (!line.StartsWith("data: ", StringComparison.Ordinal))
                continue;
            var json = line[6..];
            DaemonEventDto? ev = null;
            try
            {
                ev = JsonSerializer.Deserialize<DaemonEventDto>(json, JsonOptions);
            }
            catch (JsonException)
            {
                // Skip malformed SSE data frame without crashing stream.
            }
            if (ev is not null)
                yield return ev;
        }
    }

    private async Task<HttpResponseMessage> SendAsync(HttpMethod method, string route,
        object? payload, TimeSpan timeout, CancellationToken callerToken)
    {
        var startedAt = System.Diagnostics.Stopwatch.GetTimestamp();
        int? statusCode = null;
        var outcome = "failed";
        var token = credentials.ReadToken();
        if (string.IsNullOrWhiteSpace(token))
            throw new LilaApiException(LilaApiFailureKind.Authentication,
                "No Lila API credential is saved. Add it in Settings.");

        using var timeoutSource = CancellationTokenSource.CreateLinkedTokenSource(callerToken);
        timeoutSource.CancelAfter(timeout);
        using var request = new HttpRequestMessage(method, new Uri(BaseAddress, route));
        request.Headers.Authorization = new System.Net.Http.Headers.AuthenticationHeaderValue("Bearer", token);
        if (payload is not null)
            request.Content = CreateJsonContent(payload);

        try
        {
            var response = await http.SendAsync(request, HttpCompletionOption.ResponseHeadersRead,
                timeoutSource.Token).ConfigureAwait(false);
            statusCode = (int)response.StatusCode;
            if (response.StatusCode == HttpStatusCode.Unauthorized)
            {
                response.Dispose();
                throw new LilaApiException(LilaApiFailureKind.Authentication,
                    "Lila rejected the saved API credential.", HttpStatusCode.Unauthorized);
            }
            if (response.StatusCode == HttpStatusCode.Conflict)
            {
                var message = await ReadSafeErrorAsync(response, timeoutSource.Token).ConfigureAwait(false)
                              ?? "The voice session is in an invalid state.";
                response.Dispose();
                throw new LilaApiException(LilaApiFailureKind.Conflict, message, HttpStatusCode.Conflict);
            }
            if (response.StatusCode == HttpStatusCode.BadGateway ||
                response.StatusCode == HttpStatusCode.ServiceUnavailable)
            {
                var message = await ReadSafeErrorAsync(response, timeoutSource.Token).ConfigureAwait(false)
                              ?? "Lila's service or voice backend is not ready.";
                response.Dispose();
                throw new LilaApiException(LilaApiFailureKind.Service, message, response.StatusCode);
            }
            if (!response.IsSuccessStatusCode)
            {
                var status = response.StatusCode;
                var serverMessage = await ReadSafeErrorAsync(response, timeoutSource.Token).ConfigureAwait(false);
                response.Dispose();
                var kind = status == HttpStatusCode.BadRequest || status == HttpStatusCode.LengthRequired ||
                           status == HttpStatusCode.RequestEntityTooLarge ||
                           status == HttpStatusCode.UnsupportedMediaType
                    ? LilaApiFailureKind.InvalidRequest : LilaApiFailureKind.Service;
                throw new LilaApiException(kind,
                    serverMessage ?? $"Lila returned HTTP {(int)status}.", status);
            }
            outcome = response.IsSuccessStatusCode ? "success" : "http_error";
            return response;
        }
        catch (OperationCanceledException ex) when (!callerToken.IsCancellationRequested)
        {
            throw new LilaApiException(LilaApiFailureKind.Timeout,
                "The request to Lila timed out. Its server-side outcome may be uncertain.",
                innerException: ex);
        }
        catch (HttpRequestException ex)
        {
            throw new LilaApiException(LilaApiFailureKind.Connection,
                "Cannot connect to Lila at 127.0.0.1:8765.", innerException: ex);
        }
        finally
        {
            RequestCompleted?.Invoke(new ApiRequestDiagnostic(method.Method, route, statusCode,
                System.Diagnostics.Stopwatch.GetElapsedTime(startedAt), outcome));
        }
    }

    private static HttpContent CreateJsonContent(object payload)
    {
        var bytes = JsonSerializer.SerializeToUtf8Bytes(payload, payload.GetType(), JsonOptions);
        if (bytes.Length > MaxJsonRequestBytes)
            throw new ArgumentOutOfRangeException(nameof(payload),
                "Serialized JSON request exceeds the server request limit.");

        // ByteArrayContent exposes a known length. JsonContent leaves its length
        // unknown, so HTTP/1.1 sends Transfer-Encoding: chunked, which the local
        // API deliberately rejects in favor of bounded Content-Length requests.
        var content = new ByteArrayContent(bytes);
        content.Headers.ContentType = new MediaTypeHeaderValue("application/json")
        {
            CharSet = "utf-8"
        };
        return content;
    }

    private static async Task<string?> ReadSafeErrorAsync(HttpResponseMessage response,
        CancellationToken cancellationToken)
    {
        try
        {
            await using var stream = await response.Content.ReadAsStreamAsync(cancellationToken)
                .ConfigureAwait(false);
            var bytes = new byte[MaxErrorBodyBytes + 1];
            var count = 0;
            while (count < bytes.Length)
            {
                var read = await stream.ReadAsync(bytes.AsMemory(count, bytes.Length - count),
                    cancellationToken).ConfigureAwait(false);
                if (read == 0)
                    break;
                count += read;
            }
            if (count == 0 || count > MaxErrorBodyBytes)
                return null;

            using var document = JsonDocument.Parse(bytes.AsMemory(0, count));
            if (!document.RootElement.TryGetProperty("error", out var error) ||
                error.ValueKind != JsonValueKind.String)
                return null;

            var value = error.GetString();
            if (string.IsNullOrWhiteSpace(value))
                return null;
            var safe = new string(value.Where(character => !char.IsControl(character) || character == ' ')
                .Take(300).ToArray()).Trim();
            return safe.Length == 0 ? null : safe;
        }
        catch (Exception ex) when (ex is JsonException or IOException or InvalidOperationException)
        {
            return null;
        }
    }

    private static async Task<T> ReadAsync<T>(HttpResponseMessage response,
        CancellationToken cancellationToken)
    {
        try
        {
            return await response.Content.ReadFromJsonAsync<T>(JsonOptions, cancellationToken)
                       .ConfigureAwait(false)
                   ?? throw new JsonException("Empty API response.");
        }
        catch (JsonException ex)
        {
            throw new LilaApiException(LilaApiFailureKind.InvalidResponse,
                "Lila returned an invalid response.", response.StatusCode, ex);
        }
    }
}

public sealed record ApiRequestDiagnostic(string Method, string Route, int? StatusCode,
    TimeSpan Duration, string Outcome);
