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
    public static readonly Uri BaseAddress = new("http://127.0.0.1:8765/");
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
