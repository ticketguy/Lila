using System.Net;

namespace Lila.Desktop.Core.Api;

public sealed record LilaHealth(
    string Status,
    bool Ready,
    bool VoiceEnabled,
    string? Model,
    string? Error);

public sealed record VoiceSessionStatus(
    string State,
    string? SessionId,
    double? ElapsedSeconds,
    string? Error);

public sealed record VoiceStartResult(
    string State,
    string SessionId,
    double? ElapsedSeconds,
    string? Error);

public sealed record VoiceStopResult(
    string SessionId,
    string Transcript,
    string Response,
    VoiceSessionStatus Status);

public sealed record VoiceCancelResult(
    string State,
    string? SessionId,
    string? Error);

public sealed record ChatResult(string Response);

public enum LilaApiFailureKind
{
    Authentication,
    Connection,
    Timeout,
    InvalidRequest,
    Service,
    Conflict,
    InvalidResponse
}

public sealed class LilaApiException(LilaApiFailureKind kind, string message,
    HttpStatusCode? statusCode = null, Exception? innerException = null)
    : Exception(message, innerException)
{
    public LilaApiFailureKind Kind { get; } = kind;
    public HttpStatusCode? StatusCode { get; } = statusCode;
}
