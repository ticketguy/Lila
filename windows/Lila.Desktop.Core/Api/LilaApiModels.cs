using System.Net;
using System.Text.Json;

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
    VoiceSessionStatus Status)
{
    public IReadOnlyList<ToolResultView> ToolResults { get; init; } = Array.Empty<ToolResultView>();
    public int ToolResultsOmitted { get; init; }
}

public sealed record VoiceCancelResult(
    string State,
    string? SessionId,
    string? Error);

public sealed record OwnerVoiceStartResult(string SessionId, string State, string VoiceChallengeCode);

public sealed record ChatResult(string Response)
{
    public IReadOnlyList<ToolResultView> ToolResults { get; init; } = Array.Empty<ToolResultView>();
    public int ToolResultsOmitted { get; init; }
}

public sealed record ApprovalRequestView(
    string InvocationId, string Status, string CapabilityId, string ContractVersion,
    string OperationDigest, string ChallengeId, string VoiceChallengeCode, string KeyId,
    string? TargetKind, string? TargetResolverId, JsonElement? Target,
    JsonElement Arguments, string PolicyReason,
    string PolicyDetail, string? Risk, double ExpiresInSeconds,
    bool ApprovalReady = false, string? DisplayAction = null,
    JsonElement DisplayValues = default, string? DisplayBlockReason = null);

public sealed record ApprovalListResult(IReadOnlyList<ApprovalRequestView> Requests);

public sealed record ApprovalDetailResult(string Kind, ApprovalRequestView? Request, ApprovalStatusResult? Status);

public sealed record ApprovalStatusResult(string? InvocationId, string Status, string ReasonCode);

public sealed record AuthenticationProviderAvailability(
    string ProviderId, bool Available, bool Enrolled, string ReasonCode);

public sealed record OwnerStatusResult(
    bool Enrolled, bool SessionValid, IReadOnlyList<AuthenticationProviderAvailability> Providers);

public sealed record DaemonEventDto(
    string SchemaVersion,
    long EventId,
    long Cursor,
    string EventType,
    double Timestamp,
    JsonElement Payload);

public sealed record EventBatchResult(
    string SchemaVersion,
    long SinceCursor,
    long NextCursor,
    bool GapDetected,
    IReadOnlyList<DaemonEventDto> Events);

public sealed record DaemonTaskDto(
    string TaskId,
    string Kind,
    string Description,
    string Status,
    double Progress,
    string? Error);

public sealed record TaskListResult(IReadOnlyList<DaemonTaskDto> Tasks);

public sealed record TraceListResult(IReadOnlyList<JsonElement> Traces);

public sealed record SentinelHostFactDto(string Source, string Metric, object Value, bool ReadOnly = true);

public sealed record SentinelTriageResult(string HighestTier, bool AutonomousRemediationAttempted, bool ScreenCaptureUsed, JsonElement Coverage, IReadOnlyList<JsonElement> Indicators);

public sealed record PerceptionGestureResult(string OwnerGestureToken, string Scope);

public sealed record PerceptionCaptureResult(string CaptureId, string Scope, string ImageDigest, bool Persisted, bool Stale, double TtlRemainingS);

public sealed record CapabilityCandidateListResult(JsonElement CatalogVersion, IReadOnlyList<JsonElement> Candidates);

public sealed record ToolResultView
{
    public string Tool { get; init; } = "";
    public string Label { get; init; } = "";
    public bool Success { get; init; }
    public string Status { get; init; } = "";
    public string VerificationStatus { get; init; } = "";
    public string Output { get; init; } = "";
    public bool OutputTruncated { get; init; }
    public string? Error { get; init; }
    public bool ErrorTruncated { get; init; }
    public JsonElement? Data { get; init; }
    public string DataStatus { get; init; } = "";
}

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
