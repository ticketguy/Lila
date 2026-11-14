using System.Net.Security;
using System.Security.Cryptography;
using System.Security.Cryptography.X509Certificates;

namespace Lila.Desktop.Core.Api;

/// <summary>Builds the production Island handler from explicitly provisioned certificates.</summary>
public static class LilaMutualTlsHandler
{
    public static HttpClientHandler CreateFromEnvironment(Func<string, string?>? readSetting = null)
    {
        readSetting ??= Environment.GetEnvironmentVariable;
        var clientThumbprint = Normalize(readSetting("LILA_TLS_CLIENT_CERT_SHA256"));
        var serverThumbprint = Normalize(readSetting("LILA_TLS_SERVER_CERT_SHA256"));
        if (clientThumbprint.Length != 64 || serverThumbprint.Length != 64)
            throw new InvalidOperationException("Lila mTLS client and pinned server certificates must be configured.");

        var clientCertificate = FindClientCertificate(clientThumbprint);
        var handler = new HttpClientHandler { AllowAutoRedirect = false };
        handler.ClientCertificateOptions = ClientCertificateOption.Manual;
        handler.ClientCertificates.Add(clientCertificate);
        handler.ServerCertificateCustomValidationCallback = (_, certificate, _, errors) =>
            certificate is not null &&
            !errors.HasFlag(SslPolicyErrors.RemoteCertificateNotAvailable) &&
            IsPinnedServerCertificate(new X509Certificate2(certificate), serverThumbprint);
        return handler;
    }

    public static bool IsPinnedServerCertificate(X509Certificate2? certificate, string sha256Pin)
    {
        var pin = Normalize(sha256Pin);
        if (certificate is null || pin.Length != 64 ||
            DateTimeOffset.UtcNow < certificate.NotBefore || DateTimeOffset.UtcNow > certificate.NotAfter)
            return false;
        return CryptographicOperations.FixedTimeEquals(
            SHA256.HashData(certificate.RawData), Convert.FromHexString(pin));
    }

    private static X509Certificate2 FindClientCertificate(string thumbprint)
    {
        using var store = new X509Store(StoreName.My, StoreLocation.CurrentUser);
        store.Open(OpenFlags.ReadOnly | OpenFlags.OpenExistingOnly);
        foreach (var candidate in store.Certificates)
        {
            if (!candidate.HasPrivateKey || DateTimeOffset.UtcNow < candidate.NotBefore ||
                DateTimeOffset.UtcNow > candidate.NotAfter)
                continue;
            if (CryptographicOperations.FixedTimeEquals(
                    SHA256.HashData(candidate.RawData), Convert.FromHexString(thumbprint)))
                return new X509Certificate2(candidate);
        }
        throw new InvalidOperationException("The configured, valid Island client certificate is unavailable.");
    }

    private static string Normalize(string? value) =>
        new((value ?? string.Empty).Where(Uri.IsHexDigit).Select(char.ToUpperInvariant).ToArray());
}
