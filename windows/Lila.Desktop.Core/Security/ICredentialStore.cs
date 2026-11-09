namespace Lila.Desktop.Core.Security;

public interface ICredentialStore
{
    string? ReadToken();
    void WriteToken(string token);
    void DeleteToken();
}
