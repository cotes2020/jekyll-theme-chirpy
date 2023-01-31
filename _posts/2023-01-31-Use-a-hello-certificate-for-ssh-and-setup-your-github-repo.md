---
layout: post
title: Use a hello certificate for ssh and setup your github repo
date: 2023-01-31 00:05
category: 
- Instructions
author: Eric Marquez
tags:
- SSH
- Windows
- Windows Hello
- GitHub
- Keys
- PowerShell
summary: Use the windows hello certificate as the ssh key for github
---
* Table of Contents
{:toc}

Recently, I need to setup SSH keys on my Github account and ran into an issue setting this up. I thought I would document my setup procedure. For the most part, the process is pretty straight forward, but it's a little different than the Linux environment. All my steps are using an elevated PowerShell command window.

The first section will show how to geneate your ssh key using a Windows Hello certificate. After the key is generated, it will be add to the ssh-agent service. To complete the setup, I'll show where to add the new key to the my GitHub account.

## Generate the ssh key from a Windows Hello certificate

1. SSH keys are stored in the `$ENV:USERPROFILE\.ssh` directory.

```powershell
$thumbprint = (Get-ChildItem Microsoft.PowerShell.Security\Certificate::CurrentUser -Recurse | Where-Object {$_.Issuer -match 'Windows Hello'} | Select-Object -First 1).Thumbprint

# find the thumbprint
$path = 'c:\temp\base64cert.cer'
Set-Content -Path $path -Value ([convert]::tobase64string((Get-Item cert:\currentuser\my\$thumbprint).RawData)) -Encoding Ascii
```

convert to pem and that pem will useable for a ssh-keygen

```powershell
# Convert the cer to a pem format and use the pem to generate the ssh key.
certutil -encode 'c:\temp\base64cert.cer' 'c:\temp\hello.pem'
ssh-keygen.exe  -f 'c:\temp\hello.pem' -f $Env:USERPROFILE\.ssh\id_testkey

# Clean up the certificates that were generated.
Remove-Item ('c:\temp\base64cert.cer', 'c:\temp\hello.pem')
```

## Setup the SSH-Agent service

By default the ssh-agent service is disabled. Set the service to automatic, this will allow it to start when the computer is rebooted. The PowerShell console will need to have elevated privileges.

```powershell
Get-Service ssh-agent | Set-Service -StartupType Automatic
```

Start the service ssh-agent service.

```powershell
Start-Service ssh-agent
```

To check if the ssh-agent service is setup correct, run the following:

```powershell
Get-Service ssh-agent | select *

UserName            : LocalSystem
Description         : Agent to hold private keys used for public key authentication.
DelayedAutoStart    : False
BinaryPathName      : C:\WINDOWS\System32\OpenSSH\ssh-agent.exe
StartupType         : Automatic
Name                : ssh-agent
RequiredServices    : {}
CanPauseAndContinue : False
CanShutdown         : False
CanStop             : True
DisplayName         : OpenSSH Authentication Agent
DependentServices   : {}
MachineName         : .
ServiceName         : ssh-agent
ServicesDependedOn  : {}
StartType           : Automatic
ServiceHandle       :
Status              : Running
ServiceType         : Win32OwnProcess
Site                :
Container           :
```

 If you SSH-Agent does not have a Status of Running, this needs to be started.  Open the PowerShell window with elevated privileges.

 ```powershell
Get-Service -Name ssh-agent | Set-Service -StartupType Automatic -Status Running
```

Now load your key files into ssh-agent

```powershell
ssh-add $env:USERPROFILE\.ssh\id_testkey
```

Set the ssh profile for GitHub in the SSH Config file.  This will tell SSH how to proceed for future ssh connections.

```Powershell
Get-Content $env:UserProfile\.ssh\config

Host github.com-repo-0
    Hostname github.com
    IdentityFile=C:/Users/myusername/.ssh/id_testkey

Host github.com-repo-1
    Hostname github.com
    IdentityFile=C:/Users/myusername/.ssh/id_testkey
```

## Add the ssh key to GitHub

Then add your ssh key to github.  This can be done in one of two places.

1. The user profile for the entire account
    1.1 Under your user profile image in the right corner. Select Settings.

    {:style="clear: left"}
    ![github drop down menu under settings](https://ebmarquez.blob.core.windows.net/public-read/image/blog/sshkey/github20220906234032.png){:.align-left width="35%"}

   1.2. Next select ssh and GPG keys

    ![Left rail access section, ssh and GPG keys](https://ebmarquez.blob.core.windows.net/public-read/image/blog/sshkey/github20220906234238.png){:.align-left width="35%"}

   1.3. Select the New ssh key button

    ![ssh key button](https://ebmarquez.blob.core.windows.net/public-read/image/blog/sshkey/github_button_20220906234335.png){:.align-left width="35%"}

   1.4. Copy the public key contents to your clipboard.

    ```powershell
        Get-Content $env:userprofile\.ssh\id_testkey.pub | clip.exe
    ```

    1.5. paste in the key

    ![ssh key text box](https://ebmarquez.blob.core.windows.net/public-read/image/blog/sshkey/github20220906234431.png){:.align-left width="100%"}

2. In your windows powershell console, test the ssh key.

    ```console
    ssh -T git@github.com
    Hi ebmarquez! You've successfully authenticated, but GitHub does not provide shell access.
    ```

3. In your GitHub repo set the git configuration to utilize the new ssh key.

## Configure the local git repo

If you have already cloned the repo the following git config command will set the correct ssh key that should be utilized by git.

```powershell
git config core.sshCommand "ssh -i C:\Users\myusername\.ssh\id_testkey -F /dev/null"
```

## References

- [Key-based authentication in OpenSSH for Windows \| Microsoft Docs](https://docs.microsoft.com/en-us/windows-server/administration/openssh/openssh_keymanagement)
- [Microsoft Learn certutil](https://learn.microsoft.com/en-us/windows-server/administration/windows-commands/certutil)
- [Microsoft TechNet Export Certificate using base 64 cer format with powershell](https://social.technet.microsoft.com/Forums/en-US/37a213b9-f185-482e-b610-295f2056506e/export-certificate-using-base-64-cer-format-with-powershell-)
