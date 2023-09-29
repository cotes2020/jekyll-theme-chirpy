---
layout: post
title: From unprivileged user to system - KrbRelayUp
categories: [Microsoft, Windows]
tags:
- privesc
- windows
- ad
- redteam
- exploit
lang: fr-FR
date: 2023-01-19 21:42 +0100
description: In this article, I will show you a very useful exploit which allows you to perform a privilege escalation in a Windows environnement. From an unprivileged user to nt autority\system.
image: /assets/img/KrbRelayUp/preview.png
---
## Summary
In this article, I will show you a very useful exploit which allows you to perform a privilege escalation in a Windows environnement. **From an unprivileged user to `nt autority\system`**.
## KrbRelayUp
[KrbRelayUp](https://github.com/Dec0ne/KrbRelayUp) is an exploit made by **Dec0ne** compiling the work of [KrbRelay](https://github.com/cube0x0/KrbRelay) (**cube0x0**) and other tools like [Rubeus](https://github.com/GhostPack/Rubeus/).

A very accurate description of the exploit by his creator :
> This is essentially a universal no-fix local privilege escalation in windows domain environments where LDAP signing is not enforced (the default settings). 

It means that **every Windows machine** in a domain is vulnerable (and will be) as long as nobody changed the default settings and enforced the LDAP signing setting.
And the cool thing is that you don't need to have a special privilege or be an Administrator.

That being said, you can see how powerful this tool is.
> UPDATE : since **october 2022** the exploit from this repository has been **fixed** (see discussion [here](https://twitter.com/_Imm0/status/1583187655222706177)). You can see that there is an error when connecting to LDAP [https://github.com/Dec0ne/KrbRelayUp/issues/31](https://github.com/Dec0ne/KrbRelayUp/issues/31). However the [manual](https://gist.github.com/tothi/bf6c59d6de5d0c9710f23dae5750c4b9) exploitation might work.
{: .prompt-warning }
## Prerequisites
Before running the `.exe`, you have to download the whole project here : [https://github.com/Dec0ne/KrbRelayUp](https://github.com/Dec0ne/KrbRelayUp).
Once you have downloaded it, open the `KrbRelayUp.sln` file (which is the project file) with [Microsoft Visual Studio Community](https://visualstudio.microsoft.com/fr/free-developer-offers/).
Then right-click on the project and hit the "Build" button :
![Build .exe](/assets/img/KrbRelayUp/KrbRelayUp_1.png){: .shadow }
And there you go ! You have successfully built your `.exe`, you can now use it !
![Build .exe](/assets/img/KrbRelayUp/KrbRelayUp_2.png){: .shadow }
## Usage
As I write this article, there are 3 methods which have been developed for KrbRelayUp :
`RBCD` (Ressource Based Constrained Delegation), `ShadowCred` and `ADCS` method.

In this article, I will only show you the `RBCD` method, which is the default method (because it's the only one I have tested so far).

Like you can see, I'm a basic domain user without any Administrator privilege :
![whoami](/assets/img/KrbRelayUp/KrbRelayUp_3.png){: .shadow }

Then, when I execute this command, KrbRelayUp.exe will perform an RBCD and ultimately a `new system shell` will pop :
```shell
KrbRelayUp.exe full -Domain corp.local --CreateNewComputerAccount --ComputerName FakeMachine01$ --ComputerPassword passw@rd123
```
* `-Domain` : the name of the domain where your machine is
* `--CreateNewComputerAccount` : create a new computer account if you don't have one yet
* `--ComputerName` : the name of your computer account (if it's a new one just write whatever you want)
* `--ComputerPassword` : the password of your computer account
![execution](/assets/img/KrbRelayUp/KrbRelayUp_4.png){: .shadow }

In my case, I have graphical interface to interact with this new system shell. But what if I don't have one ?

Don't worry ! KrbRelayUp have your back with the `--sc` option. 
The `--sc <bin_path>` option allows you to specify the path of your binary (revershell or beacon) to run as system instead of cmd.exe.
![execution sc](/assets/img/KrbRelayUp/KrbRelayUp_5.png){: .shadow }
## Limitations
KrbRelayUp is a very cool tool but it's not magic !

An up to date `Windows Defender` will catch KrbRelayUp pretty easily (old Defender won't so you are free to use it on HTB or THM old machines :D).

AVs/EDRs might also catch it, but as you have the source code you can modify it (starting by removing some recognisable strings to avoid static detection).

Also, a few `Sigma`[^1] and `Elastic`[^2] rules  have been created by the community so a SOC might be able to detect KrbRelayUp behavior.

Lastly, if LDAP signing is enforced you will not be able to perform the exploit.[^3]

---
> **Disclaimer** : The technics presented in this article should not be used outside of a legal environnement. Also I'm not responsible for the use of these techniques.
{: .prompt-danger }

---
[^1]: [https://github.com/tsale/Sigma_rules/blob/main/windows_exploitation/KrbRelayUp.yml](https://github.com/tsale/Sigma_rules/blob/main/windows_exploitation/KrbRelayUp.yml), [https://github.com/SigmaHQ/sigma/blob/master/rules/windows/process_creation/proc_creation_win_hack_krbrelayup.yml](https://github.com/SigmaHQ/sigma/blob/master/rules/windows/process_creation/proc_creation_win_hack_krbrelayup.yml)
[^2]: [https://github.com/elastic/detection-rules/blob/fb6ee2c69864ffdfe347bf3b050cb931f53067a6/rules/windows/privilege_escalation_krbrelayup_suspicious_logon.toml](https://github.com/elastic/detection-rules/blob/fb6ee2c69864ffdfe347bf3b050cb931f53067a6/rules/windows/privilege_escalation_krbrelayup_suspicious_logon.toml), [https://github.com/elastic/detection-rules/blob/main/rules/windows/credential_access_kerberoasting_unusual_process.toml](https://github.com/elastic/detection-rules/blob/main/rules/windows/credential_access_kerberoasting_unusual_process.toml)
[^3]: [https://twitter.com/wdormann/status/1518999885550440451](https://twitter.com/wdormann/status/1518999885550440451)