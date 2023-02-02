---
layout: post
title: Windows 11 Ransomware protection blocks the install of some PowerShell modules from being installed
date: 2021-08-31 00:00:00 00:00
comments: true
author: ebmarquez
categories:
 - productivity
tags: 
 - development
 - productivity
 - windows11
---


While windows 11 is not yet released I came across a frustrating issue when I was attempting to install a PowerShell module in PS 7.1.4. If you have the Windows Security feature Ransomware protection enabled. This will block apps and services from making some changes to protected folders in OneDrive. In my environment I have OneDrive operating as my Documents folder. Within the Documents folder is where the `$ENV:PSModulePath` stores my modules (`Documents\PowerShell\Modules`). When I had PWSH install a new module it was attempting to write to this location which is now protected.

When installing new modules I always launch a new PWSH window as an elevated session. Even with elevated privileges this was still blocked by the ransomware protection.  When I thing about it, it should be protected from pwsh. The error was complaining about session not having Administrator rights.

```powershell
PS C:\Windows\System32> Install-Module Az -AllowClobber -Force
Install-Package: C:\program files\powershell\7\Modules\PowerShellGet\PSModule.psm1:9711
Line |
9711 |  … talledPackages = PackageManagement\Install-Package @PSBoundParameters
     |                     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
     | Administrator rights are required to install or update. Log on to the computer
     | with an account that has Administrator rights, and then try again, or install by
     | adding "-Scope CurrentUser" to your command. You can also try running the Windows
     | PowerShell session with elevated rights (Run as Administrator).
```

At the time I was doing this, I found the error puzzling.  I didn't yet notice the new popup from security alerting me of a app being blocked.  When the ransomware protection kicks in a new notification popup will appear and ask for permission. To allow the new module to be installed, I needed to allow this action by accepting it in the protection history. After allowing it I can re-run the same `Install-Module` action to install the module. 
![pwsh in protect history](/assets/img/Windows-11-Ransomeware-Protection-Blocks-Install-of-PowerShell-Modules/pwsh-protection-history.png)

One thing to note about doing this. The pwsh.exe will be put listed in the "Allow an app though controlled folder access". I plan to remove pwsh.exe from this trusted app list.  Now that I know pwsh is blocked from adding new modules to the Modules folder, I can enable it when I update my modules.


