---
title: Web sites cannot be started unless WAS and the World Wide Web Publishing Service are running
date: 2017-10-31T12:31:00+01:00
author: Wolfgang Ofner
categories: [Miscellaneous]
tags: [Continous Deployment, IIS, PowerShell]
---
Today I was working on a new Continuous Development task for one of my Projects. I was testing if it works and got an error message. After Fixing the error I rerun the deployment and got an error in an earlier step where I use a PowerShell script to stop the Default WebAppPool. It was strange because the script worked fine before. I found out that the problem was that the World Wide Web Publishing Service wasn&#8217;t running. In this post, I will tell you how to fix this problem.

### The error

During the CD I got the following error while executing the PowerShell script: **[error]Command execution stopped because the preference variable &#8220;ErrorActionPreference&#8221; or common parameter is set to Stop: Access is denied. (Exception from HRESULT: 0x80070005 (E_ACCESSDENIED))**. The script only contains two lines of code:

Import-Module WebAdministration  
Stop-WebSite &#8216;Default Web Site&#8217;

After I encountered this error I connected to the Server and saw that the WebAppPool (IIS &#8211;> Sites &#8211;> Default Web Site) wasn&#8217;t running. I clicked on Start and got the following error message:

[<img loading="lazy" class="aligncenter size-full wp-image-293" src="/assets/img/posts/2017/10/Default-Web-Site-starten.png" alt="Default Web Site error" width="409" height="148" />](/assets/img/posts/2017/10/Default-Web-Site-starten.png)

### Starting the World Wide Web Publishing Service (W3SVC)

After that, I went to the **Services** (Computer&#8211;> right click on manage &#8211;> Configuration &#8211;> Services or directly Services in the Start menu) and saw that the **World Wide Web Publishing Service** wasn&#8217;t running. Right-click the service and select **Start**. Then the service should be running. You can see that on the left column where you should have the Options Stop and Restart the service.

[<img loading="lazy" class="aligncenter wp-image-294" src="/assets/img/posts/2017/10/World-Wide-Web-Publishing-Service.-oder-einfach-auf-Services-klicken.png" alt="World Wide Web Publishing Service" width="700" height="504" />](/assets/img/posts/2017/10/World-Wide-Web-Publishing-Service.-oder-einfach-auf-Services-klicken.png)

After I started the services manually, the PowerShell script could stop the WebAppPool automatically.

### Further reading

For more information see <a href="https://technet.microsoft.com/en-us/library/aa997600%28v=exchg.80%29.aspx" target="_blank" rel="noopener">Technet</a>.