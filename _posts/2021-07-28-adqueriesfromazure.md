---
title: Execute AD queries protected by modern authentication from Azure without VPN
date: 2021-07-28 00:00
categories: [powershell]
tags: [identity,AAD, Powershell]
---

# Introduction

I have this question in mind since several months… What would I do if I wanted to run Active Directory (AD) queries from Azure without a VPN gateway? Basically, I wanted to do a quick Proof Of Concept (POC) with the [hybrid connector (HBC) for app service](https://docs.microsoft.com/en-us/azure/app-service/app-service-hybrid-connections). PS Remoting with HBC over azure function (Windows Operating system) [is not new](https://docs.microsoft.com/en-us/azure/azure-functions/functions-hybrid-powershell). But because I like challenges (or because I’m stupid…) I’ve decided to test with a app service hosted on a Linux OS since the feature is [GA](https://azure.microsoft.com/en-us/updates/azure-app-service-hybrid-connections-for-linux-apps-is-now-available). In other words, the plan will be to use a **Powershell Linux container** hosted on an app service, enable the **hybrid connector**, do a **PS remoting session** on an on-premises domain joined Windows box to finally reach Active Directory… What could go wrong?

You can find the code [here](https://github.com/SCOMnewbie/psoauth2/tree/main/Examples/Powershell/10-WebAPI-HybridConnector).

The idea looks like this:

![idea](/assets/img/2021-07-28/idea.png)

During this article, we will:
* implement the solution
* then show it in action 
* next explain why I’ve spent far more hours than I’ve thought for this POC
* conclude with cost explanation because this POC is not free

# Implementation

## Step1 – Prepare the remote machine (on premises)

This machine **must be domain joined**. According to the hybrid connector documentation, the machine has to be at least a Windows 2012, I’ve personally used a Windows 10 machine.

Let’s now enable the feature **Powershell AD management tools**.

Next, we will have to create a **winrm HTTPS listener** which allow **basic authentication** (**winrm set winrm/config/service/auth '@{Basic="true"}'**). There is a lot of documentation around this topic. I think I’ve followed this [one](https://cloudblogcenter.com/2020/05/23/using-azure-hybrid-connections-in-azure-functions). After several fails with NTLM, I’ve decided to go the basic authentication way over HTTPS. It works with NTLM when you use Azure function, I guess it should work with Linux too. But I never succeed to make it work.

{% include note.html content="I’ve used a self-signed certificate during this POC." %}

Next, we will create a **new local admin account**. Local admin because this is the quickest way to use PS remoting sessions. I’m sure there is tutorials to allow non-admin users to connect.

That’s it for now, we will come back later on this machine.

## Step2 – Configure the Azure infrastructure

Nothing complicated here, I’ve created a **Linux Docker container S1 SKU App service** and a **basic ACR** (container registry) in the same region.

Then, I’ve added 2 configurations in the web app’s app settings:

![appsettings](/assets/img/2021-07-28/appsettings.png)

This represents the credentials (local creds previously created) that our web app will use to **instantiate the remote session to the on prem machine**. Doing this avoid hardcoding login/password into the script. As usual, a more secure approach is to use a Keyvault and a reference link.

Once the App Service is created, go to network and configure the hybrid connector once done (follow the doc), don’t forget to **download the agent** (Download connection manager) we will install it just after.

![hbc](/assets/img/2021-07-28/hbc.png)

## Step3 – Back to the remote machine (on premises)

It’s time to **install the hybrid agent** we’ve previously downloaded. Simple to install, next, next sign-ins with your AAD account and you should be connected:

![hbcmanager](/assets/img/2021-07-28/hbcmanager.png)

The last step is to **store your corporate password encrypted with a specific key** (more info later). Here the code you can use on the device to protect your password: 

```Powershell
#Generate a random 256 bit AES key
$Key = New-Object Byte[] 32
[Security.Cryptography.RNGCryptoServiceProvider]::Create().GetBytes($Key)
#Flat the key and put it into the clipboard
$key | %{$str += "$_," } ; $str.Substring(0,$str.Length -1) | clip
#Generate the encrypted password file with the previously created key and store it into a specific path (change with your path).
"your password" | ConvertTo-SecureString -AsPlainText -Force | ConvertFrom-SecureString -Key $key | Out-File 'C:\TEMP\secret.txt'

# Create a PSCredential from the encrypted file with the key (Re-hydrate)
# $OrgCreds = New-Object -TypeName System.Management.Automation.PSCredential -ArgumentList 'Corp SamAccountName', (Get-Content 'C:\TEMP\secret.txt' | ConvertTo-SecureString -Key $key)
# Get-aduser "blabla" -credential $Orgcreds

```

**Keep the $Key in your clipboard** you will need it later. We’re done with this machine, let’s built our API hosted in the cloud!

## Step4 - Build our API

For simplicity, I’ve used the same AAD application I’ve created in the previous article where I have app roles defined (read.access, write.access, admin.access) in my app registration. I will use almost the same code as before for the startpode.ps1.  Long story short, I’ve kept the Pode Middleware configuration, removed the previously created routes and added a route called '/api/readfromhybrid' where the $ReadAccessRequired middleware make sure your token is valid and your user authorized to consume this API. If need more information, you can have more info [here](https://scomnewbie.github.io/posts/apiwithpode/). 

Changes in startpode.ps1 compared to last article:

* On top of the file, we now import a new module called **PSWSMAN** and we execute the command **Disable-WSManCertVerification -All**. As the name suggest and because we use self-signed certs, we will have to disable both CA/CN checks.
* Line 12/13 we just fetch the local creds exposed in the app settings.
* I removed all previous routes and added one '/api/readfromhybrid'.
  
Let’s explain the code of the route:

```Powershell
$SamAccountName = $WebEvent.Query['samaccountname']

$Script = {
    param($SamAccountName)
    # AES key that has been generated from the target machine (hardcoded for this POC, replace with your clipboard value)
    [Byte[]]$Key = @(214,234,128,193,229,128,66,17,101,143,108,66,153,101,6,206,34,149,225,88,255,29,161,70,47,47,99,224,230,46,4,73)
    # Rehydrate the PSCredential with hardcoded myorgaccount (use your AD service account here) and the encrypted password file located on the remote host
    $OrgCreds = New-Object -TypeName System.Management.Automation.PSCredential -ArgumentList 'myorgaccount', (Get-Content 'C:\TEMP\secret.txt' | ConvertTo-SecureString -Key $Key)
    #Then do a simple get AD user with a query parameter
    get-aduser $SamAccountName -Credential $OrgCreds
}

# Use the local remote local creds to generate a pssession and send the scriptblock script trough the HTTPS listener
$creds = [system.management.automation.pscredential]::new($Using:UserName,$Using:Password)
$Data = invoke-command -ComputerName "<Hybrid Endpoint Name (machine name)>" -Authentication Basic -Credential $creds -Port 5986 -UseSSL -ScriptBlock $Script -ArgumentList $SamAccountName

# Should have a better token validation here
Write-PodeJsonResponse -Value $Data

```

* Line 172 - This API is waiting for a **query parameter** called samaccountname. This parameter will be passed to the remote machine.
* Line 174 To 182 - We define a **scriptblock that will be executed on the remote machine**. Instead of hardcoding the corporate credential into the container (possible security issue), I’ve hardcoded the previously created AES key (the one you should have into your clipboard). Of course, you can use KV/App settings again, I know that’s not a perfect solution but after all the pain (see later), I’ve decided to take this shortcut. Line 179, I simply re-hydrate the PSCredential on the remote machine and then do a get-ADUser…
* Line 184-185 - We “simply” do an **invoke-command** with the local remote machine credentials and pass both the scriptblock and the HTTP query parameter. Because our container is not domain joined, I’ve decided to use the basic authentication (therefore you have to enable it in step1) over HTTPS. This is where the **hybrid connector Blackmagic** happen! 
* Line 189 - **return the data received over the 443 TCP port** from the remote machine (I’m still amazed by this feature), of course error handling should be implemented here.

# In action

Create your image from the dockerfile

```Powershell
docker build -t funwithhybrid .
```

Then tag your container to match your ACR name

```Powershell
Docker tag funwithhybrid:latest "<your ACR name>.azurecr.io/funwithhybrid:latest"
```

Then sign-in to your ACR

```Powershell
az acr login -n 'your ACR name'
```

And finally push your container to your ACR

```Powershell
docker push 'your ACR name'.azurecr.io/funwithhybrid:latest
```

{% include note.html content="Bad note for Microsoft regarding the admins credential you HAVE TO enable them on your ACR if you plan to use app service for containers. I hope Microsoft will enable the managed Identity feature soon." %}

Now you have your container in the ACR, configure your app service to use your image:

![webapp](/assets/img/2021-07-28/webapp.png)

Once your site is up and running, you can take your preferred API client in my case Thunder client VS Code extension and call your API **https://<Your app service name>.azurewebsites.net/api/readfromhybrid?samaccountname=<your SamAccountName>**. 

Because it’s **AAD protected** (this part is covered in the multiple articles that I’ve published already), if I try with an expired JWT token, I receive:

![expired](/assets/img/2021-07-28/expired.png)

If now I use a valid token:

![valid](/assets/img/2021-07-28/validtoken.png)

{% include note.html content="For this POC, my remote endpoint is a remote machine connected to a domain controller via VPN because why not… In other words, the flow goes from the web app to the hybrid connector (Service Bus) to my laptop connected to a remote DC trough VPN. In addition, I have several apps running already on this cheap app service plan. This is why (I hope) it’s not uncommon to wait more than 10 seconds … But it works! " %}

I can run AD queries from a Linux container protected with Azure Active Directory without VPN. **Challenge completed**!

Now let’s explain why I’ve spent more than 8 hours to make it works…

# The pains

## Pain1 – PS Remoting over WSMAN on Linux

The **PS Remoting over WSWAN on a Linux** machine was a big pain. PS Team recommends using SSH today, but I can’t use SSH in this case.
Quickly I’ve found this [article](https://www.bloggingforlogging.com/2020/08/21/wacky-wsman-on-linux/) where **Jordan Borean** explains the reason of my multiple fails... A big thank you to him for both this article who popularizes this complicated topic and for the [PSWSMAN module](https://github.com/jborean93/omi).

## Pain2 – Re-hydrate the credentials on the remote machine

My first try was to use Azure ARC and [my function](https://github.com/SCOMnewbie/Azure/blob/master/Identity-AAD/New-ARCAccessTokenMSI.ps1) to retrieve credentials from Keyvault with the local MSI endpoint. Sadly the ARC agent is running under the network context and not the local user one. In other words, I never succeed to fetch an access token. 
Then my second idea was to use a ConvertTo-Securestring with CLIXML commands, sadly the Hybrid agent is running in a different context again ... Therefore, I’ve decided to generate my own AES key and then use it to encrypt/decrypt a file. User context doesn’t matter in this case.

## Pain3 – Lack of debugging

Between Docker, Pode, Azure web app, the Hybrid Connector and finally the remote machine itself, trust me that I’ve created a LLLOOOTTT of containers before the solution be a success … The lack of logs was real.

# Cost

This POC is not free of charge.

## Hybrid Connector

It cost me 0.50 € for a week. When you look at the doc, you can see:

![price1](/assets/img/2021-07-28/price1.png)

It’s still far cheaper than even the smaller VPN GW you can configure.

## Container Registry

It cost me 0,7 € and I’ve used the basic SKU without egress or build time. From the doc you can see:

![price2](/assets/img/2021-07-28/price2.PNG)

## App Service Plan

To use the hybrid connector, you **can’t use the consumption tier** but instead the Standard and above. I’ve decided to use the S1 (cheapest in my case) which cost me something like 10 € for the week. When you check the doc, you can see:

![price3](/assets/img/2021-07-28/price3.png)

I personally run several apps on this app service plan so the cost is not “crazy”. 

# Conclusion

As a big fan of JEA (Just Enough Admin) endpoints, where you can control who can access to what (among other things) but **only from an AD domain**, now I know I can bring **Azure AD and modern authentication** to the picture too. When you think about it, controlling an on-premises PS endpoint from Azure is cool don’t you think? If you plan for intensive requests on AD, I guess the VPN should be a better option. At least it can be an interesting test to do.

Anyway, before closing this article, I wanted to thank again both **Jordan Borean** for his amazing work on the PSWSMAN module and **Matthew Kelly** for his wonderful Pode module.

See you for the next one.
