---
title: Passwordless deployment from anywhere with Azure ARC
date: 2021-02-19 00:00
categories: [identity]
tags: [identity,Powershell,ARC]
---

# Introduction

Instead of re-inventing the wheel, I will simply paste this definition that I've found in the Azure ARC overview. "Azure Arc enables you to manage your entire environment, with a single pane of glass, by projecting your existing resources into Azure Resource Manager. **You can now manage virtual machines, Kubernetes clusters, and databases as if they are running in Azure**." ARC is not a simple product, but a set of functionalities where you can pick and choose what you’re interested in. Today, our focus will be on **ARC for servers which is GA**.

Following a video @AzureAndChill and @ChristosMatskas published on their [425](https://www.youtube.com/channel/UCIPMDupgTRsJY5sxcdBEtCg) channel, I've discovered that the **ARC agent exposes a local identity endpoint** you can play with to get an access token like you do in Azure directly with a VM for example.

Now what does it mean?

- No more chicken egg problem where you must have a secret to access your vault which hosts all your other secrets.
- You can do any actions on any destinations (On-premises, AWS,GCP,…) without commit a single password in your repo (dduuhh) or the pipeline provider (Github, AzDO, Gitlab,…). The counter part is that you must use today a self-hosted agent. I hope tomorrow GH/AzDO will provide this feature with their runners directly.
- Your developers do not need to know any password. You provide only references to password, not the password itself.
- Even if it’s not officially supported, I know ARC is working well on Windows 10. It’s cool to access a Keyvault from a local dev machine without providing a password in a config file.
- This part of ARC is free of charge. Only the guest policy usage is not free, but we don’t need it here.

![01](/assets/img/2021-02-19/01.png)

{% include note.html content="If you want to go deeper in the ARC subject, Thomas Maurer (mister hybrid cloud) published a nice [article](https://www.thomasmaurer.ch/2020/12/get-started-with-the-azure-arc-jumpstart/) which summarize what ARC is." %} 
I don’t know for you, but I’m excited to see what we will do in this demo, let’s start!

# Context and Managed Identity

For this article, I've **enrolled my local dev machine to ARC**, there is a lot of good documentation, I won't cover this part. Once the onboarding done, you should now have a **new Service Principal** (Enterprise Application) which represents the identity of our dev machine or our self-hosted runner in our pipeline. My plan in this article will be to see what we can do from our local machine compared to what we can do from Azure directly with [managed identities]( https://docs.microsoft.com/en-us/azure/active-directory/managed-identities-azure-resources/overview).
To help us in this article, I've built a small **function to generate access tokens for various audiences** like Keyvault, ARM and so on... You can find the script [here](https://github.com/SCOMnewbie/Azure/blob/master/Identity-AAD/New-ARCAccessTokenMSI.ps1).

{% include warning.html content="I am sorry, since few days, Windows is considering my invoke-expression or iex as a threat and **Defender is blocking it** ... For now, don't hesitate to review the code, and copy/paste or dot source the functions in your console to load it in memory." %}

```powershell
#Load our function New-ARCAccessTokenMSI in memory.
# UPDATE: Windows consider today this script as a threat ... Let's copy paste it in your console instead.
iex ((New-Object System.Net.WebClient).DownloadString('https://raw.githubusercontent.com/SCOMnewbie/Azure/master/Identity-AAD/New-ARCAccessTokenMSI.ps1'))
# Generate an access token for Keyvault
$KVToken = New-ARCAccessTokenMSI -Audience Keyvault

```

Now you should have received an access token for the **Keyvault audience**. You can use [JWT.ms](https://JWT.ms) to verify/check few interesting things:

- The **aud** property should be: https://vault.azure.net. It's normal, as explain in previous article, the audience is an important part of the modern auth world. Here it's not an access token that we will be able to use to access graph or ARM, but Keyvault endpoint only.
- The **appId** property, which is as the name suggest, the Id of our service principal. You should copy it somewhere for a later usage in this demo.
- The **appidacr** property which explains we've used a certificate to authenticate. Pretty cool that we use cert based auth without any action (ok, except install the agent) don’t you think?
- The **ver** property which is equal to 1.0. It means ARC today cannot use the 2.0 Microsoft Identity endpoint. It's not a big deal, but it's interesting to be aware.
  
Now we have our token, let's play with Keyvault!

# Keyvault

To simplify this demo, I've created another function which can fetch a secret from Keyvault using REST call and access token. As before, you can verify the wrapper function [here](https://github.com/SCOMnewbie/Azure/blob/master/Identity-AAD/Get-KeyvaultSecret.ps1).

{% include note.html content="Keyvault does not provide free SKU. It will be cheap, but not free" %}

In this demo, we will get a secret value from KV, so what do we need? A Keyvault (thx captain obvious!). I won't cover this part, but here what we need for this demo:

- Create a Keyvault with <span style="color:red">**Role based access policies enabled**</span>. For this demo let's called it MyKeyvault
- Create a Secret and it's value called MySuperSecret for this demo.

```powershell
#In the same console, let's load the wrapper function
iex ((New-Object System.Net.WebClient).DownloadString('https://raw.githubusercontent.com/SCOMnewbie/Azure/master/Identity-AAD/Get-KeyvaultSecret.ps1'))
#Let's now grob our secret from anywhere (On-premises, AWS, GCP, ...) without ANY password!
Get-KeyvaultSecretValue -KeyVaultName "MyKeyvault" -SecretName "MySuperSecret" -AccessToken $KVToken
```

And voilà, **you've fetched a secret value from Keyvault** from a machine not necessarily hosted on Azure **without any password**. You remember the famous chicken-egg problem? It’s just disappeared 😊. **Of course, it means your machine must be proper secured to avoid malicious actions**. Now it means with ARC, you can use **any self-hosted agent** (AzDo, Github, Gitlab,…) on **Windows/Linux** and **get rid of ANY credentials** that you have to provide usually in those providers. Even your devs has no idea of the passwords and can’t access it in fact. They simply commit their changes, and the agent take them and deploy the modification. You don’t have to rotate the keys every time someone leave the team. This is so cool!

{% include important.html content="Keep it mind that people locally logged on the ARC agent can fetch the secret value too, don't forget to roll all your secret once the maintenance is done. The maintenance of your agents become the weakest link." %}

# Graph API

Using ARC, we've seen a new service principal is now created without any App registration. In other words, we can't define which scope to request when we ask an access token for the graph API endpoint. Does it mean we're done? Let's see...

For this demo, we will simply create an AAD groups and add our serice principal owner on it. Let's create the AAD groups and don't forget to assign our SP as owner of the group:

![02](/assets/img/2021-02-19/02.png)

Once created, take the ObjectId of our group. In my case, it's 906212d7-186d-4e30-ad65-9fdb3b1efad0.

```powershell
#In the same console, let's generate a Graph token
$GraphToken = New-ARCAccessTokenMSI -Audience GraphAPI
#Define variable with my group objectId
$GroupId = "906212d7-186d-4e30-ad65-9fdb3b1efad0"
# Create Graph URI
$GraphURI = "https://graph.microsoft.com/v1.0/groups/$GroupId"
# Call now graph API
Invoke-RestMethod -ContentType 'application/json' -Headers @{'Authorization' = "Bearer $GraphToken"} -Uri $GraphURI

```

And as you can see, you can take actions on **objects you're the owner**!

 Let's try to remove our group for fun...

```powershell
Invoke-RestMethod -ContentType 'application/json' -Headers @{'Authorization' = "Bearer $GraphToken"} -Uri $WorkingGrapURI -Method delete

```

And it's done...

Let's go a deeper and see if we can read groups/users with this MSI. By default, and it's normal, you will receive an access deny if you try to get the URL "https://graph.microsoft.com/v1.0/groups/". But what if we grant our MSI permissions!

```powershell
# I will do this in cloud shell, to avoid having to install the AzureAD module but you can do it locally too. Once logged
Connect-AzureAD # Need Global admin right because we will play with admin consent and Graph API
# Get Graph app properties. This is a well known Id
$GraphApp = Get-AzureADServicePrincipal -Filter "AppId eq '00000003-0000-0000-c000-000000000000'"
#get Role to read group objects
$GroupReadPermission = $graphApp.AppRoles | where-Object {$_.Value -eq "Group.Read.All"}
#get Role to read user objects
$UserReadPermission = $graphApp.AppRoles | where-Object {$_.Value -eq "User.Read.All"}
#use the MSI objectId. You ca have it from the portal (Enterprise App > All application)
$MSI = Get-AzureADServicePrincipal -ObjectId '9597393d-ff2a-44ee-9c5e-ebe327582873'
# Grant read group/user permission
New-AzureADServiceAppRoleAssignment -Id $GroupReadPermission.Id -ObjectId $MSI.ObjectId -PrincipalId $MSI.ObjectId -ResourceId $graphApp.ObjectId
New-AzureADServiceAppRoleAssignment -Id $UserReadPermission.Id -ObjectId $MSI.ObjectId -PrincipalId $MSI.ObjectId -ResourceId $graphApp.ObjectId

#Now regenerate a new token
$GraphToken = New-ARCAccessTokenMSI -Audience GraphAPI
$ReadGroupsURI = "https://graph.microsoft.com/v1.0/groups/"
$ReadUsersURI = "https://graph.microsoft.com/v1.0/users/"

#And do the call
Invoke-RestMethod -ContentType 'application/json' -Headers @{'Authorization' = "Bearer $GraphToken"} -Uri $ReadGroupsURI
Invoke-RestMethod -ContentType 'application/json' -Headers @{'Authorization' = "Bearer $GraphToken"} -Uri $ReadUsersURI

```

You Service Principal should look like this:

![03](/assets/img/2021-02-19/03.png)

And ttaaddaa! So let's do a step back and try to understand what does it mean? It means you can do **"Infrastructure As Code" with AAD without password** again. You can manage groups, users, app assignment, or anything graph related from Git without creds...

# Storage Account

Imagine now you need to interact with table, blobs or something else. Do you think we can read our blob without keys/SAS token? Let's see!

For this demo, I've created a storage account (testarcfanf), a container (arc) and blob called hellofromarc.txt. Then, I've gave our **MSI the RBAC data reader on this specific container**.

```powershell
# Generate a new access token for storage audience
$StorageToken = New-ARCAccessTokenMSI -Audience StorageAccount
# Let's hardcode our blob URL
$blobURI = "https://testarcfanf.blob.core.windows.net/arc/hellofromarc.txt" #Use your value with your storage account/container/blob value
# Create our storage account header
$headers = @{
	'Authorization' = "Bearer $StorageToken"
	'x-ms-version' = "2020-04-08"
}
# let's access our blob file
Invoke-RestMethod -Uri $blobURI -Headers $headers -Method Get
#In my case, I will see
Hello from Azure ARC!

```

Victory again!

# Resource manager

With the previous demos, we've had to play with REST directly which is not necessarily convenient. Do you think we can **play with our well-known Az module** too?
For this demo, I've decided to grant my **MSI reader access at the subscription level**. Another thing we will need is to get the AppId of our Service Principal. You can decide to open the Enterprise Application tile or for fun, in [Graph Explorer](https://aka.ms/ge), you can use this query: https://graph.microsoft.com/v1.0/servicePrincipals?$filter=DisplayName eq 'your computer name'&$select=Appid and grab it from there.

```powershell
# Define variables
$SubId = '<Your SubscriptionId>'
$AccountId = '<Your AppId (MSI Service Principal)>'

# Try to connect like we do usually in Azure
Connect-AzAccount -MSI 
#Generate an error. I guess it's because MS didn't include ARC in the equation yet. I hope it will come one day. 
#Instead let's generate a new access token, this time for ARM
$ARMToken = New-ARCAccessTokenMSI -Audience ARM

#Connect as an MSI using the AT instead
Connect-AzAccount -AccessToken $ARMToken -AccountId $AccountId -Subscription $SubId

#And you're in as your MSI! Let's now list our resource groups
$RGs = Invoke-AzRest -Path "/subscriptions/$SubId/resourcegroups?api-version=2020-06-01" -Method Get
# and here your RGs
$RGs.Content | ConvertFrom-Json | select -ExpandProperty value

# and is we prefer the regular Az commands, we can use instead this for the same result
Get-AzResourceGroup

```

It means that we can from anywhere (not only Azure), do automation with Powershell (~~Pretty sure we can with CLI too~~) or deploy resources in Azure without secrets again ...

{% include important.html content="One thing to keep in mind is that you **don't receive any refresh token**. You only have today an access token which means **the task you plan to do has to be done in the hour** otherwise you will get authorization deny." %}

# Conclusion

During this article, we've just scratched the Azure ARC surface. But in this specific case, we can now do almost anything without any password, FOR FREE, which is awesome, thank you Microsoft for that.

**Here few take away** from this article:

- If you're ok with the idea to manage self-hosted agent, ARC for servers is a must have to improve your security posture if you're not in Azure. For a VM already hosted in Azure, it doesn't really make sense, just enabled the system managed identity instead.
- Now if you add a Keyvault to our ARC agent, you can now deploy anything, anywhere without providing any credentials to your self-hosted agent(environment variable), your pipeline provider (GH/Gitlab/AzDo/..) and even your devs! The only "weak" point is the person who has access to the machine which has to be tracked and monitored.
- ARC is not designed for workstations (Intune instead), but it's working well on my Win10. I have no idea if we can enable ARC for Linux client OS too.

**Side note**: I'm convinced that Hybrid is the way to go in the future (for big companies). Having a mix of on-prem/multiple public cloud providers infrastructure gives you an open mindset when you're in a project planning phase. The dark side of this is how the hell you keep this infrastructure secured! Today with IAAS, containers and K8s clusters spread everywhere, it becomes challenging to keep the security score high. This is where ARC starts to shine, I don't think the other public cloud providers have this kind of offer and I will keep an eye on this promising suit of tools!

Anyway, I've enjoyed playing with MSI for this article. See you in the next one.

# References

[Azure ARC Overview](https://docs.microsoft.com/en-us/azure/azure-arc/)

[Jumpstart project](https://www.thomasmaurer.ch/2020/12/get-started-with-the-azure-arc-jumpstart/)

[425 Show](https://www.youtube.com/channel/UCIPMDupgTRsJY5sxcdBEtCg)
