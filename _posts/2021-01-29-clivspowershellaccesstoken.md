---
title: Compare get-access-token and Get-AzAccessToken
date: 2021-01-29 00:00
categories: [identity]
tags: [identity,Powershell,CLI]
---

# Introduction

Before going deeper in identity topics, I just wanted to play with this “new” Get-AzAccessToken. The Azure Powershell community was “upset” because this command exists since a long time in CLI. This cmdlet belongs to the Az.Accounts module and has been released in November 2020. As I’ve said in the previous article, my goal is to explain modern authentications and sadly this article won’t help, therefore I prefer to start with it. In this article, I will expose few ways to use the two commands (CLI and Ps) and give some general tips around identity.

# Azure CLI

## Basic usage

Once you've [installed the CLI](https://docs.microsoft.com/fr-fr/cli/azure/install-azure-cli), it's time to play a little with it:

{% include note.html content="Don't forget you can keep your **CLI up to date automatically** by simply typing **az config set auto-upgrade.enable=yes**. By default this setting is set to false." %}

Let's generate an access token for Microsoft graph API and put it in clipboard

```powershell
#Once connected
az login
# Let's generate a token for this context
az account get-access-token --resource https://graph.microsoft.com | ConvertFrom-Json | select -ExpandProperty accessToken | clip
```

Let's now paste this JWT token into [jwt.ms](https://jwt.ms/), go into the **claims** tab and check the **appid** property. You should see:

![CLI VS Powershell 02](/assets/img/2021-01-29/02.png)

But wait we didn't specified any clientId (AppId)? Remember last article? To establish a connection, you need an AppId, a redirectURL and scopes. Here we just type a command once logged in (az login). In fact this command is a wrapper (imagine like a Powershell function) which abstract a lot of parameters. Now this appId is in fact a well know clientId that Microsoft expose from their tenant. We will disscuss about how the app registration/Enterprise application concept if working.

{% include note.html content="Powershell has also his own AppId. I use it in this [piece of code](https://github.com/SCOMnewbie/Azure/tree/master/DumpAADFromGuestUser) where I explain how to dump an entiere tenant from a guest account if AAD is define with default security values." %}

If now we take a look at the scp property, we can see that **scopes** as already been defined:

![CLI VS Powershell 01](/assets/img/2021-01-29/01.png)

We can see that without specifying any resources (scopes) we received a lot of permissions! As you can imagine, only delegated permissions are "granted". Funny fact 1: Microsoft graph API do not expose user_impersonation scope compares to most of the other MS APIs. Funny fact 2: Check your AAD you won't see an Enterprise app called CLI or Powershell within your tenant where we should but you have graph explorer 😊.

Let's play and see what we can do with it! We can start with simple one, the /me endpoint:

```powershell
#Use a basic user account to log in (non admin)
az login
#Get a token for the audience https://graph.microsoft.com
$token = az account get-access-token --resource https://graph.microsoft.com | ConvertFrom-Json | select -ExpandProperty accessToken
#Generate a header
$Headers = @{
    'Authorization' = $token
    "Content-Type"  = 'application/json'
}
$uri = "https://graph.microsoft.com/v1.0/me"
Invoke-RestMethod -Method get -Uri $uri -Headers $Headers
```

It's working and it's normal because we have the **User.ReadWrite.All permission** in our **scope**. Here we're simply say show me my user context. Now imagine you want to restrict the portal access to the basic users. You can easily do this from the **Azure Active Directory** tile and then **User Settings**. Let's do it, and once done, now basic user can’t read applications (Enterprise and App registration) anymore.

You can find more information on how to play with applications REST api [here](https://docs.microsoft.com/en-us/graph/api/application-list?view=graph-rest-1.0&tabs=http). if you check the permissions tab, we can see:

![CLI VS Powershell 03](/assets/img/2021-01-29/03.png)

First, event if the application scope is not explicitely defined in our "allowed" scope, we can see from the screenshot that the /applications endpoint allow you to query the url if you received the Directory.AccessAsUser.All scope which is the case here.
Second, we can also see this scope is considered as **most privileged**.

This is where I consider Azure CLI or Powershell (see below) not helpful to understand modern authentications. There is so much abstraction done and do much "power" given to a basic user that I consider you lose the granularity modern authentication brings. But don't get me wrong, I also totaly understand why Microsoft has decided to take those shortcuts (improve user experience).

Now the funny part, remember that **our basic user can't access the application anymore from the admin portal** since we've changed the user settings. But if you regenerate a token, guess what? It’s working because CLI give you the Directory.AccessAsUser.All scope…

```powershell
#Use a basic user account to log in (non admin)
$token = az account get-access-token --resource https://graph.microsoft.com | ConvertFrom-Json | select -ExpandProperty accessToken
$Headers = @{
    'Authorization' = $token
    "Content-Type"  = 'application/json'
}
$uri = "https://graph.microsoft.com/v1.0/applications"
Invoke-RestMethod -Method get -Uri $uri -Headers $Headers
```

{% include tip.html content="Instead of generate a token, construct the request and send it, CLI offer a shortcut. **az rest --uri https://graph.microsoft.com/v1.0/applications**. CLI does exactly the same thing but in a oneliner." %}

In this first part, I wanted to explain that Microsoft did a good job to abstract the authentications complexity, but don't forget those scopes are pretty permission wide. Here we've requested an access token (AT) for the https://graph.microsoft.com audience but the CLI give you the possibility to generate an AT for the ARM fabric (https://manage.azure.com), or the old ms'graph endpoint, Keyvault and so on...

## Advanced usage

If now we want to request only /applications endpoint and nothing else (why it's another story, but let's imagine lol). We will have to create an app registration (clientID/client secret), allow the specific scope and test! Let's start by creating the App registration:

```powershell
# Connect to CLI as at least application administrator AAD role
#Create my app, a secret without role assignment
$MyApp = az ad sp create-for-rbac -n "getonlyapplications" --skip-assignment
```

For simplicity, let's assign applications permission read only from the portal. Under your newly created app registration, you should now configure this:

![CLI VS Powershell 04](/assets/img/2021-01-29/04.png)

{% include important.html content="Make sure you choose **delegated** and **not application** permission!" %}

Let's now connect as our SP and query few things:

```powershell
# Convert $Myapps to a usable object thx Powershell shell
$Myapp = $myapp | ConvertFrom-Json
#Connect as the SP
az login --service-principal --username $MyApp.appId --password $MyApp.password --tenant $MyApp.tenant --allow-no-subscriptions
#We can now use the shortcut to contact our applications endpoint
$Apps = az rest --uri "https://graph.microsoft.com/v1.0/applications" | ConvertFrom-Json | select -ExpandProperty value

#Here you should receive a forbidden message an it's normal
```

So what is happening? We will go deeper in another article, but we start to hit the limit of the CLI with custom api. If you check the **expose an api** menu, you should see:

![CLI VS Powershell 05](/assets/img/2021-01-29/05.png)

**IMPORTANT CONCEPT:**

https://<your app>/user_impersonation means that you allow this application to execute **delegated** actions on behalf of the user context. delegated means: **Only actions that user can do under his user context will be allowed through this clientId**. In other words, this is how you **make sure a user can break only things he owns and nothing else**. For example, a user will be able to add/remove members from the groups where he **has the owner privilege**. But here, at least from my knowledge, you can't do this from the CLI with the **basic commands** like **az login** and get-access-token (we will later, but with code behind). If you remember all the back and forth between your app and AAD (from the previous article), in this case your native app is the Azure CLI and you don't have a way to use a redirectURI for example.

Ok so let's "fix" our current issue. We know that we can't do impersonation. Let's stick **to application instead of delegated permission**.

![CLI VS Powershell 06](/assets/img/2021-01-29/06.png)

And now we will be able to execute our action this time not on behalf of a user but as a **daemon/Service Principal**. **Application permission** can be dangerous when you don't understand what you're doing. You can assign high permissions and you won't be able to track who do what when which can be a security risk. This is global admins mandate to understand this part.

```powershell
#Reconnect to generate a new token with the new permission
az login --service-principal --username $MyApp.appId --password $MyApp.password --tenant $MyApp.tenant --allow-no-subscriptions
#can we use the same URL this time?
$Apps = az rest --uri "https://graph.microsoft.com/v1.0/applications" | ConvertFrom-Json | select -ExpandProperty value
# Yes! Let's now generate an AT and do the same thing
$token = az account get-access-token --resource https://graph.microsoft.com/ | ConvertFrom-Json | select -ExpandProperty accessToken
$Headers = @{
    'Authorization' = $token
    "Content-Type"  = 'application/json'
}
$uri = "https://graph.microsoft.com/v1.0/applications"
Invoke-RestMethod -Method get -Uri $uri -Headers $Headers
#If now we decide to change the endpoint to /me (remember, it worked before)
$uri = "https://graph.microsoft.com/v1.0/me"
#Now you have an access denied which is normal because your clientID only expose the application.read.all scope
Invoke-RestMethod -Method get -Uri $uri -Headers $Headers
```

If you now paste your AT in jwt.ms, you will see this time under the roles property that only the application.read.all permission is exposed.

Here another thing that I don't really like in this experience which makes the whole modern auth topic fuzzy. Instead of typing https://graph.microsoft.com/.default or https://graph.microsoft.com/Application.Read.All, we type https://graph.microsoft.com/ only which is inaccurate. But if you try to type the 2 others, you will receive an error message where you shouldn't.

If you want more information about the ./default endpoint, you can find more information [here](https://docs.microsoft.com/en-us/azure/active-directory/develop/v2-permissions-and-consent).

## Summary

We can do a lot of fun things with the Azure CLI, MS did a great job to abstract the complexity,but I don't think this is how I will be able to explain how modern authentication is working.

# Azure Powershell

## Introduction

Let's now switch to Azure Powershell. Since few months now, we now have a module that is using the MSAL library (compared to CLI which still rely on ADAL). **MSAL is a library to generate tokens in OIDC/OAUTH 2.0 world** (ADAL will be deprecated in June 2020). Within the Az.Accounts module we have this cmdlet called Get-AzAccessToken, let's try to have some fun with it.

## Basic usage

Let's start by listing all the resource groups we have in our Azure subscription. Of course, the account you will use when you will do you Connect-AzAccount need to have at least **reader access** (or more restricted access if you want to be the "class delegate"...)

```powershell
#Once connected with an account (not a service Principal)
Connect-azaccount
#Let's generate an access token to the ARM fabric audience and make it a bearer token
$Token = "Bearer {0}" -f (Get-AzAccessToken -Resource "https://management.azure.com").Token
#Let's define our subscriptionId
$subId = "<your sub Id>"
# Define our URI
$Uri "https://management.azure.com/subscriptions/$subId/resourcegroups?api-version=2020-06-01"
#Create the header as before
$Headers = @{
    'Authorization' = $token
    "Content-Type"  = 'application/json'
}
# And let's call the endpoint
Invoke-RestMethod -Headers $Headers -Uri $uri -Method get
# You should now see all resource groups your users can see in the subscription
#Let's now imagine you have a Keyvault already created (in RBAC mode) and you want to store a new secret, lets try!
$uri = "https://<your vault>.vault.azure.net/secrets/<your secret name>?api-version=7.1"
#Let's now create a body that will contains your password
$Body = @{
    value = "MySuperSecret"
} | ConvertTo-Json
#Let's now call our Keyvault endpoint
Invoke-RestMethod -Headers $Headers -Uri $uri -Method PUT -Body $body
#You should now receive an error regarding the wrong audience and it's normal. Don't forget that previously, we've generated an AT why the https://management.azure.com audience.
# Keyvault is using another one. Let's generate a new token with the proper audience. You can use this:
$Token = "Bearer {0}" -f (Get-AzAccessToken -ResourceTypeName KeyVault).Token
#or the previous method if you prefer
$Token = "Bearer {0}" -f (Get-AzAccessToken -Resource "https://vault.azure.net").Token
#Regenerate the hearders
$Headers = @{
    'Authorization' = $token
    "Content-Type"  = 'application/json'
}
#You're good to go
Invoke-RestMethod -Headers $Headers -Uri $uri -Method get
# Now you have a new secret stored in our Keyvault
```

Like CLI, Azure Powershell abstract all redirectURI, scopes, clientID, to get your AT. Do you think like AZ CLI, Azure Powershell has his own invoke rest method? Let's see!

```powershell
#Let's see what do we have
get-command -noun *rest* -module az.*
# Yeah Invoke-AzRestMethod ! Let's try as before with CLI
Invoke-AzRest -Path "https://graph.microsoft.com/v1.0/users" -Method get
#You should receive a weird error message like API version is mandatory ...
#If you check the help, you will see that this command is only for ARM endpoint, not graph.
help Invoke-AzRest -online
#If this time we try an ARM URI has the example propose
Invoke-AzRest -Path "/subscriptions/$subId/resourcegroups?api-version=2020-06-01"-Method get
#You should now receive your data
```

To sumarise this basic part, as always, the Azure Powershell experience seems behind the CLI but for basic actions we can rely on it.

## Advanced usage

We will use the same application that we've previously created.

```powershell
#Create a PSCredential object
$Encryptedpassword = ConvertTo-SecureString $MyApp.password -AsPlainText -Force
$credential = New-Object System.Management.Automation.PSCredential ($MyApp.appId, $Encryptedpassword)
#Let's now connect to my tenant as a service principal
Connect-AzAccount -Credential $credential -Tenant $MyApp.tenant -ServicePrincipal
#Remember this application only have the possibility to list applications through graph API audience. So let's do something dummy like generating a token for the ARM audience instead of graph
$Token = "Bearer {0}" -f (Get-AzAccessToken -Resource "https://management.azure.com").Token
#Defin the URI you want to reach
$Uri "https://management.azure.com/subscriptions/$subId/resourcegroups?api-version=2020-06-01"
# And try to get your resource group
Invoke-RestMethod -ContentType 'application/json' -Headers @{'Authorization' = $token} -Uri $uri -Method get
# This time, you should receive a message telling you that this application <ClientID> does not have the right to do action on Microsoft.Resources/subscriptions/resourcegroups/read and it's normal
#Let's now generate a token this time for the proper audience
$Token = "Bearer {0}" -f (Get-AzAccessToken -Resource https://graph.microsoft.com/).Token
$uri = "https://graph.microsoft.com/v1.0/applications"
Invoke-RestMethod -ContentType 'application/json' -Headers @{'Authorization' = $token} -Uri $uri -Method get
# And this time you should be able to list your applications. For fun let's try to list /users because we have the proper audience and the Az Powershell magic abstraction, we don't have to regenrate a token
$uri = "https://graph.microsoft.com/v1.0/users"
Invoke-RestMethod -ContentType 'application/json' -Headers @{'Authorization' = $token} -Uri $uri -Method get
#And you should have an access denied which is normal because this application is allowed to list only applications
```

## Summary

We have almost the same experience with the Az Powershell except the Invoke-restmethod which require more work to compete the CLI.

# Conclusion

The purpose of this article wasn't to explain how modern authentication is working because as I've tried to demonstrate a lot of abstraction is made by Microsoft to help the user experience. Here few take away from this article:

- Understand the difference between **delegated Vs application permission** is critical.
- An access token is generated to an **audience**. You can’t use this token to contact another audience. It will be clearer when we will create our custom API.
- **Scopes** are also an important part. As I’ve tried to explain here, those tools do a lot of abstraction. Later we will see that you can’t even have an AT if you don’t specific a scope.
  
There is far more to cover regarding identity, as you will see it's a wide playground where we can learn a lot of things! See you in the next one.


{% include tip.html content="If you use a lot of Az CLI or Powershell another take away can be to abuse the --debug or -debug when you type a command. For example Get-AzResourceGroup -Debug will give you all requests that is done in the backend. Pretty useful from time to time" %}

# references

[ADAL Vs MSAL](https://docs.microsoft.com/en-us/azure/active-directory/develop/msal-migration)

[Graph API rest reference](https://docs.microsoft.com/en-us/graph/api/overview?view=graph-rest-1.0)

[Azure rest reference](https://docs.microsoft.com/en-us/rest/api/azure/)