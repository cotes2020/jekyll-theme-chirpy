---
title: Build a Powershell API with Pode
date: 2021-07-07 00:00
categories: [powershell]
tags: [identity,AAD, Powershell]
---

# Introduction

I’ve tried to explain previously how to secure a backend api where I’ve used Azure functions to demonstrate how to protect it, but what if we're not on Azure? I know you can run the Az functions anywhere with Arc for Kubernetes but here, I wanted to create something quick and simple for my demos. This is when I’ve felt in love of [Pode](https://badgerati.github.io/Pode/) which is a **cross platforms Powershell module to help you to create websites, schedulers API and more…**
Because I couldn’t find a lot of resources except the great documentation(https://badgerati.github.io/Pode/Tutorials/Basics/), I’ve decided to share my experience after couple hours playing with it.

{% include note.html content="For the fun, I’ve created a small node SPA frontend from where I authenticate my users and then execute my various Pode endpoints from both Azure WebApp and GCP Cloud Run. Don’t hesitate to ask if you’re interested in how to do it." %}

# Implementation

You can find the code [here](https://github.com/SCOMnewbie/psoauth2/tree/main/Examples/Powershell/09-WebAPI-With-Pode). 

I wanted to create a **“model”** that I can **reuse for later**. Let’s focus on **startpode.ps1** for now which is the masterpiece of this POC.

```Powershell

Import-Module -Name Pode
Import-Module './usr/psoauth2.psm1' # Not comming from the gallery

# Get environment variable from WebApp/Keyvault
$Myappsecret = $env:Myappsecret
# App Roles my API will validate
$WriteAccessRoles = @('Write.Access','Admin.Access') #Admin has access to both roles
$ReadAccessRoles = @('Read.Access','Admin.Access') #Admin has access to both roles

```

1. First import both the Pode and Psoauth2 module. Both modules will be **available within our web server**.
2. Then we fetch our application secret from **environment variable**. We will need this secret later to do the OBO/Client credential flow…
3. Finally I declare the 3 app roles that I’ve used in my previous Az functions demo. We will need them later for the authorization check part.

Now that our parameters/variables are defined, let’s start our Pode server:

```Powershell

Start-PodeServer {
     #Declare global Middleware (mainly because GCP and Cloud Run does not allow you to allow specific CORS from the portal)
     Add-PodeMiddleware -Name 'MandatoryAuthorizationHeader' -ScriptBlock {
        Add-PodeHeader -Name 'Access-Control-Allow-Origin' -Value '*'  # * Because it's a POC, use your specific frontend URL here
        Add-PodeHeader -Name 'Access-Control-Allow-Methods' -Value 'GET, OPTIONS'
        Add-PodeHeader -Name 'Access-Control-Allow-Headers' -Value 'Content-Type,Authorization'
        
        return $true
    }

    # Allow the option method in each route
    Add-PodeRoute -Method Options -Path * -ScriptBlock {
        return $true
    }

```

Here we define the Cross-origin resource sharing (**CORS**). You don’t really need this except if you **plan to use this api from a website/Single Page App** (which was my case). With this code, we simply say Pode to add **specific headers to all our responses**. Simple as that!

```Powershell

    #Redirect errors to terminal. Cool to have logs redirected to containers logs for tracking
    New-PodeLoggingMethod -Terminal | Enable-PodeErrorLogging
    
    # Configure Pode to listen on 8080 in HTTP (with localhost, you "break" Docker)
    Add-PodeEndpoint -Address * -Port 8080 -Protocol Http

```

Then you define how you plan to **handle errors**. In this case I wanted to redirect errors directly in console to catch them in the default WebApp/Cloud Run **container log tracking**. Next you define the bindings, port and protocols…

Pode helps you to automatically **choose config file based on a specific environment**:

```Powershell

    # Get App setting from psd1 files (can't be located outside Start-PodeServer)
    # Depending if there is an exposed environment variable called PODE_ENVIRONMENT with dev, it will take server.psd1 by default
    # https://badgerati.github.io/Pode/Tutorials/Configuration/#environments

    $Config = Get-PodeConfig # Gather Audience, TenantId, Allowed issuers

```

Now the big part and because it’s a POC I just wanted to spend few hours to build it. Therefore, I’ve decided to mainly **copy/paste code between different role permission** (Read/Write/Admin access). I’m sure there is smarter way to do it, don’t hesitate to send me some ideas :D.

```Powershell

# You need to be authenticated AND have a role declared in the $WriteAccessRoles array
    $WriteAccessRequired = {
        # Same as above
        $EmptyAuthorizationHeader = [string]::IsNullOrWhiteSpace($WebEvent.Request.Headers.Authorization)
        if ($EmptyAuthorizationHeader) {
            Set-PodeResponseStatus -Code 403
            return $false
        }

        try{
            $Configs = $using:Config
            $Aud = $Configs['Audience']
            $Iss = $Configs['Issuers']  # Warning: Array

            $RequestAccessToken = $WebEvent.Request.Headers.Authorization
            $DecodedToken = ConvertFrom-Jwt -Token $RequestAccessToken
            Test-AADToken -AccessToken $RequestAccessToken -Aud $Aud -iss $Iss
        }
        catch{

            Set-PodeResponseStatus -Code 403
            $_ | Write-PodeErrorLog
            return $false
        }

        # Now token and authentication has been validated, let's validate the role
        $Authorized = $false
        [array]$Roles = $DecodedToken.TokenPayload.roles
        $Roles.ForEach({if($_ -in $using:WriteAccessRoles){$Authorized = $true}})
        if(-not $Authorized){
            Set-PodeResponseStatus -Code 401
            return $false
        } # You don't have the app role in the access token > 401

        return $true
    }

```

Here we define a scriptblock we will use **later in the route’s configuration**. This is where we will validate our token, as I’ve explained in the previous article, we will:
* Make sure the Authorization header is present
* Validate that’s a bearer token
* Grab the configuration from the server.xx.psd1 file. 
* Validate the AAD signature
* Validate the claims
* And validate the roles
  
**If we don’t like something, we drop the request as usual!**

Now that we’ve defined all the rules we will apply to our routes, let’s take an example:

```Powershell

# This route require to be authenticated AND have the write.access or Admin.Access permission
    Add-PodeRoute -Method Get -Path '/api/authorizationheaderwithwriteaccess' -Middleware $WriteAccessRequired  -ScriptBlock {
        
        # Get info from config file
        $Configs = $using:Config
        $Aud = $Configs['Audience']
        $TenantId = $Configs['TenantId']

        $Splatting = @{
            ClientId             = $Aud                                    # Define in server.psd1
            TenantId             = $TenantId                               # Define in server.psd1
            Scope                = 'https://graph.microsoft.com/.default'
            secret               = $using:Myappsecret                      # Define in top of this script from environement variable (using: runspace)
            verbose              = $true
        }

        # Let's request a server to server token (should be cached and track in a DB to avoid hammering AAD)
        $Apptoken = New-APIServerToServerToken @Splatting

        #Let's now read all users tenant with this new token
        $uri = 'https://graph.microsoft.com/v1.0/users'
        $Users = Invoke-RestMethod -ContentType 'application/json' -Headers @{'Authorization' = "Bearer $Apptoken" } -Uri $uri -Method get

        Write-PodeJsonResponse -Value $Users
    }

```

Here we create a route called /api/authorizationheaderwithwriteaccess (sorry for the bad naming) where we define to use the **$writeaccessRequired scriptblock Middleware** created just above. For the rest, it’s like my other articles, I simply do a client credential call to Graph api using the application context and send back the result. **Super easy and super clean to require authentication and authorization within your api!**

# Container

For the container part, nothing fancy here. I simply pull the Alpine Powershell image, add few files into it, run Pode under a non-Root user and expose the port 8080 … You can check the Dockerfile for more info. The only “bad” part according to me is the 227 Mb generated image which is a lot … I’m currently trying to learn C# and ASP.net, let’s see in few months what will be the size of the image when I will do my multi stages…

# Conclusion

**Thank you again Matthew Kelly (@Badgerati)**, your module is gold and deserve more visibility. When I see what I’ve succeed to create in couple hours, I can’t imagine what we can do with it if we decide to spend some time on it. Thank you again for this great experience! See you for the next one...
