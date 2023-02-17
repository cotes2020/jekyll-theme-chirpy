---
title: How to use modern authentication
date: 2021-06-09 00:00
categories: [identity]
tags: [identity,Powershell,AAD]
---

# Introduction

I'm learning modern authentications since several months and I still learn new things every day! I'm super happy to publicly share what I’m doing on my free time and I hope it will help some of you to **better understand how modern authentication is working**. In the past months, I’ve created several articles to explain what modern authentication is. This time we will **make it real** and we will use two modules that I’ve released few weeks ago. The first one to create AAD apps (useful for multi tiers apps) which can be downloaded [here](https://github.com/SCOMnewbie/PSAADApplication) or you can read my previous [post](https://scomnewbie.github.io/posts/createaadapplications) for more information. And a module called **psoauth2** which will help us to interact and learn modern authentication with AAD. You can find the module [here](https://github.com/SCOMnewbie/psoauth2) and in fact this is the masterpiece of this initiative.

{% include important.html content="This module has been created to help people to better understand modern auth. My advice is to use the MSAL.PS module if you want to use Powershell for production workload. I will propose few examples with MSAL.PS soon. Yeah I know it's weird to tell you to not use a module on which I've spent 2 months to build..." %}

When I’ve started to work on modern auth with Powershell, I only wanted to execute graph queries from my scripts. I have to admit the online available resources weren’t great at that time (client_credential is not the only flow you can use). This is when I decided to create something to explain my understandings to people.

Now, to be honest **Powershell is not where you will be able to get the full benefit of the identity platform (AAD)** but PosH is good for demos and debugging. Some more "advanced" language better interacts with the platform (dynamic consent, identity.web library and asp.net, conditional access context auth request, ...). Does it mean we can't learn how modern authentication is working with Powershell? Nnnnaaahhhh.

{% include note.html content="**I will update this article over time.** It takes a lot of time to build those demos (even more for advanced ones) and to put everything 'on paper'." %}
 
Finally, those modules as provided as-is without warranty 😊.

Let’s have fun!

This initiative will propose:

* To use Powershell mainly to explain how to play with AAD.
* Several demos where each demo tries to focus on a specific topic
* A lot of tips through comments into the code (in the module or through the examples)
* A module created to learn oauth/OIDC. 

This initiative won't propose:

* To explain what identity is from scratch. Read my previous articles of follow the [425 show](https://www.twitch.tv/425show) for more information.
* To explain how to use a Powershell module. Ask if you need help.

Why did I start this initiative?

* Because modern auth is not a simple topic but it's not so complicated when you understand how it's working. Thanks to Microsoft.
* Because a lot of co-workers struggle with the subject (dev or IT).
* Because I wanted to give a little contribution and spending some time to write things.
* Because Powershell does not have a lot of published articles around modern authentication.

# Work done

This module provides 7 commands:
* New-Accesstoken
  * Manage a **local cache** for access token, refresk tokens, Id Tokens, secrets for both Windows and Linux.
  * Propose several flows:
    * **Device code flow**
    * **Client_credential flow** with both secrets and certs
    * **Auth code flow with PKCE** with or without secret
  * Manage **refresh token when access token is expired** (No interaction)
* Clear-tokenCache to simply remove the local cache
* **Revoke-RefreshTokens** to ask AAD to invalidate all current refresh tokens that has been generated before for a user.
* Test-AADToken which **validate if a token is valid or not**. We will use this function later once we will play with advanced concepts.
* **ConvertFrom-Jwt** to decode the JWT with header,payload and signature. I had to make this command public because I will use it later with Azure functions.
* New-APIOnBehalfToken to call the **On Behalf Of flow**. This command is public because Azure functions do not like my caching method that I use with New-Accesstoken.
* New-APIServerToServerToken which is a basic client credential but again without caching for the same reason above.

# Work to do

* Auth code PKCE with certs
* Create multi tier demos with advanced concepts
* Add MSAL.PS examples
* Create a C# desktop app (I already have a good usecase.... Now I need to learn C# lol)
* I may add my MSI Azure ARC command in this module too later.

# How to use this module

There is two important parts:

* In the **psoauth2\psoauth2 folder**, you will be able to find the latest usable Powershell module. I don't think it's a good idea to expose this module though the PS Gallery (not prod ready).
* In the **examples folders**, you will be able to find **multiples demos** where I explain different aspects of what modern auth is and how you can use it. Over time, I will fill this folder with more cases (multi-tiers app) and with other languages too.

{% include note.html content="Don't forget to use your TenantId/SubscriptionId if you want to play with the demos (always on top of the script)" %}

I recommend to simply follow demos in order.
# Prerequisites

* You will need Global admin permission to run those demos (admin consent). You can request a dev AAD environment for free [here](https://developer.microsoft.com/en-us/microsoft-365/dev-program).
* You will need to use the PSAADApplication module. You can find it in the Example folder or directly [here](https://github.com/SCOMnewbie/PSAADApplication/tree/main/PSAADApplication/PSAADApplication) for the latest version.
* You will need the AZ module and/or the AZ CLI too (for certs demos).
* In some demos, you will need an **active subscription** with a Keyvault already created.
# Basic usecases

The simple usecases will contain examples with only one tier. For now it's only Powershell, but later I plan to use other langages.

## Confidential app with secret for subscription role assignment

* Script is located [here](https://github.com/SCOMnewbie/psoauth2/blob/main/Examples/01-Script-ConfApp-Secret-AzureAssignment.ps1).
  
* Takeaways:
  * Simple to implement (dev/test)
  * Make sure you implement a key/secret rotation in your application to avoid expiration (Event based notification)
  * No user assignation with this method
  * AAD audit logs can be challenging. You don't know the user context, the task run as an application.
  * You have to store your secret somewhere (not committed in plaintext in your repo)

* Picture of what we will do:

![01](/assets/img/2021-06-09/01.png)

## Confidential app with certificate for subscription role assignment

* Script is located [here](https://github.com/SCOMnewbie/psoauth2/blob/main/Examples/02-Script-ConfApp-Cert-AzureAssignment.ps1).
  
* Takeaways:
  * Certificates are better than secrets when you have a good certificates hygiene in your company. Secrets are catched by proxies or other auditing tools, certificates no.
  * Certificates is more complicated to implement than secrets but it's not impossible. My current user experience is really not optimal for now (more info in the script file). Microsoft should work on this part to make it more accessible.
  * Make sure you implement a key/secret rotation in your application to avoid expiration (Event based notification)
  * No user assignation with this method too
  * AAD audit logs can be challenging
  * Keep your private key safe
  * Use **pem** KV policy is you plan to use CLI or from a Linux box

* Picture of what we will do:

![02](/assets/img/2021-06-09/02.png)

## Confidential app with secret to call graph API

* Script is located [here](https://github.com/SCOMnewbie/psoauth2/blob/main/Examples/03-Script-ConfApp-Secret-Application-GraphAPI.ps1).
* Takeaways:
  * This is what you see everywhere on Internet.
  * We need the TeanantId as a new information to provide. We will use exclusively **single tenant app** for now.
  * This request will be executed as an application permission (will have access to all tenant resources)
  * Secret should stay protected
  * Now it's not just CLI or Powershell. You can do this with multiple runtimes. You can find Microsoft libraries [here](https://docs.microsoft.com/en-us/azure/active-directory/develop/reference-v2-libraries) or any certified OIDC libraries [here](https://openid.net/developers/certified/). The concept is still the same.
  * We will use the Graph beta endpoint (following a friends need), but use the V1.0 in production, not the Beta version.

* Picture of what we will do:

![03](/assets/img/2021-06-09/03.png)

## Confidential app with cert to call graph API

* Script is located [here](https://github.com/SCOMnewbie/psoauth2/blob/main/Examples/04-Script-ConfApp-Cert-GraphAPI.ps1).
* Takeaways:
  * Client credential flow should be used to server to server (no user interraction).
  * Because there is no interraction (and so no dynamic scope consent), **.default** scope has to be use.
  * New-AccessToken with ClientCredentialFlow parameter (psoauth2 module cmdlet) can be used to generate an access token and keep it locally into cache (should work on Linux machine too)
  * psoauth2 module manage local token caching.
  * When you use cert auth, you end up create a custom JWT, you use your local private key to sign the token and on the other side, AAD will decode it with the public one. This is called an assertion.
  * it's doesn't matter if you commit the thumbprint.
  * use pfx KV policy if you plan to use the cert from Windows.

* Picture of what we will do:

![04](/assets/img/2021-06-09/04.png)

## Public app with NO SECRET delegated permission to call graph API (Auth code flow with PKCE + Device Code)

Context:

    Now the fun really begins. Having over-privileged API that you use with secrets/certs is cool (application permission), but the real benefit of modern authentication is the delegated permission part where you don't need any secrets. The platform will use the permission you have as user and act on behalf of you. In other words, if you can't do things with your account, you won't be able to do it through the API (delegated permissions). In the demo, we will create a public app (no secret/cert), require assignment on it (people have to be assigned to authenticate to it) and as before (with a right scope this time) request privilege action from graph (write user auth methods in the demo and remove user account from the picture).

* Script is located [here](https://github.com/SCOMnewbie/psoauth2/blob/main/Examples/05-Script-Public-Delegated-GraphAPI.ps1).
* Takeaways:
  * Public application does mean you add more ways to authenticate to your application. For example device code flow or ROPC flow (don't use it...).
  * You don't necessary need a secret to use modern authentication. 
  * Auth Code with PKCE should be the way to go first (even for confidential apps). We will use the New-AccessToken cmdlet.
  * We will use the device code flow too (with the New-AccessToken cmdlet)
  * The request will run with a delegated permission (on behelf of user privileges)
  * We can assign who can access our app now (in comparison to the client credential flow)
  * Don't forget to read comments even on the verbose lines (lot of useful information)
  * Now in AAD logs, you will see user ABC did XYZ action trough application AppID. In means that if you're a Global admin, you will be able to do GA stuff without commiting any secrets!
  * With user assignment, you can even say, GA1 can use the resource X but not resource Y. (we do the test in the demo)

* Picture of what we will do:

In the demo we're are just talking about Global admins accounts (2 differents accounts), but in the picture we're adding 2 regular users accounts too. The idea of this picture is to explain what delegated permission is. To be able to call your API (at least in this demo), you has to be first allowed to request a token (user assignment) and then you have to have the permission to do an action with your account. In other words, in this picture, only User account A which is global admin will be able to deactivate a user account.

![04](/assets/img/2021-06-09/05.png)

## Public app delegated permission with refresh token

Being able to call our API with delegated permission is cool, but do I have to authenticate every hours to my application? This is where refresh token comes into place. The goal of this demo will be to explain how you can get an access token. As before, read the demo file to have deeper information.

* Script is located [here](https://github.com/SCOMnewbie/psoauth2/blob/main/Examples/06-Script-Public-Delegated-GraphAPI-RefreshToken.ps1).
* Takeaways:
  * All authentication flows do not generate an refresh token. For exemple client credential don't.
  * openid scope allow your app to receive an Id token. This scope is like adding [cmdletbinding] to your function to then be able to use offline_access and so on...
  * offline_access is the scope you have to configure to receive a refresh token
  * Use the oid + sub claim to generate a unique id for a use between all tenants. The profile scope add those claims (check the demo).
  * To play with the refresh token, you will simply use the same command New-AccessToken but only when your access token will be expired (add verbose)
  * Funny usage of the PSAADApplication module where I will create 5 pre-configured applications with a "simple" swich statement.
  * You can add other claims in the token you will receive in both the Id and Access Token. Check out the joker application (optionalClaims).

# Advanced usecases
## Multi-tiers applications (frontend with backend api)

* You have all information in this [article](https://scomnewbie.github.io/posts/protectbackendapi/).
* Takeaways:
  * Even if our **frontend** does not have secret, we’re still be able to configure the AAD app as a **confidential app** to accept only one authentication flow which is the auth code flow here.
  * **App role** exist within an app. It's a good solution to **implement authorization** within your app. Just look at the roles claims in the token from your api.
  * **The API is responsible** of validating both the **token** AND the **authorization**.
  * **Easy Auth** is cool to **protect your exposed api** without any effort “for free”. You need a **better protection**, use service like **APIM**.
  * If you plan to expose your single tenant backend API to only a set of users, don’t forget to enabled the the **user assignment required** under the properties menu (Enterprise app)

# Conclusion

I hope this article has been useful, don't hesitate to contact me on Twitter/Github if you have question or concerns, I'm always open to feedbacks. 
