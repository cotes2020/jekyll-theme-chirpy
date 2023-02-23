---
title: Simplest way to generate tokens with Azure AD (not best practices)  
date: 2023-02-23 00:00
categories: [identity]
tags: [identity, AAD]
---

# Introduction

I’ve received recently a question which was “what would you do if you had to generate tokens (access and/or id) for non-coders?” In this case, the end user won’t be able to use MSAL or any other Oauth2 libraries, the idea is to click on a link, copy the token and paste it somewhere else.

{% include important.html content="What I will explain in this article is not best practices (implicit flow). You should always use the auth code flow with PKCE for interactive use case." %}

# Create the app registration 

Create a new app registration:

![appreg](/assets/img/2023-02-23/01.png)

Click on **expose an api** (left menu). Click on **set** to tell AAD you want to expose something and **add a scope in this case myscope**:

![appreg02](/assets/img/2023-02-23/02.png)

In the **Authentication** pane, select your platform, **add https://jwt.ms** as redirect uri and **check Access and/or id token** to enable the implicit flow. 

![appreg03](/assets/img/2023-02-23/03.png)

That’s it, now you can secure your app with the enterprise application part where you can enforce who can generate tokens with the **assignment required at yes.**

# Generate tokens 

Now to generate tokens you have to create an encoded url you provide the link to your end user. 

For Access token, create something like this: 

https://login.microsoftonline.com/<tenantId>/oauth2/v2.0/authorize?client_id=<AppId>&response_type=token&scope=api%3A%2F%2F<AppId>%2F<yourscope>&response_mode=fragment&redirect_uri=https%3A%2F%2Fjwt.ms

In my case, it was: 

https://login.microsoftonline.com/<tenantId>/oauth2/v2.0/authorize?client_id=85993318-4486-44c7-b3e4-fecd52e7b95b&response_type=token&scope=api%3A%2F%2F85993318-4486-44c7-b3e4-fecd52e7b95b%2Fmyscope&response_mode=fragment&redirect_uri=https%3A%2F%2Fjwt.ms 

Once your users click on the link, the Azure Authorize endpoint will **ask you your identity** and then **redirect the user directly to the jwt.ms page with your token:**

![accesstoken](/assets/img/2023-02-23/04.png)

If you want **id token** instead, you can provide a link like this one: 

https://login.microsoftonline.com/<tenantId>/oauth2/v2.0/authorize?client_id=<appId>&response_type=id_token&scope=offline_access%20openid&response_mode=fragment&redirect_uri=https%3A%2F%2Fjwt.ms&nonce=678910 

For example, in my case: 

https://login.microsoftonline.com/<tenantId>/oauth2/v2.0/authorize?client_id=85993318-4486-44c7-b3e4-fecd52e7b95b&response_type=id_token&scope=offline_access%20openid&response_mode=fragment&redirect_uri=https%3A%2F%2Fjwt.ms&nonce=678910 

And again, jwt.ms will show you your id token: 

![idtoken](/assets/img/2023-02-23/05.png)

# Conclusion

Here is a quick and dirty way to generate tokens. Don’t forget you shouldn’t use this for prod environment. 

# links 

1. [AAD Implicit flow](https://learn.microsoft.com/en-us/azure/active-directory/develop/v2-oauth2-implicit-grant-flow)