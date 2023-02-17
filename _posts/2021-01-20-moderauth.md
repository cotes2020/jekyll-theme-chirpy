---
title: What modern authentication is?
date: 2021-01-20 00:00
categories: [identity]
tags: [identity,AzureAD]     # TAG names should always be lowercase
---

# preamble

I'm not an Identity guru, it’s just a subject that I like, and I want to share what I’ve gathered over the months of learning. I consider this topic critical enough to spend a big number of personal/professional hours in this world where every day a new leak appears in the news. Why I consider this topic important? I've spent many years working on AD (since NT4 even is it wasn't "AD") and the concept that I've learned since can't be use anymore in a full Internet connected world. Protocols like Kerberos or NTLM does not work very well outside of your company’s walls this is where Open ID and Oauth2.0 start to shine. For me, modern auth will become the replacement of those “legacy” protocols sooner or later.

Modern authentication is not a simple topic, you may have to read this article several times to understand everything (and it’s normal). I still have some blurry areas from time to time even after the crazy numbers of learning's hours. This is not the first time that I try to explain modern authentications, I've writen several articles already and every times I consider the result to messy. Let's do another try! This time, I will do multiple articles where we will do both theory and hands on.

Finally, I’m not a developer, and my main language is PowerShell. I will do my best to provide examples in other languages (later).

# Introduction

Not so long ago, sharing user/password to access your data from a third-party application like an email application, was considered as a normal practice. When you do this, you basically trust the application/provider to keep your secrets secure. When I say trust, don't forget that now the application has your login password and the application can now behave as you. The app can send an email but if you think, the **app can also now delete your mailbox** because there is no notion of roles or scopes. Sadly, every day we can see news regarding leaks. All those leaks have something in common, Identity. Securing identities is not a simple task to do and if it’s not your job, you can’t claim you know how to protect credentials efficiently. This is sadly what those providers/applications promise you and fail. This is when modern authentication comes into place. Providers or applications prefer now to trust a worldwide known Identity provider (Idp) to delegate the identity part to someone who knows how to make a transaction secure. Compared to legacy authentications, modern authentication does not share any password, but instead signed tokens once the Idp consider the request valid. I won’t talk about device management, conditional access, CASB or other E5 cool Microsoft Azure features, but to consider a request valid, AAD propose a lot of options and you just decide which ones you want to use.

{% include warning.html content="Don’t give your password to an application except if you understand what you’re doing." %}

# What modern authentication is?

## The notion of trust

Let’s explain this concept with a story!
**You as owner of a resource** (can be a picture, an address book, an email, graph API, a custom homemade API…) want to interact with it through a third-party tool. Let’s say, you want to send an email from a native application.
This **resource** (your email) is hosted on a server and you can access it through an **API**. You can interact with those APIs through **scopes** (or **resources**). A scope, which is a permission or an action you can do, can be read, write, send, delete, or anything else exposed by the API.  
Then you have an **authorization server** (authZ) which host an **identity of you**, the resource owner. This identity is called a **Service Principal**. If this server manages **authentication (authN)** too, we are talking about an **Identity Provider (IDP)**. Today, authorizations are mainly managed by the OAuth2.0 protocol and the authentications by Open Id Connect (OidC) protocol. In other words, **the identity provider should be the only place where you store your identity information like password**, MFA information, …
Then we have the **resource server** which host the email you want to send. This resource must have a **trust relationship with the IDP** too. If you have multiple servers with your data, each server will have to trust the same IDP. Therefore, managing credentials is not something that all companies can do correctly. You need a certain reputation in order to gain third parties trust.
Finally, you want to send your email from a client. This client can be a mobile app, a web server, a Single Page Application (SPA) or a native client. Depending on the type of application, you shouldn't use the same pattern, which is why modern auth seems more complicated than other identity protocols.

{% include note.html content="Modern authentication can be translated by how to give your permission to another tool to do a specific action on behalf of you." %}

**Summary:**

![modern auth 01](/assets/img/2021-01-20/01.png)

 If I can’t provide my credentials to the resource server, what should I pass instead? 
Here a high-level picture of how modern authentication is working:

![modern auth 02](/assets/img/2021-01-20/02.png)

A lot of third parties (server1) trust an Identity Provider, in this case Azure Active Directory(AAD), therefore it should be the only place where you store your credentials. Then, AAD distribute trusted dynamic generated signed tokens to your API/Server/Application instead of forwarding your password directly.
Let’s now have a deeper look on AAD

{% include note.html content="I will focus on Azure Active Directory, but most of concepts are used into other Idps as well." %}

If we continue or story, we want to access, through a native client (script/WPF/Go app/...) you have installed on your local machine, a mailbox which is hosted on a server/application somewhere. The client application has first to be created/registered into the IDP. **The IDP will reply with a client ID which is globally unique to all Azure AD Tenants**. This is how you can create an application that will be available cross AS Tenants.

{% include note.html content="An AAD tenant (or Directory), is a logical place hosted and managed by the Idp where you will find users/groups/devices/enterprise apps/...Companies have their own tenant and a Client Application can be available cross tenants (more information later). For example, an Office 365 subscription, which is composed of multiple applications, belongs for to a specific tenant. Those applications are used by multiple tenants without any security concerns" %}

{% include warning.html content="You will see later multiple flows exist to be authenticated to your API. To explain this story, I will use the **authorization code flow** which is the more secure and the one to can use in ***almost*** every cases." %}

## Modern authentication require authentication flow

When you (the resource owner) try to access the data (your mail), the client redirects the request to the IDP with his **ClientID** (now the app is registered in the tenant), **scopes** (what we want to do) and a **redirectURI** (where to go once logged).Then, the IDP will verify if yes or no you’re **already authenticated**. If not, the IDP ask the user to authenticate. Once authenticated, the IDP will verify if you’ve **consented to the scopes**. If not, the IDP ask you to consent before proceeding and store this “fact” into his database for later usage. If everything is correct, the IDP send back an authorization code to the redirectionURI.

{% include note.html content="We will discuss more about consent later. But consent in modern authentication is exactly the same as when you install an app on your smartphone, and you receive a prompt 'This app require access to your webcam, are you OK?'" %}

So if we try to recap, here what we're doing:

![modern auth 03](/assets/img/2021-01-20/03.png)

## Public Vs Confidential application

This is where we have to discuss about **public Vs confidential** applications.
In Oauth2.0, a **public application** is an application where secrets can’t be protected. In other words, a **native application**, a **Single Page App (SPA)** or a **mobile app** are public application. In other words, a native app can be reverse engineered or a Blazor page can be easily sniffed. Therefore, you shouldn’t put any secret into those type of applications.
As a contrary, a **confidential application** can protect secrets.  A **web server**, a **backend API** can hold secrets (or certificates) in a secure way without exposing them to unauthorized parties. The application can exchange secrets, access keys, in a secure manner because everything happen in the backend, far from the public.

{% include tip.html content="A clientID or a tenant Id are not a secure string, you don’t have to protect them (compared to secrets, certificates, access keys, tokens,...)." %}

{% include note.html content="If **implicit flow** is used the last part with the **authorization code** sent back to the authorize server **does not exist**. **The Idp send directly the access token directly without refresh token**. This flow is less secure compared to the authorization code flow and you should use it only in specific cases." %}

## Now that I have my code can I connect to my API?

Now that we receive our authorization code, we will send it back to the IDP (**another endpoint** called token instead of authorize) in conjunction with the **ClientID** and **potentially a secret** (for confidential apps). This is when the IDP will reply with an **access token, a refresh token (because of this specific flow) and eventually an Id Token**. You receive an Id token if you’re already authenticated (auhtN). Then we simply use those credentials to our API to access our data.
As we can see in this table, different flows do not provide the same results:

![modern auth 04](/assets/img/2021-01-20/04.png)

If we try to put everything in a picture, it should look like this:

![modern auth 05](/assets/img/2021-01-20/05.png)

So now you’ve received your **Access token** (AT) and Refresh token (RT) what can you do? An AT is a **short lived** (one hour) encoded based64 token that you can use to access your API (or mail in our story). Those two tokens should be considered as secrets and shouldn’t be committed in your version control or exposed anywhere. Even if you can, you shouldn’t decode the AT to use the values within it. **AT purpose is for application only**, if you need to control authentication (who can access) and authorization (what the user can do), use the **Id token instead**. **After an hour**, the AT is now expired and **RT should be used instead to generate a new pair of AT/RT**. And again after an hour. By default, and if the Idp does not want to verify your identity (because you change your public IP, an admin deactivate your account, ...), you can reuse the RT during **90 days** without any prompt. I consider this flow as a semi-interractive one (semi because every 90 days you have to put your credentials again). **A good example is the AZ powershell module**, you don't have to re-authenticate to your context (subscription) even after a reboot. You connect to your subscription and once authenticated, Azure stored both tokens locally and use them when needed.

{% include important.html content="In a production environment, in addition to the ClientId, Scope and redirectURI (step 2) you should generate from the Client App a challenge code too. Check [PKCE](https://tools.ietf.org/html/rfc7636) for more information. In a future hands on, we will use this protection" %}

# Summary

This is an holitic view of what modern authentication is. In the following articles, we will go deeper in specific subjects. With this one, I've tried to stay high level. See you in the next one where we will talk about Azure AD, MSAL, Enterprise app, App registration, token signature...

# references

- [AAD Allowed athentication flows](https://docs.microsoft.com/fr-fr/azure/active-directory/develop/msal-authentication-flows)
- [425show](https://www.youtube.com/channel/UCIPMDupgTRsJY5sxcdBEtCg)
- [Microsoft Identity Plateform]( https://docs.microsoft.com/en-us/azure/active-directory/develop/)
- [jwt.io](https://jwt.io/)
- [jwt.ms](https://jwt.ms/)
- [Public Vs confidential application](https://docs.microsoft.com/bs-latn-ba/azure/active-directory/develop/msal-client-applications)
- [Azure AD fundamentals](https://www.youtube.com/playlist?list=PLLasX02E8BPD5vC2XHS_oHaMVmaeHHPLy)
- [Glossary](https://docs.microsoft.com/en-us/azure/active-directory/develop/developer-glossary)

