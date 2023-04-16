---
title: How to protect a Pode web api with Azure AD 
date: 2023-04-16 00:00
categories: [identity]
tags: [identity, AAD, Pode]
---

# Introduction

I’ve recently done a knowledge sharing session about how to protect web API with Azure AD. Long story short, I’ve decided to use the Pode module in addition to MSAL.PS and ValidateAADJwt which is a module that I’ve published which optimize the Azure AD token verification. 

In this article I will expose a template that I’ve created that I can paste for later usage where I expose several grouped routes where each group use the middleware to verify if the received token is first a valid one (exp, signature, iss, aud validation) and then add more claim validation. 

{% include note.html content="The ValidateAADJwt, which is a cross platform module, downloads public key used for signature validation to speed up later verification." %}

# Web API  

I won’t explain the Azure AD configuration, don’t hesitate to ask if you need help. You will be able to find the code [here](https://github.com/SCOMnewbie/Azure/tree/master/Identity-AAD/Pode). 

In this example, I will expose 3 groups of routes: 

* Anonymous for fun. 
* Admin where the token received required to be valid and with the admin property in claim role. 
* NonAdmin where the token received required to be valid and with the nonadmin property in claim role. 

Note: 
{% include note.html content="I’ve used some random key pair just to get a valid dockerfile." %} 

The only comment I want to add in addition to the well commented code is defaultRule.ps1 (Under authentication folder). You don’t have to touch this file, this one will validate the required fields. If you need to validate something else, just add a new file like adminRole.ps1 and put your logic. Then simply add a new line ine the startpode.ps1 (line 19). 

I hope the code will be helpful to some of you. 

Cheers