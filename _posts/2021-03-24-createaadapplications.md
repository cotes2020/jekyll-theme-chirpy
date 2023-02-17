---
title: Another way to create Azure AD applications
date: 2021-03-24 00:00
categories: [identity]
tags: [identity,Powershell,AAD]
---

# Introduction

I’m still enjoying learning AAD identity topics. In the previous articles, I’ve explained the differences between app registration and Enterprise app. Since I’ve started this learning path, I had to create dozens of applications in my lab. At a certain point, I’ve decided **to automate the AAD app creation**. Today, and according to my knowledge, you can create AAD app with the **AzureAD/Az Powershell modules** or with the **CLI**. Those options seem to have **some limitations** in terms of configuration. Then you have ARM template which is a limitation by itself when you’re talking about AAD application creation (and no I don’t consider [deployment scripts](https://docs.microsoft.com/en-us/azure/azure-resource-manager/templates/deployment-script-template) as a solution). Finally, I know you can use [**Terraform**]( https://registry.terraform.io/providers/hashicorp/azuread/latest/docs) or [**Pulumi**]( https://www.pulumi.com/docs/reference/pkg/azuread/) which seems **pretty complete** to create AAD apps. I wanted to spend few hours to understand **how things work and using SDKs/libraries do not help** in this case. Therefore, I’ve created this repository where We can do a step by step and comment on each line of code. You can find the code [here](https://github.com/SCOMnewbie/PSAADApplication).

{% include important.html content="UPDATE: I've converted this repo in a real usable module instead of the tremendous copy/paste. You can find the module (psd1/psm1) in the PSAADApplication folder." %}

# Why

The main goals of this [repo](https://github.com/SCOMnewbie/PSAADApplication) is to:

* **Learn** by myself without relying on libraries. It’s even more important for the next article where I will talk about acquiring tokens.
* **Help people** which still have some issue to understand public/confidential app concept. Doing this demo live can show/explain this topic in detail.
* Quickly **create/destroy a demo environment**.
* Explain most of the AAD “attributes” like **approles/ Oauth2permissions/Optional Claims** an so on during the demo
* Explain how things work when you do a **az ad sp create …** Several actions are made with this single command.
* Explain that we can **implement standardization in our application**. For example, force the token endpoint version to 2.0 only.

# Code explanation

## Template creation

**All apps** (RBAC/Desktop/SPA/…) that we create during this demo **depend of a “generated” json**. I’m using the word “generate” first because working on json directly makes me sick. And then because we have to **calculate properties (guid/displayname)** during the json creation. Here how to use the repo:

* There is templates in the **Templates folder**. This is what I use during the demo (demo.ps1). There is a lot of **useful comments in those files**.
* ~~ During the demo, before executing the Convert-SettingsToJson cmdlet, you have to make sure you’ve <span style="color:red">**copied/pasted the good template in the Convert-SettingsToJson.ps1 file**</span>. You have to do this on each app (sorry about that). Help from smarter people than me is appreciated to improve this part. ~~

{% include note.html content="I didn’t find a smart way to avoid copy/pasting here, if you have ideas, I’m all ears!" %}

## Demo

Not a lot to say here. At the end of this demo, you will be able to:

* Create a RBAC better than with the CLI (troll /off)
* Create a public/confidential app based on a switch (New-AppRegistration -ConfidentialApp)
* Configure the App with:
  * Logo
  * Credentials (secrets)
  * IdentifierUris (standardize your deployment)
  * App Roles
  * Expose an API
  * Force the Token endpoint to V2 (standardize your deployment)
  * Optional Token claims
  * Enforce assignment on your service principal

Things that I don't cover in this demo:

* Publisher verification
* Certificates (for now)

{% include note.html content="You can use the Get-GraphAPIScopesInfo cmdlet to extract the required Id if you need to call Graph API in your application." %}

# Conclusion

I hope this demo will be useful to some of you. I don’t know if the time that I've spent to build this demo worth it, but it was a fun ride which is the more important point. See you in the next article(s) where this time we will start to build things and use all those applications!

# References

* [Graph REST Reference](https://docs.microsoft.com/en-us/graph/api/overview?view=graph-rest-1.0)
* [App Registration manifest](https://docs.microsoft.com/en-us/azure/active-directory/develop/reference-app-manifest)