---
title: Let’s play with ARC for servers and Powershell (Key vault part) 
date: 2023-02-19 00:00
categories: [powershell]
tags: [Powershell, ARC]
---

# Introduction

In the previous article, we’ve played with the storage api and exposed few limitations, this time we will play with Key vault. In this article, we will see how we can interact with a Key vault through an ARC agent. With this article, we will: 

1. Expose some limitations
2. Fetch a secret with REST calls directly 
3. Fetch a secret from Az-Keyvault module with an **unusual way** 
4. Play with the official **secret management module** 

# Limitations 

For this article, I’ve created a demo Key vault (testacrfun) in RBAC mode, a secret (mysecrettohide) and I’ve granted the role “Key Vault Secrets User” to the ARC agent.  

Like the previous article, we will connect with the accesstoken parameter and try to fetch the value of our secret. 

```Powershell 
$token = Get-AccessTokenWithAzIdentity -Audience ARM 
$null = connect-AzAccount -AccessToken $token -AccountId <your arc appid>  
Get-AzKeyVaultSecret -VaultName <your vault> -Name <your secret> 
``` 

Here the result: 

![limitation01](/assets/img/2023-02-19/01.png)

For the same reason as with the Az.Storage module (previous article), when you use connect-azaccount, you’re connected to the Azure control plane. When you request a secret, you use the data plane … 

# Use the REST api directly 

The first option is to use REST calls directly. You can for example load this [function](https://raw.githubusercontent.com/SCOMnewbie/Azure/master/Identity-AAD/Get-KeyvaultSecret.ps1) in memory and see the result. 

```Powershell 
$tokenKV = Get-AccessTokenWithAzIdentity -Audience Keyvault 
Get-KeyvaultSecretValue -KeyVaultName '<your key vault>' -SecretName '<your secret>' -AccessToken $tokenKV  
``` 

Here the result: 

![rest01](/assets/img/2023-02-19/02.png)

I’m using this technique very often, and it’s working really well. You can also fetch certificate and Keys the same way. 

{% include note.html content="Don’t forget to add the RBAC (Key vault secret user)" %}

# Use the Az.Keyvault module 

When I’ve read the code behind [connect-azaccount](https://github.com/Azure/azure-powershell/blob/main/src/Accounts/Accounts/Account/ConnectAzureRmAccount.cs), I’ve discovered by chance the parameter KeyVaultAccessToken (line 146). Let’s try! 

```Powershell 
$tokenKV = Get-AccessTokenWithAzIdentity -Audience Keyvault 
$token = Get-AccessTokenWithAzIdentity -Audience ARM 
$null = connect-AzAccount -AccessToken $token -AccountId <your ARC appId> -KeyVaultAccessToken $tokenKv 
Get-AzKeyVaultSecret -VaultName '<your key vaul>' -Name '<your secret>' 
``` 

Here the result: 

![module01](/assets/img/2023-02-19/03.png)

{% include note.html content="Yes this is the same cmdlet we ran during the limitation chapter, but this time it’s working!" %}

This is really cool! The limitation we have with the Az.Storage module **does not exist with the Az.keyvault module!** We can connect with both control and Keyvault data plane token at the same time! 

# Secret management module 

Let’s think a little. If we can connect to both control and data plane it means, we should be able to use the [Microsoft.PowerShell.SecretManagement](https://learn.microsoft.com/en-us/powershell/module/microsoft.powershell.secretmanagement/?view=ps-modules) module no? Let’s try! 

{% include note.html content="find-module Microsoft.PowerShell.SecretManagement | install-module" %}

```Powershell 
$tokenKV = Get-AccessTokenWithAzIdentity -Audience Keyvault 
$token = Get-AccessTokenWithAzIdentity -Audience ARM 
$null = connect-AzAccount -AccessToken $token -AccountId <your ARC appId> -KeyVaultAccessToken $tokenKv 
$KVParams = @{ AZKVaultName = "<your vault name>"; SubscriptionId = $AzSubID} 
Register-SecretVault -Module Az.KeyVault -Name KeyVaultStore -VaultParameters $KVParams 
Get-Secret -Name <your secret name> -AsPlainText 
``` 

Here the result: 

![secretmodule01](/assets/img/2023-02-19/04.png)

{% include important.html content="It seems you have to grant RBAC at the vault directly, not at the secret level." %}

I find this one so cool!  

# Conclusion 

As you’ve seen during this article, **ARC helps you to remove the chicken egg/secret 0 from your pipeline**. I really hope Microsoft will add the same feature I’ve discovered with Keyvault to the storage module (and more in fact). See you for the next one.