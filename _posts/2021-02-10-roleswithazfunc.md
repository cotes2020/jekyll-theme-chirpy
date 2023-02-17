---
title: Add roles to an Azure function?
date: 2021-02-10 00:00
categories: [identity]
tags: [identity,Powershell,AzureAD]
---

# Introduction

Following the previous post where I've enabled OAUth on an azure function with easy auth, few minutes after I had another idea... The problem? I have no idea if it's stupid or if it's a good idea :D. I hope someone will be able to tell me... So here the idea, **now we can choose who can execute our Az function**, **why not adding roles** in this picture?. Why? Why not in fact :p. When you read the [doc](https://docs.microsoft.com/en-us/azure/app-service/app-service-authentication-how-to#access-user-claims), it seems that App Service passes user claims to your application by using special headers. What does it means? No idea, let's find out...

{% include important.html content="Again, I have no idea if it's good idea or not, I really hope someone will tell me if yes or no is secured enough. For example here I won't even check any token signature because ... there won't have any tokens to verify." %}

# Demo

To start, we will have to **modify our app registration** and **add roles** to it. You can do it from the manifest or from the GUI. You should now have something like this:

![01](/assets/img/2021-02-10/role01.png)

Let's now **switch to our Enterprise app** and **remove all previous default assignments**. Then **assign two users**. In my lab, one has the two roles (admin + reader) and the other account reader only. You should now see something like this:

![01](/assets/img/2021-02-10/role02.png)

Now it seems that **App service expose some user claims through special headers**. The one we will be interested of is called **x-ms-client-principal** which is base64 encoded json which contains a lot of useful information like caller's IP, roles, objectId,... Now we just have to tweak our function a little:

```powershell
using namespace System.Net

param($Request, $TriggerMetadata)

#https://docs.microsoft.com/en-us/azure/app-service/app-service-authentication-how-to#access-user-claims
$UserInfoStr = $Request.Headers['x-ms-client-principal']
# Decode the base64 string and then return a JSON
$UserInfoObj = [System.Text.Encoding]::UTF8.GetString([System.Convert]::FromBase64String($UserInfoStr)) | ConvertFrom-Json

#Force a type array
$UserRoles = @()
# Populate the array with roles values
$UserRoles += $UserInfoObj.claims | where typ -eq roles | select -ExpandProperty val

#Then dummy if (of course not prod ready)
if("Admin_Role" -in $UserRoles){

    $name = $Request.Query.Name
    $body = "Hello, $name, you can see this because you're an admin!"
}
else{
    $name = $Request.Query.Name
    $body = "Hello, $name, sorry you don't have enough privilege. Try to be a friend with an admin"
}

# Associate values to output bindings by calling 'Push-OutputBinding'.
Push-OutputBinding -Name Response -Value ([HttpResponseContext]@{
    StatusCode = [HttpStatusCode]::OK
    Body = $body
})
```

Now if we do the same thing as the [previous article](https://scomnewbie.github.io/posts/authenticatedazfunc/) and connect wih the admin account we should see:

```powershell
# Here you can use anonymous if you prefer and replace with your URL function
$Uri = "https://testfuncaadfanf.azurewebsites.net/api/httpNoRole?code=EpogLPlkqeSiVlOOvQ73eAnTSfLC8ZSMMz4KrRpEcckNvgxCf/81fw==&name=fanf"
# generate our public tokens
$token = Get-MsalToken -ClientId $clientId -TenantId $TenantId -Scopes  "https://testfuncaadfanf.azurewebsites.net/user_impersonation" -DeviceCode

$Headers = @{
    'Authorization' = $("Bearer " + $token.AccessToken)
    'Content-Type'  = 'application/json'
}

Invoke-RestMethod -Uri $URI -Headers $Headers

```

The function will return **Hello, fanf, you can see this because you're an admin!** and if you do the same with the reader account, instead you should see Hello, fanf, **sorry** you don't have enough privilege. **Try to be a friend with an admin.**

# Conclusion

Again, is it useful? Is it stupid? Does it make any sense to do this within an Az function? I don't know lol. Now my objective will be to ask if it's at least secure. Don't forget you can use this process only if you run your workflow on App Service. If you try to run your function somewhere else, it won't work and you will have to use the other path with ID/Access tokens instead. See you in the next one.

# References

[App Service authentication](https://docs.microsoft.com/en-us/azure/app-service/app-service-authentication-how-to#access-user-claims)
