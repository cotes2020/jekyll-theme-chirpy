---
title: Azure ARC with GCP workflows and Powershell
date: 2023-03-17 00:00
categories: [identity]
tags: [identity, AAD, ARC]
---

# Introduction

In GCP, people use a service account associated with private keys to run their workflows. The problem is that you have to commit the associated private key somewhere. This is where [workload identities](https://cloud.google.com/kubernetes-engine/docs/how-to/workload-identity) start to shine. Long story short, it’s a way to delegate credentials validation (federation) against another identity provider and in my case, as usual it will be Azure AD. The good news is that if you have a managed identity, you can authenticate to GCP without any private keys. Guess what, I have my ARC agent!

During this article, we will cover two scenarios where we will request organization projects with REST APIs first and then with gcloud CLI. Following the 2 previous articles related to Azure ARC where I’ve talked about the [storage API](https://scomnewbie.github.io/posts/arcwithstorageapi/) or the [Keyvault API](https://scomnewbie.github.io/posts/arcwithkeyvaultapi/), this time we will use a custom one. 

# Configuration

## AzureAD

First, we have to create an app registration that will represent our workload identity pool.

![01](/assets/img/2023-03-17/01.png)

Now let's "expose an api":

![02](/assets/img/2023-03-17/02.png)

Then create a Role into the app role menu:

![03](/assets/img/2023-03-17/04.png)

Now switch to the Enterprise app and make sure you’ve checked the assignment required: 

![04](/assets/img/2023-03-17/03.png)

Now we know that only assigned people or application will be able to fetch an AAD token.
Finally force the V2 in manifest (required for later):

![05](/assets/img/2023-03-17/13.png)

Let’s now [assign your ARC agent](https://learn.microsoft.com/en-us/graph/api/serviceprincipal-post-approleassignments?view=graph-rest-1.0&tabs=http) (Same agent as previous articles) with graph api. 

{% include note.html content="Because we can’t assign application/Service principal through the portal, we will have to assign it with Graph API." %}

![06](/assets/img/2023-03-17/05.png)

Once done, you will see your arc agent under user and group: 

![07](/assets/img/2023-03-17/06.png)

## GCP

In your project, create a workload identity pool and create a new provider with few parameters like: 

* A pool name 
* The issuer URL with the format https://login.microsoftonline/your aadtenantid/v2.0 thanks to the tokenacceptance 2 in the manifest 
* The allowed audience which is your GCP that represent your app registration. 

![08](/assets/img/2023-03-17/07.png)

Let’s now grant access to a service account: 

![09](/assets/img/2023-03-17/08.png)

Now you have to grant proper RBAC to your service account on you rorganization/project. 

And that’s it. If we summarize, on AAD side, only our ARC agent can generate a token (thanks to the assignment required) and on GCP side, we now have a service account linked to the workload identity pool with proper permission. Let’s now see in action! 

# Action

## REST

Like the previous articles, we will need to load several functions in memory. Let’s start with the [Get-AccessTokenWithAzIdentity](https://raw.githubusercontent.com/SCOMnewbie/Azure/master/Identity-AAD/Get-AccessTokenWithAzIdentity.ps1) function which will help us to generate AAD access tokens for our custom audience from any Azure compute workflows (VM, ACI, ARC, function,...). Let’s now add [Get-GCPAccessTokenFromAAD](https://raw.githubusercontent.com/SCOMnewbie/Azure/master/Identity-AAD/Get-GCPAccessTokenFromAAD.ps1) Which will help us to generate GCP access tokens.  

{% include note.html content="Contrary to Azure, the GCP access tokens are opaque tokens." %}

```Powershell 

$AADToken = Get-AccessTokenWithAzIdentity -Audience Custom -CustomScope "api://<your aad app registration>" 

$splat = @{ 
    WorkloadIdentityPoolProjectNumber = "<project number>" 
    WorkloadIdentityPoolName = "<pool name>" 
    WorkloadIdentityPoolProviderName = "<provider name>" 
    ServiceAccountEmailAddress = "<service account>@<project ID>.iam.gserviceaccount.com" 
    AADAccessToken = $AADToken 
} 

Get-GCPAccessTokenFromAAD @splat

``` 

Here the result:

![10](/assets/img/2023-03-17/09.png)

So now what can we do with this token? This is where you have to (sadly?) use the GCP REST api reference <del>joke</del> eeuuhh website. I didn’t find a central way to find all APIs in the same place, every time I have to go back in the search engine type words like “project rest api reference” and cross my fingers...  

In this article, we will play with the [projects endpoint](https://cloud.google.com/resource-manager/reference/rest/v1/projects).

And then joke number 2, the API versioning: 

![11](/assets/img/2023-03-17/10.png)

Each API has several versions where each version is unique in terms of how you interact with it and the results you will have. Anyway, I don’t want to be an asshole but come on Google, try to be in your customer shoes a little bit … 

Now you know which API and version you want to use, you simply have to call the endpoint with: 

``` Powershell 

$GCPToken = Get-GCPAccessTokenFromAAD @splat 
irm -uri https://cloudresourcemanager.googleapis.com/v3/projects?parent=organizations%<your organization id> -Headers @{'Authorization' = "Bearer $GCPToken"} | % projects | select -f 1 

```

Here the result: 

![12](/assets/img/2023-03-17/11.png)

In other words, we can now, from our AAD token, sign in as a service account and fetch Google Cloud Platform information.  

Because the REST api is really a pain to use (compared to Azure), my first thought was to use the CLI and see if the result if better…

## Gcloud CLI

Because the REST API is complicated, it’s legitimate to say that the CLI will be simpler … The short answer is no sadly. There is no official documentation for what we plan to do (using workload identity), or it’s well hidden.  

According to the [documentation](https://cloud.google.com/sdk/docs/authorizing#:~:text=The%20gcloud%20auth%20login%20command%20authorizes%20access%20by%20using%20workload%20identity%20federation%2C%20which%20provides%20access%20to%20external%20workloads%2C%20or%20by%20using%20a%20service%20account%20key.), the CLI is compatible with workload identities. From the doc, it’s written that using a “configuration” cred-file should be the way to go: 

![13](/assets/img/2023-03-17/14.png)

But do you think there is an example of how to do this with a workload identity where the OIDC access token you send is usable only for one hour? No 😀 

This is where I’ve created another function to help me in this joke. It’s now time to load another function in memory called [New-GCPWorkloadIdentityTemplate](https://raw.githubusercontent.com/SCOMnewbie/Azure/master/Identity-AAD/New-GCPWorkloadIdentityTemplate.ps1). This function will require the AAD token and will create 2 files (yes in fact, it’s two, not one ...). One dedicated file for the whole configuration like before (Project number, WI pool, ...) and another with the generated AAD accesstoken.

```Powershell 

$AADToken = Get-AccessTokenWithAzIdentity -Audience Custom -CustomScope "api://b86d723f-cbfe-42e4-a11b-8efb388befba"

$HashArguments = @{ 
    ProjectNumber = "<project number>" 
    WorkloadIdentityPoolName = 'aad-pool' 
    ProviderName = 'aad-pool' 
    ServiceAccountEmail = "<service account>@<project id>.iam.gserviceaccount.com" 
    AccessToken = $AADToken  
} 

New-GCPWorkloadIdentityTemplate @HashArguments 
gcloud auth login --cred-file="gcpconfig.json" -q #q ) quiet
gcloud projects list --format=json | ConvertFrom-Json | select -f 1 

``` 

Here the result: 

![14](/assets/img/2023-03-17/12.png)

# Conclusion

In summary, I had to find a way to work with GCP in a secure way but in my opinion, this platform requires a lot of work from the provider to really compete with the 2 others. In addition, what I’ve done with GCP, you can do it with **AWS as well if you plan to use the assume role with web identity**. Don’t hesitate to ask if it’s something you like to read. See you in the next one.