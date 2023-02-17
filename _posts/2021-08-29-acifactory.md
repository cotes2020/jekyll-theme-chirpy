---
title: Kubernetes is not the only way...
date: 2021-08-29 00:00
categories: [powershell]
tags: [Powershell, Container]
---

# Introduction

Before people start to bash me, I just want to clarify few things. I’m not pros or cons Kubernetes (K8s), **I’m actually a K8s n00b** and learn it is in my to-do list. But I understand how it works and what people has to do to make it work properly and securely in a production environment. When you talk about environment **lifecycle**, **cluster upgrades** (control plane, nodes, containers), **networking**, **identity and access management**, **resource wasting**, **internal rebilling**, **cluster spreading** to just name a few topics… We can easily say that **yes, this tool can answer most of the needs** but <span style="color:red">**only if you’re prepared with the extra operational tasks and responsibilities this tool require**</span>. My point is, don't forget there is other solutions to run your container :).

This article can be considered as an alternative option. We will work with Azure Container Instance (ACI) and Azure function. I will demonstrate a recent idea to execute various workflows without too much configuration overhead. During the last article, I’ve explained I need to run scripts that can exceed the 10 minutes limit that Azure Function with consumption SKU has, therefore ACI becomes a perfect fit. Now if we use Azure function to orchestrate your container groups, **how do you manage the state of your variables between all steps**? 

Serverless is stateless, sadly when you have more than one step in your worflow (like step3 depends on step2 which depends on step1) you have to store your state somewhere like a database. But in my case, where I only need to run scripts, I’m only interested in passing variables from one step to another easily and dynamically **without extra infrastructure**.

As usual, this article will use Powershell and I assume the concept can be re-used with other languages. 

{% include important.html content="This demo code is not production ready, here I just want to explain the concept without going to deep." %}

The goal of this article will be to demo a way to execute scripts in a modular manner where all scripts can use the same “concept” without worrying about infrastructure, management, or operation overhead…

# Implementation

The solution look like this:

![Desktop View](/assets/img/2021-08-29/aciv1.png)

1. Is the initial phase where we basically define our dataset of variables we will pass to our pipeline and store in a queue. In addition, we can imagine another Az function with timer trigger which will call the HTTP or you can also replace the HTTP one by the timer one if you don’t need external interaction or body parameter.
2. Is when the Az function is triggered by the previous step. Implementing async method like this make our solution more flexible and scalable.
3. Is when a function will ask the ARM fabric to deploy our ACI with a container hosted in a Container Registry (ACR). Once deployed, our container will do his job...
4. Is once the work is done, the container will contact another Az function to store variables in a second queue to keep our variables for next a later step.
5. Is finally when we ask the fabric to clean the container group.

Let’s deploy this solution and provide explanations for each step to explain how everything works…

As usual, the code will be available on Github [HERE](https://github.com/SCOMnewbie/Azure/tree/master/ACI/Simple_Deployment).

## Resource Groups (RG)

We will need two RGs. The **management one which will oversee the orchestration part** (control plane). And the **deployment RG which will basically host all our container groups** (data plane). The big difference between the two (except the infra of course) is that our Az function will have contributor access on the deployment one and reader to the control plane. I consider this as a safety net and of course the contributor role can be tweaked to be more granular if needed.

## Function App

The function app is the angular stone of this idea, we will use it as the orchestrator. Let’s create a **consumption Powershell based function app and then enabled the system managed identity**. Now that our function has a proper identity, let’s **grant it reader role to the management RG and contributor to the deployment RG**.

{% include note.html content="Instead of contributor, you can fine-grained the RBAC with custom roles." %}

To avoid duplicating code between functions, I’ve created a simple **module called loadme** where we will find all functions we will re-use over and over. Thanks to the portal, it’s easy to add this module to our function app. On the **App Service Editor** menu and the click **Go**.

![01](/assets/img/2021-08-29/01.png)

Now you can easily **create the required Modules folder** and then **drag and drop the loadme module files** like this:

![02](/assets/img/2021-08-29/02.png)

At this point, all functions located under this function app will be available to be called.

Now under App files (left menu), make sure your **profile.ps1** looks like this:

![03](/assets/img/2021-08-29/03.png)

And the **requirements.psd1** like that:

![04](/assets/img/2021-08-29/04.png)

At this point, our functions will be able to read/write into our resource groups. In addition, we’re now prepared to connect to Azure with the Az.Accounts module, deployed to it with the Az.Resources module and finally use our own custom code with the loadme one! **We will come back later to this function app to create the functions themselves**…

## User managed Identity (User MSI)

Because we will **create and delete our container group(s) all the time**, if we decide to use a **system managed identity** for our ACI, we will quickly mess up our RBAC table with things like this:

![05](/assets/img/2021-08-29/05.png)

To avoid this, we will create a **user MSI that our ACI will use during the deployment**. Let’s now create it. **Once done, let’s assigned our previously created function app the role Managed identity operator on the user MSI resource itself**. Effectively, the inherited reader access does not grant enough permission when the function app will deploy our ACI with this user MSI identity.

![06](/assets/img/2021-08-29/06.png)

{% include note.html content="For the person(s) still with me, this User MSI part is not really mandatory. It is because our Template Specs will require it, but long story short we will use it in the next article…" %}

## Template specs

I won’t be kind in this section. My first plan was to use the CLI but it’s not efficient to bring it into our Az function runtime. Then, **I’ve tried the Az.ContainerInstance Powershell module which is, sorry to say this, the worse Powershell module that I’ve seen**. Nothing is working, the doc is inaccurate, some cmdlets does not even work, I’m surprised that this module went through testing phases …

The last choice that I had was in fact to deploy an ARM template directly. But if I don’t want to store a state in a database, do you think I want to store an ARM template in a storage account? This is where **Template specs start to shine**. The setup is simple, you create a resource called template specs, and paste the content of the **ACI.json** file. This is a custom template that I’ve quickly created for this proof of concept where we have to specify a User MSI for later usage. 

## Container Registry (ACR)

Nothing complicated here, create an ACR and sadly make sure the **admin credentials are enabled** … I know it sucks but I didn’t find a way to use Azure AD credentials to fetch a container image. 

Because reader role is not enough again, I’ve granted **contributor role on the ACR resource itself to our function app**. For fine grained RBAC, we have to use custom role again...

![07](/assets/img/2021-08-29/07.png)

### Container

Instead of building it locally with Docker and send it to the ACR, we will ask the ACR to build it for us. Let’s connect to our ACR with our AAD creds (Docker desktop and Az CLI are required):

![acrlogin](/assets/img/2021-08-29/acrlogin.png)

Change directory in the Docker folder and then type:

![acrupload](/assets/img/2021-08-29/acrupload.png)

We should now have a new image called demo/demoacivariable:v1 in our ACR, let’s verify:

![acruploaddone](/assets/img/2021-08-29/acruploaddone.png)

At this point, most of the infrastructure is done. We will now create our various functions that we will host on our function app and explain each of them. Then we will finish to explain what the container is doing.

## Az functions
### The initJob

All functions we will add to our function app can be found [here](https://github.com/SCOMnewbie/Azure/tree/master/ACI/Simple_Deployment/AzFunction).

As the name suggest, this is the starting point. This is where we will configure all variables we will consume during the pipeline. In addition to both calculated and hardcoded variables, the HTTP trigger give us the possibility to add information from the requester through body or query parameters. The main idea with this function is that it’s **the only place where you define your variables for the whole pipeline**.

This function has 2 output bindings. HTTP to give a state back the requestor to propose error control and the queue one because we have to drop our variables somewhere…

Here an example:

![bindings](/assets/img/2021-08-29/bindings.png)

{% include important.html content="No secret should be stored in this hashtable. An option can be a Keyvault reference like we will see later." %}

### DQinitjob

This second function will be triggered by the message added in the initjob queue. This Az function will use custom functions located in the loadme module.

Starting from this function, **no variable should be set manually anymore except the ones you don’t want to pass to the next step**. We only consume what is coming from the previous job (initjob in this case). The **New-VariableFactory** function takes a hashtable as a parameter and create variables with the same name/value in the session. In other words, **if you have a key called ___runId  with the value ‘ABC’ in the initjob, this function will create a variable called $___runId with the value ‘ABC’ in your current Powershell session**.  

Because our function app has an identity and a contributor role applied to the ACR, we will be able to fetch an access token for the https://management.azure.com/ resource that we will use to extract the ACR admin credentials **without using any password**!

With ACI, you can expose environment variables to your container group if you follow a specific format. Now, **because we don’t share secrets**, **why not simply expose everything to our container**? This is where the function **New-ACIEnvGenerator** comes into place and where the “naming convention” starts to make sense. This function will take all keys from the initjob and format them to be consumed by the container.

Here for example, we create new “formatted” variable ($envVars) that will contain all current session variables which are starting with our prefix.

``` Powershell

$EnvVars = New-ACIEnvGenerator -Variables $(Get-Variable -Name '___*' ) # our prefix

```

Then, we just deploy container group with a big splatting:

``` Powershell

$splatting = @{
    Name = $___ContainerName
    ResourceGroupName = $___ACIRG
    TemplateSpecId = $___ACITemplateSpecId
    ContainerName = $___ContainerName
    ImageName = $___Imagename
    EnvironmentVariables = $EnvVars
    imageRegistryCredentialsServer = $___ACRNameFullName
    imageRegistryCredentialsUsername = $($ACRInfo.Username)
    imageRegistryCredentialsPassword = $(ConvertTo-SecureString $($ACRInfo.passwords[0].value) -AsPlainText)
    UserMSIResourceId = $___UserMSIResourceId
}
# Deploy the ACI from template specs
New-AzResourceGroupDeployment @splatting

```

As you can see, in less than 10 lines of code, we **get an access token**, **fetch ACR credential**, **expose dynamic environment variables** to our container and finally **deploy** it!

Once the ACI is created, we can check the exposed environment variables:

![08](/assets/img/2021-08-29/08.png)

And the container logs:

![09](/assets/img/2021-08-29/09.png)

{% include note.html content="A good idea should be to extract logs send them to a storage blob or something else but it’s not the scope of this article." %}

Our container will run once the ACI is deployed but … What our container is doing? For now, this is a **demo script** (DemoACIScript.ps1) so we just display the current environment variables, create a new variable which start with our prefix convention. This is where the function **New-HashtableFactory** is interesting. We basically **create a hashtable with all variables which start with a prefix in our session**. Then we simply send this hashtable to another Azure function with the hashtable as a payload.

The idea with this container is simple. **You always have something to do and you always want to notify something/someone when the job is done**. Now, because we can have more/less data to share to a later step, re-using the dynamic prefixed variables idea help us to **circle back the information**.

### Stage2 (I’m really bad in naming …)

This function just helps us to store the hashtable which comes from the container once the job is done to another queue.
### RemoveInfra

This final function is triggered by the **message posted by the stage2 function** into a second queue. **The message contains all variables from the initjob AND the new variable we’ve added from the container**. With this function, as before, we **re-hydrate** all variables in memory, and then simply **destroy our container group** in this case.

Why do we want to remove our infrastructure? 
* You update your container image, you’re sure you use the latest image everytime.
* You need to pass new environment variables.
* Cleaner resource group helps for my peace of mind …

## Things to keep in mind

As you’ve seen, in term of **responsibility, we let the cloud provider manage almost everything**. We’re only responsible of the orchestration with the Az func and to keep our container up to date. In fact, we can even ask Azure to keep our container up to date automatically with ACR tasks.

Now it’s not because you have less responsibilities to manage that there is nothing to do, here what I have in mind to improve the idea:
* Several has to be monitored, here some ideas:
    * Poison’s queues
    * ACI deletion fail
    * Number of runs per day with Azure monitor
* Download the ACI logs (from your container) and upload it somewhere before we delete the infrastructure.
# Conclusion

During this article, we’ve seen a way to deploy and orchestrate container(s) easily without a lot of configurations overhead. Effectively, using PaaS products allow us to mainly focus on what we want to run and not what we have to do to make our code to run effectively and securely. In the next article we will add more Azure AD topics to see how we can rely on Keyvault to pass secrets! See you in the next one.


