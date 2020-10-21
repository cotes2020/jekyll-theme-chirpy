---
title: Implement Azure Functions and WebJobs
date: 2018-07-04T20:00:59+02:00
author: Wolfgang Ofner
categories: [Cloud]
tags: [70-532, Azure, Certification, Exam, learning]
---
Azure Functions is a serverless compute service that enables you to run code-on-demand without having to explicitly provision or manage infrastructure. Use Azure Function to run a script or piece of code in response to a variety of events from sources such as:

  * HTTP requests
  * Timers
  * Webhooks
  * Azure Cosmos DB
  * Blob
  * Queues
  * Event Hub

When it comes to implementing background processing tasks, the main options in Azure are Azure Functions and WebJobs. Azure Functions are built on top of WebJobs.

If you already have an app service running a website or a web API and you require a background process to run in the same context, a WebJob makes the most sense. You can share compute resources between the website or API and the WebJob and the also share libraries between them.

If you want to externalize a process so that it runs and scales independently from your web application or API environment, Azure Functions are the more modern serverless technology to choose.

## Create Azure Functions

  1. In the Azure portal click on +Create a resource, search for Function App and click Create.
  2. Provide a name, subscription, resource group, hosting plan, location, and storage plan.
  3. Click Create.

<div id="attachment_1371" style="width: 322px" class="wp-caption aligncenter">
  <a href="/wp-content/uploads/2018/07/Create-an-Azure-Function.jpg"><img aria-describedby="caption-attachment-1371" loading="lazy" class="wp-image-1371 size-full" src="/wp-content/uploads/2018/07/Create-an-Azure-Function.jpg" alt="Create your first Azure Functions" width="312" height="633" /></a>
  
  <p id="caption-attachment-1371" class="wp-caption-text">
    Create an Azure Function
  </p>
</div>

## Implement a Webhook function

To implement a Webhook function, follow these steps:

  1. Open Visual Studio and make sure that you have the Azure Functions and Web Jobs Tool extension installed.
  2. Click New Project, expand the Visual C# and Cloud node, select Azure Functions and provide a name.
  3. Select the Empty template and click Create.
  4. Right-click on your solution, select Add and then New Item.
  5. Select Azure Function, provide a name and then select Generic WebHook.

<div id="attachment_1372" style="width: 710px" class="wp-caption aligncenter">
  <a href="/wp-content/uploads/2018/07/Create-a-Webhook-for-your-Azure-function.jpg"><img aria-describedby="caption-attachment-1372" loading="lazy" class="wp-image-1372" src="/wp-content/uploads/2018/07/Create-a-Webhook-for-your-Azure-function.jpg" alt="Create a Webhook for your Azure function" width="700" height="623" /></a>
  
  <p id="caption-attachment-1372" class="wp-caption-text">
    Create a Webhook for your Azure function
  </p>
</div>

<ol start="6">
  <li>
    Click OK to generate an initial implementation for your function.
  </li>
  <li>
    Start the application.
  </li>
  <li>
    In the output, you can see the Webhook URL.
  </li>
</ol>

<div id="attachment_1373" style="width: 420px" class="wp-caption aligncenter">
  <a href="/wp-content/uploads/2018/07/The-local-URL-of-your-Azure-Function.jpg"><img aria-describedby="caption-attachment-1373" loading="lazy" class="size-full wp-image-1373" src="/wp-content/uploads/2018/07/The-local-URL-of-your-Azure-Function.jpg" alt="The local URL of your Azure Function" width="410" height="67" /></a>
  
  <p id="caption-attachment-1373" class="wp-caption-text">
    The local URL of your Azure Function
  </p>
</div>

<ol start="9">
  <li>
    Post a JSON payload to the function using any tool that can issue HTTP requests to test the function.
  </li>
</ol>

<div id="attachment_1374" style="width: 710px" class="wp-caption aligncenter">
  <a href="/wp-content/uploads/2018/07/Test-your-Azure-Function.jpg"><img aria-describedby="caption-attachment-1374" loading="lazy" class="wp-image-1374" src="/wp-content/uploads/2018/07/Test-your-Azure-Function.jpg" alt="Test your Azure Function" width="700" height="446" /></a>
  
  <p id="caption-attachment-1374" class="wp-caption-text">
    Test your Azure Function
  </p>
</div>

## Create an event processing function

To create an event processing function, follow these steps:

  1. Open your Function App in the Azure portal and click + sign on the Functions blade.
  2. Select Timer and C# and click Create this function.

<div id="attachment_1375" style="width: 1228px" class="wp-caption aligncenter">
  <a href="/wp-content/uploads/2018/07/Create-an-event-processing-funciton.jpg"><img aria-describedby="caption-attachment-1375" loading="lazy" class="wp-image-1375 size-full" src="/wp-content/uploads/2018/07/Create-an-event-processing-funciton.jpg" alt="Create an event processing function" width="1218" height="548" /></a>
  
  <p id="caption-attachment-1375" class="wp-caption-text">
    Create an event processing function
  </p>
</div>

<ol start="3">
  <li>
    This creates a function that runs based on a timer. You can edit the .json file to adjust the settings for the function.
  </li>
</ol>

## Implement an Azure connected function

To create an Azure connected function using Azure Queues, follow these steps:

  1. Open your Function App in the Azure portal and click + sign on the Functions blade.
  2. Click on Queue trigger.
  3. Select a language, provide a name for the function and queue and select a storage account.
  4. Click Create.

<div id="attachment_1379" style="width: 710px" class="wp-caption aligncenter">
  <a href="/wp-content/uploads/2018/07/Create-a-queue-triggered-function.jpg"><img aria-describedby="caption-attachment-1379" loading="lazy" class="wp-image-1379" src="/wp-content/uploads/2018/07/Create-a-queue-triggered-function.jpg" alt="Create a queue triggered function" width="700" height="259" /></a>
  
  <p id="caption-attachment-1379" class="wp-caption-text">
    Create a queue triggered function
  </p>
</div>

<ol start="5">
  <li>
    The function will be created with a simple implementation already which is triggered when a message is queued.
  </li>
  <li>
    If you already had a storage account, the information will be filled in automatically in the Integrate tab. If not, then you have to enter your storage account, the account key, and the connection string.
  </li>
</ol>

<div id="attachment_1380" style="width: 1473px" class="wp-caption aligncenter">
  <a href="/wp-content/uploads/2018/07/Add-the-storage-account-information-to-your-function.jpg"><img aria-describedby="caption-attachment-1380" loading="lazy" class="size-full wp-image-1380" src="/wp-content/uploads/2018/07/Add-the-storage-account-information-to-your-function.jpg" alt="Add the storage account information to your function" width="1463" height="733" /></a>
  
  <p id="caption-attachment-1380" class="wp-caption-text">
    Add the storage account information to your function
  </p>
</div>

<ol start="7">
  <li>
    To test the function, click Run on the main tab. On the right side, you can see the input and on the bottom, you can see the output.
  </li>
</ol>

<div id="attachment_1383" style="width: 1706px" class="wp-caption aligncenter">
  <a href="/wp-content/uploads/2018/07/Test-your-function-with-the-queue.jpg"><img aria-describedby="caption-attachment-1383" loading="lazy" class="size-full wp-image-1383" src="/wp-content/uploads/2018/07/Test-your-function-with-the-queue.jpg" alt="Test your function with the queue" width="1696" height="776" /></a>
  
  <p id="caption-attachment-1383" class="wp-caption-text">
    Test your function with the queue
  </p>
</div>

## Integrate a function with storage

To create a function integrated with Azure Storage Blobs, follow these steps:

  1. Open your Function App in the Azure portal and click + sign on the Functions blade.
  2. Click on Blob trigger.
  3. Select a language, provide a name for the function and queue and select a storage account.
  4. Click Create.

<div id="attachment_1381" style="width: 710px" class="wp-caption aligncenter">
  <a href="/wp-content/uploads/2018/07/Create-an-integrated-function-with-storage.jpg"><img aria-describedby="caption-attachment-1381" loading="lazy" class="wp-image-1381" src="/wp-content/uploads/2018/07/Create-an-integrated-function-with-storage.jpg" alt="Create an integrated function with storage" width="700" height="242" /></a>
  
  <p id="caption-attachment-1381" class="wp-caption-text">
    Create an integrated function with storage
  </p>
</div>

<ol start="5">
  <li>
    The function will be created with a simple implementation already which is triggered when a message is queued.
  </li>
  <li>
    Azure fills out the storage account in the Integrated tab if you already have one. If not, then you have to enter your storage account, the account key, and the connection string.
  </li>
</ol>

<div id="attachment_1382" style="width: 710px" class="wp-caption aligncenter">
  <a href="/wp-content/uploads/2018/07/Add-the-storage-account-information-to-your-function-with-blob-storage.jpg"><img aria-describedby="caption-attachment-1382" loading="lazy" class="wp-image-1382" src="/wp-content/uploads/2018/07/Add-the-storage-account-information-to-your-function-with-blob-storage.jpg" alt="Add the storage account information to your function with blob storage" width="700" height="345" /></a>
  
  <p id="caption-attachment-1382" class="wp-caption-text">
    Add the storage account information to your function with blob storage
  </p>
</div>

<ol start="7">
  <li>
    To test the function, click Run on the main tab. On the right side, you can see the input and on the bottom, you can see the output. When you run the test for the first time, it will fail probably. If it fails, you probably don&#8217;t have the workitem.txt file in your blob storage.
  </li>
</ol>

## Design and implement a custom binding

Function triggers indicate how a function is invoked. There is a number of predefined triggers, some already discussed in previous sections, including:

  * HTTP triggers
  * Event triggers
  * Queues and topic triggers
  * Storage triggers

Every function must have one trigger. The trigger is usually associated with a data payload that is supplied to the function. Bindings are a declarative way to map data to and from function code. Using the Integrate tab, you can provide connection setting for such a data binding activity.

You can also create custom input and output bindings to assist with reducing code bloat in your functions by encapsulating reusable, declarative work into the binding.

## Debug a Function

You can run your Azure Function locally in your Visual Studio 2017 and debug it like any other C# application.

## API Proxies

If you have a solution with many functions, you will find it can become work to manage given different URLs, naming, and versioning potentially related to each other. An API Proxy acts as a single point of entry to functions from the outside world. Instead of calling the individual function URLs, you provide a proxy as a facade to your different function URLs.

Furthermore, API proxies can modify the requests and responses on the fly.

API proxies make sense in HTTP-bound Azure Functions. They may work for other event-driven functions, however, HTTP triggers are best suited for their functionality. As an alternative, you can use API Management for a fully featured solution.

## Integrate with App Service Plan

Functions can operate in two different modes:

  * With a consumption plan where your function is allocated dynamically to the amount of computing power required to execute under the current load.
  * With an App Service plan where your function is assigned a specific app service hosting plan and is limited to the resources available to that hosting plan.

## Conclusion

This post showed different variations of Azure Functions and how to create them. Since there are so many applications for Azure Functions, it is impossible to go too much into detail. You should know enough to get started on your own though.

For more information about the 70-532 exam get the <a href="http://amzn.to/2EWNWMF" target="_blank" rel="noopener">Exam Ref book from Microsoft</a> and continue reading my blog posts. I am covering all topics needed to pass the exam. You can find an overview of all posts related to the 70-532 exam <a href="https://www.programmingwithwolfgang.com/prepared-for-the-70-532-exam/" target="_blank" rel="noopener">here</a>.