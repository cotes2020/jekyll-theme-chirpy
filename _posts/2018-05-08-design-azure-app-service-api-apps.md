---
title: Design Azure App Service API Apps
date: 2018-05-08T22:07:41+02:00
author: Wolfgang Ofner
categories: [Cloud]
tags: [70-532, Azure, Certification, Exam, learning]
---
Azure API Apps provide a quick and easy way to create and consume scalable RESTful APIs, using the language of your choice. The Azure portal helps you to enable CORS to support access to your API from any client and Swagger support makes generating client code to use your API simple.

## Create and deploy API Apps

There are different ways to create and deploy your API Apps. You could use Visual Studio to create a new API Apps project and publish it to a new API app. Additionally to Visual Studio, you can use the Azure portal, Azure CLI, or PowerShell to provision a new API App service.

### Creating a new API App from the Azure portal

To create a new API app, follow these steps:

  1. In the Azure portal click on +Create a resource, search for API App and click Create.
  2. On the API App blade, provide a name, subscription, resource group and App Service plan.
  3. Click Create.

<div id="attachment_1282" style="width: 315px" class="wp-caption aligncenter">
  <a href="/wp-content/uploads/2018/05/Create-a-new-API-App.jpg"><img aria-describedby="caption-attachment-1282" loading="lazy" class="size-full wp-image-1282" src="/wp-content/uploads/2018/05/Create-a-new-API-App.jpg" alt="Create a new API App" width="305" height="410" /></a>
  
  <p id="caption-attachment-1282" class="wp-caption-text">
    Create a new API App
  </p>
</div>

After the API App is deployed, you can download a sample project in ASP.Net, NodeJs or Java by clicking on Quickstart under the Deployment menu.

<div id="attachment_1283" style="width: 489px" class="wp-caption aligncenter">
  <a href="/wp-content/uploads/2018/05/Download-a-sample-project.jpg"><img aria-describedby="caption-attachment-1283" loading="lazy" class="size-full wp-image-1283" src="/wp-content/uploads/2018/05/Download-a-sample-project.jpg" alt="Download a sample project" width="479" height="295" /></a>
  
  <p id="caption-attachment-1283" class="wp-caption-text">
    Download a sample project
  </p>
</div>

### Creating and deploying a new API App with Visual Studio

To create a new API App open Visual Studio 2017 and create a new ASP.NET project with the Azure API App template.

<div id="attachment_1287" style="width: 710px" class="wp-caption aligncenter">
  <a href="/wp-content/uploads/2018/05/Create-a-new-ASP.Net-application-with-the-Azure-API-App-template.jpg"><img aria-describedby="caption-attachment-1287" loading="lazy" class="wp-image-1287" src="/wp-content/uploads/2018/05/Create-a-new-ASP.Net-application-with-the-Azure-API-App-template.jpg" alt="Create a new ASP.Net application with the Azure API App template" width="700" height="457" /></a>
  
  <p id="caption-attachment-1287" class="wp-caption-text">
    Create a new ASP.Net application with the Azure API App template
  </p>
</div>

Visual Studio creates a new API App project and adds NuGet packages, such as:

  * Newtsonsoft.Json for deserializing requests and serializing responses to and from API app.
  * Swashbuckle to add Swagger for rich discovery and documentation for your API REST endpoints.

Follow these steps to deploy your API App from Visual Studio:

<li style="list-style-type: none;">
  <ol>
    <li>
      Right-click on your project in the Solution Explorer and click Publish.
    </li>
    <li>
      In the Publish dialog, select the Create New option on the App Service tab and click Publish.
    </li>
    <li>
      On the Create App Service blade, provide an App name, subscription, resource group and hosting plan.
    </li>
  </ol>
</li>

<div id="attachment_1288" style="width: 710px" class="wp-caption aligncenter">
  <a href="/wp-content/uploads/2018/05/Create-App-Service.jpg"><img aria-describedby="caption-attachment-1288" loading="lazy" class="wp-image-1288" src="/wp-content/uploads/2018/05/Create-App-Service.jpg" alt="Create App Service" width="700" height="525" /></a>
  
  <p id="caption-attachment-1288" class="wp-caption-text">
    Create App Service
  </p>
</div>

<ol start="4">
  <li>
    Click Create.
  </li>
  <li>
    After the API is deployed, the browser will be automatically opened.
  </li>
</ol>

## Automate API discovery by using Swashbuckle

Swagger is a popular, open source framework backed by a large ecosystem of tools that helps you design, build, document, and consume your RESTful APIs.

Generating Swagger metadata manually can be a very tedious process. If you build your API using ASP.NET or ASP.NET Core, you can use the Swashbuckle NuGet package to automatically do this for you, saving a lot of time creating the metadata and maintaining it. Additionally, Swashbuckle also contains an embedded version of swagger-ui, which it will automatically serve up once Swashbuckle is installed.

### Use Swashbuckle in your API App project

To work with swagger, follow these steps:

  1. Go to <a href="https://swagger.io/swagger-ui/" target="_blank" rel="noopener">https://swagger.io/swagger-ui/</a> and either download swagger or use the live demo. I used the live demo and entered the path to my API definition, https://wolfgangapi.azurewebsites.net/swagger/docs/v1. You can find the definition in the Azure portal by clicking on API definition under the API menu.
  2. Click Explore and the available API endpoints will be displayed.

<div id="attachment_1289" style="width: 710px" class="wp-caption aligncenter">
  <a href="/wp-content/uploads/2018/05/The-available-endpoints-of-your-API.jpg"><img aria-describedby="caption-attachment-1289" loading="lazy" class="wp-image-1289" src="/wp-content/uploads/2018/05/The-available-endpoints-of-your-API.jpg" alt="The available endpoints of your API" width="700" height="364" /></a>
  
  <p id="caption-attachment-1289" class="wp-caption-text">
    The available endpoints of your API
  </p>
</div>

<ol start="3">
  <li>
    If you get an error message, you probably have to enable CORS. You can do that in the Azure portal, in your Web API by clicking on CORS under the API menu. For more information see <a href="#EnableCORS">Enable CORS to allow clients to consume API and Swagger interfaces</a>.
  </li>
  <li>
    To test your API endpoints, click on one of them, for example, GET /api/Values.
  </li>
  <li>
    Click on Try it out.
  </li>
  <li>
    Click Execute and the result will be displayed after a couple of seconds.
  </li>
</ol>

<div id="attachment_1291" style="width: 710px" class="wp-caption aligncenter">
  <a href="/wp-content/uploads/2018/05/Test-your-API-with-swagger.jpg"><img aria-describedby="caption-attachment-1291" loading="lazy" class="wp-image-1291" src="/wp-content/uploads/2018/05/Test-your-API-with-swagger.jpg" alt="Test your API with swagger" width="700" height="271" /></a>
  
  <p id="caption-attachment-1291" class="wp-caption-text">
    Test your API with swagger
  </p>
</div>

## Enable CORS to allow clients to consume API and Swagger interfaces {#EnableCORS}

To enable CORS (Cross-Origin Resource Sharing), follow these steps:

  1. In the Azure portal on your API App service, click on CORS under the API menu.
  2.  On the CORS blade, enter the allowed origins or enter an asterisk to allow all origins.
  3. Click Save.

<div id="attachment_1284" style="width: 512px" class="wp-caption aligncenter">
  <a href="/wp-content/uploads/2018/05/Enable-CORS.jpg"><img aria-describedby="caption-attachment-1284" loading="lazy" class="wp-image-1284 size-full" src="/wp-content/uploads/2018/05/Enable-CORS.jpg" alt="Enable CORS for your API Apps" width="502" height="537" /></a>
  
  <p id="caption-attachment-1284" class="wp-caption-text">
    Enable CORS
  </p>
</div>

## Use Swagger API metadata to generate client code for an API App

There are tools available to generate client code for your API Apps that have Swagger API definitions, like the Swagger.io online editor.

To generate client code for your API App that has Swagger API metadata, follow these steps:

  1. In the Azure portal, go to your API App and select API definition under the API menu.
  2. On the API definition, copy the API definition.

<div id="attachment_1293" style="width: 625px" class="wp-caption aligncenter">
  <a href="/wp-content/uploads/2018/05/Copy-the-API-definition.jpg"><img aria-describedby="caption-attachment-1293" loading="lazy" class="size-full wp-image-1293" src="/wp-content/uploads/2018/05/Copy-the-API-definition.jpg" alt="Copy the API definition" width="615" height="514" /></a>
  
  <p id="caption-attachment-1293" class="wp-caption-text">
    Copy the API definition
  </p>
</div>

<ol start="3">
  <li>
    With the API definition copied, go to <a href="http://editor.swagger.io/" target="_blank" rel="noopener">http://editor.swagger.io/</a> to use the Swagger.io online editor.
  </li>
  <li>
    Click on File &#8211;> Import URL and paste your API definition.
  </li>
  <li>
    Click OK.
  </li>
</ol>

<div id="attachment_1294" style="width: 710px" class="wp-caption aligncenter">
  <a href="/wp-content/uploads/2018/05/Paste-the-API-definition-into-the-swagger-editor.jpg"><img aria-describedby="caption-attachment-1294" loading="lazy" class="wp-image-1294" src="/wp-content/uploads/2018/05/Paste-the-API-definition-into-the-swagger-editor.jpg" alt="Paste the API definition into the swagger editor" width="700" height="199" /></a>
  
  <p id="caption-attachment-1294" class="wp-caption-text">
    Paste the API definition into the swagger editor
  </p>
</div>

<ol start="6">
  <li>
    The discovered API endpoints will be displayed on the right side.
  </li>
  <li>
    Click on Generate Client and select your desired language.
  </li>
</ol>

<div id="attachment_1295" style="width: 710px" class="wp-caption aligncenter">
  <a href="/wp-content/uploads/2018/05/Download-the-client-application-for-your-desired-language.jpg"><img aria-describedby="caption-attachment-1295" loading="lazy" class="wp-image-1295" src="/wp-content/uploads/2018/05/Download-the-client-application-for-your-desired-language.jpg" alt="Download the client application for your desired language" width="700" height="589" /></a>
  
  <p id="caption-attachment-1295" class="wp-caption-text">
    Download the client application for your desired language
  </p>
</div>

<ol start="8">
  <li>
    The client application will be downloaded as a .zip file.
  </li>
</ol>

## Monitor API Apps

App Service provides built-in monitoring capabilities, such as resource quotas and metrics. You can set up alerts and automatic scaling based on these metrics. Additionally, Azure provides built-in diagnostics to assist with debugging. A combination of the monitoring capabilities and logging should provide you with the information you need to monitor the health of your API App, and determine whether it is able to meet capacity demands.

### Using quotas and metrics

The resource limits of API Apps are defined by the App Service plan associated with the app.

If you exceed the CPI and bandwidth quotas, your app will respond with a 403 HTTP error. Therefore, you should keep an eye on your resource usage. Exceeding memory quotas causes an application reset, and exceeding the file system quota will cause write operations to fail, even to logs. If you need more resources, you can upgrade your App Service plan.

### Enable and review diagnostics logs

To enable and review diagnostics logs, see my last post <a href="https://www.programmingwithwolfgang.com/design-azure-app-service-web-app/#ImplementDiagnostics" target="_blank" rel="noopener">Design Azure App Service Web App</a>.

## Conclusion

This post showed how to create an Azure API App in the Azure portal and how to create and deploy an API application using Visual Studio 2017. Next, I showed how to test your API endpoints with Swagger and how to create a client application using the online editor of Swagger. During the testing process, I showed how to configure CORS, otherwise, you won&#8217;t be able to test your endpoints.

The last section explained how to monitor your API Apps and how quotas work.

For more information about the 70-532 exam get the <a href="http://amzn.to/2EWNWMF" target="_blank" rel="noopener">Exam Ref book from Microsoft</a> and continue reading my blog posts. I am covering all topics needed to pass the exam. You can find an overview of all posts related to the 70-532 exam <a href="https://www.programmingwithwolfgang.com/prepared-for-the-70-532-exam/" target="_blank" rel="noopener">here</a>.