---
title: Implement API Management
date: 2018-07-01T14:55:25+02:00
author: Wolfgang Ofner
categories: [Cloud]
tags: [70-532, Azure, Certification, Exam, learning]
---
Azure API Management is a turnkey solution for publishing, managing, securing, and analyzing APIs to both external and internal customers in minutes. You can create an API gateway for backend services hosted anywhere. Many modern APIs protect themselves by rate-limiting consumers, meaning, limiting how many requests can be made in a certain amount of time. Traditionally, there is a lot of work that goes into that process. When you use API Management to manage your API, you can easily secure it and protect it from abuse and overuse with an API key, JWT validation, IP filtering, and through quotas and rate limits.

If you have several APIs as part of your solution, and they are hosted across several services or platforms, you can group them all behind a single static IP and domain, simplifying communication, protection, and reducing maintenance of consumer software due to API locations changing. You also can scale API Management on demand in one or more geographical locations. Its built-in response caching also help with improving latency and scaling.

Hosting your APIs on the API Management platform also makes it easier for developers to use your APIs, by offering self-service API key management, and an auto-generate API catalog through the developer portal. APIs are also documented and come with code examples, reducing developer on-boarding time using your APIs.

## The components of API Management

API Management is made up of the following components:

  * The API gateway is the endpoint that: 
      * Accepts API calls and routes them to your backends.
      * Verifies API keys, JWT tokens, certificates, and other credentials.
      * Enforces usage quotas and rate limits.
      * Caches backend responses where set up.
      * Logs call metadata for analytics.
  * The publisher portal is the administrative interface where you set up your API program. Use it to: 
      * Define or import API schemas.
      * Package APIs into products.
      * Get insights from analytics.
      * Manage users.
  * The developer portal serves as the main web presence for developers, where they can: 
      * Read the API documentation.
      * Try out an API via the interactive console.
      * Create an account and subscribe to get API keys.
      * Access analytics on their own usage.

## Create managed APIs

Before you can create APIs, you must first create a service instance.

### Create an API Management service

To create an API Management service, follow these steps:

  1. In the Azure portal click on +Create a resource, search for API Management and click Create.
  2. Provide a name, subscription, resource group, location, organization name, and administrator email.
  3. Click Create.

<div id="attachment_1361" style="width: 587px" class="wp-caption aligncenter">
  <a href="https://www.programmingwithwolfgang.com/wp-content/uploads/2018/06/Create-an-API-Management-service-in-the-Azure-Portal.jpg"><img aria-describedby="caption-attachment-1361" loading="lazy" class="size-full wp-image-1361" src="https://www.programmingwithwolfgang.com/wp-content/uploads/2018/06/Create-an-API-Management-service-in-the-Azure-Portal.jpg" alt="Create an API Management service in the Azure Portal" width="577" height="531" srcset="https://www.programmingwithwolfgang.com/wp-content/uploads/2018/06/Create-an-API-Management-service-in-the-Azure-Portal.jpg 577w, https://www.programmingwithwolfgang.com/wp-content/uploads/2018/06/Create-an-API-Management-service-in-the-Azure-Portal-300x276.jpg 300w" sizes="(max-width: 577px) 100vw, 577px" /></a>
  
  <p id="caption-attachment-1361" class="wp-caption-text">
    Create an API Management service in the Azure Portal
  </p>
</div>

### Add a product

A product contains one or more APIs, as well as constraints such as a usage quota and terms of use. You can create several products to group APIs with their own usage rules. Developers can subscribe to a product once it is published, and then begin using its APIs.

To add and publish a new product, follow these steps:

  1. On the API Management service, select the Products blade under the API Management menu.
  2. On the Products blade, click +Add.
  3. Provide a name and description. The remaining fields are settings for the level of protection and can stay as they are.
  4. Change the state to published.
  5. Click on Select API and select Echo API.
  6. Click Create.

<div id="attachment_1362" style="width: 442px" class="wp-caption aligncenter">
  <a href="https://www.programmingwithwolfgang.com/wp-content/uploads/2018/06/Add-a-product-to-you-API-Management-instance.jpg"><img aria-describedby="caption-attachment-1362" loading="lazy" class="wp-image-1362" src="https://www.programmingwithwolfgang.com/wp-content/uploads/2018/06/Add-a-product-to-you-API-Management-instance.jpg" alt="Add a product to you API Management instance" width="432" height="700" srcset="https://www.programmingwithwolfgang.com/wp-content/uploads/2018/06/Add-a-product-to-you-API-Management-instance.jpg 577w, https://www.programmingwithwolfgang.com/wp-content/uploads/2018/06/Add-a-product-to-you-API-Management-instance-185x300.jpg 185w" sizes="(max-width: 432px) 100vw, 432px" /></a>
  
  <p id="caption-attachment-1362" class="wp-caption-text">
    Add a product to you API Management instance
  </p>
</div>

### Create a new API

To create a new API, follow these steps:

  1. On the API Management service, select the APIs blade under the API Management menu.
  2. On the APIs blade, select Blank API (or choose one of the existing templates).
  3. Provide a display name, name, the Web Service URL, which is the HTTP endpoint of your API and an API URL affix.
  4. Select HTTP, HTTPS or both as an URL schema.
  5. Add the previously added product in the Products drop-down.
  6. Click Create.

<div id="attachment_1363" style="width: 710px" class="wp-caption aligncenter">
  <a href="https://www.programmingwithwolfgang.com/wp-content/uploads/2018/06/Create-a-new-API.jpg"><img aria-describedby="caption-attachment-1363" loading="lazy" class="wp-image-1363" src="https://www.programmingwithwolfgang.com/wp-content/uploads/2018/06/Create-a-new-API.jpg" alt="Create a new API" width="700" height="451" srcset="https://www.programmingwithwolfgang.com/wp-content/uploads/2018/06/Create-a-new-API.jpg 892w, https://www.programmingwithwolfgang.com/wp-content/uploads/2018/06/Create-a-new-API-300x193.jpg 300w, https://www.programmingwithwolfgang.com/wp-content/uploads/2018/06/Create-a-new-API-768x495.jpg 768w" sizes="(max-width: 700px) 100vw, 700px" /></a>
  
  <p id="caption-attachment-1363" class="wp-caption-text">
    Create a new API
  </p>
</div>

### Add an operation to your API

Before you can use your new API, you must add one or more operations. Their operations do things like enable service documentation, the interactive API console, or set operation limits.

To add an operation, follow these steps:

  1. On your previously created API, click on +Add operation.
  2. Provide a name, URL and HTTP verb.
  3. Optionally, you can create parameters.
  4. Click Create.

<div id="attachment_1364" style="width: 710px" class="wp-caption aligncenter">
  <a href="https://www.programmingwithwolfgang.com/wp-content/uploads/2018/06/Add-an-operation-to-your-API.jpg"><img aria-describedby="caption-attachment-1364" loading="lazy" class="wp-image-1364" src="https://www.programmingwithwolfgang.com/wp-content/uploads/2018/06/Add-an-operation-to-your-API.jpg" alt="Add an operation to your API" width="700" height="539" srcset="https://www.programmingwithwolfgang.com/wp-content/uploads/2018/06/Add-an-operation-to-your-API.jpg 958w, https://www.programmingwithwolfgang.com/wp-content/uploads/2018/06/Add-an-operation-to-your-API-300x231.jpg 300w, https://www.programmingwithwolfgang.com/wp-content/uploads/2018/06/Add-an-operation-to-your-API-768x591.jpg 768w" sizes="(max-width: 700px) 100vw, 700px" /></a>
  
  <p id="caption-attachment-1364" class="wp-caption-text">
    Add an operation to your API
  </p>
</div>

## Configure API Management policies

API Management policies allow you as the publisher, to determine the behavior of your APIs through configuration, requiring no code changes. There are many built-in policies, like allowing cross-domain calls, authenticate requests or setting rate limits. The policies statements you choose affect both inbound requests and outbound responses. Policies can be applied globally, or scoped to the Product, API, or Operation level.

To configure a policy, follow these steps:

  1. In the API Management service, select the APIs blade and click on All APIs.
  2. Click on +Add on the Inbound processing tab.
  3. Click Code View on the top right corner.
  4. Add your desired policy, for example, to enable CORS from all domains.
  5. After you are done, click Save

<div id="attachment_1365" style="width: 710px" class="wp-caption aligncenter">
  <a href="https://www.programmingwithwolfgang.com/wp-content/uploads/2018/06/Add-a-policy-for-CORS.jpg"><img aria-describedby="caption-attachment-1365" loading="lazy" class="wp-image-1365" src="https://www.programmingwithwolfgang.com/wp-content/uploads/2018/06/Add-a-policy-for-CORS.jpg" alt="Add a policy for CORS" width="700" height="457" srcset="https://www.programmingwithwolfgang.com/wp-content/uploads/2018/06/Add-a-policy-for-CORS.jpg 929w, https://www.programmingwithwolfgang.com/wp-content/uploads/2018/06/Add-a-policy-for-CORS-300x196.jpg 300w, https://www.programmingwithwolfgang.com/wp-content/uploads/2018/06/Add-a-policy-for-CORS-768x501.jpg 768w" sizes="(max-width: 700px) 100vw, 700px" /></a>
  
  <p id="caption-attachment-1365" class="wp-caption-text">
    Add a policy for CORS
  </p>
</div>

To see all available policies, go to the <a href="https://docs.microsoft.com/en-us/azure/api-management/api-management-policies" target="_blank" rel="noopener">Azure documentation</a>.

## Protect APIs with rate limits

Protecting your published APIs by throttling incoming requests is one of the most attractive offerings of API Management. Limiting incoming requests helps you controlling your resource costs, preventing you from unnecessarily scaling up your services to meet unexpected demand. Rate limiting, or throttling, is common practice when providing APIs. Oftentimes, API publishers offer varying levels of access to their APIs. For instance, you may choose to offer a free tier with very restrictive rate limits, and various paid tiers offering higher request rates. This is where API Management&#8217;s product comes into play.

### Create a product to scope rate limits to a group of APIs

To create a rate limited API, follow these steps:

  1. On the API Management service, select the Products blade under the API Management menu.
  2. On the Products blade, click +Add.
  3. Provide a name and description.
  4. Add your previously created API.
  5. Click Create.

<div id="attachment_1366" style="width: 524px" class="wp-caption aligncenter">
  <a href="https://www.programmingwithwolfgang.com/wp-content/uploads/2018/06/Create-a-new-Product-to-limit-the-requests.jpg"><img aria-describedby="caption-attachment-1366" loading="lazy" class="wp-image-1366" src="https://www.programmingwithwolfgang.com/wp-content/uploads/2018/06/Create-a-new-Product-to-limit-the-requests.jpg" alt="Create a new Product to limit the requests" width="514" height="700" srcset="https://www.programmingwithwolfgang.com/wp-content/uploads/2018/06/Create-a-new-Product-to-limit-the-requests.jpg 594w, https://www.programmingwithwolfgang.com/wp-content/uploads/2018/06/Create-a-new-Product-to-limit-the-requests-220x300.jpg 220w" sizes="(max-width: 514px) 100vw, 514px" /></a>
  
  <p id="caption-attachment-1366" class="wp-caption-text">
    Create a new Product to limit the requests
  </p>
</div>

<ol start="6">
  <li>
    After the new product is published, open it and select Policies.
  </li>
  <li>
    Add the following code to the inbound rule: <div>
      <rate-limitcalls=&#8221;10&#8243;renewal-period=&#8221;60&#8243;>
    </div>
    
    <div>
      </rate-limit>
    </div>
    
    <div>
      <quotacalls=&#8221;200&#8243;renewal-period=&#8221;604800&#8243;>
    </div>
    
    <div>
      </quota>
    </div>
  </li>
  
  <li>
    Click Save
  </li>
</ol>

### Advanced rate limiting

If you want to avoid high-usage consumers limit access to occasional users, by using up the pool of available resources, consider using the rate-limit-by-key and quota-by-key policies. These are more flexible rate limiting policies that allow defining an expression to track traffic usage by user-level information such as IP address or user identity.

The following code limits the rate by the users IP address:

<span class="fontstyle0"><rate-limit-by-key calls=&#8221;10&#8243;<br /> renewal-period=&#8221;60&#8243;<br /> counter-key=&#8221;@(context.Request.IpAddress)&#8221; /></span>

## Add caching to improve performance

Caching is a great way to limit your resource consumption, like bandwidth, as well as reduce latency for infrequently changing data.

To add caching to your API, follow these steps:

  1. In the API Management service, click the APIs blade and select the Echo API.
  2. Click on the Operations blade and select Retrieve resource (cached).
  3. Switch to the Caching tab.
  4. Here you can configure caching and also copy the code from the code view to use it in your own API.

## Monitor APIs

API Management provides a few methods by which you can monitor resource usage, service health, activities, and analytics. If you want real-time monitoring, as well as richer debugging, you can enable diagnostics on your logic app and send events to OMS with Log Analytics, or to other services, such as Azure Storage, and Event Hubs. Select the Diagnostics logs under the Monitoring menu from your API Management service, and then select Turn on diagnostics to archive your gateway logs and metrics to a storage account, stream to an Event Hub, or send to Log Analytics on OMS.

Like all Azure resources, you can view several metrics under the Metrics blade or create alerts under the Alerts blade.

## Conclusion

In this post, I introduced Azure API Management. I showed how to create an API Management service, and add APIs, products and operations to it and how to enable and configure policies on different scopes. Additionally to the configuration, I mentioned how to monitor APIs and how to enable alerts.

For more information about the 70-532 exam get the <a href="http://amzn.to/2EWNWMF" target="_blank" rel="noopener">Exam Ref book from Microsoft</a> and continue reading my blog posts. I am covering all topics needed to pass the exam. You can find an overview of all posts related to the 70-532 exam <a href="https://www.programmingwithwolfgang.com/prepared-for-the-70-532-exam/" target="_blank" rel="noopener">here</a>.