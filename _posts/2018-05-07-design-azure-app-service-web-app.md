---
title: Design Azure App Service Web App
date: 2018-05-07T18:47:57+02:00
author: Wolfgang Ofner
categories: [Cloud]
tags: [70-532, Azure, Certification, Exam, learning]
---
Azure Web App provides a managed service for hosting your web applications and APIs with infrastructure services such as security, load balancing, and scaling provided as part of the service. Additionally, it has integrated DevOps experience from code repositories and from Docker image repositories. You pay for computing resources according to your App Service Plan and scale settings.

## Define and manage App Service plans

An App Service plan defines the supported feature set and capacity of a group of virtual machine resources that are hosting one or more web apps, logic apps, mobile apps, or API apps.

Each App Service plan is configured with a pricing tier. There are four pricing tiers available: Free, Shared, Basic, and Standard). An App Service plan is unique to the region, resource group, and subscription.

### Creating a new App Service plan

To create a new App Service plan, follow these steps:

  1. In the Azure portal, click on +Create a resource, search for App Service Plan and click Create.
  2. On the New App Service Plan blade, provide an App service plan name, subscription, resource group, operating system, location, and pricing tier.
  3. Click Create.

<div id="attachment_1267" style="width: 318px" class="wp-caption aligncenter">
  <a href="https://www.programmingwithwolfgang.com/wp-content/uploads/2018/05/Create-a-new-App-Service-plan.jpg"><img aria-describedby="caption-attachment-1267" loading="lazy" class="size-full wp-image-1267" src="https://www.programmingwithwolfgang.com/wp-content/uploads/2018/05/Create-a-new-App-Service-plan.jpg" alt="Create a new App Service plan" width="308" height="460" srcset="https://www.programmingwithwolfgang.com/wp-content/uploads/2018/05/Create-a-new-App-Service-plan.jpg 308w, https://www.programmingwithwolfgang.com/wp-content/uploads/2018/05/Create-a-new-App-Service-plan-201x300.jpg 201w" sizes="(max-width: 308px) 100vw, 308px" /></a>
  
  <p id="caption-attachment-1267" class="wp-caption-text">
    Create a new App Service plan
  </p>
</div>

### Creating a new Web App

To create a new Web App, follow these steps:

  1. In the Azure portal, click on +Create a resource, search for Web App and click Create.
  2. On the Web App blade, provide an App name, subscription, resource group, OS, and  App Service plan. You can take either the previously created one or create a new one.
  3. Click Create.

<div id="attachment_1268" style="width: 577px" class="wp-caption aligncenter">
  <a href="https://www.programmingwithwolfgang.com/wp-content/uploads/2018/05/Create-a-new-Web-App.jpg"><img aria-describedby="caption-attachment-1268" loading="lazy" class="size-full wp-image-1268" src="https://www.programmingwithwolfgang.com/wp-content/uploads/2018/05/Create-a-new-Web-App.jpg" alt="Create a new Web App" width="567" height="459" srcset="https://www.programmingwithwolfgang.com/wp-content/uploads/2018/05/Create-a-new-Web-App.jpg 567w, https://www.programmingwithwolfgang.com/wp-content/uploads/2018/05/Create-a-new-Web-App-300x243.jpg 300w" sizes="(max-width: 567px) 100vw, 567px" /></a>
  
  <p id="caption-attachment-1268" class="wp-caption-text">
    Create a new Web App
  </p>
</div>

### Review App Service plan settings

After you have created a new App Service plan, you can change the settings, following these steps:

  1. In the Azure portal, open your App Service plans. There you can see all your App Service plans, the number of apps deployed for each plan and its pricing tier.

<div id="attachment_1269" style="width: 710px" class="wp-caption aligncenter">
  <a href="https://www.programmingwithwolfgang.com/wp-content/uploads/2018/05/An-overview-of-my-App-Service-plans.jpg"><img aria-describedby="caption-attachment-1269" loading="lazy" class="wp-image-1269" src="https://www.programmingwithwolfgang.com/wp-content/uploads/2018/05/An-overview-of-my-App-Service-plans.jpg" alt="An overview of my App Service plans" width="700" height="171" srcset="https://www.programmingwithwolfgang.com/wp-content/uploads/2018/05/An-overview-of-my-App-Service-plans.jpg 1209w, https://www.programmingwithwolfgang.com/wp-content/uploads/2018/05/An-overview-of-my-App-Service-plans-300x73.jpg 300w, https://www.programmingwithwolfgang.com/wp-content/uploads/2018/05/An-overview-of-my-App-Service-plans-768x188.jpg 768w, https://www.programmingwithwolfgang.com/wp-content/uploads/2018/05/An-overview-of-my-App-Service-plans-1024x251.jpg 1024w" sizes="(max-width: 700px) 100vw, 700px" /></a>
  
  <p id="caption-attachment-1269" class="wp-caption-text">
    An overview of my App Service plans
  </p>
</div>

<ol start="2">
  <li>
    Click on your App Service plan to manage its settings.
  </li>
  <li>
    You can scale up or scale out your Service plan or you can integrate a VNET.
  </li>
</ol>

## Configure Web App settings

You can configure the following groups of settings for your Web App application:

  * Load balancing
  * IIS related settings
  * App settings and connection strings
  * Slot management
  * Debugging
  * Application type and library versions

### Configure Web App settings in the Azure Portal

To manage your Web App settings, follow these steps:

  1. In the Azure portal, open the Webb App, you want to configure and select Application settings under the Settings menu.
  2. On the Application settings, you can choose the following settings: 
      * Select language support for .Net, PHP, Python, or Java and set the desired version of the language.
      * Choose between a 32 or 64 bit runtime environment.
      * Enable or disable web sockets.
      * Enable Always On. This configures that the web application will be kept in the memory all the time. Therefore the load time for the next request is reduced.
      * Chose the type of pipeline for IIS. Integrated is the more modern pipeline and Classic would only be used for legacy applications.

<div id="attachment_1270" style="width: 710px" class="wp-caption aligncenter">
  <a href="https://www.programmingwithwolfgang.com/wp-content/uploads/2018/05/Change-the-language-and-runtime-settings-of-your-application.jpg"><img aria-describedby="caption-attachment-1270" loading="lazy" class="wp-image-1270" src="https://www.programmingwithwolfgang.com/wp-content/uploads/2018/05/Change-the-language-and-runtime-settings-of-your-application.jpg" alt="Change the language and runtime settings of your application" width="700" height="536" srcset="https://www.programmingwithwolfgang.com/wp-content/uploads/2018/05/Change-the-language-and-runtime-settings-of-your-application.jpg 835w, https://www.programmingwithwolfgang.com/wp-content/uploads/2018/05/Change-the-language-and-runtime-settings-of-your-application-300x230.jpg 300w, https://www.programmingwithwolfgang.com/wp-content/uploads/2018/05/Change-the-language-and-runtime-settings-of-your-application-768x588.jpg 768w" sizes="(max-width: 700px) 100vw, 700px" /></a>
  
  <p id="caption-attachment-1270" class="wp-caption-text">
    Change the language and runtime settings of your application
  </p>
</div>

<ol start="3">
  <li>
    Turn on ARR Affinity to enable sticky sessions. Sticky session means that a user will always be routed to the same host machine. The downside of sticky sessions is that the performance might be lower.
  </li>
</ol>

<div id="attachment_1271" style="width: 561px" class="wp-caption aligncenter">
  <a href="https://www.programmingwithwolfgang.com/wp-content/uploads/2018/05/ARR-Affinity-settings.jpg"><img aria-describedby="caption-attachment-1271" loading="lazy" class="size-full wp-image-1271" src="https://www.programmingwithwolfgang.com/wp-content/uploads/2018/05/ARR-Affinity-settings.jpg" alt="ARR Affinity settings" width="551" height="124" srcset="https://www.programmingwithwolfgang.com/wp-content/uploads/2018/05/ARR-Affinity-settings.jpg 551w, https://www.programmingwithwolfgang.com/wp-content/uploads/2018/05/ARR-Affinity-settings-300x68.jpg 300w" sizes="(max-width: 551px) 100vw, 551px" /></a>
  
  <p id="caption-attachment-1271" class="wp-caption-text">
    ARR Affinity settings
  </p>
</div>

<ol start="4">
  <li>
    When you first create your web app, the auto swap settings are disabled. You must first create a new slot and from the slow, you may configure auto swap to another slot.
  </li>
  <li>
    The next setting is a really cool one, remote debugging. Enable remote debugging if you run into situations where deployed applications are not functioning as expected. Remote debugging can be enabled for Visual Studio 2012 &#8211; 2017.
  </li>
</ol>

<div id="attachment_1272" style="width: 486px" class="wp-caption aligncenter">
  <a href="https://www.programmingwithwolfgang.com/wp-content/uploads/2018/05/Remote-debugging.jpg"><img aria-describedby="caption-attachment-1272" loading="lazy" class="size-full wp-image-1272" src="https://www.programmingwithwolfgang.com/wp-content/uploads/2018/05/Remote-debugging.jpg" alt="Remote debugging" width="476" height="144" srcset="https://www.programmingwithwolfgang.com/wp-content/uploads/2018/05/Remote-debugging.jpg 476w, https://www.programmingwithwolfgang.com/wp-content/uploads/2018/05/Remote-debugging-300x91.jpg 300w" sizes="(max-width: 476px) 100vw, 476px" /></a>
  
  <p id="caption-attachment-1272" class="wp-caption-text">
    Remote debugging
  </p>
</div>

<ol start="6">
  <li>
    The application settings override any settings with the same name from your application. The connection strings override the matching ones from your application as well. The value of the connection string is hidden unless you click on it.
  </li>
</ol>

<div id="attachment_1273" style="width: 710px" class="wp-caption aligncenter">
  <a href="https://www.programmingwithwolfgang.com/wp-content/uploads/2018/05/Application-and-connection-settings.jpg"><img aria-describedby="caption-attachment-1273" loading="lazy" class="wp-image-1273" src="https://www.programmingwithwolfgang.com/wp-content/uploads/2018/05/Application-and-connection-settings.jpg" alt="Application and connection settings" width="700" height="218" srcset="https://www.programmingwithwolfgang.com/wp-content/uploads/2018/05/Application-and-connection-settings.jpg 1030w, https://www.programmingwithwolfgang.com/wp-content/uploads/2018/05/Application-and-connection-settings-300x93.jpg 300w, https://www.programmingwithwolfgang.com/wp-content/uploads/2018/05/Application-and-connection-settings-768x239.jpg 768w, https://www.programmingwithwolfgang.com/wp-content/uploads/2018/05/Application-and-connection-settings-1024x319.jpg 1024w" sizes="(max-width: 700px) 100vw, 700px" /></a>
  
  <p id="caption-attachment-1273" class="wp-caption-text">
    Application and connection settings
  </p>
</div>

## Configure Web App certificates and custom domains

When you first create your web app, it is accessible through yourwebappname.azurewebsites.net. To map to a more user-friendly domain, you must set up a custom domain name.

If you use HTTPS, you need to utilize an SSL certificate. You can use your SSL certificate with your web app in one of two ways:

  1. You can use the built-in wildcard SSL certificate that is associated with the *.azurewebsites.net domain.
  2. More commonly you use a certificate you can purchase for your custom domain from a third-party authority.

### Mapping custom domain names

The mapping is captured in domain name system (DNS) records that are maintained by your domain registrar. Two types of DNS records effectively express this purpose:

  1. A records (address records) map your domain name to the UP address of your website.
  2. CNAME records (alias records) map a subdomain of your custom domain name to the canonical name of your website,

CNAME records enable you to map only subdomains.

### Configuring a custom domain

You need to access your domain registrar setup for the domain while also editing the configuration for your web app in the Azure portal. Note that custom domain names are not supported by the Free App service plan pricing tier.

To configure a custom domain, follow these steps:

  1. In the Azure portal, go to your Web App and select Custom domains under the Settings menu. Custom domains are not available on the Free pricing tier.
  2. On the Custom domains blade, click on +Add hostname and enter your Hostname. Then choose if you want to set up an A record or CNAME record.

<div id="attachment_1274" style="width: 710px" class="wp-caption aligncenter">
  <a href="https://www.programmingwithwolfgang.com/wp-content/uploads/2018/05/Add-a-hostname-for-your-custom-domain.jpg"><img aria-describedby="caption-attachment-1274" loading="lazy" class="wp-image-1274" src="https://www.programmingwithwolfgang.com/wp-content/uploads/2018/05/Add-a-hostname-for-your-custom-domain.jpg" alt="Add a hostname for your custom domain" width="700" height="160" srcset="https://www.programmingwithwolfgang.com/wp-content/uploads/2018/05/Add-a-hostname-for-your-custom-domain.jpg 1694w, https://www.programmingwithwolfgang.com/wp-content/uploads/2018/05/Add-a-hostname-for-your-custom-domain-300x69.jpg 300w, https://www.programmingwithwolfgang.com/wp-content/uploads/2018/05/Add-a-hostname-for-your-custom-domain-768x175.jpg 768w, https://www.programmingwithwolfgang.com/wp-content/uploads/2018/05/Add-a-hostname-for-your-custom-domain-1024x234.jpg 1024w" sizes="(max-width: 700px) 100vw, 700px" /></a>
  
  <p id="caption-attachment-1274" class="wp-caption-text">
    Add a hostname for your custom domain
  </p>
</div>

<ol start="3">
  <li>
    To set up an A record, select A Record and follow the instruction in the blade. The instructions are: <ul>
      <li>
        First add a TXT record at your domain name registrar, pointing to the default Azure domain for your web app, to verify you own the domain name. The new TXT record should point to yourwebappname.azurewebsites.net.
      </li>
      <li>
        Additionally, add an A record pointing to the IP address shown on the blade, for your web app.
      </li>
    </ul>
  </li>
  
  <li>
    To set up a CNAME record, select CNAME Record and follow the instruction in the blade. The instructions are: <ul>
      <li>
        If using a CNAME record, following the instructions provided by your domain name registrar, add a new CNAME record with the name of the subdomain, and for the value, specify your web app&#8217;s default Azure domain with yourwebappname.azurewebsites.net.
      </li>
    </ul>
  </li>
  
  <li>
    Save your DNS changes. Note that it may take up to 48 hours to propagate the changes across DNS.
  </li>
  <li>
    Click Add Hostname again to configure your custom domain. Enter the domain name and select Validate again. If validation passed, select Add Hostname to complete the assignment.
  </li>
</ol>

### Configure SSL certificates

To configure SSL certificates for your custom domain, you need to have access to an SSL certificate that includes your custom domain name, including the CNAME if it is not a wildcard certificate.

To assign an SSL certificate, follow these steps:

  1. In the Azure portal, go to your Web App and select SSL Configuration under the Settings menu.
  2. On the SSL settings blade, import an existing certificate, or upload a new one.
  3. Select Add Binding to set up the correct binding. You can set up binding that point at your naked domain (programmingwithwolfgang.com), or to a particular CNAME (www.programmingwithwolfgang.com), as long as the certificate supports it.
  4. Select ServerName Indication (SNI) or IP Based SSL as SSL Type.

## Manage Web Apps by using the API, Azure PowerShell, and Xplat-CLI

Additionally, to managing your Web App in the Azure portal, programmatic or script-based access is available.

The following options are available:

<div class="table-responsive">
  <table class="table table-striped table-bordered table-hover">
    <tr>
      <td>
        Configuration Method
      </td>
      
      <td>
        Description
      </td>
    </tr>
    
    <tr>
      <td>
        Azure Resource Manager (ARM)
      </td>
      
      <td>
        Azure Resource Manager provides a consistent management layer for the management tasks you can perform using PowerShell, Azure CLI, Azure portal, REST API, and other development tools.
      </td>
    </tr>
    
    <tr>
      <td>
        REST API
      </td>
      
      <td>
        The REST API enables you to deploy and manage Azure infrastructure resources using HTTP request and JSON payloads.
      </td>
    </tr>
    
    <tr>
      <td>
        PowerShell
      </td>
      
      <td>
        PowerShell provides cmdlets for interacting with ARM to manage infrastructure resources. The PowerShell modules can be installed to Windows, macOS, or Linux.
      </td>
    </tr>
    
    <tr>
      <td>
        Azure CL
      </td>
      
      <td>
        Azure CLI (also known as XplatCLI) is a command line experience for managing Azure resources. This is an open source SDK that works on Windows, macOS, and Linux platforms to create, manage, and monitor web apps.
      </td>
    </tr>
  </table>
</div>

## Implement diagnostics, monitoring, and analytics {#ImplementDiagnostics}

Without diagnostics, monitoring, and analytics, you can&#8217;t effectively investigate the cause of a failure, nor can you proactively prevent potential problems before your users experience them. Web Apps provide multiple forms of logs, features for monitoring availability and automatically sending an email alert when the availability crosses a threshold, features for monitoring your web app resource usage, and integration with Azure Analytics via Application Insights.

App Services are also governed by quotas depending on the App Service plan you have chosen. Free and Shared apps have CPU, memory, bandwidth, and file system quotas. When reached, the web app no longer runs until the next cycle, or the App Service plan is changed. Basic, Standard, and Premium App Services are only limited by file system quotas based on the SKU size selected for the host.

### Configure diagnostic logs

There are five different types of logs:

<div class="table-responsive">
  <table class="table table-striped table-bordered table-hover">
    <tr>
      <td>
        Log
      </td>
      
      <td>
        Description
      </td>
    </tr>
    
    <tr>
      <td>
        Event Log
      </td>
      
      <td>
        The Event logs are equivalent to the Windows Event Log on a Windows Server. It is useful for capturing unhandled exceptions. Only one XML file is created per web app.
      </td>
    </tr>
    
    <tr>
      <td>
        Web server logs
      </td>
      
      <td>
        Web server logs are textual files that create a text entry for each HTTP request to the web app.
      </td>
    </tr>
    
    <tr>
      <td>
        Detailed error message logs
      </td>
      
      <td>
        These HTML flies are generated by the web server and log the error messages for failed requests that result in an HTTP status code of 400 or higher. One error message is captured per HTML file.
      </td>
    </tr>
    
    <tr>
      <td>
        Failed request tracing logs
      </td>
      
      <td>
        Additionally to the error message, the stack trace that led to a failed HTTP request is captured in these XML documents that are present with an XSL style sheet for in-browser consumption. One failed request trace is captured per XML file.
      </td>
    </tr>
    
    <tr>
      <td>
        Application diagnostic logs
      </td>
      
      <td>
        These text-based trace logs are created by web application code in a manner specific to the platform the application is built in using logging or tracing utilities.
      </td>
    </tr>
  </table>
</div>

To enable these diagnostic settings, follow these steps:

  1. In the Azure portal, open your Webb App and click Diagnostic logs under the Monitoring menu.
  2. On the Diagnostics logs blade, enable your desired logs.
  3. If you enabled Application Logging, you have to provide a storage account.
  4. If you enabled Web server logging, you have to provide either a storage account or you can log to the file system.
  5. Click Save.

<div id="attachment_1275" style="width: 479px" class="wp-caption aligncenter">
  <a href="https://www.programmingwithwolfgang.com/wp-content/uploads/2018/05/Configure-logging-for-your-Web-App-application.jpg"><img aria-describedby="caption-attachment-1275" loading="lazy" class="wp-image-1275" src="https://www.programmingwithwolfgang.com/wp-content/uploads/2018/05/Configure-logging-for-your-Web-App-application.jpg" alt="Configure logging for your Web App application" width="469" height="700" srcset="https://www.programmingwithwolfgang.com/wp-content/uploads/2018/05/Configure-logging-for-your-Web-App-application.jpg 527w, https://www.programmingwithwolfgang.com/wp-content/uploads/2018/05/Configure-logging-for-your-Web-App-application-201x300.jpg 201w" sizes="(max-width: 469px) 100vw, 469px" /></a>
  
  <p id="caption-attachment-1275" class="wp-caption-text">
    Configure logging for your Web App application
  </p>
</div>

<ol start="6">
  <li>
    If you enabled Application Logging or Web server logging to the file system, you can see the logs as a stream by clicking on Log stream under the Monitoring blade.
  </li>
</ol>

<div id="attachment_1276" style="width: 710px" class="wp-caption aligncenter">
  <a href="https://www.programmingwithwolfgang.com/wp-content/uploads/2018/05/Application-log-stream.jpg"><img aria-describedby="caption-attachment-1276" loading="lazy" class="wp-image-1276" src="https://www.programmingwithwolfgang.com/wp-content/uploads/2018/05/Application-log-stream.jpg" alt="Application log stream" width="700" height="341" srcset="https://www.programmingwithwolfgang.com/wp-content/uploads/2018/05/Application-log-stream.jpg 1679w, https://www.programmingwithwolfgang.com/wp-content/uploads/2018/05/Application-log-stream-300x146.jpg 300w, https://www.programmingwithwolfgang.com/wp-content/uploads/2018/05/Application-log-stream-768x374.jpg 768w, https://www.programmingwithwolfgang.com/wp-content/uploads/2018/05/Application-log-stream-1024x499.jpg 1024w" sizes="(max-width: 700px) 100vw, 700px" /></a>
  
  <p id="caption-attachment-1276" class="wp-caption-text">
    Application log stream
  </p>
</div>

### Configure endpoint monitoring

You can monitor many different resources such as:

  * Average Response Time
  * CPU Time
  * HTTP 2xx
  * Data In
  * Data Out
  * HTTP 3xx
  * HTTP 4xx
  * Requests
  * HTTP Server Errors

To customize the shown metrics, follow these steps:

  1. In the Azure portal, open your Webb App and click on one of the graphs on the Overview blade.
  2. On the Metrics blade, add or remove all metrics, you want to display on the graph.
  3. Enter a title, select a chart type and set a time range.

<div id="attachment_1277" style="width: 710px" class="wp-caption aligncenter">
  <a href="https://www.programmingwithwolfgang.com/wp-content/uploads/2018/05/Customize-the-metrics-graph.jpg"><img aria-describedby="caption-attachment-1277" loading="lazy" class="wp-image-1277" src="https://www.programmingwithwolfgang.com/wp-content/uploads/2018/05/Customize-the-metrics-graph.jpg" alt="Customize the metrics graph" width="700" height="376" srcset="https://www.programmingwithwolfgang.com/wp-content/uploads/2018/05/Customize-the-metrics-graph.jpg 1679w, https://www.programmingwithwolfgang.com/wp-content/uploads/2018/05/Customize-the-metrics-graph-300x161.jpg 300w, https://www.programmingwithwolfgang.com/wp-content/uploads/2018/05/Customize-the-metrics-graph-768x413.jpg 768w, https://www.programmingwithwolfgang.com/wp-content/uploads/2018/05/Customize-the-metrics-graph-1024x550.jpg 1024w" sizes="(max-width: 700px) 100vw, 700px" /></a>
  
  <p id="caption-attachment-1277" class="wp-caption-text">
    Customize the metrics graph
  </p>
</div>

<ol start="4">
  <li>
    Additionally, you can set an alarm, so you get an email when a certain threshold is reached, for example, 90% CPU usage.
  </li>
  <li>
    Click Save and close.
  </li>
</ol>

## Design and configure a Web App for scale and resilience

App Services can scale up and down by adjusting the instance size and scale out or in by changing the number of instances serving requests. Scaling can be configured to be automatic. The advantage of automatic scaling up or out is that at peaks the resources are increased. As a result, your customers don&#8217;t experience any problems with the performance. During less busy hours, scaling down or helps you to save costs.

Additionally, to scaling within the same datacenter, you can also scale a web app by deploying to multiple regions around the globe and then utilizing Microsoft Azure Traffic Manager to direct web app traffic to the appropriate region based on a round-robin strategy or according to performance. Alternately, you can configure Traffic Manager to use alternate regions as targets for failover if the primary region becomes unavailable.

### Configure scaling

To configure scaling for your web app, follow these steps.

  1. In the Azure portal, open your Service Plan and click Scale up under the Settings blade.
  2. On the Scale up blade, select your desired pricing tier and click Apply.
  3. Select Scale out under the Settings blade and select the desired instance count to scale out to.

<div id="attachment_1278" style="width: 710px" class="wp-caption aligncenter">
  <a href="https://www.programmingwithwolfgang.com/wp-content/uploads/2018/05/Configure-scaling-out.jpg"><img aria-describedby="caption-attachment-1278" loading="lazy" class="wp-image-1278" src="https://www.programmingwithwolfgang.com/wp-content/uploads/2018/05/Configure-scaling-out.jpg" alt="Configure scaling out" width="700" height="335" srcset="https://www.programmingwithwolfgang.com/wp-content/uploads/2018/05/Configure-scaling-out.jpg 1180w, https://www.programmingwithwolfgang.com/wp-content/uploads/2018/05/Configure-scaling-out-300x144.jpg 300w, https://www.programmingwithwolfgang.com/wp-content/uploads/2018/05/Configure-scaling-out-768x368.jpg 768w, https://www.programmingwithwolfgang.com/wp-content/uploads/2018/05/Configure-scaling-out-1024x490.jpg 1024w" sizes="(max-width: 700px) 100vw, 700px" /></a>
  
  <p id="caption-attachment-1278" class="wp-caption-text">
    Configure scaling out
  </p>
</div>

<ol start="4">
  <li>
    If you select Enable autoscale, you can create conditions based on metrics and rules in order for the site to automatically scale out or in.
  </li>
</ol>

## Conclusion

This post talked about how App Service is created and how the settings can be changed. Then, I showed that you can configure your Web App directly in the Azure portal and that you can override the settings from your application can be overridden.

Next, I showed how to apply a custom domain and an SSL certificate to your Web App. Note that certificates and custom domains are not supported by the Free pricing tier.

After the custom domain and SSL was configured, I showed how to enable various logs and how to customize the monitoring graphs to give you exactly the information you need.

The last section talked about scaling out or up to increase the used resources and scaling in or down to save costs during less busy hours.

For more information about the 70-532 exam get the <a href="http://amzn.to/2EWNWMF" target="_blank" rel="noopener">Exam Ref book from Microsoft</a> and continue reading my blog posts. I am covering all topics needed to pass the exam. You can find an overview of all posts related to the 70-532 exam <a href="https://www.programmingwithwolfgang.com/prepared-for-the-70-532-exam/" target="_blank" rel="noopener">here</a>.