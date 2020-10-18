---
title: Design and Implement DevOps
date: 2018-08-09T22:22:12+02:00
author: Wolfgang Ofner
categories: [DevOps, Cloud]
tags: [70-532, Azure, Certification, Exam, learning]
---
DevOps is a combination of development (Dev) and Operations (Ops). The ultimate goal of DevOps is automation and repeatability which allows for increased deployment frequency without the burden that manual deployments bring.

## **Instrument an application with telemetry**

Application Insights is an extensible analytics service for application developers on multiple platforms that helps you understand the performance and usage of your live applications. With it, you can monitor your web application, collect custom telemetry, automatically detect performance anomalies, and use its powerful analytics tools to help you diagnose issues and understand what users do with your app. It works with web applications hosted on Azure, on-premise, or in another cloud provider.

To get started, you only need to provision an Application Insights resource in Azure, and then install a small package in your application. The things can instrument are not limited just to the web application, but also any background components, and JavaScript within its web pages. You can also pull telemetry from host environments, such as performance counters, Dicker logs, or Azure diagnostics.

The following telemetry can be collected from server web apps:

  * HTTP requests
  * Dependencies such as calls to SQL Databases, HTTP calls to external services, Azure Cosmos DB
  * Exceptions and stack traces
  * Performance Counters, if you use Status Monitor, Azure monitoring or the Application Insights collected writer

The following telemetry can be collected from client web pages:

  * Page view counts
  * AJAX calls requests made from a running script
  * Page view load data
  * User and session counters
  * Authenticated user IDs

The standard telemetry modules that run out of the box when using the Application Insights SDK send load, performance and usage metrics, exception reports, client information such as IP address, and calls to external services.

## **Discover application performance issues by using Application Insights**

System performances depends on several factors. Each factor is typically measured through key performance indicators (KPIs), such as the number of database transactions per second or the volume of network requests.

Application Insights can help you quickly identify any application failures. It also tells you about any performance issues and exceptions.

When you open any Application Insights resource you see basic performance data on the overview blade. Clicking on any of the charts allows you to drill down into the related data to see more detail and related requests, as well as viewing different time ranges.

If your application is built on ASP.NET or ASP.NET Core, you can turn on Application Insight&#8217;s profiling to view detailed profiles of live requests. In addition to displaying hot paths that are using the most response times, the Profiler shows which lines in the application code slowed down performance.

To enable the Profiler, follow these steps:

  1. In your Application Insights resource, select the Performance blade and select Profiler.

<div id="attachment_2106" style="width: 710px" class="wp-caption aligncenter">
  <a href="https://www.programmingwithwolfgang.com/wp-content/uploads/2018/08/Create-a-Profiler.jpg"><img aria-describedby="caption-attachment-2106" loading="lazy" class="wp-image-2106" src="https://www.programmingwithwolfgang.com/wp-content/uploads/2018/08/Create-a-Profiler.jpg" alt="Create a Profiler" width="700" height="306" srcset="https://www.programmingwithwolfgang.com/wp-content/uploads/2018/08/Create-a-Profiler.jpg 1702w, https://www.programmingwithwolfgang.com/wp-content/uploads/2018/08/Create-a-Profiler-300x131.jpg 300w, https://www.programmingwithwolfgang.com/wp-content/uploads/2018/08/Create-a-Profiler-1024x448.jpg 1024w, https://www.programmingwithwolfgang.com/wp-content/uploads/2018/08/Create-a-Profiler-768x336.jpg 768w, https://www.programmingwithwolfgang.com/wp-content/uploads/2018/08/Create-a-Profiler-1536x672.jpg 1536w" sizes="(max-width: 700px) 100vw, 700px" /></a>
  
  <p id="caption-attachment-2106" class="wp-caption-text">
    Create a Profiler
  </p>
</div>

<ol start="2">
  <li>
    Select Add Linked Apps from the top of the Configure Application Insights Profiler blade.
  </li>
  <li>
    Select the application you wish to link to see all its available slots. Click Add to link them to your Application Insights resource.
  </li>
  <li>
    After linking, select Enable Profiler from the of the Configure Application Insights Profiler blade.
  </li>
</ol>

## Deploy Visual Studio Team Services with Continuous Integration (CI) and Continuous Delivery (CD)

Continuous Integration is a practice by which the development team members integrate their work frequently, usually daily. An automated build verifies each integration, typically along with tests to detect integration errors quickly, while it&#8217;s easier and less costly to fix them. The output, or artifacts, generated by the CI systems are fed to the release pipeline to streamline and enable frequent deployments.

Continuous Delivery is a process where the full software delivery lifecycle is automated including tests, and deployment to one or more test and production environments. Azure App Services supports deployment slots, into which you can deploy development, staging, and production builds from the CD process. Automated release pipelines consume the artifacts that the CI systems produce, and deploys them as new version and fixes to existing systems.

Your source code must be hosted in a version control system. VSTS provides GIT and Team Foundation Version Control. Additionally, GitHub, Subversion, Bitbucket, or any other Git repository can be integrated with the Build service.

To configure the CI/CD pipeline from the Azure portal, follow these steps:

  1. In the Azure portal click on +Create a resource, search for Web App and click Create.
  2. Provide a name, subscription and APP Service plan.
  3. Click Create.

<div id="attachment_2107" style="width: 322px" class="wp-caption aligncenter">
  <a href="https://www.programmingwithwolfgang.com/wp-content/uploads/2018/08/Create-a-new-Web-App.jpg"><img aria-describedby="caption-attachment-2107" loading="lazy" class="size-full wp-image-2107" src="https://www.programmingwithwolfgang.com/wp-content/uploads/2018/08/Create-a-new-Web-App.jpg" alt="Create a new Web App for DevOps" width="312" height="460" srcset="https://www.programmingwithwolfgang.com/wp-content/uploads/2018/08/Create-a-new-Web-App.jpg 312w, https://www.programmingwithwolfgang.com/wp-content/uploads/2018/08/Create-a-new-Web-App-203x300.jpg 203w" sizes="(max-width: 312px) 100vw, 312px" /></a>
  
  <p id="caption-attachment-2107" class="wp-caption-text">
    Create a new Web App
  </p>
</div>

<ol start="4">
  <li>
    After the web app is deployed, open it in the Azure portal and select the Deployment Center blade under the Deployment menu.
  </li>
  <li>
    Select VSTS and then click Continue.
  </li>
</ol>

<div id="attachment_2108" style="width: 710px" class="wp-caption aligncenter">
  <a href="https://www.programmingwithwolfgang.com/wp-content/uploads/2018/08/Create-Continuous-Delivery-with-VSTS.jpg"><img aria-describedby="caption-attachment-2108" loading="lazy" class="wp-image-2108" src="https://www.programmingwithwolfgang.com/wp-content/uploads/2018/08/Create-Continuous-Delivery-with-VSTS.jpg" alt="Create Continuous Delivery with VSTS" width="700" height="553" srcset="https://www.programmingwithwolfgang.com/wp-content/uploads/2018/08/Create-Continuous-Delivery-with-VSTS.jpg 1206w, https://www.programmingwithwolfgang.com/wp-content/uploads/2018/08/Create-Continuous-Delivery-with-VSTS-300x237.jpg 300w, https://www.programmingwithwolfgang.com/wp-content/uploads/2018/08/Create-Continuous-Delivery-with-VSTS-1024x809.jpg 1024w, https://www.programmingwithwolfgang.com/wp-content/uploads/2018/08/Create-Continuous-Delivery-with-VSTS-768x607.jpg 768w" sizes="(max-width: 700px) 100vw, 700px" /></a>
  
  <p id="caption-attachment-2108" class="wp-caption-text">
    Create Continuous Delivery with VSTS
  </p>
</div>

<ol start="6">
  <li>
    On the next page, select App Service Kudu as your build server and click Continue.
  </li>
  <li>
    Provide your VSTS Account, select a project, repository, and branch you want to use for the deployment and click Continue.
  </li>
  <li>
    After everything is set up, Azure Continuous Delivery executes the build and initiates the deployment.
  </li>
</ol>

## Deploy CI/CD with third-party platform DevOps tools (Jenkins, GitHub, Chef, Puppet, TeamCity)

Azure allows you to continuously integrate and deploy with any of the leading DevOps tools, targeting any Azure service.

Out of the box, Azure App Services integrates with source code repositories such as GitHub to enable a continuous DevOps workflow.

Follow these steps to enable continuous deployment from a GitHub repository:

  1. Publish your application source code to GitHub.
  2. In the Azure portal, open your web app and select Deployment options under the Deployment menu.
  3. On the Deployment option blade, select GitHub, authorize Azure to use Github, select a project and a branch.
  4. Click OK.

<div id="attachment_2109" style="width: 325px" class="wp-caption aligncenter">
  <a href="https://www.programmingwithwolfgang.com/wp-content/uploads/2018/08/Enable-continuous-deployment-from-GitHub.jpg"><img aria-describedby="caption-attachment-2109" loading="lazy" class="size-full wp-image-2109" src="https://www.programmingwithwolfgang.com/wp-content/uploads/2018/08/Enable-continuous-deployment-from-GitHub.jpg" alt="Enable continuous deployment from GitHub" width="315" height="453" srcset="https://www.programmingwithwolfgang.com/wp-content/uploads/2018/08/Enable-continuous-deployment-from-GitHub.jpg 315w, https://www.programmingwithwolfgang.com/wp-content/uploads/2018/08/Enable-continuous-deployment-from-GitHub-209x300.jpg 209w" sizes="(max-width: 315px) 100vw, 315px" /></a>
  
  <p id="caption-attachment-2109" class="wp-caption-text">
    Enable continuous deployment from GitHub
  </p>
</div>

When you push a change to your repository, your app is automatically updated with the latest changes.

## Conclusion

This post gave a short overview of how to use various tools in Azure to implement DevOps. DevOps is a mix of Development and Operations and should help to deploy faster and with fewer errors due to automation. Azure supports different DevOps tools like VSTS or GitHub.

For more information about the 70-532 exam get the <a href="http://amzn.to/2EWNWMF" target="_blank" rel="noopener noreferrer">Exam Ref book from Microsoft</a> and continue reading my blog posts. I am covering all topics needed to pass the exam. You can find an overview of all posts related to the 70-532 exam <a href="https://www.programmingwithwolfgang.com/prepared-for-the-70-532-exam/" target="_blank" rel="noopener noreferrer">here</a>.