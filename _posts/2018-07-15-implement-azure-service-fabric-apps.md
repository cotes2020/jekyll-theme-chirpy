---
title: Design and Implement Azure Service Fabric apps
date: 2018-07-15T21:03:41+02:00
author: Wolfgang Ofner
categories: [Cloud]
tags: [70-532, Azure, Certification, Exam, learning]
---
Azure Service Fabric is a platform that makes it easy to package, deploy, and manage distributed solutions at scale. It provides an easy programming model for building microservice solutions with a simple, familiar, and easy to understand development experience that supports stateless and stateful services and actor patterns. In addition, to providing a packaging and deployment solution for these native components, Service Fabric also supports the deployment of guest executable and containers as part of the same managed and distributes system.

<div class="table-responsive">
  <table class="table table-striped table-bordered table-hover">
    <tr>
      <td>
        Native and executable component
      </td>
      
      <td>
        Description
      </td>
    </tr>
    
    <tr>
      <td>
        Stateless Services
      </td>
      
      <td>
        Stateless-Fabric-aware services that run without a managed state.
      </td>
    </tr>
    
    <tr>
      <td>
        Stateful Services
      </td>
      
      <td>
        Stateless-Fabric-aware services that run with a managed state where the state is close to the compute.
      </td>
    </tr>
    
    <tr>
      <td>
        Actors
      </td>
      
      <td>
        A higher level programming model built on top of stateful services.
      </td>
    </tr>
    
    <tr>
      <td>
        Guest Executable
      </td>
      
      <td>
        Can be any application or service that may be cognizant or not cognizant of Service Fabric.
      </td>
    </tr>
    
    <tr>
      <td>
        Containers
      </td>
      
      <td>
        Both Linux and Windows containers are supported by Service Fabric and may be cognizant or not cognizant of Service Fabric.
      </td>
    </tr>
  </table>
</div>

##  Create a Service Fabric application

A Service Fabric application can consist of one or more services. The application defines the deployment package for the service, and each service can have its own configuration, code, and data. A Service Fabric cluster can host multiple applications, and each has its own independent deployment upgrade lifecycle.

To create a new Service Fabric application, follow these steps:

  1. Open Visual Studio and select File -> New -> Project.
  2. In the New Project dialog, select Service Fabric Application within the Cloud category. Provide a name and click OK.

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2018/07/Create-a-Service-Fabric-application.jpg"><img loading="lazy" src="/assets/img/posts/2018/07/Create-a-Service-Fabric-application.jpg" alt="Create a Service Fabric application" /></a>
  
  <p>
    Create a Service Fabric application
  </p>
</div>

<ol start="3">
  <li>
    Select Stateful Service from the list of services and provide a name.
  </li>
  <li>
    Click OK.
  </li>
</ol>

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2018/07/Select-a-Template-for-your-Fabric-Service.jpg"><img loading="lazy" src="/assets/img/posts/2018/07/Select-a-Template-for-your-Fabric-Service.jpg" alt="Select a Template for your Fabric Service" /></a>
  
  <p>
    Select a Template for your Fabric Service
  </p>
</div>

<ol start="5">
  <li>
    Expand the PackageRoot folder in the Solution Explorer and you will find the ServiceManifest.xml file there. This file describes the service deployment package and related information. It includes a section that describes the service type that is initialized when the Service Fabric runtime starts the service.
  </li>
</ol>

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2018/07/The-service-type-description.jpg"><img aria-describedby="caption-attachment-1392" loading="lazy" class="size-full wp-image-1392" src="/assets/img/posts/2018/07/The-service-type-description.jpg" alt="The service type description" /></a>
  
  <p>
    The service type description
  </p>
</div>

### Configure your Service Fabric application

  1. A service type is created for the project. In this case, the type is defined in the Simulator.cs file. This service type is registered in Program.cs when the program starts so that the Service Fabric runtime knows which type to initialize when it creates an instance of the service.

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2018/07/Registering-a-service-type-in-the-main-method.jpg"><img loading="lazy" src="/assets/img/posts/2018/07/Registering-a-service-type-in-the-main-method.jpg" alt="Registering a service type in the main method" /></a>
  
  <p>
    Registering a service type in the main method
  </p>
</div>

<ol start="2">
  <li>
    The template produces a default implementation for the service type, with a RunAsync method that increments a counter every second. This counter value is persisted with the service in a dictionary using the StateManager, available through the service base type StatefulService. This counter is used to represent the number of leads generated for the purpose of this example.
  </li>
</ol>

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2018/07/RunAsync-which-increments-a-counter-every-second.jpg"><img loading="lazy" src="/assets/img/posts/2018/07/RunAsync-which-increments-a-counter-every-second.jpg" alt="RunAsync which increments a counter every second" /></a>
  
  <p>
    RunAsync which increments a counter every second
  </p>
</div>

<ol start="3">
  <li>
    The service will run, and increment the counter as it runs persisting the value, but by default, this service does not expose any methods for a client to call. Before you can create an RPC listener you have to add the required NuGet package, Microsoft.ServiceFabric.Services.Remoting.
  </li>
  <li>
    Create a new service interface using the IService marker interface from the previously installed NuGet, that indicates this service can be called remotely.
  </li>
</ol>

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2018/07/Create-the-ISimulatorService-interface.jpg"><img aria-describedby="caption-attachment-1395" loading="lazy" class="size-full wp-image-1395" src="/assets/img/posts/2018/07/Create-the-ISimulatorService-interface.jpg" alt="Create the ISimulatorService interface" /></a>
  
  <p>
    Create the ISimulatorService interface
  </p>
</div>

<ol start="5">
  <li>
    Implement the previously created interface on the Simulator service type, and include an implementation of the GetLeads method to return the value of the counter.
  </li>
</ol>

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2018/07/Implementation-of-the-GetLeads-method.jpg"><img loading="lazy" src="/assets/img/posts/2018/07/Implementation-of-the-GetLeads-method.jpg" alt="Implementation of the GetLeads method" /></a>
  
  <p>
    Implementation of the GetLeads method
  </p>
</div>

<ol start="6">
  <li>
    To expose this method to clients, add an RPC listener to the service. Modify the CreateServiceReplicaListeners method in the Simulator service type implementation, to add a call to the CreateServiceReplicaListeners method.
  </li>
</ol>

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2018/07/Modify-the-CreateServiceReplicaListeners-method.jpg"><img aria-describedby="caption-attachment-1397" loading="lazy" class="size-full wp-image-1397" src="/assets/img/posts/2018/07/Modify-the-CreateServiceReplicaListeners-method.jpg" alt="Modify the CreateServiceReplicaListeners method" /></a>
  
  <p>
    Modify the CreateServiceReplicaListeners method
  </p>
</div>

## Add a web front end to a Service Fabric application

In this section, I will create a web front end to call the stateful service endpoint which I create previously.

To add a web app to your Service Fabric application, follow these steps:

  1. Right-click the Services node of your Service Fabric application and select Add and then New Service Fabric Service&#8230;
  2. In the template dialog, select Stateless ASP.NET Core, provide a name and click OK.

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2018/07/Create-a-web-app-in-your-Service-Fabric-app.jpg"><img loading="lazy" src="/assets/img/posts/2018/07/Create-a-web-app-in-your-Service-Fabric-app.jpg" alt="Create a web app in your Service Fabric app" /></a>
  
  <p>
    Create a web app in your Service Fabric app
  </p>
</div>

<ol start="3">
  <li>
    On the next page select Web Application (Model-View-Controller) and click OK.
  </li>
</ol>

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2018/07/Select-the-mvc-template-for-your-web-app.png"><img loading="lazy" src="/assets/img/posts/2018/07/Select-the-mvc-template-for-your-web-app.png" alt="Select the mvc template for your web app" /></a>
  
  <p>
    Select the MVC template for your web app
  </p>
</div>

<ol start="4">
  <li>
    Expand the PackageRoot folder in the Solution Explorer and you will find the ServiceManifest.xml file there. This file describes the service deployment package and related information. It includes a section that describes the HTTP endpoint where your web app will listen for requests.
  </li>
</ol>

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2018/07/The-HTTP-endpoint-description.jpg"><img aria-describedby="caption-attachment-1400" loading="lazy" class="size-full wp-image-1400" src="/assets/img/posts/2018/07/The-HTTP-endpoint-description.jpg" alt="The HTTP endpoint description" /></a>
  
  <p>
    The HTTP endpoint description
  </p>
</div>

<ol start="5">
  <li>
    The new WebApp type is defined in the WebApp.cs, which inherits from StatelessService. For the service to listen for HTTP requests, the CreateServiceInstanceListeners() method sets up the WebListener.
  </li>
</ol>

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2018/07/The-WebApp-class.jpg"><img aria-describedby="caption-attachment-1401" loading="lazy" class="size-full wp-image-1401" src="/assets/img/posts/2018/07/The-WebApp-class.jpg" alt="The WebApp class" /></a>
  
  <p>
    The WebApp class
  </p>
</div>

<ol start="6">
  <li>
    The next step is to call the stateful service that returns the leads counter value, from the stateless web app.
  </li>
  <li>
    Make a copy of the service interface defined for the service type, ISimulatorService.
  </li>
</ol>

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2018/07/The-copied-ISimulatorService-interface.jpg"><img aria-describedby="caption-attachment-1402" loading="lazy" class="size-full wp-image-1402" src="/assets/img/posts/2018/07/The-copied-ISimulatorService-interface.jpg" alt="The copied ISimulatorService interface" /></a>
  
  <p>
    The copied ISimulatorService interface
  </p>
</div>

<ol start="8">
  <li>
    Modify the ConfigureServices instruction in the WebApp.cs to inject an instance of Fabric client.
  </li>
</ol>

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2018/07/The-modified-CreateServiceInstanceListeners-method.jpg"><img loading="lazy" src="/assets/img/posts/2018/07/The-modified-CreateServiceInstanceListeners-method.jpg" alt="The modified CreateServiceInstanceListeners method" /></a>
  
  <p>
    The modified CreateServiceInstanceListeners method
  </p>
</div>

<ol start="9">
  <li>
    Modify the HomeController to use the FabricClient via dependency injection.
  </li>
</ol>

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2018/07/Inject-FabricClient-into-the-HomeController.jpg"><img aria-describedby="caption-attachment-1404" loading="lazy" class="size-full wp-image-1404" src="/assets/img/posts/2018/07/Inject-FabricClient-into-the-HomeController.jpg" alt="Inject FabricClient into the HomeController" /></a>
  
  <p>
    Inject FabricClient into the HomeController
  </p>
</div>

<ol start="10">
  <li>
    Modify the Index method in the HomeController to use the FabricClient instance to call the Simulator service
  </li>
</ol>

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2018/07/Modify-the-Index-method-to-call-the-Simulator-service.jpg"><img loading="lazy" src="/assets/img/posts/2018/07/Modify-the-Index-method-to-call-the-Simulator-service.jpg" alt="Modify the Index method to call the Simulator service" /></a>
  
  <p>
    Modify the Index method to call the Simulator service
  </p>
</div>

<ol start="11">
  <li>
    Update the Index.cshtml view to display the counter for each partition.
  </li>
</ol>

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2018/07/Modify-the-Index.cshtml-view.jpg"><img aria-describedby="caption-attachment-1406" loading="lazy" class="size-full wp-image-1406" src="/assets/img/posts/2018/07/Modify-the-Index.cshtml-view.jpg" alt="Modify the Index.cshtml view" /></a>
  
  <p>
    Modify the Index.cshtml view
  </p>
</div>

### Deploy and run your Web App

  1. To run the web app and stateful service, you can publish it to the local Service Fabric cluster. Right-click the Service Fabric application node in the Solution Explorer and select Publish. From the Publish Service Fabric Application dialog, select a target profile matching one of the local cluster options, and click Publish.

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2018/07/Deploy-to-a-local-cluster-with-error-message.png"><img aria-describedby="caption-attachment-1407" loading="lazy" class="size-full wp-image-1407" src="/assets/img/posts/2018/07/Deploy-to-a-local-cluster-with-error-message.png" alt="Deploy to a local cluster with error message" /></a>
  
  <p>
    Deploy to a local cluster with an error message
  </p>
</div>

<ol start="2">
  <li>
    If you get an error message as on the screenshot shown above, start PowerShell as administrator and run the following code: & <span class="hljs-string">&#8220;<span class="hljs-variable">$ENV:ProgramFiles</span>\Microsoft SDKs\Service Fabric\ClusterSetup\DevClusterSetup.ps1&#8243;</span>. This creates a local cluster.
  </li>
  <li>
    The installation takes a couple of minutes.
  </li>
  <li>
    Once the installation is done, close and re-open the Publish window and the error should be gone.
  </li>
  <li>
    Deploy your application and then access your web app at http://localhost:8527 (or whatever port you configured in the ServiceManifest.xml in your web app).
  </li>
</ol>

You can find the code of the demo on <a href="https://github.com/WolfgangOfner/Azure-ServiceFabricDemo" target="_blank" rel="noopener">GitHub</a>.

## Build an Actor-based service

The actor model is a superset of the Service Fabric stateful model. Actors are simple POCO objects that have many features that make them an isolated, independent unit of compute and state with single-threaded execution.

To create a new Service Fabric application based on the Actor service template, follow these steps:

  1. Open Visual Studio and select File -> New -> Project.
  2. In the Cloud category select Service Fabric Application, provide a name and click OK.
  3. Select Actor Service from the templates list, provide a name and Click OK.

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2018/07/Create-an-Actor-service.jpg"><img loading="lazy" src="/assets/img/posts/2018/07/Create-an-Actor-service.jpg" alt="Create an Actor service" /></a>
  
  <p>
    Create an Actor service
  </p>
</div>

## Monitor and diagnose services

The Azure portal offers several features to monitor and evaluate the performance or resource consumption of your application at runtime.

## Deploy an application to a container

Service Fabric can run processes and containers side by side, and containers can be Linux or Windows based containers. If you have an existing container image and wish to deploy this to an existing Service Fabric cluster, you can follow these steps to create a new Service Fabric application and set it up to deploy and run the container in your cluster.

  1. Open Visual Studio and select File -> New -> Project.
  2. In the Cloud category select Service Fabric Application, provide a name and click OK.
  3. Select Container from the templates list, provide a name and container image and Click OK.
  4. Expand the PackageRoot folder in the Solution Explorer and you will find the ServiceManifest.xml file there. Modify the Resources section to add a UriScheme, Port and Protocol setting for the service point.

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2018/07/Add-a-UriScheme-Port-and-Protocol-to-the-ServiceManifest.xml-file.jpg"><img aria-describedby="caption-attachment-1411" loading="lazy" class="size-full wp-image-1411" src="/assets/img/posts/2018/07/Add-a-UriScheme-Port-and-Protocol-to-the-ServiceManifest.xml-file.jpg" alt="Add a UriScheme, Port and Protocol to the ServiceManifest.xml file" /></a>
  
  <p>
    Add a UriScheme, Port and Protocol to the ServiceManifest.xml file
  </p>
</div>

<ol start="5">
  <li>
    Open the ApplicationManifest.xml file. Create a policy for the container to host a PortBinding by adding the Policies section to the ServiceManifestImport section. Additionally, indicate the container port for your container.
  </li>
</ol>

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2018/07/Create-a-PortBinding-policy-for-the-container.jpg"><img aria-describedby="caption-attachment-1412" loading="lazy" class="size-full wp-image-1412" src="/assets/img/posts/2018/07/Create-a-PortBinding-policy-for-the-container.jpg" alt="Create a PortBinding policy for the container" /></a>
  
  <p>
    Create a PortBinding policy for the container
  </p>
</div>

<ol start="6">
  <li>
    The application is configured and ready to be published.
  </li>
</ol>

## Migrate apps from cloud services

You can migrate your existing cloud service, both web and worker roles to Service Fabric applications.

## Scale a Service Fabric app

To scale a Service Fabric app, you have to understand Instances, Partitions, and Replicas.

By default, the Service Fabric tooling produces three publish profiles that you can use to deploy your application.

<div class="table-responsive">
  <table class="table table-striped table-bordered table-hover">
    <tr>
      <td>
        Publish profile
      </td>
      
      <td>
        Description
      </td>
    </tr>
    
    <tr>
      <td>
        Local.1Node.xml
      </td>
      
      <td>
        To deploy against the local 1-node cluster.
      </td>
    </tr>
    
    <tr>
      <td>
        Local.5Node.xml
      </td>
      
      <td>
        To deploy against the local 5-node cluster.
      </td>
    </tr>
    
    <tr>
      <td>
        Cloud.xml
      </td>
      
      <td>
        To deploy against a Cloud cluster.
      </td>
    </tr>
  </table>
</div>

The publish profiles indicate the settings for the number of instances and partitions for each service.

<div class="table-responsive">
  <table class="table table-striped table-bordered table-hover">
    <tr>
      <td>
        Publish profile parameter
      </td>
      
      <td>
        Description
      </td>
    </tr>
    
    <tr>
      <td>
        WebApp_InstanceCount
      </td>
      
      <td>
        Specifies the number of instances the WebApp service must have within the cluster.
      </td>
    </tr>
    
    <tr>
      <td>
        Simulator_PartitionCount
      </td>
      
      <td>
        Specifies the number of partitions (for the stateful service) the Simulator service must have within the cluster.
      </td>
    </tr>
    
    <tr>
      <td>
        Simulator_MinReplicaSetSize
      </td>
      
      <td>
        Specifies the minimum number of replicas required for each partition that the WebApp service should have within the cluster.
      </td>
    </tr>
    
    <tr>
      <td>
        Simulator_TargetReplicaSetSize
      </td>
      
      <td>
        Specifies the number of target replicas required for each partition that the WebApp service should have within the cluster.
      </td>
    </tr>
  </table>
</div>

## Create, secure, upgrade, and save Service Fabric Cluster in Azure

To publish your Service Fabric application to Azure in production, you will create a cluster and have to learn how to secure it. Also, you should know how to upgrade applications with zero downtime, and configure the application to scale following Azure&#8217;s best practices

## Conclusion

This post gave an overview of Azure Service Fabric and its features and how to deploy it.

For more information about the 70-532 exam get the <a href="http://amzn.to/2EWNWMF" target="_blank" rel="noopener">Exam Ref book from Microsoft</a> and continue reading my blog posts. I am covering all topics needed to pass the exam. You can find an overview of all posts related to the 70-532 exam <a href="/prepared-for-the-70-532-exam/" target="_blank" rel="noopener">here</a>.