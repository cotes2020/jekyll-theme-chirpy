---
title: Develop Azure App Service Mobile App
date: 2018-06-29T22:21:13+02:00
author: Wolfgang Ofner
categories: [Cloud]
tags: [70-532, Azure, Certification, Exam, learning]
---
A Mobile App in Azure App Service provides a platform for the development of mobile applications, providing a combination of backend Azure hosted services with device side development frameworks that streamline the integration of the backend services.

Mobile Apps enables the development of applications across a variety of platforms, targeting native iOS, Android, and Windows apps, cross-platform Xamarin and Cordova.

Azure Mobile Apps provide functionality for:

<div class="table-responsive">
  <table class="table table-striped table-bordered table-hover">
    <tr>
      <td>
        Functionality
      </td>
      
      <td>
        Description
      </td>
    </tr>
    
    <tr>
      <td>
        Authentication and authorization
      </td>
      
      <td>
        Enables integration with identity providers including Azure Active Directory, Google, Twitter, Facebook, and Microsoft.
      </td>
    </tr>
    
    <tr>
      <td>
        Data Access
      </td>
      
      <td>
        Enables access to tabular data stored in an Azure SQL Database or an on-premise SQL Server via an automatically provisioned and mobile-friendly OData v3 data source
      </td>
    </tr>
    
    <tr>
      <td>
        Offline sync
      </td>
      
      <td>
        Enables reads as well as create, update, and delete activity to happen against the supporting tables even when the device is not connected to the network, and coordinates the synchronization of data between local and cloud stores as dictated by the application logic.
      </td>
    </tr>
    
    <tr>
      <td>
        Push notifications
      </td>
      
      <td>
        Enables the sending of push notifications to app users via Azure Notifications Hubs, which in turn supports the sending of notifications across the most popular push notifications services for Apple, Google, Windows and Amazon devices.
      </td>
    </tr>
  </table>
</div>

##  Create a mobile app

The following steps are necessary to create a mobile app:

  1. Identify the target device platforms you want your app to target.
  2. Prepare your development environment.
  3. Deploy an Azure Mobile App Service instance.
  4. Configure the Azure Mobile App Service.
  5. Configure your client application.
  6. Augment your project with authentication/authorization, offline data sync, or push notification capabilities.

### Identify the target device platform

Decide if you want to develop your Mobile App as native Android, Cordova, native iOS, Windows, Xamarin Android, Xamarin Forms or Xamarion iOS application.

### Prepare your development environment

The requirements for your development environment vary depending on the device platforms you wish to target. For example, iOS developers use macOS and Xcode whereas Android developers macOS or Windows and Android Studio use.

### Deploy an Azure Mobile App Service

To create an Azure Mobile App Service instance, follow these steps:

  1. In the Azure portal click on +Create a resource, search for Mobile App and click Create.
  2. Provide a name, subscription, resource group and, service plan.
  3. Click Create

<div id="attachment_1347" style="width: 326px" class="wp-caption aligncenter">
  <a href="/assets/img/posts/2018/06/Create-a-Mobile-App.jpg"><img aria-describedby="caption-attachment-1347" loading="lazy" class="size-full wp-image-1347" src="/assets/img/posts/2018/06/Create-a-Mobile-App.jpg" alt="Create a Mobile App" width="316" height="416" /></a>
  
  <p id="caption-attachment-1347" class="wp-caption-text">
    Create a Mobile App
  </p>
</div>

### Configure the Mobile App

  1. After the Mobile App is deployed, follow these steps to configure it:  
    Open your Mobile App in the Azure portal and click on Quick Start under the Deployment menu.
  2. On the Quickstart blade, select the platform you wish to target, for example, Windows (C#).
  3. Click on &#8220;You will need a database in order to complete this quick start. Click here to create one.
  4. On the data Connections blade, click +Add to create a new SQL database or select one from the list if you already have one.
  5. Select a backend language, for example, C#.
  6. Download the zip file and unpack it.
  7. Compile the application and then deploy it to your Mobile App to Azure.
  8. To deploy your app right click on the project in Visual Studio, select Publish and then select your previously created Mobile App by clicking on Select Existing in the Azure App Service tab.

<div id="attachment_1349" style="width: 710px" class="wp-caption aligncenter">
  <a href="/assets/img/posts/2018/06/Deploy-your-Mobile-App-to-Azure.jpg"><img aria-describedby="caption-attachment-1349" loading="lazy" class="wp-image-1349" src="/assets/img/posts/2018/06/Deploy-your-Mobile-App-to-Azure.jpg" alt="Deploy your Mobile App to Azure" width="700" height="518" /></a>
  
  <p id="caption-attachment-1349" class="wp-caption-text">
    Deploy your Mobile App to Azure
  </p>
</div>

<ol start="9">
  <li>
    After the deployment is finished, the browser is opened and displays that your Mobile App is running.
  </li>
</ol>

<div id="attachment_1351" style="width: 537px" class="wp-caption aligncenter">
  <a href="/assets/img/posts/2018/06/The-Mobile-App-is-deployed-and-running.jpg"><img aria-describedby="caption-attachment-1351" loading="lazy" class="size-full wp-image-1351" src="/assets/img/posts/2018/06/The-Mobile-App-is-deployed-and-running.jpg" alt="The Mobile App is deployed and running" width="527" height="443" /></a>
  
  <p id="caption-attachment-1351" class="wp-caption-text">
    The Mobile App is deployed and running
  </p>
</div>

### Configure your client application

You can create a new application from the Quickstart blade, or by connecting an existing application. To download the template application, click on Create a new app and then download under section 3 Configure your client application on the Quickstart blade.

<div id="attachment_1348" style="width: 491px" class="wp-caption aligncenter">
  <a href="/assets/img/posts/2018/06/Download-templates-for-your-Mobile-App.jpg"><img aria-describedby="caption-attachment-1348" loading="lazy" class="wp-image-1348" src="/assets/img/posts/2018/06/Download-templates-for-your-Mobile-App.jpg" alt="Download templates for your Mobile App" width="481" height="700" /></a>
  
  <p id="caption-attachment-1348" class="wp-caption-text">
    Download templates for your Mobile App
  </p>
</div>

### Add authentication to a mobile app

To enable the integration of identity providers like Facebook or Twitter, follow these steps:

  1. For each identity provider, you want to support, you need to follow the provider&#8217;s specific instructions to register your app and retrieve the credentials needed to authenticate using that provided.
  2. Open your Mobile App and select Authentication / Authorization under the Settings blade.
  3. On the Authentication / Authorization configure the provider you want to use. For example, for Twitter, you have to enter your API Key and API Secret which you get on Twitter&#8217;s developer page.
  4. In the Allowed external redirect URLs textbox, enter your callback URL, for example, myapp://myauth.callback.

<div id="attachment_1354" style="width: 497px" class="wp-caption aligncenter">
  <a href="/assets/img/posts/2018/06/Enable-Twitter-authentication-and-authorization.jpg"><img aria-describedby="caption-attachment-1354" loading="lazy" class="wp-image-1354" src="/assets/img/posts/2018/06/Enable-Twitter-authentication-and-authorization.jpg" alt="Enable Twitter authentication and authorization" width="487" height="700" /></a>
  
  <p id="caption-attachment-1354" class="wp-caption-text">
    Enable Twitter authentication and authorization
  </p>
</div>

<ol start="5">
  <li>
    Click Save.
  </li>
  <li>
    In your C# application decorate all controller and/or action which should be only accessed be logged in users with the [Authorize] attribute.
  </li>
  <li>
    Add your authentication logic to the project.
  </li>
  <li>
    Run your application in your local simulator or device to verify the authentication flow.
  </li>
</ol>

The steps 6-8 are only an overview because an exact explanation would be too much for here. If you don&#8217;t know how the Authorize attribute works, create a new ASP.NET MVC project form the template in Visual Studio. There it is implemented and you can see that an action or controller with this attribute redirects the user to the login page.

### Add offline sync to a Mobile App

The offline data sync capability comes from a mix of client-side SDK and service-side features. These features include, for example, support for conflict detection when the same record is changed on both the client and backend, and it allows for the conflicts to be resolved.

For the conflicts to be resolved you need a table that leverages Mobile App easy tables on the service side. This is usually a SQL Database exposed by Mobile Apps using the OData endpoint.

On the client side, the Azure Mobile App  SDKs provide an interface referred to as a SyncTable that wraps access to the remote easy table. When using a Synctable all the CRUD operations work from a local store, whose implementation is device platform specific. For example, iOS uses the local store based on Core Data whereas Xamarin and Androids local store is based on SQL lite. Changes to the data are made through a sync context object that tracks the changes that are made across all tables. This sync context maintains an operation queue that is an ordered list of create, update and delete operations that have been performed against the data locally.

### Add push notifications to a Mobile App

Push notifications enable you to send app-specific messages to your app running across a variety of platforms. In Azure Mobile Apps, push notification capabilities are provided by Azure Notification Hubs. Notification Hubs abstract your application from the complexities of dealing with the various push notification systems (PNS) that are specific to each platform. Notification Hubs support the sending of notifications across the most popular push notification services for Apple, Google, Windows, and Amazon.

To add push notifications, follow these steps:

  1. Open your Mobile App and select Push under the Settings menu.
  2. On the Push blade, connect to a Notification Hub. If you don&#8217;t have one yet, you can create one on the blade.
  3. After your Notification Hub is connected, click on Configure push notification services.
  4. On the Push notification services blade, enter the configuration of the notification service you want to use.
  5. Click Save.
  6. Configure your app project to respond to push notifications.

## Conclusion

In this post, I showed how to leverage Azure Mobile App services to easily create a mobile app using modern features like push notifications and offline sync.

For more information about the 70-532 exam get the <a href="http://amzn.to/2EWNWMF" target="_blank" rel="noopener">Exam Ref book from Microsoft</a> and continue reading my blog posts. I am covering all topics needed to pass the exam. You can find an overview of all posts related to the 70-532 exam <a href="/prepared-for-the-70-532-exam/" target="_blank" rel="noopener">here</a>.