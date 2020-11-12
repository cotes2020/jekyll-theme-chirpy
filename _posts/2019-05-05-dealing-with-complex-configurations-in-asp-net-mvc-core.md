---
title: Dealing with Complex Configurations in ASP.NET Core MVC
date: 2019-05-05T11:14:45+02:00
author: Wolfgang Ofner
categories: [ASP.NET]
tags: [NET Core, ASP.NET Core MVC, 'C#']
---
If you have to deal with a large number of hosting environments, configuring all of them in the Startup class can become messy. In the following sections, I will describe different ways that the Startup class can be used for complex configurations. You can find the source code of the following demo on <a href="https://github.com/WolfgangOfner/MVC-Core-Complex-Configurations" target="_blank" rel="noopener noreferrer">GitHub</a>.

## Creating different External Configuration Files

The default configuration for the application performed by the Program class looks for JSON configuration files that are specific to the hosting environment being used to run the application. A file called appsettings.production.json can be used to store settings that are specific to the production platform.

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2019/04/Loading-the-appsettings.json-file-according-to-the-environment.jpg"><img loading="lazy" src="/assets/img/posts/2019/04/Loading-the-appsettings.json-file-according-to-the-environment.jpg" alt="Loading the appsettings.json file according to the environment" /></a>
  
  <p>
    Loading the appsettings.json file according to the environment
  </p>
</div>

When you load configuration data from a platform-specific file, the configuration settings it contains override any existing data with the same names. The appsettings.json file will be loaded when the application starts, followed by the appsettings.development.json file if the application is running in a development environment.

## Creating different Configuration Methods

Selecting different configuration data files can be useful but doesn’t provide a complete solution for complex configurations because data files don’t contain C# statements. If you want to vary the configuration statements used to create services or register middleware components, then you can use different methods, where the name of the method includes the hosting environment:

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2019/04/Using-different-method-names-in-the-Startup-class.jpg"><img loading="lazy"  title="Using different method names in the Startup class to handle complex configurations" src="/assets/img/posts/2019/04/Using-different-method-names-in-the-Startup-class.jpg" alt="Using different method names in the Startup class" /></a>
  
  <p>
    Using different method names in the Startup class
  </p>
</div>

When ASP.NET Core looks for the ConfigureServices and Configure methods in the Startup class, it first checks to see whether there are methods that include the name of the hosting environment. You can define separate methods for each of the environments that you need to support and rely on the default methods being called if there are no environment-specific methods available. Note that <span class="fontstyle0">the default methods are not called if there are environment-specific methods defined.</span>

## Creating different Configuration Classes to handle Complex Configurations

Using different methods means you don’t have to use if statements to check the hosting environment name, but it can result in large classes, which is a problem in itself. For especially complex configurations, the final progression is to create a different configuration class for each hosting environment. When ASP.NET looks for the Startup class, it first checks to see whether there is a class whose name includes the current hosting environment.

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2019/04/Choose-the-Startup-class-at-runtime.jpg"><img loading="lazy"  title="Choose the Startup class at runtime to deal with your complex configurations" src="/assets/img/posts/2019/04/Choose-the-Startup-class-at-runtime.jpg" alt="Choose the Startup class at runtime" /></a>
  
  <p>
    Choose the Startup class at runtime
  </p>
</div>

Rather than specifying a class, the UseStartup method is given the name of the assembly that it should use. When the application starts, ASP.NET will look for a class whose name includes the hosting environment, such as StartupDevelopment or StartupProduction, and fall back to using the regular Startup class if one does not exist.

## Conclusion

In this post, I showed different approaches on how to handle the configuration of your application. You can use different files, different methods or even different classes, depending on how complex your configuration will be.

For more details about complex configurations, I highly recommend the book &#8220;<a href="https://www.amazon.com/Pro-ASP-NET-Core-MVC-2/dp/148423149X" target="_blank" rel="noopener noreferrer">Pro ASP.NET Core MVC 2</a>&#8220;. You can find the source code of this demo on <a href="https://github.com/WolfgangOfner/MVC-Core-Complex-Configurations" target="_blank" rel="noopener noreferrer">GitHub</a>.