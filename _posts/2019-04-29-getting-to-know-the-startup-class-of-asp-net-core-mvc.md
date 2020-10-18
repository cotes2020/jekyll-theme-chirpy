---
title: Getting to know the Startup Class of ASP.Net Core MVC
date: 2019-04-29T11:15:53+02:00
author: Wolfgang Ofner
categories: [ASP.NET]
tags: [.net core, ASP.NET Core MVC, 'C#']
---
Every .NET Core web application has a Program class with a static Main method.

The Startup class of .Net core is the new version of the Global.asax file. This class is responsible for starting the application. The most important part of it is .UseStartup<Startup>(). This delegate calls the Startup class in which all the configuration for the web application is handled. The UseStartup method relies on a type parameter to identify the class that will configure the .NET Core application. This means if you don&#8217;t want to use the default Startup class, you can edit the call, for example with .UseStartup<MyStartup>().

The Startup class has two methods, ConfigureServices and Configure, that tell ASP.NET Core which features are available and how they should be used.

You can find the source code for this demo on <a href="https://github.com/WolfgangOfner/MVC-Core-Startup" target="_blank" rel="noopener noreferrer">GitHub</a>. On the following screenshot, you can see the default Startup class of a .NET Core:

<div id="attachment_1621" style="width: 710px" class="wp-caption aligncenter">
  <a href="https://www.programmingwithwolfgang.com/wp-content/uploads/2019/04/The-default-Startup-class-of-ASP.NET-Core-MVC-2.jpg"><img aria-describedby="caption-attachment-1621" loading="lazy" class="wp-image-1621" src="https://www.programmingwithwolfgang.com/wp-content/uploads/2019/04/The-default-Startup-class-of-ASP.NET-Core-MVC-2.jpg" alt="The default Startup class of ASP.NET Core MVC 2" width="700" height="599" srcset="https://www.programmingwithwolfgang.com/wp-content/uploads/2019/04/The-default-Startup-class-of-ASP.NET-Core-MVC-2.jpg 924w, https://www.programmingwithwolfgang.com/wp-content/uploads/2019/04/The-default-Startup-class-of-ASP.NET-Core-MVC-2-300x257.jpg 300w, https://www.programmingwithwolfgang.com/wp-content/uploads/2019/04/The-default-Startup-class-of-ASP.NET-Core-MVC-2-768x657.jpg 768w" sizes="(max-width: 700px) 100vw, 700px" /></a>
  
  <p id="caption-attachment-1621" class="wp-caption-text">
    The default Startup class of ASP.NET Core MVC 2
  </p>
</div>

When the ASP.NET Core starts, the application creates a new instance of the Startup class and calls the ConfigureServices method to create its services. After the services are created, the application calls the Configure method. This method sets up middlewares (the request pipeline) which are used to handle incoming HTTP requests. Examples for middlewares are logging and authentication.

## Adding MVC functionality in the Startup class

To enable the MVC functionality, you have to add MVC service to your service collection and a default route to your application. This can be done with services.AddMvc() (optional with a specific version) and app.UseMvcWithDefaultRoute(). That&#8217;s already enough to start your web application.

<div id="attachment_1622" style="width: 546px" class="wp-caption aligncenter">
  <a href="https://www.programmingwithwolfgang.com/wp-content/uploads/2019/04/Add-MVC-functionality-to-the-web-application.jpg"><img aria-describedby="caption-attachment-1622" loading="lazy" class="wp-image-1622 size-full" src="https://www.programmingwithwolfgang.com/wp-content/uploads/2019/04/Add-MVC-functionality-to-the-web-application.jpg" alt="Add MVC functionality to the Startup class of your web application" width="536" height="169" srcset="https://www.programmingwithwolfgang.com/wp-content/uploads/2019/04/Add-MVC-functionality-to-the-web-application.jpg 536w, https://www.programmingwithwolfgang.com/wp-content/uploads/2019/04/Add-MVC-functionality-to-the-web-application-300x95.jpg 300w" sizes="(max-width: 536px) 100vw, 536px" /></a>
  
  <p id="caption-attachment-1622" class="wp-caption-text">
    Add MVC functionality to the web application
  </p>
</div>

The add.MVC() method sets up every service that MVC needs. There is no need to add every needed service and therefore the configuration of your application stays small and simple.

## Taking a closer look at the Configure Method of the Startup class

The Configure method has two parameters of type IApplicationBuilder and IHostingEnvironment. The IApplicationBuilder is used to set up the functionality of the middleware pipeline whereas the IHostingEnvironment enables the application to differentiate between different environment types, for example, testing and production.

### Using the Application Builder

Almost every application will use the IApplicationBuilder because it is used to set up the MVC or your custom middlewares. Setting up the MVC pipeline can be done by using UseMvcWithDefaultRoute or with UseMvc. The first method will set up with a default route, containing {controller}/{action}/{id?}. UseMvc can be used if you want to set up your own routes using lambda expressions.

Usually, applications use the UseMvc method, even if the routes look the same as the default route. This approach makes the used routing logic more obvious to other developers and easier to add new routes later.

<div id="attachment_1637" style="width: 408px" class="wp-caption aligncenter">
  <a href="https://www.programmingwithwolfgang.com/wp-content/uploads/2019/04/Adding-a-the-default-route-with-UseMvc.jpg"><img aria-describedby="caption-attachment-1637" loading="lazy" class="wp-image-1637 size-full" src="https://www.programmingwithwolfgang.com/wp-content/uploads/2019/04/Adding-a-the-default-route-with-UseMvc.jpg" alt="Adding a the default route with UseMvc in the Startup class" width="398" height="95" srcset="https://www.programmingwithwolfgang.com/wp-content/uploads/2019/04/Adding-a-the-default-route-with-UseMvc.jpg 398w, https://www.programmingwithwolfgang.com/wp-content/uploads/2019/04/Adding-a-the-default-route-with-UseMvc-300x72.jpg 300w" sizes="(max-width: 398px) 100vw, 398px" /></a>
  
  <p id="caption-attachment-1637" class="wp-caption-text">
    Adding a the default route with UseMvc
  </p>
</div>

MVC also sets up content-generating middleware, therefore your custom middlewares should be registered first and the MVC one last.

### Using the Hosting Environment

The IHostingEnvironment interface provides the following properties and information about the hosting environment:

<div class="table-responsive">
  <table class="table table-striped table-bordered table-hover">
    <tr>
      <td>
        Name
      </td>
      
      <td>
        Description
      </td>
    </tr>
    
    <tr>
      <td>
        ApplicationName
      </td>
      
      <td>
        This returns the name of the application which is set by the hosting platform.
      </td>
    </tr>
    
    <tr>
      <td>
        EnvironmentName
      </td>
      
      <td>
        This string describes the current environment, for example, test or production.
      </td>
    </tr>
    
    <tr>
      <td>
        ContentRootPath
      </td>
      
      <td>
        This property returns the path that contains the application&#8217;s content and configuration files.
      </td>
    </tr>
    
    <tr>
      <td>
        WebRootPath
      </td>
      
      <td>
        The string which specifies the location of the container for static content. Usually, this is the wwwroot folder.
      </td>
    </tr>
    
    <tr>
      <td>
        ContentRootFileProvider
      </td>
      
      <td>
        This property returns an object that implements the IFileProvider interface and that can be used to read files from the folder specified by the<br /> ContentRootPath property.
      </td>
    </tr>
    
    <tr>
      <td>
        WebRootFileProvider
      </td>
      
      <td>
        This property returns an object that implements the IFileProvider interface and that can be used to read files from the folder specified by the<br /> WebRootPath property.
      </td>
    </tr>
  </table>
</div>

The ContentRootPath and WebRootPath might be interesting properties but are not needed in most real-world applications because MVC provides a built-in middleware which can be used to deliver static content.  
Probably the most important property is EnvironmentName because it allows you to modify the configuration of the application based on the environment in which it is running. There are three conventional environments: development, staging, and production. You can set the current hosting environment by setting an environment variable called ASPNETCORE_ENVIRONMENT.

You can set this variable by selecting ConfiguringAppsProperties from the Visual Studio Project menu and switching to the Debug tag. There double-click the Value field of the environment variable and change it, for example, to Staging. It is common practice to use Staging but the value is not case-sensitive, so you could also use staging. Additionally, you can use every name you want, these previously mentioned ones are only conventional ones.

<div id="attachment_1638" style="width: 610px" class="wp-caption aligncenter">
  <a href="https://www.programmingwithwolfgang.com/wp-content/uploads/2019/04/Setting-the-environment-variable-to-Development.jpg"><img aria-describedby="caption-attachment-1638" loading="lazy" class="size-full wp-image-1638" src="https://www.programmingwithwolfgang.com/wp-content/uploads/2019/04/Setting-the-environment-variable-to-Development.jpg" alt="Setting the environment variable to Development" width="600" height="164" srcset="https://www.programmingwithwolfgang.com/wp-content/uploads/2019/04/Setting-the-environment-variable-to-Development.jpg 600w, https://www.programmingwithwolfgang.com/wp-content/uploads/2019/04/Setting-the-environment-variable-to-Development-300x82.jpg 300w" sizes="(max-width: 600px) 100vw, 600px" /></a>
  
  <p id="caption-attachment-1638" class="wp-caption-text">
    Setting the environment variable to Development
  </p>
</div>

You can use IHostingEnvironment.EnvironmentName in the Configure method to determine which hosting environment is being used. Also, you could use one of the extension methods of IHostingEnvironment:

<div class="table-responsive">
  <table class="table table-striped table-bordered table-hover">
    <tr>
      <td>
        Name
      </td>
      
      <td>
        Description
      </td>
    </tr>
    
    <tr>
      <td>
        IsDevelopment()
      </td>
      
      <td>
        Returns true if the hosting environment name is Development.
      </td>
    </tr>
    
    <tr>
      <td>
        IsStaging()
      </td>
      
      <td>
        Returns true if the hosting environment name is Staging.
      </td>
    </tr>
    
    <tr>
      <td>
        IsProduction()
      </td>
      
      <td>
        Returns true if the hosting environment name is Production.
      </td>
    </tr>
    
    <tr>
      <td>
        IsEnvironment(env)
      </td>
      
      <td>
        Returns true if the hosting environment name matches the variable.
      </td>
    </tr>
  </table>
  
  <p>
    On the following screenshot, I set up all my custom middlewares only if the application is running on the Development environment. This is useful for gathering debugging or diagnostic information.
  </p>
</div>

<div id="attachment_1640" style="width: 523px" class="wp-caption aligncenter">
  <a href="https://www.programmingwithwolfgang.com/wp-content/uploads/2019/04/Register-the-middleware-only-if-the-hosting-environment-is-Development.jpg"><img aria-describedby="caption-attachment-1640" loading="lazy" class="size-full wp-image-1640" src="https://www.programmingwithwolfgang.com/wp-content/uploads/2019/04/Register-the-middleware-only-if-the-hosting-environment-is-Development.jpg" alt="Register the middleware only if the hosting environment is Development" width="513" height="173" srcset="https://www.programmingwithwolfgang.com/wp-content/uploads/2019/04/Register-the-middleware-only-if-the-hosting-environment-is-Development.jpg 513w, https://www.programmingwithwolfgang.com/wp-content/uploads/2019/04/Register-the-middleware-only-if-the-hosting-environment-is-Development-300x101.jpg 300w" sizes="(max-width: 513px) 100vw, 513px" /></a>
  
  <p id="caption-attachment-1640" class="wp-caption-text">
    Register the middleware only if the hosting environment is Development
  </p>
</div>

In production, none of these middlewares would be added.

## Configuring Exception Handling in the Startup class

In a classic ASP.Net MVC application, you had to configure the exception handling in the web.config. The custom error section there often was configured to show detailed error pages and was changed to hide an exception and redirect on an error page on the production environment.

With ASP.NET Core, you can easily configure this using the IHostingEnvironment and a built-in exception handler middleware.

<div id="attachment_1641" style="width: 326px" class="wp-caption aligncenter">
  <a href="https://www.programmingwithwolfgang.com/wp-content/uploads/2019/04/Configure-exception-handling-depending-on-the-hosting-environment.jpg"><img aria-describedby="caption-attachment-1641" loading="lazy" class="size-full wp-image-1641" src="https://www.programmingwithwolfgang.com/wp-content/uploads/2019/04/Configure-exception-handling-depending-on-the-hosting-environment.jpg" alt="Configure exception handling depending on the hosting environment" width="316" height="156" srcset="https://www.programmingwithwolfgang.com/wp-content/uploads/2019/04/Configure-exception-handling-depending-on-the-hosting-environment.jpg 316w, https://www.programmingwithwolfgang.com/wp-content/uploads/2019/04/Configure-exception-handling-depending-on-the-hosting-environment-300x148.jpg 300w" sizes="(max-width: 316px) 100vw, 316px" /></a>
  
  <p id="caption-attachment-1641" class="wp-caption-text">
    Configure exception handling depending on the hosting environment
  </p>
</div>

The UseStatusCodePages method adds descriptive messages to responses that contain no content, such as 404 &#8211; Not Found responses, which can be useful since not all browsers show their own messages to the user. The UseDeveloperExceptionPage method sets up an error-handling middleware component that displays details of the exception in the response, including the exception trace. This isn’t information that should be displayed to users, so the call to UseDeveloperExceptionPage is made only in the development hosting environment, which is detected using the IHostingEnvironmment object.

On any other environment, the user gets redirected to /Home/Error. You can create the Error action in the Home controller and display a nice error page to the user, providing some helpful information.

## Enabling Browser Link

Browser Link is a feature in Visual Studio that creates a communication channel between the development environment and one or more web browsers. You can use Browser Link to refresh your web application in several browsers at once, which is useful for cross-browser testing.

The server-side part of Browser Link is implemented as a middleware component that must be added to the Startup class as part of the application configuration, without which the Visual Studio  
integration won’t work. Browser Link is useful only during development and should not be used in staging or production because it edits the responses generated by other middleware components to insert JavaScript code that opens HTTP connections back to the server side so that it can receive reload notifications.

<div id="attachment_1642" style="width: 527px" class="wp-caption aligncenter">
  <a href="https://www.programmingwithwolfgang.com/wp-content/uploads/2019/04/Enable-Browser-Link.jpg"><img aria-describedby="caption-attachment-1642" loading="lazy" class="wp-image-1642 size-full" src="https://www.programmingwithwolfgang.com/wp-content/uploads/2019/04/Enable-Browser-Link.jpg" alt="Enable Browser Link in your Startup class" width="517" height="110" srcset="https://www.programmingwithwolfgang.com/wp-content/uploads/2019/04/Enable-Browser-Link.jpg 517w, https://www.programmingwithwolfgang.com/wp-content/uploads/2019/04/Enable-Browser-Link-300x64.jpg 300w" sizes="(max-width: 517px) 100vw, 517px" /></a>
  
  <p id="caption-attachment-1642" class="wp-caption-text">
    Enable Browser Link
  </p>
</div>

You also have to install the Microsoft.VisualStudio.Web.BrowserLink Nuget package.

## Enabling Static Content

The UseStaticFiles method adds a short-circuiting middleware which provides access to the files in the wwwroot folder so that the application can load images, JavaScript and CSS files.

<div id="attachment_1643" style="width: 526px" class="wp-caption aligncenter">
  <a href="https://www.programmingwithwolfgang.com/wp-content/uploads/2019/04/Enable-static-content.jpg"><img aria-describedby="caption-attachment-1643" loading="lazy" class="size-full wp-image-1643" src="https://www.programmingwithwolfgang.com/wp-content/uploads/2019/04/Enable-static-content.jpg" alt="Enable static content" width="516" height="62" srcset="https://www.programmingwithwolfgang.com/wp-content/uploads/2019/04/Enable-static-content.jpg 516w, https://www.programmingwithwolfgang.com/wp-content/uploads/2019/04/Enable-static-content-300x36.jpg 300w" sizes="(max-width: 516px) 100vw, 516px" /></a>
  
  <p id="caption-attachment-1643" class="wp-caption-text">
    Enable static content
  </p>
</div>

I can&#8217;t think of an application which doesn&#8217;t need static content, no matter of the environment it is running in. Therefore, I add it for all environments. This change will also make the default page work since it is loading the necessary CSS files now.

## Conclusion

In this post, I showed different ways to configure your application using the Startup class. I talked about adding static content, exceptions and environment specific configurations

For more details about the Startup class, I highly recommend the book &#8220;<a href="https://www.amazon.com/Pro-ASP-NET-Core-MVC-2/dp/148423149X" target="_blank" rel="noopener noreferrer">Pro ASP.NET Core MVC 2</a>&#8220;. You can find the source code for this demo on <a href="https://github.com/WolfgangOfner/MVC-Core-Startup" target="_blank" rel="noopener noreferrer">GitHub</a>.