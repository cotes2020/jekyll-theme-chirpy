---
title: Configure ASP.NET Core MVC
date: 2019-05-03T11:17:40+02:00
author: Wolfgang Ofner
categories: [ASP.NET]
tags: [.net core, ASP.NET Core MVC, 'C#']
---
Some configurations like the connection string usually change on every environment your application is running on. Instead of hard-coding this information into your application, ASP.NET Core enables you to provide configuration data through different sources such as environment variables, command-line arguments or JSON files. You can find the source code for this demo on <a href="https://github.com/WolfgangOfner/MVC-Core-Configure" target="_blank" rel="noopener noreferrer">GitHub</a>.

On the following screenshot, I configure my application in the BuildWebHost method of the program class to read the appsettings.json file, the environment variables and if available the command line arguments.

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2019/04/Configure-the-application-to-read-settings-from-a-json-file-environment-variables-and-the-command-line-parameter.jpg"><img aria-describedby="caption-attachment-1645" loading="lazy" class="size-full wp-image-1645" src="/assets/img/posts/2019/04/Configure-the-application-to-read-settings-from-a-json-file-environment-variables-and-the-command-line-parameter.jpg" alt="Configure the application to read settings from a json file, environment variables and the command line parameter" /></a>
  
  <p>
    Configure the application to read settings from a JSON file, environment variables and the command line parameter
  </p>
</div>

Appsettings.json is the conventional name for the configuration file but you can choose any name you like. The two additional parameters mark the file as optional and enable reloading when the file changes. This enables you to change the configuration during runtime without the need of restarting the web server like you had to do when you changed, for example, the web.config file. Only because you can change the configuration during runtime doesn&#8217;t mean that you should since it is a recipe for downtime.

The ConfigureAppConfiguration method is used to handle the configuration data and its arguments are a WebHostBuilderContext object and an IConfigurationBuilder object. The WebHostBuilderContext class has two properties:

  * HostingEnvironment
  * Configuration

The HostingEnvironment provides information about the hosting environment in which the application is running whereas the Configuration property provides read-only access to the configuration data.

The IConfigurationBuilder provides three extension methods:

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
        AddJsonFile
      </td>
      
      <td>
        This method is used to load configuration data from a JSON file, such as appsettings.json.
      </td>
    </tr>
    
    <tr>
      <td>
        AddEnvironmentVariables
      </td>
      
      <td>
        This method is used to load configuration data from environment variables.
      </td>
    </tr>
    
    <tr>
      <td>
        AddCommandLine
      </td>
      
      <td>
        This method is used to load configuration data from the command-line arguments used to start the application.
      </td>
    </tr>
  </table>
</div>

###  Creating the JSON Configuration File

The most common uses for the appsettings.json file are to store your connection strings and logging setting, but you can store any data that your application needs. On the following screenshot, I add the ShortCircuitMiddleware section containing EnableBrowserShortCircuit with the value true to the appsettings.json file.

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2019/04/Adding-values-to-the-appsettings.json-file.jpg"><img loading="lazy" size-full" src="/assets/img/posts/2019/04/Adding-values-to-the-appsettings.json-file.jpg" alt="Adding values to the appsettings.json file to configure your application" /></a>
  
  <p>
    Adding values to the appsettings.json file
  </p>
</div>

In JSON everything has to be quoted exception bool and number values. If you want to add a new section, add a comma after the closing bracket of the ShortCircuitMiddleware section. Be aware to not add a trailing comma at the end if you don&#8217;t have another section there. This and missing quotes are the most common mistakes in a JSON file.

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2019/04/Adding-more-values-to-the-appsettings.json-file.jpg"><img loading="lazy" size-full" src="/assets/img/posts/2019/04/Adding-more-values-to-the-appsettings.json-file.jpg" alt="Adding more values to the appsettings.json file to configure your application" /></a>
  
  <p>
    Adding more values to the appsettings.json file
  </p>
</div>

### Using Configuration Data

The Startup class can access the configuration data by defining a constructor with an IConfiguration argument. When the UseStartup method is called in the Program class, the configuration data prepared by the ConfigureAppConfiguration is used to create the Startup object.

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2019/04/Setting-the-Configuration-in-the-Startup-constructor.jpg"><img aria-describedby="caption-attachment-1648" loading="lazy" class="size-full wp-image-1648" src="/assets/img/posts/2019/04/Setting-the-Configuration-in-the-Startup-constructor.jpg" alt="Setting the Configuration in the Startup constructor" /></a>
  
  <p>
    Setting the Configuration in the Startup constructor
  </p>
</div>

The IConfiguration object is received by the constructor and assigned to a property called Configuration, which can then be used to access the configuration data that has been loaded from  
environment variables, the command line, and the appsettings.json file. To obtain a value, you navigate through the structure of the data to the configuration section you require. The IConfigurationInterface defines the following member variables to do that:

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
        [key]
      </td>
      
      <td>
        The indexer is used to obtain a string value for a specific key.
      </td>
    </tr>
    
    <tr>
      <td>
        GetSection(name)
      </td>
      
      <td>
        This method returns an IConfiguration object that represents a section of the configuration data.
      </td>
    </tr>
    
    <tr>
      <td>
        GetChildren()
      </td>
      
      <td>
        This method returns an enumeration of the IConfiguration objects that represent the subsections of the current configuration object.
      </td>
    </tr>
  </table>
</div>

Additionally, the IConfiguration interface provides the following extension methods to get and convert values from string into other data types:

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
        GetValue<T>(keyName)
      </td>
      
      <td>
        This method gets the value associated with the specified key and attempts to convert it to the type T.
      </td>
    </tr>
    
    <tr>
      <td>
        GetValue<T>(keyName, defaultValue)
      </td>
      
      <td>
        This method gets the value associated with the specified key and attempts to convert it to the type T. The default value will be used if there is no value for the key in the configuration data.
      </td>
    </tr>
  </table>
</div>

With these extension methods, I read the configuration and if EnableBrowserShortCircuiting is true, I add the ShortCircuitMiddleware to my application.

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2019/04/Use-the-configuration-to-decide-if-the-middleware-should-be-enabled-or-not.jpg"><img loading="lazy" src="/assets/img/posts/2019/04/Use-the-configuration-to-decide-if-the-middleware-should-be-enabled-or-not.jpg" alt="Use the configuration to decide if the middleware should be enabled or not" /></a>
  
  <p>
    Use the configuration to decide if the middleware should be enabled or not
  </p>
</div>

&nbsp;

It is important not to assume that a configuration value will be specified. It is good practice to program defensively and check for null using the null conditional operator to ensure that the ShortCircuitMiddleware section was received before its value is used.

## Configure Logging

Many of the built-in middlewares already generate logging output. To set up the logging, you have to set it up in the Program class:

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2019/04/Configure-logging-in-the-Program-class.jpg"><img aria-describedby="caption-attachment-1651" loading="lazy" class="size-full wp-image-1651" src="/assets/img/posts/2019/04/Configure-logging-in-the-Program-class.jpg" alt="Configure logging in the Program class" /></a>
  
  <p>
    Configure logging in the Program class
  </p>
</div>

The ConfigureLogging method sets up the logging system using a lambda function that receives a WebHostingBuilderContext object and an ILoggingBuilder object. The ILoggingBuilder interface provides the following extension methods:

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
        AddConfiguration
      </td>
      
      <td>
        This method is used to configure the logging system using the configuration data that has been loaded from the appsettings.json file, from the command line, or from<br /> environment variables.
      </td>
    </tr>
    
    <tr>
      <td>
        AddConsole
      </td>
      
      <td>
        This method sends logging messages to the console, which is useful when starting the application using the dotnet run command.
      </td>
    </tr>
    
    <tr>
      <td>
        AddDebug
      </td>
      
      <td>
        This method sends logging messages to the debug output window when the Visual Studio debugger is running.
      </td>
    </tr>
    
    <tr>
      <td>
        AddEventLog
      </td>
      
      <td>
        This method sends logging messages to the Windows Event Log, which is useful if you deploy to Windows Server and want the log messages from the ASP.NET Core<br /> MVC application to be incorporated with those from other types of application.
      </td>
    </tr>
  </table>
</div>

### Understanding the Logging Configuration Data

Configuration data for the logging is usually defined in the appsettings.json file.

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2019/04/Configuring-logging-in-the-appsettings.json-file.jpg"><img aria-describedby="caption-attachment-1657" loading="lazy" class="size-full wp-image-1657" src="/assets/img/posts/2019/04/Configuring-logging-in-the-appsettings.json-file.jpg" alt="Configuring logging in the appsettings.json file" /></a>
  
  <p>
    Configuring logging in the appsettings.json file
  </p>
</div>

ASP.NET has seven debugging levels:

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
        None
      </td>
      
      <td>
        This level is used to disable logging messages.
      </td>
    </tr>
    
    <tr>
      <td>
        Trace
      </td>
      
      <td>
        This level is used for messages that are useful during development but that are not required in production.
      </td>
    </tr>
    
    <tr>
      <td>
        Debug
      </td>
      
      <td>
        This level is used for detailed messages required by developers to debug problems.
      </td>
    </tr>
    
    <tr>
      <td>
        Information
      </td>
      
      <td>
        This level is used for messages that describe the general operation of the application.
      </td>
    </tr>
    
    <tr>
      <td>
        Warning
      </td>
      
      <td>
        This level is used for messages that describe events that are unexpected but that do not interrupt the application.
      </td>
    </tr>
    
    <tr>
      <td>
        Error
      </td>
      
      <td>
        This level is used for messages that describe errors that interrupt the application.
      </td>
    </tr>
    
    <tr>
      <td>
        Critical
      </td>
      
      <td>
        This level is used for messages that describe catastrophic failures.
      </td>
    </tr>
  </table>
</div>

###  Creating Custom Log Messages

The logging messages in the previous section were generated by the ASP.NET Core and MVC components that handled the HTTP request and generated the response. This kind of message can provide useful information, but you can also generate custom log messages that are specific to your application.

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2019/04/Create-your-own-log-message.jpg"><img aria-describedby="caption-attachment-1658" loading="lazy" class="size-full wp-image-1658" src="/assets/img/posts/2019/04/Create-your-own-log-message.jpg" alt="Create your own log message" /></a>
  
  <p>
    Create your own log message
  </p>
</div>

The ILogger interface defines the functionality required to create log entries and to obtain an object that implements this interface. The value of the constructor argument is provided automatically through dependency injection. The ILogger interface is in the Microsoft.Extensions.Logging namespace. This namespace defines extension methods for each logging level.

## Configuring MVC Services

When you call AddMvc in the ConfigureServices method of the Startup class, it sets up all the services that are required for MVC applications. This has the advantage of convenience because it registers all the MVC services in a single step but does mean that some additional work is required to reconfigure the services to change the default behavior. The AddMvc method returns an object that implements the IMvcBuilder interface and MVC provides a set of extension methods that can be used for advanced configuration. These extension methods are:

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
        AddMvcOptions
      </td>
      
      <td>
        This method configures the services used by MVC. For more details see the next table.
      </td>
    </tr>
    
    <tr>
      <td>
        AddFormatterMappings
      </td>
      
      <td>
        This method is used to configure a feature that allows clients to specify the data format they receive.
      </td>
    </tr>
    
    <tr>
      <td>
        AddJsonOptions
      </td>
      
      <td>
        This method is used to configure the way that JSON data is created.
      </td>
    </tr>
    
    <tr>
      <td>
        AddRazorOptions
      </td>
      
      <td>
        This method is used to configure the Razor view engine.
      </td>
    </tr>
    
    <tr>
      <td>
        AddViewOptions
      </td>
      
      <td>
        This method is used to configure how MVC handles views, including which view engines are used.
      </td>
    </tr>
  </table>
</div>

The AddMvcOptions method configures the most important MVC services. It accepts a function that receives an MvcOptions object, which provides one of the following set of configuration properties:

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
        Conventions
      </td>
      
      <td>
        This property returns a list of the model conventions that are used to customize how MVC creates controllers and actions.
      </td>
    </tr>
    
    <tr>
      <td>
        Filters
      </td>
      
      <td>
        This property returns a list of the global filters.
      </td>
    </tr>
    
    <tr>
      <td>
        FormatterMappings
      </td>
      
      <td>
        This property returns the mappings used to allow clients to specify the data format they receive.
      </td>
    </tr>
    
    <tr>
      <td>
        InputFormatters
      </td>
      
      <td>
        This property returns a list of the objects used to parse request data.
      </td>
    </tr>
    
    <tr>
      <td>
        ModelValidatorProviders
      </td>
      
      <td>
        This property returns a list of the objects used to validate data.
      </td>
    </tr>
    
    <tr>
      <td>
        OutputFormatters
      </td>
      
      <td>
        This property returns a list of the classes that format data sent from API controllers.
      </td>
    </tr>
    
    <tr>
      <td>
        RespectBrowserAcceptHeader
      </td>
      
      <td>
        This property specifies whether the Accept header is taken into account when deciding what data format to use for a response
      </td>
    </tr>
  </table>
</div>

These configuration options are used to fine-tune the way your MVC application works.

## Conclusion

In this post, I talked about configuring your ASP.NET Core MVC application. I showed how to enable logging, add built-in MVC functionalities and how to create a JSON file from which the application can read its configuration. If you want to learn more about more complex configurations, check my post <a href="/dealing-with-complex-configurations-in-asp-net-mvc-core" target="_blank" rel="noopener noreferrer">Dealing with Complex Configurations in ASP.NET MVC Core</a>.

For more details about the configuring ASP.NET Core, I highly recommend the book &#8220;<a href="https://www.amazon.com/Pro-ASP-NET-Core-MVC-2/dp/148423149X" target="_blank" rel="noopener noreferrer">Pro ASP.NET Core MVC 2</a>&#8220;. You can find the source code for this demo on <a href="https://github.com/WolfgangOfner/MVC-Core-Configure" target="_blank" rel="noopener noreferrer">GitHub</a>.