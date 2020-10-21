---
title: Dependency Injection in ASP.NET Core MVC
date: 2019-05-07T11:32:49+02:00
author: Wolfgang Ofner
categories: [ASP.NET]
tags: [.net core, ASP.NET Core MVC, 'C#', Dependency Injection]
---
Dependency Injection is a technique that helps to create flexible applications and simplifies unit testing. .NET Core brings dependency injection out of the box, therefore you don&#8217;t have to use any third party tools like Autofac or Ninject anymore.

## Setting up the Demo Application

You can find the source code of the following demo on <a href="https://github.com/WolfgangOfner/MVC-Core-DependencyInjection/tree/master/Set%20up" target="_blank" rel="noopener noreferrer">GitHub</a>.

I created a repository which will provide basic operations for my Customer class like Add and Delete.

<div id="attachment_1680" style="width: 472px" class="wp-caption aligncenter">
  <a href="/wp-content/uploads/2019/04/Setting-up-a-repository-class-for-basic-operations.jpg"><img aria-describedby="caption-attachment-1680" loading="lazy" class="size-full wp-image-1680" src="/wp-content/uploads/2019/04/Setting-up-a-repository-class-for-basic-operations.jpg" alt="Setting up a repository class for basic operations" width="462" height="534" /></a>
  
  <p id="caption-attachment-1680" class="wp-caption-text">
    Setting up a repository class for basic operations
  </p>
</div>

Next, I created a view which will display the customer&#8217;s name and age in a table.

<div id="attachment_1681" style="width: 577px" class="wp-caption aligncenter">
  <a href="/wp-content/uploads/2019/04/The-view-to-display-the-customer-data.jpg"><img aria-describedby="caption-attachment-1681" loading="lazy" class="wp-image-1681" src="/wp-content/uploads/2019/04/The-view-to-display-the-customer-data.jpg" alt="The view to display the customer data" width="567" height="700" /></a>
  
  <p id="caption-attachment-1681" class="wp-caption-text">
    The view to display the customer data
  </p>
</div>

Lastly, I call this view from the controller with the CustomerRepository creating some data.

<div id="attachment_1683" style="width: 377px" class="wp-caption aligncenter">
  <a href="/wp-content/uploads/2019/04/Hardcoding-the-usage-of-the-CustomerRepository-in-the-controller.jpg"><img aria-describedby="caption-attachment-1683" loading="lazy" class="size-full wp-image-1683" src="/wp-content/uploads/2019/04/Hardcoding-the-usage-of-the-CustomerRepository-in-the-controller.jpg" alt="Hardcoding the usage of the CustomerRepository in the controller" width="367" height="71" /></a>
  
  <p id="caption-attachment-1683" class="wp-caption-text">
    Hard-coding the usage of the CustomerRepository in the controller
  </p>
</div>

The CustomerRepository is hard-coded into the controller. This means that if you want to call the view with a different repository, you have to change the code and recompile it. Also, unit testing is really hard if you don&#8217;t have any interfaces to fake. This is where dependency injection comes in handy.

## Preparing for Dependency Injection

The term dependency injection (DI) describes an approach to creating loosely coupled components, which are used automatically by MVC. This means that controllers and other components don&#8217;t need to have any knowledge of how the types they require are created. Dependency injection might seem abstract in the beginning but let&#8217;s take a look at the modified HomeController:

<div id="attachment_1685" style="width: 389px" class="wp-caption aligncenter">
  <a href="/wp-content/uploads/2019/04/Preparing-the-Home-Controller-for-dependency-injection.jpg"><img aria-describedby="caption-attachment-1685" loading="lazy" class="size-full wp-image-1685" src="/wp-content/uploads/2019/04/Preparing-the-Home-Controller-for-dependency-injection.jpg" alt="Preparing the Home Controller for dependency injection" width="379" height="213" /></a>
  
  <p id="caption-attachment-1685" class="wp-caption-text">
    Preparing the Home Controller for dependency injection
  </p>
</div>

I am passing an IRepository object to the constructor of the class. From now on the class doesn&#8217;t have to care to get the right object. The right object will be passed. This behavior is also known as the Hollywood Principle. The only problem with this code is, that nobody tells MVC which object to pass as the IRepository and therefore it passes null which will lead to an exception.

<div id="attachment_1686" style="width: 710px" class="wp-caption aligncenter">
  <a href="/wp-content/uploads/2019/04/Exception-when-starting-the-application.jpg"><img aria-describedby="caption-attachment-1686" loading="lazy" class="wp-image-1686" src="/wp-content/uploads/2019/04/Exception-when-starting-the-application.jpg" alt="Exception when starting the application" width="700" height="102" /></a>
  
  <p id="caption-attachment-1686" class="wp-caption-text">
    Exception when starting the application
  </p>
</div>

In the next section, I will configure MVC, so it knows which object it should inject as IRepository.

### Hollywood Principle

The Hollywood Principle says: &#8220;Don&#8217;t call us, We will call you&#8221;. This means don&#8217;t do anything until further notice. In a more technical term: don&#8217;t instantiate an object yourself and do work with it, wait until someone calls you with an object on which you can do your operation.

For example, I have a calculator class which logs all the steps of the calculation. Instead of instantiating the logger itself, the calculator class expects an ILogger interface which it will use to log. If I want to use this calculator class now, I provide it with my logger. This can be a file logger, a console logger or a database logger. The class doesn&#8217;t care about. All it does is ILogger.Log(&#8220;My log message&#8221;). This behavior makes the classes loosely coupled which means they can be extended and tested (most of the time) easily.

### Configuring the Service Provider

In the previous section, I added the IRepository interface to the Home controller but it ended in an exception. To fix this problem, all you have to do is the following line to the ConfigureServices method of the Startup class:

services.AddTransient<IRepository, ProductRepository>();

<div id="attachment_1688" style="width: 581px" class="wp-caption aligncenter">
  <a href="/wp-content/uploads/2019/04/Register-the-IRepository-interface-as-ProductRepository.jpg"><img aria-describedby="caption-attachment-1688" loading="lazy" class="size-full wp-image-1688" src="/wp-content/uploads/2019/04/Register-the-IRepository-interface-as-ProductRepository.jpg" alt="Register the IRepository interface as ProductRepository" width="571" height="90" /></a>
  
  <p id="caption-attachment-1688" class="wp-caption-text">
    Register the IRepository interface as ProductRepository
  </p>
</div>

There are three different Methods to register a service, depending on its scope:

  * Transient
  * Scoped
  * Singleton

I will explain the differences in the section &#8220;Understanding Service Life Cycles&#8221;. For now, add the line and that&#8217;s all you have to do to fix the previous exception.

<div id="attachment_1689" style="width: 369px" class="wp-caption aligncenter">
  <a href="/wp-content/uploads/2019/04/The-application-works-again.jpg"><img aria-describedby="caption-attachment-1689" loading="lazy" class="size-full wp-image-1689" src="/wp-content/uploads/2019/04/The-application-works-again.jpg" alt="The application works again" width="359" height="187" /></a>
  
  <p id="caption-attachment-1689" class="wp-caption-text">
    The application works again
  </p>
</div>

## Using Dependency Injection for Concrete Types

Dependency injection can also be used for concrete types, which are not accessed through interfaces. While this doesn’t provide the loose-coupling advantages of using an interface, it is a useful technique because it allows objects to be accessed anywhere in an application and puts concrete types under lifecycle management.

In the following example, I created a new WeatherService class and added it to the Home controller. There, I created a new action which returns only the string provided by the service. It is not the most useful implementation but it shows how it works.

<div id="attachment_1690" style="width: 607px" class="wp-caption aligncenter">
  <a href="/wp-content/uploads/2019/04/Adding-the-weather-service-to-the-Home-controller.jpg"><img aria-describedby="caption-attachment-1690" loading="lazy" class="size-full wp-image-1690" src="/wp-content/uploads/2019/04/Adding-the-weather-service-to-the-Home-controller.jpg" alt="Adding the weather service to the Home controller" width="597" height="336" /></a>
  
  <p id="caption-attachment-1690" class="wp-caption-text">
    Adding the weather service to the Home controller
  </p>
</div>

Next, I register the WeatherService in the Startup class. Since there is no mapping between a service type and an implementation type in this solution, you have to use an override of the AddTransient method which accepts a single type parameter that tells the service provider that it should instantiate the WeatherService class to resolve a dependency on this type.

<div id="attachment_1691" style="width: 582px" class="wp-caption aligncenter">
  <a href="/wp-content/uploads/2019/04/Register-the-WeatherService-in-the-Startup-class.jpg"><img aria-describedby="caption-attachment-1691" loading="lazy" class="size-full wp-image-1691" src="/wp-content/uploads/2019/04/Register-the-WeatherService-in-the-Startup-class.jpg" alt="Register the WeatherService in the Startup class" width="572" height="106" /></a>
  
  <p id="caption-attachment-1691" class="wp-caption-text">
    Register the WeatherService in the Startup class
  </p>
</div>

The advantages of this approach are that the service provider will resolve any dependencies declared by the concrete class and that you can change the configuration so that more specialized sub-classes are used to resolve dependencies for a concrete class. Concrete classes are managed by the service provider and are also subject to lifecycle features, which I will talk about in the next section.

If you call the new action, you will see the weather information provided by the WeatherService.

<div id="attachment_1692" style="width: 395px" class="wp-caption aligncenter">
  <a href="/wp-content/uploads/2019/04/Getting-data-from-the-weather-service.jpg"><img aria-describedby="caption-attachment-1692" loading="lazy" class="size-full wp-image-1692" src="/wp-content/uploads/2019/04/Getting-data-from-the-weather-service.jpg" alt="Getting data from the weather service" width="385" height="90" /></a>
  
  <p id="caption-attachment-1692" class="wp-caption-text">
    Getting data from the weather service
  </p>
</div>

## Understanding Dependency Injection Service Life Cycle

In the last example, I added a dependency using the AddTransient method. This is one of four different ways that type mappings can be defined. The following table shows the extension methods for the service provider dependency injection:

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
        AddTransient<service, implType>()
      </td>
      
      <td>
        This method tells the service provider to create a new instance of the implementation type for every dependency on the service type.
      </td>
    </tr>
    
    <tr>
      <td>
        AddTransient<service>()
      </td>
      
      <td>
        This method is used to register a single type, which will be instantiated for every dependency.
      </td>
    </tr>
    
    <tr>
      <td>
        AddTransient<service>(factoryFunc)
      </td>
      
      <td>
        This method is used to register a factory function that will be invoked to create an implementation object for every dependency on the service type.
      </td>
    </tr>
    
    <tr>
      <td>
        AddTransient<service>(factoryFunc) or AddScoped<service>() or AddScoped<service>(factoryFunc)
      </td>
      
      <td>
        These methods tell the service provider to reuse instances of the implementation type so that all service requests made by components associated with a<br /> common scope, which is usually a single HTTP request, share the same object. These methods follow the same pattern as the corresponding AddTransient methods.
      </td>
    </tr>
    
    <tr>
      <td>
        AddSingleton<service, implType>() or AddSingleton<service>() or AddSingleton<service>(factoryFunc)
      </td>
      
      <td>
        These methods tell the service provider to create a new instance of the implementation type for the first service request and then reuse it for every subsequent service<br /> request.
      </td>
    </tr>
    
    <tr>
      <td>
        AddSingleton<service>(instance)
      </td>
      
      <td>
        This method provides the service provider with an object that should be used to service all service requests.
      </td>
    </tr>
  </table>
</div>

###  Using the Transient Life Cycle

The simplest way to start using dependency injection is to use the AddTransient method, which tells the service provider to create a new instance of the implementation type whenever it needs to resolve a dependency. The transient life cycle incurs the cost of creating a new instance of the implementation class every time a dependency is resolved, but the advantage is that you don&#8217;t have to worry about managing concurrent access or ensure that objects can be safely reused for multiple requests.

In my experience, the transient life cycle is used for most of the applications.

#### Using a Factory Function

One version of the AddTransient method accepts a factory function that is invoked every time there is a dependency on the service type. This allows the object that is created to be varied so that different dependencies receive instances of different types or instances that are configured differently.

To demonstrate this behavior, I extended the ConfigureServices method of the Startup class. First, I inject an IHostingEnvironment object which indicates on which environment the application is running. Afterward, I check this variable and if it is development, I instantiate a ProductRepository as IRepository. Otherwise, I instantiate a CustomerRepository.

<div id="attachment_1693" style="width: 616px" class="wp-caption aligncenter">
  <a href="/wp-content/uploads/2019/04/Instantiate-the-IRepository-interface-depending-on-the-hosting-environment.jpg"><img aria-describedby="caption-attachment-1693" loading="lazy" class="size-full wp-image-1693" src="/wp-content/uploads/2019/04/Instantiate-the-IRepository-interface-depending-on-the-hosting-environment.jpg" alt="Instantiate the IRepository interface depending on the hosting environment" width="606" height="425" /></a>
  
  <p id="caption-attachment-1693" class="wp-caption-text">
    Instantiate the IRepository interface depending on the hosting environment
  </p>
</div>

### Using the Scoped Life Cycle

This life cycle creates a single object from the implementation class that is used to resolve all the dependencies associated with a single scope, which generally means a single HTTP request. Since the default scope is the HTTP request, this life cycle allows for a single object to be shared by all the components that process a request and is most often used for sharing common context data when writing custom classes, such as routes.

Note that there are also versions of the AddScoped method that accept a factory function and that can be used to register a concrete type. These methods work in the same way as the AddTransient method.

<div id="attachment_1694" style="width: 393px" class="wp-caption aligncenter">
  <a href="/wp-content/uploads/2019/04/Adding-a-scoped-dependency.jpg"><img aria-describedby="caption-attachment-1694" loading="lazy" class="wp-image-1694 size-full" src="/wp-content/uploads/2019/04/Adding-a-scoped-dependency.jpg" alt="Adding a scoped for your dependency injection" width="383" height="21" /></a>
  
  <p id="caption-attachment-1694" class="wp-caption-text">
    Adding a scoped dependency
  </p>
</div>

### Using the Singleton Life Cycle

The singleton life cycle ensures that a single object is used to resolve all the dependencies for a given service type. When using this life cycle, you must ensure that the implementation classes used to resolve dependencies are safe for concurrent access.

<div id="attachment_1695" style="width: 306px" class="wp-caption aligncenter">
  <a href="/wp-content/uploads/2019/04/Adding-a-singleton-dependency.jpg"><img aria-describedby="caption-attachment-1695" loading="lazy" class="wp-image-1695 size-full" src="/wp-content/uploads/2019/04/Adding-a-singleton-dependency.jpg" alt="Adding a singleton for your dependency injection" width="296" height="23" /></a>
  
  <p id="caption-attachment-1695" class="wp-caption-text">
    Adding a singleton dependency
  </p>
</div>

## Dependency Injection in Actions

The standard way to declare a dependency is through the constructor. Additionally to the standard way, MVC provides the functionality to inject an action, called action injection. Action injection allows dependencies to be declared through parameters to action methods. To be more precise, action injection is provided by the model binding system. All you have to do is using the FromService attribute before your parameter. Also, don&#8217;t forget to register your service in the Startup class.

<div id="attachment_1699" style="width: 549px" class="wp-caption aligncenter">
  <a href="/wp-content/uploads/2019/05/Using-action-injection.jpg"><img aria-describedby="caption-attachment-1699" loading="lazy" class="size-full wp-image-1699" src="/wp-content/uploads/2019/05/Using-action-injection.jpg" alt="Using action injection" width="539" height="73" /></a>
  
  <p id="caption-attachment-1699" class="wp-caption-text">
    Using action injection
  </p>
</div>

MVC uses the services provider to get in instance of the CustomerService class which is used to load all the customers. Using action injection is less common than the constructor injection, but it can be useful when you have a dependency on an object that is expensive to create and that is only required in only one of the actions of a controller. Using constructor injection resolves the dependencies for all action methods, even if the one used to handle the request doesn&#8217;t use the implementation object.

## Dependency Injection in Properties

The third way to use dependency injection provided by MVC is property injection. Here, a set of specialized attributes is used to receive specific types via property injection in the controllers and view components. You won&#8217;t need to use these attributes if you derive your controllers from the Controller base class because the context information is exposed through convenience properties.

The following table shows the specialized property injection attributes:

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
        ControllerContext
      </td>
      
      <td>
        This attribute sets a ControllerContext property, which provides a superset of the functionality of the ActionContext class.
      </td>
    </tr>
    
    <tr>
      <td>
        ActionContext
      </td>
      
      <td>
        This attribute sets an ActionContext property to provide context information to action methods. The Controller classes expose the context information through an ActionContext property.
      </td>
    </tr>
    
    <tr>
      <td>
        ViewContext
      </td>
      
      <td>
        This attribute sets a ViewContext property to provide context data for view operations.
      </td>
    </tr>
    
    <tr>
      <td>
        ViewComponentContext
      </td>
      
      <td>
        This attribute sets a ViewComponentContext property for view components.
      </td>
    </tr>
    
    <tr>
      <td>
        ViewDataDictionary
      </td>
      
      <td>
        This attribute sets a ViewDataDictionary property to provide access to the model binding data.
      </td>
    </tr>
  </table>
</div>

I have never used property binding and don&#8217;t see a use case where to use it. Therefore, I am not going into more detail here.

## Manually Requesting an Implementation Object

MVC provides the dependency feature for you. There can be occasions when it can be useful to create an implementation for an interface without relying on dependency injection. In these situations, you can work directly with the service provider.

<div id="attachment_1700" style="width: 710px" class="wp-caption aligncenter">
  <a href="/wp-content/uploads/2019/05/Resolve-an-object-manually.jpg"><img aria-describedby="caption-attachment-1700" loading="lazy" class="wp-image-1700" src="/wp-content/uploads/2019/05/Resolve-an-object-manually.jpg" alt="Resolve an object manually wihtout dependency injection" width="700" height="160" /></a>
  
  <p id="caption-attachment-1700" class="wp-caption-text">
    Resolve an object manually
  </p>
</div>

The HttpContext object returned by the property of the same name defines a RequestServices method that returns an IServiceProvider object. This is known as the service locator pattern, which some developers believe should be avoided. I also think that it should be avoided but there might be some cases where it is reasonable to use it. For example, if the normal way of receiving dependencies through the constructor can&#8217;t be used for some reasons.

## Conclusion

Today, I talked about the different types of dependency injection and how to use them with your ASP.NET Core MVC application. If you start with dependency injection, I would only use constructor injection since it is the most common form of it.

For more details about complex configurations, I highly recommend the book &#8220;<a href="https://www.amazon.com/Pro-ASP-NET-Core-MVC-2/dp/148423149X" target="_blank" rel="noopener noreferrer">Pro ASP.NET Core MVC 2</a>&#8220;. You can find the source code of this demo on <a href="https://github.com/WolfgangOfner/MVC-Core-DependencyInjection" target="_blank" rel="noopener noreferrer">GitHub</a>.