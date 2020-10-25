---
title: Decorator Pattern in .NET Core 3.1
date: 2020-07-01T08:51:30+02:00
author: Wolfgang Ofner
categories: [Design Pattern]
tags: [.net core 3.1, 'C#', software architecture]
---
The decorator pattern is a structural design pattern used for dynamically adding behavior to a class without changing the class. You can use multiple decorators to extend the functionality of a class whereas each decorator focuses on a single-tasking, promoting separations of concerns. Decorator classes allow functionality to be added dynamically without changing the class thus respecting the open-closed principle.

You can see this behavior on the UML diagram where the decorator implements the interface to extend the functionality.

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2020/06/Decorator-Pattern-UML.jpg"><img loading="lazy" src="/assets/img/posts/2020/06/Decorator-Pattern-UML.jpg" alt="Decorator Pattern UML" /></a>
  
  <p>
    Decorator Pattern UML (Source)
  </p>
</div>

## When to use the Decorator Pattern

The decorator pattern should be used to extend classes without changing them. Mostly this pattern is used for cross-cutting concerns like logging or caching. Another use case is to modify data that is sent to or from a component.

## Decorator Pattern Implementation in ASP .NET Core 3.1

You can find the code of the demo on <a href="https://github.com/WolfgangOfner/.NetCore-DecoratorPattern" target="_blank" rel="noopener noreferrer">Github</a>.

I created a DataService class with a GetData method which returns a list of ints. Inside the loop, I added a Thread.Sleep to slow down the data collection a bit to make it more real-world like.

```csharp  
public class DataService : IDataService
{
    public List<int> GetData()
    {
        var data = new List<int>();

        for (var i = 0; i < 10; i++)
        {
            data.Add(i);

            Thread.Sleep(350);
        }

        return data;
    }
}  
```

This method is called in the GetData action and then printed to the website. The first feature I want to add with a decorator is logging. To achieve that, I created the DataServiceLoggingDecorator class and implement the IDataService interface. In the GetData method, I add a stopwatch to measure how long collecting data takes and then log the time it took.

```csharp  
public class DataServiceLoggingDecorator : IDataService
{
    private readonly IDataService _dataService;
    private readonly ILogger<DataServiceLoggingDecorator> _logger;

    public DataServiceLoggingDecorator(IDataService dataService, ILogger<DataServiceLoggingDecorator> logger)
    {
        _dataService = dataService;
        _logger = logger;
    }

    public List<int> GetData()
    {
        _logger.LogInformation("Starting to get data");
        var stopwatch = Stopwatch.StartNew();

        var data = _dataService.GetData();

        stopwatch.Stop();
        var elapsedTime = stopwatch.ElapsedMilliseconds;

        _logger.LogInformation($"Finished getting data in {elapsedTime} milliseconds");

        return data;
    }
}  
```

Additionally, I want to add caching also using a decorator. To do that, I created the DataServiceCachingDecorator class and also implemented the IDataService interface. To cache the data, I use IMemoryCache and check the cache if it contains my data. If not, I load it and then add it to the cache. If the cache already has the data, I simply return it. The cache item is valid for 2 hours.

```csharp  
public class DataServiceCachingDecorator : IDataService
{
    private readonly IDataService _dataService;
    private readonly IMemoryCache _memoryCache;

    public DataServiceCachingDecorator(IDataService dataService, IMemoryCache memoryCache)
    {
        _dataService = dataService;
        _memoryCache = memoryCache;
    }

    public List<int> GetData()
    {
        const string cacheKey = "data-key";

        if (_memoryCache.TryGetValue<List<int>>(cacheKey, out var data))
        {
            return data;
        }

        data = _dataService.GetData();
        
        _memoryCache.Set(cacheKey, data, TimeSpan.FromMinutes(120));

        return data;
    }
}  
```

All that is left to do is to register the service and decorator in the ConfigureServices method of the startup class with the following code:

```csharp  
services.AddTransient<IDataService, DataService>();

services.AddScoped(serviceProvider =>  
{  
    var logger = serviceProvider.GetService<ILogger<DataServiceLoggingDecorator>>();  
    var memoryCache = serviceProvider.GetService<IMemoryCache>();
    
    IDataService concreteService = new DataService();  
    IDataService loggingDecorator = new DataServiceLoggingDecorator(concreteService, logger);  
    IDataService cacheingDecorator = new DataServiceCachingDecorator(loggingDecorator, memoryCache);
    
    return cacheingDecorator;  
});  
```

With everything in place, I can call the GetData method from the service which gets logged and the data placed in the cache. When I call the method again, the data will be loaded from the cache.

```csharp  
public class HomeController : Controller
{
    private readonly IDataService _dataService;
    private readonly ILogger<HomeController> _logger;

    public HomeController(ILogger<HomeController> logger, IDataService dataService)
    {
        _logger = logger;
        _dataService = dataService;
    }

    public IActionResult Index()
    {
        return View();
    }

    public IActionResult GetData()
    {
        var data = _dataService.GetData();

        return View(data);
    }
}	  
```

Start the application and click on Get Data. After a couple of seconds, you will see the data displayed.

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2020/06/Displaying-the-loaded-data.jpg"><img aria-describedby="caption-attachment-2243" loading="lazy" class="size-full wp-image-2243" src="/assets/img/posts/2020/06/Displaying-the-loaded-data.jpg" alt="Displaying the loaded data" /></a>
  
  <p>
    Displaying the loaded data
  </p>
</div>

When you load the site again, the data will be displayed immediately due to the cache.

## Conclusion

The decorator pattern can be used to extend classes without changing the code. This helps you to achieve the open-closed principle and separation of concerns. Mostly the decorator pattern is used for cross-cutting concerns like caching and logging.

You can find the code of the demo on <a href="https://github.com/WolfgangOfner/.NetCore-DecoratorPattern" target="_blank" rel="noopener noreferrer">Github</a>.