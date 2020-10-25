---
title: Middleware in ASP.NET Core MVC
date: 2019-05-01T11:17:11+02:00
author: Wolfgang Ofner
categories: [ASP.NET]
tags: [.net core, ASP.NET Core MVC, 'C#']
---
Middleware is the term used for the components that are combined to form the request pipeline. This pipeline is arranged like a chain. The request is either returned by the middleware or passed to the next one until a response is sent back. Once a response is created, the response will travel the chain back, passing all middlewares again, which allows them to modify this response.

It may not be intuitive at first, but this allows for a lot of flexibility in the way the parts of an application are combined.

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2019/04/Middlewares-in-ASP.NET-Core-MVC.jpg"><img loading="lazy" src="/assets/img/posts/2019/04/Middlewares-in-ASP.NET-Core-MVC.jpg" alt="Middlewares in ASP.NET Core MVC" /></a>
  
  <p>
    Middlewares in ASP.NET Core MVC (<a href="https://docs.microsoft.com/en-us/aspnet/core/fundamentals/middleware/?view=aspnetcore-2.2">Source</a>)
  </p>
</div>

You can find the source code of the following demo on <a href="https://github.com/WolfgangOfner/MVC-Core-Middleware" target="_blank" rel="noopener noreferrer">GitHub</a>.

## Creating a Content-Generating Middleware

A middleware can be a pretty simple class since it doesn&#8217;t implement an interface and doesn&#8217;t derive from a base class. Instead, the constructor takes a RequestDelegate object and defines an Invoke method. The RequestDelegate links to the next middleware in the chain and the Invoke method is called when the ASP.NET application receives an HTTP request. All the information of the HTTP requests and responses are provided through the HttpContext in the Invoke method.

On the following screenshot, you can see a simple implementation of a middleware which returns a string  if the request path is /contentmiddleware

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2019/04/The-content-creating-middleware-implementation.jpg"><img loading="lazy" src="/assets/img/posts/2019/04/The-content-creating-middleware-implementation.jpg" alt="The content-creating middleware implementation" /></a>
  
  <p>
    The content-creating middleware implementation
  </p>
</div>

The request pipeline or chain of middlewares is created in the Configure method of the Startup class. All you have to do is app.UseMiddleware<MyMiddleware>(); A request is only passed through a middleware when it is registered in the Startup class.

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2019/04/Register-the-content-creating-middleware.jpg"><img loading="lazy" src="/assets/img/posts/2019/04/Register-the-content-creating-middleware.jpg" alt="Register the content-creating middleware" /></a>
  
  <p>
    Register the content-creating middleware
  </p>
</div>

That&#8217;s already everything you have to do to use the middleware. Start the application and enter /contentmiddleware and you will see the response from the middleware.

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2019/04/Response-from-the-content-creating-middleware.jpg"><img loading="lazy" src="/assets/img/posts/2019/04/Response-from-the-content-creating-middleware.jpg" alt="Response from the content-creating middleware" /></a>
  
  <p>
    Response from the content-creating middleware
  </p>
</div>

## Creating a Short-Circuiting Middleware

A short-circuiting middleware intercepts the request before the content generating components (for example a controller) is reached. The main reason for doing this is performance. This type of middleware is called short-circuiting because it doesn&#8217;t always forward the request to the next component in the chain.  For example if your application doesn&#8217;t allow Chrome users, the middleware can check the client agent and if it is Chrome, a response with an error message is created.

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2019/04/The-short-circuiting-middleware.jpg"><img loading="lazy" src="/assets/img/posts/2019/04/The-short-circuiting-middleware.jpg" alt="The short-circuiting middleware" /></a>
  
  <p>
    The short-circuiting middleware
  </p>
</div>

It is important to note that middlewares are called in the same order as they are registered in the Startup class. The middleware which is registered first will handle the request first. Short-circuiting middlewares should always be placed at the front of the chain.

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2019/04/The-order-of-the-middlewares-is-important.jpg"><img loading="lazy" src="/assets/img/posts/2019/04/The-order-of-the-middlewares-is-important.jpg" alt="The order of the middlewares is important" /></a>
  
  <p>
    The order of the middlewares is important
  </p>
</div>

If you make a request from Chrome (also from Edge since it is using Chromium now), you will see the access denied message.

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2019/04/Calls-with-Chrome-are-blocked.jpg"><img loading="lazy" src="/assets/img/posts/2019/04/Calls-with-Chrome-are-blocked.jpg" alt="Calls with Chrome are blocked" /></a>
  
  <p>
    Calls with Chrome are blocked
  </p>
</div>

Calls with a different browser, for example, Firefox still create a response.

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2019/04/Calls-with-Firefox-are-allowed.jpg"><img loading="lazy" src="/assets/img/posts/2019/04/Calls-with-Firefox-are-allowed.jpg" alt="Calls with Firefox are allowed" /></a>
  
  <p>
    Calls with Firefox are allowed
  </p>
</div>

## <span style="color: #000000;">Creating a Request-Editing Middleware</span>

A request-editing middleware changes the requests before it reaches other components but doesn&#8217;t create a response. This can be used to prepare the request for easier processing later or for enriching the request with platform-specific features. The following example shows a middleware which sets a context item if the browser used is Edge.

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2019/04/The-request-editing-middleware.jpg"><img loading="lazy" src="/assets/img/posts/2019/04/The-request-editing-middleware.jpg" alt="The request-editing middleware" /></a>
  
  <p>
    The request-editing middleware
  </p>
</div>

## Creating a Response-Editing Middleware

Since there is a request-editing middleware, it won&#8217;t be surprising that there is also a response-editing middleware which changes the response before it is sent to the user. This type of middleware is often used for logging or handling errors.

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2019/04/The-response-editing-middleware.jpg"><img loading="lazy" src="/assets/img/posts/2019/04/The-response-editing-middleware.jpg" alt="The response-editing middleware" /></a>
  
  <p>
    The response-editing middleware
  </p>
</div>

Now register the middleware in your Startup class. It may not be intuitive but it is important that a response-editing middleware is registered first because the response passes all middlewares in the reverse order of the request. This means that the first middleware processes the request first and the response last.

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2019/04/Register-the-response-editing-middleware-first.jpg"><img loading="lazy" src="/assets/img/posts/2019/04/Register-the-response-editing-middleware-first.jpg" alt="Register the response-editing middleware first" /></a>
  
  <p>
    Register the response-editing middleware first
  </p>
</div>

If you start your application and enter an URL which your system doesn&#8217;t know (and don&#8217;t use Chrome), you will see your custom error text.

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2019/04/Response-created-by-the-middleware.jpg"><img loading="lazy" src="/assets/img/posts/2019/04/Response-created-by-the-middleware.jpg" alt="Response created by the middleware" /></a>
  
  <p>
    Response created by the middleware
  </p>
</div>

## Conclusion

In this post, I talked about middlewares. I explained what they are and what different types there are.

For more details about complex configurations, I highly recommend the book &#8220;<a href="https://www.amazon.com/Pro-ASP-NET-Core-MVC-2/dp/148423149X" target="_blank" rel="noopener noreferrer">Pro ASP.NET Core MVC 2</a>&#8220;. You can find the source code of the demo on <a href="https://github.com/WolfgangOfner/MVC-Core-Middleware" target="_blank" rel="noopener noreferrer">GitHub</a>.