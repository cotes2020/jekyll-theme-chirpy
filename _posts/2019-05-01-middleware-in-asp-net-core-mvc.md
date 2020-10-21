---
title: Middleware in ASP.NET Core MVC
date: 2019-05-01T11:17:11+02:00
author: Wolfgang Ofner
categories: [ASP.NET]
tags: [.net core, ASP.NET Core MVC, 'C#']
---
Middleware is the term used for the components that are combined to form the request pipeline. This pipeline is arranged like a chain. The request is either returned by the middleware or passed to the next one until a response is sent back. Once a response is created, the response will travel the chain back, passing all middlewares again, which allows them to modify this response.

It may not be intuitive at first, but this allows for a lot of flexibility in the way the parts of an application are combined.

<div id="attachment_1623" style="width: 610px" class="wp-caption aligncenter">
  <a href="/wp-content/uploads/2019/04/Middlewares-in-ASP.NET-Core-MVC.jpg"><img aria-describedby="caption-attachment-1623" loading="lazy" class="size-full wp-image-1623" src="/wp-content/uploads/2019/04/Middlewares-in-ASP.NET-Core-MVC.jpg" alt="Middlewares in ASP.NET Core MVC" width="600" height="384" /></a>
  
  <p id="caption-attachment-1623" class="wp-caption-text">
    Middlewares in ASP.NET Core MVC (<a href="https://docs.microsoft.com/en-us/aspnet/core/fundamentals/middleware/?view=aspnetcore-2.2">Source</a>)
  </p>
</div>

You can find the source code of the following demo on <a href="https://github.com/WolfgangOfner/MVC-Core-Middleware" target="_blank" rel="noopener noreferrer">GitHub</a>.

## Creating a Content-Generating Middleware

A middleware can be a pretty simple class since it doesn&#8217;t implement an interface and doesn&#8217;t derive from a base class. Instead, the constructor takes a RequestDelegate object and defines an Invoke method. The RequestDelegate links to the next middleware in the chain and the Invoke method is called when the ASP.NET application receives an HTTP request. All the information of the HTTP requests and responses are provided through the HttpContext in the Invoke method.

On the following screenshot, you can see a simple implementation of a middleware which returns a string  if the request path is /contentmiddleware

<div id="attachment_1626" style="width: 710px" class="wp-caption aligncenter">
  <a href="/wp-content/uploads/2019/04/The-content-creating-middleware-implementation.jpg"><img aria-describedby="caption-attachment-1626" loading="lazy" class="wp-image-1626" src="/wp-content/uploads/2019/04/The-content-creating-middleware-implementation.jpg" alt="The content-creating middleware implementation" width="700" height="344" /></a>
  
  <p id="caption-attachment-1626" class="wp-caption-text">
    The content-creating middleware implementation
  </p>
</div>

The request pipeline or chain of middlewares is created in the Configure method of the Startup class. All you have to do is app.UseMiddleware<MyMiddleware>(); A request is only passed through a middleware when it is registered in the Startup class.

<div id="attachment_1627" style="width: 481px" class="wp-caption aligncenter">
  <a href="/wp-content/uploads/2019/04/Register-the-content-creating-middleware.jpg"><img aria-describedby="caption-attachment-1627" loading="lazy" class="size-full wp-image-1627" src="/wp-content/uploads/2019/04/Register-the-content-creating-middleware.jpg" alt="Register the content-creating middleware" width="471" height="88" /></a>
  
  <p id="caption-attachment-1627" class="wp-caption-text">
    Register the content-creating middleware
  </p>
</div>

That&#8217;s already everything you have to do to use the middleware. Start the application and enter /contentmiddleware and you will see the response from the middleware.

<div id="attachment_1628" style="width: 419px" class="wp-caption aligncenter">
  <a href="/wp-content/uploads/2019/04/Response-from-the-content-creating-middleware.jpg"><img aria-describedby="caption-attachment-1628" loading="lazy" class="size-full wp-image-1628" src="/wp-content/uploads/2019/04/Response-from-the-content-creating-middleware.jpg" alt="Response from the content-creating middleware" width="409" height="117" /></a>
  
  <p id="caption-attachment-1628" class="wp-caption-text">
    Response from the content-creating middleware
  </p>
</div>

## Creating a Short-Circuiting Middleware

A short-circuiting middleware intercepts the request before the content generating components (for example a controller) is reached. The main reason for doing this is performance. This type of middleware is called short-circuiting because it doesn&#8217;t always forward the request to the next component in the chain.  For example if your application doesn&#8217;t allow Chrome users, the middleware can check the client agent and if it is Chrome, a response with an error message is created.

<div id="attachment_1629" style="width: 634px" class="wp-caption aligncenter">
  <a href="/wp-content/uploads/2019/04/The-short-circuiting-middleware.jpg"><img aria-describedby="caption-attachment-1629" loading="lazy" class="size-full wp-image-1629" src="/wp-content/uploads/2019/04/The-short-circuiting-middleware.jpg" alt="The short-circuiting middleware" width="624" height="335" /></a>
  
  <p id="caption-attachment-1629" class="wp-caption-text">
    The short-circuiting middleware
  </p>
</div>

It is important to note that middlewares are called in the same order as they are registered in the Startup class. The middleware which is registered first will handle the request first. Short-circuiting middlewares should always be placed at the front of the chain.

<div id="attachment_1630" style="width: 481px" class="wp-caption aligncenter">
  <a href="/wp-content/uploads/2019/04/The-order-of-the-middlewares-is-important.jpg"><img aria-describedby="caption-attachment-1630" loading="lazy" class="size-full wp-image-1630" src="/wp-content/uploads/2019/04/The-order-of-the-middlewares-is-important.jpg" alt="The order of the middlewares is important" width="471" height="98" /></a>
  
  <p id="caption-attachment-1630" class="wp-caption-text">
    The order of the middlewares is important
  </p>
</div>

If you make a request from Chrome (also from Edge since it is using Chromium now), you will see the access denied message.

<div id="attachment_1631" style="width: 710px" class="wp-caption aligncenter">
  <a href="/wp-content/uploads/2019/04/Calls-with-Chrome-are-blocked.jpg"><img aria-describedby="caption-attachment-1631" loading="lazy" class="wp-image-1631" src="/wp-content/uploads/2019/04/Calls-with-Chrome-are-blocked.jpg" alt="Calls with Chrome are blocked" width="700" height="385" /></a>
  
  <p id="caption-attachment-1631" class="wp-caption-text">
    Calls with Chrome are blocked
  </p>
</div>

Calls with a different browser, for example, Firefox still create a response.

<div id="attachment_1632" style="width: 611px" class="wp-caption aligncenter">
  <a href="/wp-content/uploads/2019/04/Calls-with-Firefox-are-allowed.jpg"><img aria-describedby="caption-attachment-1632" loading="lazy" class="size-full wp-image-1632" src="/wp-content/uploads/2019/04/Calls-with-Firefox-are-allowed.jpg" alt="Calls with Firefox are allowed" width="601" height="119" /></a>
  
  <p id="caption-attachment-1632" class="wp-caption-text">
    Calls with Firefox are allowed
  </p>
</div>

## <span style="color: #000000;">Creating a Request-Editing Middleware</span>

A request-editing middleware changes the requests before it reaches other components but doesn&#8217;t create a response. This can be used to prepare the request for easier processing later or for enriching the request with platform-specific features. The following example shows a middleware which sets a context item if the browser used is Edge.

<div id="attachment_1633" style="width: 710px" class="wp-caption aligncenter">
  <a href="/wp-content/uploads/2019/04/The-request-editing-middleware.jpg"><img aria-describedby="caption-attachment-1633" loading="lazy" class="wp-image-1633" src="/wp-content/uploads/2019/04/The-request-editing-middleware.jpg" alt="The request-editing middleware" width="700" height="250" /></a>
  
  <p id="caption-attachment-1633" class="wp-caption-text">
    The request-editing middleware
  </p>
</div>

## Creating a Response-Editing Middleware

Since there is a request-editing middleware, it won&#8217;t be surprising that there is also a response-editing middleware which changes the response before it is sent to the user. This type of middleware is often used for logging or handling errors.

<div id="attachment_1634" style="width: 710px" class="wp-caption aligncenter">
  <a href="/wp-content/uploads/2019/04/The-response-editing-middleware.jpg"><img aria-describedby="caption-attachment-1634" loading="lazy" class="wp-image-1634" src="/wp-content/uploads/2019/04/The-response-editing-middleware.jpg" alt="The response-editing middleware" width="700" height="276" /></a>
  
  <p id="caption-attachment-1634" class="wp-caption-text">
    The response-editing middleware
  </p>
</div>

Now register the middleware in your Startup class. It may not be intuitive but it is important that a response-editing middleware is registered first because the response passes all middlewares in the reverse order of the request. This means that the first middleware processes the request first and the response last.

<div id="attachment_1635" style="width: 481px" class="wp-caption aligncenter">
  <a href="/wp-content/uploads/2019/04/Register-the-response-editing-middleware-first.jpg"><img aria-describedby="caption-attachment-1635" loading="lazy" class="size-full wp-image-1635" src="/wp-content/uploads/2019/04/Register-the-response-editing-middleware-first.jpg" alt="Register the response-editing middleware first" width="471" height="147" /></a>
  
  <p id="caption-attachment-1635" class="wp-caption-text">
    Register the response-editing middleware first
  </p>
</div>

If you start your application and enter an URL which your system doesn&#8217;t know (and don&#8217;t use Chrome), you will see your custom error text.

<div id="attachment_1636" style="width: 710px" class="wp-caption aligncenter">
  <a href="/wp-content/uploads/2019/04/Response-created-by-the-middleware.jpg"><img aria-describedby="caption-attachment-1636" loading="lazy" class="wp-image-1636" src="/wp-content/uploads/2019/04/Response-created-by-the-middleware.jpg" alt="Response created by the middleware" width="700" height="101" /></a>
  
  <p id="caption-attachment-1636" class="wp-caption-text">
    Response created by the middleware
  </p>
</div>

## Conclusion

In this post, I talked about middlewares. I explained what they are and what different types there are.

For more details about complex configurations, I highly recommend the book &#8220;<a href="https://www.amazon.com/Pro-ASP-NET-Core-MVC-2/dp/148423149X" target="_blank" rel="noopener noreferrer">Pro ASP.NET Core MVC 2</a>&#8220;. You can find the source code of the demo on <a href="https://github.com/WolfgangOfner/MVC-Core-Middleware" target="_blank" rel="noopener noreferrer">GitHub</a>.