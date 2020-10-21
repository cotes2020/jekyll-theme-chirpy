---
title: Controllers and Actions
date: 2018-01-15T16:52:28+01:00
author: Wolfgang Ofner
categories: [ASP.NET]
tags: [Action, ActionResult, Controller, MVC, Redirect]
---
<span class="fontstyle0">In this post, I want to talk about how controllers and actions interact with each other and present several built-in functions.  The MVC Framework is endlessly customizable and extensible. As a result, it is possible to implement your own controller which I will shortly talk about and show how to do that.</span>

## Setting up the project

I created a new ASP.NET MVC project with the empty template and add folders and core references for MVC.

<div id="attachment_601" style="width: 510px" class="wp-caption aligncenter">
  <a href="/wp-content/uploads/2018/01/Set-up-project.jpg"><img aria-describedby="caption-attachment-601" loading="lazy" class="wp-image-601" title="Set up the ASP.NET MVC project for Actions and Controller" src="/wp-content/uploads/2018/01/Set-up-project.jpg" alt="Set up the ASP.NET MVC project for Actions and Controller" width="500" height="326" /></a>
  
  <p id="caption-attachment-601" class="wp-caption-text">
    Set up the ASP.NET MVC project
  </p>
</div>

Throughout this post, I will only use the default route. This post describes the basics of controllers and actions but I will not explain every single simple step. Therefore it is expected that you know at least how to create a controller and how to call its Index action.

## Passing data to an action

Following I will show several options, how to pass data to the action. First I start with the simplest way to call an action though. The easiest way to call an action is by entering an URL which is mapped to a route and redirected to the fitting action.

<div id="attachment_602" style="width: 325px" class="wp-caption aligncenter">
  <a href="/wp-content/uploads/2018/01/Simple-Action.jpg"><img aria-describedby="caption-attachment-602" loading="lazy" class="size-full wp-image-602" src="/wp-content/uploads/2018/01/Simple-Action.jpg" alt="Simple Action" width="315" height="135" /></a>
  
  <p id="caption-attachment-602" class="wp-caption-text">
    Simple Action
  </p>
</div>

The Index action does nothing except returning a view to the user&#8217;s browser. Therefore it does not take or process any data.

### Passing data to the parameter

A simple way to pass data to an action is by using a parameter for the action. The parameter will be added at the end of the URL.

<div id="attachment_603" style="width: 307px" class="wp-caption aligncenter">
  <a href="/wp-content/uploads/2018/01/Action-with-int-parameter.jpg"><img aria-describedby="caption-attachment-603" loading="lazy" class="size-full wp-image-603" src="/wp-content/uploads/2018/01/Action-with-int-parameter.jpg" alt="Action with int parameter" width="297" height="96" /></a>
  
  <p id="caption-attachment-603" class="wp-caption-text">
    Action with int parameter
  </p>
</div>

As you can see, I used int as the data type for the parameter. If the user enters a parameter which can not be converted to an int, ASP.NET MVC throws an exception and displays the yellow screen of death. To prevent this exception, it is possible to set a default parameter. If the framework can&#8217;t convert the parameter provided by the user, the default parameter is used.

Using a parameter to pass data to a method makes it also way easier to unit test.

It is also interesting to note that parameter can&#8217;t have out or ref parameter. It wouldn&#8217;t make any sense though and therefore ASP.NET MVC would throw an exception.

### Passing multiple parameters to the action

There are two options on how to pass multiple parameters to an action. The first option is to use the catchall variable in the route. To do that, you have to configure the route first, as shown in the following screenshot.

<div id="attachment_604" style="width: 583px" class="wp-caption aligncenter">
  <a href="/wp-content/uploads/2018/01/Catchall-route.jpg"><img aria-describedby="caption-attachment-604" loading="lazy" class="size-full wp-image-604" src="/wp-content/uploads/2018/01/Catchall-route.jpg" alt="Catchall route" width="573" height="105"  /></a>
  
  <p id="caption-attachment-604" class="wp-caption-text">
    Catchall route
  </p>
</div>

After setting up the route, you can use the catchall variable as the parameter and add as many variables as you want to it. The variables are all passed as one string and have to be parsed. The following screenshot shows one id and three variables in the catchall are passed to the method.

<div id="attachment_605" style="width: 723px" class="wp-caption aligncenter">
  <a href="/wp-content/uploads/2018/01/Catchall-action-1.jpg"><img aria-describedby="caption-attachment-605" loading="lazy" class="size-full wp-image-605" src="/wp-content/uploads/2018/01/Catchall-action-1.jpg" alt="Catchall action" width="713" height="119"  /></a>
  
  <p id="caption-attachment-605" class="wp-caption-text">
    Catchall action
  </p>
</div>

The second way is to use multiple parameters. This method is used to pass data from a form to the action.

<div id="attachment_606" style="width: 556px" class="wp-caption aligncenter">
  <a href="/wp-content/uploads/2018/01/Action-with-multiple-parameter.jpg"><img aria-describedby="caption-attachment-606" loading="lazy" class="size-full wp-image-606" src="/wp-content/uploads/2018/01/Action-with-multiple-parameter.jpg" alt="Action with multiple parameter" width="546" height="116"  /></a>
  
  <p id="caption-attachment-606" class="wp-caption-text">
    Action with multiple parameter
  </p>
</div>

### Retrieving data within an action

Besides parameter, there are several other sources from where you can receive data. These data sources can be:

  * Request
  * User
  * Server
  * HttpContext
  * RouteData

I would suggest to type these classes into your action and look through the options provided by the intellisense. On the following screenshot, I show some of these options.

<div id="attachment_607" style="width: 392px" class="wp-caption aligncenter">
  <a href="/wp-content/uploads/2018/01/Action-with-different-sources-to-get-some-data.jpg"><img aria-describedby="caption-attachment-607" loading="lazy" class="size-full wp-image-607" src="/wp-content/uploads/2018/01/Action-with-different-sources-to-get-some-data.jpg" alt="Action with different sources to get some data" width="382" height="315" /></a>
  
  <p id="caption-attachment-607" class="wp-caption-text">
    Action with different sources to get some data
  </p>
</div>

## ActionResult types

Action results help to make the code easier and cleaner and also make unit testing easier. The ActionResult class is an abstract class which can be used. But there are also a whole bunch of derived classes which can be used to increase the readability of the action. If you are familiar with the command patter, you will recognize it here. ActionResults pass around objects that describe the operations which are performed.

As I just mentioned, there is a whole list of action results. Following I will present some of them.

### Returning a view

Returning a view is the easiest way to present an HTML page to the user. Using ActionResult as return type works and compiles. Though I prefer using ViewResult for views because it tells me faster what the actual return type is. There are different overloaded versions of View().

#### View without parameter

The simplest is without a parameter. When using this version, the MVC framework searches for a view with the same name as the action.

<div id="attachment_609" style="width: 225px" class="wp-caption aligncenter">
  <a href="/wp-content/uploads/2018/01/View-without-parameter.jpg"><img aria-describedby="caption-attachment-609" loading="lazy" class="size-full wp-image-609" src="/wp-content/uploads/2018/01/View-without-parameter.jpg" alt="View without parameter" width="215" height="91" /></a>
  
  <p id="caption-attachment-609" class="wp-caption-text">
    View without parameter
  </p>
</div>

The framework searches in the following locations:

  * <span class="fontstyle2">/Areas/</span><span class="fontstyle4"><AreaName></span><span class="fontstyle2">/Views</span><span class="fontstyle4">/<ControllerName></span><span class="fontstyle2">/</span><span class="fontstyle4"><ViewName></span><span class="fontstyle2">.aspx</span>
  * /Views/<ControllerName>/<ViewName>.aspx
  * <span class="fontstyle2">/</span><span class="fontstyle2">Areas/</span><span class="fontstyle4"><AreaName></span><span class="fontstyle2">/Views/</span><span class="fontstyle4"><ControllerName></span><span class="fontstyle2">/</span><span class="fontstyle4"><ViewName></span><span class="fontstyle2">.ascx</span>
  * <span class="fontstyle2">/Views/<ControllerName>/<ViewName>.ascx</span>
  * <span class="fontstyle2">/Areas/</span><span class="fontstyle4"><AreaName></span><span class="fontstyle2">/Views/Shared/</span><span class="fontstyle4"><ViewName></span><span class="fontstyle2">.aspx</span>
  * <span class="fontstyle2">/Views/Shared/<span class="fontstyle4"><ViewName></span>.aspx</span>
  * <span class="fontstyle2">/</span><span class="fontstyle2">Areas/</span><span class="fontstyle4"><AreaName></span><span class="fontstyle2">/Views/Shared/</span><span class="fontstyle4"><ViewName></span><span class="fontstyle2">.ascx</span>
  * <span class="fontstyle2">/Views/Shared/<span class="fontstyle4"><ViewName></span>.ascx</span>
  * <span class="fontstyle2">/Areas/</span><span class="fontstyle4"><AreaName></span><span class="fontstyle2">/Views/</span><span class="fontstyle4"><ControllerName></span><span class="fontstyle2">/</span><span class="fontstyle4"><ViewName></span><span class="fontstyle2">.cshtml</span>
  * <span class="fontstyle2">/Views/<span class="fontstyle4"><ControllerName></span>/<span class="fontstyle4"><ViewName></span>.cshtml</span>
  * <span class="fontstyle2">/Areas/</span><span class="fontstyle4"><AreaName></span><span class="fontstyle2">/Views/</span><span class="fontstyle4"><ControllerName></span><span class="fontstyle2">/</span><span class="fontstyle4"><ViewName></span><span class="fontstyle2">.vbhtml</span>
  * <span class="fontstyle2">/Views/<span class="fontstyle4"><ControllerName></span>/<span class="fontstyle4"><ViewName></span>.vbhtml</span>
  * <span class="fontstyle2">/Areas/</span><span class="fontstyle4"><AreaName></span><span class="fontstyle2">/Views/Shared/<ViewName>.cshtml</span>
  * <span class="fontstyle2">/Views/Shared/<span class="fontstyle4"><ViewName></span>.cshtml</span>
  * <span class="fontstyle2">/Areas/</span><span class="fontstyle4"><AreaName></span><span class="fontstyle2">/Views/Shared/</span><span class="fontstyle4"><ViewName></span><span class="fontstyle2">.vbhtm</span>
  * <span class="fontstyle2">/Views/Shared/<span class="fontstyle4"><ViewName></span>.vbhtml</span>

The MVC framework also searches for ASPX views, C# and Visual Basic .NET Razor templates. This behavior guarantees backward compatibility with earlier versions of the MVC framework

<span class="fontstyle0">If the file is found in one of the locations, it is returned and the search stops. If no file is found, a yellow screen of death is shown with some information which locations where searched.</span>

#### View with name parameter

By default, the framework searches for a view with the same name as the action. If you want to return a view from the same controller but with a different name, you have to pass the name as the parameter in the view.

<div id="attachment_608" style="width: 236px" class="wp-caption aligncenter">
  <a href="/wp-content/uploads/2018/01/View-with-view-name.jpg"><img aria-describedby="caption-attachment-608" loading="lazy" class="size-full wp-image-608" src="/wp-content/uploads/2018/01/View-with-view-name.jpg" alt="View with view name" width="226" height="97" /></a>
  
  <p id="caption-attachment-608" class="wp-caption-text">
    View with view name
  </p>
</div>

#### View with path

Sometimes the view you want to return is in a different directory. Then you can pass the path to the view as the parameter. Note: Don&#8217;t forget the tilde at the beginning of the path. Otherwise, you won&#8217;t get the expected view.

<div id="attachment_610" style="width: 385px" class="wp-caption aligncenter">
  <a href="/wp-content/uploads/2018/01/View-with-path.jpg"><img aria-describedby="caption-attachment-610" loading="lazy" class="size-full wp-image-610" src="/wp-content/uploads/2018/01/View-with-path.jpg" alt="View with path" width="375" height="93" /></a>
  
  <p id="caption-attachment-610" class="wp-caption-text">
    View with path
  </p>
</div>

It is very uncommon to use this feature and it is a hint that your design might not be optimal.

#### View with a view model

All versions of the view which I showed above also have the option to pass a view model. To pass the view model to the view, simply add it as a parameter. On the following screenshot, I pass the path to the view and the view model (which in this case is just a string).

<div id="attachment_611" style="width: 513px" class="wp-caption aligncenter">
  <a href="/wp-content/uploads/2018/01/View-with-path-and-view-model.jpg"><img aria-describedby="caption-attachment-611" loading="lazy" class="size-full wp-image-611" src="/wp-content/uploads/2018/01/View-with-path-and-view-model.jpg" alt="View with path and view model" width="503" height="132" /></a>
  
  <p id="caption-attachment-611" class="wp-caption-text">
    View with path and view model
  </p>
</div>

When you use only a view model which is a string, the ASP.NET MVC framework thinks that it is the name of the view you want to return. To tell the framework that it is a model, you have to prefix the model with the model: keyword.

<div id="attachment_613" style="width: 369px" class="wp-caption aligncenter">
  <a href="/wp-content/uploads/2018/01/View-with-model-keyword.jpg"><img aria-describedby="caption-attachment-613" loading="lazy" class="size-full wp-image-613" src="/wp-content/uploads/2018/01/View-with-model-keyword.jpg" alt="View with model keyword" width="359" height="93" /></a>
  
  <p id="caption-attachment-613" class="wp-caption-text">
    View with model keyword
  </p>
</div>

### Passing data with a view model

Previously I showed how to pass a model to the view. Now there are two different ways to use this view model in the view.  Either you cast the model to its data type and then use it inline, for example, @(((DateTime)Model).ToLocalTime()) or you can declare the model and its data type and then use it like a normal C# variable.

<div id="attachment_612" style="width: 410px" class="wp-caption aligncenter">
  <a href="/wp-content/uploads/2018/01/View-with-view-model.jpg"><img aria-describedby="caption-attachment-612" loading="lazy" class="size-full wp-image-612" src="/wp-content/uploads/2018/01/View-with-view-model.jpg" alt="View with view model" width="400" height="253" /></a>
  
  <p id="caption-attachment-612" class="wp-caption-text">
    View with view model
  </p>
</div>

Casting the object works but it makes the view pretty messy. Therefore I recommend using strongly typed views. It is important to note that when declaring the model it starts with a lower case and when using it, it starts with an upper case.

### Passing data with view bag

Another way to pass data to a view is with the view bag. The difference between the view bag and the data model is that the view bag is a dynamic object. The view bag doesn&#8217;t have to be passed to the view but can be used in the view like the view model.

<div id="attachment_615" style="width: 290px" class="wp-caption aligncenter">
  <a href="/wp-content/uploads/2018/01/Adding-data-into-the-view-bag.jpg"><img aria-describedby="caption-attachment-615" loading="lazy" class="size-full wp-image-615" src="/wp-content/uploads/2018/01/Adding-data-into-the-view-bag.jpg" alt="Adding data into the view bag" width="280" height="143" /></a>
  
  <p id="caption-attachment-615" class="wp-caption-text">
    Adding data into the view bag
  </p>
</div>

Due to its dynamic behavior, I don&#8217;t have to declare the view bag variable before using it. I can simply assign my data to it.

<div id="attachment_614" style="width: 389px" class="wp-caption aligncenter">
  <a href="/wp-content/uploads/2018/01/View-with-view-bag.jpg"><img aria-describedby="caption-attachment-614" loading="lazy" class="size-full wp-image-614" src="/wp-content/uploads/2018/01/View-with-view-bag.jpg" alt="View with view bag" width="379" height="112" /></a>
  
  <p id="caption-attachment-614" class="wp-caption-text">
    View with view bag
  </p>
</div>

### Passing data with temp data

Temp data works like the view bag with one difference. After reading the data from temp data, the values get marked for deletion. The usage is similar to the usage of a session.

<div id="attachment_617" style="width: 436px" class="wp-caption aligncenter">
  <a href="/wp-content/uploads/2018/01/Assigning-values-to-temp-data.jpg"><img aria-describedby="caption-attachment-617" loading="lazy" class="size-full wp-image-617" src="/wp-content/uploads/2018/01/Assigning-values-to-temp-data.jpg" alt="Assigning values to temp data" width="426" height="92" /></a>
  
  <p id="caption-attachment-617" class="wp-caption-text">
    Assigning values to temp data
  </p>
</div>

If the value is a primitive data type, it can be used like the session. If it is an object, you have to cast it.

<div id="attachment_618" style="width: 478px" class="wp-caption aligncenter">
  <a href="/wp-content/uploads/2018/01/Using-temp-data-in-a-view.jpg"><img aria-describedby="caption-attachment-618" loading="lazy" class="size-full wp-image-618" src="/wp-content/uploads/2018/01/Using-temp-data-in-a-view.jpg" alt="Using temp data in a view" width="468" height="79" /></a>
  
  <p id="caption-attachment-618" class="wp-caption-text">
    Using temp data in a view
  </p>
</div>

#### Keeping data

There are three different ways to store the values from temp data:

  * Assign it to the view bag
  * Assign it to a variable
  * Use Peek() or Keep()

Peek returns the value without marking it for deletion. Keep marks the value to keep it after it was accessed. When the value is accessed a second time, it gets marked for removal if you don&#8217;t use Keep again.

<div id="attachment_619" style="width: 355px" class="wp-caption aligncenter">
  <a href="/wp-content/uploads/2018/01/Ways-to-store-data-from-temp-data.jpg"><img aria-describedby="caption-attachment-619" loading="lazy" class="size-full wp-image-619" src="/wp-content/uploads/2018/01/Ways-to-store-data-from-temp-data.jpg" alt="Ways to store data from temp data" width="345" height="361" /></a>
  
  <p id="caption-attachment-619" class="wp-caption-text">
    Ways to store data from temp data
  </p>
</div>

#### Returning an HTTP status code

ASP.NET MVC hat to built-in classes for returning an HTTP status code

  * HttpNotFoundResult
  * HttpUnauthorizedResult

Additionally, you can return every HTTP status code you want with HttpStatusCodeResult. Pass the status code and a message as parameter. HttpNotFoundResult has a short form, HttpNotFound. The return value for a status code is HttpStatusCodeResult.

<div id="attachment_626" style="width: 459px" class="wp-caption aligncenter">
  <a href="/wp-content/uploads/2018/01/Working-with-HttpStatusCode.jpg"><img aria-describedby="caption-attachment-626" loading="lazy" class="size-full wp-image-626" src="/wp-content/uploads/2018/01/Working-with-HttpStatusCode.jpg" alt="Working with HttpStatusCode" width="449" height="241" /></a>
  
  <p id="caption-attachment-626" class="wp-caption-text">
    Working with HttpStatusCode
  </p>
</div>

### More results

Additionally, to the action results, I just presented there are some more like:

  * PartialViewResult
  * FileResult
  * JsonResult
  * JavaScriptResult
  * EmptyResult

The names are speaking for themselves. Therefore I won&#8217;t go into more detail about them.

## Performing redirects

A common performance of an action is to redirect the user to another URL. Often this URL is another action which generates an output for the user.

#### The Post / Redirect / Get pattern

The <a href="https://en.wikipedia.org/wiki/Post/Redirect/Get" target="_blank" rel="noopener">Post / Redirect / Get pattern</a> helps to avoid the problem of resubmitting a form for the second time which could cause unexpected results. This problem can occur when you return HTML after processing a POST and the user clicks the reload button of his browser which results in resubmitting his form.

With the Post / Redirect / get pattern the POST request is processed and then the browser gets redirected to another URL. The redirect generates a GET request. Since GET requests don&#8217;t (shouldn&#8217;t) modify the state of the application, any resubmission of the request won&#8217;t cause any problems.

#### Temporary and permanent redirects

There are two different HTTP status codes for redirects:

  * 301 for permanent redirection
  * 302 for temporary redirection

The HTTP code 302 is often used, especially when using the Post / Redirect / Get pattern. Be careful when using this cause because it instructs the recipient to never again use the requested URL and use the new URL instead. If you don&#8217;t know which code you should use, use temporary redirects.

ASP.NET MVC offers for every redirect a temporary and permanent implementation. For example, you can use Redirect() for a temporary redirect or RedirectPermanent() for a permanent redirection.

#### <span class="fontstyle0">Redirect to a view</span>

If you want to redirect to a view, use the Redirect or RedirectPermanent method with the folder and index name as a string parameter. You don&#8217;t have to provide the file ending of the view. The return value of this is RedirectResult.

<div id="attachment_621" style="width: 378px" class="wp-caption aligncenter">
  <a href="/wp-content/uploads/2018/01/Redirect-to-view.jpg"><img aria-describedby="caption-attachment-621" loading="lazy" class="size-full wp-image-621" src="/wp-content/uploads/2018/01/Redirect-to-view.jpg" alt="Redirect to view" width="368" height="181" /></a>
  
  <p id="caption-attachment-621" class="wp-caption-text">
    Redirect to view
  </p>
</div>

#### Redirect to a route

To redirect from an action to a route use RedirectToRoute or RedirectToRoutePermanent. The methods take an anonymous type with the route information. The return value is RedirectToRouteResult.

<div id="attachment_622" style="width: 446px" class="wp-caption aligncenter">
  <a href="/wp-content/uploads/2018/01/Redirect-to-route.jpg"><img aria-describedby="caption-attachment-622" loading="lazy" class="size-full wp-image-622" src="/wp-content/uploads/2018/01/Redirect-to-route.jpg" alt="Redirect to route" width="436" height="329" /></a>
  
  <p id="caption-attachment-622" class="wp-caption-text">
    Redirect to route
  </p>
</div>

#### Redirect to an action

More elegantly than redirecting to a route is redirecting to an action. RedirectToAction and RedirectToActionPermanent are wrapper for RedirectToRoute and RedirectToRoutePermanent but make the code cleaner in my opinion. The values provided as a parameter for the action and controller are not verified at compile time. This means that you are responsible for making sure that your target exists.

&nbsp;

<div id="attachment_625" style="width: 710px" class="wp-caption aligncenter">
  <a href="/wp-content/uploads/2018/01/Redirect-to-action.jpg"><img aria-describedby="caption-attachment-625" loading="lazy" class="wp-image-625" src="/wp-content/uploads/2018/01/Redirect-to-action.jpg" alt="Redirect to action" width="700" height="239" /></a>
  
  <p id="caption-attachment-625" class="wp-caption-text">
    Redirect to action
  </p>
</div>

## Conclusion

In this post, I showed how to pass data into an action and how to retrieve data within an action from different sources. Then I presented some of the ActionResult types which are built-in into the MVC framework and also how to pass data between actions using ViewBag and TempData.. In the last part, I talked about the different ways how to perform redirects.

For more details about how controller and actions work, I highly recommend the books <a href="http://amzn.to/2mgRbTy" target="_blank" rel="noopener">Pro ASP.NET MVC 5</a> and <a href="http://amzn.to/2mfQ0nA" target="_blank" rel="noopener">Pro ASP.NET MVC 5 Plattform</a>.

I uploaded the source code to <a href="https://github.com/WolfgangOfner/MVC-ActionAndController" target="_blank" rel="noopener">GitHub</a>.