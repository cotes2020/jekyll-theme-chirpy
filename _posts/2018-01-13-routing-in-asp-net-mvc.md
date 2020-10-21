---
title: Routing in ASP.NET MVC
date: 2018-01-13T15:34:13+01:00
author: Wolfgang Ofner
categories: [ASP.NET]
tags: [MVC, Routing]
---
Routing in ASP.NET MVC can be a really complex topic. In this post, I want to present the most used URL Patterns and explain how they work.

## Routing with Routes

Routes can be created by adding them to the RouteCollection or by decorating actions or controller with attributes. In this section, I will show different approaches how to build a route and adding it to the RouteCollection. Adding a route happens in the RouteConfig class in the App_Start folder at the start of the application.

### Default route

The ASP.NET MVC framework comes out of the box with a default route. The template also displays the property names of the route attributes, so it is easier for a beginner to understand what&#8217;s going on. Let&#8217;s have a look at the default route:

[<img loading="lazy" class="aligncenter size-full wp-image-536" src="/wp-content/uploads/2018/01/1-Default-Route-implemented-by-MVC.jpg" alt="Default Route implemented by MVC" width="658" height="186" />](/wp-content/uploads/2018/01/1-Default-Route-implemented-by-MVC.jpg)

Every route has a unique name. The name of the default route is Default. The url attribute describes the pattern of the url. The default pattern is Controller/Action/Id. The defaults property sets default properties for the controller, action and sets the id as optional. The default values are used when no values for the attribute is passed. Valid URLs for this route are for example:

  * /
  * /Home
  * /Admin
  * /Home/Index
  * /Home/Index/123abc
  * /Home/Index/abc

The names of the properties (name, url and defaults) are grayed out because they are not needed. They are there only for an easier understanding, especially for beginners.

Additionally to the default route, the ASP.NET template implements routes.IgnoreRoute(&#8220;{resource}.axd/{*pathInfo}&#8221;);. You should keep this rule to prevent ASP.NET MVC from trying to handle .axd files. These files don&#8217;t exist physically and therefore are handled by an HttpHandler.

### Simple route

Now it&#8217;s time to implement our own routes. The simplest route takes a controller and an action with no defaults or additional parameters.

[<img loading="lazy" class="aligncenter size-full wp-image-538" src="/wp-content/uploads/2018/01/2-Simple-Route-Controller-Action.jpg" alt="Simple Route Controller - Action" width="458" height="40" />](/wp-content/uploads/2018/01/2-Simple-Route-Controller-Action.jpg)

If the user types into his browser myurl.com/Home/Index the Index action in the Home controller is called. If the user only enters /Home the route won&#8217;t find a suiting action because no default action is defined.

### Static route segments

ASP.NET MVC also offers the possibility of static route segments. This means that if the route contains a certain word that a specific controller and/or action are called.

[<img loading="lazy" class="aligncenter size-full wp-image-539" src="/wp-content/uploads/2018/01/3-Fixed-and-variable-parts.jpg" alt="Fixed and variable parts" width="525" height="149" />](/wp-content/uploads/2018/01/3-Fixed-and-variable-parts.jpg)

The screenshot above shows three different variations of a static segment in the route. The first route calls the ShowArchievePosts action in the Posts controller when the user enters /Blog/Archive. The second route calls an action entered in the Posts controller when the user enters/Blog/ActionName.

The third route is selected when the user input starts with /InternalBlog. If the user doesn&#8217;t enter anything else the default controller and action are called. The user can also enter a controller or a controller and an action.

Another possibility to add a static part to a route is to prefix the controller or action as part of its name.

[<img loading="lazy" class="aligncenter size-full wp-image-541" src="/wp-content/uploads/2018/01/4-prefix-controller.jpg" alt="Prefix controller" width="395" height="37" />](/wp-content/uploads/2018/01/4-prefix-controller.jpg)

This route is matched when the controller begins with External. This means that the URL /ExternalHome/Index would call the Index action in the Home controller.

### <span class="fontstyle0">Route ordering</span>

Now it gets a bit trickier. The routes are added to the RouteCollection as they appear in the RegisterRoutes method. After the user entered a URL, ASP.NET searches through the RouteCollection until it finds a fitting route. A fitting route does not mean that it leads to the result which the user expects.

[<img loading="lazy" class="aligncenter size-full wp-image-540" src="/wp-content/uploads/2018/01/4.5-Route-order.jpg" alt="Route order" width="630" height="141" />](/wp-content/uploads/2018/01/4.5-Route-order.jpg)

Let&#8217;s take a look at the two roots from above. The first route is the default route with a default controller and action and the second route has the static segment InternalBlog in front of the controller. What happens if the user enters &#8220;/InternalBlog/Posts/Display&#8221;? You might think that the second route is selected. But that&#8217;s not the case. The entered URL fits the first route where InternalBlog = controller, Posts = action, Display = id. If the application doesn&#8217;t have an InternalBlog controller with the Posts action, an error message is displayed.

[<img loading="lazy" class="aligncenter size-full wp-image-546" src="/wp-content/uploads/2018/01/4.6-Route-not-found-error.jpg" alt="Route not found error" width="1234" height="270" />](/wp-content/uploads/2018/01/4.6-Route-not-found-error.jpg)

### Custom segment variables

The default route already showed that it is possible to add a variable after the action. For example, the route /Home/Index/123 call the Index action from the HomeController with the parameter 123. The parameter name of the action must match the variable name in the route. Otherwise, it will be null.

[<img loading="lazy" class="aligncenter size-full wp-image-547" src="/wp-content/uploads/2018/01/Default-Route.jpg" alt="Default Route" width="627" height="93" />](/wp-content/uploads/2018/01/Default-Route.jpg)

[<img loading="lazy" class="aligncenter size-full wp-image-543" src="/wp-content/uploads/2018/01/5.5-Custom-segment-controller-and-action.jpg" alt="Custom segment controller and action" width="312" height="140" />](/wp-content/uploads/2018/01/5.5-Custom-segment-controller-and-action.jpg)

The default route sets the id as UrlParameter.Optional which means that this parameter is optional (what a surprise).

### Variable amount of segment variables

Like params in C#, the routing in ASP.NET MVC offers a feature to take a variable amount of variables. To achieve that use the *catchall keyword.

[<img loading="lazy" class="aligncenter size-full wp-image-548" src="/wp-content/uploads/2018/01/Catchall.jpg" alt="Catchall" width="509" height="116" />](/wp-content/uploads/2018/01/Catchall.jpg)

This allows the user to enable any amount of variables into the URL. A fitting URL would be for example /Home/Index/User/Detail/123. User/Detail/123 would be passed as catchall parameter to the Index action in the HomeController.

[<img loading="lazy" class="aligncenter size-full wp-image-549" src="/wp-content/uploads/2018/01/Catchall-action.jpg" alt="Catchall action" width="315" height="137" />](/wp-content/uploads/2018/01/Catchall-action.jpg)

The catchall string contains User/Detail/123.

### Default values for attributes

I already showed that it is possible to set default values for controller, actions and attributes in the route. It is also possible to set default values for attributes in the action.  This is done as in normal C# with variable = defaultValue, for example string id = &#8220;1&#8221;.

### [<img loading="lazy" class="aligncenter size-full wp-image-550" src="/wp-content/uploads/2018/01/Default-attribute-in-action.jpg" alt="Default attribute in action" width="291" height="146" />](/wp-content/uploads/2018/01/Default-attribute-in-action.jpg)

### Variable constraints

The routing in ASP.NET MVC enables you to restrict the data type and the range of the entered attributes. To restrict a variable to int for example use, variable = new IntRouteConstraint(). There are several classes like FloatRouteConstraint() or AlphaRouteConstraint().

To restrict a variable to a certain range, use the RangeRouteConstraint class, for example variable = new RangeRouteConstraint(min, max).

[<img loading="lazy" class="aligncenter size-full wp-image-558" src="/wp-content/uploads/2018/01/Route-variable-constraints.jpg" alt="Route variable constraints" width="426" height="202" />](/wp-content/uploads/2018/01/Route-variable-constraints.jpg)

### Restrict HTTP method

Not only controller, actions and variables can be restricted, also the HTTP method can be restricted. To restrict a route to a certain HTTP method use httpMethod = new HttPMethodConstraint(&#8220;GET&#8221;). Instead of get, you could use any HTTP verb.

[<img loading="lazy" class="aligncenter size-full wp-image-559" src="/wp-content/uploads/2018/01/HTTP-Method-constraint.jpg" alt="HTTP Method constraint" width="463" height="105" />](/wp-content/uploads/2018/01/HTTP-Method-constraint.jpg)

## Attribute Routing in ASP.NET MVC

Additionally, to creating routes it is possible to decorate controller and actions with route attributes.

### Enabling attribute routing

To enable attribute routing you have to add routes.MapMvcAttributeRoutes(); to the RegisterRoutes method. Next, you have to set the route attribute on an action and the desired route, for example [Route(&#8220;MyRoute&#8221;]). Now you can call your action with /MyRoute

[<img loading="lazy" class="aligncenter size-full wp-image-553" src="/wp-content/uploads/2018/01/Enable-attribute-routing.jpg" alt="Enable attribute routing" width="416" height="73" />](/wp-content/uploads/2018/01/Enable-attribute-routing.jpg)

[<img loading="lazy" class="aligncenter size-full wp-image-551" src="/wp-content/uploads/2018/01/Attribute-routing-action.jpg" alt="Attribute routing action" width="247" height="154" />](/wp-content/uploads/2018/01/Attribute-routing-action.jpg)

[<img loading="lazy" class="aligncenter size-full wp-image-552" src="/wp-content/uploads/2018/01/Attribute-routing-result.jpg" alt="Attribute routing result" width="298" height="127" />](/wp-content/uploads/2018/01/Attribute-routing-result.jpg)

It is also possible to decorate an action with several route attributes.

### Attribute routing with variables

With attribute routes, it is also possible to add variables which can be processed in the action as parameters. To declare a variable wrap it in curly brackets. The name in the route must match the name of the parameter, otherwise, the parameter will be null.

[<img loading="lazy" class="aligncenter size-full wp-image-555" src="/wp-content/uploads/2018/01/Attribute-routing-with-variables.jpg" alt="Attribute routing with variables" width="364" height="115" />](/wp-content/uploads/2018/01/Attribute-routing-with-variables.jpg)

### Constraints for route attributes

The variables in the route attribute can be restricted to a certain data type. This would be useful for the id. Ids are usually int, so it makes sense to expect an int id. To do that you only have to add the data type after the variable name within the brackets, separated by a colon. Don&#8217;t forget to change the data type of the parameter, otherwise ASP.NET won&#8217;t match the variable with the parameter.

[<img loading="lazy" class="aligncenter size-full wp-image-556" src="/wp-content/uploads/2018/01/Attribute-routing-with-variables-constraint.jpg" alt="Attribute routing with variables constraint" width="400" height="113" />](/wp-content/uploads/2018/01/Attribute-routing-with-variables-constraint.jpg)

## Routing Static Files

If the user enters a path to a static file, for example, an image or a pdf file, the routing in ASP.NET MVC forwards the user to this file, if it exists.

[<img loading="lazy" class="aligncenter size-full wp-image-560" src="/wp-content/uploads/2018/01/Static-file.jpg" alt="Static file" width="371" height="94" />](/wp-content/uploads/2018/01/Static-file.jpg)

The mechanisms for routing in ASP.NET MVC searches first the path of the file. Only then it evaluates the routes. To prevent this behavior use set RouteExistingFiles to true.

[<img loading="lazy" class="aligncenter size-full wp-image-563" src="/wp-content/uploads/2018/01/Evaluate-routes-before-static-files.jpg" alt="Evaluate routes before static files" width="453" height="116" />](/wp-content/uploads/2018/01/Evaluate-routes-before-static-files.jpg)

It is also possible to ignore the routing for a certain file type with routes.IgnoreRoute(path).

[<img loading="lazy" class="aligncenter size-full wp-image-564" src="/wp-content/uploads/2018/01/Ignore-Route-for-all-html-files-in-staticcontent.jpg" alt="Ignore route for all html files in staticcontent" width="386" height="38" />](/wp-content/uploads/2018/01/Ignore-Route-for-all-html-files-in-staticcontent.jpg)

The example above shows that routing is ignored for all HTML files in the StaticContent folder. You can use {filename} as a variable for all file names in this directory.

## Conclusion

I showed several approaches for routing in ASP.NET MVC using the routing method and using attributes on actions and controllers. Routing is a pretty complex topic and there is way more to about it than what I presented. For more details about routing, I highly recommend the books <a href="http://amzn.to/2mgRbTy" target="_blank" rel="noopener noreferrer">Pro ASP.NET MVC 5</a> and <a href="http://amzn.to/2mfQ0nA" target="_blank" rel="noopener noreferrer">Pro ASP.NET MVC 5 Plattform</a>.

I uploaded the source code to <a href="https://github.com/WolfgangOfner/MVC-Routing" target="_blank" rel="noopener noreferrer">GitHub</a> if you want to download it and play a bit around with different routes.