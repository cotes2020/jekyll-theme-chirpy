---
title: Model Binding in ASP.NET MVC
date: 2018-01-22T22:53:33+01:00
author: Wolfgang Ofner
categories: [ASP.NET]
tags: [model binding, MVC]
---
ASP.NET MVC creates objects using the model binding process with the data which is sent by the browser in an HTTP request. The action method parameters are created through model binding from the data in the request.

## Setting up the project

I created a new ASP.NET MVC project with the empty template and add folders and core references for MVC.

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2018/01/Set-up-project.jpg"><img loading="lazy"title="Setting up the Model Binding ASP.NET MVC project" src="/assets/img/posts/2018/01/Set-up-project.jpg" alt="Setting up the Model Binding ASP.NET MVC project" /></a>
  
  <p>
    Setting up the ASP.NET MVC project
  </p>
</div>

Then I create a simple Home controller and a view to display some information about customers.

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2018/01/Home-controller-with-two-actions-for-working-with-customers.jpg"><img loading="lazy" src="/assets/img/posts/2018/01/Home-controller-with-two-actions-for-working-with-customers.jpg" alt="Home controller with two actions for working with customers" /></a>
  
  <p>
    Home controller with two actions for working with customers
  </p>
</div>

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2018/01/View-to-display-a-customer.jpg"><img loading="lazy" src="/assets/img/posts/2018/01/View-to-display-a-customer.jpg" alt="View to display a customer" /></a>
  
  <p>
    View to display a customer
  </p>
</div>

I also installed the Bootstrap NuGet to make the forms look a bit nicer.

## How Model Binding works

Model binding is a simple way to connect C# code with an HTTP request. Most MVC frameworks use some form of model binding to get the data from the request for an action. If you used MVC before, it is very likely that you already used model binding, even if you didn&#8217;t realize it.

Action invokers rely on model binders to bind the data from the request to the data of the C# code. When you use a parameter for an action, model binders generate these parameters before an action is invoked. This process starts after the request is received and is processed by the routing engine. Model binders are defined by the <a href="https://msdn.microsoft.com/en-us/library/system.web.mvc.imodelbinder(v=vs.118).aspx" target="_blank" rel="noopener noreferrer">IModelBinder interface</a>.

There can be several model binder in an MVC application and you can also create your own. In this post, I will only talk about the built-in binder, DefaultModelBinder.

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2018/01/Displaying-customer-with-id-1.jpg"><img aria-describedby="caption-attachment-722" loading="lazy" class="size-full wp-image-722" src="/assets/img/posts/2018/01/Displaying-customer-with-id-1.jpg" alt="Displaying customer with id 1" /></a>
  
  <p>
    Displaying customer with id 1
  </p>
</div>

### Searching parameter for Model Binding

On the screenshot above, you can see that I passed a parameter in the URL. After the MVC framework received the process and did the routing, the action invoker examines the Index method and finds an int parameter. Then the model binder for int calls its BindModel method to bind the value from the URL to the method parameter. The MVC framework searches four locations for a suiting parameter. If one is found, the search is finished and the value is processed. The locations for the search are:

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
        Request.Form
      </td>
      
      <td>
        Values provided by the user in HTML form elements
      </td>
    </tr>
    
    <tr>
      <td>
        RouteData.Values
      </td>
      
      <td>
        The values obtained using the application routes
      </td>
    </tr>
    
    <tr>
      <td>
        Request.QueryString
      </td>
      
      <td>
        Data included in the query string portion of the request URL
      </td>
    </tr>
    
    <tr>
      <td>
        Request.Files
      </td>
      
      <td>
        Files that have been uploaded as part of the request
      </td>
    </tr>
  </table>
</div>

In the example from above the model binder searches Request.Form[&#8220;id&#8221;] and then RouteData.Values[&#8220;id&#8221;]. The needed data can be found in the route information and therefore the search is finished.

Note that I am using the default route. It is also important that the variable name in the route and the parameter name are the same. If I named the parameter customerId, the model binder wouldn&#8217;t be able to match the id with the customerId.

## Binding primitive data types

When the model binder encounters a primitive data type, it tries to convert it into the needed type. If the conversion fails, an exception message is displayed because int is not nullable (except you handle the exception as I described in <a href="/built-filter-asp-net-mvc/" target="_blank" rel="noopener noreferrer">Built-in Filter in ASP.NET MVC</a>)

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2018/01/Converting-exception-leading-to-yellow-screen-of-death.jpg"><img loading="lazy" src="/assets/img/posts/2018/01/Converting-exception-leading-to-yellow-screen-of-death.jpg" alt="Converting exception leading to yellow screen of death" /></a>
  
  <p>
    Converting exception leading to yellow screen of death
  </p>
</div>

You can prevent this from happening but you have to make sure that your code can handle the id if it is null.

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2018/01/Nullable-int-parameter.jpg"><img aria-describedby="caption-attachment-708" loading="lazy" class="size-full wp-image-708" src="/assets/img/posts/2018/01/Nullable-int-parameter.jpg" alt="Nullable int parameter" /></a>
  
  <p>
    Nullable int parameter
  </p>
</div>

An even simpler alternative is to use default parameter.

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2018/01/Default-value-for-the-action-parameter.jpg"><img aria-describedby="caption-attachment-709" loading="lazy" class="size-full wp-image-709" src="/assets/img/posts/2018/01/Default-value-for-the-action-parameter.jpg" alt="Default value for the action parameter" /></a>
  
  <p>
    Default value for the action parameter
  </p>
</div>

If the model binder can&#8217;t bind the user input to the parameter, the default value is used.

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2018/01/Displaying-default-customer-after-casting-failed.jpg"><img aria-describedby="caption-attachment-725" loading="lazy" class="size-full wp-image-725" src="/assets/img/posts/2018/01/Displaying-default-customer-after-casting-failed.jpg" alt="Displaying default customer after casting failed" /></a>
  
  <p>
    Displaying default customer after casting failed
  </p>
</div>

Default parameters prevent the application from crashing if the model binding process fails but don&#8217;t forget if the value supplied by the user is valid for your application. For example, there is probably no customer with the id -1.

## Binding complex data types

If the DefaultModelBinder class encounters a complex type in the parameter, it uses reflections to get the public properties and then binds to each of them in turn.

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2018/01/Complex-data-type-Customer-as-action-parameter.jpg"><img aria-describedby="caption-attachment-726" loading="lazy" class="size-full wp-image-726" src="/assets/img/posts/2018/01/Complex-data-type-Customer-as-action-parameter.jpg" alt="Complex data type Customer as action parameter" /></a>
  
  <p>
    Complex data type Customer as action parameter
  </p>
</div>

If you use HTML Helper methods, the helper sets the name attributes of the element to match the format that the model binder uses.

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2018/01/Name-set-by-HTML-helper.jpg"><img aria-describedby="caption-attachment-727" loading="lazy" class="size-full wp-image-727" src="/assets/img/posts/2018/01/Name-set-by-HTML-helper.jpg" alt="Name set by HTML helper" /></a>
  
  <p>
    Name set by HTML helper
  </p>
</div>

### Using custom prefixes

Sometimes you don&#8217;t want to bind the data to the type the HTML generates for you. This means that the prefixes containing the view won&#8217;t correspond to the structure that the model binder is expecting and therefore your data won&#8217;t be processed properly. For example, I have some address information in a form and pass the form to an action. This action only needs some properties and therefore I create a new class called AddressShort.

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2018/01/New-action-for-address-details.jpg"><img aria-describedby="caption-attachment-711" loading="lazy" class="size-full wp-image-711" src="/assets/img/posts/2018/01/New-action-for-address-details.jpg" alt="New action for address details" /></a>
  
  <p>
    New action for address details
  </p>
</div>

If you pass the Customer class with the Address property to the action, it can&#8217;t be passed and the AddressShort object will only contain null objects. The values are null because the name attributes have the prefix Address in the HTML form and the model binder is looking for this type when trying to bind the AddressShort type. You can fix this by telling the model binder which prefix it should look for with the Bind attribute in the action.

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2018/01/The-Bind-attribute-in-the-action.jpg"><img aria-describedby="caption-attachment-712" loading="lazy" class="size-full wp-image-712" src="/assets/img/posts/2018/01/The-Bind-attribute-in-the-action.jpg" alt="The Bind attribute in the action" /></a>
  
  <p>
    The Bind attribute in the action
  </p>
</div>

I am not a big fan of this syntax because I think it makes the code messy but it is an easy way to achieve the desired behavior. If you call the DisplayAddressShort action from the CreateCustomer action, the AddressShort object will contain the country and city of the new customer.

### Binding only selected properties

Sometimes you don&#8217;t want the user to see sensitive data. You could hide the information in the HTML or create a new view model and only send the information you want to display. A simpler solution is to tell the model binder not to bind the properties which you don&#8217;t want to display. You can tell the model binder to not bind a property by using the Exclude attribute in the action.

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2018/01/Action-excluding-the-city-in-the-binding-process.jpg"><img loading="lazy" size-full" src="/assets/img/posts/2018/01/Action-excluding-the-city-in-the-binding-process.jpg" alt="Action excluding the city from the binding process" /></a>
  
  <p>
    Action excluding the city from the binding process
  </p>
</div>

On the screenshot above, you can see that I excluded the city from being bound. Another approach would be to include the properties I want to bind with the Include attribute. You can also use the Include and Exclude property in a class. On the following screenshot, I show how to use the Include property in the model class.

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2018/01/Applying-the-Include-property-on-a-class.jpg"><img loading="lazy" size-full" src="/assets/img/posts/2018/01/Applying-the-Include-property-on-a-class.jpg" alt="Applying the Include property in a class" /></a>
  
  <p>
    Applying the Include property in a class
  </p>
</div>

If you have the Bind attribute in the class and in the action, it only binds if neither the class nor the action excludes a property.

## Binding to Arrays and Collections

Model binding arrays and collections are supported by the model binder and can be achieved very easily.

### Binding to Arrays

To demonstrate how binding to an array works, I create a new action which takes a string array as parameter:

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2018/01/Action-with-an-array-parameter.jpg"><img aria-describedby="caption-attachment-716" loading="lazy" class="size-full wp-image-716" src="/assets/img/posts/2018/01/Action-with-an-array-parameter.jpg" alt="Action with an array parameter" /></a>
  
  <p>
    Action with an array parameter
  </p>
</div>

The model binder searches for all items with the name attribute countries and then create an array containing these values. Since it is not possible to assign a default value to an array, you have to check in the code whether the array is null. In this example, I let the user enter three countries, pass them as an array into the action and then return the array to display it in the view. It is important that all text boxes which take countries have the name set to countries which is the same name as the parameter. If these two don&#8217;t match, the model binder can&#8217;t bind the values to the parameter.

### Binding to collections

Binding to collections works as binding to arrays. I create a new action and let the user enter cities this time. The only difference is that the parameter is of type IList:

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2018/01/Action-with-a-list-parameter.jpg"><img aria-describedby="caption-attachment-717" loading="lazy" class="size-full wp-image-717" src="/assets/img/posts/2018/01/Action-with-a-list-parameter.jpg" alt="Action with a list parameter" /></a>
  
  <p>
    Action with a list parameter
  </p>
</div>

Again, I check whether the list is null and then pass the values to the view to display them. I also changed the name of the text boxes to cities to match the list parameter.

### Binding collections of complex data types

Binding to a collection of a complex data type is not different than binding to a normal collection except the naming of the items is a bit different. The names start with the index in square brackets followed by a period and the name of the property. For example [0].City

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2018/01/Names-starting-with-0-to-bind-to-a-collection.jpg"><img aria-describedby="caption-attachment-728" loading="lazy" class="size-full wp-image-728" src="/assets/img/posts/2018/01/Names-starting-with-0-to-bind-to-a-collection.jpg" alt="Names starting with [0] to bind to a collection" /></a>
  
  <p>
    Names starting with [0] to bind to a collection
  </p>
</div>After the values are sent to the action, the model binder binds the objects to the list. Starting with all items starting with [0] are added as the first object, all items starting with [1] as the second item and so on.

## Conclusion

In this post, I showed how the default model binder works and how you can bind primitive and complex data types. I also explained how to bind to arrays and collection.

For more details about model binding, I highly recommend the books <a href="http://amzn.to/2mgRbTy" target="_blank" rel="noopener noreferrer">Pro ASP.NET MVC 5</a> and <a href="http://amzn.to/2mfQ0nA" target="_blank" rel="noopener noreferrer">Pro ASP.NET MVC 5 Plattform</a>.

You can find the source code on <a href="https://github.com/WolfgangOfner/MVC-ModelBinding" target="_blank" rel="noopener noreferrer">GitHub</a>.