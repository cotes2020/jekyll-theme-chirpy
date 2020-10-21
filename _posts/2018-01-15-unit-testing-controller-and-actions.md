---
title: Unit Testing Controllers and Actions
date: 2018-01-15T20:10:13+01:00
author: Wolfgang Ofner
categories: [ASP.NET]
tags: [FluentAssertions, MVC, TDD, xUnit]
---
In <a href="https://www.programmingwithwolfgang.com/controllers-and-actions/" target="_blank" rel="noopener">my last post</a>, I showed how to work with controllers and actions. This included passing data, return types of actions and redirects. In this post, I want to check if my implemented features work as I expect them to by unit testing controllers and actions.

## Setting up the project

I use the project which I created the last time. For the testing, I add a class library project called ControllersAndActions.Test. In this project, I add one class called ActionTests. Usually, I would use several classes for every class I test. To make things simple, I only use this one class this time. After creating the class, I install xUnit and FluentAssertions. You can choose the testing framework of your liking. FluentAssertions is a great tool to make the asserts more readable.

## Testing the name of the view

The first tests I write will test what view is returned by the action. To do that, I call an action and then the view name of the returned value should be the name of the view I expect to be called.

<div id="attachment_636" style="width: 485px" class="wp-caption aligncenter">
  <a href="/wp-content/uploads/2018/01/Testing-returned-view-name-of-action.jpg"><img aria-describedby="caption-attachment-636" loading="lazy" class="size-full wp-image-636" src="/wp-content/uploads/2018/01/Testing-returned-view-name-of-action.jpg" alt="Testing returned view name of action" width="475" height="519" /></a>
  
  <p id="caption-attachment-636" class="wp-caption-text">
    Testing returned view name of action
  </p>
</div>

If your action returns an ActionResult instead of a ViewResult, you have to cast the object first before you can access the ViewName property.

## Testing ViewBag values

More interesting than testing if the right view, is called is testing the values of the view bag. You can access the view bag with the returned object of the action and compare it with the value you expect it to be.

<div id="attachment_637" style="width: 445px" class="wp-caption aligncenter">
  <a href="/wp-content/uploads/2018/01/Testing-ViewBag-value.jpg"><img aria-describedby="caption-attachment-637" loading="lazy" class="size-full wp-image-637" src="/wp-content/uploads/2018/01/Testing-ViewBag-value.jpg" alt="Testing ViewBag value" width="435" height="193" /></a>
  
  <p id="caption-attachment-637" class="wp-caption-text">
    Testing ViewBag value
  </p>
</div>

## Testing redirects

When testing the redirect to a controller, the return object of the action has two interesting properties when testing redirects. The first property is Permanent which is a bool indicating whether the redirect was permanent. The second property is URL which you can compare to the URL you expect.

<div id="attachment_638" style="width: 378px" class="wp-caption aligncenter">
  <a href="/wp-content/uploads/2018/01/Testing-redirect-to-a-controller.jpg"><img aria-describedby="caption-attachment-638" loading="lazy" class="size-full wp-image-638" src="/wp-content/uploads/2018/01/Testing-redirect-to-a-controller.jpg" alt="Testing redirect to a controller" width="368" height="386" /></a>
  
  <p id="caption-attachment-638" class="wp-caption-text">
    Testing redirect to a controller
  </p>
</div>

Testing the redirection to a route is a bit different. You have to compare the route values for the controller and action to the values you expect them to be.

<div id="attachment_639" style="width: 555px" class="wp-caption aligncenter">
  <a href="/wp-content/uploads/2018/01/Testing-redirect-to-a-route.jpg"><img aria-describedby="caption-attachment-639" loading="lazy" class="size-full wp-image-639" src="/wp-content/uploads/2018/01/Testing-redirect-to-a-route.jpg" alt="Testing redirect to a route" width="545" height="199" /></a>
  
  <p id="caption-attachment-639" class="wp-caption-text">
    Testing redirect to a route
  </p>
</div>

## Testing the returned status code

The next interesting property in the return object is the status code. With this property, you can check if the returned HTTP status code is what you expect.

<div id="attachment_640" style="width: 342px" class="wp-caption aligncenter">
  <a href="/wp-content/uploads/2018/01/Testing-the-status-code.jpg"><img aria-describedby="caption-attachment-640" loading="lazy" class="size-full wp-image-640" src="/wp-content/uploads/2018/01/Testing-the-status-code.jpg" alt="Testing the status code" width="332" height="174" /></a>
  
  <p id="caption-attachment-640" class="wp-caption-text">
    Testing the status code
  </p>
</div>

## Testing the ViewModel

The last test for today tests if the view model has the expected data type. To do that use the ViewData.Model property of the return value.

<div id="attachment_641" style="width: 540px" class="wp-caption aligncenter">
  <a href="/wp-content/uploads/2018/01/Testing-ViewData-data-type.jpg"><img aria-describedby="caption-attachment-641" loading="lazy" class="size-full wp-image-641" src="/wp-content/uploads/2018/01/Testing-ViewData-data-type.jpg" alt="Testing ViewData data type" width="530" height="178" /></a>
  
  <p id="caption-attachment-641" class="wp-caption-text">
    Testing ViewData data type
  </p>
</div>

## Conclusion

In this post, I presented several test cases for unit testing controller and actions. For these tests, I accessed various properties of the return object of the action call.

For more details about how controller and actions work, I highly recommend the books <a href="http://amzn.to/2mgRbTy" target="_blank" rel="noopener">Pro ASP.NET MVC 5</a> and <a href="http://amzn.to/2mfQ0nA" target="_blank" rel="noopener">Pro ASP.NET MVC 5 Plattform</a>.

You can find the source code on <a href="https://github.com/WolfgangOfner/MVC-ActionAndController" target="_blank" rel="noopener">GitHub</a>.