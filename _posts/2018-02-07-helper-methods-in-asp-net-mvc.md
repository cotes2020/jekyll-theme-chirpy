---
title: Helper Methods in ASP.NET MVC
date: 2018-02-07T18:57:46+01:00
author: Wolfgang Ofner
categories: [ASP.NET]
tags: [form, helper, HTML]
---
Writing HTML forms can be a tedious task. To help the developers, the ASP.NET MVC framework offers a wide range of helper methods which make creating an HTML form way easier.

ASP.NET MVC offers for every feature the option to use your own implementation. In this post, I will only talk about the built-in helper methods and not about custom helper methods.

## Setting up the project for Model Validation

I created a new ASP.NET MVC project with the empty template and add folders and core references for MVC.

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2018/01/Set-up-project.jpg"><img loading="lazy" src="/assets/img/posts/2018/01/Set-up-project.jpg" alt="Setting up the ASP.NET MVC project" /></a>
  
  <p>
    Setting up the ASP.NET MVC project
  </p>
</div>

In the next step, I create a model classes, Customer and a Role enum.

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2018/02/Implementation-of-the-Customer-class.jpg"><img aria-describedby="caption-attachment-793" loading="lazy" class="size-full wp-image-793" src="/assets/img/posts/2018/02/Implementation-of-the-Customer-class.jpg" alt="Implementation of the Customer class" /></a>
  
  <p>
    Implementation of the Customer class
  </p>
</div>

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2018/02/Implementation-of-the-Role-enum.jpg"><img aria-describedby="caption-attachment-776" loading="lazy" class="size-full wp-image-776" src="/assets/img/posts/2018/02/Implementation-of-the-Role-enum.jpg" alt="Implementation of the Role enum" /></a>
  
  <p>
    Implementation of the Role enum
  </p>
</div>

In the Home controller, I create two actions, both with the name CreateCustomer. One sends a new Customer object to the view and the other action takes an HTTP post request with the customer. This is the standard approach to deal with HTML forms.

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2018/02/Actions-in-the-Home-controller.jpg"><img aria-describedby="caption-attachment-777" loading="lazy" class="size-full wp-image-777" src="/assets/img/posts/2018/02/Actions-in-the-Home-controller.jpg" alt="Actions in the Home controller" /></a>
  
  <p>
    Actions in the Home controller
  </p>
</div>

## Using built-in Helper Methods

The MVC framework offers many helper methods out of the box to manage your HTML code for elements. First, I want to show you how to build a form without any helper methods and then extend this form piece by piece.

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2018/02/View-with-a-simple-HTML-form.jpg"><img loading="lazy" src="/assets/img/posts/2018/02/View-with-a-simple-HTML-form.jpg" alt="View with a simple HTML form" /></a>
  
  <p>
    View with a simple HTML form
  </p>
</div>

This is a very simple view. I added Bootstrap and some simple CSS. Usually, I would use a layout file but since the focus of this post is on helper methods, I think this approach is fine. Note that I have to set the name property of every input field. This name corresponds to a property of the model. If I don&#8217;t set the name, the <a href="/model-binding-in-asp-net-mvc/" target="_blank" rel="noopener noreferrer">model binder </a>wouldn&#8217;t be able to bind the data.

### Creating a form

The most used helper method is Html.BeginForm. This helper creates the form tag and has 13 different versions. You can use either @{Html.BeginForm();} to create an open form tag and @{Html.EndForm();} to create a closing form tag or you can use the simpler version @using(Html.BeginForm()){ and add the closing bracket } at the end of the form. I think I never saw anything else than the approach with the using since it&#8217;s so much simpler.

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2018/02/Creating-a-form-using-Html.BeginForm.jpg"><img aria-describedby="caption-attachment-781" loading="lazy" class="size-full wp-image-781" src="/assets/img/posts/2018/02/Creating-a-form-using-Html.BeginForm.jpg" alt="Creating a form using Html.BeginForm()" /></a>
  
  <p>
    Creating a form using Html.BeginForm()
  </p>
</div>

Here are the different overloaded versions of the BefginForm helper method:

<div class="table-responsive">
  <div class="table-responsive">
    <table class="table table-striped table-bordered table-hover">
      <tr>
        <td>
          Overload
        </td>
        
        <td>
          Description
        </td>
      </tr>
      
      <tr>
        <td>
          BeginForm()
        </td>
        
        <td>
          Creates a form which posts back to the action method it originated from
        </td>
      </tr>
      
      <tr>
        <td>
          BeginForm(action, controller)
        </td>
        
        <td>
          Creates a form which posts back to the action method and controller, specified as strings
        </td>
      </tr>
      
      <tr>
        <td>
          BeginForm(action, controller, method)
        </td>
        
        <td>
          As for the previous overload, but allows you to specify the value for the method attribute using a value from the System.Web.Mvc.FormMethod enumeration
        </td>
      </tr>
      
      <tr>
        <td>
          BeginForm(action, controller, method, attributes)
        </td>
        
        <td>
          As for the previous overload, but allows you to specify attributes for the form element an object whose properties are used as the attribute names
        </td>
      </tr>
      
      <tr>
        <td>
          BeginForm(action, controller, routeValues, method, attributes)
        </td>
        
        <td>
          Like the previous overload, but allows you to specify values for the variable route segments in your application routing configuration as an object whose properties correspond to the routing variables
        </td>
      </tr>
    </table>
  </div>
  
  <p>
    <a href="https://msdn.microsoft.com/en-us/library/system.web.mvc.html.formextensions.beginform(v=vs.118).aspx" target="_blank" rel="noopener noreferrer">Source 1</a>, <a href="http://amzn.to/2mgRbTy" target="_blank" rel="noopener noreferrer">Source 2</a>
  </p>
  
  <h3>
    Specifying the route for a form
  </h3>
  
  <p>
    When you use the BeginForm helper method, you leave the routing to the MVC framework. It takes the first route in the <a href="/routing-in-asp-net-mvc/" target="_blank" rel="noopener noreferrer">routing configuration</a>. If you want to use a specific route for your form, use BeginRouteForm instead.
  </p>
  
  <p>
    To demonstrate the BeginRouteForm helper method, I added a new route in the RouteConfig. I added the new route before the default one and named it MyFormRoute.
  </p>
  
  <div class="col-12 col-sm-10 aligncenter">
    <a href="/assets/img/posts/2018/02/New-route-for-BeginRouteForm-helper-method.jpg"><img aria-describedby="caption-attachment-782" loading="lazy" class="size-full wp-image-782" src="/assets/img/posts/2018/02/New-route-for-BeginRouteForm-helper-method.jpg" alt="New route for BeginRouteForm helper method" /></a>
    
    <p>
      New route for BeginRouteForm helper method
    </p>
  </div>
  
  <p>
    In the BeginRouteForm method, I pass the route name as the parameter. The MVC framework now finds the first route and routes to the CreateCustomer action in the Home controller. (I know this behavior isn&#8217;t very clever but it shows how BeginRouteForm works)
  </p>
  
  <div class="col-12 col-sm-10 aligncenter">
    <a href="/assets/img/posts/2018/02/Using-BeginRouteForm.jpg"><img aria-describedby="caption-attachment-783" loading="lazy" class="size-full wp-image-783" src="/assets/img/posts/2018/02/Using-BeginRouteForm.jpg" alt="Using BeginRouteForm with the route name as parameter" /></a>
    
    <p>
      Using BeginRouteForm with the route name as parameter
    </p>
  </div>
  
  <p>
    The BeginRouteForm helper method also has several overloaded versions which enable you to specify the form element in more detail.
  </p>
  
  <h2>
    Using HTML Input Helper Methods
  </h2>
  
  <p>
    In the previous example, I create the HTML code by hand and assigned the model data to the value attribute. The ASP.NET MVC offers several HTML Helper methods which make creating forms easier.
  </p>
  
  <p>
    The following input helper methods are available out of the box:
  </p>
  
  <div class="table-responsive">
    <table class="table table-striped table-bordered table-hover">
      <tr>
        <td>
          HTML Element
        </td>
        
        <td>
          Example
        </td>
      </tr>
      
      <tr>
        <td>
          Check box
        </td>
        
        <td>
          Html.CheckBox(&#8220;myCheckbox&#8221;, false)
        </td>
      </tr>
      
      <tr>
        <td>
          Hidden field
        </td>
        
        <td>
          Html.Hidden(&#8220;myHidden&#8221;, &#8220;val&#8221;)
        </td>
      </tr>
      
      <tr>
        <td>
          Radio button
        </td>
        
        <td>
          Html.RadioButton(&#8220;myRadiobutton&#8221;, &#8220;val&#8221;, true)
        </td>
      </tr>
      
      <tr>
        <td>
          Password
        </td>
        
        <td>
          Html.Password(&#8220;myPassword&#8221;, &#8220;val&#8221;)
        </td>
      </tr>
      
      <tr>
        <td>
          Text area
        </td>
        
        <td>
          Html.TextArea(&#8220;myTextarea&#8221;, &#8220;val&#8221;, 5, 20, null)
        </td>
      </tr>
      
      <tr>
        <td>
          Text box
        </td>
        
        <td>
          Html.TextBox(&#8220;myTextbox&#8221;, &#8220;val&#8221;)
        </td>
      </tr>
    </table>
  </div>
  
  <p>
    I replace the HTML input elements with Html.TextBox. The syntax might feel strange in the beginning but once you got used to it, you won&#8217;t go back to writing plain HTML forms.
  </p>
  
  <div class="col-12 col-sm-10 aligncenter">
    <a href="/assets/img/posts/2018/02/Replacing-the-HTML-input-fields-with-MVC-HTML-input-helper-methods.jpg"><img aria-describedby="caption-attachment-784" loading="lazy" class="size-full wp-image-784" src="/assets/img/posts/2018/02/Replacing-the-HTML-input-fields-with-MVC-HTML-input-helper-methods.jpg" alt="Replacing the HTML input fields with MVC HTML input helper methods" /></a>
    
    <p>
      Replacing the HTML input fields with MVC HTML input helper methods
    </p>
  </div>
  
  <h3>
    Generating HTML elements from a view model
  </h3>
  
  <p>
    Previously, I showed you how to create input elements using HTML helper methods. The problem with this approach is that I had to specify the name of the element and had to make sure that this name fits the view model property which I want to bind. If I had misspelled a name, the MVC framework wouldn&#8217;t be able to bind the property to the element.
  </p>
  
  <p>
    The solution to this problem is another overloaded version of the helper method which takes only a string as a parameter. The string is the property name and the MVC framework automatically creates the name attribute for you.
  </p>
  
  <div class="col-12 col-sm-10 aligncenter">
    <a href="/assets/img/posts/2018/02/Generating-input-elements-from-a-property.jpg"><img aria-describedby="caption-attachment-786" loading="lazy" class="size-full wp-image-786" src="/assets/img/posts/2018/02/Generating-input-elements-from-a-property.jpg" alt="Generating input elements from a property" /></a>
    
    <p>
      Generating input elements from a property
    </p>
  </div>
  
  <p>
    Note that I had to use null as the second parameter to add a CSS class. If you don&#8217;t need CSS you only need the string parameter. This string parameter is used to search the ViewBag and then the Model for a corresponding property. For example for the first input element with personId the MVC framework searches for ViewBag.personId and @Model.personId. It finds the property in the model and therefore is able to bind it. As always, the first value that is found is used.
  </p>
  
  <h3>
    Using strongly typed Input Helper Methods
  </h3>
  
  <p>
    Each input helper method has also a strongly typed version which works with lambda expressions. The view model is passed to the helper method and you can select the needed property. The strongly typed input helper methods can only be used in a strongly typed view.
  </p>
  
  <div class="col-12 col-sm-10 aligncenter">
    <a href="/assets/img/posts/2018/02/Creating-input-element-using-strongly-typed-input-helper-methods.jpg"><img aria-describedby="caption-attachment-787" loading="lazy" class="size-full wp-image-787" src="/assets/img/posts/2018/02/Creating-input-element-using-strongly-typed-input-helper-methods.jpg" alt="Creating input element using strongly typed input helper methods" /></a>
    
    <p>
      Creating input element using strongly typed input helper methods
    </p>
  </div>
  
  <p>
    I prefer this approach because it is less likely that you mistype a property name and with the lambda expressions you also have IntelliSense support.
  </p>
  
  <h3>
    Creating Select elements
  </h3>
  
  <p>
    So far, I only talked about input elements. The MVC framework also offers helper methods for drop-down lists.
  </p>
  
  <div class="col-12 col-sm-10 aligncenter">
    <a href="/assets/img/posts/2018/02/HTML-helper-methods-for-select-elements.jpg"><img aria-describedby="caption-attachment-791" loading="lazy" class="size-full wp-image-791" src="/assets/img/posts/2018/02/HTML-helper-methods-for-select-elements.jpg" alt="HTML helper methods for select elements" /></a>
    
    <p>
      HTML helper methods for select elements
    </p>
  </div>
  
  <p>
    The difference between a normal drop-down list and multiple-select is that the drop-down list allows only one element to be selected.
  </p>
  
  <div class="col-12 col-sm-10 aligncenter">
    <a href="/assets/img/posts/2018/02/Creating-a-drop-down-list-with-HTML-helper-methods.jpg"><img aria-describedby="caption-attachment-788" loading="lazy" class="size-full wp-image-788" src="/assets/img/posts/2018/02/Creating-a-drop-down-list-with-HTML-helper-methods.jpg" alt="Creating a drop-down list with HTML helper methods" /></a>
    
    <p>
      Creating a drop-down list with HTML helper methods
    </p>
  </div>
  
  <p>
    The drop-down list displays all available roles. the DropDownList and ListBox HTML helper methods work with IEnumerable. Therefore, I have to use GetNames to get all the names of the roles.
  </p>
  
  <h2>
    Conclusion
  </h2>
  
  <p>
    Today I talked about different approaches on how to create input fields with HTML helper methods which are built-in into the ASP.NET MVC framework. With all this new knowledge, it&#8217;s time to go out and write some code 😉
  </p>
  
  <p>
    For more details about model validation, I highly recommend the books <a href="http://amzn.to/2mgRbTy" target="_blank" rel="noopener noreferrer">Pro ASP.NET MVC 5</a> and <a href="http://amzn.to/2mfQ0nA" target="_blank" rel="noopener noreferrer">Pro ASP.NET MVC 5 Plattform</a>.
  </p>
  
  <p>
    You can find the source code with all examples on <a href="https://github.com/WolfgangOfner/MVC-HelperMethods" target="_blank" rel="noopener noreferrer">GitHub</a>.
  </p>
</div>