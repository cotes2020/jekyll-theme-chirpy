---
title: Templated Helper Methods
date: 2018-02-08T23:36:06+01:00
author: Wolfgang Ofner
categories: [ASP.NET]
tags: [attribute, scaffolding, template]
---
I talked about HTML Helper Methods in <a href="/helper-methods-in-asp-net-mvc/" target="_blank" rel="noopener noreferrer">my last post</a>. Although it was a pretty lengthy post, I only talked about the basic helper methods. The problem with this approach was that I had to decide which input field I want to render. I want to tell the framework which property I want to be displayed and let the framework decide which HTML element gets rendered. Therefore, I want to talk about templated helper methods today and show how they can be used to easily create views with even less programming to do.

## Setting up the project

I use the same project as last time. The only changes I made was adding an address and the birthday to the customer class. You can download the code <a href="https://github.com/WolfgangOfner/MVC-TemplatedHelperMethods" target="_blank" rel="noopener noreferrer">here</a>.

<div id="attachment_798" style="width: 305px" class="wp-caption aligncenter">
  <a href="/wp-content/uploads/2018/02/Implementation-of-the-Address-class.jpg"><img aria-describedby="caption-attachment-798" loading="lazy" class="size-full wp-image-798" src="/wp-content/uploads/2018/02/Implementation-of-the-Address-class.jpg" alt="Implementation of the Address class" width="295" height="278" /></a>
  
  <p id="caption-attachment-798" class="wp-caption-text">
    Implementation of the Address class
  </p>
</div>

## Using Templated Helper Methods

The simplest templated helper method is Html.Editor and the strongly typed version Html.EditorFor. I like the strongly typed version due to it&#8217;s IntelliSense support and pass the property name as the parameter. The MVC framework renders the element which it thinks fits best.

<div id="attachment_802" style="width: 492px" class="wp-caption aligncenter">
  <a href="/wp-content/uploads/2018/02/Using-Html.EditorFor-to-create-HTML-elements.jpg"><img aria-describedby="caption-attachment-802" loading="lazy" class="size-full wp-image-802" src="/wp-content/uploads/2018/02/Using-Html.EditorFor-to-create-HTML-elements.jpg" alt="Using Html.EditorFor to create HTML elements" width="482" height="337" /></a>
  
  <p id="caption-attachment-802" class="wp-caption-text">
    Using Html.EditorFor to create HTML elements
  </p>
</div>

The output can vary depending on what browser you use.  When you go to the devTools and inspect the rendered elements, you will see that the type for CustomerId is number and the type of Birthday is datetime whereas all the others have the type text. This type tells the browser which element it should render.

The MVC framework has the following built-in templated helper methods:

<div class="table-responsive">
  <table class="table table-striped table-bordered table-hover">
    <tr>
      <td>
        Helper
      </td>
      
      <td>
        Example
      </td>
      
      <td>
        Description
      </td>
    </tr>
    
    <tr>
      <td>
        Display
      </td>
      
      <td>
        Html.Display(&#8220;PropertyName&#8221;)
      </td>
      
      <td>
        Renders a read-only view of the specified model property, choosing an HTML element according to the property’s type and metadata
      </td>
    </tr>
    
    <tr>
      <td>
        DisplayFor
      </td>
      
      <td>
        Html.DisplayFor(x => x.PropertyName)
      </td>
      
      <td>
        Strongly typed version of the Display helper
      </td>
    </tr>
    
    <tr>
      <td>
        Editor
      </td>
      
      <td>
        Html.Editor(&#8220;PropertyName&#8221;)
      </td>
      
      <td>
        Renders an editor for the specified model property, choosing an HTML element according to the property’s type and metadata
      </td>
    </tr>
    
    <tr>
      <td>
        EditorFor
      </td>
      
      <td>
        Html.EditorFor(x => x.PropertyName)
      </td>
      
      <td>
        Strongly typed version of the Editor helper
      </td>
    </tr>
    
    <tr>
      <td>
        Label
      </td>
      
      <td>
        Html.Label(&#8220;PropertyName&#8221;)
      </td>
      
      <td>
        Renders an HTML label element referring to the specified model property
      </td>
    </tr>
    
    <tr>
      <td>
        LabelFor
      </td>
      
      <td>
        Html.LabelFor(x => x.PropertyName)
      </td>
      
      <td>
        Strongly typed version of the Label helper
      </td>
    </tr>
  </table>
</div>

## Displaying the Model with Templated Helper Methods

So far I have used templated helper methods for every property. The MVC framework also offers methods to process an entire model. This process is called scaffolding.

The following templated helper methods are available for the scaffolding process:

<div class="table-responsive">
  <table class="table table-striped table-bordered table-hover">
    <tr>
      <td>
        Helper
      </td>
      
      <td>
        Example
      </td>
      
      <td>
        Description
      </td>
    </tr>
    
    <tr>
      <td>
        DisplayForModel
      </td>
      
      <td>
        Html.DisplayForModel()
      </td>
      
      <td>
        Renders a read-only view of the entire model object
      </td>
    </tr>
    
    <tr>
      <td>
        EditorForModel
      </td>
      
      <td>
        Html.EditorForModel()
      </td>
      
      <td>
        Displays editor elements for the entire model object
      </td>
    </tr>
    
    <tr>
      <td>
        LabelForModel
      </td>
      
      <td>
        Html.LabelForModel()
      </td>
      
      <td>
        Renders an HTML label element referring to the entire model object
      </td>
    </tr>
  </table>
</div>

With these templated helper methods it is possible to render forms with just a couple lines of code. This makes them easy to read and easy to change.

<div id="attachment_799" style="width: 486px" class="wp-caption aligncenter">
  <a href="/wp-content/uploads/2018/02/Creating-a-form-using-templated-helper-methods-with-the-model.jpg"><img aria-describedby="caption-attachment-799" loading="lazy" class="size-full wp-image-799" src="/wp-content/uploads/2018/02/Creating-a-form-using-templated-helper-methods-with-the-model.jpg" alt="Creating a form using templated helper methods with the model" width="476" height="169" /></a>
  
  <p id="caption-attachment-799" class="wp-caption-text">
    Creating a form using templated helper methods with the model
  </p>
</div>

When you look at the result you can see that there are some problems with the output:

[<img loading="lazy" width="259" height="451" class="size-full wp-image-800" src="/wp-content/uploads/2018/02/Output-of-the-form-which-was-created-with-the-templated-helper-methods.jpg" alt="&quot;<yoastmark" />](/wp-content/uploads/2018/02/Output-of-the-form-which-was-created-with-the-templated-helper-methods.jpg)

The Html.LabelForModel method didn&#8217;t render anything in the headline, and the role is a textbox instead of a drop-down list. (this time it is not the fault of IE) Additionally, the Address is not rendered and probably I don&#8217;t want to display the Id to the user.

The solution to this problem is to add attributes to the model class to tell the framework what and how it should render the properties.

## Using Model Attributes

The MVC framework renders the HTML fields which it thinks fit best. As you saw in the last example, this is not always what you expect or want. I like the templated helper methods because they make the view so simple but I also have to ensure that they render what I want. I can add some metadata to my model to tell the MVC framework how to handle the property. ASP.NET MVC has a variety of built-in attributes which help me to display what I want with the scaffolding process.

### Using Metadata for Data Values

Attributes offer a convenient way to tell the rendering engine which datatype it renders. For example, I can tell it that I only want to display the date part of the birthday or that I want to render a password field which masks the user&#8217;s input.

The ASP.NET MVC framework offers the following datatype attributes:

<div class="table-responsive">
  <table class="table table-striped table-bordered table-hover">
    <tr>
      <td>
        DataType Value
      </td>
      
      <td>
        Description
      </td>
    </tr>
    
    <tr>
      <td>
        Date
      </td>
      
      <td>
        Displays the date part of a DateTime
      </td>
    </tr>
    
    <tr>
      <td>
        DateTime
      </td>
      
      <td>
        Displays a date and time (this is the default behavior for System.DateTime values)
      </td>
    </tr>
    
    <tr>
      <td>
        EmailAddress
      </td>
      
      <td>
        Displays the data as an e-mail address (using a link (a) element with a mailto href)
      </td>
    </tr>
    
    <tr>
      <td>
        MultilineText
      </td>
      
      <td>
        Renders the value in a textarea element
      </td>
    </tr>
    
    <tr>
      <td>
        Password
      </td>
      
      <td>
        Displays the data so that individual characters are masked from view
      </td>
    </tr>
    
    <tr>
      <td>
        PhoneNumber
      </td>
      
      <td>
        Displays a phone number
      </td>
    </tr>
    
    <tr>
      <td>
        Text
      </td>
      
      <td>
        Displays a single line of text
      </td>
    </tr>
    
    <tr>
      <td>
        Time
      </td>
      
      <td>
        Displays the time part of a DateTime
      </td>
    </tr>
    
    <tr>
      <td>
        Url
      </td>
      
      <td>
        Displays the data as a URL (using an HTML link (a) element)
      </td>
    </tr>
  </table>
  
  <h3>
    Hide or Display Elements using Metadata
  </h3>
  
  <p>
    Earlier I said that I don&#8217;t want to display the CustomerId to the user. It is common not to display all information, for example, the id or primary key of an element is usually not relevant to the user. To hide a property, I decorate it with the HiddenInput attribute and set the DisplayValue to false.
  </p>
  
  <div id="attachment_803" style="width: 285px" class="wp-caption aligncenter">
    <a href="/wp-content/uploads/2018/02/Applying-the-HiddenInput-attribute-to-the-CustomerId.jpg"><img aria-describedby="caption-attachment-803" loading="lazy" class="size-full wp-image-803" src="/wp-content/uploads/2018/02/Applying-the-HiddenInput-attribute-to-the-CustomerId.jpg" alt="Applying the HiddenInput attribute to the CustomerId" width="275" height="58" /></a>
    
    <p id="caption-attachment-803" class="wp-caption-text">
      Applying the HiddenInput attribute to the CustomerId
    </p>
  </div>
  
  <p>
    I have to set the DisplayValue property to false because otherwise, the MVC framework would render a read-only field.
  </p>
  
  <p>
    Another approach to hide a property is to exclude it from scaffolding with [ScaffoldColumn(false)]. The problem with the excluding is that it doesn&#8217;t get sent to the view. Therefore if the user returns the view for example after editing some information, the id is not included and so I don&#8217;t know which user was sent back.
  </p>
  
  <h3>
    Using Attributes to display Property Names
  </h3>
  
  <p>
    The scaffolding process displays the name of the property as the label. The problem is that property names are rarely useful to the user. No user wants to read FirstName. A user expects First name. Therefore, I can decorate properties with the Display attribute and a class with the DisplayName attribute. As the parameter, I pass the name which I want to be displayed.
  </p>
  
  <div id="attachment_806" style="width: 317px" class="wp-caption aligncenter">
    <a href="/wp-content/uploads/2018/02/Decorating-the-model-properties-with-the-Display-attribute.jpg"><img aria-describedby="caption-attachment-806" loading="lazy" class="size-full wp-image-806" src="/wp-content/uploads/2018/02/Decorating-the-model-properties-with-the-Display-attribute.jpg" alt="Decorating the model properties with the Display attribute" width="307" height="254" /></a>
    
    <p id="caption-attachment-806" class="wp-caption-text">
      Decorating the model properties with the Display attribute
    </p>
  </div>
  
  <p>
    When I start the application now, nice names are displayed and also Person in the header will be rendered.
  </p>
  
  <div id="attachment_807" style="width: 283px" class="wp-caption aligncenter">
    <a href="/wp-content/uploads/2018/02/Output-of-the-form-with-the-Display-attribute-for-naming-the-properties.jpg"><img aria-describedby="caption-attachment-807" loading="lazy" class="size-full wp-image-807" src="/wp-content/uploads/2018/02/Output-of-the-form-with-the-Display-attribute-for-naming-the-properties.jpg" alt="Output of the form with the Display attribute for naming the properties" width="273" height="379" /></a>
    
    <p id="caption-attachment-807" class="wp-caption-text">
      Output of the form with the Display attribute for naming the properties
    </p>
  </div>
  
  <h3>
    Applying Metadata to automatically created Classes
  </h3>
  
  <p>
    It is not always possible to apply attributes to classes because they are automatically generated by tools like the Entity Framework. (Actually, you can add attributes but they will be overridden the next time the class gets updated and therefore generated again). The solution to this problem is to add a partial class of the class you want to extend with the same properties and add the attributes there. To be able to do that the original class has to be partial as well. Fortunately Entity Framework creates partial classes. Then you have to add the MetadataType attribute to the class with typeof(your class) as the parameter.
  </p>
  
  <div id="attachment_808" style="width: 357px" class="wp-caption aligncenter">
    <a href="/wp-content/uploads/2018/02/Implementation-of-the-partial-customer-class-with-attributes.jpg"><img aria-describedby="caption-attachment-808" loading="lazy" class="size-full wp-image-808" src="/wp-content/uploads/2018/02/Implementation-of-the-partial-customer-class-with-attributes.jpg" alt="Implementation of the partial customer class with attributes" width="347" height="327" /></a>
    
    <p id="caption-attachment-808" class="wp-caption-text">
      Implementation of the partial customer class with attributes
    </p>
  </div>
  
  <p>
    On the screenshot above, I created a partial class and called it CustomerWithAttributes. Then I added all attributes from the Customer class which has attributes. Additionally to these changes, I made the Customer class partial and added the MetadataType attribute to it with the CustomerWithAttributes class as the parameter.
  </p>
  
  <div id="attachment_809" style="width: 359px" class="wp-caption aligncenter">
    <a href="/wp-content/uploads/2018/02/Adding-the-MetaData-attribute-to-the-customer-class.jpg"><img aria-describedby="caption-attachment-809" loading="lazy" class="size-full wp-image-809" src="/wp-content/uploads/2018/02/Adding-the-MetaData-attribute-to-the-customer-class.jpg" alt="Adding the MetaData attribute to the customer class" width="349" height="78" /></a>
    
    <p id="caption-attachment-809" class="wp-caption-text">
      Adding the MetaData attribute to the customer class
    </p>
  </div>
  
  <p>
    I also removed all attributes from this class since I want to simulate an automatically generated class. If you run your application now, it will look as before. Note that the partial classes have to be in the same namespace.
  </p>
  
  <h3>
    Displaying Complex Type Properties with Templated Helper Methods
  </h3>
  
  <p>
    The next part I want to fix is the display of the Address and its properties. The Address properties weren&#8217;t rendered yet because the EditorFor helper can only operate on simple types such int or string. Therefore it does not work recursively and as a result complex data types get ignored. The reason why the MVC framework is not rendering properties recursively is that it could easily trigger lazy-loading which will result in rendering all elements from the database.
  </p>
  
  <p>
    To render complex datatype, I have to explicitly tell the MVC framework how to do that by creating a separate call to the templated helper method:
  </p>
  
  <div id="attachment_810" style="width: 494px" class="wp-caption aligncenter">
    <a href="/wp-content/uploads/2018/02/Adding-a-render-method-for-the-Address-property.jpg"><img aria-describedby="caption-attachment-810" loading="lazy" class="size-full wp-image-810" src="/wp-content/uploads/2018/02/Adding-a-render-method-for-the-Address-property.jpg" alt="Adding a render method for the Address property" width="484" height="187" /></a>
    
    <p id="caption-attachment-810" class="wp-caption-text">
      Adding a render method for the Address property
    </p>
  </div>
  
  <p>
    The only change I had to make was adding the line with EditorFor and all properties of the Address class will be rendered after the previously rendered properties.
  </p>
  
  <h2>
    Conclusion
  </h2>
  
  <p>
    In this post, I refactored the application of my last post to use templated helper methods which enables you to create a view with only a couple of codes. To help the render engine, I showed how to add additional information on what you want to be rendered by applying attributes to the model class. The result is that the view is simple to understand and can be easily extended or changed.
  </p>
  
  <p>
    For more details about model validation, I highly recommend the books <a href="http://amzn.to/2mgRbTy" target="_blank" rel="noopener noreferrer">Pro ASP.NET MVC 5</a> and <a href="http://amzn.to/2mfQ0nA" target="_blank" rel="noopener noreferrer">Pro ASP.NET MVC 5 Plattform</a>.
  </p>
  
  <p>
    You can find the source code with all examples on <a href="https://github.com/WolfgangOfner/MVC-TemplatedHelperMethods" target="_blank" rel="noopener noreferrer">GitHub</a>.
  </p>
</div>