---
title: Model Validation in ASP.NET MVC
date: 2018-02-03T18:33:31+01:00
author: Wolfgang Ofner
categories: [ASP.NET]
tags: [ajax, attributes, Javascript, model validation]
---
Model validation is the process of checking whether the user input is suitable for <a href="https://www.programmingwithwolfgang.com/model-binding-in-asp-net-mvc/" target="_blank" rel="noopener noreferrer">model binding</a> and if not it should provide useful error messages to the user. The first part is to ensure that only valid entries are made. This should filter inputs which don&#8217;t make any sense. This could be a birth date in the future or an appointment date in the past. As important as checking for valid data is to inform the user about the wrong input and help him to enter the information in the expected form. Without any help, it can be frustrating and annoying for the user which might end up in losing a potential customer.

To help the developer, ASP.NET MVC provides several possibilities for model validation.

## Setting up the project for Model Validation

I created a new ASP.NET MVC project with the empty template and add folders and core references for MVC.

<div id="attachment_601" style="width: 796px" class="wp-caption aligncenter">
  <a href="/wp-content/uploads/2018/01/Set-up-project.jpg"><img aria-describedby="caption-attachment-601" loading="lazy" class="wp-image-601 size-full" title="Setting up the ASP.NET MVC project for model validation" src="/wp-content/uploads/2018/01/Set-up-project.jpg" alt="Setting up the ASP.NET MVC project for model validation" width="786" height="513" /></a>
  
  <p id="caption-attachment-601" class="wp-caption-text">
    Setting up the ASP.NET MVC project for model validation
  </p>
</div>

Then I create a simple Home controller with two actions. The first action returns a view to the user to enter customer information and the second action processes the user input. In a real-world solution, the customer information would be stored probably in a database. In this example, I don&#8217;t do anything with it except passing to a second view to display the information.

<div id="attachment_737" style="width: 550px" class="wp-caption aligncenter">
  <a href="/wp-content/uploads/2018/01/Implementation-of-the-Home-controller.jpg"><img aria-describedby="caption-attachment-737" loading="lazy" class="size-full wp-image-737" src="/wp-content/uploads/2018/01/Implementation-of-the-Home-controller.jpg" alt="Implementation of the Home controller" width="540" height="291" /></a>
  
  <p id="caption-attachment-737" class="wp-caption-text">
    Implementation of the Home controller
  </p>
</div>

To work with customers, I need to implement a customer model. I want to keep the customer as simple as possible and therefore the class only has basic properties.

<div id="attachment_738" style="width: 422px" class="wp-caption aligncenter">
  <a href="/wp-content/uploads/2018/01/Implementation-of-the-customer-class.jpg"><img aria-describedby="caption-attachment-738" loading="lazy" class="size-full wp-image-738" src="/wp-content/uploads/2018/01/Implementation-of-the-customer-class.jpg" alt="Implementation of the customer class" width="412" height="185" /></a>
  
  <p id="caption-attachment-738" class="wp-caption-text">
    Implementation of the customer class
  </p>
</div>

I keep the views simple too. The RegisterCustomer view takes a name, birthday and has a checkbox to accept the terms and conditions.

<div id="attachment_739" style="width: 669px" class="wp-caption aligncenter">
  <a href="/wp-content/uploads/2018/01/Register-new-customer-view.jpg"><img aria-describedby="caption-attachment-739" loading="lazy" class="size-full wp-image-739" src="/wp-content/uploads/2018/01/Register-new-customer-view.jpg" alt="Register new customer view" width="659" height="244" /></a>
  
  <p id="caption-attachment-739" class="wp-caption-text">
    Register new customer view
  </p>
</div>

Right now the user can enter whatever he wants and the application accepts it. To create a valid customer the user have to provide a name, a birth date (in the dd/mm/yyyy format) in the past and he must accept the terms and condition.

To enforce these requirements, I use model validation.

## Explicitly validating a Model

One possible approach of checking if the model is valid is checking directly in the action.

<div id="attachment_741" style="width: 597px" class="wp-caption aligncenter">
  <a href="/wp-content/uploads/2018/01/Validating-the-model-directly-in-the-action.jpg"><img aria-describedby="caption-attachment-741" loading="lazy" class="size-full wp-image-741" src="/wp-content/uploads/2018/01/Validating-the-model-directly-in-the-action.jpg" alt="Validating the model directly in the action" width="587" height="430" /></a>
  
  <p id="caption-attachment-741" class="wp-caption-text">
    Validating the model directly in the action
  </p>
</div>

I check if every field contains the expected value. If not, I add a model error to the Modelstate. After I checked every attribute, I return the RegisterComplete view if the Modelstate is valid and if not I return the input view again.

I added some CSS to the HTML which colors the border of the element with the model error in red.

<div id="attachment_743" style="width: 386px" class="wp-caption aligncenter">
  <a href="/wp-content/uploads/2018/02/CSS-to-highlight-model-error-in-the-view.jpg"><img aria-describedby="caption-attachment-743" loading="lazy" class="size-full wp-image-743" src="/wp-content/uploads/2018/02/CSS-to-highlight-model-error-in-the-view.jpg" alt="CSS to highlight model error in the view" width="376" height="311" /></a>
  
  <p id="caption-attachment-743" class="wp-caption-text">
    CSS to highlight model error in the view (<a href="http://amzn.to/2mgRbTy" target="_blank" rel="noopener noreferrer">Source</a>)
  </p>
</div>

<div id="attachment_744" style="width: 345px" class="wp-caption aligncenter">
  <a href="/wp-content/uploads/2018/02/Highlighted-errors-in-the-view.jpg"><img aria-describedby="caption-attachment-744" loading="lazy" class="size-full wp-image-744" src="/wp-content/uploads/2018/02/Highlighted-errors-in-the-view.jpg" alt="Highlighted errors in the view" width="335" height="223" /></a>
  
  <p id="caption-attachment-744" class="wp-caption-text">
    Highlighted errors in the view
  </p>
</div>

## Displaying error messages

Highlighting input fields which contain wrong values is a nice beginning but it doesn&#8217;t tell the user what is wrong. The MVC framework provides several helper methods to display useful error messages to the user. The simplest one is Html.ValidationSumary().

<div id="attachment_745" style="width: 672px" class="wp-caption aligncenter">
  <a href="/wp-content/uploads/2018/02/Displaying-error-messages-with-Html.ValidationSummar.jpg"><img aria-describedby="caption-attachment-745" loading="lazy" class="size-full wp-image-745" src="/wp-content/uploads/2018/02/Displaying-error-messages-with-Html.ValidationSummar.jpg" alt="Displaying error messages with Html.ValidationSummar" width="662" height="143" /></a>
  
  <p id="caption-attachment-745" class="wp-caption-text">
    Displaying error messages with Html.ValidationSummar
  </p>
</div>

This helper method adds all error messages above the form. If there are none, nothing will be rendered before the form.

<div id="attachment_746" style="width: 340px" class="wp-caption aligncenter">
  <a href="/wp-content/uploads/2018/02/Displaying-error-messages-to-the-user.jpg"><img aria-describedby="caption-attachment-746" loading="lazy" class="size-full wp-image-746" src="/wp-content/uploads/2018/02/Displaying-error-messages-to-the-user.jpg" alt="Displaying error messages to the user" width="330" height="265" /></a>
  
  <p id="caption-attachment-746" class="wp-caption-text">
    Displaying error messages to the user
  </p>
</div>

The Html.ValidationSummary helper has several overloaded implementations. See the following tables for the different versions:

<div class="table-responsive">
  <table class="table table-striped table-bordered table-hover">
    <tr>
      <td>
        Overloaded Method
      </td>
      
      <td>
        Description
      </td>
    </tr>
    
    <tr>
      <td>
        Html.ValidationSummary()
      </td>
      
      <td>
        Generates a summary for all validation errors
      </td>
    </tr>
    
    <tr>
      <td>
        Html.ValidationSummary(bool)
      </td>
      
      <td>
        If the bool parameter is true, then only model-level errors are displayed. If the parameter is false, then all errors are shown.
      </td>
    </tr>
    
    <tr>
      <td>
        Html.ValidationSummary(string)
      </td>
      
      <td>
        Displays a message (contained in the string parameter) before a summary of all the validation errors.
      </td>
    </tr>
    
    <tr>
      <td>
        Html.ValidationSummary(bool, string)
      </td>
      
      <td>
        Displays a message before the validation errors. If the bool parameter is true, only model-level errors will be shown.
      </td>
    </tr>
  </table>
</div>

### Model level validation messages

Another approach to validate a model is model level validation. This is used when you have to ensure that two or more properties interact correctly. I know it&#8217;s a stupid example but for whatever reason, you don&#8217;t allow a customers name to be Wolfgang with his birthday yesterday. Adding a model level error to the ModelState is also achieved by using the AddModelError with the difference that the first parameter is an empty string. Additionally, you have to pass true as the parameter for the ValidationSummary in the view.

<div id="attachment_748" style="width: 765px" class="wp-caption aligncenter">
  <a href="/wp-content/uploads/2018/02/Model-level-error-implementation.jpg"><img aria-describedby="caption-attachment-748" loading="lazy" class="size-full wp-image-748" src="/wp-content/uploads/2018/02/Model-level-error-implementation.jpg" alt="Model level error implementation" width="755" height="92" /></a>
  
  <p id="caption-attachment-748" class="wp-caption-text">
    Model level error implementation
  </p>
</div>

<div id="attachment_747" style="width: 368px" class="wp-caption aligncenter">
  <a href="/wp-content/uploads/2018/02/Displaying-the-model-level-error-message.jpg"><img aria-describedby="caption-attachment-747" loading="lazy" class="size-full wp-image-747" src="/wp-content/uploads/2018/02/Displaying-the-model-level-error-message.jpg" alt="Displaying the model level error message" width="358" height="247" /></a>
  
  <p id="caption-attachment-747" class="wp-caption-text">
    Displaying the model level error message
  </p>
</div>

### Property level validation messages

Displaying all error messages on the top of the form might be ok if the form is as short as mine but if the form has 20 rows, the user will be confused with all the error messages. The solution for this is the ValidationMessageFor HTML helper. This method takes a lambda expression with the name of the property of which it should display the error message.

<div id="attachment_751" style="width: 670px" class="wp-caption aligncenter">
  <a href="/wp-content/uploads/2018/02/Implementation-of-the-ValidationMessageFor-helper-for-every-input-field.jpg"><img aria-describedby="caption-attachment-751" loading="lazy" class="size-full wp-image-751" src="/wp-content/uploads/2018/02/Implementation-of-the-ValidationMessageFor-helper-for-every-input-field.jpg" alt="Implementation of the ValidationMessageFor helper for every input field" width="660" height="245" /></a>
  
  <p id="caption-attachment-751" class="wp-caption-text">
    Implementation of the ValidationMessageFor helper for every input field
  </p>
</div>

<div id="attachment_749" style="width: 336px" class="wp-caption aligncenter">
  <a href="/wp-content/uploads/2018/02/Displaying-an-error-message-for-every-input-field.jpg"><img aria-describedby="caption-attachment-749" loading="lazy" class="size-full wp-image-749" src="/wp-content/uploads/2018/02/Displaying-an-error-message-for-every-input-field.jpg" alt="Displaying an error message for every input field" width="326" height="325" /></a>
  
  <p id="caption-attachment-749" class="wp-caption-text">
    Displaying an error message for every input field
  </p>
</div>

## Validation using the Model Binder

The default <a href="https://www.programmingwithwolfgang.com/model-binding-in-asp-net-mvc/" target="_blank" rel="noopener noreferrer">model binder</a> performs validation during the binding process. If it can&#8217;t bind a property, it will display an error message. For example, if you leave the birthday empty, the model binder will display a message prompting the user to enter a birthday.

<div id="attachment_754" style="width: 313px" class="wp-caption aligncenter">
  <a href="/wp-content/uploads/2018/02/Error-message-by-model-binder-because-of-empty-birthday-field.jpg"><img aria-describedby="caption-attachment-754" loading="lazy" class="size-full wp-image-754" src="/wp-content/uploads/2018/02/Error-message-by-model-binder-because-of-empty-birthday-field.jpg" alt="Error message by model binder because of empty birthday field" width="303" height="253" /></a>
  
  <p id="caption-attachment-754" class="wp-caption-text">
    Error message by model binder because of empty birthday field
  </p>
</div>

The model binder also displays an error message if you try to enter an invalid value, for example, a string as birthday.

<div id="attachment_755" style="width: 335px" class="wp-caption aligncenter">
  <a href="/wp-content/uploads/2018/02/Error-message-by-model-binder-because-of-the-wrong-value-in-the-birthday-field.jpg"><img aria-describedby="caption-attachment-755" loading="lazy" class="size-full wp-image-755" src="/wp-content/uploads/2018/02/Error-message-by-model-binder-because-of-the-wrong-value-in-the-birthday-field.jpg" alt="Error message by model binder because of the wrong value in the birthday field" width="325" height="245" /></a>
  
  <p id="caption-attachment-755" class="wp-caption-text">
    Error message by model binder because of the wrong value in the birthday field
  </p>
</div>

Modern browsers like Chrome or Edge don&#8217;t even let the user enter a string. They offer a date picker to help the user entering the right value. The Internet Explorer, on the other hand, doesn&#8217;t offer this feature (surprise, surprise IE sucks).

## Validation using Metadata

The MVC framework enables you to add attributes to properties of a model. The advantage of this approach is that the attributes and therefore the model validation are always enforced when the model is used.

<div id="attachment_752" style="width: 621px" class="wp-caption aligncenter">
  <a href="/wp-content/uploads/2018/02/Using-Metadata-in-the-model-class.jpg"><img aria-describedby="caption-attachment-752" loading="lazy" class="size-full wp-image-752" src="/wp-content/uploads/2018/02/Using-Metadata-in-the-model-class.jpg" alt="Using Metadata in the model class" width="611" height="237" /></a>
  
  <p id="caption-attachment-752" class="wp-caption-text">
    Using built-in validation attributes in the model class
  </p>
</div>

I applied the Required attribute with and without a specific error message and the Range attribute to ensure that the checkbox is checked. If there is no custom error, a default error will be displayed. It feels unnatural to use the Range attribute for a checkbox but the framework doesn&#8217;t provide an attribute for bool operations. Therefore, I use the Range attribute and set the range from true to true. ASP.NET MVC provides five useful built-in attributes for validation:

<div class="table-responsive">
  <table class="table table-striped table-bordered table-hover">
    <tr>
      <td>
        Attribute
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
        Compare
      </td>
      
      <td>
        [Compare(&#8220;Password&#8221;)]
      </td>
      
      <td>
        Two properties must have the same value. This is useful<br /> when you ask the user to provide the same information<br /> twice, such as an e-mail address or a password.
      </td>
    </tr>
    
    <tr>
      <td>
        Range
      </td>
      
      <td>
        [Range(5, 10)]
      </td>
      
      <td>
        A numeric value must lie between the specified<br /> minimum and maximum values. To specify a boundary<br /> on only one side, use a MinValue or MaxValue<br /> constant—for example, [Range(int.MinValue, 10)].
      </td>
    </tr>
    
    <tr>
      <td>
        RegularExpression
      </td>
      
      <td>
        [RegularExpression(&#8220;regex&#8221;)]
      </td>
      
      <td>
        A string value must match the specified regular<br /> expression pattern. Note that the pattern has to match the<br /> entire user-supplied value, not just a substring within it.<br /> By default, it matches case sensitively, but you can make<br /> it case insensitive by applying the (?i) modifier—that is,<br /> [RegularExpression(&#8220;(?i)regex&#8221;)].
      </td>
    </tr>
    
    <tr>
      <td>
        Required
      </td>
      
      <td>
        [Required]
      </td>
      
      <td>
        The value must not be empty or be a string consisting<br /> only of spaces. If you want to treat whitespace as valid,<br /> use [Required(AllowEmptyStrings = true)].
      </td>
    </tr>
    
    <tr>
      <td>
        StringLength
      </td>
      
      <td>
        [StringLength(12)]
      </td>
      
      <td>
        A string value must not be longer than the specified<br /> maximum length. You can also specify a minimum<br /> length: [StringLength(12, MinimumLength=8)].
      </td>
    </tr>
  </table>
  
  <h2>
    Validation using a custom property attribute
  </h2>
  
  <p>
    The ASP.NET MVC framework is highly custom able and therefore it is easy to implement your own attribute for validating a property. I don&#8217;t like the Range attribute for a checkbox and therefore I will create my own called MustBeTrue.
  </p>
  
  <p>
    As I already said, implementing your own attribute is pretty simple. You only have to create a new class which derives from ValidationAttribute and then overrides the IsValid method.
  </p>
  
  <div id="attachment_756" style="width: 406px" class="wp-caption aligncenter">
    <a href="/wp-content/uploads/2018/02/Implementation-of-a-custom-attribute-for-property-validation.jpg"><img aria-describedby="caption-attachment-756" loading="lazy" class="size-full wp-image-756" src="/wp-content/uploads/2018/02/Implementation-of-a-custom-attribute-for-property-validation.jpg" alt="Implementation of a custom attribute for property validation" width="396" height="147" /></a>
    
    <p id="caption-attachment-756" class="wp-caption-text">
      Implementation of a custom attribute for property validation
    </p>
  </div>
  
  <p>
    Now I can replace the Range attribute with the MustBeTrue attribute on the property.
  </p>
  
  <div id="attachment_760" style="width: 426px" class="wp-caption aligncenter">
    <a href="/wp-content/uploads/2018/02/Applying-my-custom-attribute-to-a-property.jpg"><img aria-describedby="caption-attachment-760" loading="lazy" class="size-full wp-image-760" src="/wp-content/uploads/2018/02/Applying-my-custom-attribute-to-a-property.jpg" alt="Applying my custom attribute to a property" width="416" height="61" /></a>
    
    <p id="caption-attachment-760" class="wp-caption-text">
      Applying my custom attribute to a property
    </p>
  </div>
  
  <p>
    As you can see, I can still use the ErrorMessage property to display the same error message as before.
  </p>
  
  <h3>
    Implementing a custom property by deriving from a built-in property
  </h3>
  
  <p>
    You can also derive from a built-in validation property, for example, Required, and extend its functionality. I create a new validation property which derives from RequiredAttribute and override the IsValid method again. Then I return IsValid from the base class and my additional check. In this case, I want to make sure that the date is in the past.
  </p>
  
  <div id="attachment_762" style="width: 524px" class="wp-caption aligncenter">
    <a href="/wp-content/uploads/2018/02/Extending-the-Required-attribute-with-my-custom-validation.jpg"><img aria-describedby="caption-attachment-762" loading="lazy" class="size-full wp-image-762" src="/wp-content/uploads/2018/02/Extending-the-Required-attribute-with-my-custom-validation.jpg" alt="Extending the Required attribute with my custom validation" width="514" height="153" /></a>
    
    <p id="caption-attachment-762" class="wp-caption-text">
      Extending the Required attribute with my custom validation
    </p>
  </div>
  
  <h2>
    Performing client-side validation
  </h2>
  
  <p>
    All previous validation techniques are for server-side validation. This means that the user sends the data to the server, then it gets validated and sent back. Sending data back and forth uses a lot of bandwidth which can lead to problems if you have a very popular application or if the user has a slow connection, for example on his mobile phone. Therefore it would be better to do some validation on the client even before sending the data to the server. This reduces the amount of data sent back and forth and also increases the user experience.
  </p>
  
  <h3>
    Setting up client-side validation
  </h3>
  
  <p>
    To enable client-side validation change the values of ClientValidationEnabled and UnobtrusiveJavaScriptEnabled in the web.config to true.
  </p>
  
  <div id="attachment_763" style="width: 406px" class="wp-caption aligncenter">
    <a href="/wp-content/uploads/2018/02/Configuring-web.config-for-client-side-validation.jpg"><img aria-describedby="caption-attachment-763" loading="lazy" class="size-full wp-image-763" src="/wp-content/uploads/2018/02/Configuring-web.config-for-client-side-validation.jpg" alt="Configuring web.config for client-side validation" width="396" height="39" /></a>
    
    <p id="caption-attachment-763" class="wp-caption-text">
      Configuring web.config for client-side validation
    </p>
  </div>
  
  <p>
    The next step is to install the following NuGet packages:
  </p>
  
  <ul>
    <li>
      jQuery
    </li>
    <li>
      jQuery.Validation
    </li>
    <li>
      Microsoft.jQuery.Unobtrusive.Validation
    </li>
  </ul>
  
  <p>
    The last step is to add these three Javascript files to your layout or view.
  </p>
  
  <h3>
    Using client-side validation
  </h3>
  
  <p>
    Using client-side validation is pretty simple. Just add validation properties to your model, for example, Required or StringLength.
  </p>
  
  <div id="attachment_764" style="width: 330px" class="wp-caption aligncenter">
    <a href="/wp-content/uploads/2018/02/Adding-validation-attributes-to-the-model-for-client-side-validation.jpg"><img aria-describedby="caption-attachment-764" loading="lazy" class="size-full wp-image-764" src="/wp-content/uploads/2018/02/Adding-validation-attributes-to-the-model-for-client-side-validation.jpg" alt="Adding validation attributes to the model for client-side validation" width="320" height="110" /></a>
    
    <p id="caption-attachment-764" class="wp-caption-text">
      Adding validation attributes to the model for client-side validation
    </p>
  </div>
  
  <div id="attachment_765" style="width: 614px" class="wp-caption aligncenter">
    <a href="/wp-content/uploads/2018/02/Error-message-from-client-side-validation.jpg"><img aria-describedby="caption-attachment-765" loading="lazy" class="size-full wp-image-765" src="/wp-content/uploads/2018/02/Error-message-from-client-side-validation.jpg" alt="Error message from client-side validation" width="604" height="232" /></a>
    
    <p id="caption-attachment-765" class="wp-caption-text">
      Error message from client-side validation
    </p>
  </div>
  
  <p>
    Thanks to client-side validation the user gets informed that the name has to be at least two characters long even before the form is sent to the server.
  </p>
</div>

Client-side validation is a great feature. However, you still have to do server-side validation to ensure the integrity of the data.

## Performing remote validation

Remote validation is a mix of server-side and client-side validation. It is often used to check whether an entered username or email is already taken. The check is performed in the background using ajax without the user doing anything. It is a pretty lightweight request but it still should be used for only some input fields since every input is sent back to the server which leads to a lot of requests.

### Implementing remote validation

Implementing remote validation consists of two steps. The first step is to implement an action in your controller which returns a JsonResult and takes one string parameter.

<div id="attachment_766" style="width: 596px" class="wp-caption aligncenter">
  <a href="/wp-content/uploads/2018/02/Implementation-of-the-remote-validation-in-the-controller.jpg"><img aria-describedby="caption-attachment-766" loading="lazy" class="size-full wp-image-766" src="/wp-content/uploads/2018/02/Implementation-of-the-remote-validation-in-the-controller.jpg" alt="Implementation of the remote validation in the controller" width="586" height="122" /></a>
  
  <p id="caption-attachment-766" class="wp-caption-text">
    Implementation of the remote validation in the controller
  </p>
</div>

In the action, I check if the customer name is Wolfgang. If it is Wolfgang, I return an error message. Note that the parameter is Name with an upper case N. I have to do this because the parameter has to match the name of the input field you want to check. I also have to allow Json GET requests because they are disallowed by the MVC framework by default.

The second step is to add the Remote attribute to the Name property of the Customer class. Additionally, I add the validation method name and the controller.

<div id="attachment_767" style="width: 288px" class="wp-caption aligncenter">
  <a href="/wp-content/uploads/2018/02/Adding-the-remote-attribute-to-the-property-of-the-model-class.jpg"><img aria-describedby="caption-attachment-767" loading="lazy" class="size-full wp-image-767" src="/wp-content/uploads/2018/02/Adding-the-remote-attribute-to-the-property-of-the-model-class.jpg" alt="Adding the remote attribute to the property of the model class" width="278" height="90" /></a>
  
  <p id="caption-attachment-767" class="wp-caption-text">
    Adding the remote attribute to the property of the model class
  </p>
</div>

That&#8217;s it. When you enter a name, you will get an error message as soon as you have finished entering Wolfgang. If you change one letter, the error message will disappear.

<div id="attachment_768" style="width: 311px" class="wp-caption aligncenter">
  <a href="/wp-content/uploads/2018/02/Remote-validation-of-the-customer-name.jpg"><img aria-describedby="caption-attachment-768" loading="lazy" class="size-full wp-image-768" src="/wp-content/uploads/2018/02/Remote-validation-of-the-customer-name.jpg" alt="Remote validation of the customer name" width="301" height="240" /></a>
  
  <p id="caption-attachment-768" class="wp-caption-text">
    Remote validation of the customer name
  </p>
</div>

## Conclusion

In this post, I showed different approaches on how to perform model validation. Model validation is necessary to ensure data integrity and check whether the user entered the valid data.

For more details about model validation, I highly recommend the books <a href="http://amzn.to/2mgRbTy" target="_blank" rel="noopener noreferrer">Pro ASP.NET MVC 5</a> and <a href="http://amzn.to/2mfQ0nA" target="_blank" rel="noopener noreferrer">Pro ASP.NET MVC 5 Plattform</a>.

You can find the source code on <a href="https://github.com/WolfgangOfner/MVC-ModelValidation" target="_blank" rel="noopener noreferrer">GitHub</a>.