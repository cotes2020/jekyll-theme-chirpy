---
title: Document your Microservice with Swagger
date: 2020-04-15T17:37:49+02:00
author: Wolfgang Ofner
categories: [ASP.NET]
tags: [.net core 3.1, 'C#', CQRS, docker, docker-compose, MediatR, microservice, RabbitMQ, Swagger]
---
Swagger is an open-source toolset that can be easily integrated into your solution and which helps you to document and test your APIs. It is so simple that even none technical people can use it. <a href="/programming-microservices-net-core-3-1/" target="_blank" rel="noopener noreferrer">In my last post</a>, I created two Microservices and today I will explain how I integrated Swagger.

## What is Swagger?

Swagger is an open-source toolset from <a href="https://swagger.io/" target="_blank" rel="noopener noreferrer">Smartbear</a> and is based on the OpenAPI specificiation. For .NET Core, you can install Swagger easily as Nuget package and configure it in the Startup class of your solution. In simple words, Swagger wraps the XML comments of your API in a nice looking functional GUI which shows your the available actions and models and also lets you send requests to the actions. You can even attach authentication objects like a JWT.

## Install Swagger

You can find the code of  the finished demo on <a href="https://github.com/WolfgangOfner/MicroserviceDemo" target="_blank" rel="noopener noreferrer">GitHub</a>.

To implement Swagger, I installed the Swashbuckle.AspNetCore NuGet package in the API project. Next, I added the path to the XML file which contains all the XML comments of the actions and models in the ConfigureServices method in the Startup class.

```csharp
services.AddSwaggerGen(c =>  
{  
    var xmlFile = $"{Assembly.GetExecutingAssembly().GetName().Name}.xml";  
    var xmlPath = Path.Combine(AppContext.BaseDirectory, xmlFile);  
    c.IncludeXmlComments(xmlPath);  
});  
```

The next step is to tell ASP .net core to use Swagger and its UI. You can add both in the Configure method of the Startup class. Additionally, I configured Swagger to load the the GUI when starting your solution.

```csharp  
app.UseSwagger();  
app.UseSwaggerUI(c =>  
{  
    c.SwaggerEndpoint("/swagger/v1/swagger.json", "Customer API V1");  
    c.RoutePrefix = string.Empty;  
});  
```

That&#8217;s all you have to configure in Swagger. Now I only have to make two adjustments to the starting project. First, I tell the project to create the XML file by opening the properties of the project. Go to the Build tab and check XML documentation file. It is important that you use All Configurations as Configuration in the dropdown on top All Configurations.

<div id="attachment_1889" style="width: 710px" class="wp-caption aligncenter">
  <a href="/assets/img/posts/2020/04/Create-the-XML-documentation-file.jpg"><img aria-describedby="caption-attachment-1889" loading="lazy" class="wp-image-1889" src="/assets/img/posts/2020/04/Create-the-XML-documentation-file.jpg" alt="Create the XML documentation file" width="700" height="682" /></a>
  
  <p id="caption-attachment-1889" class="wp-caption-text">
    Create the XML documentation file
  </p>
</div>

After you configured your project to create the XML documentation file, you will get warnings that an XML comment is missing on top of your actions. I find these warnings pretty annoying, so I suppress them by adding 1591 in the text box for Suppress warnings in the Build tab (see screenshot above).

The last step is to remove the launch URL from the launchSettings.json file. You can just remove the line, otherwise, the Swagger GUI won&#8217;t be loaded when the application is started and you have to call its URL.

That&#8217;s all you have to do to set up Swagger. Before I test it, I will add some XML comments to the actions, attributes to my model and some more configuration.

### Adding XML comments to API Actions

The XML comment on an action describes what the action does, what the parameters are, what it returns and what return codes it can produce. Usually, I have the opinion that the name of the method and parameter should describe themselves but in this case, we need a comment for the Swagger GUI. To create an XML comment write three / on top of an action. This will give you the template of the comment.

Additionally, I add the response codes and the ProducesResponseType attribute which will help users of the GUI to understand what return codes can be expected from the API.

```csharp  
/// <summary>  
/// Action to create a new customer in the database.  
/// </summary>  
/// <param name="createCustomerModel">Model to create a new customer</param>  
/// <returns>Returns the created customer</returns>  
/// <response code="200">Returned if the customer was created</response>  
/// <response code="400">Returned if the model couldn&#8217;t be parsed or the customer couldn&#8217;t be saved</response>  
/// <response code="422">Returned when the validation failed</response>  
[ProducesResponseType(StatusCodes.Status200OK)]  
[ProducesResponseType(StatusCodes.Status400BadRequest)]  
[ProducesResponseType(StatusCodes.Status422UnprocessableEntity)]  
[HttpPost]  
public async Task<ActionResult<Customer>> Customer([FromBody] CreateCustomerModel createCustomerModel)  
{  
```

### Adding Attributes to the Model

The Swagger UI for .NET Core also includes the models of your application. The UI shows which models are available, what properties they have including their data type and their attributes, e.g. if the property is required. To use this feature, you only have to add the attribute to the property of your models. Swagger creates everything out of the box by itself.

```csharp  
public class CreateCustomerModel  
{  
    [Required]  
    public string FirstName { get; set; }
    
    [Required]  
    public string LastName { get; set; }
    
    public DateTime? Birthday { get; set; }
    
    public int? Age { get; set; }  
}  
```

### Personalize the Swagger UI

Swagger is also easily extensible. You can load your own CSS, or change the headline or information displayed on top of the page. For now, I will add my contact information so developers or customers can contact me if they have a problem. To add your contact information use the SwaggerDoc extension and pass an OpenApiInfo object inside the AddSwaggerGen extension in the Startup class.

```csharp  
services.AddSwaggerGen(c =>  
{  
    c.SwaggerDoc("v1", new OpenApiInfo  
    {  
        Version = "v1",  
        Title = "Customer Api",  
        Description = "A simple API to create or update customers",  
        Contact = new OpenApiContact  
    {  
        Name = "Wolfgang Ofner",  
        Email = "Wolfgang@programmingwithwolfgang.com",  
        Url = new Uri("https://www.programmingwithwolfgang.com/")  
    }  
    });
    
    var xmlFile = $"{Assembly.GetExecutingAssembly().GetName().Name}.xml";  
    var xmlPath = Path.Combine(AppContext.BaseDirectory, xmlFile);  
    c.IncludeXmlComments(xmlPath);  
});  
```

## Testing Swagger

Everything is set up and when you start the application, you should see the Swagger UI.

<div id="attachment_1893" style="width: 565px" class="wp-caption aligncenter">
  <a href="/assets/img/posts/2020/04/Swagger-UI.jpg"><img aria-describedby="caption-attachment-1893" loading="lazy" class="wp-image-1893" src="/assets/img/posts/2020/04/Swagger-UI.jpg" alt="Swagger UI" width="555" height="700" /></a>
  
  <p id="caption-attachment-1893" class="wp-caption-text">
    Swagger UI
  </p>
</div>

On top of the page, you can see my headline and my contact information. Then you can see my two actions, even in different colors for the different HTTP verbs they use and then you can see my models. Next to the post action, you can see the XML comment I added to describe the method. The put action doesn&#8217;t have an XML comment yet, therefore no text is displayed.

When you click on an action, it opens and shows you information about the parameter, and also shows the responses which I added previously in the XML comment.

<div id="attachment_1894" style="width: 710px" class="wp-caption aligncenter">
  <a href="/assets/img/posts/2020/04/Swagger-UI-information-about-an-action.jpg"><img aria-describedby="caption-attachment-1894" loading="lazy" class="wp-image-1894" src="/assets/img/posts/2020/04/Swagger-UI-information-about-an-action.jpg" alt="Swagger UI information about an action" width="700" height="467" /></a>
  
  <p id="caption-attachment-1894" class="wp-caption-text">
    Swagger UI information about an action
  </p>
</div>

When you click on the Try it out button in the top right corner, Swagger will already create a request for you with all parameters. You can edit them and then send the request to the server and the UI will show you the reply.

<div id="attachment_1895" style="width: 710px" class="wp-caption aligncenter">
  <a href="/assets/img/posts/2020/04/Testing-an-aciton-of-the-API.jpg"><img aria-describedby="caption-attachment-1895" loading="lazy" class="wp-image-1895" src="/assets/img/posts/2020/04/Testing-an-aciton-of-the-API.jpg" alt="Testing an action of the API" width="700" height="491" /></a>
  
  <p id="caption-attachment-1895" class="wp-caption-text">
    Testing an action of the API
  </p>
</div>

After clicking on Try it out, you can define the format of your request on the top right. I leave the default application/json and also leave the created model as it is. When you click on Execute, the response and also the sent request and the request URL will be shown below. In my case, the response is a code 200 and a Customer JSON object. On the bottom right is a button where you can even download the result. This might be useful if you have a test document and want to attach the result to it.

## Conclusion

Today, I talked about Swagger, one of my favorite NuGet packages. Swagger can be used to document and test your application and make this information easily accessible even for none technical people. Swagger is also very easy to set up and can be extended and modify to fit your needs.

<a href="/cqrs-in-asp-net-core-3-1" target="_blank" rel="noopener noreferrer">In my next post</a>, I will explain the implementation of CQRS and how it can be used to split up your read and write operations.

Note: On October 11, I removed the Solution folder and moved the projects to the root level. Over the last months I made the experience that this makes it quite simpler to work with Dockerfiles and have automated builds and deployments.

You can find the code of  the finished demo on <a href="https://github.com/WolfgangOfner/MicroserviceDemo" target="_blank" rel="noopener noreferrer">GitHub</a>.