---
title: Azure Static Web App with Blazor
date: 2020-10-04T16:51:05+02:00
author: Wolfgang Ofner
categories: [Cloud, DevOps]
tags: [.net core, Azure Function, Azure Static Web App, Blazor, 'C#']
---
Last week at Ignite Microsoft announced that the preview of Azure Static Web App now also supports Blazor WebAssembly. In this post, I will show how to create a Blazor client-side (WebAssembly) application and an Azure Function to retrieve some data. Then Azure Static Web App will automatically deploy the Blazor app and the Function.

You can find the code of the demo on <a href="https://github.com/WolfgangOfner/BlazorAzureStaticWebsite" target="_blank" rel="noopener noreferrer">Github</a>.

## Create a Blazor WebAssembly Application

To create a Blazor WebAssembly application, you need Visual Studio 2019 and the <a href="https://dotnet.microsoft.com/download/dotnet-core/3.1" target="_blank" rel="noopener noreferrer">.NET Core SDK 3.1.300 or later</a>.

In Visual Studio create a new project and select Blazor App as your template.

<div id="attachment_2405" style="width: 710px" class="wp-caption aligncenter">
  <a href="/wp-content/uploads/2020/10/Select-Blazor-App-as-the-template.jpg"><img aria-describedby="caption-attachment-2405" loading="lazy" class="wp-image-2405" src="/wp-content/uploads/2020/10/Select-Blazor-App-as-the-template.jpg" alt="Select Blazor App as the template" width="700" height="485" /></a>
  
  <p id="caption-attachment-2405" class="wp-caption-text">
    Select Blazor App as the template
  </p>
</div>

Enter a name and then select Blazor WebAssembly App to create a client-side Blazor application.

<div id="attachment_2406" style="width: 710px" class="wp-caption aligncenter">
  <a href="/wp-content/uploads/2020/10/Create-a-Blazor-WebAssembly-application.jpg"><img aria-describedby="caption-attachment-2406" loading="lazy" class="wp-image-2406" src="/wp-content/uploads/2020/10/Create-a-Blazor-WebAssembly-application.jpg" alt="Create a Blazor WebAssembly application" width="700" height="485" /></a>
  
  <p id="caption-attachment-2406" class="wp-caption-text">
    Create a Blazor WebAssembly application
  </p>
</div>

Click Create and Visual Studio creates a sample Blazor WebAssembly application.

## Create an Azure Function in Visual Studio

To create an Azure Function, you need the <a href="https://docs.microsoft.com/en-gb/azure/azure-functions/functions-run-local?tabs=linux%2Ccsharp%2Cbash" target="_blank" rel="noopener noreferrer">Azure Functions Core Tools</a>.

In your previously created solution, right-click on the solution file, click add a new project, and select Azure Functions as the template.

<div id="attachment_2407" style="width: 710px" class="wp-caption aligncenter">
  <a href="/wp-content/uploads/2020/10/Select-Azure-Function-as-the-template.jpg"><img aria-describedby="caption-attachment-2407" loading="lazy" class="wp-image-2407" src="/wp-content/uploads/2020/10/Select-Azure-Function-as-the-template.jpg" alt="Select Azure Function as the template" width="700" height="485" /></a>
  
  <p id="caption-attachment-2407" class="wp-caption-text">
    Select Azure Function as the template
  </p>
</div>

Enter a name and then select Http trigger to trigger the function via HTTP calls.

<div id="attachment_2408" style="width: 710px" class="wp-caption aligncenter">
  <a href="/wp-content/uploads/2020/10/Select-HTTP-Trigger-for-the-Azure-Function.jpg"><img aria-describedby="caption-attachment-2408" loading="lazy" class="wp-image-2408" src="/wp-content/uploads/2020/10/Select-HTTP-Trigger-for-the-Azure-Function.jpg" alt="Select HTTP Trigger for the Azure Function" width="700" height="485" /></a>
  
  <p id="caption-attachment-2408" class="wp-caption-text">
    Select HTTP Trigger for the Azure Function
  </p>
</div>

## Get Data from the Azure Function

To call the Azure Function and get some data about products, I edit the already existing FetchData page. I change the code block to call the Azure Function and then cast the result into a product array. Then, I will loop over the array in the HTML code and display all elements in a table.

[code language=&#8221;CSharp&#8221;]  
@code {  
private Product[] products;

protected override async Task OnInitializedAsync()  
{  
try  
{  
products = await Http.GetFromJsonAsync<Product[]>("/Api/Product");  
}  
catch (Exception ex)  
{  
Console.WriteLine(ex.ToString());  
}  
}  
}  
[/code]

In the Function, I create a new list with three products and return it. Note that the FunctionName, &#8220;Product&#8221;, is the same as in the call in the code block.

[code language=&#8221;CSharp&#8221;]  
[FunctionName("Product")]  
public static IActionResult Run([HttpTrigger(AuthorizationLevel.Anonymous, "get", Route = null)] HttpRequest req, ILogger log)  
{  
var products = new List<Product>  
{  
new Product  
{  
Name = "Book",  
Description = "A great book",  
Price = 9.99m  
},  
new Product  
{  
Name = "Phone",  
Description = "A good phone",  
Price = 149.99m  
},  
new Product  
{  
Name = "Car",  
Description = "A bad car ",  
Price = 999.99m  
}  
};

return new OkObjectResult(products);  
}  
[/code]

The Product class is in a class library so both projects can reference it.

To test that everything is working as expected, start the Blazor app and the Azure Function. Click on the Fetch data and you will see three products displayed. In the console application of the Azure Function, you can see its URL and some log information. To test only the Azure Function, you could also call the displayed URL. This will return the products as JSON.

<div id="attachment_2409" style="width: 710px" class="wp-caption aligncenter">
  <a href="/wp-content/uploads/2020/10/Test-the-implementation-locally.jpg"><img aria-describedby="caption-attachment-2409" loading="lazy" class="wp-image-2409" src="/wp-content/uploads/2020/10/Test-the-implementation-locally.jpg" alt="Test the implementation locally" width="700" height="356" /></a>
  
  <p id="caption-attachment-2409" class="wp-caption-text">
    Test the implementation locally
  </p>
</div>

## Create an Azure Static Web App

Go to the Azure Portal, click on New, search for Static Webb App, and click on Create.

<div id="attachment_2410" style="width: 564px" class="wp-caption aligncenter">
  <a href="/wp-content/uploads/2020/10/Create-the-Static-Web-App.jpg"><img aria-describedby="caption-attachment-2410" loading="lazy" class="size-full wp-image-2410" src="/wp-content/uploads/2020/10/Create-the-Static-Web-App.jpg" alt="Create the Static Web App" width="554" height="221" /></a>
  
  <p id="caption-attachment-2410" class="wp-caption-text">
    Create the Static Web App
  </p>
</div>

Enter a name, location, and your Github repository. Then select Blazor as Build Presets and enter the location of your Blazor project and of your Azure Function project. Click Review + create and your application will be deployed.

<div id="attachment_2411" style="width: 604px" class="wp-caption aligncenter">
  <a href="/wp-content/uploads/2020/10/Configure-the-Static-Web-App.jpg"><img aria-describedby="caption-attachment-2411" loading="lazy" class="wp-image-2411" src="/wp-content/uploads/2020/10/Configure-the-Static-Web-App.jpg" alt="Configure the Static Web App" width="594" height="700" /></a>
  
  <p id="caption-attachment-2411" class="wp-caption-text">
    Configure the Static Web App
  </p>
</div>

The deployment takes only a moment. After it is finished, click on the URL that is displayed in your Static Web App.

<div id="attachment_2412" style="width: 710px" class="wp-caption aligncenter">
  <a href="/wp-content/uploads/2020/10/The-Static-Web-App-got-deployed.jpg"><img aria-describedby="caption-attachment-2412" loading="lazy" class="wp-image-2412" src="/wp-content/uploads/2020/10/The-Static-Web-App-got-deployed.jpg" alt="The Static Web App got deployed" width="700" height="121" /></a>
  
  <p id="caption-attachment-2412" class="wp-caption-text">
    The Static Web App got deployed
  </p>
</div>

This opens your application in a new window and when you click on Fetch data, you will see your products.

<div id="attachment_2415" style="width: 710px" class="wp-caption aligncenter">
  <a href="/wp-content/uploads/2020/10/Testing-the-deployed-Blazor-application.jpg"><img aria-describedby="caption-attachment-2415" loading="lazy" class="wp-image-2415" src="/wp-content/uploads/2020/10/Testing-the-deployed-Blazor-application.jpg" alt="Testing the deployed Blazor application" width="700" height="323" /></a>
  
  <p id="caption-attachment-2415" class="wp-caption-text">
    Testing the deployed Blazor application
  </p>
</div>

Below the URL, you see a link to the Github Actions which are responsible for the deployment. Alternatively, you could also open your Github repository and click on Actions there. When you open the Github Actions, you will see the deployment from the Azure Static Web App.

<div id="attachment_2414" style="width: 710px" class="wp-caption aligncenter">
  <a href="/wp-content/uploads/2020/10/Github-Actions-deployed-the-Blazor-application.jpg"><img aria-describedby="caption-attachment-2414" loading="lazy" class="wp-image-2414" src="/wp-content/uploads/2020/10/Github-Actions-deployed-the-Blazor-application.jpg" alt="Github Actions deployed the Blazor application" width="700" height="246" /></a>
  
  <p id="caption-attachment-2414" class="wp-caption-text">
    Github Actions deployed the Blazor application
  </p>
</div>

## Conclusion

<a href="/azure-static-web-apps/" target="_blank" rel="noopener noreferrer">I already wrote an article about Azure Static Web Apps</a> when they got announced in May. Back then I had some problems and the functionality was very limited. With these improvements, especially with the support of Blazor, Microsoft is going definitively in the right direction. To have a realistic application, the Azure Function could get some data from a database. Another great benefit is that it is still free and you only pay for the Azure Functions (the first 400,000 seconds are free).

You can find the code of the demo on <a href="https://github.com/WolfgangOfner/BlazorAzureStaticWebsite" target="_blank" rel="noopener noreferrer">Github</a>.