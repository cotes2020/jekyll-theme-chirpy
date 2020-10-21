---
title: Azure Static Web Apps
date: 2020-05-30T17:46:33+02:00
author: Wolfgang Ofner
categories: [Cloud, Programming]
tags: [Azure Function, 'C#', Github Action, React, Static Web Apps]
---
Azure Static Web Apps were announced at the Build conference this year and they allow you to host your static websites for free. You can use HTML, plain Javascript of front-end frameworks like Angular or React. The website retrieves data, if needed, from Azure Functions. A great feature of Azure Static Web Apps is that you don&#8217;t have to configure your CI/CD pipeline. This is done for you by Azure and Github using Git Actions.

Today, I want to show how to create a React front-end and how to retrieve data from an Azure Function. You can find the code of the React demo <a href="https://github.com/WolfgangOfner/React-Azure-Static-Web-App/tree/master" target="_blank" rel="noopener noreferrer">here</a> and the code of the Azure Function demo <a href="https://github.com/WolfgangOfner/Azure-Function-StaticWebApp" target="_blank" rel="noopener noreferrer">here</a>.

## Azure Functions

Azure Function is an event-driven serverless feature which can be used to automate workloads or connect different parts of an application. They can be flexible scaled and they offer different triggers, for example, run some code when a message is written to a queue when a web API call was made or many more. You can use different programming languages like Javascript, Node.js, or .Net Core.

Azure Functions offer a consumption plan, where you pay what you used, a premium plan, or an App Service plan. I like the consumption plan because it gives you 400,000 GB/s for free every month. This should be more than enough for small projects.

### Creating an Azure Function in the Azure Portal

To create an Azure Function in the Azure portal, search for Function App, and click on Create. In the Basics tab of the creation process, enter your basic information like the resource group, the region, and the runtime.

<div id="attachment_2123" style="width: 710px" class="wp-caption aligncenter">
  <a href="/wp-content/uploads/2020/05/Create-an-Azure-Function.jpg"><img aria-describedby="caption-attachment-2123" loading="lazy" class="wp-image-2123" src="/wp-content/uploads/2020/05/Create-an-Azure-Function.jpg" alt="Create an Azure Function" width="700" height="608" /></a>
  
  <p id="caption-attachment-2123" class="wp-caption-text">
    Create an Azure Function
  </p>
</div>

In the next screenshot, you can see a summary of all my provided information. Note that I selected Windows as my operating system. First I wanted to use Linux but for whatever reason, I couldn&#8217;t deploy to the Linux Azure Function.

<div id="attachment_2124" style="width: 481px" class="wp-caption aligncenter">
  <a href="/wp-content/uploads/2020/05/Reviewing-the-Azure-Function-before-creating-it.jpg"><img aria-describedby="caption-attachment-2124" loading="lazy" class="wp-image-2124" src="/wp-content/uploads/2020/05/Reviewing-the-Azure-Function-before-creating-it.jpg" alt="Reviewing the Azure Function before creating it" width="471" height="700" /></a>
  
  <p id="caption-attachment-2124" class="wp-caption-text">
    Reviewing the Azure Function before creating it
  </p>
</div>

Click on Create and the Azure Function deployment starts.

### Creating an Azure Function using Visual Studio

To create my Azure Function, I am using Visual Studio 2019. You can also use Visual Studio Code. If you want to use Javascript for your Azure Function, you even have to use VS Code since Visual Studio only supports C#. Create a new project and select the Azure Functions template.

<div id="attachment_2121" style="width: 710px" class="wp-caption aligncenter">
  <a href="/wp-content/uploads/2020/05/Select-the-Azure-Function-template.jpg"><img aria-describedby="caption-attachment-2121" loading="lazy" class="wp-image-2121" src="/wp-content/uploads/2020/05/Select-the-Azure-Function-template.jpg" alt="Select the Azure Function template" width="700" height="485" /></a>
  
  <p id="caption-attachment-2121" class="wp-caption-text">
    Select the Azure Function template
  </p>
</div>

In the next window select HTTP trigger and set the Authorization level to Anonymous. This configuration starts the code execution every time an HTTP request is received.

<div id="attachment_2122" style="width: 710px" class="wp-caption aligncenter">
  <a href="/wp-content/uploads/2020/05/Select-Http-trigger-and-set-the-Authorization-level-to-Anonymous.jpg"><img aria-describedby="caption-attachment-2122" loading="lazy" class="wp-image-2122" src="/wp-content/uploads/2020/05/Select-Http-trigger-and-set-the-Authorization-level-to-Anonymous.jpg" alt="Select Http trigger and set the Authorization level to Anonymous" width="700" height="485" /></a>
  
  <p id="caption-attachment-2122" class="wp-caption-text">
    Select HTTP trigger and set the Authorization level to Anonymous
  </p>
</div>

After the Azure Function is created, I change the code so it returns a JSON list of products:

[code language=&#8221;csharp&#8221;]  
public static class Function1  
{  
[FunctionName("Function1")]  
public static async Task<IActionResult> Run(  
[HttpTrigger(AuthorizationLevel.Anonymous, "get", "post", Route = null)]  
HttpRequest req,  
ILogger log)  
{  
var products = new List<Product>  
{  
new Product  
{  
Name = "Phone",  
Price = 999.90m,  
Description = "This is the description of the phone"  
},  
new Product  
{  
Name = "Book",  
Price = 99.90m,  
Description = "The best book you will ever read"  
},  
new Product  
{  
Name = "TV",  
Price = 15.49m,  
Description = "Here you can see an awesome TV"  
}  
};

return new OkObjectResult(JsonConvert.SerializeObject(products));  
}  
}  
[/code]

The Product class has the following properties:

[code language=&#8221;csharp&#8221;]  
public class Product  
{  
public string Name { get; set; }

public decimal Price { get; set; }

public string Description { get; set; }  
}  
[/code]

Start the application and a console window will appear telling you the URL of your function. Enter this URL into your browser and you should see the JSON list displayed.

<div id="attachment_2125" style="width: 710px" class="wp-caption aligncenter">
  <a href="/wp-content/uploads/2020/05/Testing-the-Azure-Function-locally.jpg"><img aria-describedby="caption-attachment-2125" loading="lazy" class="wp-image-2125" src="/wp-content/uploads/2020/05/Testing-the-Azure-Function-locally.jpg" alt="Testing the Azure Function locally" width="700" height="412" /></a>
  
  <p id="caption-attachment-2125" class="wp-caption-text">
    Testing the Azure Function locally
  </p>
</div>

### Deployment of the Azure Function

You can deploy the Azure Function directly from within Visual Studio. You can deploy to an existing Azure Function or even create a new one. Since I already created one, I will deploy it to this one. To make things even easier, I will download the publish profile from the Function in the Azure portal by clicking on Get publish profile on the Overview tab.

<div id="attachment_2126" style="width: 710px" class="wp-caption aligncenter">
  <a href="/wp-content/uploads/2020/05/Download-the-publish-profile-of-the-Azure-Function.jpg"><img aria-describedby="caption-attachment-2126" loading="lazy" class="wp-image-2126" src="/wp-content/uploads/2020/05/Download-the-publish-profile-of-the-Azure-Function.jpg" alt="Download the publish profile of the Azure Function" width="700" height="194" /></a>
  
  <p id="caption-attachment-2126" class="wp-caption-text">
    Download the publish profile of the Azure Function
  </p>
</div>

After you downloaded the publish profile, in Visual Studio right-click on your project and select Publish. This opens a new window, select Import Profile and then select your previously downloaded publish profile.

<div id="attachment_2127" style="width: 710px" class="wp-caption aligncenter">
  <a href="/wp-content/uploads/2020/05/Import-the-downloaded-publish-profile.jpg"><img aria-describedby="caption-attachment-2127" loading="lazy" class="wp-image-2127" src="/wp-content/uploads/2020/05/Import-the-downloaded-publish-profile.jpg" alt="Import the downloaded publish profile" width="700" height="491" /></a>
  
  <p id="caption-attachment-2127" class="wp-caption-text">
    Import the downloaded publish profile
  </p>
</div>

After the publish profile is imported, click on Publish and the Azure Function will be published to Azure.

### Configuring and Testing of the Azure Function

In the Azure portal, click on the Functions tab of your Azure Function. There you will see your previously deployed function.

<div id="attachment_2128" style="width: 562px" class="wp-caption aligncenter">
  <a href="/wp-content/uploads/2020/05/The-published-function-appears-in-the-Azure-Portal-under-Functions.jpg"><img aria-describedby="caption-attachment-2128" loading="lazy" class="size-full wp-image-2128" src="/wp-content/uploads/2020/05/The-published-function-appears-in-the-Azure-Portal-under-Functions.jpg" alt="The published function appears in the Azure Portal under Functions" width="552" height="453" /></a>
  
  <p id="caption-attachment-2128" class="wp-caption-text">
    The published function appears in the Azure Portal under Functions
  </p>
</div>

Click on the Function (Function1 in my case) and then click on Test. This opens a new panel where you can send a request to your function to test its functionality.

<div id="attachment_2130" style="width: 710px" class="wp-caption aligncenter">
  <a href="/wp-content/uploads/2020/05/Call-the-Azure-Function-to-test-its-functionality.jpg"><img aria-describedby="caption-attachment-2130" loading="lazy" class="wp-image-2130" src="/wp-content/uploads/2020/05/Call-the-Azure-Function-to-test-its-functionality.jpg" alt="Call the Azure Function to test its functionality" width="700" height="477" /></a>
  
  <p id="caption-attachment-2130" class="wp-caption-text">
    Call the Azure Function to test its functionality
  </p>
</div>

Click on Run and the JSON list of your products should be displayed in the Output tab.

<div id="attachment_2129" style="width: 710px" class="wp-caption aligncenter">
  <a href="/wp-content/uploads/2020/05/The-function-returns-the-JSON-with-products.jpg"><img aria-describedby="caption-attachment-2129" loading="lazy" class="wp-image-2129" src="/wp-content/uploads/2020/05/The-function-returns-the-JSON-with-products.jpg" alt="The function returns the JSON with products" width="700" height="178" /></a>
  
  <p id="caption-attachment-2129" class="wp-caption-text">
    The function returns the JSON with products
  </p>
</div>

Next call the function from your browser. The URL is <YourAzureFunctionName>.azurewebsites.net/api/YourFunction. Enter the URL in your browser and you will see the JSON displayed.

<div id="attachment_2131" style="width: 710px" class="wp-caption aligncenter">
  <a href="/wp-content/uploads/2020/05/Test-the-function-call-in-the-browser.jpg"><img aria-describedby="caption-attachment-2131" loading="lazy" class="wp-image-2131" src="/wp-content/uploads/2020/05/Test-the-function-call-in-the-browser.jpg" alt="Test the function call in the browser" width="700" height="84" /></a>
  
  <p id="caption-attachment-2131" class="wp-caption-text">
    Test the function call in the browser
  </p>
</div>

If we create the React app now and try to call the Azure Function, it won&#8217;t work. The reason why it won&#8217;t work is that CORS is not configured and therefore the request will be blocked. The configure CORS, open then the CORS tab, and enter http://localhost:3000. This will be the URL of the React app during the development.

<div id="attachment_2133" style="width: 489px" class="wp-caption aligncenter">
  <a href="/wp-content/uploads/2020/05/Configure-CORS.jpg"><img aria-describedby="caption-attachment-2133" loading="lazy" class="size-full wp-image-2133" src="/wp-content/uploads/2020/05/Configure-CORS.jpg" alt="Configure CORS" width="479" height="540" /></a>
  
  <p id="caption-attachment-2133" class="wp-caption-text">
    Configure CORS
  </p>
</div>

I also tried * to allow all requests but it didn&#8217;t work for me.

## Create the React App which will be deployed using Static Web Apps

You should have basic knowledge of React. If you are new to React, take a look at the <a href="https://reactjs.org/docs/create-a-new-react-app.html" target="_blank" rel="noopener noreferrer">documentation</a> to install node.js and npm.

Open a new Powershell window and create a new react app with the following command:

[code language=&#8221;powershell&#8221;]  
npx create-react-app react-static-web-app  
[/code]

This will create a react project, named react-static-web-app. Go inside the project folder in Powershell and open Visual Studio Code with the following code:

[code language=&#8221;powershell&#8221;]  
code .  
[/code]

I will change the application to call my Azure Function and then display the returned list with Bootstrap cards. First, I create a new folder, called components, and create a new file inside this folder called products.js. Then I add the following code to the new file:

[code language=&#8221;javascript&#8221;]  
import React from &#8216;react&#8217;

const Products = ({ products }) => {  
return (  
<div>  
{products.map((product) => (  
<div class="card">  
<div class="card-body">  
<h5 class="card-title">{product.Name}</h5>  
<h6 class="card-subtitle mb-2 text-muted">{product.Price}</h6>  
<p class="card-text">{product.Description}</p>  
</div>  
</div>  
))}  
</div>  
)  
};

export default Products  
[/code]

This method takes a list of products and displays every item. The next step is to implement the Azure Function call in the App.js file.

[code language=&#8221;javascript&#8221;]  
import React, { Component } from &#8216;react&#8217;;  
import Products from &#8216;./components/products&#8217;;

class App extends Component {  
render() {  
return (  
<Products products={this.state.products} />  
)  
}

state = {  
products: []  
};

componentDidMount() {  
fetch(&#8216;https://staticwebappwolfgang.azurewebsites.net/api/Function1&#8217;)  
.then(res => res.json())  
.then((data) => {  
this.setState({ products: data })  
})  
.catch(console.log)  
}  
}

export default App;  
[/code]

Lastly, I add the Bootstrap css file in the index.html file which is located in the Public folder.

[code language=&#8221;html&#8221;]  
<link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.3.1/css/bootstrap.min.css" integrity="sha384-ggOyR0iXCbMQv3Xipma34MD+dH/1fQ784/j6cY/iJTQUOhcWr7x9JvoRxT2MZw1T" crossorigin="anonymous">  
[/code]

Open a new terminal in VS Code and start the application with:

[code language=&#8221;powershell&#8221;]  
npm start  
[/code]

This automatically opens your browser and should display your product list.

<div id="attachment_2132" style="width: 316px" class="wp-caption aligncenter">
  <a href="/wp-content/uploads/2020/05/Testing-the-react-app-with-the-data-from-the-Azure-Function.jpg"><img aria-describedby="caption-attachment-2132" loading="lazy" class="size-full wp-image-2132" src="/wp-content/uploads/2020/05/Testing-the-react-app-with-the-data-from-the-Azure-Function.jpg" alt="Testing the react app with the data from the Azure Function" width="306" height="448" /></a>
  
  <p id="caption-attachment-2132" class="wp-caption-text">
    Testing the React app with the data from the Azure Function
  </p>
</div>

Check in the React app into Github and let&#8217;s deploy it with Static Web Apps.

## Create Static Web Apps

In the Azure portal, search for Static Web App and click on Create to start the deployment process.

<div id="attachment_2134" style="width: 438px" class="wp-caption aligncenter">
  <a href="/wp-content/uploads/2020/05/Create-a-Static-Web-App.jpg"><img aria-describedby="caption-attachment-2134" loading="lazy" class="size-full wp-image-2134" src="/wp-content/uploads/2020/05/Create-a-Static-Web-App.jpg" alt="Create a Static Web Apps" width="428" height="197" /></a>
  
  <p id="caption-attachment-2134" class="wp-caption-text">
    Create a Static Web App
  </p>
</div>

On the Basics tab enter a name, select a region, and select your Github repository and branch. Note the region is only the initial region for the deployment. After the deployment, Static Web Apps deploys your application globally to give your users the best possible user experience.

<div id="attachment_2135" style="width: 710px" class="wp-caption aligncenter">
  <a href="/wp-content/uploads/2020/05/Configure-the-Static-Web-App.jpg"><img aria-describedby="caption-attachment-2135" loading="lazy" class="wp-image-2135" src="/wp-content/uploads/2020/05/Configure-the-Static-Web-App.jpg" alt="Configure the Static Web Apps" width="700" height="680" /></a>
  
  <p id="caption-attachment-2135" class="wp-caption-text">
    Configure the Static Web App
  </p>
</div>

On the Build tab, I removed the Api location because I have my Azure Function already deployed. The Static Web Apps are still in preview and I couldn&#8217;t deploy a C# Azure Function because the build failed. The build demanded that the direction of the function must be set in the function.json file. With C#, you can&#8217;t edit the function.json file because it is created during the build. I was able to deploy a Javascript Azure Function using Static Web Apps though.

<div id="attachment_2136" style="width: 321px" class="wp-caption aligncenter">
  <a href="/wp-content/uploads/2020/05/Configure-the-location-of-the-app-and-api.jpg"><img aria-describedby="caption-attachment-2136" loading="lazy" class="size-full wp-image-2136" src="/wp-content/uploads/2020/05/Configure-the-location-of-the-app-and-api.jpg" alt="Configure the location of the app and api" width="311" height="293" /></a>
  
  <p id="caption-attachment-2136" class="wp-caption-text">
    Configure the location of the app and api
  </p>
</div>

On the following screenshot, you can see all my entered information. Click on Create and the deployment process starts.

<div id="attachment_2137" style="width: 650px" class="wp-caption aligncenter">
  <a href="/wp-content/uploads/2020/05/Overview-of-the-Static-Web-App-before-creation.jpg"><img aria-describedby="caption-attachment-2137" loading="lazy" class="size-full wp-image-2137" src="/wp-content/uploads/2020/05/Overview-of-the-Static-Web-App-before-creation.jpg" alt="Overview of the Static Web Apps before creation" width="640" height="495" /></a>
  
  <p id="caption-attachment-2137" class="wp-caption-text">
    Overview of the Static Web App before creation
  </p>
</div>

After the deployment is finished, you can see the URL of the Static Web App and also a link to the created Github Action.

<div id="attachment_2140" style="width: 710px" class="wp-caption aligncenter">
  <a href="/wp-content/uploads/2020/05/Overview-of-the-published-Static-Web-App.jpg"><img aria-describedby="caption-attachment-2140" loading="lazy" class="wp-image-2140" src="/wp-content/uploads/2020/05/Overview-of-the-published-Static-Web-App.jpg" alt="Overview of the published Static Web Apps" width="700" height="141" /></a>
  
  <p id="caption-attachment-2140" class="wp-caption-text">
    Overview of the published Static Web App
  </p>
</div>

Click on GitHub Action runs and you will be redirected to the Action inside your Github repository.

<div id="attachment_2138" style="width: 710px" class="wp-caption aligncenter">
  <a href="/wp-content/uploads/2020/05/The-Git-Action-is-building-your-application.jpg"><img aria-describedby="caption-attachment-2138" loading="lazy" class="wp-image-2138" src="/wp-content/uploads/2020/05/The-Git-Action-is-building-your-application.jpg" alt="The Git Action is building your application" width="700" height="163" /></a>
  
  <p id="caption-attachment-2138" class="wp-caption-text">
    The Git Action is building your application
  </p>
</div>

Click on the CI/CD pipeline and you can see more detailed information.

<div id="attachment_2139" style="width: 710px" class="wp-caption aligncenter">
  <a href="/wp-content/uploads/2020/05/The-Git-Action-run-finished.jpg"><img aria-describedby="caption-attachment-2139" loading="lazy" class="wp-image-2139" src="/wp-content/uploads/2020/05/The-Git-Action-run-finished.jpg" alt="The Git Action run finished" width="700" height="463" /></a>
  
  <p id="caption-attachment-2139" class="wp-caption-text">
    The Git Action run finished
  </p>
</div>

Before you can test your deployed React app, you have to enter its URL in the CORS settings of your Azure Function. You can find the URL on the overview tab of your Static Web Apps.

<div id="attachment_2141" style="width: 650px" class="wp-caption aligncenter">
  <a href="/wp-content/uploads/2020/05/Add-the-URL-of-the-Static-Web-App-to-the-CORS-Setting-of-the-Azure-Function.jpg"><img aria-describedby="caption-attachment-2141" loading="lazy" class="wp-image-2141 size-full" src="/wp-content/uploads/2020/05/Add-the-URL-of-the-Static-Web-App-to-the-CORS-Setting-of-the-Azure-Function.jpg" alt="Add the URL of the Static Web Apps to the CORS Setting of the Azure Function" width="640" height="561" /></a>
  
  <p id="caption-attachment-2141" class="wp-caption-text">
    Add the URL of the Static Web App to the CORS Setting of the Azure Function
  </p>
</div>

After you entered the URL in the CORS settings, call it in your browser and you should see your React app displaying the list of products.

<div id="attachment_2142" style="width: 466px" class="wp-caption aligncenter">
  <a href="/wp-content/uploads/2020/05/Testing-the-Static-Web-App-with-the-Azure-Function.jpg"><img aria-describedby="caption-attachment-2142" loading="lazy" class="wp-image-2142 size-full" src="/wp-content/uploads/2020/05/Testing-the-Static-Web-App-with-the-Azure-Function.jpg" alt="Testing the Static Web Apps with the Azure Function" width="456" height="439" /></a>
  
  <p id="caption-attachment-2142" class="wp-caption-text">
    Testing the Static Web App with the Azure Function
  </p>
</div>

## Conclusion

Static Web Apps are a great new feature to quickly deploy your static website and host it globally. The feature is still new and in preview therefore it is no surprise that everything is not working perfectly yet. Once all problems are solved, I think it will be a great tool for a simple website, especially since it is free.

In my next post, I will show you how to host your website for free and also extend the Azure Function to be able to read data from a database.

You can find the code of the React demo <a href="https://github.com/WolfgangOfner/React-Azure-Static-Web-App/tree/master" target="_blank" rel="noopener noreferrer">here</a> and the code of the Azure Function demo <a href="https://github.com/WolfgangOfner/Azure-Function-StaticWebApp" target="_blank" rel="noopener noreferrer">here</a>.