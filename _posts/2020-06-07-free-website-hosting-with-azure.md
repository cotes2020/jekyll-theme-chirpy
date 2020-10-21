---
title: Free Website Hosting with Azure
date: 2020-06-07T22:53:43+02:00
author: Wolfgang Ofner
categories: [Cloud]
tags: [Azure, Azure Functions, Azure Static Web Apps, 'C#', Cosmos DB, React]
---
<a href="https://www.programmingwithwolfgang.com/azure-static-web-apps/" target="_blank" rel="noopener noreferrer">Last week, I talked about hosting your static website with Azure Static Web Apps.</a> Today, I will extend this example using a free Cosmos DB for the website data and Azure Functions to retrieve them. This approach will give you free website hosting and global distribution of your website.  
You can find the demo code of the Azure Static Web App <a href="https://github.com/WolfgangOfner/React-Azure-Static-Web-App" target="_blank" rel="noopener noreferrer">here</a> and the code for the Azure Functions <a href="https://github.com/WolfgangOfner/AzureFunctions-CosmosDb" target="_blank" rel="noopener noreferrer">here</a>.

## Azure Cosmos DB

Cosmos DB is a high-end NoSQL database that offers incredible speed and global distribution. Cosmos DB is way too comprehensive to talk about all the features here. I am using it because it offers a free tier which should give you enough compute resources for a static website.

### Create a Free Tier Cosmos DB

In the Azure Portal search for Azure Cosmos DB, select it and click on Create or select Azure Cosmos DB from the left panel and then click on Create Azure Cosmos DB account.

<div id="attachment_2155" style="width: 710px" class="wp-caption aligncenter">
  <a href="/wp-content/uploads/2020/06/Create-a-new-Cosmos-DB.jpg"><img aria-describedby="caption-attachment-2155" loading="lazy" class="wp-image-2155" src="/wp-content/uploads/2020/06/Create-a-new-Cosmos-DB.jpg" alt="Create a new Cosmos DB" width="700" height="368" /></a>
  
  <p id="caption-attachment-2155" class="wp-caption-text">
    Create a new Cosmos DB
  </p>
</div>

On the next page, select a resource group and make sure that the Free Tier Discount is applied. After filling out all information click on Review + create.

<div id="attachment_2156" style="width: 632px" class="wp-caption aligncenter">
  <a href="/wp-content/uploads/2020/06/Set-up-the-free-tier-of-the-Cosmos-DB.jpg"><img aria-describedby="caption-attachment-2156" loading="lazy" class="wp-image-2156" src="/wp-content/uploads/2020/06/Set-up-the-free-tier-of-the-Cosmos-DB.jpg" alt="Set up the free tier of the Cosmos DB" width="622" height="700" /></a>
  
  <p id="caption-attachment-2156" class="wp-caption-text">
    Set up the free tier of the Cosmos DB
  </p>
</div>

The deployment will take around ten minutes.

### Add Data to the Cosmos Database

After the deployment is finished, navigate to the Data Explorer tab in your Cosmos DB account. Click on New Container and a new tab is opened on the right side. There enter a Database id, Container id, and Partition key and click OK.

<div id="attachment_2157" style="width: 710px" class="wp-caption aligncenter">
  <a href="/wp-content/uploads/2020/06/Create-a-new-catabase-and-container-in-the-Azure-Cosmos-DB.jpg"><img aria-describedby="caption-attachment-2157" loading="lazy" class="wp-image-2157" src="/wp-content/uploads/2020/06/Create-a-new-catabase-and-container-in-the-Azure-Cosmos-DB.jpg" alt="Create a new database and container in the Azure Cosmos DB" width="700" height="400" /></a>
  
  <p id="caption-attachment-2157" class="wp-caption-text">
    Create a new database and container in the Azure Cosmos DB
  </p>
</div>

Open the newly created database and the Products container and click on New Item. This opens an editor where you can add your products as JSON.

<div id="attachment_2158" style="width: 710px" class="wp-caption aligncenter">
  <a href="/wp-content/uploads/2020/06/Add-data-to-the-container.jpg"><img aria-describedby="caption-attachment-2158" loading="lazy" class="wp-image-2158" src="/wp-content/uploads/2020/06/Add-data-to-the-container.jpg" alt="Add data to the container for your Free Website Hosting " width="700" height="242" /></a>
  
  <p id="caption-attachment-2158" class="wp-caption-text">
    Add data to the container
  </p>
</div>

Again, Azure Cosmos DB is too big to go into any details in this post. For the free hosting of your website, it is only important to know that I added the data for the website into the database. The next step is to edit the Azure Function so it doesn&#8217;t return a static list but uses the Azure Cosmos DB instead.

## Using an Azure Function to read Data from Cosmos DB

I am re-using the Azure Function from my last post. If you don&#8217;t have any yet, create a new Azure Function with an HTTP trigger. To connect to the Cosmos DB, I am installing the Microsoft.Azure.Cosmos NuGet package and create a private variable with which I will access the data.

[code language=&#8221;CSharp&#8221;]  
private static Container _container;  
[/code]

Next, I create a method that will create a connection to the container in the database.

[code language=&#8221;CSharp&#8221;]  
private static async Task SetUpDatabaseConnection()  
{  
var cosmosClient = new CosmosClient("https://staticwebappdemocosmosdb.documents.azure.com:443",  
"qCEzM0xsroClwt54p7aICi3yBa0bWn4rAaiQNiIPp74LHcA7Wbm9D1iHszfaJ0icRcTwiW74KbMbn4WrMqnyfg==", new CosmosClientOptions());  
Database database = await cosmosClient.CreateDatabaseIfNotExistsAsync("StaticWebAppDatabase");  
_container = await database.CreateContainerIfNotExistsAsync("Products", "/Name", 400);  
}  
[/code]

To connect to the Azure Cosmos DB container, you have to enter your URI and primary key. You can find them in the Keys tab of your Cosmos DB account.

<div id="attachment_2159" style="width: 710px" class="wp-caption aligncenter">
  <a href="/wp-content/uploads/2020/06/Get-the-Uri-and-Primary-Key-of-the-Cosmos-DB.jpg"><img aria-describedby="caption-attachment-2159" loading="lazy" class="wp-image-2159" src="/wp-content/uploads/2020/06/Get-the-Uri-and-Primary-Key-of-the-Cosmos-DB.jpg" alt="Get the Uri and Primary Key of the Cosmos DB for your Free Website Hosting " width="700" height="233" /></a>
  
  <p id="caption-attachment-2159" class="wp-caption-text">
    Get the Uri and Primary Key of the Cosmos DB
  </p>
</div>

In the next method, I am creating an iterator that will return all my products. I add these products to a list and return the list. You can filter the query by providing a filter statement in the GetItemQueryIterator method.

[code language=&#8221;CSharp&#8221;]  
private static async Task<List<Product>> GetAllProducts()  
{  
var feedIterator = _container.GetItemQueryIterator<Product>();  
var products = new List<Product>();

while (feedIterator.HasMoreResults)  
{  
foreach (var item in await feedIterator.ReadNextAsync())  
{  
{  
products.Add(item);  
}  
}  
}

return products;  
}  
[/code]

In the Run method of the Azure Function, I am calling both methods and convert the list to a JSON object before returning it.

[code language=&#8221;CSharp&#8221;]  
[FunctionName("Function1")]  
public static async Task<IActionResult> Run(  
[HttpTrigger(AuthorizationLevel.Anonymous, "get", "post", Route = null)] HttpRequest req, ILogger log)  
{  
await SetUpDatabaseConnection();

return new OkObjectResult(JsonConvert.SerializeObject(await GetAllProducts()));  
}  
[/code]

I keep the Product class as it is.

[code language=&#8221;CSharp&#8221;]  
public class Product  
{  
public string Name { get; set; }

public decimal Price { get; set; }

public string Description { get; set; }  
}  
[/code]

Start the Azure Function, enter the URL displayed in the command line and you will see your previously entered data.

<div id="attachment_2160" style="width: 710px" class="wp-caption aligncenter">
  <a href="/wp-content/uploads/2020/06/Test-the-Azure-Function-locally.jpg"><img aria-describedby="caption-attachment-2160" loading="lazy" class="wp-image-2160" src="/wp-content/uploads/2020/06/Test-the-Azure-Function-locally.jpg" alt="Test the Azure Function locally" width="700" height="118" /></a>
  
  <p id="caption-attachment-2160" class="wp-caption-text">
    Test the Azure Function locally
  </p>
</div>

The last step is to deploy the Azure Function. In my last post, I already imported the publish profile. Since nothing has changed, I can right-click on my project, select Publish and then Publish again.

<div id="attachment_2161" style="width: 710px" class="wp-caption aligncenter">
  <a href="/wp-content/uploads/2020/06/Publish-the-Azure-Function.jpg"><img aria-describedby="caption-attachment-2161" loading="lazy" class="wp-image-2161" src="/wp-content/uploads/2020/06/Publish-the-Azure-Function.jpg" alt="Publish the Azure Function for your Free Website Hosting " width="700" height="233" /></a>
  
  <p id="caption-attachment-2161" class="wp-caption-text">
    Publish the Azure Function
  </p>
</div>

## Testing the Free Website Hosting Implementation

Open the URL of your Azure Static Web App and the data from the Cosmos DB will be displayed.

<div id="attachment_2162" style="width: 468px" class="wp-caption aligncenter">
  <a href="/wp-content/uploads/2020/06/The-data-from-the-database-is-displayed-in-the-React-app.jpg"><img aria-describedby="caption-attachment-2162" loading="lazy" class="size-full wp-image-2162" src="/wp-content/uploads/2020/06/The-data-from-the-database-is-displayed-in-the-React-app.jpg" alt="The data from the database is displayed in the React app" width="458" height="434" /></a>
  
  <p id="caption-attachment-2162" class="wp-caption-text">
    The data from the database is displayed in the React app
  </p>
</div>

## Conclusion

Today, I showed how to use Azure Cosmos DB, Azure Functions and Azure Static Web Apps to achieve free website hosting and also a global distribution of the website. You can find the demo code of the Azure Static Web App <a href="https://github.com/WolfgangOfner/React-Azure-Static-Web-App" target="_blank" rel="noopener noreferrer">here</a> and the code for the Azure Functions <a href="https://github.com/WolfgangOfner/AzureFunctions-CosmosDb" target="_blank" rel="noopener noreferrer">here</a>.

During Ignite in September 2020, Microsoft announced new features for Static Web Apps. From now on it is also possible to host Blazor apps and the connection with the Azure Function got improved a lot. You can find my post about it <a href="https://www.programmingwithwolfgang.com/azure-static-web-app-with-blazor/" target="_blank" rel="noopener noreferrer">here</a>.