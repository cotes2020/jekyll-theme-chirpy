---
title: Implement Azure Search
date: 2018-05-02T17:05:35+02:00
author: Wolfgang Ofner
categories: [Cloud]
tags: [70-532, Azure, Certification, Exam, learning]
---
Azure Search is a Platform as a Service offering that gives developers APIs needed to add search functionality to their applications. Primarily this means full-text search. The typical example is how Google and Bing search works. They don&#8217;t care what tense you use, it spell checks for you, and finds similar topics based on the search term. It also offers term highlighting and can ignore noise words, as well as many other search-related features. Applying these features inside your application can give your users a rich and comforting search experience.

## Create a service index

There are four different types of Azure Search accounts: Free, Basic, Standard, and High-density. The free tier only allows 50 MB of data storage and 10,000 documents. The higher you go with the pricing tier, the more documents you can index and the faster a search returns results. Computer resources for Azure Search are sold through Search Units (SUs). The basic level allows 3 search units whereas the high-density tier offers up to 36 SUs. Additionally, all of the paid pricing tiers offer load-balancing over three or more replicas.

### Create an Azure Search service

To create an Azure Search service, follow these steps:

  1. In the Azure portal click on +Create a resource, search for Azure Search and then click on Create.
  2. On the New Search Service blade, provide an URL, subscription, resource group, location, and pricing tier.
  3. Click Create.

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2018/05/Create-a-new-Azure-Search-Service.jpg"><img loading="lazy" src="/assets/img/posts/2018/05/Create-a-new-Azure-Search-Service.jpg" alt="Create a new Azure Search Service" /></a>
  
  <p>
    Create a new Azure Search Service
  </p>
</div>

### Scale an existing Azure Search Service

You can only scale an Azure Search Service with a paid pricing tier.

  1. In your Azure Search Service, click on Scale under the Settings menu.
  2. On the Scale blade, select your desired partitions and replicas.
  3. Click Save.

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2018/05/Scale-your-Azure-Search-Service.jpg"><img loading="lazy" src="/assets/img/posts/2018/05/Scale-your-Azure-Search-Service.jpg" alt="Scale your Azure Search Service" /></a>
  
  <p>
    Scale your Azure Search Service
  </p>
</div>

Replicas distribute workloads across multiple nodes. Partitions allow for scaling the document count as well as faster data ingestion by spanning your index over multiple Azure Search Units.

## Implement Azure Search using C#

To add data to Azure Search, create an index. An index contains documents used by Azure Search. For instance, a car dealer might have a document describing each car they sell. An index is similar to a SQL Server table and documents are similar to rows in those tables.

You can find the code for the following demo on <a href="https://github.com/WolfgangOfner/Azure-Search" target="_blank" rel="noopener">GitHub</a>. To add data to an index and search it using C#, follow these steps:

  1. Create a new C# console application with Visual Studio 2017.
  2. Install the Microsoft.Azure.Search NuGet package.
  3. Set up the connection to your Azure Search Service. Note that the serviceName is only the name you entered. Not a URI.

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2018/05/Set-up-the-connection-to-the-Azure-Search-Service-account.jpg"><img loading="lazy" src="/assets/img/posts/2018/05/Set-up-the-connection-to-the-Azure-Search-Service-account.jpg" alt="Set up the connection to the Azure Search Service account" /></a>
  
  <p>
    Set up the connection to the Azure Search Service account
  </p>
</div>

<ol start="4">
  <li>
    Add the System.ComponentModel.DataAnnotations reference to your project.
  </li>
  <li>
    Create a POCO for the cars you want to be searchable:
  </li>
</ol>

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2018/05/The-car-class-for-the-search.jpg"><img loading="lazy" src="/assets/img/posts/2018/05/The-car-class-for-the-search.jpg" alt="The car class for the search" /></a>
  
  <p>
    The car class for the search
  </p>
</div>

<ol start="6">
  <li>
    The next step is to create an index. The following code will create an index object with field objects that define the correct schema based on the car POCO. The FieldBuilder class iterates over the properties of the Car POCO using reflections.
  </li>
</ol>

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2018/05/Create-an-index-object-for-the-search.jpg"><img loading="lazy" src="/assets/img/posts/2018/05/Create-an-index-object-for-the-search.jpg" alt="Create an index object for the search" /></a>
  
  <p>
    Create an index object for the search
  </p>
</div>

<ol start="7">
  <li>
    Next, I create a batch of cars to upload:
  </li>
</ol>

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2018/05/Create-a-car-array.jpg"><img loading="lazy" src="/assets/img/posts/2018/05/Create-a-car-array.jpg" alt="Create a car array" /></a>
  
  <p>
    Create a car array
  </p>
</div>

<ol start="8">
  <li>
    This array of cars are the documents which will be searched. Create a batch object and upload it.
  </li>
</ol>

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2018/05/Upload-the-car-array.jpg"><img loading="lazy" src="/assets/img/posts/2018/05/Upload-the-car-array.jpg" alt="Upload the car array" /></a>
  
  <p>
    Upload the car array
  </p>
</div>

<ol start="9">
  <li>
    When I started with the Azure Search, I had problems finding results, although I uploaded my documents. Then I figured out that it is not working when you add the index and documents and search through it right away. You have to wait for a second to make it work. Therefore I added Thread.Sleep(1000).
  </li>
</ol>

### Search for a brand

  1. That&#8217;s all you have to do to set up the search. The next step is to search something. To do that, create a SearchParameters object containing all properties, which the search should return. Then execute the search with the string you are looking for and your SearchParameters object.

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2018/05/Execute-the-search.jpg"><img loading="lazy" src="/assets/img/posts/2018/05/Execute-the-search.jpg" alt="Execute the search" /></a>
  
  <p>
    Execute the search
  </p>
</div>

<ol start="2">
  <li>
    The search should return two objects. You can iterate through the results and, for example, print the type of the car.
  </li>
</ol>

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2018/05/Iterate-through-the-search-result.jpg"><img loading="lazy" src="/assets/img/posts/2018/05/Iterate-through-the-search-result.jpg" alt="Iterate through the search result" /></a>
  
  <p>
    Iterate through the search result
  </p>
</div>

## Conclusion

This post provided an overview of the Azure Search Service and how it can be implemented using C#.

You can find the code of the demo on <a href="https://github.com/WolfgangOfner/Azure-Search" target="_blank" rel="noopener">GitHub</a>.

For more information about the 70-532 exam get the <a href="http://amzn.to/2EWNWMF" target="_blank" rel="noopener">Exam Ref book from Microsoft</a> and continue reading my blog posts. I am covering all topics needed to pass the exam. You can find an overview of all posts related to the 70-532 exam <a href="/prepared-for-the-70-532-exam/" target="_blank" rel="noopener">here</a>.