---
title: 'Blazor - Getting Started'
date: 2020-06-22T22:41:29+02:00
author: Wolfgang Ofner
categories: [ASP.NET, Frontend]
tags: [.net core, Blazor, 'C#', SignalR]
---
Blazor was introduced with .net core 3.0 and is a web UI single page application (SPA) framework. Developers write the code in HTML, CSS, C#, and optionally Javascript. Blazor was developed by Steve Sanderson and presented  in July 2018. The main advantage over Javascript frameworks is that you can run C# directly in the browser which should be appealing for enterprise developers who don&#8217;t want to deal with Javascript

## Blazor Features

Blazor comes with the following features:

  * Run C# code inside your browser as WebAssembly (WASM)
  * Blazor is a mix of browser and Razor
  * WASM is more locked down than Javascript and therefore more secure
  * Messages between the browser and the backend are sent via SignalR
  * There is no direct DOM access but you can interact with the DOM using Javascript
  * Modern browser support WASM: Firefox, Chrome, Edge, Safari (IE 11 does not)
  * Components can be shared via Nuget

There are two different versions: Blazor Server and Blazor Webassembly. I will talk about the difference in more detail in my next post. For now, you only have to remember that the Server version runs on the server and sends code to the browser and the Webassembly version runs directly in the browser and calls APIs, like Angular or React apps.

## Create your First Blazor Server App

To follow the demo you need Visual Studio 2019 and at least .net core 3.0. You can find the source of the demo on <a href="https://github.com/WolfgangOfner/Blazor-Server" target="_blank" rel="noopener noreferrer">Github</a>.

To create a new Blazor App, select the Blazor template in Visual Studio.

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2020/06/Select-the-Blazor-App-template.jpg"><img loading="lazy" src="/assets/img/posts/2020/06/Select-the-Blazor-App-template.jpg" alt="Select the Blazor App template" /></a>
  
  <p>
    Select the Blazor App template
  </p>
</div>

After entering a name and location, select Blazor Server App, and click on Create.

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2020/06/Create-a-Blazor-Server-App.jpg"><img loading="lazy" src="/assets/img/posts/2020/06/Create-a-Blazor-Server-App.jpg" alt="Create a Blazor Server App" /></a>
  
  <p>
    Create a Blazor Server App
  </p>
</div>

This creates the solution and also a demo application. Start the application and you will see a welcome screen and a menu on the left side. Click on Fetch data to see some information about the weather forecast or on Counter to see a button which increases the amount when you click it.

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2020/06/The-counter-function-of-the-demo-app.jpg"><img loading="lazy" src="/assets/img/posts/2020/06/The-counter-function-of-the-demo-app.jpg" alt="The counter function of the demo app" /></a>
  
  <p>
    The counter function of the demo app
  </p>
</div>

All these features are implemented with HTML, CSS, and C# only. No need for Javascript.

### Create a new Component

Open the Counter.razor file in the Pages folder and you can see the whole code for the implementation of the Counter feature.

```csharp  
@page "/counter"

<h1>Counter</h1>

<p>Current count: @currentCount</p>

<button class="btn btn-primary" @onclick="IncrementCount">Click me</button>

@code {
    private int currentCount = 0;

    private void IncrementCount()
    {
        currentCount++;
    }
}  
```

On the top, you have the page attribute which indicates the route. In this example, when you enter /counter, this page will be rendered. The logic of the functionality is in the code block and the button has an onclick event. This event calls the IncrementCount method which then increases the count of the currentCode variable.

### Create a new Page

In this section, I will add a new Blazor component with a dropdown menu, event handling, and binding of a variable.

Right-click on the Pages folder and add a Blazor Component with the name DisplayProducts. Then add the following code:

```csharp  
@page "/products"

<h3>Product:</h3>

@code {  
    private List<Product> Products; 
    
    protected override void OnInitialized()
    {
        Products = new List<Product>
        {
            new Product{Id = 1, Name = "Phone", Price = 999.99m},
            new Product{Id = 2, Name = "Book", Price = 5.99m},
            new Product{Id = 3, Name = "Car", Price = 19999.99m}
        };
    }
    
    public class Product
    {
        public int Id { get; set; }
    
        public string Name { get; set; }
    
        public decimal Price { get; set; }
    }  
}
```

This code sets the route to products and displays a headline. In the code block, I overwrite the OnInitialized method and create a class Product. The OnInitialized method is executed during the load of the page. You might know this from Webforms. Next, I add a dropdown menu and add each element of my Products list.

```csharp  
<select size="1" style="width:10%">  
@foreach (var product in Products)  
{  
    <option value="@product.Id.ToString()">@product.Name &#8211; $@product.Price</option>  
}  
</select>  
```

As the last step, I add an onchange event to the dropdown menu and the code of the executed method in the code block. The selected value will be displayed below the dropdown menu. If you haven&#8217;t selected anything, the whole div tag won&#8217;t be displayed. My whole component looks like the following:

```csharp  
@page "/products"

@if (Products != null)
{
    <h3>Product:</h3>
    <select @onchange="ProductSelected" size="1" style="width:10%">
        @foreach (var product in Products)
        {
            <option value="@product.Id.ToString()">@product.Name - $@product.Price</option>
        }
    </select>

    if (SelectedProduct != null)
    {
        <br />
        <div>Selected Product: @SelectedProduct.Name</div>
    }
}

@code {
    private List<Product> Products;

    Product SelectedProduct;

    void ProductSelected(ChangeEventArgs args)
    {
        SelectedProduct = (from p in Products where p.Id == Convert.ToInt32(args.Value.ToString()) select p).FirstOrDefault();
    }


    protected override void OnInitialized()
    {
        Products = new List<Product>
        {
            new Product{Id = 1, Name = "Phone", Price = 999.99m},
            new Product{Id = 2, Name = "Book", Price = 5.99m},
            new Product{Id = 3, Name = "Car", Price = 19999.99m}
        };
    }

    public class Product
    {
        public int Id { get; set; }

        public string Name { get; set; }

        public decimal Price { get; set; }
    }
}  
```

Run the application, navigate to /products and you will see the dropdown menu. Select a product and it will be displayed below the dropdown.

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2020/06/Testing-my-first-Blazor-Component.jpg"><img loading="lazy" src="/assets/img/posts/2020/06/Testing-my-first-Blazor-Component.jpg" alt="Testing my first Blazor Component" /></a>
  
  <p>
    Testing my first Blazor Component
  </p>
</div>

## Conclusion

Blazor is a new frontend framework that enables developers to run C# in the browser. This might be a great asset for enterprises that already use .Net but don&#8217;t want to deal with Javascript frameworks.

You can find the source of the demo on <a href="https://github.com/WolfgangOfner/Blazor-Server" target="_blank" rel="noopener noreferrer">Github</a>.