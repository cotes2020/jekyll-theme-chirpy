---
title: Blazor Server vs. Blazor WebAssembly
date: 2020-06-29T09:20:00+02:00
author: Wolfgang Ofner
categories: [ASP.NET, Frontend]
tags: [Blazor Server, Blazor WebAssembly, 'C#', WASM, Blazor]
---
Blazor Server was release with .net core 3.0 in September 2019 and Blazor WebAssembly (WASM) was released in May 2020. Today, I will talk about the differences, when to use what version, and everything else you need to know about Blazor Server vs. Blazor WebAssembly.

## Blazor Server vs. Blazor WebAssembly Features

Before, I go into the details of each version of Blazor, let&#8217;s have a look at their features.

### WebAssembly Hosting Model

  * WASM runs in the browser on the client.
  * The first request to the WASM application downloads the CLR, Assemblies, JavaScript, CSS (React and Angular work similar).
  * It runs in the secure WASM sandbox.
  * The Blazor Javascript handler accesses the DOM (Document Object Model).

### Server Hosting Model

  * The C# code runs on the server.
  * Javascript hooks are used to access the DOM.
  * Binary messages are used to pass information between the browser and the server using SignalR.
  * If something is changed the server sends back DOM update messages.

### SignalR

SignalR is an integral part of Blazor and offers these features:

  * It is free, open-source and a first-class citizen of .net core
  * Sends async messages over persistent connections.
  * Connections are two-way.
  * Every client has its own connection.
  * Azure SignalR Services offers a free tier.
  * Example applications are chat applications or the car tracking in the Uber app.

## Blazor Server vs. Blazor WebAssembly Project Templates

To create both versions of Blazor you should have an up to date version of Visual Studio 2019. Blazor WASM was added in May 2020, whereas Blazor Server was included in the launch of VS 2019.

You can find the code of the demo on <a href="https://github.com/WolfgangOfner/Blazor-ServervsClient" target="_blank" rel="noopener noreferrer">Github</a>.

### Blazor Server

In Visual Studio create a new project and select Blazor App and then Blazor Server App.

<div id="attachment_2228" style="width: 710px" class="wp-caption aligncenter">
  <a href="/wp-content/uploads/2020/06/Create-a-new-Blazor-Server-App.jpg"><img aria-describedby="caption-attachment-2228" loading="lazy" class="wp-image-2228" src="/wp-content/uploads/2020/06/Create-a-new-Blazor-Server-App.jpg" alt="Create a new Blazor Server App" width="700" height="482" /></a>
  
  <p id="caption-attachment-2228" class="wp-caption-text">
    Create a new Blazor Server App
  </p>
</div>

The template creates a sample application where you can increase a counter and load data about the weather forecast. Start the application, open the Counter feature and click a couple of times on the button. The blazor.server.js Javascript intercepts the button click action and uses SignalR to send messages to the server where the code is executed. The blazor.server.js file is loaded in the _Host.cshtml file.

[code language=&#8221;Javascript&#8221;]  
<script src="_framework/blazor.server.js"></script>  
[/code]

You can see the messages sent to the server when you open the developer tools, go to the Network tab and click the button several times.

<div id="attachment_2229" style="width: 563px" class="wp-caption aligncenter">
  <a href="/wp-content/uploads/2020/06/SignalR-messages-between-server-and-browser.png"><img aria-describedby="caption-attachment-2229" loading="lazy" class="size-full wp-image-2229" src="/wp-content/uploads/2020/06/SignalR-messages-between-server-and-browser.png" alt="SignalR messages between server and browser" width="553" height="541" /></a>
  
  <p id="caption-attachment-2229" class="wp-caption-text">
    SignalR messages between server and browser
  </p>
</div>

### Blazor WebAssembly

In Visual Studio create a new project and select Blazor App and then Blazor WebAssembly App.

<div id="attachment_2230" style="width: 710px" class="wp-caption aligncenter">
  <a href="/wp-content/uploads/2020/06/Create-a-new-Blazor-WebAssembly-App.png"><img aria-describedby="caption-attachment-2230" loading="lazy" class="wp-image-2230" src="/wp-content/uploads/2020/06/Create-a-new-Blazor-WebAssembly-App.png" alt="Create a new Blazor WebAssembly App" width="700" height="484" /></a>
  
  <p id="caption-attachment-2230" class="wp-caption-text">
    Create a new Blazor WebAssembly App
  </p>
</div>

Creating a WASM App creates three projects: client, server, and shared whereas shared is a .Net Standard project. Start the application, open the Network tab of the developer tools and you will see that 6.6 MB got downloaded. If you start the application in Release mode, only 2.3 MB are downloaded. Switching to Release turns on the linker and therefore only necessary files are sent.

<div id="attachment_2231" style="width: 400px" class="wp-caption aligncenter">
  <a href="/wp-content/uploads/2020/06/All-the-files-which-are-sent-to-the-client.jpg"><img aria-describedby="caption-attachment-2231" loading="lazy" class="wp-image-2231" src="/wp-content/uploads/2020/06/All-the-files-which-are-sent-to-the-client.jpg" alt="All the files which are sent to the client" width="390" height="700" /></a>
  
  <p id="caption-attachment-2231" class="wp-caption-text">
    All the files which are sent to the client
  </p>
</div>

The Dotnet.wasm file is a webassembly file which contains a mono-based .net runtime.

## Scalability

WASM doesn’t need scaling for the client part because it runs in the browser. You have to scale the server part in the same way you scale a classic web server. For the server part, Microsoft estimates that you need around 85 KB of RAM for every client. Microsoft tested how much a Blazor server can handle. The test is successful when the latency is below 200 milliseconds. To keep it close to a real-world application, the test was a button click per second.

  * A server with 1 CPU and 3.5 GB ram could support 5k concurrent connections.
  * A server with 4 CPU and 14 GB ram could support 20k concurrent connections.

SignalR web sockets use port 80, which can be offloaded from the server to Azure SignalR Service which allows 100k  concurrent users and 100 million messages per day. The free offering allows 20 concurrent users and 20k messages per day.

## Pros and Cons Blazor Server

Advantages

  * Faster loading than WASM
  * Access to secure resources like databases or cloud-based services
  * Supports browsers which don’t support WASM like IE 11
  * C# code is not sent to the browser

Disadvantages

  * Extra latency due to sending data back and forth
  * No offline support
  * Scalability can be challenging
  * Server required (serverless possible)

## Pros and Cons Blazor WebAssembly

Advantages

  * Faster UI Code
  * When performance matters use WASM
  * Offline support
  * Can be distributed via CDN, no server required (except for API)
  * Any .Net standard 2.0 C# can be run

Disadvantages

  * An API layer is required if you want to access secure resources
  * Debugging still a bit limited

## Conclusion

Blazor Server and WebAssembly application both have their advantages and disadvantages. If you want to serve a large number of users and don&#8217;t have secret code running, use WASM. It also offers offline support. Use Blazor Server if performance is important or if you have sensitive code that you don&#8217;t want to send to the browser.

You can find the code of the demo on <a href="https://github.com/WolfgangOfner/Blazor-ServervsClient" target="_blank" rel="noopener noreferrer">Github</a>.