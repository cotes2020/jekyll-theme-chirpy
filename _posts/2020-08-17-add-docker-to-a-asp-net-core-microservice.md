---
title: Add Docker to an ASP .NET Core Microservice
date: 2020-08-17T18:31:42+02:00
author: Wolfgang Ofner
categories: [Docker]
tags: [.net core 3.1, docker, microservice]
---
Microservices need to be deployed often and fast. To achieve this, they often run inside a Docker container. In this post, I will show how easy it is to add Docker support to a project using Visual Studio.

<a href="/run-the-ci-pipeline-during-a-pull-request/" target="_blank" rel="noopener noreferrer">My last post</a> explained how to create a CI pipeline and protect the master branch in Azure DevOps. In my next post, I will add another CI pipeline to build the Docker container.

## What is Docker?

Docker is the most popular container technology. It is written in Go and open-source. A container can contain a Windows or Linux application and will run the same, no matter where you start it. This means it runs the same way during development, on the testing environment, and on the production environment. This eliminates the famous “It works on my machine”.

Another big advantage is that Docker containers share the host system kernel. This makes them way smaller than a virtual machine and enables them to start within seconds or even less. For more information about Docker, check out <a href="https://www.docker.com/resources/what-container" target="_blank" rel="noopener noreferrer">Docker.com</a>. There you can also download Docker Desktop which you will need to run Docker container on your machine.

For more information, see my post &#8220;<a href="/dockerize-an-asp-net-core-microservice-and-rabbitmq/" target="_blank" rel="noopener noreferrer">Dockerize an ASP .NET Core Microservice and RabbitMQ</a>&#8221;

## Add Docker Support to the Microservice

You can find the code of the demo on <a href="https://github.com/WolfgangOfner/.NetCoreMicroserviceCiCdAks/tree/AddDocker" target="_blank" rel="noopener noreferrer">Github</a>.

Open the solution with Visual Studio 2019 and right-click on the CustomerApi project. Then click on Add and select Docker Support&#8230;

<div id="attachment_2351" style="width: 680px" class="wp-caption aligncenter">
  <a href="/wp-content/uploads/2020/08/Add-Docker-Support-to-the-API-project.jpg"><img aria-describedby="caption-attachment-2351" loading="lazy" class="size-full wp-image-2351" src="/wp-content/uploads/2020/08/Add-Docker-Support-to-the-API-project.jpg" alt="Add Docker Support to the API project" width="670" height="600" /></a>
  
  <p id="caption-attachment-2351" class="wp-caption-text">
    Add Docker Support to the API project
  </p>
</div>

This opens a window where you can select the operating system for your project. If you don&#8217;t have a requirement to use Windows, I would always use Linux since it is smaller and therefore faster.

<div id="attachment_2352" style="width: 370px" class="wp-caption aligncenter">
  <a href="/wp-content/uploads/2020/08/Select-the-operating-system-for-the-Docker-container.jpg"><img aria-describedby="caption-attachment-2352" loading="lazy" class="size-full wp-image-2352" src="/wp-content/uploads/2020/08/Select-the-operating-system-for-the-Docker-container.jpg" alt="Select the operating system for the Docker container" width="360" height="156" /></a>
  
  <p id="caption-attachment-2352" class="wp-caption-text">
    Select the operating system for the Docker container
  </p>
</div>

After you clicked OK, a Dockerfile was added to the project and in Visual Studio you can see that you can start the project with Docker now.

<div id="attachment_2353" style="width: 632px" class="wp-caption aligncenter">
  <a href="/wp-content/uploads/2020/08/Start-the-project-in-Docker.jpg"><img aria-describedby="caption-attachment-2353" loading="lazy" class="size-full wp-image-2353" src="/wp-content/uploads/2020/08/Start-the-project-in-Docker.jpg" alt="Start the project in Docker" width="622" height="79" /></a>
  
  <p id="caption-attachment-2353" class="wp-caption-text">
    Start the project in Docker
  </p>
</div>

Click F5 to start the application and your browser will open. To prove that this application runs inside a Docker container, check what containers are running. You can do this with the following command:

[code language=&#8221;PowerShell&#8221;]  
docker ps  
[/code]

On the following screenshot, you can see that I have one container running with the name CustomerApi and it runs on port 32770. This is the same port as the browser opened.

<div id="attachment_2354" style="width: 710px" class="wp-caption aligncenter">
  <a href="/wp-content/uploads/2020/08/The-microservice-is-running-inside-a-Docker-container.jpg"><img aria-describedby="caption-attachment-2354" loading="lazy" class="wp-image-2354" src="/wp-content/uploads/2020/08/The-microservice-is-running-inside-a-Docker-container.jpg" alt="The microservice is running inside a Docker container" width="700" height="82" /></a>
  
  <p id="caption-attachment-2354" class="wp-caption-text">
    The microservice is running inside a Docker container
  </p>
</div>

## The Dockerfile explained

Visual Studio generates a so-called multi-stage Dockerfile. This means that several images are used to keep the output image as small as possible. The first line in the Dockerfile uses the ASP .NET Core 3.1 runtime and names it base. Additionally, the ports 80 and 443 are exposed so we can access the container with HTTP and HTTPs later.

[code language=&#8221;text&#8221;]  
FROM mcr.microsoft.com/dotnet/core/aspnet:3.1-buster-slim AS base  
WORKDIR /app  
EXPOSE 80  
EXPOSE 443  
[/code]

The next section uses the .NET Core 3.1 SDK to build the project. This image is only used for the build and won&#8217;t be present in the output container. As a result, the container will be smaller and therefore will start faster. Additionally the projects are copied into the container.

[code language=&#8221;text&#8221;]  
FROM mcr.microsoft.com/dotnet/core/sdk:3.1-buster AS build  
WORKDIR /src  
COPY ["Solution/CustomerApi/CustomerApi.csproj", "Solution/CustomerApi/"]  
COPY ["Solution/CustomerApi.Domain/CustomerApi.Domain.csproj", "Solution/CustomerApi.Domain/"]  
COPY ["Solution/CustomerApi.Service/CustomerApi.Service.csproj", "Solution/CustomerApi.Service/"]  
COPY ["Solution/CustomerApi.Data/CustomerApi.Data.csproj", "Solution/CustomerApi.Data/"]  
[/code]

Next, I restore the Nuget packages of the CustomerApi and then build the CustomerApi project.

[code language=&#8221;text&#8221;]  
RUN dotnet restore "Solution/CustomerApi/CustomerApi.csproj"  
COPY . .  
WORKDIR "/src/Solution/CustomerApi"  
RUN dotnet build "CustomerApi.csproj" -c Release -o /app/build  
[/code]

The last part of the Dockerfile publishes the CustomerApi project. The last line sets the entrypoint as a dotnet application and that the CustomerApi.dll should be run.

[code language=&#8221;text&#8221;]  
FROM build AS publish  
RUN dotnet publish "CustomerApi.csproj" -c Release -o /app/publish  
FROM base AS final  
WORKDIR /app  
COPY &#8211;from=publish /app/publish .  
ENTRYPOINT ["dotnet", "CustomerApi.dll"]  
[/code]

## Conclusion

In today&#8217;s DevOps culture it is necessary to change applications fast and often. Additionally, microservices should run inside a container whereas Docker is the defacto standard container. This post showed how easy it is to add Docker to a microservice in Visual Studio.

You can find the code of this demo on <a href="https://github.com/WolfgangOfner/.NetCoreMicroserviceCiCdAks/tree/AddDocker" target="_blank" rel="noopener noreferrer">Github</a>.