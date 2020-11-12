---
title: Dockerize an ASP .NET Core Microservice and RabbitMQ
date: 2020-04-21T11:24:50+02:00
author: Wolfgang Ofner
categories: [Docker, ASP.NET]
tags: [NET Core 3.1, 'C#', CQRS, docker, docker-compose, MediatR, microservice, RabbitMQ, Swagger]
---
<a href="/rabbitmq-in-an-asp-net-core-3-1-microservice/" target="_blank" rel="noopener noreferrer">In my last post</a>, I added RabbitMQ to my two microservices which finished all the functional requirements. Microservices became so popular because they can be easily deployed using Docker. Today I will dockerize my microservices and create Docker container which can be run anywhere as long as Docker is installed. I will explain most of the Docker commands but basic knowledge about starting and stopping containers is recommended.

## What is Docker?

Docker is the most popular container technology. It is written in Go and open-source. A container can contain a Windows or Linux application and will run the same, no matter where you start it. This means it runs the same way during development, on the testing environment, and on the production environment. This eliminates the famous &#8220;It works on my machine&#8221;.

Another big advantage is that Docker containers share the host system kernel. This makes them way smaller than a virtual machine and enables them to start within seconds or even less. For more information about Docker, check out <a href="https://www.docker.com/resources/what-container" target="_blank" rel="noopener noreferrer">Docker.com</a>. There you can also download Docker Desktop which you will need to run Docker container on your machine.

## What is Dockerhub?

<a href="https://hub.docker.com/" target="_blank" rel="noopener noreferrer">Dockerhub</a> is like GitHub for Docker containers. You can sign up for free and get unlimited public repos and one private repo. There are also enterprise plans which give you more private repos, build pipelines for your containers and security scanning.

To Dockerize an application means that you create a Docker container or at least a Dockerfile which describes how to create the container. You can upload the so-called container image to container registries like Dockerhub so other developers can easily download and run it.

Dockerhub is the go-to place if you want to download official container images. The RabbitMQ from the last post was downloaded from there or you can download Redis, SQL Server from Microsoft or thousands of other popular applications.

## Dockerize the Microservices

You can find the code of  the finished demo on <a href="https://github.com/WolfgangOfner/MicroserviceDemo" target="_blank" rel="noopener noreferrer">GitHub</a>.

Visual Studio makes it super easy to dockerize your application. All you have to do it to right-click on the API project and then select Add &#8211;> Docker Support.

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2020/04/Dockerize-your-application.jpg"><img loading="lazy" src="/assets/img/posts/2020/04/Dockerize-your-application.jpg" alt="Dockerize your application" /></a>
  
  <p>
    Dockerize your application
  </p>
</div>

This opens a new window where you can select Linux or Windows as OS for the container. I am always going for Linux as my default choice because the image is way smaller than Windows and therefore starts faster. Also, all my other containers run on Linux and on Docker Desktop, you can only run containers with the same OS at a time. After clicking OK, the Dockerfile and .dockerignore files are added. That&#8217;s all you have to do to dockerize the application.

### Dockerfile and .dockerignore Files

The .dockerignore file is like the .gitignore file and contains extensions and paths which should not be copied into the container. Default extensions in the .dockerignore file are .vs, /bin or /obj. The .dockerignore file is not required to dockerize your application but highly recommended.

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2020/04/Content-of-the-.dockerignore-file.jpg"><img loading="lazy" src="/assets/img/posts/2020/04/Content-of-the-.dockerignore-file.jpg" alt="Content of the .dockerignore file" /></a>
  
  <p>
    Content of the .dockerignore file
  </p>
</div>

The Dockerfile is a set of instructions to build and run an image. Visual Studio creates a multi-stage Dockerfile which means that it builds the application but only adds necessary files and images to the container image. The Dockerfile uses the .NET Core SDK to build the image but uses the way smaller .NET Core runtime image inside of the container. Let&#8217;s take a look at the different stages of the Dockerfile.

### Understanding the multi-stage Dockerfile

```docker
FROM mcr.microsoft.com/dotnet/core/aspnet:3.1-buster-slim AS base  
WORKDIR /app  
EXPOSE 80  
EXPOSE 443  
```

The first part downloads the .NET Core runtime 3.1 image from Docker hub and gives it the name base which will be used later on. Then it sets the working directory to /app which will also be later used. Lastly, the ports 80 and 443 are exposed which tells Docker to listen to these two ports when the container is running.

```docker 
FROM mcr.microsoft.com/dotnet/core/sdk:3.1-buster AS build  
WORKDIR /src  
COPY ["CustomerApi/CustomerApi.csproj", "CustomerApi/"]  
COPY ["CustomerApi.Domain/CustomerApi.Domain.csproj", "CustomerApi.Domain/"]  
COPY ["CustomerApi.Messaging.Send/CustomerApi.Messaging.Send.csproj", "CustomerApi.Messaging.Send/"]  
COPY ["CustomerApi.Service/CustomerApi.Service.csproj", "CustomerApi.Service/"]  
COPY ["CustomerApi.Data/CustomerApi.Data.csproj", "CustomerApi.Data/"]  
RUN dotnet restore "CustomerApi/CustomerApi.csproj"  
COPY . .  
WORKDIR "/src/CustomerApi"  
RUN dotnet build "CustomerApi.csproj" -c Release -o /app/build  
```

The next section downloads the .NET Core 3.1 SDK from Dockerhub and names it build. Then the working directory is set to /src and all project files (except test projects) of the solution are copied inside the container. Then dotnet restore is executed to restore all NuGet packages and the working directory is changed to the directory of the API project. Note that the path starts with /src, the working directory path I set before I copied the files inside the container. Lastly, dotnet build is executed which builds the project with the Release configuration into the path /app/build.

```docker
FROM build AS publish  
RUN dotnet publish "CustomerApi.csproj" -c Release -o /app/publish  
```

The build image in the first line of the next section is the SDK image which we downloaded before and named build. We use it to run dotnet publish which publishes the CustomerApi project.

```docker 
FROM base AS final  
WORKDIR /app  
COPY &#8211;from=publish /app/publish .  
ENTRYPOINT ["dotnet", "CustomerApi.dll"]  
```

The last section uses the runtime image and sets the working directory to /app. Then the published files from the last step are copied into the working directory. The dot means that it is copied to your current location, therefore /app. The Entrypoint command tells Docker to configure the container as an executable and to run the CustomerApi.dll when the container starts.

For more details on the Dockerfile and .dockerignore file check out the <a href="https://docs.docker.com/engine/reference/builder/" target="_blank" rel="noopener noreferrer">official documentation</a>.

## Test the Dockerized Application

After adding the Docker support to your application, you should be able to select Docker as a startup option in Visual Studio. When you select Docker for the first time, Visual Studio will run the Dockerfile, therefore build and create the container. This might take a bit because the images for the .NET Core runtime and SDK need to be downloaded. After the first download, they are cached and can be quickly reused.

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2020/04/Select-Docker-as-startup-option.jpg"><img loading="lazy" src="/assets/img/posts/2020/04/Select-Docker-as-startup-option.jpg" alt="Dockerize the application and select Docker as startup option" /></a>
  
  <p>
    Select Docker as startup option
  </p>
</div>

Click F5 or on the Docker button and your application should start as you are used to. If you don&#8217;t believe me that it is running inside a Docker container, you can check the running containers in PowerShell with the command docker ps.

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2020/04/Check-the-running-containers.jpg"><img loading="lazy" src="/assets/img/posts/2020/04/Check-the-running-containers.jpg" alt="Check the running containers" /></a>
  
  <p>
    Check the running containers
  </p>
</div>

The screenshot above shows that the customerapi image was started two minutes ago, that it is running for two minutes and that it maps the port 32789 to port 80 and 32788 to 433. To stop a running container, you can use docker stop [id]. In my case. this would be docker stop f25727f43d6b. You don&#8217;t have to use the full id, like in git. Docker only needs to clearly identify the image you want to stop. So you could use docker stop f25.

## Build the Dockerfile without Visual Studio

You don&#8217;t need Visual Studio to create a Docker image. This is useful when you want to create the image and then push it to a container registry like Docker hub. You should always do this in a build pipeline but its useful to know how to do it by hand and sometimes you need it to quickly test something.

Open Powershell and navigate to the folder containing the CustomerApi.sln file. To build an image, you can use docker build \[build context\] \[location of Dockerfile\]. Optionally, you can add a tag by using -t Tagname. Use

```powershell  
docker build -t customerapi . -f CustomerApi/Dockerfile  
```

to build the Dockerfile which is in your current file with the tag name customerapi. This will download the needed images (or use them from the cache) and start to build your image. Step 7 fails because a directory can&#8217;t be found though.

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2020/04/Build-the-docker-image.jpg"><img loading="lazy" src="/assets/img/posts/2020/04/Build-the-docker-image.jpg" alt="Build the docker image" /></a>
  
  <p>
    Build the docker image
  </p>
</div>

To confirm that your image was really created, use docker images.

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2020/04/Confirm-that-the-image-was-created.jpg"><img loading="lazy" src="/assets/img/posts/2020/04/Confirm-that-the-image-was-created.jpg" alt="Confirm that the image was created" /></a>
  
  <p>
    Confirm that the image was created
  </p>
</div>

### Start the newly built Image

To start an image use docker run [-p &#8220;port outside of the container&#8221;:&#8221;port inside the container&#8221;] name of the image to start. In my example:

```powershell  
docker run -p 32789:80 -p 32788:443 customerapi .  
```

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2020/04/Run-the-previously-created-image.jpg"><img loading="lazy" src="/assets/img/posts/2020/04/Run-the-previously-created-image.jpg" alt="Run the previously created image dockerize" /></a>
  
  <p>
    Run the previously created image
  </p>
</div>

After the container is started, open localhost:32789 and you should see the Swagger UI of the API. If you use the HTTP port, you will get a connection closed error. HTTPS is currently not working because we have to provide a certificate so kestrel can process HTTPs requests. I will explain <a href="/asp-net-core-with-https-in-docker" target="_blank" rel="noopener noreferrer">in my next post</a> how to add a certificate to the container. For now, I will only use the HTTP port.

### Push the Image to Dockerhub

We confirmed that the image is running, and now it is time to share it and therefore to upload it to Dockerhub. Dockerhub is the default registry in Docker Desktop. Use docker login to login in your Dockerhub account.

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2020/04/Log-in-into-Dockerhub.jpg"><img loading="lazy" src="/assets/img/posts/2020/04/Log-in-into-Dockerhub.jpg" alt="Log in into Dockerhub" /></a>
  
  <p>
    Log in into Dockerhub
  </p>
</div>

Next, I have to tag the image I want to upload with the name of my Dockerhub account and the name of the repository I want to use. I do this with docker tag Image DockerhubAccount/repository.

```powershell  
docker tag customerapi wolfgangofner/customerapi  
```

The last step is to push the image to Dockerhub using docker push tagname.

```powershell  
docker push wolfgangofner/customerapi  
```

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2020/04/Push-the-image-to-Dockerhub.jpg"><img loading="lazy" src="/assets/img/posts/2020/04/Push-the-image-to-Dockerhub.jpg" alt="Push the image to Dockerhub dockerize" /></a>
  
  <p>
    Push the image to Dockerhub
  </p>
</div>

To confirm that the image was pushed to Dockerhub, I open my repositories and see the newly create customerapi there.

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2020/04/Confirm-that-the-image-was-pushed-to-Dockerhub.jpg"><img loading="lazy" src="/assets/img/posts/2020/04/Confirm-that-the-image-was-pushed-to-Dockerhub.jpg" alt="Confirm that the image was pushed to Dockerhub dockerize" /></a>
  
  <p>
    Confirm that the image was pushed to Dockerhub
  </p>
</div>

### Testing the uploaded Image

To confirm that everything worked fine, I will download the image and run it on any machine. The only requirement is that Docker is installed. When you click on the repository, you can see the command to download the image. In my example, this is docker pull wolfgangofner/customerapi. I will use docker run because this runs the image and if it is not available automatically pull it too.

```powershell  
docker run -p 32789:80 -p 32788:443 wolfgangofner/customerapi  
```

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2020/04/Run-the-previously-uploaded-image.jpg"><img loading="lazy" src="/assets/img/posts/2020/04/Run-the-previously-uploaded-image.jpg" alt="Run the previously uploaded image dockerize" /></a>
  
  <p>
    Run the previously uploaded image
  </p>
</div>

Open localhost:32789 and the Swagger UI will appear.

For practice purposes, you can dockerize the OrderApi. The steps are identical to the steps for the CustomerApi.

## Conclusion

Today, I showed how to dockerize the microservices to create immutable Docker images which I can easily share using Dockerhub and run everywhere the same way. Currently, only the HTTP port of the application works because we haven&#8217;t provided an SSL certificate to process HTTPS requests. <a href="/asp-net-core-with-https-in-docker" target="_blank" rel="noopener noreferrer">In my next post</a>, I will create a development certificate and start the image with it.

Note: On October 11, I removed the Solution folder and moved the projects to the root level. I also edited this post to reflect the changes. Over the last months I made the experience that this makes it quite simpler to work with Dockerfiles and have automated builds and deployments.

You can find the code of  the finished demo on <a href="https://github.com/WolfgangOfner/MicroserviceDemo" target="_blank" rel="noopener noreferrer">GitHub</a>.