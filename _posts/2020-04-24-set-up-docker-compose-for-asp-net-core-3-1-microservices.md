---
title: Set up Docker-Compose for ASP .NET Core 3.1 Microservices
date: 2020-04-24T11:04:16+02:00
author: Wolfgang Ofner
categories: [Microservice, Docker]
tags: [.net core 3.1, 'C#', CQRS, docker, docker-compose, MediatR, microservice, RabbitMQ, Swagger]
---
<a href="https://www.programmingwithwolfgang.com/asp-net-core-with-https-in-docker" target="_blank" rel="noopener noreferrer">In my last post</a>, I described how to dockerize my microservices. To start the whole application, you have to start both microservices and RabbitMq. Today, I will add a docker-compose file which is a tool to easily run multi-container applications.

## What is the Docker-Compose File?

Docker-compose is a yml file in which you can set up all the containers your application needs to run. Simplified, it executes several docker run commands after each other. If you are used to docker run commands, the content of the compose file will look familiar. Let&#8217;s have a look at the content of the file:

[code language=&#8221;text&#8221;]  
version: "3.6"  
services:  
rabbitmq:  
container_name: rabbitmq  
ports:  
&#8211; 5672:5672  
&#8211; 15672:15672  
environment:  
&#8211; RABBITMQ\_DEFAULT\_USER=user  
&#8211; RABBITMQ\_DEFAULT\_PASS=password  
image: rabbitmq:3-management

customerapi:  
container_name: customerapi  
ports:  
&#8211; 8000:80  
&#8211; 8001:443  
environment:  
&#8211; "ASPNETCORE_URLS=https://+;http://+"  
&#8211; Kestrel\_\_Certificates\_\_Default__Path=/app/Infrastructure/Certificate/cert-aspnetcore.pfx  
&#8211; Kestrel\_\_Certificates\_\_Default__Password=SecretPassword  
image: wolfgangofner/customerapi  
restart: on-failure  
depends_on:  
&#8211; rabbitmq  
[/code]

This file describes two images, rabbitmq, and customerapi. Let&#8217;s have a closer look at the customerapi definition:

  * Ports: The container is listening to the ports 8000 and 8001 and redirects the request to the ports 80 and 443 inside the container.
  * Environment: This section provides environment variables and their value to enable Kestrel to process SSL requests
  * Image: This specifies which image should be used. If it is not available locally, it will be downloaded from Dockerhub.
  * Restart: Here you can configure the restart policy. This container is always restarting on failure. Other options are always and until-stopped.
  * Depends on: This section specifies dependencies. It only specifies that the rabbitmq container has to be started before the customerapi container. It doesn&#8217;t guarantee that the container is already finished starting up

To get started with docker-compose, I highly recommend the website <a href="https://composerize.com/" target="_blank" rel="noopener noreferrer">composerize.com</a>. On this website, you can paste a docker run command and it will give you a compose file with your parameters.

## Start multiple Container with Docker-Compose

You can find the code of  the finished demo on <a href="https://github.com/WolfgangOfner/MicroserviceDemo" target="_blank" rel="noopener noreferrer">GitHub</a>.

To execute your compose file, open Powershel, and navigate to the location of your file. In my demo code, I am providing the compose file in the root folder. Once you navigated to your file, use the following command: docker-compose up -d. The -d parameter executes the command detached. This means that the containers run in the background and don&#8217;t block your Powershell window.

<div id="attachment_1953" style="width: 692px" class="wp-caption aligncenter">
  <a href="https://www.programmingwithwolfgang.com/wp-content/uploads/2020/04/Start-multiple-container-with-docker-compose.jpg"><img aria-describedby="caption-attachment-1953" loading="lazy" class="size-full wp-image-1953" src="https://www.programmingwithwolfgang.com/wp-content/uploads/2020/04/Start-multiple-container-with-docker-compose.jpg" alt="Start multiple container with docker-compose" width="682" height="620" srcset="https://www.programmingwithwolfgang.com/wp-content/uploads/2020/04/Start-multiple-container-with-docker-compose.jpg 682w, https://www.programmingwithwolfgang.com/wp-content/uploads/2020/04/Start-multiple-container-with-docker-compose-300x273.jpg 300w" sizes="(max-width: 682px) 100vw, 682px" /></a>
  
  <p id="caption-attachment-1953" class="wp-caption-text">
    Start multiple container with docker-compose
  </p>
</div>

To prove that all three containers are running, use docker ps.

<div id="attachment_1954" style="width: 710px" class="wp-caption aligncenter">
  <a href="https://www.programmingwithwolfgang.com/wp-content/uploads/2020/04/Check-all-running-container.jpg"><img aria-describedby="caption-attachment-1954" loading="lazy" class="wp-image-1954" src="https://www.programmingwithwolfgang.com/wp-content/uploads/2020/04/Check-all-running-container.jpg" alt="Check all running container" width="700" height="84" srcset="https://www.programmingwithwolfgang.com/wp-content/uploads/2020/04/Check-all-running-container.jpg 1202w, https://www.programmingwithwolfgang.com/wp-content/uploads/2020/04/Check-all-running-container-300x36.jpg 300w, https://www.programmingwithwolfgang.com/wp-content/uploads/2020/04/Check-all-running-container-1024x123.jpg 1024w, https://www.programmingwithwolfgang.com/wp-content/uploads/2020/04/Check-all-running-container-768x92.jpg 768w" sizes="(max-width: 700px) 100vw, 700px" /></a>
  
  <p id="caption-attachment-1954" class="wp-caption-text">
    Check all running container
  </p>
</div>

Another great feature of docker-compose is, that you can stop all your application with a single command: docker-compose down.

## Build and run Containers

You don&#8217;t have to use images from Docker hub in your compose file, you can also build images and then run them. To build an image, use the build section and set the context to the location of the Dockerfile.

[code language=&#8221;text&#8221;]  
version: "3.6"  
services:  
rabbitmq:  
container_name: rabbitmq  
ports:  
&#8211; 5672:5672  
&#8211; 15672:15672  
environment:  
&#8211; RABBITMQ\_DEFAULT\_USER=user  
&#8211; RABBITMQ\_DEFAULT\_PASS=password  
image: rabbitmq:3-management

customerapi:  
container_name: customerapi  
ports:  
&#8211; 8000:80  
&#8211; 8001:443  
environment:  
&#8211; "ASPNETCORE_URLS=https://+;http://+"  
&#8211; Kestrel\_\_Certificates\_\_Default__Path=/app/Infrastructure/Certificate/cert-aspnetcore.pfx  
&#8211; Kestrel\_\_Certificates\_\_Default__Password=SecretPassword  
build:  
context: ./CustomerApi  
dockerfile: CustomerApi/Dockerfile  
restart: on-failure  
depends_on:  
&#8211; rabbitmq

orderapi:  
container_name: order  
ports:  
&#8211; 9000:80  
&#8211; 9001:443  
environment:  
&#8211; "ASPNETCORE_URLS=https://+;http://+"  
&#8211; Kestrel\_\_Certificates\_\_Default__Path=/app/Infrastructure/Certificate/cert-aspnetcore.pfx  
&#8211; Kestrel\_\_Certificates\_\_Default__Password=SecretPassword  
build:  
context: ./OrderApi  
dockerfile: OrderApi/Dockerfile  
restart: on-failure  
depends_on:  
&#8211; rabbitmq  
[/code]

I named this file docker-compose.Build. You can use the -f parameter to specify the file in your docker-compose command: docker-compose -f docker-compose.Build.yml up -d.

<div id="attachment_1956" style="width: 710px" class="wp-caption aligncenter">
  <a href="https://www.programmingwithwolfgang.com/wp-content/uploads/2020/04/Build-images-with-docker-compose.jpg"><img aria-describedby="caption-attachment-1956" loading="lazy" class="wp-image-1956" src="https://www.programmingwithwolfgang.com/wp-content/uploads/2020/04/Build-images-with-docker-compose.jpg" alt="Build images with docker-compose" width="700" height="361" srcset="https://www.programmingwithwolfgang.com/wp-content/uploads/2020/04/Build-images-with-docker-compose.jpg 963w, https://www.programmingwithwolfgang.com/wp-content/uploads/2020/04/Build-images-with-docker-compose-300x155.jpg 300w, https://www.programmingwithwolfgang.com/wp-content/uploads/2020/04/Build-images-with-docker-compose-768x396.jpg 768w" sizes="(max-width: 700px) 100vw, 700px" /></a>
  
  <p id="caption-attachment-1956" class="wp-caption-text">
    Build images with docker-compose
  </p>
</div>

## How can Containers talk to each other?

When starting multiple containers with a compose file, a default network is created in which all containers are placed. Containers can reach each other with the container name. For example, the customerapi can send data to the queue using the name rabbitmq .

## Conclusion

In today&#8217;s post, I talked about docker-compose and how it can be used to easily set up applications with multiple containers.With this post, most of the features of the two microservices are implemented. <a href="https://www.programmingwithwolfgang.com/build-net-core-in-ci-pipeline-in-azure-devops" target="_blank" rel="noopener noreferrer">With my next post</a>, I will start to focus more on the DevOps process. This means that I will create CI pipelines, run tests automatically during pull requests and later on will automatically deploy the microservices.

Note: On October 11, I removed the Solution folder and moved the projects to the root level. Over the last months I made the experience that this makes it quite simpler to work with Dockerfiles and have automated builds and deployments.

You can find the code of  the finished demo on <a href="https://github.com/WolfgangOfner/MicroserviceDemo" target="_blank" rel="noopener noreferrer">GitHub</a>.