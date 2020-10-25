---
title: ASP .NET Core with HTTPS in Docker
date: 2020-04-22T11:30:30+02:00
author: Wolfgang Ofner
categories: [Docker, ASP.NET]  
tags: [.net core 3.1, 'C#', CQRS, docker, docker-compose, MediatR, microservice, RabbitMQ, SSL, Swagger]
---
<a href="/dockerize-an-asp-net-core-microservice-and-rabbitmq/" target="_blank" rel="noopener noreferrer">In my last post</a>, I dockerized my ASP .NET Core 3.1 microservices but the HTTPS connection didn&#8217;t work. Kestrel needs a certificate to process HTTPS requests. Today, I will show you how to create a development certificate and how to provide it to your Docker container so you can use ASP .NET Core with HTTPS in Docker.

## Start a Docker Container without a Certificate

Before I start, I want to show you what happens when you try to start a .NET Core application without a valid certificate. I use the following command to start the container:

```powershell  
docker run -p 32789:80 -p 32788:443 -e "ASPNETCORE_URLS=https://+;http://+" wolfgangofner/customerapi  
```

This command sets a port mapping, adds an environment variable and starts the image customerapi from my Dockerhub repository. Executing this command will result in the following exception:

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2020/04/Start-a-.net-core-application-without-a-certificate.jpg"><img loading="lazy" src="/assets/img/posts/2020/04/Start-a-.net-core-application-without-a-certificate.jpg" alt="Start a .NET Core application without a certificate" /></a>
  
  <p>
    Start a .NET Core application without a certificate
  </p>
</div>

As you can see, Kestrel can&#8217;t start because no certificate was specified and no developer certificate could be found. When you start your application inside a Docker container within Visual Studio, Visual Studio manages the certificate for you. But without Visual Studio, you have to create the certificate yourself.

## Creating a Certificate to use ASP .NET Core with HTTPS in Docker

You can create a certificate with the following command: <span class="">dotnet dev-certs https -ep [Path of the certificate]-p [Password]. I create the certificate under D:\temp and set Password as its password.</span>

<div>
  <div class="col-12 col-sm-10 aligncenter">
    <a href="/assets/img/posts/2020/04/Creating-the-certificate.jpg"><img loading="lazy" src="/assets/img/posts/2020/04/Creating-the-certificate.jpg" alt="Creating the certificate to use ASP .Net Core with HTTPS in Docker" /></a>
    
    <p>
      Creating the certificate
    </p>
  </div>
</div>

Note that you must set a password. Otherwise, Kestrel won&#8217;t be able to use the certificate.

## Provide the Certificate to the Docker Image

After creating the certificate, you only have to share it with your container and the .NET Core application should start. I use the following command:

```powershell  
docker run -p 32789:80 -p 32788:443 -e Kestrel\_\_Certificates\_\_Default\_\_Path=/app/Infrastructure/Certificate/certificate.pfx -e Kestrel\_\_Certificates\_\_Default\_\_Password=Password -e "ASPNETCORE_URLS=https://+;http://+"-v D:\temp\:/app/Infrastructure/Certificate wolfgangofner/customerapi  
```

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2020/04/Start-a-.net-core-application-and-provide-a-certificate.jpg"><img loading="lazy" src="/assets/img/posts/2020/04/Start-a-.net-core-application-and-provide-a-certificate.jpg" alt="Start a .NET Core application and provide a certificate" /></a>
  
  <p>
    Start a .NET Core application and provide a certificate
  </p>
</div>

When you open https://localhost:32788, you should see the Swagger UI.

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2020/04/Testing-the-application-with-HTTPS.jpg"><img loading="lazy" size-full" src="/assets/img/posts/2020/04/Testing-the-application-with-HTTPS.jpg" alt="Testing ASP .Net Core with HTTPS in Docker" /></a>
  
  <p>
    Testing the application with HTTPS
  </p>
</div>

### Explaining the Docker Parameter

In this section, I will shortly explain the used parameter from the example above.

  * -p maps the container from the host inside the container. -p 32789:80 means that the container is listening on port 32789 and redirects it to port 80 inside the application.
  * -e is used to provide an environment variable. Kestrel\_\_Certificates\_\_Default\_\_Path tells the application where the certificate is located and Kestrel\_\_Certificates\_\_Default\_\_Password tells the application the password of the certificate. &#8220;ASPNETCORE_URLS=https://+;http://+&#8221; tells Kestrel to listen for HTTP and HTTPS requests.
  * -v creates a volume that allows you to share files from your computer with the container.

### Using the Certificate in the Container

I created a certificate and copied it into the container during the build. To do that you have to remove .pfx from the .gitignore file. Note that you should never share your certificate or put it inside a container. I only did it to simplify this demo. To use the certificate inside the container, use the following command:

```powershell  
docker run -p 32789:80 -p 32788:443 -e Kestrel\_\_Certificates\_\_Default\_\_Path=/app/Infrastructure/Certificate/cert-aspnetcore.pfx -e Kestrel\_\_Certificates\_\_Default\_\_Password=SecretPassword -e "ASPNETCORE_URLS=https://+;http://+" wolfgangofner/customerapi  
```

## Conclusion

This post showed how to create a certificate and how to provide it to your application inside a Docker container. This enables you to use ASP .NET Coree with HTTPS in Docker.

<a href="/set-up-docker-compose-for-asp-net-core-3-1-microservices" target="_blank" rel="noopener noreferrer">In my next post</a>, I will create a docker-compose file which will help you to start both microservices and RabbitMQ with a single command.

You can find the code of  the finished demo on <a href="https://github.com/WolfgangOfner/MicroserviceDemo" target="_blank" rel="noopener noreferrer">GitHub</a>.