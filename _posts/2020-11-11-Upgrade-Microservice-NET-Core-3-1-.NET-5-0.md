---
title: Upgrade a Microservice from .NET Core 3.1 to .NET 5.0
date: 2020-11-11
author: Wolfgang Ofner
categories: [ASP.NET, Docker]
tags: [Azure DevOps, CI, docker, xUnit, NET 5.0]
---

Microsoft released the next major release, called .NET 5.0 which succeeds .NET Core 3.1. .NET 5.0 comes with a lot of improvements and also with C# 9. It also is the first step of a unified .NET platform and is the first version of Microsofts new release cycle. From now on, Microsoft will release a new version of .NET every November. .NET 6.0 will be released in November 2021, .NET 7.0 in November 2022, and so on.

Today, I want to show how to upgrade a microservice and its build pipeline from .NET Core 3.1 to .NET 5.0. You can find the code of this demo on [Github](https://github.com/WolfgangOfner/MicroserviceDemo).

## System Requirements for .NET 5.0
To use .NET 5.0 you have to install the .NET 5.0 SDK from the [dotnet download page](https://dotnet.microsoft.com/download/dotnet/5.0) and [Visual Studio 16.8 or later](https://visualstudio.microsoft.com/downloads).

## Uprgrade from .NET Core 3.1 to .NET 5.0
To upgrade your solution to .NET 5.0, you have to update the TargetFramework in every .csproj file of your solution. Replace 

```xml  
<TargetFramework>netcoreapp3.1</TargetFramework>
```
with
```xml  
<TargetFramework>net5.0</TargetFramework>
```

Instead of updating all project files and next year updating them again, I created a new file called common.props in the root folder of the solution. This file contains the following code:

```xml  
<?xml version="1.0" encoding="utf-8"?>
<Project xmlns="http://schemas.microsoft.com/developer/msbuild/2003">

  <PropertyGroup>
    <DefaultTargetFramework>net5.0</DefaultTargetFramework>
  </PropertyGroup>

  <PropertyGroup Label="C#">
    <LangVersion>latest</LangVersion>
    <TargetLatestRuntimePatch>true</TargetLatestRuntimePatch>
  </PropertyGroup>

</Project>
```

This file defines the C# version I am using and sets DefaultTargetFramework to net5.0. Additionally, I have a Directory.Build.props file with the following content:

```xml  
<?xml version="1.0" encoding="utf-8"?>
<Project xmlns="http://schemas.microsoft.com/developer/msbuild/2003">

  <ImportGroup Condition=" '$(MSBuildProjectExtension)' == '.csproj' ">
    <Import Project=".\file-version.props" />
      <Import Project=".\common.props" />
  </ImportGroup>

</Project>
```

This file links the common.props file to the .csproj files. The file-version file is not needed now and I will talk about it in a later post.

After setting this up, I can use this variable in my project files and can update with it all my projects with one change in a single file. Update the TargetFramework of all your .csproj files with the following code:

```xml  
<TargetFramework>$(DefaultTargetFramework)</TargetFramework>
```

After updating all project files, update all Nuget packages of your solution. You can do this by right-clicking your solution --> Manage Nuget Packages for Solution...

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2020/11/Update-Nuget-packages.JPG"><img loading="lazy" src="/assets/img/posts/2020/11/Update-Nuget-packages.JPG" alt="Update Nuget packages" /></a>
  
  <p>
    Update your Nuget packages
  </p>
</div>

That's it. Your solution is updated to .NET 5.0. Build the solution to check that you have no errors.

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2020/11/Build-the-solution.JPG"><img loading="lazy" src="/assets/img/posts/2020/11/Build-the-solution.JPG" alt="Build the solution" /></a>
  
  <p>
    Build the solution
  </p>
</div>

Additionally, run all your tests to make sure your code still works.

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2020/11/Run-all-unit-tests.JPG"><img loading="lazy" src="/assets/img/posts/2020/11/Run-all-unit-tests.JPG" alt="Run all unit tests" /></a>
  
  <p>
    Run all unit tests
  </p>
</div>

Lastly, I update the path to the XML comments in the CustomerApi.csproj file with the following code:

```xml  
<PropertyGroup Condition="'$(Configuration)|$(Platform)'=='Debug|AnyCPU'">
  <DocumentationFile>obj\Release\net5.0\CustomerApi.xml</DocumentationFile>
  <NoWarn>1701;1702;1591</NoWarn>
</PropertyGroup>

<PropertyGroup Condition="'$(Configuration)|$(Platform)'=='Release|AnyCPU'">
  <DocumentationFile>obj\Release\net5.0\CustomerApi.xml</DocumentationFile>
  <NoWarn>1701;1702;1591</NoWarn>
</PropertyGroup>
```

## Update CI pipeline

There are no changes required in the CI pipeline because the solution is built-in Docker. Therefore, I have to update the Dockerfile. Replace the following two lines:

```docker
FROM mcr.microsoft.com/dotnet/core/aspnet:3.1-buster-slim AS base
FROM mcr.microsoft.com/dotnet/core/sdk:3.1-buster AS build
```

with 

```docker
FROM mcr.microsoft.com/dotnet/aspnet:5.0 AS base
FROM mcr.microsoft.com/dotnet/sdk:5.0 AS build
```

This tells Docker to use the new .NET 5.0 images to build and run the application. Additionally, I have to copy the .props files into my Docker image with the following code inside the Dockerfile:

```docker
COPY ["*.props", "./"]
```

Check in your changes and the build in Azure DevOps will run successfully.

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2020/11/The-Net-5-build-was-successful.JPG"><img loading="lazy" src="/assets/img/posts/2020/11/The-Net-5-build-was-successful.JPG" alt="The .NET 5.0 build was successful" /></a>
  
  <p>
    The .NET 5.0 build was successful
  </p>
</div>

## Conclusion
Today, I showed how easy it can be to upgrade .NET Core 3.1 to .NET 5.0. To upgrade was so easy because I kept my solution up to date and because microservices are small solutions that are way easier to upgrade than big monolithic applications. The whole upgrade for both my microservices took around 10 minutes. I know that a real-world microservice will have more code than mine but nevertheless, it is quite easy to update it. If you are coming from .NET Core 2.x or even .NET 4.x, the upgrade might be harder.

You can find the code of this demo on [Github](https://github.com/WolfgangOfner/MicroserviceDemo).