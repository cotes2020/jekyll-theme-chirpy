---
title: Build .NET Core in a CI Pipeline in Azure DevOps
date: 2020-10-13T17:20:53+02:00
author: Wolfgang Ofner
categories: [DevOps]
tags: [.net core, Azure Devops, CI, Continous Integration, DevOps, Github]
---
A crucial feature of DevOps is to give the developer fast feedback if their code changes work. This can be done by automatically building code and running tests every time changes are checked-in. Today, I will show how to create a CI pipeline (continuous integration) for ASP .NET Core.

## Create a .NET Core CI Pipeline in Azure DevOps

<a href="/programming-microservices-net-core-3-1" target="_blank" rel="noopener noreferrer">Over the series of my last posts</a>, I created two ASP .NET Core microservices. In this post, I create a CI pipeline to build all projects and run all unit tests in the repository. You can find the code of the demo on <a href="https://github.com/WolfgangOfner/MicroserviceDemo" target="_blank" rel="noopener noreferrer">Github</a>.

In your Azure DevOps project, go to Pipelines and click Create Pipeline.

<div id="attachment_2306" style="width: 710px" class="wp-caption aligncenter">
  <a href="/assets/img/posts/2020/08/Create-a-new-CI-Pipeline.jpg"><img aria-describedby="caption-attachment-2306" loading="lazy" class="wp-image-2306" src="/assets/img/posts/2020/08/Create-a-new-CI-Pipeline.jpg" alt="Create a new CI Pipeline" width="700" height="361" /></a>
  
  <p id="caption-attachment-2306" class="wp-caption-text">
    Create a new CI Pipeline
  </p>
</div>

In the next window, select where you have your code stored. I select GitHub for this Demo. Usually, I have my code directly in Azure DevOps, then I would select Azure Repos Git. On the bottom, you can see &#8220;Use the classic editor&#8221;. This opens the old task-based editor. You shouldn&#8217;t use this anymore since the new standard is to use YML pipelines. This enables you to have your pipeline in your source control.

<div id="attachment_2307" style="width: 608px" class="wp-caption aligncenter">
  <a href="/assets/img/posts/2020/08/Select-where-your-code-is.jpg"><img aria-describedby="caption-attachment-2307" loading="lazy" class="size-full wp-image-2307" src="/assets/img/posts/2020/08/Select-where-your-code-is.jpg" alt="Select where your code is" width="598" height="571" /></a>
  
  <p id="caption-attachment-2307" class="wp-caption-text">
    Select where your code is
  </p>
</div>

### Authorize Github

Since the code is on GitHub, I have to authorize Azure Pipelines to access my repositories. If the code was on an Azure Repos, this step wouldn&#8217;t be necessary.

<div id="attachment_2308" style="width: 535px" class="wp-caption aligncenter">
  <a href="/assets/img/posts/2020/08/Authorize-Azure-Pipelines-to-access-Github.jpg"><img aria-describedby="caption-attachment-2308" loading="lazy" class="wp-image-2308" src="/assets/img/posts/2020/08/Authorize-Azure-Pipelines-to-access-Github.jpg" alt="Authorize Azure Pipelines to access Github" width="525" height="700" /></a>
  
  <p id="caption-attachment-2308" class="wp-caption-text">
    Authorize Azure Pipelines to access Github
  </p>
</div>

After authorizing Azure Pipelines for GitHub, all your repositories will be displayed. Search and select for the repository, you want to make the CI pipeline for. In my case, I select the MicroserviceDemo repository.

<div id="attachment_2486" style="width: 692px" class="wp-caption aligncenter">
  <a href="/assets/img/posts/2020/10/Find-your-repository.jpg"><img aria-describedby="caption-attachment-2486" loading="lazy" class="size-full wp-image-2486" src="/assets/img/posts/2020/10/Find-your-repository.jpg" alt="Find your repository" width="682" height="290" /></a>
  
  <p id="caption-attachment-2486" class="wp-caption-text">
    Find your repository
  </p>
</div>

On the next window, I have to approve to install Azure Pipelines in my Github repository. This allows Azure DevOps to access the repository and write to the code. This is necessary because the CI pipeline will be added to the source control. Again, this is not necessary when you have your code in Azure DevOps.

<div id="attachment_2487" style="width: 710px" class="wp-caption aligncenter">
  <a href="/assets/img/posts/2020/10/Approve-the-access-to-the-Github-repository.jpg"><img aria-describedby="caption-attachment-2487" loading="lazy" class="wp-image-2487" src="/assets/img/posts/2020/10/Approve-the-access-to-the-Github-repository.jpg" alt="Approve the access to the Github repository" width="700" height="610" /></a>
  
  <p id="caption-attachment-2487" class="wp-caption-text">
    Approve the access to the Github repository
  </p>
</div>

### Select a Template

On the next step, select a template for your CI pipeline. Azure DevOps offers many templates like Docker, Kubernetes, PHP, or Node.js. Since my application is a .NET Core microservice, I select the ASP.NET Core template.

<div id="attachment_2314" style="width: 493px" class="wp-caption aligncenter">
  <a href="/assets/img/posts/2020/08/Select-the-ASP.NET-Core-template-for-your-CI-Pipeline.jpg"><img aria-describedby="caption-attachment-2314" loading="lazy" class="size-full wp-image-2314" src="/assets/img/posts/2020/08/Select-the-ASP.NET-Core-template-for-your-CI-Pipeline.jpg" alt="Select the ASP.NET Core template for your CI Pipeline" width="483" height="633" /></a>
  
  <p id="caption-attachment-2314" class="wp-caption-text">
    Select the ASP.NET Core template for your CI Pipeline
  </p>
</div>

That&#8217;s it. The template created a simple CI pipeline and you can use it to build your .NET Core solution. In the next section, I will go into more detail about the functionality and add more steps.

<div id="attachment_2488" style="width: 541px" class="wp-caption aligncenter">
  <a href="/assets/img/posts/2020/10/The-CI-Pipeline-got-created-from-the-template.jpg"><img aria-describedby="caption-attachment-2488" loading="lazy" class="size-full wp-image-2488" src="/assets/img/posts/2020/10/The-CI-Pipeline-got-created-from-the-template.jpg" alt="The CI Pipeline got created from the template" width="531" height="374" /></a>
  
  <p id="caption-attachment-2488" class="wp-caption-text">
    The CI Pipeline got created from the template
  </p>
</div>

## Add more steps to the CI Pipeline

Before we add more steps to the CI pipeline, let&#8217;s have a look at what the template created.

### Analyze the CI Pipeline from the Template

Above the .yml editor, you can see the path to the pipeline yml file in your source control. I recommend having a pipelines folder of the root folder of your solution. In my case, it is WolfgangOfner/MicroserviceDemo/CustomerApi/pipelines/NetCore-CI-azure-pipeline.yml (and also one for the OrderApi but the principle is the same). It is a good practice to name the file to describe what they do. You can find the finished pipeline on <a href="https://github.com/WolfgangOfner/MicroserviceDemo/blob/master/CustomerApi/pipelines/NetCore-CI-azure-pipeline.yml" target="_blank" rel="noopener noreferrer">Github</a> or a bit further down.

The lines 1 through 14 configure the pipeline. The trigger section defines that the pipeline is automatically triggered when something changes on the master branch inside the CustomerApi folder. This means that this pipeline does not run when you change something on the OrderApi project. The pool section defines that the pipeline is executed on an ubuntu agent and the variables section lets you define variables for the pipeline. By default, only the buildConfiguration is set to Release.

On line 16 starts the first build step that executes a dotnet build in sets the configuration to Release by using the previously defined buildConfiguration variable. Additionally, a display name is set to identify the step easier in the logs.

The .yml editor can be a pain and overwhelming, especially when you are starting with pipelines. Once you are used to it, it is way better than the old one though.

### Add more steps to the CI Pipeline

I plan to add several new steps to restore NuGet packages, build the solution, run tests, and publish the solution. The publish step should only be run when the pipeline was triggered by the master branch.

I remove the build script and select the .NET Core task on the right side. I select restore as command and *\*/CustomerApi\*.csproj as the path to the projects to restore each project starting with CustomerApi. Then click Add, to add the task to your pipeline. Make sure that your cursor is at the beginning of the next line under steps.

<div id="attachment_2489" style="width: 300px" class="wp-caption aligncenter">
  <a href="/assets/img/posts/2020/10/Add-Nuget-restore-to-the-CI-pipeline.jpg"><img aria-describedby="caption-attachment-2489" loading="lazy" class="wp-image-2489" src="/assets/img/posts/2020/10/Add-Nuget-restore-to-the-CI-pipeline.jpg" alt="Add Nuget restore to the CI pipeline" width="290" height="700" /></a>
  
  <p id="caption-attachment-2489" class="wp-caption-text">
    Add Nuget restore to the CI pipeline
  </p>
</div>

I repeat the process of adding new .NET Core tasks but I use build to build all projects, test to run all projects that have Test at the end of the project name, and then publish the CustomerApi project. Since I already built and restored the test projects, you can see that I use the &#8211;no-restore and &#8211;no-build arguments for them. The whole pipeline looks as follows:

```yaml  
name : NetCore-CustomerApi-CI
trigger:
  branches:
    include:
      - master
  paths:
    include:
      - CustomerApi/*
 
pool:
  vmImage: 'ubuntu-latest'
 
variables:
  buildConfiguration: 'Release'
 
steps:
- task: DotNetCoreCLI@2
  inputs:
    command: 'restore'
    projects: '**/CustomerApi*.csproj'
  displayName: 'Restore Nuget Packages'
 
- task: DotNetCoreCLI@2
  inputs:
    command: 'build'
    projects: '**/CustomerApi*.csproj'
    arguments: '--no-restore'
  displayName: 'Build projects'
 
- task: DotNetCoreCLI@2
  inputs:
    command: 'test'
    projects: '**/*Test.csproj'
    arguments: '--no-restore --no-build'
  displayName: 'Run Tests'
 
- task: DotNetCoreCLI@2
  inputs:
    command: 'publish'
    publishWebProjects: false
    projects: '**/CustomerApi.csproj'
    arguments: '--configuration $(buildConfiguration) --no-restore'
    modifyOutputPath: false
  displayName: 'Publish CustomerApi'
```

Click Save and Run the pipeline will be added to your source control and then executed.

<div id="attachment_2490" style="width: 710px" class="wp-caption aligncenter">
  <a href="/assets/img/posts/2020/10/The-CI-Pipeline-is-running.jpg"><img aria-describedby="caption-attachment-2490" loading="lazy" class="wp-image-2490" src="/assets/img/posts/2020/10/The-CI-Pipeline-is-running.jpg" alt="The CI Pipeline is running" width="700" height="424" /></a>
  
  <p id="caption-attachment-2490" class="wp-caption-text">
    The CI Pipeline is running
  </p>
</div>

After the build is finished, you see a summary and that all 38 tests passed.

<div id="attachment_2491" style="width: 710px" class="wp-caption aligncenter">
  <a href="/assets/img/posts/2020/10/All-unit-tests-passed.jpg"><img aria-describedby="caption-attachment-2491" loading="lazy" class="wp-image-2491" src="/assets/img/posts/2020/10/All-unit-tests-passed.jpg" alt="All unit tests passed" width="700" height="544" /></a>
  
  <p id="caption-attachment-2491" class="wp-caption-text">
    All unit tests passed
  </p>
</div>

You don&#8217;t see anything under Code Coverage. I will cover this in a later post.

### Run Tasks only when the Master Branch triggered the build

Currently, the publish task runs always, even if I don&#8217;t want to create a release. It would be more efficient to run this task only when the build was triggered by the master branch. To do that, I add a custom condition to the publish task. I want to run the publish only when the previous steps succeeded and when the build was not triggered by a pull request. I do this with the following code:

```yaml  
- task: DotNetCoreCLI@2
  inputs:
    command: 'publish'
    publishWebProjects: false
    projects: '**/CustomerApi.csproj'
    arguments: '--configuration $(buildConfiguration) --no-restore'
    modifyOutputPath: false    
  displayName: 'Publish CustomerApi'
  condition: and(succeeded(), ne(variables['Build.Reason'], 'PullRequest'))  
```

Adding a second pipeline for the OrderApi works exactly the same except that instead of CustomerApi, you use OrderApi.

## Conclusion

CI pipelines help developers to get fast feedback about their code changes. These pipelines can build code and run tests every time something changed. In my next post, I will show how to add code coverage to the results to get even more information about the code changes, and then I will show how to run your code with every pull request. Later, I will extend the pipeline to build and create Docker images and also to deploy them to Kubernetes.

You can find the code of the demo on <a href="https://github.com/WolfgangOfner/MicroserviceDemo" target="_blank" rel="noopener noreferrer">Github</a>.