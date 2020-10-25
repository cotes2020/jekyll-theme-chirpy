---
title: Create a .NET Core CI Pipeline in Azure DevOps
date: 2020-08-03T19:24:06+02:00
author: Wolfgang Ofner
layout: post
categories: [DevOps]
tags: [.net core, Azure Devops, CI, continuous integration, DevOps]
---
A crucial feature of DevOps is to give the developer fast feedback if their code changes work. This can be done by automatically building code and running tests every time changes are checked-in. Today, I will show how to create a CI pipeline (continuous integration) for ASP .NET Core.

## Create a .NET Core CI Pipeline in Azure DevOps

In my <a href="/creating-a-microservice-with-net-core-3-1/" target="_blank" rel="noopener noreferrer">last post, Creating a Microservice with .NET Core 3.1</a>, I created a new ASP .NET Core microservice. I will use the CI pipeline to build all projects and run all unit tests in the repository. You can find the code of the demo on <a href="https://github.com/WolfgangOfner/.NetCoreMicroserviceCiCdAks/tree/NetCoreCiPipeline" target="_blank" rel="noopener noreferrer">Github</a>.

In your Azure DevOps project, go to Pipelines and click Create Pipeline.

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2020/08/Create-a-new-CI-Pipeline.jpg"><img loading="lazy" src="/assets/img/posts/2020/08/Create-a-new-CI-Pipeline.jpg" alt="Create a new CI Pipeline" /></a>
  
  <p>
    Create a new CI Pipeline
  </p>
</div>

In the next window, select where you have your code stored. I select GitHub for this Demo. Usually, I have my code directly in Azure DevOps, then I would select Azure Repos Git. On the bottom, you can see &#8220;Use the classic editor&#8221;. This opens the old task-based editor. You shouldn&#8217;t use this anymore since the new standard is to use YML pipelines. This enables you to have your pipeline in your source control.

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2020/08/Select-where-your-code-is.jpg"><img loading="lazy" src="/assets/img/posts/2020/08/Select-where-your-code-is.jpg" alt="Select where your code is" /></a>
  
  <p>
    Select where your code is
  </p>
</div>

### Authorize Github

Since the code is on GitHub, I have to authorize Azure Pipelines to access my repositories. If the code was on an Azure Repos, this step wouldn&#8217;t be necessary.

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2020/08/Authorize-Azure-Pipelines-to-access-Github.jpg"><img loading="lazy" src="/assets/img/posts/2020/08/Authorize-Azure-Pipelines-to-access-Github.jpg" alt="Authorize Azure Pipelines to access Github" /></a>
  
  <p>
    Authorize Azure Pipelines to access Github
  </p>
</div>

After authorizing Azure Pipelines for GitHub, all your repositories will be displayed. Search and select for the repository, you want to make the CI pipeline for. In my case, I select the .NetCoreMicroserviceCiCdAks repository.

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2020/10/Select-your-repository.jpg"><img loading="lazy" src="/assets/img/posts/2020/10/Select-your-repository.jpg" alt="Select your repository" /></a>
  
  <p>
    Select your repository
  </p>
</div>

On the next window, I have to approve to install Azure Pipelines in my Github repository. This allows Azure DevOps to access the repository and write to the code. This is necessary because the CI pipeline will be added to the source control. Again, this is not necessary when you have your code in Azure DevOps.

### Select a Template

On the next step, select a template for your CI pipeline. Azure DevOps offers many templates like Docker, Kubernetes, PHP, or Node.js. Since my application is a .NET Core microservice, I select the ASP.NET Core template.

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2020/08/Select-the-ASP.NET-Core-template-for-your-CI-Pipeline.jpg"><img loading="lazy" src="/assets/img/posts/2020/08/Select-the-ASP.NET-Core-template-for-your-CI-Pipeline.jpg" alt="Select the ASP.NET Core template for your CI Pipeline" /></a>
  
  <p>
    Select the ASP.NET Core template for your CI Pipeline
  </p>
</div>

That&#8217;s it. The template created a simple CI pipeline and you can use it to build your .NET Core solution. In the next section, I will go into more detail about the functionality and add more steps.

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2020/10/The-template-created-a-simple-CI-Pipeline.jpg"><img loading="lazy" src="/assets/img/posts/2020/10/The-template-created-a-simple-CI-Pipeline.jpg" alt="The template created a simple CI Pipeline" /></a>
  
  <p>
    The template created a simple CI Pipeline
  </p>
</div>

## Add more steps to the CI Pipeline

Before we add more steps to the CI pipeline, let&#8217;s have a look at what the template created.

### Analyze the CI Pipeline from the Template

Above the .yml editor, you can see the path to the pipeline yml file in your source control. In my case it is WolfgangOfner/.NetCoreMicroserviceCiCdAks/dotnetcoreCIPipeline.yml. I renamed the file because I want to add more later.

Line 1 through 8 configures the pipeline. The trigger section defines that the pipeline is automatically triggered when something changes on the master branch. The pool section defines that the pipeline is executed on an ubuntu agent and the variables section lets you define variables for the pipeline. By default, only the buildConfiguration is set to Release.

On line 10 starts the first build step that executes a dotnet build in sets the configuration to Release by using the previously defined buildConfiguration variable. Additionally, a display name is set to identify the step easier in the logs.

The .yml editor can be a pain and overwhelming especially when you are starting with pipelines. Once you are used to it, it is way better than the old editor.

### Add more steps to the CI Pipeline

I plan to add several new steps to restore NuGet packages, build the solution, run tests, and publish the solution. The publish step should only be run when the pipeline was triggered by the master branch.

I remove the build script and select the .NET Core task on the right side. I select restore as command and *\*/\*.csproj as the path to the projects. This will restore all available projects. Then click Add, to add the task to your pipeline. Make sure that your cursor is at the beginning of the next line under steps.

&nbsp;

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2020/10/Add-a-Nuget-restore-step-to-the-CI-Pipeline.jpg"><img loading="lazy" src="/assets/img/posts/2020/10/Add-a-Nuget-restore-step-to-the-CI-Pipeline.jpg" alt="Add a Nuget restore step to the CI Pipeline" /></a>
  
  <p>
    Add a Nuget restore step to the CI Pipeline
  </p>
</div>

I repeat the process of adding new .NET Core tasks but I use build to build all projects, test to run all projects that have Test at the end of the project name, and then publish the CustomerApi project. The whole pipeline looks as follows:

```yaml  
trigger:
- master
 
pool:
  vmImage: 'ubuntu-latest'
 
variables:
  buildConfiguration: 'Release'
 
steps:
- task: DotNetCoreCLI@2
  inputs:
    command: 'restore'
    projects: '**/*.csproj'
  displayName: 'Restore Nuget Packages'
 
- task: DotNetCoreCLI@2
  inputs:
    command: 'build'
    projects: '**/*.csproj'
    arguments: '--no-restore'
  displayName: 'Build projects'
 
- task: DotNetCoreCLI@2
  inputs:
    command: 'test'
    projects: '**/*Test.csproj'
    arguments: '--no-restore'
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

Click Save and Run and and the pipeline will be added to your source control and then executed.I created a new branch to test if everything is fine.

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2020/10/Running-the-CI-Pipeline.jpg"><img loading="lazy" src="/assets/img/posts/2020/10/Running-the-CI-Pipeline.jpg" alt="Running the CI Pipeline" /></a>
  
  <p>
    Running the CI Pipeline
  </p>
</div>

After the build is finished, you see a summary and that all 52 tests passed.

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2020/10/All-unit-tests-passed.jpg"><img loading="lazy" src="/assets/img/posts/2020/10/All-unit-tests-passed.jpg" alt="All unit tests passed" /></a>
  
  <p>
    All unit tests passed
  </p>
</div>

You don&#8217;t see anything under Code Coverage. I will cover this in a later post.

### Run Tasks only when the Master Branch triggered the build

Currently, the publish task runs always, even if I don&#8217;t want to create a release. It would be more efficient to run this task only when the build was triggered by the master branch. To do that, I add a custom condition to the publish task. I want to run the publish only when the previous steps succeeded and when the branch name is master. I do this with the following code:

```yaml  
- task: DotNetCoreCLI@2
  inputs:
    command: 'publish'
    publishWebProjects: false
    projects: '**/CustomerApi.csproj'
    arguments: '--configuration $(buildConfiguration) --no-restore'
    modifyOutputPath: false    
  displayName: 'Publish CustomerApi'
  condition: and(succeeded(), eq(variables['Build.SourceBranch'], 'refs/heads/master')) 
```

Save the pipeline and run it with any branch but the master branch. You will see that the publish task is skipped.

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2020/08/The-Publish-Task-got-skipped.jpg"><img loading="lazy" src="/assets/img/posts/2020/08/The-Publish-Task-got-skipped.jpg" alt="The Publish Task got skipped" /></a>
  
  <p>
    The Publish Task got skipped
  </p>
</div>

## Conclusion

CI pipelines help developers to get fast feedback about their code changes. These pipelines can build code and run tests every time something changed. In my next post, I will show how to add code coverage to the results to get even more information about the code changes, and then I will show how to run your code with every pull request. Later, I will extend the pipeline to build and create Docker images and also to deploy them to Kubernetes.

You can find the code of the demo on <a href="https://github.com/WolfgangOfner/.NetCoreMicroserviceCiCdAks/tree/NetCoreCiPipeline" target="_blank" rel="noopener noreferrer">Github</a>.