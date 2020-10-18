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

In my <a href="https://www.programmingwithwolfgang.com/creating-a-microservice-with-net-core-3-1/" target="_blank" rel="noopener noreferrer">last post, Creating a Microservice with .NET Core 3.1</a>, I created a new ASP .NET Core microservice. I will use the CI pipeline to build all projects and run all unit tests in the repository. You can find the code of the demo on <a href="https://github.com/WolfgangOfner/.NetCoreMicroserviceCiCdAks/tree/NetCoreCiPipeline" target="_blank" rel="noopener noreferrer">Github</a>.

In your Azure DevOps project, go to Pipelines and click Create Pipeline.

<div id="attachment_2306" style="width: 710px" class="wp-caption aligncenter">
  <a href="https://www.programmingwithwolfgang.com/wp-content/uploads/2020/08/Create-a-new-CI-Pipeline.jpg"><img aria-describedby="caption-attachment-2306" loading="lazy" class="wp-image-2306" src="https://www.programmingwithwolfgang.com/wp-content/uploads/2020/08/Create-a-new-CI-Pipeline.jpg" alt="Create a new CI Pipeline" width="700" height="361" srcset="https://www.programmingwithwolfgang.com/wp-content/uploads/2020/08/Create-a-new-CI-Pipeline.jpg 1332w, https://www.programmingwithwolfgang.com/wp-content/uploads/2020/08/Create-a-new-CI-Pipeline-300x155.jpg 300w, https://www.programmingwithwolfgang.com/wp-content/uploads/2020/08/Create-a-new-CI-Pipeline-1024x528.jpg 1024w, https://www.programmingwithwolfgang.com/wp-content/uploads/2020/08/Create-a-new-CI-Pipeline-768x396.jpg 768w" sizes="(max-width: 700px) 100vw, 700px" /></a>
  
  <p id="caption-attachment-2306" class="wp-caption-text">
    Create a new CI Pipeline
  </p>
</div>

In the next window, select where you have your code stored. I select GitHub for this Demo. Usually, I have my code directly in Azure DevOps, then I would select Azure Repos Git. On the bottom, you can see &#8220;Use the classic editor&#8221;. This opens the old task-based editor. You shouldn&#8217;t use this anymore since the new standard is to use YML pipelines. This enables you to have your pipeline in your source control.

<div id="attachment_2307" style="width: 608px" class="wp-caption aligncenter">
  <a href="https://www.programmingwithwolfgang.com/wp-content/uploads/2020/08/Select-where-your-code-is.jpg"><img aria-describedby="caption-attachment-2307" loading="lazy" class="size-full wp-image-2307" src="https://www.programmingwithwolfgang.com/wp-content/uploads/2020/08/Select-where-your-code-is.jpg" alt="Select where your code is" width="598" height="571" srcset="https://www.programmingwithwolfgang.com/wp-content/uploads/2020/08/Select-where-your-code-is.jpg 598w, https://www.programmingwithwolfgang.com/wp-content/uploads/2020/08/Select-where-your-code-is-300x286.jpg 300w" sizes="(max-width: 598px) 100vw, 598px" /></a>
  
  <p id="caption-attachment-2307" class="wp-caption-text">
    Select where your code is
  </p>
</div>

### Authorize Github

Since the code is on GitHub, I have to authorize Azure Pipelines to access my repositories. If the code was on an Azure Repos, this step wouldn&#8217;t be necessary.

<div id="attachment_2308" style="width: 535px" class="wp-caption aligncenter">
  <a href="https://www.programmingwithwolfgang.com/wp-content/uploads/2020/08/Authorize-Azure-Pipelines-to-access-Github.jpg"><img aria-describedby="caption-attachment-2308" loading="lazy" class="wp-image-2308" src="https://www.programmingwithwolfgang.com/wp-content/uploads/2020/08/Authorize-Azure-Pipelines-to-access-Github.jpg" alt="Authorize Azure Pipelines to access Github" width="525" height="700" srcset="https://www.programmingwithwolfgang.com/wp-content/uploads/2020/08/Authorize-Azure-Pipelines-to-access-Github.jpg 557w, https://www.programmingwithwolfgang.com/wp-content/uploads/2020/08/Authorize-Azure-Pipelines-to-access-Github-225x300.jpg 225w" sizes="(max-width: 525px) 100vw, 525px" /></a>
  
  <p id="caption-attachment-2308" class="wp-caption-text">
    Authorize Azure Pipelines to access Github
  </p>
</div>

After authorizing Azure Pipelines for GitHub, all your repositories will be displayed. Search and select for the repository, you want to make the CI pipeline for. In my case, I select the .NetCoreMicroserviceCiCdAks repository.

<div id="attachment_2492" style="width: 697px" class="wp-caption aligncenter">
  <a href="https://www.programmingwithwolfgang.com/wp-content/uploads/2020/10/Select-your-repository.jpg"><img aria-describedby="caption-attachment-2492" loading="lazy" class="size-full wp-image-2492" src="https://www.programmingwithwolfgang.com/wp-content/uploads/2020/10/Select-your-repository.jpg" alt="Select your repository" width="687" height="295" srcset="https://www.programmingwithwolfgang.com/wp-content/uploads/2020/10/Select-your-repository.jpg 687w, https://www.programmingwithwolfgang.com/wp-content/uploads/2020/10/Select-your-repository-300x129.jpg 300w" sizes="(max-width: 687px) 100vw, 687px" /></a>
  
  <p id="caption-attachment-2492" class="wp-caption-text">
    Select your repository
  </p>
</div>

On the next window, I have to approve to install Azure Pipelines in my Github repository. This allows Azure DevOps to access the repository and write to the code. This is necessary because the CI pipeline will be added to the source control. Again, this is not necessary when you have your code in Azure DevOps.

### Select a Template

On the next step, select a template for your CI pipeline. Azure DevOps offers many templates like Docker, Kubernetes, PHP, or Node.js. Since my application is a .NET Core microservice, I select the ASP.NET Core template.

<div id="attachment_2314" style="width: 493px" class="wp-caption aligncenter">
  <a href="https://www.programmingwithwolfgang.com/wp-content/uploads/2020/08/Select-the-ASP.NET-Core-template-for-your-CI-Pipeline.jpg"><img aria-describedby="caption-attachment-2314" loading="lazy" class="size-full wp-image-2314" src="https://www.programmingwithwolfgang.com/wp-content/uploads/2020/08/Select-the-ASP.NET-Core-template-for-your-CI-Pipeline.jpg" alt="Select the ASP.NET Core template for your CI Pipeline" width="483" height="633" srcset="https://www.programmingwithwolfgang.com/wp-content/uploads/2020/08/Select-the-ASP.NET-Core-template-for-your-CI-Pipeline.jpg 483w, https://www.programmingwithwolfgang.com/wp-content/uploads/2020/08/Select-the-ASP.NET-Core-template-for-your-CI-Pipeline-229x300.jpg 229w" sizes="(max-width: 483px) 100vw, 483px" /></a>
  
  <p id="caption-attachment-2314" class="wp-caption-text">
    Select the ASP.NET Core template for your CI Pipeline
  </p>
</div>

That&#8217;s it. The template created a simple CI pipeline and you can use it to build your .NET Core solution. In the next section, I will go into more detail about the functionality and add more steps.

<div id="attachment_2493" style="width: 587px" class="wp-caption aligncenter">
  <a href="https://www.programmingwithwolfgang.com/wp-content/uploads/2020/10/The-template-created-a-simple-CI-Pipeline.jpg"><img aria-describedby="caption-attachment-2493" loading="lazy" class="size-full wp-image-2493" src="https://www.programmingwithwolfgang.com/wp-content/uploads/2020/10/The-template-created-a-simple-CI-Pipeline.jpg" alt="The template created a simple CI Pipeline" width="577" height="418" srcset="https://www.programmingwithwolfgang.com/wp-content/uploads/2020/10/The-template-created-a-simple-CI-Pipeline.jpg 577w, https://www.programmingwithwolfgang.com/wp-content/uploads/2020/10/The-template-created-a-simple-CI-Pipeline-300x217.jpg 300w" sizes="(max-width: 577px) 100vw, 577px" /></a>
  
  <p id="caption-attachment-2493" class="wp-caption-text">
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

<div id="attachment_2494" style="width: 338px" class="wp-caption aligncenter">
  <a href="https://www.programmingwithwolfgang.com/wp-content/uploads/2020/10/Add-a-Nuget-restore-step-to-the-CI-Pipeline.jpg"><img aria-describedby="caption-attachment-2494" loading="lazy" class="wp-image-2494" src="https://www.programmingwithwolfgang.com/wp-content/uploads/2020/10/Add-a-Nuget-restore-step-to-the-CI-Pipeline.jpg" alt="Add a Nuget restore step to the CI Pipeline" width="328" height="700" srcset="https://www.programmingwithwolfgang.com/wp-content/uploads/2020/10/Add-a-Nuget-restore-step-to-the-CI-Pipeline.jpg 359w, https://www.programmingwithwolfgang.com/wp-content/uploads/2020/10/Add-a-Nuget-restore-step-to-the-CI-Pipeline-141x300.jpg 141w" sizes="(max-width: 328px) 100vw, 328px" /></a>
  
  <p id="caption-attachment-2494" class="wp-caption-text">
    Add a Nuget restore step to the CI Pipeline
  </p>
</div>

I repeat the process of adding new .NET Core tasks but I use build to build all projects, test to run all projects that have Test at the end of the project name, and then publish the CustomerApi project. The whole pipeline looks as follows:

[code language=&#8221;text&#8221;]  
trigger:  
&#8211; master

pool:  
vmImage: &#8216;ubuntu-latest&#8217;

variables:  
buildConfiguration: &#8216;Release&#8217;

steps:  
&#8211; task: DotNetCoreCLI@2  
inputs:  
command: &#8216;restore&#8217;  
projects: &#8216;*\*/\*.csproj&#8217;  
displayName: &#8216;Restore Nuget Packages&#8217;

&#8211; task: DotNetCoreCLI@2  
inputs:  
command: &#8216;build&#8217;  
projects: &#8216;*\*/\*.csproj&#8217;  
arguments: &#8216;&#8211;no-restore&#8217;  
displayName: &#8216;Build projects&#8217;

&#8211; task: DotNetCoreCLI@2  
inputs:  
command: &#8216;test&#8217;  
projects: &#8216;*\*/\*Test.csproj&#8217;  
arguments: &#8216;&#8211;no-restore&#8217;  
displayName: &#8216;Run Tests&#8217;

&#8211; task: DotNetCoreCLI@2  
inputs:  
command: &#8216;publish&#8217;  
publishWebProjects: false  
projects: &#8216;**/CustomerApi.csproj&#8217;  
arguments: &#8216;&#8211;configuration $(buildConfiguration) &#8211;no-restore&#8217;  
modifyOutputPath: false  
displayName: &#8216;Publish CustomerApi&#8217;  
[/code]

Click Save and Run and and the pipeline will be added to your source control and then executed.I created a new branch to test if everything is fine.

<div id="attachment_2495" style="width: 710px" class="wp-caption aligncenter">
  <a href="https://www.programmingwithwolfgang.com/wp-content/uploads/2020/10/Running-the-CI-Pipeline.jpg"><img aria-describedby="caption-attachment-2495" loading="lazy" class="wp-image-2495" src="https://www.programmingwithwolfgang.com/wp-content/uploads/2020/10/Running-the-CI-Pipeline.jpg" alt="Running the CI Pipeline" width="700" height="307" srcset="https://www.programmingwithwolfgang.com/wp-content/uploads/2020/10/Running-the-CI-Pipeline.jpg 1193w, https://www.programmingwithwolfgang.com/wp-content/uploads/2020/10/Running-the-CI-Pipeline-300x132.jpg 300w, https://www.programmingwithwolfgang.com/wp-content/uploads/2020/10/Running-the-CI-Pipeline-1024x450.jpg 1024w, https://www.programmingwithwolfgang.com/wp-content/uploads/2020/10/Running-the-CI-Pipeline-768x337.jpg 768w" sizes="(max-width: 700px) 100vw, 700px" /></a>
  
  <p id="caption-attachment-2495" class="wp-caption-text">
    Running the CI Pipeline
  </p>
</div>

After the build is finished, you see a summary and that all 52 tests passed.

<div id="attachment_2491" style="width: 1018px" class="wp-caption aligncenter">
  <a href="https://www.programmingwithwolfgang.com/wp-content/uploads/2020/10/All-unit-tests-passed.jpg"><img aria-describedby="caption-attachment-2491" loading="lazy" class="size-full wp-image-2491" src="https://www.programmingwithwolfgang.com/wp-content/uploads/2020/10/All-unit-tests-passed.jpg" alt="All unit tests passed" width="1008" height="783" srcset="https://www.programmingwithwolfgang.com/wp-content/uploads/2020/10/All-unit-tests-passed.jpg 1008w, https://www.programmingwithwolfgang.com/wp-content/uploads/2020/10/All-unit-tests-passed-300x233.jpg 300w, https://www.programmingwithwolfgang.com/wp-content/uploads/2020/10/All-unit-tests-passed-768x597.jpg 768w" sizes="(max-width: 1008px) 100vw, 1008px" /></a>
  
  <p id="caption-attachment-2491" class="wp-caption-text">
    All unit tests passed
  </p>
</div>

You don&#8217;t see anything under Code Coverage. I will cover this in a later post.

### Run Tasks only when the Master Branch triggered the build

Currently, the publish task runs always, even if I don&#8217;t want to create a release. It would be more efficient to run this task only when the build was triggered by the master branch. To do that, I add a custom condition to the publish task. I want to run the publish only when the previous steps succeeded and when the branch name is master. I do this with the following code:

[code language=&#8221;text&#8221;]  
&#8211; task: DotNetCoreCLI@2  
inputs:  
command: &#8216;publish&#8217;  
publishWebProjects: false  
projects: &#8216;**/CustomerApi.csproj&#8217;  
arguments: &#8216;&#8211;configuration $(buildConfiguration) &#8211;no-restore&#8217;  
modifyOutputPath: false  
displayName: &#8216;Publish CustomerApi&#8217;  
condition: and(succeeded(), eq(variables[&#8216;Build.SourceBranch&#8217;], &#8216;refs/heads/master&#8217;))  
[/code]

Save the pipeline and run it with any branch but the master branch. You will see that the publish task is skipped.

<div id="attachment_2509" style="width: 710px" class="wp-caption aligncenter">
  <a href="https://www.programmingwithwolfgang.com/wp-content/uploads/2020/08/The-Publish-Task-got-skipped.jpg"><img aria-describedby="caption-attachment-2509" loading="lazy" class="wp-image-2509" src="https://www.programmingwithwolfgang.com/wp-content/uploads/2020/08/The-Publish-Task-got-skipped.jpg" alt="The Publish Task got skipped" width="700" height="347" srcset="https://www.programmingwithwolfgang.com/wp-content/uploads/2020/08/The-Publish-Task-got-skipped.jpg 1020w, https://www.programmingwithwolfgang.com/wp-content/uploads/2020/08/The-Publish-Task-got-skipped-300x149.jpg 300w, https://www.programmingwithwolfgang.com/wp-content/uploads/2020/08/The-Publish-Task-got-skipped-768x380.jpg 768w" sizes="(max-width: 700px) 100vw, 700px" /></a>
  
  <p id="caption-attachment-2509" class="wp-caption-text">
    The Publish Task got skipped
  </p>
</div>

## Conclusion

CI pipelines help developers to get fast feedback about their code changes. These pipelines can build code and run tests every time something changed. In my next post, I will show how to add code coverage to the results to get even more information about the code changes, and then I will show how to run your code with every pull request. Later, I will extend the pipeline to build and create Docker images and also to deploy them to Kubernetes.

You can find the code of the demo on <a href="https://github.com/WolfgangOfner/.NetCoreMicroserviceCiCdAks/tree/NetCoreCiPipeline" target="_blank" rel="noopener noreferrer">Github</a>.