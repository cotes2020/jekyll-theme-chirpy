---
title: Create Automatic Builds for .Net and .Net Core Applications with Azure Devops
date: 2019-11-18T16:15:30+01:00
author: Wolfgang Ofner
categories: [DevOps]
tags: [.net core, ASP.NET Core MVC, ASP.NET MVC, Azure Devops, Azure Devops Services Continous Integration]
---
Automated build processes should be a no-brainer nowadays but unfortunately, I still come across quite many projects which do their builds manually. In today&#8217;s post, I want to give an introduction to setting up an automated build pipeline for a .net and .net core application with Microsoft&#8217;s Azure DevOps Server. The DevOps Server is the on-premise version of Azure DevOps Services, but they are so similar that its negligible which one you use for this demo. I will use Azure DevOps Services. You can create a free account <a href="https://azure.microsoft.com/en-us/services/devops/" target="_blank" rel="noopener noreferrer">here</a>.

## Creating your first Build Pipeline for a .net Framework Application

In your Azure DevOps project go to Pipelines and then again Pipelines. There click on Create Pipeline and the wizard starts.

<div id="attachment_1785" style="width: 703px" class="wp-caption aligncenter">
  <a href="/assets/img/posts/2019/11/Start-the-create-pipeline-wizzard.jpg"><img aria-describedby="caption-attachment-1785" loading="lazy" class="wp-image-1785 size-full" src="/assets/img/posts/2019/11/Start-the-create-pipeline-wizzard.jpg" alt="Start the create pipeline wizard" width="693" height="591" /></a>
  
  <p id="caption-attachment-1785" class="wp-caption-text">
    Start the create pipeline wizard
  </p>
</div>

Here you can use either the new YAML editor which is selected by default or the classic editor. I use the classic editor by clicking on Use the classic editor at the bottom. In the next window, select the source of your code. The editor offers a wide range of repositories like Azure Repos Git, GitHub or even Subversion. Since I manage my code in Azure DevOps Services, I select Azure Repos Git and the wizard automatically selects your current project, repository and default branch. Click Continue to get to the template selection.

<div id="attachment_1786" style="width: 710px" class="wp-caption aligncenter">
  <a href="/assets/img/posts/2019/11/Select-the-repository-for-your-pipeline.jpg"><img aria-describedby="caption-attachment-1786" loading="lazy" class="wp-image-1786" src="/assets/img/posts/2019/11/Select-the-repository-for-your-pipeline.jpg" alt="Select the repository for your pipeline" width="700" height="438" /></a>
  
  <p id="caption-attachment-1786" class="wp-caption-text">
    Select the repository for your pipeline
  </p>
</div>

Azure DevOps Services offers a wide range of different templates as a starting point for your pipeline. There are templates for ASP .NET, Docker, Azure Functions and many more. You don&#8217;t have to select a template and could start without one by selecting Empty job or by importing a YAML file by selecting YAML. Since I have an ASP .NET MVC application, I select ASP. NET.

<div id="attachment_1787" style="width: 710px" class="wp-caption aligncenter">
  <a href="/assets/img/posts/2019/11/Select-a-template-for-your-pipeline.jpg"><img aria-describedby="caption-attachment-1787" loading="lazy" class="wp-image-1787" src="/assets/img/posts/2019/11/Select-a-template-for-your-pipeline.jpg" alt="Select a template for your pipeline" width="700" height="444" /></a>
  
  <p id="caption-attachment-1787" class="wp-caption-text">
    Select a template for your pipeline
  </p>
</div>

Click on Apply and the pipeline will be created for you.

<div id="attachment_1788" style="width: 710px" class="wp-caption aligncenter">
  <a href="/assets/img/posts/2019/11/The-created-build-pipeline.jpg"><img aria-describedby="caption-attachment-1788" loading="lazy" class="wp-image-1788" src="/assets/img/posts/2019/11/The-created-build-pipeline.jpg" alt="The created build pipeline" width="700" height="427" /></a>
  
  <p id="caption-attachment-1788" class="wp-caption-text">
    The created build pipeline
  </p>
</div>

### Inspecting the Settings of the Build Pipeline

On top of your build pipeline, you can see six tabs with settings. Let&#8217;s check them out and see what we can configure there.

#### Tasks

The Tasks tab is the heart of your build pipeline and the part where all the magic happens. I will go over the steps in more detail in the next section.

#### Variables

In the Variables tab, you can create variables that can be used in every step of the pipeline. For builds, I barely use variables but you could set the username and password of an external service that you want to call. Azure DevOps already comes with a couple of predefined variables like the build configuration or platform.

<div id="attachment_1789" style="width: 710px" class="wp-caption aligncenter">
  <a href="/assets/img/posts/2019/11/Variables-for-the-build-pipeline.jpg"><img aria-describedby="caption-attachment-1789" loading="lazy" class="wp-image-1789" src="/assets/img/posts/2019/11/Variables-for-the-build-pipeline.jpg" alt="Variables for the build pipeline" width="700" height="221" /></a>
  
  <p id="caption-attachment-1789" class="wp-caption-text">
    Variables for the build pipeline
  </p>
</div>

With the checkbox on the right (Settable at queue time), you can configure that these variables can be changed when you create a build. On the left are also Variable groups. There you can also set variables. The difference is that you can reuse a variable group for several pipelines whereas the Pipeline variables are specific for your pipeline.

#### Triggers

The Triggers tab configures continuous integration. On the following screenshot, you can see that I enabled continuous integration and trigger a build every time a change to a branch with the pattern release/* was made. (When I want to deploy I create a branch from master with the name release/sprint-XX. This puts the sprint-XX branch in the release folder and also triggers the build)

<div id="attachment_1790" style="width: 710px" class="wp-caption aligncenter">
  <a href="/assets/img/posts/2019/11/Enable-continuous-integration.jpg"><img aria-describedby="caption-attachment-1790" loading="lazy" class="wp-image-1790" src="/assets/img/posts/2019/11/Enable-continuous-integration.jpg" alt="Enable continuous integration" width="700" height="344" /></a>
  
  <p id="caption-attachment-1790" class="wp-caption-text">
    Enable continuous integration
  </p>
</div>

It is also possible to schedule builds. For example, if you have integration tests that take a long time to finish, you don&#8217;t want to run them at every commit. You can run them every night, for example, Monday to Friday at 2 am or on the weekend.

<div id="attachment_1791" style="width: 710px" class="wp-caption aligncenter">
  <a href="/assets/img/posts/2019/11/Schedule-builds.jpg"><img aria-describedby="caption-attachment-1791" loading="lazy" class="wp-image-1791" src="/assets/img/posts/2019/11/Schedule-builds.jpg" alt="Schedule builds" width="700" height="326" /></a>
  
  <p id="caption-attachment-1791" class="wp-caption-text">
    Schedule builds
  </p>
</div>

#### Options

Here, you can edit the timeout for your builds and the build number format. I edited the number format to display the branch and then the date and build number. By default, only the date and build number are displayed. To do that, I am using the built-in variable Build.SourceBranchName.

&nbsp;

<div id="attachment_1794" style="width: 710px" class="wp-caption aligncenter">
  <a href="/assets/img/posts/2019/11/Include-the-branch-name-in-the-build-format-number.jpg"><img aria-describedby="caption-attachment-1794" loading="lazy" class="wp-image-1794" src="/assets/img/posts/2019/11/Include-the-branch-name-in-the-build-format-number.jpg" alt="Include the branch name in the build format number" width="700" height="315" /></a>
  
  <p id="caption-attachment-1794" class="wp-caption-text">
    Include the branch name in the build format number
  </p>
</div>

#### Retention

Under this tab, you can configure your retention policies for your builds. The default is to keep 10 good builds for 30 days. I leave this as it is.

<div id="attachment_1795" style="width: 710px" class="wp-caption aligncenter">
  <a href="/assets/img/posts/2019/11/Configure-the-retention-policy-of-your-build-pipeline.jpg"><img aria-describedby="caption-attachment-1795" loading="lazy" class="wp-image-1795" src="/assets/img/posts/2019/11/Configure-the-retention-policy-of-your-build-pipeline.jpg" alt="Configure the retention policy of your build pipeline" width="700" height="416" /></a>
  
  <p id="caption-attachment-1795" class="wp-caption-text">
    Configure the retention policy of your build pipeline
  </p>
</div>

#### History

The history tab shows not the history of your builds but the history of your build pipeline changes. I highly recommend you to always make a comment when you save changes in your pipeline.

<div id="attachment_1796" style="width: 710px" class="wp-caption aligncenter">
  <a href="/assets/img/posts/2019/11/The-history-of-the-changes-to-the-pipeline.jpg"><img aria-describedby="caption-attachment-1796" loading="lazy" class="wp-image-1796" src="/assets/img/posts/2019/11/The-history-of-the-changes-to-the-pipeline.jpg" alt="The history of the changes to the pipeline" width="700" height="367" /></a>
  
  <p id="caption-attachment-1796" class="wp-caption-text">
    The history of the changes to the pipeline
  </p>
</div>

### Inspecting the Build Tasks created by the ASP. NET Template

Now that all the settings are as wished, let&#8217;s look at the steps of the deployment.

#### Pipeline

Here you can configure the agent pool, agent and artifact name for your build. The agent pool groups agents together. An agent builds (and later deploys) your application. Since I am using Azure DevOps Services, I leave the agent pool at Azure Pipelines because I want to use the agents which are hosted by Microsoft. For the agent, I also leave it as it is. If you are running a .Net core build, you could switch to Ubuntu.

#### Get sources

Under Get sources, you can change the project, repository and default branch for your build. You can also configure that a clean should be performed before the build. I set Clean to true and the Clean options to All build directories. Additionally, you could set a tag automatically for each build or each successful build.

<div id="attachment_1798" style="width: 710px" class="wp-caption aligncenter">
  <a href="/assets/img/posts/2019/11/Configure-the-clean-options.jpg"><img aria-describedby="caption-attachment-1798" loading="lazy" class="wp-image-1798" src="/assets/img/posts/2019/11/Configure-the-clean-options.jpg" alt="Configure the clean options" width="700" height="476" /></a>
  
  <p id="caption-attachment-1798" class="wp-caption-text">
    Configure the clean options
  </p>
</div>

#### Use NuGet

The first set of your build pipeline is used to set up the NuGet.exe which will be used in the next step to restore your NuGet packages. I leave all the settings at their default values.

#### NuGet restore

This step restores all the Nuget packages of your solution. By default the NuGet packages in all projects. You can change this by changing the Path to solution from *\*\\*.sln to whatever fits you.

#### Build solution

The build solution step builds your application and also publishes it. The publish is necessary to deploy it later. You can also configure which version of Visual Studio should be used. You can choose from 2012 on all versions. Latest always selects the newest version.

#### Test Assemblies

Here, all test assemblies with test in their names are executed. You can select to run only impacted tests to speed up your build process. This setting executes only tests that were affected by the last commit. You can also configure to run all tests in isolation or even rerun failed tests.

#### Publish symbols path

This task publishes all *.pdb files which can be used for remote debugging.

#### Publish Artifact

Publish Artifact is necessary for an automated deployment. This step publishes all the files which you want to deploy later. If you don&#8217;t do this, the release pipeline won&#8217;t find any files to deploy.

## Running your first build

Now that everything is set up, you can run your first build by clicking Save & queue and then Save and run.

<div id="attachment_1800" style="width: 710px" class="wp-caption aligncenter">
  <a href="/assets/img/posts/2019/11/Run-your-build.jpg"><img aria-describedby="caption-attachment-1800" loading="lazy" class="wp-image-1800" src="/assets/img/posts/2019/11/Run-your-build.jpg" alt="Run your build" width="700" height="394" /></a>
  
  <p id="caption-attachment-1800" class="wp-caption-text">
    Run your build
  </p>
</div>

This starts the build process. You can see the status by clicking on Agent job 1. On the following screenshot, you can see that the build step is already done and the tests are run right now.

<div id="attachment_1801" style="width: 710px" class="wp-caption aligncenter">
  <a href="/assets/img/posts/2019/11/The-status-of-the-running-build-pipeline.jpg"><img aria-describedby="caption-attachment-1801" loading="lazy" class="wp-image-1801" src="/assets/img/posts/2019/11/The-status-of-the-running-build-pipeline.jpg" alt="The status of the running build " width="700" height="477" /></a>
  
  <p id="caption-attachment-1801" class="wp-caption-text">
    The status of the running build pipeline
  </p>
</div>

After a couple of minutes, your build should finish successfully.

## Creating a Build Pipeline for a .net Core Application

Creating a build pipeline for .net core is the same process as for a .net framework application. The only difference is that I select ASP.NET Core as the template this time.

<div id="attachment_1804" style="width: 710px" class="wp-caption aligncenter">
  <a href="/assets/img/posts/2019/11/Select-.net-core-as-your-build-template.jpg"><img aria-describedby="caption-attachment-1804" loading="lazy" class="wp-image-1804" src="/assets/img/posts/2019/11/Select-.net-core-as-your-build-template.jpg" alt="Select .net core as your build template" width="700" height="232" /></a>
  
  <p id="caption-attachment-1804" class="wp-caption-text">
    Select .net core as your build template
  </p>
</div>

You can see a difference in the created build pipeline though.

<div id="attachment_1805" style="width: 710px" class="wp-caption aligncenter">
  <a href="/assets/img/posts/2019/11/The-.net-core-build-pipeline.jpg"><img aria-describedby="caption-attachment-1805" loading="lazy" class="wp-image-1805" src="/assets/img/posts/2019/11/The-.net-core-build-pipeline.jpg" alt="The .net core pipeline" width="700" height="527" /></a>
  
  <p id="caption-attachment-1805" class="wp-caption-text">
    The .net core build pipeline
  </p>
</div>

## Why I like the .net Core Build Pipeline better

There are several reasons why I like the .net core build pipeline better than the one for a .net framework application.

The first thing which got my attention is that all tasks look the same. All tasks except the last one use the dotnet CLI. The only difference is the argument (restore, build, publish and test). This means that I know exactly what&#8217;sgoing on and due to the similarity of the tasks, they are easy to configure.

An even better new feature is that it is possible to zip a project during the publishing process. Before that, you had to use a Powershell script or the Archive files of Azure DevOps to create the zip. But this meant that you have an additional step and also duplicate data since you have the &#8220;normal&#8221; published files and afterwards the zipped ones. Zipping the files is necessary to save storage space but more importantly to speed up the deployment process.

## Conclusion

This post showed how you can quickly set up an automated build pipeline for your .net or .net core application in Azure DevOps. This is the first step to increasing the development velocity and reducing the bugs and later on also the deployment time.