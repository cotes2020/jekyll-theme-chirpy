---
title: Create a Docker Image in an Azure DevOps CI Pipeline
date: 2020-09-14T14:34:59+02:00
author: Wolfgang Ofner
layout: post
categories: [DevOps, Docker]
tags: [Azure Devops, CI, DevOps, docker, Docker Hub]
---
<a href="https://www.programmingwithwolfgang.com/add-docker-to-a-asp-net-core-microservice/" target="_blank" rel="noopener noreferrer">In my last post</a>, I showed how to build a .NET Core Microservice inside a Docker container. Today, I want to build this microservice in an Azure DevOps CI pipeline and push the image to Dockerhub.

## Set up a Service Connection to Docker Hub

Before I create the new CI Pipeline for building the Docker image, I set up a connection to Docker Hub to push my image to its repository. To do that in Azure DevOps, click on Project Settings &#8211;> Service connections &#8211;> New service connection.

<div id="attachment_2371" style="width: 710px" class="wp-caption aligncenter">
  <a href="https://www.programmingwithwolfgang.com/wp-content/uploads/2020/09/Create-a-new-service-connection.jpg"><img aria-describedby="caption-attachment-2371" loading="lazy" class="wp-image-2371" src="https://www.programmingwithwolfgang.com/wp-content/uploads/2020/09/Create-a-new-service-connection.jpg" alt="Create a new service connection" width="700" height="353" srcset="https://www.programmingwithwolfgang.com/wp-content/uploads/2020/09/Create-a-new-service-connection.jpg 1850w, https://www.programmingwithwolfgang.com/wp-content/uploads/2020/09/Create-a-new-service-connection-300x151.jpg 300w, https://www.programmingwithwolfgang.com/wp-content/uploads/2020/09/Create-a-new-service-connection-1024x516.jpg 1024w, https://www.programmingwithwolfgang.com/wp-content/uploads/2020/09/Create-a-new-service-connection-768x387.jpg 768w, https://www.programmingwithwolfgang.com/wp-content/uploads/2020/09/Create-a-new-service-connection-1536x775.jpg 1536w" sizes="(max-width: 700px) 100vw, 700px" /></a>
  
  <p id="caption-attachment-2371" class="wp-caption-text">
    Create a new service connection
  </p>
</div>

This opens a pop-up where you select Docker Registry.

<div id="attachment_2372" style="width: 478px" class="wp-caption aligncenter">
  <a href="https://www.programmingwithwolfgang.com/wp-content/uploads/2020/09/Select-Docker-Registry-for-your-service-connection.jpg"><img aria-describedby="caption-attachment-2372" loading="lazy" class="size-full wp-image-2372" src="https://www.programmingwithwolfgang.com/wp-content/uploads/2020/09/Select-Docker-Registry-for-your-service-connection.jpg" alt="Select Docker Registry for your service connection" width="468" height="287" srcset="https://www.programmingwithwolfgang.com/wp-content/uploads/2020/09/Select-Docker-Registry-for-your-service-connection.jpg 468w, https://www.programmingwithwolfgang.com/wp-content/uploads/2020/09/Select-Docker-Registry-for-your-service-connection-300x184.jpg 300w" sizes="(max-width: 468px) 100vw, 468px" /></a>
  
  <p id="caption-attachment-2372" class="wp-caption-text">
    Select Docker Registry for your service connection
  </p>
</div>

On the next page, select Docker Hub as your Registry type, enter your Docker ID, password and set a name for your connection. Then click Verify and save.

<div id="attachment_2373" style="width: 367px" class="wp-caption aligncenter">
  <a href="https://www.programmingwithwolfgang.com/wp-content/uploads/2020/09/Configure-the-service-connection.jpg"><img aria-describedby="caption-attachment-2373" loading="lazy" class="wp-image-2373" src="https://www.programmingwithwolfgang.com/wp-content/uploads/2020/09/Configure-the-service-connection.jpg" alt="Configure the service connection for the DevOps CI pipeline" width="357" height="700" srcset="https://www.programmingwithwolfgang.com/wp-content/uploads/2020/09/Configure-the-service-connection.jpg 480w, https://www.programmingwithwolfgang.com/wp-content/uploads/2020/09/Configure-the-service-connection-153x300.jpg 153w" sizes="(max-width: 357px) 100vw, 357px" /></a>
  
  <p id="caption-attachment-2373" class="wp-caption-text">
    Configure the service connection
  </p>
</div>

## Create a new Azure DevOps CI Pipeline

After setting up the service connection, create a new CI Pipeline. Select the source code location and then any template. After the yml file is created, delete its content. For more details on creating a Pipeline, see my post &#8220;<a href="https://www.programmingwithwolfgang.com/build-net-core-in-a-ci-pipeline-in-azure-devops/" target="_blank" rel="noopener noreferrer">Build .Net Core in a CI Pipeline in Azure DevOps</a>&#8220;.

<div id="attachment_2367" style="width: 553px" class="wp-caption aligncenter">
  <a href="https://www.programmingwithwolfgang.com/wp-content/uploads/2020/09/Create-an-empty-Pipeline.jpg"><img aria-describedby="caption-attachment-2367" loading="lazy" class="wp-image-2367 size-full" src="https://www.programmingwithwolfgang.com/wp-content/uploads/2020/09/Create-an-empty-Pipeline.jpg" alt="Create an empty DevOps CI pipeline" width="543" height="187" srcset="https://www.programmingwithwolfgang.com/wp-content/uploads/2020/09/Create-an-empty-Pipeline.jpg 543w, https://www.programmingwithwolfgang.com/wp-content/uploads/2020/09/Create-an-empty-Pipeline-300x103.jpg 300w" sizes="(max-width: 543px) 100vw, 543px" /></a>
  
  <p id="caption-attachment-2367" class="wp-caption-text">
    Create an empty Pipeline
  </p>
</div>

### Configure the Pipeline

First, you have to set up some basic configuration for the pipeline. I will give it a name, set a trigger to run the pipeline every time a commit is made to master and use an Ubuntu agent. You can do this with the following code:

[code language=&#8221;text&#8221;]  
name : Docker-CI  
trigger:  
branches:  
include:  
&#8211; master

pool:  
vmImage: &#8216;ubuntu-latest&#8217;  
[/code]

In the next section, I set up the variables for my pipeline. Since this is a very simple example, I only need one for the image name. I define a name and set the tag to the build id using the built-in variable $(Build.BuildId). This increases the tag of my image automatically every time when a build runs.

[code language=&#8221;text&#8221;]  
variables:  
ImageName: &#8216;wolfgangofner/microservicedemo:$(Build.BuildId)&#8217;  
[/code]

If you want better versioning of the Docker images, use one of the many extensions from the marketplace. In my projects, we use one of our own plugins which you can find <a href="https://marketplace.visualstudio.com/items?itemName=4tecture.BuildVersioning" target="_blank" rel="noopener noreferrer">here</a>.

### Build the Docker Image

Now that everything is set up, let&#8217;s add a task to build the image. Before you can do that, you have to add a stage and a job. You can use whatever name you want for your stage and job. For now, you only need one. It is good practice to use a meaningful name though.

Inside the job, add a task for Docker. Inside this task add your previously created service connection, the location to the dockerfile, an image name, and the build context. As the command use Build an Image. Note that I use version 1 because version 2 was not working and resulted in an error I could not resolve.

[code language=&#8221;text&#8221;]  
stages:  
&#8211; stage: Build  
displayName: Build image  
jobs:  
&#8211; job: DockerImage  
displayName: Build and push Docker image  
steps:  
&#8211; task: Docker@1  
displayName: &#8216;Build the Docker image&#8217;  
inputs:  
containerregistrytype: &#8216;Container Registry&#8217;  
dockerRegistryEndpoint: &#8216;Docker Hub&#8217;  
command: &#8216;Build an image&#8217;  
dockerFile: &#8216;**/Dockerfile&#8217;  
imageName: &#8216;$(ImageName)&#8217;  
includeLatestTag: true  
useDefaultContext: false  
buildContext: &#8216;.&#8217;  
[/code]

You can either add the YAML code from above or click on the Docker task on the right side. You can also easily edit a task by clicking Settings right above the task. This will open the task on the right side.

<div id="attachment_2374" style="width: 710px" class="wp-caption aligncenter">
  <a href="https://www.programmingwithwolfgang.com/wp-content/uploads/2020/09/Edit-the-Docker-task.jpg"><img aria-describedby="caption-attachment-2374" loading="lazy" class="wp-image-2374" src="https://www.programmingwithwolfgang.com/wp-content/uploads/2020/09/Edit-the-Docker-task.jpg" alt="Edit the Docker task in the DevOps CI pipeline" width="700" height="532" srcset="https://www.programmingwithwolfgang.com/wp-content/uploads/2020/09/Edit-the-Docker-task.jpg 713w, https://www.programmingwithwolfgang.com/wp-content/uploads/2020/09/Edit-the-Docker-task-300x228.jpg 300w" sizes="(max-width: 700px) 100vw, 700px" /></a>
  
  <p id="caption-attachment-2374" class="wp-caption-text">
    Edit the Docker task
  </p>
</div>

Save the pipeline and run it. This should give you a green build.

### Push the Image to Docker Hub

The last step is to push the image to a registry. For this example, I use Docker Hub because it is publicly available but you can also use a private one like Azure Container Registry (ACR) or even a private Docker Hub repository.

Add the following code to your pipeline:

[code language=&#8221;text&#8221;]  
&#8211; task: Docker@1  
displayName: &#8216;Push the Docker image to Dockerhub&#8217;  
inputs:  
containerregistrytype: &#8216;Container Registry&#8217;  
dockerRegistryEndpoint: &#8216;Docker Hub&#8217;  
command: &#8216;Push an image&#8217;  
imageName: &#8216;$(ImageName)&#8217;  
condition: and(succeeded(), ne(variables[&#8216;Build.Reason&#8217;], &#8216;PullRequest&#8217;))  
[/code]

Here I set a display name, the container registry &#8220;Container Registry&#8221; which means Docker Hub, and select my previously created service connection &#8220;Docker Hub&#8221;. The command indicates that I want to push an image and I set the image name from the previously created variable. This task only runs when the previous task was successful and when the build is not triggered by a pull request.

### The finished Azure DevOps CI Pipeline

The finished pipeline looks as follows:

[code language=&#8221;text&#8221;]  
name : Docker-CI  
trigger:  
branches:  
include:  
&#8211; master

pool:  
vmImage: &#8216;ubuntu-latest&#8217;

variables:  
ImageName: &#8216;wolfgangofner/microservicedemo:$(Build.BuildId)&#8217;

stages:  
&#8211; stage: Build  
displayName: Build image  
jobs:  
&#8211; job: Build  
displayName: Build and push Docker image  
steps:  
&#8211; task: Docker@1  
displayName: &#8216;Build the Docker image&#8217;  
inputs:  
containerregistrytype: &#8216;Container Registry&#8217;  
dockerRegistryEndpoint: &#8216;Docker Hub&#8217;  
command: &#8216;Build an image&#8217;  
dockerFile: &#8216;**/Dockerfile&#8217;  
imageName: &#8216;$(ImageName)&#8217;  
includeLatestTag: true  
useDefaultContext: false  
buildContext: &#8216;.&#8217;

&#8211; task: Docker@1  
displayName: &#8216;Push the Docker image to Dockerhub&#8217;  
inputs:  
containerregistrytype: &#8216;Container Registry&#8217;  
dockerRegistryEndpoint: &#8216;Docker Hub&#8217;  
command: &#8216;Push an image&#8217;  
imageName: &#8216;$(ImageName)&#8217;  
condition: and(succeeded(), ne(variables[&#8216;Build.Reason&#8217;], &#8216;PullRequest&#8217;))  
[/code]

You can also find the code of the CI pipeline on <a href="https://github.com/WolfgangOfner/.NetCoreMicroserviceCiCdAks/blob/DockerCiPipeline/Docker-CI.yml" target="_blank" rel="noopener noreferrer">Github</a>.

## Testing the Azure DevOps CI Pipeline

Save the pipeline and run it. The build should succeed and a new image should be pushed to Docker Hub.

<div id="attachment_2376" style="width: 710px" class="wp-caption aligncenter">
  <a href="https://www.programmingwithwolfgang.com/wp-content/uploads/2020/09/The-pipeline-ran-successfully.jpg"><img aria-describedby="caption-attachment-2376" loading="lazy" class="wp-image-2376" src="https://www.programmingwithwolfgang.com/wp-content/uploads/2020/09/The-pipeline-ran-successfully.jpg" alt="The DevOps CI pipeline ran successfully" width="700" height="227" srcset="https://www.programmingwithwolfgang.com/wp-content/uploads/2020/09/The-pipeline-ran-successfully.jpg 1165w, https://www.programmingwithwolfgang.com/wp-content/uploads/2020/09/The-pipeline-ran-successfully-300x97.jpg 300w, https://www.programmingwithwolfgang.com/wp-content/uploads/2020/09/The-pipeline-ran-successfully-1024x332.jpg 1024w, https://www.programmingwithwolfgang.com/wp-content/uploads/2020/09/The-pipeline-ran-successfully-768x249.jpg 768w" sizes="(max-width: 700px) 100vw, 700px" /></a>
  
  <p id="caption-attachment-2376" class="wp-caption-text">
    The pipeline ran successfully
  </p>
</div>

The pipeline ran successfully and if I go to <a href="https://hub.docker.com/r/wolfgangofner/microservicedemo/tags" target="_blank" rel="noopener noreferrer">my repository on Docker Hub</a>, I should see a new image with the tag 232 there.

<div id="attachment_2377" style="width: 710px" class="wp-caption aligncenter">
  <a href="https://www.programmingwithwolfgang.com/wp-content/uploads/2020/09/The-new-image-got-pushed-to-Docker-Hub.jpg"><img aria-describedby="caption-attachment-2377" loading="lazy" class="wp-image-2377" src="https://www.programmingwithwolfgang.com/wp-content/uploads/2020/09/The-new-image-got-pushed-to-Docker-Hub.jpg" alt="The new image got pushed to Docker Hub" width="700" height="273" srcset="https://www.programmingwithwolfgang.com/wp-content/uploads/2020/09/The-new-image-got-pushed-to-Docker-Hub.jpg 1296w, https://www.programmingwithwolfgang.com/wp-content/uploads/2020/09/The-new-image-got-pushed-to-Docker-Hub-300x117.jpg 300w, https://www.programmingwithwolfgang.com/wp-content/uploads/2020/09/The-new-image-got-pushed-to-Docker-Hub-1024x400.jpg 1024w, https://www.programmingwithwolfgang.com/wp-content/uploads/2020/09/The-new-image-got-pushed-to-Docker-Hub-768x300.jpg 768w" sizes="(max-width: 700px) 100vw, 700px" /></a>
  
  <p id="caption-attachment-2377" class="wp-caption-text">
    The new image got pushed to Docker Hub
  </p>
</div>

## Conclusion

An automated CI pipeline to build and push new images is an integral point of every DevOps process. This post showed that it is quite simple to automate everything and create a new image every time changes are pushed to the master branch.

You can find the source code of this demo on <a href="https://github.com/WolfgangOfner/.NetCoreMicroserviceCiCdAks/tree/DockerCiPipeline" target="_blank" rel="noopener noreferrer">Github</a>.