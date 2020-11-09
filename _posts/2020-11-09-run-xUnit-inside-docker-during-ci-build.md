---
title: Run xUnit Tests inside Docker during an Azure DevOps CI Build
date: 2020-11-09
author: Wolfgang Ofner
categories: [DevOps, Docker]
tags: [Azure DevOps, CI, docker, Unit Test, xUnit]
---
Running your build inside a Docker container has many advantages like platform independence and better testability for developers. These containers also bring more complexity though. <a href="/build-docker-azure-devops-ci-pipeline/" target="_blank" rel="noopener noreferrer">In my last post</a>, I showed how to set up the build inside a Docker container. Unfortunately, I didn't get any test results or code coverage after the build succeeded. Today, I want to show how to get the test results when you run tests inside Docker and how to display them in Azure DevOps.

You can find the code of this demo on [Github](https://github.com/WolfgangOfner/.NetCoreMicroserviceCiCdAks/tree/UnitTestInCiPipeline).

## Run Tests inside Docker

Running unit tests inside a Docker container is more or less the same as building a project. First, I copy all my test projects inside the container using the COPY command:

```text  
COPY ["Tests/CustomerApi.Test/CustomerApi.Test.csproj", "Tests/CustomerApi.Test/"]  
COPY ["Tests/CustomerApi.Service.Test/CustomerApi.Service.Test.csproj", "Tests/CustomerApi.Service.Test/"]  
COPY ["Tests/CustomerApi.Data.Test/CustomerApi.Data.Test.csproj", "Tests/CustomerApi.Data.Test/"]  
```

After copying, I execute dotnet restore on all test projects.

```text  
RUN dotnet restore "Tests/CustomerApi.Test/CustomerApi.Test.csproj"
RUN dotnet restore "Tests/CustomerApi.Service.Test/CustomerApi.Service.Test.csproj"
RUN dotnet restore "Tests/CustomerApi.Data.Test/CustomerApi.Data.Test.csproj" 
```

Next, I set the label test to the build id. I will need this label later to identify the right layer of the container to copy the test results out of it. Then, I use dotnet test to run the tests in my three test projects. Additionally, I write the test result into the testresults folder and give them different names, e.g. test_results.trx.

```text  
FROM build AS test  
ARG BuildId
LABEL test=${BuildId} 
RUN dotnet test --no-build -c Release --results-directory /testresults --logger "trx;LogFileName=test_results.trx"  Tests/CustomerApi.Test/CustomerApi.Test.csproj  
RUN dotnet test --no-build -c Release --results-directory /testresults --logger "trx;LogFileName=test_results2.trx" Tests/CustomerApi.Service.Test/CustomerApi.Service.Test.csproj  
RUN dotnet test --no-build -c Release --results-directory /testresults --logger "trx;LogFileName=test_results3.trx" Tests/CustomerApi.Data.Test/CustomerApi.Data.Test.csproj 
```

That's already everything I have to change to run the tests inside the container and generate test results. You don't have to split up every command as I did but I would recommend doing so. This helps you finding problems and also is better for the caching which will increase the build time of your container. You could also use a hard-coded value for the label but if you have multiple builds running at the same time, the pipeline may publish the wrong test results since all builds have the same label name.

If you run the build, you will see the successful tests in the output of the build step.

<a style="text-align: center;" href="/assets/img/posts/The-tests-ran-inside-the-Docker-Container.JPG"><img loading="lazy" src="/assets/img/posts/The-tests-ran-inside-the-Docker-Container.JPG" alt="The tests ran inside the Docker container" /></a>

The tests ran inside the Docker container. If you try to look at the Tests tab of the built-in Azure DevOps to see the test results, you won't see the tab.

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2020/09/The-build-was-successful-but-not-Test-Results-are-showing.jpg"><img loading="lazy" src="/assets/img/posts/2020/09/The-build-was-successful-but-not-Test-Results-are-showing.jpg" alt="The build was successful but not Test Results are showing" /></a>
  
  <p>
    The build was successful but not Test Results are showing
  </p>
</div>

The Tests tab is not displayed because Azure DevOps has no test results to display. Since I ran the tests inside the container, the results are also inside the container. To display them, I have to copy them out of the docker container and publish them.

## Copy Test Results after you run Tests inside Docker

To copy the test results out of the container, first I have to pass the build id to the dockerfile. To do that, add the following line to the Docker build task:

```yaml  
arguments: '--build-arg BuildId=$(Build.BuildId)'
```

Next, I use the following PowerShell task in the CI pipeline to create an intermediate container which contains the test results.

```yaml  
- pwsh: |
   $id=docker images --filter "label=test=$(Build.BuildId)" -q | Select-Object -First 1
   docker create --name testcontainer $id
   docker cp testcontainer:/testresults ./testresults
   docker rm testcontainer
  displayName: 'Copy test results' 
```

Docker creates a new layer for every command in the Dockerfile. I can access the layer (also called intermediate container) through the label I set during the build. The script selects the first intermediate container with the label test=true and then copies the content of the testresults folder to the testresults folder of the WorkingDirectory of the build agent. Then the container is removed. Next, I can take this testresults folder and publish the test results inside it.

## Publish the Test Results

To publish the test results, I use the PublishTestResult task of Azure DevOps. I only have to provide the format of the results, what files contain results and, the path to the files. The YAML code looks as follows:

```yaml  
- task: PublishTestResults@2
  inputs:
    testResultsFormat: 'VSTest'
    testResultsFiles: '**/*.trx'
    searchFolder: '$(System.DefaultWorkingDirectory)/testresults'
  displayName: 'Publish test results' 
```

Run the CI pipeline again and after it is finished, you will see the Tests tab on the summary page. Click on it and you will see that all tests ran successfully. Azure DevOps even gives you a trophy for that.

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/The-Tests-tab-is-shown-and-you-get-even-a-trophy.JPG"><img loading="lazy" src="/assets/img/posts/The-Tests-tab-is-shown-and-you-get-even-a-trophy.JPG" alt="The Tests tab is shown and you get even a trophy" /></a>
  
  <p>
    The Tests tab is shown and you get even a trophy
  </p>
</div>

I did the same for the OrderApi project, except that I replaced CustomerApi with OrderApi in the Dockerfile.

## Conclusion

Docker containers are awesome. They can be used to run your application anywhere but also to build your application. This enables you to take your build definition and run it in Azure DevOps, as Github actions, or in Jenkins. You don't have to change anything because the logic is encapsulated inside the Dockerfile. This flexibility comes with some challenges, for example, to display the test results of the unit tests. This post showed that it is pretty simple to get these results out of the container and display in Azure DevOps.

In my next post, I will show how you can also display the code coverage of your tests. You can find the code of this demo on [Github](https://github.com/WolfgangOfner/MicroserviceDemo).