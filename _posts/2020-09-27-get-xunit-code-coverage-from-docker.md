---
title: Get xUnit Code Coverage from Docker
date: 2020-09-27T13:00:34+02:00
author: Wolfgang Ofner
categories: [DevOps, Docker]
tags: [Azure Devops, 'C#', docker, xUnit]
---
Getting code coverage in Azure DevOps is not well documented and the first time I configured it, it took me quite some time to figure out how to do it. It gets even more complicated when you run your tests inside a Docker container during the build.

<a href="/run-tests-inside-docker-during-ci/" target="_blank" rel="noopener noreferrer">In my last post</a>, I showed how to run tests inside a container during the build. Today, I want to show how to get the code coverage of these tests and how to display them in Azure DevOps.

Code coverage gives you an indication of how much of your code is covered by at least one test. Usually, the higher the better but you shouldn&#8217;t aim for 100%. As always, it depends on the project but I would recommend having around 80%.

## Setting up xUnit for Code Coverage

You can find the code of the demo on <a href="https://github.com/WolfgangOfner/.NetCoreMicroserviceCiCdAks/tree/CodeCoverage" target="_blank" rel="noopener noreferrer">Github</a>.

### Install Coverlet

I use coverlet to collect the coverage. All you have to do is installing the Nuget package. The full Nuget configuration of the test projects looks as following:

[code language=&#8221;XML&#8221;]  
<ItemGroup>  
<PackageReference Include="FakeItEasy" Version="6.2.1" />  
<PackageReference Include="FluentAssertions" Version="5.10.3" />  
<PackageReference Include="Microsoft.NET.Test.Sdk" Version="16.6.1" />  
<PackageReference Include="xunit" Version="2.4.1" />  
<PackageReference Include="xunit.runner.visualstudio" Version="2.4.2">  
<PrivateAssets>all</PrivateAssets>  
<IncludeAssets>runtime; build; native; contentfiles; analyzers; buildtransitive</IncludeAssets>  
</PackageReference>  
<PackageReference Include="coverlet.msbuild" Version="2.9.0">  
<PrivateAssets>all</PrivateAssets>  
<IncludeAssets>runtime; build; native; contentfiles; analyzers; buildtransitive</IncludeAssets>  
</PackageReference>  
</ItemGroup>  
[/code]

I am using FakeItEasy to mock objects, FluentAssertions for a more readable assertion and xUnit to run the tests.

### Collect the Code Coverage Results

After installing coverlet, the next step is to collect the coverage results. To do that, I edit the Dockerfile to enable collecting the coverage results, setting the output format, and the output directory. The code of the tests looks as follows:

[code language=&#8221;text&#8221;]  
FROM build AS test  
LABEL test=true  
RUN dotnet test -c Release &#8211;results-directory /testresults &#8211;logger "trx;LogFileName=test_results.trx" /p:CollectCoverage=true /p:CoverletOutputFormat=json%2cCobertura /p:CoverletOutput=/testresults/coverage/ -p:MergeWith=/testresults/coverage/coverage.json Tests/CustomerApi.Test/CustomerApi.Test.csproj  
RUN dotnet test -c Release &#8211;results-directory /testresults &#8211;logger "trx;LogFileName=test_results2.trx" /p:CollectCoverage=true /p:CoverletOutputFormat=json%2cCobertura /p:CoverletOutput=/testresults/coverage/ -p:MergeWith=/testresults/coverage/coverage.json Tests/CustomerApi.Service.Test/CustomerApi.Service.Test.csproj  
RUN dotnet test -c Release &#8211;results-directory /testresults &#8211;logger "trx;LogFileName=test_results3.trx" /p:CollectCoverage=true /p:CoverletOutputFormat=json%2cCobertura /p:CoverletOutput=/testresults/coverage/ -p:MergeWith=/testresults/coverage/coverage.json Tests/CustomerApi.Data.Test/CustomerApi.Data.Test.csproj  
[/code]

The output format is json and Cobertura because I want to collect the code coverage of all tests and merge them into the summary file. This is all done behind the scenes, all you have to do is using the MergeWith flag where you provide the path to the json file. You could also build the whole solution if you don&#8217;t want to configure this. The disadvantage is that you will always run all tests. This might be not wanted, especially in bigger projects where you want to separate unit tests from integration or UI tests.

This is everything you have to change in your projects to be ready to collect the coverage. The last step is to copy the results out of the container and display them in Azure DevOps.

## Display the Code Coverage Results in Azure DevOps

In my last post, I explained how to copy the test results out of the container using the label test=true. This means that besides the test results, the coverage results are also copied out of the container already. All I have to do now is to display these coverage results using the PublishCodeCoverageResults tasks from Azure DevOps. The code looks as follows:

[code language=&#8221;text&#8221;]  
&#8211; task: PublishCodeCoverageResults@1  
inputs:  
codeCoverageTool: &#8216;Cobertura&#8217;  
summaryFileLocation: &#8216;$(System.DefaultWorkingDirectory)/testresults/coverage/coverage.cobertura.xml&#8217;  
reportDirectory: &#8216;$(System.DefaultWorkingDirectory)/testresults/coverage/reports&#8217;  
displayName: &#8216;Publish code coverage results&#8217;  
[/code]

The whole code to copy the everything out of the container, display the test results and the code coverage looks as this:

[code language=&#8221;text&#8221;]  
&#8211; pwsh: |  
$id=docker images &#8211;filter "label=test=true" -q | Select-Object -First 1  
docker create &#8211;name testcontainer $id  
docker cp testcontainer:/testresults ./testresults  
docker rm testcontainer  
displayName: &#8216;Copy test results&#8217;

&#8211; task: PublishTestResults@2  
inputs:  
testResultsFormat: &#8216;VSTest&#8217;  
testResultsFiles: &#8216;*\*/\*.trx&#8217;  
searchFolder: &#8216;$(System.DefaultWorkingDirectory)/testresults&#8217;  
displayName: &#8216;Publish test results&#8217;

&#8211; task: PublishCodeCoverageResults@1  
inputs:  
codeCoverageTool: &#8216;Cobertura&#8217;  
summaryFileLocation: &#8216;$(System.DefaultWorkingDirectory)/testresults/coverage/coverage.cobertura.xml&#8217;  
reportDirectory: &#8216;$(System.DefaultWorkingDirectory)/testresults/coverage/reports&#8217;  
displayName: &#8216;Publish code coverage results&#8217;  
[/code]

Save the changes and run the CI pipeline. After the build is finished, you will see the Code Coverage tab in the summary overview where you can see the coverage of each of your projects.

<div id="attachment_2396" style="width: 444px" class="wp-caption aligncenter">
  <a href="/wp-content/uploads/2020/09/Summary-of-the-Code-Coverage-Results.jpg"><img aria-describedby="caption-attachment-2396" loading="lazy" class="size-full wp-image-2396" src="/wp-content/uploads/2020/09/Summary-of-the-Code-Coverage-Results.jpg" alt="Summary of the Code Coverage Results" width="434" height="620" /></a>
  
  <p id="caption-attachment-2396" class="wp-caption-text">
    Summary of the Code Coverage Results
  </p>
</div>

## Conclusion

The code coverage shows how much of your code is covered by at least one test. This post showed how easy it can be to display these results in Azure DevOps, even when the build runs inside a Docker container.

You can find the code of the demo on <a href="https://github.com/WolfgangOfner/.NetCoreMicroserviceCiCdAks/tree/CodeCoverage" target="_blank" rel="noopener noreferrer">Github</a>.