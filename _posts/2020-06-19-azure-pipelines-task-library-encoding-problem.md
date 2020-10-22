---
title: Azure Pipelines Task Library Encoding Problem
date: 2020-06-19T09:35:48+02:00
author: Wolfgang Ofner
categories: [DevOps]
tags: [Azure Devops, Azure Key Vault, Azure Pipelines Task Library]
---
For one of our customers, we use an Azure DevOps pipeline to read secrets from the Azure Key Vault and then write them into pipeline variables. During the operation, we encountered an encoding problem with the azure pipeline task library. The library encoded the percentage symbol during the read. Instead of Secret%Value, we got Secret%25Value. After writing the value into our pipeline variable, the value Secret%25Value was saved.

## The CD Pipeline

Our Azure DevOps CD pipeline reads all the secrets from an Azure Key Vault using the Azure Key Vault Task. Then we pass these secrets into a Powershell script and copy the values into new variables. You can see the tasks in question in the following code sample.

[code language=&#8221;text&#8221;]  
&#8211; task: AzureKeyVault@1  
inputs:  
azureSubscription: "$(KeyVaultSubscription)"  
KeyVaultName: $(KeyVaultName)  
SecretsFilter: "*"  
displayName: "Read secret values from key vault"

&#8211; task: RenameVariables@0  
inputs:  
variablesRenamingDefinition: |  
base-infra-sas1:sas1  
base-infra-sas2:sas2  
base-infra-sas3:sas3  
displayName: "Prepare stage secret variables from key vault"  
[/code]

We do this because our customer has several configurations for each environment. For example, there are secrets for test1WebServer, test1ImageServer, test1DbServer, test2Webserver and, so on in the key vault. We take these variables and write them into generic variables like WebServer, DbServer and, so on. This enables us to have one pipeline and the customer can use as many configurations as they want.

## Azure Pipelines Task Library Encoding Problem

The problem we encountered is that the Azure Key Vault Task read the secrets but encoded the % sign. Instead of Secret%Value, we got Secret%25Value and therefore wrote Secret%25Value into our variable inside the Powershell. As a result, the deployment failed since our secret is not correct. After too many hours of searching, we found an issue on Github with the same Azure pipelines task library encoding problem.  You can find a Github issue <a href="https://github.com/microsoft/azure-pipelines-task-lib/issues/627" target="_blank" rel="noopener noreferrer">here</a>. The problem occurs in version 2.9.3. The Azure Key Vault task uses version 2.8.0. You can find the code on [Github](https://github.com/microsoft/azure-pipelines-tasks/blob/master/Tasks/AzureKeyVaultV1/package.json).

## Fixing the Azure Pipelines Task Library Encoding Problem

In our Powershell script, we were using &#8220;azure-pipelines-task-lib&#8221;: &#8220;^2.8.0&#8221;. The ^ updates to all future versions, which is as of this writing 2.9.3. We use this to stay up to date without changing the version of all packages by hand. Unfortunately, the azure-pipelines-task-lib 2.8 is not compatible with 2.9.3 and therefore led to the encoding problem. Fixing the problem was as easy as removing the ^ and redeploying the task to the Azure marketplace.

## Consequences

Right now the pipeline is working again but once Microsoft updates the Azure Key Vault task to use a new version of the azure-pipelines-task-lib, the pipeline will break again. When this happens we have to update the azure-pipelines-task-lib inside our Powershell script and it should work again.

If you want to read more about CI/CD pipelines, read my article <a href="/create-automatic-build-pipeline-for-net-core/" target="_blank" rel="noopener noreferrer">Create Automatic Builds for .Net and .Net Core Applications with Azure Devops</a>.