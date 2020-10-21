---
title: Design and implement third-party PaaS
date: 2018-07-24T11:47:46+02:00
author: Wolfgang Ofner
categories: [Cloud]
tags: [70-532, Azure, Certification, Exam, learning]
---
Azure support many third-party offerings and services through the Azure Marketplace. These can be deployed through the Azure portal, using ARM, or other CLI tools.

## Implement Cloud Foundry as a third-party PaaS

Cloud Foundry is an open source PaaS for building, deploying and operating 12-factor applications developed in various languages and frameworks. It is a mature container-based application platform allowing you to easily deploy and manage production grade application on a platform that supports continuous delivery and horizontal scale, and support hybrid and multi-cloud scenarios.

There are two forms of Cloud Foundry available on Azure:

  1. Open source Cloud Foundry (OSS CF) is an entirely open source version of Cloud Foundry managed by the Cloud Foundry Foundation.
  2. Pivotal Cloud Foundry (PCF) is an enterprise distribution of Cloud Foundry from Pivotal Software Inc., which adds on a set of proprietary management tools and enterprise support.

### Deploy Cloud Foundry on Azure

to deploy a basic Pivotal Cloud Foundry on Azure, follow these steps:

  1. Before you can create a Cloud Foundry cluster, you must create an Azure Service Principle. Follow the instructions on <a href="https://github.com/cloudfoundry-incubator/bosh-azure-cpi-release/blob/master/docs/get-started/create-service-principal.md" target="_blank" rel="noopener">Github</a>.
  2. In the Azure portal click on +Create a resource, search for Pivotal Cloud Foundry on Microsoft Azure and click Create.

<div id="attachment_1418" style="width: 451px" class="wp-caption aligncenter">
  <a href="/wp-content/uploads/2018/07/Deploy-Pivotal-Cloud-Foundry-on-Azure.jpg"><img aria-describedby="caption-attachment-1418" loading="lazy" class="wp-image-1418" src="/wp-content/uploads/2018/07/Deploy-Pivotal-Cloud-Foundry-on-Azure.jpg" alt="Deploy Pivotal Cloud Foundry on Azure as a third-party PaaS" width="441" height="700" /></a>
  
  <p id="caption-attachment-1418" class="wp-caption-text">
    Deploy Pivotal Cloud Foundry on Azure
  </p>
</div>

<ol start="3">
  <li>
    Provide a storage account name prefix, paste your SSH public key, upload the azure credentials.json file, enter the Pivotal Network API token, choose a resource group, and location.
  </li>
  <li>
    Click OK.
  </li>
  <li>
    After the validation is done, click OK again.
  </li>
  <li>
    On the Buy blade, click Purchase.
  </li>
</ol>

## Implement OpenShift

The OpenShift Container Platform is a PaaS offering from Red Hat built on Kubernetes. It brings together Docker and Kubernetes and provides an API to manage these services. OpenShift simplifies the process of deploying, scaling, and operating multi-tenant applications onto containers.

There are two forms of OpenShift:

  1. The open source OpenShift Origin
  2. The enterprise-grade Red Hat OpenShift Container Platform

Both are built on the same open source technologies, with the Red Hat OpenShift Container Platform offering enterprise-grade security, compliance, and container management.

The prerequisites for installing OpenShift include:

  1. Generate an SSH key pair (Public / Private), ensuring that you do not include a passphrase with the private key.
  2. Create a Key Vault to store the SSH Private Key.
  3. Create an Azure Active Directory Service Principal.
  4. Install and configure the OpenShift CLI to manage the cluster.

Prerequisites for deploying a Red Hat OpenShift Container Platform include:

  1. OpenShift Container Platform subscription eligible for use in Azure. You need to specify the Pool Id that contains your entitlements for OpenShift.
  2. Red Hat Customer Portal login credentials. You may use either an Organization ID and Activation Key, or a username and password. It is more secure to use the Organization ID and Activation Key.

### Deploy a Red Hat OpenShift Container Platform

You can deploy from the Azure Marketplace templates, or using ARM templates. To deploy a Red Hat OpenShift Container Platform from the Marketplace, follow these steps:

  1. In the Azure portal click on +Create a resource, search for Red Hat OpenShift Container Platform and click Create.

<div id="attachment_1419" style="width: 451px" class="wp-caption aligncenter">
  <a href="/wp-content/uploads/2018/07/Deploy-a-Red-hat-OpenShift-Container-Platform.jpg"><img aria-describedby="caption-attachment-1419" loading="lazy" class="wp-image-1419" src="/wp-content/uploads/2018/07/Deploy-a-Red-hat-OpenShift-Container-Platform.jpg" alt="Deploy a Red hat OpenShift Container Platform" width="441" height="700" /></a>
  
  <p id="caption-attachment-1419" class="wp-caption-text">
    Deploy a Red hat OpenShift Container Platform
  </p>
</div>

<ol start="2">
  <li>
    On the Basics blade, provide the VM Admin username, paste your SSH public key, select a resource group and location.
  </li>
  <li>
    Click OK.
  </li>
  <li>
    On the Infrastructure blade, provide an OCP cluster name prefix, select a cluster size, provide the resource group name for your key Vault, as well as the Key Vault name and its secret name you specified in the prerequisites.
  </li>
  <li>
    Click OK.
  </li>
  <li>
    On the OpenShift Container Platform Settings blade, provide an OpenShift Admin user password, enter your Red Hat subscription manager credentials, specify whether you want to configure an Azure Cloud Provider, and select your default router subdomain.
  </li>
  <li>
    Click OK.
  </li>
  <li>
     Wait until the validation passes and click OK.
  </li>
  <li>
    Click Purchase.
  </li>
</ol>

## Provision applications by using Azure Quickstart Templates

Azure Quickstart Templates are community contributed Azure Resource Manager templates that help you quickly provision applications and solutions with minimal effort. You can search for templates in the gallery at <a href="https://azure.microsoft.com/resources/templates" target="_blank" rel="noopener">https://azure.microsoft.com/resources/templates</a>

## Build applications that leverage Azure Marketplace solutions and services

The Azure Marketplace is an online applications and services marketplace that enables start-ups and independent software vendors to offer their solutions to Azure customers around the world.

Pricing varies based on the product type. Pricing models include:

  * Free
  * Free Software Trial
  * BYOL &#8211; Bring your own license
  * Monthly Fee
  * Usage Based

## Conclusion

This post gave a short overview of how to use third-party platforms as a Service in Azure and how to leverage the existing templates.

For more information about the 70-532 exam get the <a href="http://amzn.to/2EWNWMF" target="_blank" rel="noopener">Exam Ref book from Microsoft</a> and continue reading my blog posts. I am covering all topics needed to pass the exam. You can find an overview of all posts related to the 70-532 exam <a href="https://www.programmingwithwolfgang.com/prepared-for-the-70-532-exam/" target="_blank" rel="noopener">here</a>.