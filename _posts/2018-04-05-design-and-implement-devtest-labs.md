---
title: Design and implement DevTest Labs
date: 2018-04-05T16:45:00+02:00
author: Wolfgang Ofner
categories: [Cloud]
tags: [70-532, Azure, Certification, Exam, learning]
---
Azure DevTest Labs is a service designed to help developers and testers quickly spin up virtual machines (VMs) or complete environments in Azure, enabling rapid deployment and testing of applications. This allows you to easily spin up and tear down development and test resources, minimizing waste and providing better cost control. It is also useful to create pre-provisioned environments for demos and training.

## Create DevTest Labs

To add a new DevTest Lab, follow these steps:

  1. In the Azure Portal select All services and search for DevTest Labs.
  2. Click on DevTest Labs and select + Add.
  3. Enter a Lab name, Subscription, Location and optionally add a Name and Value for tagging.
  4. By default, the Auto-shutdown option is enabled. This feature can help you save costs. Click on it, if you want to disable it or if you want to change the time for the shutdown.

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2018/03/Create-a-DevTest-Lab.jpg"><img loading="lazy" src="/assets/img/posts/2018/03/Create-a-DevTest-Lab.jpg" alt="Create a DevTest Lab" /></a>
  
  <p>
    Create a DevTest Lab
  </p>
</div>

<ol start="5">
  <li>
    Click on Create.
  </li>
</ol>

The deployment process creates a new resource group with the following resources in it:

  * The DevTest Lab instance
  * A Key vault instance
  * A Storage account
  * A virtual network

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2018/03/Deployed-resources.jpg"><img loading="lazy" src="/assets/img/posts/2018/03/Deployed-resources.jpg" alt="Deployed resources" /></a>
  
  <p>
    Deployed resources
  </p>
</div>

## Add a VM to your lab

After the DevTest Lab is deployed, you can add a VM to it following these steps:

  1. In your DevTest Lab, click +Add on the overview blade.
  2. On the Choose a base blade, select your desired image for your VM.
  3. Selecting an image opens the Virtual machine blade. Provide a machine name, user name, password, the size of your VM and optionally artifacts. Artifacts are third-party tools like Chrome or Git.

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2018/03/Create-a-Windows-2016-Server-for-your-DevTest-Lab.jpg"><img loading="lazy" src="/assets/img/posts/2018/03/Create-a-Windows-2016-Server-for-your-DevTest-Lab.jpg" alt="Create a Windows 2016 Server for your DevTest Lab" /></a>
  
  <p>
    Create a Windows 2016 Server for your DevTest Lab
  </p>
</div>

<ol start="4">
  <li>
    Click Create to start the deployment process.
  </li>
</ol>

### Claim a VM

After the VM is created, the ownership is assigned to the creator but it can be made claimable for others. To unclaim the VM,  follow these steps:

  1. Click on the VM, you want to unclaim in your DevTest Lab.
  2. On the top of the Overview blade click on unclaim.

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2018/03/Unclaim-a-VM.jpg"><img loading="lazy" src="/assets/img/posts/2018/03/Unclaim-a-VM.jpg" alt="Unclaim a VM" /></a>
  
  <p>
    Unclaim a VM
  </p>
</div>

Unclaiming a VM removes it from the My virtual machines area in the DevTest Lab and moves it to the Claimable virtual machines section.

To claim a VM, open it and select Claim machine on the top of the Overview blade.

## Create and manage custom images and formulas

The difference between a custom image and a formula is that the custom image is an image based on a VHD, whereas formulas are base on a VHD with additional pre-configurations such VM size, virtual network, and artifacts. These pre-configured settings can be overridden during the deployment process.

### Creating custom images

Custom images provide a static, immutable way to create VMs from a configuration.

Pros of using custom images:

  * VMs created from the same image are identical
  * The provisioning of the VM is fast

Cons of using custom images:

  * To update the image, it has to be recreated

You can use the Azure Portal or PowerShell to create an image from an existing VM or create one from a VHD.

### Create a custom image from a provisioned VM

To create a custom image from a provisioned VM using the Azure Portal, follow these steps:

  1. In your DevTest lab, select All virtual machines under the My Lab menu and then click on the VM, you want to use as a base for your image.

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2018/03/All-virtual-machines-in-your-DevTest-Lab.jpg"><img loading="lazy" src="/assets/img/posts/2018/03/All-virtual-machines-in-your-DevTest-Lab.jpg" alt="All virtual machines in your DevTest Lab" /></a>
  
  <p>
    All virtual machines in your DevTest Lab
  </p>
</div>

<ol start="2">
  <li>
    After you selected a VM, click on Create custom image under the Operations menu.
  </li>
  <li>
    On the Custom image blade, provide a name and select whether sysprep should be run.
  </li>
  <li>
    Click OK to create the image.
  </li>
</ol>

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2018/03/Create-a-custom-image.jpg"><img loading="lazy" src="/assets/img/posts/2018/03/Create-a-custom-image.jpg" alt="Create a custom image" /></a>
  
  <p>
    Create a custom image
  </p>
</div>

### Create a custom image from a VHD using the Azure Portal

To create a custom image from a VHD using the Azure Portal follow these steps:

  1. In your DevTest Lab instance, select Configuration and policies under the Settings tab.
  2. On the Configuration and policies blade, select Custom images under the Virtual Machine Bases menu and click +Add.
  3. Provide a name, the operating system type and select the VHD. If you don&#8217;t have any, you have to upload one first.
  4. Click OK.

###  Delete a custom image

To delete a custom image, follow these steps:

  1. In your DevTest Lab instance, select Configuration and policies under the Settings tab.
  2. On the Configuration and policies blade, select Custom images under the Virtual Machine Bases menu and click the three dots next to the image which you want to delete.
  3. Click delete and select Yes in the confirmation dialog.

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2018/03/Delete-a-custom-image.jpg"><img loading="lazy" src="/assets/img/posts/2018/03/Delete-a-custom-image.jpg" alt="Delete a custom image" /></a>
  
  <p>
    Delete a custom image
  </p>
</div>

## Formulas

Formulas provide default property values and therefore offer a fast way to create VMs from a preconfigured state.

Pros of using formulas:

  * Formulas can define default setting that custom images can&#8217;t provide.
  * The default settings from the formulas can be modified when creating a new VM.

Cons of using custom images:

  * Creating a new formula can be more time consuming than creating a VM from a custom image.

There are two ways to create a formula:

  1. From a base like a custom image or Marketplace image, use a base when you want to define all characteristics of the formula.
  2. From an existing lab VM. Use this approach when you want to create a formula which is based on an existing VM.

### Create a formula from a base

To create a formula from a base, follow these steps:

  1. In your DevTest Labs instance, select Configuration and policies under the Settings menu.
  2. On the Configuration and policies blade, select Formulas (reusable bases) under the Virtual Machine Bases menu and click + Add.
  3. On the Choose a base blade, select an image to use for the formula, for example, Windows Server 2016 Datacenter.
  4. Provide a Formula name and user name.
  5. Select the disk type and the size of your VM.
  6. Optionally add artifacts. Artifacts are third-party tools like Google Chrome or Docker which will be installed on your VM.
  7. On the Advanced blade, you can configure the IP address and the automatic delete options.
  8. After you are done with the configuration, click Create to create the formula.

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2018/04/Create-a-new-formula.jpg"><img loading="lazy" src="/assets/img/posts/2018/04/Create-a-new-formula.jpg" alt="Create a new formula" /></a>
  
  <p>
    Create a new formula
  </p>
</div>

### Create a formula from a VM

To create a formula from a VM, follow these steps:

  1. In your DevTest Labs instance, on the Overview blade, select the VM from which you want to create the new formula.

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2018/04/Select-a-VM-for-your-formula.jpg"><img loading="lazy" src="/assets/img/posts/2018/04/Select-a-VM-for-your-formula.jpg" alt="Select a VM for your formula" /></a>
  
  <p>
    Select a VM for your formula
  </p>
</div>

<ol start="2">
  <li>
    On the VM&#8217;s blade, select Create formula (reusable base) under the Operations menu.
  </li>
  <li>
    On the Create a formula blade, provide a name and optionally a description and click OK.
  </li>
</ol>

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2018/04/Create-a-new-formula-from-a-VM.jpg"><img loading="lazy" src="/assets/img/posts/2018/04/Create-a-new-formula-from-a-VM.jpg" alt="Create a new formula from a VM" /></a>
  
  <p>
    Create a new formula from a VM
  </p>
</div>

### Modify a formula

To modify the properties of an existing formula, follow these steps:

  1. In your DevTest Labs instance, select Configuration and policies under the Settings menu.
  2. On the Configuration and policies blade, select Formulas (reusable bases) under the Virtual Machine Bases menu.
  3. Select the formula you wish to modify.
  4. Make your changes on the Update formula blade and select Update when you are finished.

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2018/04/Modify-an-existing-formula.jpg"><img loading="lazy" src="/assets/img/posts/2018/04/Modify-an-existing-formula.jpg" alt="Modify an existing formula" /></a>
  
  <p>
    Modify an existing formula
  </p>
</div>

### Delete a formula

To delete an existing formula, follow these steps:

  1. In your DevTest Labs instance, select Configuration and policies under the Settings menu.
  2. On the Configuration and policies blade, select Formulas (reusable bases) under the Virtual Machine Bases menu.
  3. Click on the ellipsis to the right of the formula you want to delete and click Delete in the context menu.

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2018/04/Delete-an-existing-formula.jpg"><img loading="lazy" src="/assets/img/posts/2018/04/Delete-an-existing-formula.jpg" alt="Delete an existing formula" /></a>
  
  <p>
    Delete an existing formula
  </p>
</div>

<ol start="4">
  <li>
    On the confirmation dialog click Yes.
  </li>
</ol>

## Configure a lab to include policies and procedures

For each lab you create, you can control cost and minimize your time waste by managing policies.

### Configure allowed virtual machine sizes policy

You can configure that the creation of only specific VM sizes is allowed. To do that follow these steps:

  1. In your DevTest Labs instance, select Configuration and policies under the Settings menu.
  2. On the Configuration and policies blade, select Allowed virtual machines under the Settings menu.
  3. On the Allowed virtual machine move the slider to On and then check each VM size which you want to be allowed to be created.
  4. Click Save on the top of the blade when you are finished.

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2018/04/Enable-allowed-virtual-machine-sizes-policy.jpg"><img loading="lazy" src="/assets/img/posts/2018/04/Enable-allowed-virtual-machine-sizes-policy.jpg" alt="Enable allowed virtual machine sizes policy" /></a>
  
  <p>
    Enable allowed virtual machine sizes policy
  </p>
</div>

### Configure virtual machines per user policy

It is also possible to limit the number of virtual machines an user can create. Additionally, you can limit the number of VMs using premium OS disks. To do that, follow these steps:

  1. In your DevTest Labs instance, select Configuration and policies under the Settings menu.
  2. On the Configuration and policies blade, select Virtual machines per user under the Settings menu.
  3. Move the slider to Yes to limit the number of virtual machines and/or the number of virtual machines using premium OS disks per user.
  4. After you enabled a limitation, provide a number for the limit. The default value is 1.
  5. Click Save.

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2018/04/Limit-the-number-of-virtual-machines-and-premium-OS-disks-per-user.jpg"><img loading="lazy" src="/assets/img/posts/2018/04/Limit-the-number-of-virtual-machines-and-premium-OS-disks-per-user.jpg" alt="Limit the number of virtual machines and premium OS disks per user" /></a>
  
  <p>
    Limit the number of virtual machines and premium OS disks per user
  </p>
</div>

If a user has reached his maximum amount of VMs but tries to create another one, he will get an error message.

### Configure virtual machines per lab policy

To specify the maximum number of VMs that can be created in your current lab, follow these steps:

  1. In your DevTest Labs instance, select Configuration and policies under the Settings menu.
  2. On the Configuration and policies blade, select Virtual machines per lab under the Settings menu.
  3. Move the slider to Yes to limit the number of virtual machines and/or the number of virtual machines using premium OS disks per lab.
  4. After you enabled a limitation, provide a number for the limit. The default value is 1.
  5. Click Save.

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2018/04/Limit-the-number-of-virtual-machines-and-premium-OS-disks-per-lab.jpg"><img loading="lazy" src="/assets/img/posts/2018/04/Limit-the-number-of-virtual-machines-and-premium-OS-disks-per-lab.jpg" alt="Limit the number of virtual machines and premium OS disks per lab" /></a>
  
  <p>
    Limit the number of virtual machines and premium OS disks per lab
  </p>
</div>

### Configure auto-shutdown policy

The auto-shutdown is the most important policy for helping you to minimize cost control and also helps to prevent costs when the VMs are not in use. I always enable this policy when I do training because there is always at least one student who doesn&#8217;t turn off his VMs at the end of the day.

To enable auto-shutdown, follow these steps:

  1. In your DevTest Labs instance, select Configuration and policies under the Settings menu.
  2. On the Configuration and policies blade, select Auto-shutdown under the Schedules menu.
  3. Move the slider to On to enable auto-shutdown.
  4. Configure the shutdown time and optionally enable sending a notification and provide a webhook URL or an email address. The notification will be sent 15 minutes prior the shutdown.
  5. Click Save.

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2018/04/Configure-auto-shutdown-policy.jpg"><img loading="lazy" src="/assets/img/posts/2018/04/Configure-auto-shutdown-policy.jpg" alt="Configure auto-shutdown policy" /></a>
  
  <p>
    Configure auto-shutdown policy
  </p>
</div>

Once configured, the auto-shutdown policy will be applied to all VMs in the lab. You also can adjust the auto-shutdown policy for specific VMs. To do that follow these steps:

  1. In your DevTest Labs instance, select the VM you want to configure.
  2. On the VM blade select Auto-shutdown under the Operations menu. There you can see the same attributes as before but if you change them here, you only change the settings for this VM.
  3. After all changes are made, click Save.

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2018/04/Configure-auto-shutdown-policy-for-an-individual-VM.jpg"><img loading="lazy" src="/assets/img/posts/2018/04/Configure-auto-shutdown-policy-for-an-individual-VM.jpg" alt="Configure auto-shutdown policy for an individual VM" /></a>
  
  <p>
    Configure auto-shutdown policy for an individual VM
  </p>
</div>

I like to disable the auto-shutdown policy on my VM, so I can prepare things for the next days training on the evening without getting interrupted by the shutdown.

## Configure auto-start policy

Besides auto-shutdown, you can also configure an auto-start policy. To do that follow these steps:

  1. In your DevTest Labs instance, select Configuration and policies under the Settings menu.
  2. On the Configuration and policies blade, select Auto-start under the Schedules menu.
  3. Move the slider to On and then configure the start time and on which days the VMs should start.
  4. Click Save

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2018/04/Configure-auto-start-policy.jpg"><img loading="lazy" src="/assets/img/posts/2018/04/Configure-auto-start-policy.jpg" alt="Configure auto-start policy" /></a>
  
  <p>
    Configure auto-start policy
  </p>
</div>

## 

Once configured, the auto-start policy will be applied to all VMs in the lab. You can Opt-out the auto-start policy for a specific VM. To do that follow these steps:

  1. In your DevTest Labs instance, select the VM you want to configure.
  2. On the VM blade select Auto-start under the Operations menu. There you can see move the slider to On or Off to enable or disable auto-start.
  3. Click Save.

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2018/04/Disable-auto-start-for-a-specific-VM.jpg"><img loading="lazy" src="/assets/img/posts/2018/04/Disable-auto-start-for-a-specific-VM.jpg" alt="Disable auto-start for a specific VM" /></a>
  
  <p>
    Disable auto-start for a specific VM
  </p>
</div>

### Set expiration date policy

The expiration date policy ensures that VMs are automatically deleted at a specified date and time. To do that follow these steps:

  1. When creating a new VM click on the Advanced settings.
  2. On the Advanced blade click the calendar icon under Expiration date and select the date when the VM should be deleted. Next, select the time.
  3. Click OK and then Create.

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2018/04/Set-the-expiration-date-policy.jpg"><img loading="lazy" src="/assets/img/posts/2018/04/Set-the-expiration-date-policy.jpg" alt="Set the expiration date policy" /></a>
  
  <p>
    Set the expiration date policy
  </p>
</div>

## Configure cost management

Azure DevTest Labs was designed to manage your resources and costs effectively. Cost Management is a key feature and allows you to track to costs associated to your lab. It also enables you to view trends, set cost targets and configure alerts.

To view your Cost trend chart, select Configuration and policies in your DevTest lab and then select Cost trend under the Cost Tracking menu.

### Cost trend

To view the monthly estimated cost trend chart follow these steps:

  1. In your DevTest Labs instance, select Configuration and policies under the Settings menu.
  2. On the Configuration and policies blade, select Cost trend under the Cost Tracking menu.

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2018/04/The-Cost-trend-chart.jpg"><img loading="lazy" src="/assets/img/posts/2018/04/The-Cost-trend-chart.jpg" alt="The Cost trend chart" /></a>
  
  <p>
    The Cost trend chart
  </p>
</div>

To modify the Cost trend chart follow these steps:

  1. On the Cost trend blade, click on Manage target.
  2. On the Manage target blade, you can modify the date interval of the chart and also set a cost target threshold and notifications via webhook when a certain amount is reached.
  3. Click OK to save your changes.

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2018/04/Modify-the-cost-trend-chart.jpg"><img loading="lazy" src="/assets/img/posts/2018/04/Modify-the-cost-trend-chart.jpg" alt="Modify the cost trend chart" /></a>
  
  <p>
    Modify the cost trend chart
  </p>
</div>

### Cost by resource

With cost by resource, you get a breakdown of your costs of each resource. To do that follow these steps:

  1. In your DevTest Labs instance, select Configuration and policies under the Settings menu.
  2. On the Configuration and policies blade, select Cost by resource under the Cost Tracking menu.
  3. All your resources of the lab are listed on the Cost by resource blade.

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2018/04/Your-costs-by-resource.jpg"><img loading="lazy" src="/assets/img/posts/2018/04/Your-costs-by-resource.jpg" alt="Your costs by resource" /></a>
  
  <p>
    Your costs by resource
  </p>
</div>

## Secure access to labs

The access to your DevTest Labs is determined by Azure Role-Based Access Control (RBAC). To understand how this works, you have to understand the difference between a permission, a role, and a scope defined by RBAC.

  * Permission: Defines access to a specific action (for example, read-access to the storage account)
  * Role: A set of permissions that can be grouped and assigned to a user. For example, the reader role can read all resources.
  * Scope: A level within the hierarchy of an Azure resource, such as a resource group or a virtual machine.

With RBAC, you can segregate duties within your team into roles where you grant only the amount of access necessary to users to perform their job. The three most relevant roles to Azure DevTest Labs are Owner, DevTest Labs User, and Contributor.

### Add an owner or user at the lab level

Owners and users can be added at the lab level via the Azure Portal. This also includes external users with a valid Microsoft account. To add an owner or user, follow these steps:

  1. In your DevTest Labs instance, select Configuration and policies under the Settings menu.
  2. On the Configuration and policies blade, select Access control (IAM) under the Manage menu and click +Add.
  3. On the Add permissions blade, select Owner or DevTest Labs user as the role.
  4. Enter the name or email address and select the user.
  5. Click Save.

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2018/04/Add-a-user-to-the-lab.jpg"><img loading="lazy" src="/assets/img/posts/2018/04/Add-a-user-to-the-lab.jpg" alt="Add a user to the lab" /></a>
  
  <p>
    Add a user to the lab
  </p>
</div>

### Add an external user to a lab using PowerShell

Additionally to the Azure portal, you can add an external user to your Azure DevTest Labs using PowerShell. Before you can do that, you have to add the user as a gust to the Active Directory though. To do that follow these steps:

  1. Open the Azure Active Directory and select Users under the Manage menu.
  2. On the All users blade, click +New guest user on the top of the blade.
  3. Enter the email of the user you want to add as a guest and optionally include a message for the invitation.

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2018/04/Add-a-user-as-guest-to-your-Active-Directory.jpg"><img loading="lazy" src="/assets/img/posts/2018/04/Add-a-user-as-guest-to-your-Active-Directory.jpg" alt="Add a user as a guest to your Active Directory" /></a>
  
  <p>
    Add a user as a guest to your Active Directory
  </p>
</div>

<ol start="4">
  <li>
    After the invitation is sent, the user will show up in your Azure Active Directory.
  </li>
</ol>

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2018/04/The-Azure-Active-Directory-with-the-new-guest-user.jpg"><img loading="lazy" src="/assets/img/posts/2018/04/The-Azure-Active-Directory-with-the-new-guest-user.jpg" alt="The Azure Active Directory with the new guest user" /></a>
  
  <p>
    The Azure Active Directory with the new guest user
  </p>
</div>

&nbsp;

Now, you can add the user to your lab using PowerShell, following these steps:

  1. Create the following variables in PowerShell: 
      * $subscriptionId = &#8220;<Enter Azure subscription ID here>&#8221;
      * $labResourceGroup = &#8220;<Enter lab&#8217;s resource name here>&#8221;
      * $labName = &#8220;<Enter lab name here>&#8221;
      * $userDisplayName = &#8220;<Enter user&#8217;s display name here>&#8221;
  2. Select your subscription (you only have to do this step if you have more than one subscription) 
      * Select-AzureRmSubscription -SubscriptionId $subscriptionId
  3. Get the user object 
      * $adObject = Get-AzureRmADUser -SearchString $userDisplayName
  4. Create the role assignment 
      * $labId = (&#8216;/subscriptions/&#8217; + $subscriptionId + &#8216;/resourceGroups/&#8217; + $labResourceGroup + &#8216;/providers/Microsoft.DevTestLab/labs/&#8217; + $labName)
      * Attention: You need a / in front of subscriptions. The Microsoft documentation doesn&#8217;t have the slash and therefore it won&#8217;t work.
  5. Assign the role to the user: 
      * New-AzureRmRoleAssignment -ObjectId $adObject.Id -RoleDefinitionName &#8216;DevTest Labs User&#8217; -Scope $labId

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2018/04/Add-a-user-to-your-lab-with-PowerShell.jpg"><img loading="lazy" src="/assets/img/posts/2018/04/Add-a-user-to-your-lab-with-PowerShell.jpg" alt="Add a user to your lab with PowerShell" /></a>
  
  <p>
    Add a user to your lab with PowerShell
  </p>
</div>

After you are done, the user will appear in your lab&#8217;s Active Directory. To verify it go to your Azure DevTest Labs &#8211;> Configuration and policies &#8211;> Access control (IAM). There, you can see the previously added user with his assigned role.

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2018/04/Confirm-that-the-user-has-been-added-to-your-lab.jpg"><img loading="lazy" src="/assets/img/posts/2018/04/Confirm-that-the-user-has-been-added-to-your-lab.jpg" alt="Confirm that the user has been added to your lab" /></a>
  
  <p>
    Confirm that the user has been added to your lab
  </p>
</div>

## Use lab settings to set access rights to the environment

You can give your lab users Contributor access rights. This enables them to edit resources such as SQL Server, in the resource group that contains your lab environment. By default, lab users only have the Reader access rights. To modify the user&#8217;s rights, follow these steps:

  1. In your DevTest Labs instance, select Configuration and policies under the Settings menu.
  2. On the Configuration and policies blade, select Lab settings under the Settings menu.
  3. Select Contributor to give users the Contributor access rights.
  4. Click Save

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2018/04/Give-users-the-Contributor-or-Reader-access-rights.jpg"><img loading="lazy" src="/assets/img/posts/2018/04/Give-users-the-Contributor-or-Reader-access-rights.jpg" alt="Give users the Contributor or Reader access rights" /></a>
  
  <p>
    Give users the Contributor or Reader access rights
  </p>
</div>

## Use environments in a lab

You can use Azure Resource Manager (ARM) templates to spin up a complete environment in DevTest labs. These environments can contain multiple VMs, or a SharePoint farm.  <span class="fontstyle0">Following infrastructure-as-code and configuration-as-code best practices, environment templates are managed in source control. Azure DevTest Labs loads all ARM templates directly from your GitHub or VSTS Git repositories. As a result, Resource Manager templates can be used across the entire release cycle, from the test environment to the production environment.</span>

### Configure an ARM template repository

There are a couple of rules which you have to follow when organizing ARM templates in a repository:

  1. The master template file must be named azuredeploy.json.
  2. The parameter file must be named azuredeploy.parameters.json.
  3. <span class="fontstyle0">You can use the parameters _artifactsLocation and _artifactsLocationSasToken to construct the parametersLink URI value, allowing DevTest Labs to automatically manage nested templates.</span>
  4. <span class="fontstyle0">Metadata can be defined to specify the template display name and description. This metadata must be in a file named metadata.json. </span>

<span class="fontstyle0">On the following example metadata file, you can see how to specify the display name and description:</span>

{  
&#8220;itemDisplayName&#8221;: &#8220;Your template name&#8221;,  
&#8220;description&#8221;: &#8220;Your description&#8221;  
}

To add a repository to your Azure DevTest Labs using the Azure portal, follow these steps:

  1. In your DevTest Labs instance, select Configuration and policies under the Settings menu.
  2. On the Configuration and policies blade, select Repositories under the External Resources menu and click +Add.
  3. Enter a name, the Git clone URI, your access token and optionally a branch.
  4. Enter either a folder path that starts with / and is relative to your Git clone URI or your ARM template definition.
  5. Click Save.

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2018/04/Add-a-repository-with-templates-to-your-Azure-DevTest-Labs.jpg"><img loading="lazy" src="/assets/img/posts/2018/04/Add-a-repository-with-templates-to-your-Azure-DevTest-Labs.jpg" alt="Add a repository with templates to your Azure DevTest Labs" /></a>
  
  <p>
    Add a repository with templates to your Azure DevTest Labs
  </p>
</div>

After you added the Repository, you can use the templates when you add a new Resource on the Overview blade of your DevTest Labs.

## Conclusion

This post described the Azure DevTest Labs and showed what you can do with it and how it helps you to minimize the management tasks and also helps to keep your costs low. I talked about creating VMs, custom images, and formulas and how to apply policies like auto-shutdown to the resources within the lab. Furthermore, I showed how to add users to your lab and how to change their access permissions. The last topic was about setting up your own Git repository which helps you to create complete environments with a single deployment process.

For more information about the 70-532 exam get the <a href="http://amzn.to/2EWNWMF" target="_blank" rel="noopener">Exam Ref book from Microsoft</a> and continue reading my blog posts. I am covering all topics needed to pass the exam. You can find an overview of all posts related to the 70-532 exam <a href="/prepared-for-the-70-532-exam/" target="_blank" rel="noopener">here</a>.