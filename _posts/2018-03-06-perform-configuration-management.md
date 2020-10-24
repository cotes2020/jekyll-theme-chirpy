---
title: Perform Configuration Management
date: 2018-03-06T12:21:11+01:00
author: Wolfgang Ofner
categories: [Cloud]
tags: [70-532, Azure, Certification, Exam, learning]
---
In this post, I want to show how to use Windows PowerShell Desired State Configuration (DSC) and the VM Agent to perform various configuration management tasks like remote debugging and at the end, I will show how to use variables in a deployment template.

## Automate configuration management by using PowerShell Desired State Configuration (DSC) or VM Agent (custom script extensions)

The VM Agent gets installed automatically when creating a new VM. It is a lightweight process which can install and configure VM extensions. These extensions can be added through the portal, with PowerShell cmdlets or through the Azure Cross Platform Command Line Interface (Azure CLI)

Popular VM extensions are:

  * Octopus Deploy Tentacle Agent
  * Docker extension
  * Chef extension

### Configure VMs with Custom Script Extension

Custom Script Extensions allow you to automatically download files from Azure store and run PowerShell or Shell scripts to copy files or configure VMs.

<li style="list-style-type: none;">
  <ol>
    <li>
      Open the blade of your Windows Server
    </li>
    <li>
      Under the Settings section, select Extensions <p>
        <div id="attachment_831" style="width: 288px" class="wp-caption aligncenter">
          <a href="/assets/img/posts/2018/02/The-Extensions-option-under-Settings.jpg"><img aria-describedby="caption-attachment-831" loading="lazy" class="size-full wp-image-831" src="/assets/img/posts/2018/02/The-Extensions-option-under-Settings.jpg" alt="The Extensions option under Settings" width="278" height="581" /></a>
          
          <p id="caption-attachment-831" class="wp-caption-text">
            The Extensions option under Settings
          </p>
        </div></li> 
        
        <li>
          On the Extensions blade, click + Add on the top
        </li>
        <li>
          This opens a new blade with all available extensions. Select Custom Script Extension and press Create
        </li>
        <li>
          On the install extension blade upload a PowerShell script file which you want to run when the VM starts. The PowerShell script contains only one line to copy a file from C:\ to the desktop. The entire command is: Copy-Item &#8220;FilePath&#8221; -Destination &#8220;DestinationPath&#8221;. Before I upload the script, I create the file under C:\. <p>
            <div id="attachment_832" style="width: 652px" class="wp-caption aligncenter">
              <a href="/assets/img/posts/2018/02/Adding-the-PowerShell-script-which-will-run-when-the-VM-starts.jpg"><img aria-describedby="caption-attachment-832" loading="lazy" class="size-full wp-image-832" src="/assets/img/posts/2018/02/Adding-the-PowerShell-script-which-will-run-when-the-VM-starts.jpg" alt="Adding the PowerShell script which will run when the VM starts" width="642" height="233" /></a>
              
              <p id="caption-attachment-832" class="wp-caption-text">
                Adding the PowerShell script which will run when the VM starts
              </p>
            </div></li> 
            
            <li>
              Click OK and the script will be downloaded on the Server.
            </li>
            <li>
              When you start your VM, the VM executes the script which opens a Windows Explorer window.
            </li></ol> </li> </ol> 
            
            <div id="attachment_861" style="width: 241px" class="wp-caption aligncenter">
              <a href="/assets/img/posts/2018/03/Copied-the-file-on-the-desktop-using-the-uploaded-script.jpg"><img aria-describedby="caption-attachment-861" loading="lazy" class="size-full wp-image-861" src="/assets/img/posts/2018/03/Copied-the-file-on-the-desktop-using-the-uploaded-script.jpg" alt="Copied the file on the desktop using the uploaded script" width="231" height="231" /></a>
              
              <p id="caption-attachment-861" class="wp-caption-text">
                Copied the file on the desktop using the uploaded script
              </p>
            </div>
            
            <h2>
              Using PowerShell Desired State Configuration (DSC)
            </h2>
            
            <p>
              DSC is a management platform which was introduced with PowerShell 4.0. It has an easy, declarative syntax which simplifies configuration and management tasks. With DSC it is possible to describe what application resources you want to add or remove based on the current state of a server node. A DSC describes the state of one or more resources. These resources can be the Registry or the filesystem. Tasks you can achieve with a DSC script are:
            </p>
            
            <ul>
              <li>
                Manage server roles and Windows features
              </li>
              <li>
                Copy files and folders
              </li>
              <li>
                Deploy software
              </li>
              <li>
                Run PowerShell scripts
              </li>
            </ul>
            
            <p>
              DSC extends PowerShell (4.0 and upwards) with a Configuration keyword which is used to express the desired state of one or more target nodes.
            </p>
            
            <div id="attachment_862" style="width: 341px" class="wp-caption aligncenter">
              <a href="/assets/img/posts/2018/03/Example-of-a-DSC-script.jpg"><img aria-describedby="caption-attachment-862" loading="lazy" class="wp-image-862 size-full" src="/assets/img/posts/2018/03/Example-of-a-DSC-script.jpg" alt="Example of a DSC script for the configuration management" width="331" height="236" /></a>
              
              <p id="caption-attachment-862" class="wp-caption-text">
                Example of a DSC script (<a href="https://github.com/MicrosoftDocs/azure-docs/blob/master/articles/virtual-machines/windows/extensions-dsc-overview.md" target="_blank" rel="noopener noreferrer">Source</a>)
              </p>
            </div>
            
            <h3>
              Custom Resources
            </h3>
            
            <p>
              Many resources are already defined and exposed to DSC. Additionally, it is possible to implement custom resources by creating a PowerShell module. This module includes a MOF (Managed Object Format) file, a script module, and a module manifest. For more information on this topic see the <a href="https://docs.microsoft.com/en-us/powershell/dsc/authoringResource" target="_blank" rel="noopener noreferrer">documentation</a>.
            </p>
            
            <h3>
              Local Configuration Manager
            </h3>
            
            <p>
              The Local Configuration Manager runs on all target notes and is the engine of DSC. It enables you to do the following tasks:
            </p>
            
            <ul>
              <li>
                Pushing configurations to bootstrap a target node
              </li>
              <li>
                Pulling configuration from a specified location to bootstrap or update a target node
              </li>
            </ul>
            
            <p>
              The documentation for the Local Configuration Manager can be found <a href="https://docs.microsoft.com/en-us/powershell/dsc/metaconfig" target="_blank" rel="noopener noreferrer">here</a>.
            </p>
            
            <h2>
              Configure VMs with DSC
            </h2>
            
            <p>
              To configure a VM using DSC, you have to create a PowerShell script which describes the desired configuration state of this VM. This script selects the resources which need to be configured and provides the desired settings. After the script is created, there are several methods to run the script when the VM starts.
            </p>
            
            <h3>
              Creating a configuration script
            </h3>
            
            <p>
              To create a configuration script, create a PowerShell file and include all nodes which you want to configure. If you want to copy files, make sure that the file exists at the source and that the destination also exists.
            </p>
            
            <h3>
              Deploy a DSC script
            </h3>
            
            <p>
              Before you can deploy your DSC script, you have to pack it in a zip file. To do that open PowerShell and enter Publish-AzureVMDscConfiguration Filepath -ConfigurationArchivePath DestinationPath.
            </p>
            
            <p>
             
              <br /> Publish-AzureVMDscConfiguration .\IIS.ps1 -ConfigurationArchivePath .\IIS.ps1.zip<br /> 
              
            </p>
            
            <p>
              After the script is prepared, follow these steps to deploy it to your VM:
            </p>
            
            <ol>
              <li>
                In the Azure Portal open your VM
              </li>
              <li>
                Click on Extensions under the settings menu.
              </li>
              <li>
                On the Extensions blade click + Add, then select PowerShell Desired State Configuration and click Create.
              </li>
              <li>
                On the Install extension blade, upload your zip File, enter the full name for your script and enter the desired version of the DSC extension.
              </li>
              <li>
                Click OK to deploy your script.
              </li>
            </ol>
            
            <div id="attachment_859" style="width: 598px" class="wp-caption aligncenter">
              <a href="/assets/img/posts/2018/03/Deploy-the-DSC-script-to-your-VM.jpg"><img aria-describedby="caption-attachment-859" loading="lazy" class="size-full wp-image-859" src="/assets/img/posts/2018/03/Deploy-the-DSC-script-to-your-VM.jpg" alt="Deploy the DSC script to your VM" width="588" height="579" /></a>
              
              <p id="caption-attachment-859" class="wp-caption-text">
                Deploy the DSC script to your VM
              </p>
            </div>
            
            <h2>
              Enable remote debugging
            </h2>
            
            <p>
              You can use Visual Studio 2015 or 2017 to debug applications which run on your Windows VM. To do that following these steps:
            </p>
            
            <ol>
              <li>
                Open the Cloud Explorer in Visual Studio.
              </li>
              <li>
                Expand the node of your subscription and then the Virtual Machines node.
              </li>
              <li>
                Right-click on the VM you want to debug and select Enable Debugging. This installs the debugging extension on the VM and takes a couple minutes.
              </li>
              <li>
                After the extension is installed, right-click on the VM and select Attach Debugger. This opens a new window with a list of all available processes.
              </li>
              <li>
                Select the process you want to debug and click Attach. The probably most useful process is w3.wp.exe. This process only appears if you have a web application running on your server though.
              </li>
            </ol>
            
            <div id="attachment_854" style="width: 350px" class="wp-caption aligncenter">
              <a href="/assets/img/posts/2018/03/Enable-remote-debugging-on-your-VM-in-Visual-Studio.jpg"><img aria-describedby="caption-attachment-854" loading="lazy" class="size-full wp-image-854" src="/assets/img/posts/2018/03/Enable-remote-debugging-on-your-VM-in-Visual-Studio.jpg" alt="Enable remote debugging on your VM in Visual Studio" width="340" height="631" /></a>
              
              <p id="caption-attachment-854" class="wp-caption-text">
                Enable remote debugging on your VM in Visual Studio
              </p>
            </div>
            
            <h2>
              Implement VM template variables to configure VMs
            </h2>
            
            <p>
              When you create a new resource, for example, a VM from the marketplace, it creates a JSON template file. You can write this template by yourself or you can download the template from an existing VM and modify it. Variables can be used to simplify your template which makes it easier to read and easier to change. They also help you with your configuration management since you only have to change one value to change the configuration for your whole template. In this section, I will edit an existing template to show, how variables can be used to simplify a template.
            </p>
            
            <h3>
              Edit an existing template from the Azure Marketplace
            </h3>
            
            <ol>
              <li>
                In the Azure Portal go to the Marketplace.
              </li>
              <li>
                Enter template in the search, then select Template deployment and click Create.
              </li>
              <li>
                On the custom deployment blade select Create a Windows virtual machine. This opens the Deploy a simple Windows VM blade.
              </li>
              <li>
                Click Edit template to open the template.
              </li>
              <li>
                When you scroll down a bit, you will find the &#8220;variables&#8221; section. All variables are configured in this section. Change any of the values or add your own variables.
              </li>
              <li>
                To use the variable use &#8220;[variables(&#8216;yourVariableName&#8217;)]&#8221;.
              </li>
            </ol>
            
            <div id="attachment_857" style="width: 710px" class="wp-caption aligncenter">
              <a href="/assets/img/posts/2018/03/Variables-in-the-deployment-template.jpg"><img aria-describedby="caption-attachment-857" loading="lazy" class="wp-image-857" src="/assets/img/posts/2018/03/Variables-in-the-deployment-template.jpg" alt="Variables in the deployment template" width="700" height="520" /></a>
              
              <p id="caption-attachment-857" class="wp-caption-text">
                Variables in the deployment template
              </p>
            </div>
            
            <h2>
              Conclusion
            </h2>
            
            <p>
              In this post, I talked about what Desired State Configuration is and what you can do with it. Then, I showed how to enable remote debugging using Visual Studio 2015 or 2017 and in the last section, I explained how variables in templates work and how they can be used to simplify the template.
            </p>
            
            <p>
              For more information about the 70-532 exam get the <a href="http://amzn.to/2EWNWMF" target="_blank" rel="noopener noreferrer">Exam Ref book from Microsoft</a> and continue reading my blog posts. I am covering all topics needed to pass the exam. You can find an overview of all posts related to the 70-532 exam <a href="/prepared-for-the-70-532-exam/" target="_blank" rel="noopener noreferrer">here</a>.
            </p>