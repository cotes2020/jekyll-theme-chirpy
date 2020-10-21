---
title: Design and implement ARM VM Azure Storage
date: 2018-03-12T15:59:16+01:00
author: Wolfgang Ofner
categories: [Cloud]
tags: [70-532, Azure, Certification, Exam, learning]
---
Azure Storage provides more functionality than just attaching data disks to your VM. In this post, I will talk about creating a file storage account, how to create and access a file share using PowerShell and C# and how to enable caching for your disks.

## Plan for storage capacity

VMs in Azure have a local temp drive (D: on Windows and \dev\sdb1 on Linux) and a system disk. The disks are saved as Blob in Azure Storage. This means that this Blob governs the performance and the capacity of the disk. There are two factors when it comes to storage performance and capacity:

  1. Is the disk standard (HDD) or premium (SSD)?
  2. Is the disk managed or unmanaged?

The difference between managed and unmanaged is that unmanaged disks require the creation of an Azure Storage Account in your subscription whereas managed disks manage the Storage Account for you. This means that you only need to specify the size and type of your disk and Azure manages the rest for you. The main advantage of using managed disks is that the Storage Account does not limit the disk. See the following table with the differences between standard and premium managed and unmanaged disks:

### <span class="fontstyle2"><span style="font-size: small;">Comparison of Standard and Premium disks (<a href="http://amzn.to/2EWNWMF" target="_blank" rel="noopener noreferrer">Source</a>)</span></span>

<div class="table-responsive">
  <table class="table table-striped table-bordered table-hover">
    <tr>
      <td>
        Feature
      </td>
      
      <td>
        Standard (unmanaged)
      </td>
      
      <td>
        Standard (managed)
      </td>
      
      <td>
        Premium (unmanaged)
      </td>
      
      <td>
        Premium (managed)
      </td>
    </tr>
    
    <tr>
      <td>
        Max IOPS for storage account
      </td>
      
      <td>
        20k IOPS
      </td>
      
      <td>
        N/A
      </td>
      
      <td>
        60k -127.5k IOPS
      </td>
      
      <td>
        N/A
      </td>
    </tr>
    
    <tr>
      <td>
        Max bandwidth for storage account
      </td>
      
      <td>
        N/A
      </td>
      
      <td>
        N/A
      </td>
      
      <td>
        50 Gbps
      </td>
      
      <td>
        N/A
      </td>
    </tr>
    
    <tr>
      <td>
        Max storage capacity per storage account
      </td>
      
      <td>
        500 TB
      </td>
      
      <td>
        N/A
      </td>
      
      <td>
        35 TB
      </td>
      
      <td>
        N/A
      </td>
    </tr>
    
    <tr>
      <td>
        Max IOPS per VM
      </td>
      
      <td>
        Depends on VM size
      </td>
      
      <td>
        Depends on VM size
      </td>
      
      <td>
        Depends on VM size
      </td>
      
      <td>
        Depends on VM size
      </td>
    </tr>
    
    <tr>
      <td>
        Max throughput per VM
      </td>
      
      <td>
        Depends on VM size
      </td>
      
      <td>
        Depends on VM size
      </td>
      
      <td>
        Depends on VM size
      </td>
      
      <td>
        Depends on VM size
      </td>
    </tr>
    
    <tr>
      <td>
        Max disk size
      </td>
      
      <td>
        4TB
      </td>
      
      <td>
        32GB &#8211; 4TB
      </td>
      
      <td>
        32GB &#8211; 4TB
      </td>
      
      <td>
        32GB &#8211; 4TB
      </td>
    </tr>
    
    <tr>
      <td>
        Max 8 KB IOPS per disk
      </td>
      
      <td>
        300 &#8211; 500 IOPS
      </td>
      
      <td>
        500 IOPS
      </td>
      
      <td>
        500 &#8211; 7,500 IOPS
      </td>
      
      <td>
        120 &#8211; 7,500 IOPS
      </td>
    </tr>
    
    <tr>
      <td>
        Max throughput per disk
      </td>
      
      <td>
        60 MB/s
      </td>
      
      <td>
        60 MB/s
      </td>
      
      <td>
        100 MB/s &#8211; 250 MB/s
      </td>
      
      <td>
        25 MB/s &#8211; 250 MB/s
      </td>
    </tr>
  </table>
</div>

IOPS is a unit of measure which counts the number of input and output operations per second. Usually, Azure VMs allow the number of disks you can attach is twice the number of CPU cores of your VM.

## Configure Storage Pools

Before you can configure a storage pool, you have to add disks to your VM.

### Create new disks for your VM

Follow these steps:

  1. Open your VM in the Azure Portal.
  2. Under the Settings menu click Disks.
  3. On the Disks blade, click + Add data disk.
  4. In the drop-down menu under Name select Create disk.
  5. On the Create managed disk blade provide a Name, Resource group and the Account type (SSD or HDD).
  6. As Source type select None (empty disk) and provide your desired size.

<div id="attachment_906" style="width: 864px" class="wp-caption aligncenter">
  <a href="/wp-content/uploads/2018/03/Create-a-new-disk-for-your-VM.jpg"><img aria-describedby="caption-attachment-906" loading="lazy" class="size-full wp-image-906" src="/wp-content/uploads/2018/03/Create-a-new-disk-for-your-VM.jpg" alt="Create a new disk for your VM" width="854" height="516" /></a>
  
  <p id="caption-attachment-906" class="wp-caption-text">
    Create a new disk for your VM
  </p>
</div>

<ol start="7">
  <li>
    Click Create.
  </li>
  <li>
    You can add more disks or click Save on the top of the blade.
  </li>
</ol>

<div id="attachment_907" style="width: 710px" class="wp-caption aligncenter">
  <a href="/wp-content/uploads/2018/03/Adding-disks-to-your-VM.jpg"><img aria-describedby="caption-attachment-907" loading="lazy" class="wp-image-907" src="/wp-content/uploads/2018/03/Adding-disks-to-your-VM.jpg" alt="Adding disks to your VM" width="700" height="470" /></a>
  
  <p id="caption-attachment-907" class="wp-caption-text">
    Adding disks to your VM
  </p>
</div>

### Create a Storage Pool

Storage Pools enable you to group together a set of disks and then create a volume from the available aggregate capacity. To do that follow these steps:

  1. Connect to your Windows VM using RDP.
  2. Open the Server Manager.
  3. Click on File and Storage Services and then Storage Pools.

<div id="attachment_901" style="width: 1010px" class="wp-caption aligncenter">
  <a href="/wp-content/uploads/2018/03/Adding-a-new-Storage-Pool-to-your-VM.jpg"><img aria-describedby="caption-attachment-901" loading="lazy" class="wp-image-901" src="/wp-content/uploads/2018/03/Adding-a-new-Storage-Pool-to-your-VM.jpg" alt="Adding a new Storage Pool to your VM" width="1000" height="173" /></a>
  
  <p id="caption-attachment-901" class="wp-caption-text">
    Adding a new Storage Pool to your VM
  </p>
</div>

<ol start="4">
  <li>
    Provide a name for your Storage Pool and click Next.
  </li>
  <li>
    Select all disks which you want to add to the storage pool and click Next.
  </li>
</ol>

<div id="attachment_902" style="width: 710px" class="wp-caption aligncenter">
  <a href="/wp-content/uploads/2018/03/Add-the-physical-disks-to-the-Storage-Pool.jpg"><img aria-describedby="caption-attachment-902" loading="lazy" class="wp-image-902" src="/wp-content/uploads/2018/03/Add-the-physical-disks-to-the-Storage-Pool.jpg" alt="Add the physical disks to the Storage Pool" width="700" height="395" /></a>
  
  <p id="caption-attachment-902" class="wp-caption-text">
    Add the physical disks to the Storage Pool
  </p>
</div>

<ol start="6">
  <li>
    Click Create and then Close to create the storage pool.
  </li>
</ol>

### Create a new Virtual Disk

  1. After the storage pool is created, right-click on it and select New Virtual Disk&#8230;

<div id="attachment_904" style="width: 710px" class="wp-caption aligncenter">
  <a href="/wp-content/uploads/2018/03/Create-a-new-virtual-disk-.jpg"><img aria-describedby="caption-attachment-904" loading="lazy" class="wp-image-904" src="/wp-content/uploads/2018/03/Create-a-new-virtual-disk-.jpg" alt="Create a new virtual disk" width="700" height="228" /></a>
  
  <p id="caption-attachment-904" class="wp-caption-text">
    Create a new virtual disk
  </p>
</div>

<ol start="2">
  <li>
    Select the storage pool you just created and click OK.
  </li>
  <li>
    In the wizard enter a name for the virtual disk and click Next twice.
  </li>
  <li>
    Select Simple as your layout and click Next. You don&#8217;t need mirroring because Azure already replicates your data three times.
  </li>
  <li>
    For the provisioning select and click Next.
  </li>
  <li>
    Select Maximum size, so that the new virtual disk uses the complete capacity of the storage pool and click Next.
  </li>
  <li>
    In the Confirm selections window, click Create.
  </li>
  <li>
    After the new volume is created click Next on the first page of the wizard.
  </li>
  <li>
    Select the disk you just created and click Next.
  </li>
  <li>
    Leave the volume size as it is and click Next.
  </li>
  <li>
    Leave Assign to Drive letter selected and optionally change the drive letter, then click Next.
  </li>
  <li>
    In the last window, click Create and then Close to finish the process.
  </li>
  <li>
    After the wizard is completed, open the Windows Explorer and you can see your new drive.
  </li>
</ol>

<div id="attachment_913" style="width: 710px" class="wp-caption aligncenter">
  <a href="/wp-content/uploads/2018/03/The-Windows-Explorer-with-the-new-mapped-disk.jpg"><img aria-describedby="caption-attachment-913" loading="lazy" class="wp-image-913" src="/wp-content/uploads/2018/03/The-Windows-Explorer-with-the-new-mapped-disk.jpg" alt="The Windows Explorer with the new mapped disk" width="700" height="144" /></a>
  
  <p id="caption-attachment-913" class="wp-caption-text">
    The Windows Explorer with the new mapped disk
  </p>
</div>

You can increase the IOPS and total storage capacity if you use multiple blobs for your disks.

For Linux, you have to use the Logical Volume Manager to create the volume.

## Configure disk caching

Each disk you attach to your VM has a local cache which can improve the performance of read and write operations. This cache is outside your VM (it&#8217;s on the host of your VM) and uses a combination of memory and disks on the host. There are three caching options available:

  1. None: No caching
  2. Read-Only: The cache is only used for read operations. If the needed data is not found in the cache, it will be loaded into it form the Azure Storage. Write operations go directly into the Azure Storage.
  3. Read/Write: The cache is used for read and write operations. The write operations will be written into Azure Storage later.

The default options are Read/Write for the operating system disk and Read-Only for the data disk. Data disks can turn off caching, operating system disk can&#8217;t. The reason for this behavior is that Azure Storage can provide better performance for random I/Os than the local disk. The big advantage of caching is obviously the better performance but also minimizes caching your costs because you don&#8217;t pay anything if you don&#8217;t access your Storage Account.

### Enable disk caching

To enable caching for your disk follow these steps:

  1. Open your VM in the Azure Portal.
  2. Under the Settings menu, select Disks.
  3. Select Edit on the Disks blade.
  4. Select the Host Caching drop-down and set it to the desired configuration.

<div id="attachment_910" style="width: 710px" class="wp-caption aligncenter">
  <a href="/wp-content/uploads/2018/03/Enable-disk-caching.jpg"><img aria-describedby="caption-attachment-910" loading="lazy" class="wp-image-910" src="/wp-content/uploads/2018/03/Enable-disk-caching.jpg" alt="Enable disk caching" width="700" height="265" /></a>
  
  <p id="caption-attachment-910" class="wp-caption-text">
    Enable disk caching
  </p>
</div>

<ol start="5">
  <li>
    Click Save.
  </li>
</ol>

## Enable geo-replication

With geo-replication, you can copy your data into other data centers, even in other regions all around the world. Additionally to geo-replication, Azure created three copies of your data within the data center where they reside. Keep in mind that geo-replication is not synchronized across blob files. To save money and keep your data safe configure your VM disks to use locally redundant replication.

## Configure shared storage using Azure File storage

Azure File storage enables your VMs to access files using a shared location within the same region your VMs. The VMs don&#8217;t even have to be in the same subscription or storage account than your Azure File storage. It only has to be in the same region. It can be compared with a network drive since you can also map it like a normal network drive. Common scenarios are:

  * Support applications which need a file share
  * Centralize storage for logs or crash dumps
  * Provide access to shared application settings

To create an Azure File storage you need an Azure Storage account. The access is controlled by the storage account name and a key. As long as your VM and the File storage are in the same region, the VM can access the storage using the storage credentials.

Each share is an SMB file share and can contain an unlimited number of directories. The maximum file size is one terabyte and the maximum size of a share is five terabytes. A share has a maximum performance of 1,000 IOPS and a throughput of 60 MB/s.

## Creating a file share using Azure Storage {#CreateAzureStorage}

Before you can create a file share, you need to create a storage account. To do that follow these steps:

  1. Click on Storage accounts in the Azure Portal.
  2. Click + Add on the top of the blade.
  3. On the Create storage account blade provide a name, Subscription, Resource group and Location. Enable Secure transfer required if you want to use https only.

<div id="attachment_911" style="width: 299px" class="wp-caption aligncenter">
  <a href="/wp-content/uploads/2018/03/Create-a-new-storage-account.jpg"><img aria-describedby="caption-attachment-911" loading="lazy" class="wp-image-911" src="/wp-content/uploads/2018/03/Create-a-new-storage-account.jpg" alt="Create a new Azure storage account" width="289" height="700" /></a>
  
  <p id="caption-attachment-911" class="wp-caption-text">
    Create a new storage account
  </p>
</div>

<ol start="4">
  <li>
    Click Create.
  </li>
</ol>

With the storage account created, I can use PowerShell to create a file share. To do that I need the storage account name and the storage account key. To get this information open your storage account and click on Access keys under the Settings menu.

<div id="attachment_915" style="width: 710px" class="wp-caption aligncenter">
  <a href="/wp-content/uploads/2018/03/Keys-for-the-storage-account.jpg"><img aria-describedby="caption-attachment-915" loading="lazy" class="wp-image-915" src="/wp-content/uploads/2018/03/Keys-for-the-storage-account.jpg" alt="Keys for the storage account" width="700" height="422" /></a>
  
  <p id="caption-attachment-915" class="wp-caption-text">
    Keys for the storage account
  </p>
</div>

To create a file share using PowerShell use: $context = New-AzureStorageContext -Storage-AccountName &#8220;YourStorageAccountName&#8221; and then New-AzureStorageShare &#8220;YourShareName&#8221; -Context $context. The share name must be a valid DNS name, lowercase and between 3 and 63 characters long.

<div id="attachment_916" style="width: 710px" class="wp-caption aligncenter">
  <a href="/wp-content/uploads/2018/03/Create-a-file-share-using-PowerShell.jpg"><img aria-describedby="caption-attachment-916" loading="lazy" class="wp-image-916" src="/wp-content/uploads/2018/03/Create-a-file-share-using-PowerShell.jpg" alt="Create a file share using PowerShell" width="700" height="126" /></a>
  
  <p id="caption-attachment-916" class="wp-caption-text">
    Create a file share using PowerShell
  </p>
</div>

### Mounting the file share {#MountingTheFileShare}

To access the share follow these steps:

  1. Connect to your VM via RDP.
  2. Open PowerShell or the command promp.
  3. Enter command to add your Azure Storage account credentials to the Windows Credentials Manager: cmdkey /add:<Storage-AccountName>.file.core.windows.net /user:<StorageAccountName> /pass:<Storage-AccountKey>.
  4. Replace the values within <> with your credentials. You can find your credentials in the Azure Portal in your Storage Account.

<div id="attachment_948" style="width: 710px" class="wp-caption aligncenter">
  <a href="/wp-content/uploads/2018/03/Add-the-Azure-Storage-account-credentials-to-the-Windows-Credentials-Manager.jpg"><img aria-describedby="caption-attachment-948" loading="lazy" class="wp-image-948" src="/wp-content/uploads/2018/03/Add-the-Azure-Storage-account-credentials-to-the-Windows-Credentials-Manager.jpg" alt="Add the Azure Storage account credentials to the Windows Credentials Manager" width="700" height="64" /></a>
  
  <p id="caption-attachment-948" class="wp-caption-text">
    Add the Azure Storage account credentials to the Windows Credentials Manager
  </p>
</div>

<ol start="5">
  <li>
    To mount the file share to a drive letter use net use z: \\<Storage-AccountName>.file.core.windows.net\<ShareName>. For example net use z: \\<Storage-AccountName>.file.core.windows.net\<ShareName>.  Replace the values within <> with your storage account name and share name
  </li>
</ol>

<div id="attachment_939" style="width: 710px" class="wp-caption aligncenter">
  <a href="/wp-content/uploads/2018/03/Map-the-file-to-drive-letter-Z.jpg"><img aria-describedby="caption-attachment-939" loading="lazy" class="wp-image-939" src="/wp-content/uploads/2018/03/Map-the-file-to-drive-letter-Z.jpg" alt="Map the file to drive letter Z" width="700" height="75" /></a>
  
  <p id="caption-attachment-939" class="wp-caption-text">
    Map the file to drive letter Z
  </p>
</div>

<ol start="6">
  <li>
    Now you can find the file share in the Windows Explorer.
  </li>
</ol>

<div id="attachment_940" style="width: 710px" class="wp-caption aligncenter">
  <a href="/wp-content/uploads/2018/03/The-mapped-file-share.jpg"><img aria-describedby="caption-attachment-940" loading="lazy" class="wp-image-940" src="/wp-content/uploads/2018/03/The-mapped-file-share.jpg" alt="The mapped file share" width="700" height="224" /></a>
  
  <p id="caption-attachment-940" class="wp-caption-text">
    The mapped file share
  </p>
</div>

### Access the file share using PowerShell

You can upload or download file to and from the file share using PowerShell. Before I start, I uploaded a text file to the file share and renamed it to fileshare.txt

  1. To work on your storage account, you have to get its context using $variable = New-AzureStorageContext -StorageAccountName <Storage-Account-Name> -StorageAccountKey <Storage-AccountKey>. Replace the values within <> with your storage account name and your key.
  2. To download a file to your current directory use Get-AzureStorageFileContent -ShareName <ShareName> -Path <PathToTheFileYouWantToDownload> -Context $variable.

<div id="attachment_941" style="width: 710px" class="wp-caption aligncenter">
  <a href="/wp-content/uploads/2018/03/Download-a-file-from-your-file-share-using-PowerShell.jpg"><img aria-describedby="caption-attachment-941" loading="lazy" class="wp-image-941" src="/wp-content/uploads/2018/03/Download-a-file-from-your-file-share-using-PowerShell.jpg" alt="Download a file from your file share using PowerShell" width="700" height="57" /></a>
  
  <p id="caption-attachment-941" class="wp-caption-text">
    Download a file from your file share using PowerShell
  </p>
</div>

### Access the file share using C#

For this example, I create a new C# console application. Then follow these steps to access the file share:

  1. Install the WindowsAzure.Storage and the WindowsAzure.ConfigurationManager NuGet Packages.
  2. Add your storage account credentials to the app.config file.

<div id="attachment_943" style="width: 710px" class="wp-caption aligncenter">
  <a href="/wp-content/uploads/2018/03/Add-the-storage-connection-string-to-App.config.jpg"><img aria-describedby="caption-attachment-943" loading="lazy" class="wp-image-943" src="/wp-content/uploads/2018/03/Add-the-storage-connection-string-to-App.config.jpg" alt="Add the storage connection string to App.config" width="700" height="48" /></a>
  
  <p id="caption-attachment-943" class="wp-caption-text">
    Add the storage connection string to App.config
  </p>
</div>

<ol start="3">
  <li>
    Connect to your storage account and get the reference from the file share.
  </li>
</ol>

<div id="attachment_944" style="width: 552px" class="wp-caption aligncenter">
  <a href="/wp-content/uploads/2018/03/Connect-to-your-storage-account-and-get-the-reference-from-the-file-share.jpg"><img aria-describedby="caption-attachment-944" loading="lazy" class="size-full wp-image-944" src="/wp-content/uploads/2018/03/Connect-to-your-storage-account-and-get-the-reference-from-the-file-share.jpg" alt="Connect to your storage account and get the reference from the file share" width="542" height="136" /></a>
  
  <p id="caption-attachment-944" class="wp-caption-text">
    Connect to your storage account and get the reference from the file share
  </p>
</div>

<ol start="4">
  <li>
    Get a reference to your root directory and to the file you want to download.
  </li>
</ol>

<div id="attachment_945" style="width: 434px" class="wp-caption aligncenter">
  <a href="/wp-content/uploads/2018/03/Get-a-reference-to-your-root-directory-and-to-the-file-you-want-to-download.jpg"><img aria-describedby="caption-attachment-945" loading="lazy" class="size-full wp-image-945" src="/wp-content/uploads/2018/03/Get-a-reference-to-your-root-directory-and-to-the-file-you-want-to-download.jpg" alt="Get a reference to your root directory and to the file you want to download" width="424" height="80" /></a>
  
  <p id="caption-attachment-945" class="wp-caption-text">
    Get a reference to your root directory and to the file you want to download
  </p>
</div>

<ol start="5">
  <li>
    Download the file to your computer.
  </li>
</ol>

<div id="attachment_946" style="width: 539px" class="wp-caption aligncenter">
  <a href="/wp-content/uploads/2018/03/Download-the-file-to-your-computer.jpg"><img aria-describedby="caption-attachment-946" loading="lazy" class="size-full wp-image-946" src="/wp-content/uploads/2018/03/Download-the-file-to-your-computer.jpg" alt="Download the file to your computer" width="529" height="27" /></a>
  
  <p id="caption-attachment-946" class="wp-caption-text">
    Download the file to your computer
  </p>
</div>

<ol start="6">
  <li>
    You can also upload a file by getting a reference to your directory and then upload the file using UploadText
  </li>
</ol>

<div id="attachment_947" style="width: 649px" class="wp-caption aligncenter">
  <a href="/wp-content/uploads/2018/03/Upload-a-file-to-your-file-share.jpg"><img aria-describedby="caption-attachment-947" loading="lazy" class="size-full wp-image-947" src="/wp-content/uploads/2018/03/Upload-a-file-to-your-file-share.jpg" alt="Upload a file to your file share" width="639" height="43" /></a>
  
  <p id="caption-attachment-947" class="wp-caption-text">
    Upload a file to your file share
  </p>
</div>

You have to replace my placeholder strings with valid values for a filename or share name. My example project is on <a href="https://github.com/WolfgangOfner/Azure-StorageClient" target="_blank" rel="noopener noreferrer">GitHub</a>.

## **<span lang="EN-US"><span style="color: #000000; font-family: Calibri;">Disk encryption</span></span>**

Before you can encrypt the disk of your VM, you have to do some set up steps.

### **<span lang="EN-US"><span style="color: #000000; font-family: Calibri;">Set up</span></span>**

To set up your Azure environment to encrypt the disks of your VMs, you have to do an application registration and create a Key vault.

#### Azure Active Directory App Registration

To register an app in the AAD follow these steps:

  1. In the Azure Portal go to the Azure Active Directory.
  2. Select App registrations under the Manage menu and click on + New application registration.
  3. On the Create blade, provide a name and Sign-on URL and click on Create.

<div id="attachment_976" style="width: 321px" class="wp-caption aligncenter">
  <a href="/wp-content/uploads/2018/03/Create-an-application-registration.jpg"><img aria-describedby="caption-attachment-976" loading="lazy" class="size-full wp-image-976" src="/wp-content/uploads/2018/03/Create-an-application-registration.jpg" alt="Create an application registration" width="311" height="237" /></a>
  
  <p id="caption-attachment-976" class="wp-caption-text">
    Create an application registration
  </p>
</div>

<ol start="4">
  <li>
    On the App registrations blade, select All apps from the drop-down list on the top and copy the Application Id of your newly created app. This id is the AAD client id which I will need later.
  </li>
</ol>

<div id="attachment_977" style="width: 710px" class="wp-caption aligncenter">
  <a href="/wp-content/uploads/2018/03/Get-the-AAD-client-Id.jpg"><img aria-describedby="caption-attachment-977" loading="lazy" class="wp-image-977" src="/wp-content/uploads/2018/03/Get-the-AAD-client-Id.jpg" alt="Get the AAD client Id" width="700" height="212" /></a>
  
  <p id="caption-attachment-977" class="wp-caption-text">
    Get the AAD client Id
  </p>
</div>

<ol start="5">
  <li>
    Click on your application and then select Settings.
  </li>
  <li>
    Select Key under the Api Access.
  </li>
  <li>
    Enter a description and set the expire that for the key on the Keys blade.
  </li>
  <li>
    Click Save. After the key is created, the hidden key value is display. It is important that you copy the key because after you close the window, it won&#8217;t be displayed again. This key is the client secret for later.
  </li>
</ol>

<div id="attachment_978" style="width: 710px" class="wp-caption aligncenter">
  <a href="/wp-content/uploads/2018/03/Create-a-client-secret.jpg"><img aria-describedby="caption-attachment-978" loading="lazy" class="wp-image-978" src="/wp-content/uploads/2018/03/Create-a-client-secret.jpg" alt="Create a client secret" width="700" height="266" /></a>
  
  <p id="caption-attachment-978" class="wp-caption-text">
    Create a client secret
  </p>
</div>

#### Create a Key vault

  1. The next step is to create a Key vault. To do that click on All services and search for Key vaults.
  2. On the Key vaults blade, click on + add.
  3. Provide a Name, Subscription, Resource Group and Location on the Create key vault blade.
  4. Click on Access policies and the on + Add new.
  5. On the Add access policy blade, click on Select principal and search for your previously create application registration.
  6. In the Key permissions drop-down list, select Wrap Key.

<div id="attachment_979" style="width: 710px" class="wp-caption aligncenter">
  <a href="/wp-content/uploads/2018/03/Create-a-new-key-vault-with-the-previously-created-application-registration.jpg"><img aria-describedby="caption-attachment-979" loading="lazy" class="wp-image-979" src="/wp-content/uploads/2018/03/Create-a-new-key-vault-with-the-previously-created-application-registration.jpg" alt="Create a new key vault with the previously created application registration" width="700" height="350" /></a>
  
  <p id="caption-attachment-979" class="wp-caption-text">
    Create a new key vault with the previously created application registration
  </p>
</div>

<ol start="7">
  <li>
    In the Secret permission drop-down list, select Set.
  </li>
  <li>
    Click OK twice and then Create
  </li>
  <li>
    After your Key vault is created, click on Access policies under the Settings menu.
  </li>
  <li>
    On the Access policies blade, click on Click to show advanced access policies and select all three checkboxes.
  </li>
</ol>

<div id="attachment_980" style="width: 710px" class="wp-caption aligncenter">
  <a href="/wp-content/uploads/2018/03/Enable-all-access-policies.jpg"><img aria-describedby="caption-attachment-980" loading="lazy" class="wp-image-980" src="/wp-content/uploads/2018/03/Enable-all-access-policies.jpg" alt="Enable all access policies" width="700" height="516" /></a>
  
  <p id="caption-attachment-980" class="wp-caption-text">
    Enable all access policies
  </p>
</div>

<ol start="11">
  <li>
    Still on the Access policies blade, click on your User (mine starts with 789c&#8230; on the screenshot above).
  </li>
  <li>
    In the Key permissions drop-down list, check Select all and click OK.
  </li>
</ol>

<div id="attachment_981" style="width: 308px" class="wp-caption aligncenter">
  <a href="/wp-content/uploads/2018/03/Give-your-user-all-key-permissions.jpg"><img aria-describedby="caption-attachment-981" loading="lazy" class="size-full wp-image-981" src="/wp-content/uploads/2018/03/Give-your-user-all-key-permissions.jpg" alt="Give your user all key permissions" width="298" height="564" /></a>
  
  <p id="caption-attachment-981" class="wp-caption-text">
    Give your user all key permissions
  </p>
</div>

After all these steps, you can encrypt your Windows VM with Powershell, CLI or with a template and your Linux VM with CLI or with a template.

### **<span lang="EN-US"><span style="color: #000000; font-family: Calibri;">Windows</span></span>**

To demonstrate how to encrypt a Windows VM, I created a new Windows Server 2016 VM with the name WinServer in the resource group WinRg.

#### **<span lang="EN-US"><span style="color: #000000; font-family: Calibri;">Powershell</span></span>**

To encrypt your Windows VM using Powershell follow these steps:

  1. Login to your Azure account with Login-AzureRmAccount.
  2. Select your Subscription with Select-AzureRmSubscription -SubscriptionName &#8220;YourSubscriptionName&#8221;.
  3. $resourceGroupName = &#8220;YourResourceGroup&#8221;
  4. $vmName = &#8220;YourVmName&#8221;
  5. $clientID = &#8220;YourAadClientId&#8221; (you copied that value during the setup process)
  6. $clientSecret = &#8220;YourClientSecret&#8221; (you copied that value during the set up process)
  7. $keyVaultName = &#8220;YourKeyVaultName&#8221;
  8. $keyVault = Get-AzureRmKeyVault -VaultName $keyVaultName -ResourceGroupName $resourceGroupName
  9. $diskEncryptionKeyVaultUrl = $keyVault.VaultUri
 10. $keyVaultResourceId = $keyVault.ResourceId
 11. Set-AzureRmKeyVaultAccessPolicy -VaultName $keyVaultName -ResourceGroupName $resourceGroupName -EnabledForDiskEncryption
 12. Set-AzureRmVMDiskEncryptionExtension -ResourceGroupName $resourceGroupName -VMName $vmName -AadClientID $clientID -AadClientSecret $clientSecret -DiskEncryptionKeyVaultUrl $diskEncryptionKeyVaultUrl -DiskEncryptionKeyVaultId $keyVaultResourceId
 13. This starts the Encryption and takes around 10 -15 minutes. After the encryption is done, you can check, if your disks are encrypted with Get-AzureRmVMDiskEncryptionStatus -ResourceGroupName $resourceGroupName -VMName $vmName

<div id="attachment_989" style="width: 710px" class="wp-caption aligncenter">
  <a href="/wp-content/uploads/2018/03/Encrypt-a-Windows-VM-using-PowerShell.jpg"><img aria-describedby="caption-attachment-989" loading="lazy" class="wp-image-989" src="/wp-content/uploads/2018/03/Encrypt-a-Windows-VM-using-PowerShell.jpg" alt="Encrypt a Windows VM using PowerShell" width="700" height="515" /></a>
  
  <p id="caption-attachment-989" class="wp-caption-text">
    Encrypt a Windows VM using PowerShell
  </p>
</div>

#### **<span lang="DE-AT"><span style="color: #000000; font-family: Calibri;">Template</span></span>**

Additionally to the PowerShell encryption can you encrypt your VM with a template. Go to <a href="https://github.com/Azure/azure-quickstart-templates/tree/master/201-encrypt-running-windows-vm" target="_blank" rel="noopener noreferrer">GitHub</a> and then click on deploy to Azure. This opens the template in Azure. Enter the following values:

<div id="attachment_982" style="width: 669px" class="wp-caption aligncenter">
  <a href="/wp-content/uploads/2018/03/Enable-encryption-on-a-running-Windows-VM-template.jpg"><img aria-describedby="caption-attachment-982" loading="lazy" class="wp-image-982" src="/wp-content/uploads/2018/03/Enable-encryption-on-a-running-Windows-VM-template.jpg" alt="Enable encryption on a running Windows VM template" width="659" height="700" /></a>
  
  <p id="caption-attachment-982" class="wp-caption-text">
    Enable encryption on a running Windows VM template
  </p>
</div>

The Aad Client ID and Aad Client Secret are the values you copied during the setup process. After you entered your values, accept the terms and conditions and click on Purchase.

### **<span lang="DE-AT"><span style="color: #000000; font-family: Calibri;">Linux</span></span>**

To demonstrate how to encrypt a Windows VM, I created a new Kali Linux VM with the name Linux in the resource group WinRg.

#### **<span lang="DE-AT"><span style="color: #000000; font-family: Calibri;">Template</span></span>**

Go to <a href="https://github.com/Azure/azure-quickstart-templates/tree/master/201-encrypt-running-linux-vm" target="_blank" rel="noopener noreferrer">GitHub</a> and then click on deploy to Azure. This opens the template in Azure. Enter the following values:

<div id="attachment_985" style="width: 630px" class="wp-caption aligncenter">
  <a href="/wp-content/uploads/2018/03/Create-encryption-on-a-running-Linux-VM-with-the-template.jpg"><img aria-describedby="caption-attachment-985" loading="lazy" class="wp-image-985" src="/wp-content/uploads/2018/03/Create-encryption-on-a-running-Linux-VM-with-the-template.jpg" alt="Create encryption on a running Linux VM with the template" width="620" height="700" /></a>
  
  <p id="caption-attachment-985" class="wp-caption-text">
    Create encryption on a running Linux VM with the template
  </p>
</div>

The Aad Client ID and Aad Client Secret are the values you copied during the set up process. After you entered your values, accept the terms and conditions and click on Purchase.

**<span lang="DE-AT"><span style="color: #000000; font-family: Calibri;">CLI</span></span>**

To encrypt your VM disk with Azure CLI see the <a href="https://docs.microsoft.com/en-us/azure/security/azure-security-disk-encryption#cli-commands" target="_blank" rel="noopener noreferrer">documentation</a>.

## StoreSimple

The Azure StoreSimple Virtual Array is an integrated storage solution which manages storage tasks between an on-premises virtual array running in a hypervisor and Microsoft Azure cloud storage. It is a cost-effective file server or iSCSI server solution which is well-suited for infrequently accessed archival data. The virtual array supports the SMB and iSCSI protocol. It runs in your existing hypervisor infrastructure and provides tiering to the cloud, cloud backup, fast restore and disaster recovery features.

The following table summarizes the most important features (<a href="https://docs.microsoft.com/en-us/azure/storsimple/storsimple-ova-overview" target="_blank" rel="noopener noreferrer">Source</a>):

<div class="table-responsive">
  <table class="table table-striped table-bordered table-hover">
    <tr>
      <td>
        Feature
      </td>
      
      <td>
        StorSimple Virtual Array
      </td>
    </tr>
    
    <tr>
      <td>
        Installation requirements
      </td>
      
      <td>
        Uses virtualization infrastructure (Hyper-V or VMware)
      </td>
    </tr>
    
    <tr>
      <td>
        Availability
      </td>
      
      <td>
        Single node
      </td>
    </tr>
    
    <tr>
      <td>
        Total capacity (including cloud)
      </td>
      
      <td>
        Up to 64 TB usable capacity per virtual array
      </td>
    </tr>
    
    <tr>
      <td>
        Local capacity
      </td>
      
      <td>
        390 GB to 6.4 TB usable capacity per virtual array (need to provision 500 GB to 8 TB of disk space)
      </td>
    </tr>
    
    <tr>
      <td>
        Native protocols
      </td>
      
      <td>
        iSCSI or SMB
      </td>
    </tr>
    
    <tr>
      <td>
        Recovery time objective (RTO)
      </td>
      
      <td>
        iSCSI: less than 2 minutes regardless of size
      </td>
    </tr>
    
    <tr>
      <td>
        Recovery point objective (RPO)
      </td>
      
      <td>
        Daily backups and on-demand backups
      </td>
    </tr>
    
    <tr>
      <td>
        Storage tiering
      </td>
      
      <td>
        Uses heat mapping to determine what data should be tiered in or out
      </td>
    </tr>
    
    <tr>
      <td>
        Support
      </td>
      
      <td>
        Virtualization infrastructure supported by the supplier
      </td>
    </tr>
    
    <tr>
      <td>
        Performance
      </td>
      
      <td>
        Varies depending on underlying infrastructure
      </td>
    </tr>
    
    <tr>
      <td>
        Data mobility
      </td>
      
      <td>
        Can restore to the same device or do item-level recovery (file server)
      </td>
    </tr>
    
    <tr>
      <td>
        Storage tiers
      </td>
      
      <td>
        Local hypervisor storage and cloud
      </td>
    </tr>
    
    <tr>
      <td>
        Share size
      </td>
      
      <td>
        Tiered: up to 20 TB; locally pinned: up to 2 TB
      </td>
    </tr>
    
    <tr>
      <td>
        Volume size
      </td>
      
      <td>
        Tiered: 500 GB to 5 TB; locally pinned: 50 GB to 200 GB, maximum local reservation for tiered volumes is 200 GB.
      </td>
    </tr>
    
    <tr>
      <td>
        Snapshots
      </td>
      
      <td>
        Crash consistent
      </td>
    </tr>
    
    <tr>
      <td>
        Item-level recovery
      </td>
      
      <td>
        Yes; users can restore from shares
      </td>
    </tr>
  </table>
</div>

### Why use StorSimple

StorSimple can connect the users and servers to Azure storage in minutes, without making changes to applications. The following table show some benefits of StorSimple Virtual Array (<a href="https://docs.microsoft.com/en-us/azure/storsimple/storsimple-ova-overview#why-use-storsimple" target="_blank" rel="noopener noreferrer">Source</a>):

<div class="table-responsive">
  <table class="table table-striped table-bordered table-hover">
    <tr>
      <td>
        Feature
      </td>
      
      <td>
        Benefit
      </td>
    </tr>
    
    <tr>
      <td>
        Transparent integration
      </td>
      
      <td>
        The virtual array supports the iSCSI or the SMB protocol. The data movement between the local tier and the cloud tier is seamless and transparent to the user.
      </td>
    </tr>
    
    <tr>
      <td>
        Reduced storage costs
      </td>
      
      <td>
        With StorSimple, you provision sufficient local storage to meet current demands for the most used hot data. As storage needs grow, StorSimple tiers cold data into cost-effective cloud storage. The data is deduplicated and compressed before sending to the cloud to further reduce storage requirements and expense.
      </td>
    </tr>
    
    <tr>
      <td>
        Simplified storage management
      </td>
      
      <td>
        StorSimple provides centralized management in the cloud using StorSimple Device Manager to manage multiple devices.
      </td>
    </tr>
    
    <tr>
      <td>
        Improved disaster recovery and compliance
      </td>
      
      <td>
        StorSimple facilitates faster disaster recovery by restoring the metadata immediately and restoring the data as needed. This means normal operations can continue with minimal disruption.
      </td>
    </tr>
    
    <tr>
      <td>
        Data mobility
      </td>
      
      <td>
        Data tiered to the cloud can be accessed from other sites for recovery and migration purposes. Note that you can restore data only to the original virtual array. However, you use disaster recovery features to restore the entire virtual array to another virtual array.
      </td>
    </tr>
  </table>
</div>

For more information see the <a href="https://docs.microsoft.com/en-us/azure/storsimple/storsimple-ova-overview" target="_blank" rel="noopener noreferrer">documentation</a>.

## Conclusion

In this post, I talked about storage pools on VMs and how virtual disks are created. Then I talked about enabling geo-replication and disk caching. Next, I showed how to create a file share and how to interact with it using your VM, PowerShell or C# code. After the file share, I explained how to set up disk encryption for your Windows and Linux VMs. The last section talks about what StorSimple is and what benefits it can bring.

For more information about the 70-532 exam get the <a href="http://amzn.to/2EWNWMF" target="_blank" rel="noopener noreferrer">Exam Ref book from Microsoft</a> and continue reading my blog posts. I am covering all topics needed to pass the exam. You can find an overview of all posts related to the 70-532 exam <a href="https://www.programmingwithwolfgang.com/prepared-for-the-70-532-exam/" target="_blank" rel="noopener noreferrer">here</a>.