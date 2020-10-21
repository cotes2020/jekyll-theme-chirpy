---
title: Implement Azure Storage blobs and Azure files
date: 2018-04-08T11:17:44+02:00
author: Wolfgang Ofner
categories: [Cloud]
tags: [70-532, Azure, Certification, Exam, learning]
---
Azure provides several methods of storing files, including Azure Storage blobs and Azure Files. In this post, I want to talk about the differences between them and how to use them.

## Azure Storage blobs

Azure Storage blobs should be used when you have files that you are storing using a custom application. Microsoft provides client libraries and REST interfaces for the Azure Storage blobs with which you can store and access data at a massive scale in block blobs

### Create a blob storage account

To create a blob storage account, follow these steps:

  1. In the Azure Portal, click on +Create a resource then on Storage and then select Storage account &#8211; blob, file, table, queue.
  2. On the Create storage account blade provide a name, location, subscription, and resource group. Optionally you can choose between standard (HDD) and premium (SSD) performance and enforce HTTPS by moving the slider to Enabled under the Secure transfer required attribute.
  3. Click Create.

<div id="attachment_1064" style="width: 710px" class="wp-caption aligncenter">
  <a href="/wp-content/uploads/2018/04/Create-a-new-Azure-Storage-account.jpg"><img aria-describedby="caption-attachment-1064" loading="lazy" class="wp-image-1064" src="/wp-content/uploads/2018/04/Create-a-new-Azure-Storage-account.jpg" alt="Create a new Azure Storage blobs account" width="700" height="697" /></a>
  
  <p id="caption-attachment-1064" class="wp-caption-text">
    Create a new Azure Storage account
  </p>
</div>

Your data is always replicated three times within the same data center to ensure data security.

### Read and change data

The code of the following demo can be downloaded from <a href="https://github.com/WolfgangOfner/Azure-StorageAccount" target="_blank" rel="noopener">GitHub</a> You can read and change data by using the Azure SDK for .Net, following these steps:

<li style="list-style-type: none;">
  <ol>
    <li>
      Make sure you have the Azure SDK installed.
    </li>
    <li>
      Create a new C# console application and install the WindowsAzure.Storage NuGet Package.
    </li>
    <li>
      Connect to your storage account in your application using the following code:
    </li>
  </ol>
</li>

<div id="attachment_1151" style="width: 721px" class="wp-caption aligncenter">
  <a href="/wp-content/uploads/2018/04/Connect-to-your-storage-account.jpg"><img aria-describedby="caption-attachment-1151" loading="lazy" class="size-full wp-image-1151" src="/wp-content/uploads/2018/04/Connect-to-your-storage-account.jpg" alt="Connect to your storage account" width="711" height="55" /></a>
  
  <p id="caption-attachment-1151" class="wp-caption-text">
    Connect to your storage account
  </p>
</div>

Replace the placeholder for storage account name and storage key with your own. You can find them in the Azure Portal under Access keys in your storage account.

<div id="attachment_1065" style="width: 710px" class="wp-caption aligncenter">
  <a href="/wp-content/uploads/2018/04/Finding-the-Storage-account-name-storage-key-and-connection-string.jpg"><img aria-describedby="caption-attachment-1065" loading="lazy" class="wp-image-1065" src="/wp-content/uploads/2018/04/Finding-the-Storage-account-name-storage-key-and-connection-string.jpg" alt="Finding the Storage account name, storage key and connection string" width="700" height="203" /></a>
  
  <p id="caption-attachment-1065" class="wp-caption-text">
    Finding the Storage account name, storage key, and connection string
  </p>
</div>

<ol start="4">
  <li>
    Create a container with the following code
  </li>
</ol>

<div id="attachment_1152" style="width: 507px" class="wp-caption aligncenter">
  <a href="/wp-content/uploads/2018/04/Create-the-container-myblockcontainer-if-it-does-not-exist.jpg"><img aria-describedby="caption-attachment-1152" loading="lazy" class="size-full wp-image-1152" src="/wp-content/uploads/2018/04/Create-the-container-myblockcontainer-if-it-does-not-exist.jpg" alt="Create the container myblockcontainer if it does not exist" width="497" height="186" /></a>
  
  <p id="caption-attachment-1152" class="wp-caption-text">
    Create the container myblockcontainer if it does not exist
  </p>
</div>

Azure Storage blobs are organized in containers. Each storage account can have an unlimited amount of containers. Note that a container can&#8217;t have uppercase letters.

<ol start="5">
  <li>
    Next, set the path to the file you want to upload and upload the file
  </li>
</ol>

<div id="attachment_1153" style="width: 589px" class="wp-caption aligncenter">
  <a href="/wp-content/uploads/2018/04/Upload-a-file-to-the-container.jpg"><img aria-describedby="caption-attachment-1153" loading="lazy" class="size-full wp-image-1153" src="/wp-content/uploads/2018/04/Upload-a-file-to-the-container.jpg" alt="Upload a file to the container" width="579" height="153" /></a>
  
  <p id="caption-attachment-1153" class="wp-caption-text">
    Upload a file to the container
  </p>
</div>

<ol start="6">
  <li>
    After the file is uploaded you can find it in the Azure Portal: <ul>
      <li>
        Go to your storage account and click on Blobs on the Overview blade
      </li>
      <li>
        On the Blob service blade, click on your container. There you can see all blobs inside this container.
      </li>
    </ul>
  </li>
</ol>

<div id="attachment_1066" style="width: 710px" class="wp-caption aligncenter">
  <a href="/wp-content/uploads/2018/04/The-previously-uploaded-blob.jpg"><img aria-describedby="caption-attachment-1066" loading="lazy" class="wp-image-1066" src="/wp-content/uploads/2018/04/The-previously-uploaded-blob.jpg" alt="The previously uploaded blob" width="700" height="190" /></a>
  
  <p id="caption-attachment-1066" class="wp-caption-text">
    The previously uploaded blob
  </p>
</div>

## Set metadata on a container

Metadata can be used to determine when files have been updated or to set the content types for web artifacts. There are two forms of metadata:

  1. **System properties metadata** give you information about access, file types and more.
  2. **User-defined metadata** is a key-value pair that is specified for your application. It can be the time when a file was processed or a note of the source.

A container has only read-only system properties, while blobs have both read-only and read-write properties.

### Setting user-defined metadata

To set user-defined metadata, expand the code from before with these two lines:

<div id="attachment_1069" style="width: 456px" class="wp-caption aligncenter">
  <a href="/wp-content/uploads/2018/04/Set-user-defined-metadata.jpg"><img aria-describedby="caption-attachment-1069" loading="lazy" class="size-full wp-image-1069" src="/wp-content/uploads/2018/04/Set-user-defined-metadata.jpg" alt="Set user-defined metadata" width="446" height="92" /></a>
  
  <p id="caption-attachment-1069" class="wp-caption-text">
    Set user-defined metadata
  </p>
</div>

### Read user-defined metadata

To read the previously added user-defined metadata, add this code:

<div id="attachment_1070" style="width: 541px" class="wp-caption aligncenter">
  <a href="/wp-content/uploads/2018/04/Read-user-defined-metadata.jpg"><img aria-describedby="caption-attachment-1070" loading="lazy" class="size-full wp-image-1070" src="/wp-content/uploads/2018/04/Read-user-defined-metadata.jpg" alt="Read user-defined metadata" width="531" height="165" /></a>
  
  <p id="caption-attachment-1070" class="wp-caption-text">
    Read user-defined metadata
  </p>
</div>

If the metadata key does not exist, an exception is thrown.

### Read system properties

To read system properties, add this code:

<div id="attachment_1072" style="width: 561px" class="wp-caption aligncenter">
  <a href="/wp-content/uploads/2018/04/Read-system-metadata.jpg"><img aria-describedby="caption-attachment-1072" loading="lazy" class="size-full wp-image-1072" src="/wp-content/uploads/2018/04/Read-system-metadata.jpg" alt="Read system metadata" width="551" height="116" /></a>
  
  <p id="caption-attachment-1072" class="wp-caption-text">
    Read system metadata
  </p>
</div>

There are various system metadata. Use the IntelliSense to see all available ones.

## Store data using block and page blobs

Azure Storage Blobs have three different types of blobs:

  1. **Block blobs** are used to upload large files. A blob is divided up into blocks which allow for easy updating large files since you can insert, replace or delete an existing block. After a block is updated, the list of blocks needs to be committed for the file to actually record the update.
  2. **Page blobs** are comprised of 512- byte pages that are optimized for random read and write operations. Page blobs are useful for VHDs and other files which have frequent, random access.
  3. **Append blobs** are optimized for append operations like logging and streaming data.

### Write data to a page blob

First, you have to create a page blob:

<div id="attachment_1073" style="width: 637px" class="wp-caption aligncenter">
  <a href="/wp-content/uploads/2018/04/Create-a-page-blob.jpg"><img aria-describedby="caption-attachment-1073" loading="lazy" class="size-full wp-image-1073" src="/wp-content/uploads/2018/04/Create-a-page-blob.jpg" alt="Create a page blob" width="627" height="261" /></a>
  
  <p id="caption-attachment-1073" class="wp-caption-text">
    Create a page blob
  </p>
</div>

After the page blob is created, you can write data to it:

<div id="attachment_1074" style="width: 633px" class="wp-caption aligncenter">
  <a href="/wp-content/uploads/2018/04/Wirte-to-a-page-blob.jpg"><img aria-describedby="caption-attachment-1074" loading="lazy" class="wp-image-1074 size-full" src="/wp-content/uploads/2018/04/Wirte-to-a-page-blob.jpg" alt="Write to a page blob" width="623" height="172" /></a>
  
  <p id="caption-attachment-1074" class="wp-caption-text">
    Write to a page blob
  </p>
</div>

After some data are added to the blob, you can read it:

<div id="attachment_1075" style="width: 481px" class="wp-caption aligncenter">
  <a href="/wp-content/uploads/2018/04/Read-a-page-blob.jpg"><img aria-describedby="caption-attachment-1075" loading="lazy" class="size-full wp-image-1075" src="/wp-content/uploads/2018/04/Read-a-page-blob.jpg" alt="Read a page blob" width="471" height="134" /></a>
  
  <p id="caption-attachment-1075" class="wp-caption-text">
    Read a page blob
  </p>
</div>

## Stream data using Azure Storage blobs

Instead of downloading a whole blob, you can download it to a stream using the DownloadToStream() API. The advantage of this approach is that it avoids loading the whole blob into the memory.

## Access Azure Storage blobs securely

Azure Storage supports both HTTP and HTTPS. You should always use HTTPS though. You can authenticate in three different ways to your storage account:

  1. Shared Key: The shared key is constructed from a set of fields from the request. It is computed with the SHA-256 algorithm and encoded in Base64
  2. Shared Key Lite: The shared key lite is similar to the shared key but it is compatible with previous versions of Azure Storage.
  3. Shared Access Signature: The shared access signature grants restricted access rights to containers and blobs. Users with a shared access signature have only specific permissions to a resource for a specified amount of time.

Each call to interact with blob storage will be secured, as shown in the following code:

<div id="attachment_1076" style="width: 710px" class="wp-caption aligncenter">
  <a href="/wp-content/uploads/2018/04/Secure-access-to-your-Storage-Account.jpg"><img aria-describedby="caption-attachment-1076" loading="lazy" class="wp-image-1076" src="/wp-content/uploads/2018/04/Secure-access-to-your-Storage-Account.jpg" alt="Secure access to your Storage Account" width="700" height="52" /></a>
  
  <p id="caption-attachment-1076" class="wp-caption-text">
    Secure access to your Storage Account
  </p>
</div>

## Implement Async blob copy

Sometimes it is necessary to copy blobs between storage account, for example, before an update or when migrating files from one account to another.

The type of the blob can&#8217;t be changed during the async copy operation. Any files with the same name on the destination account will be overwritten.

When you call the API and get a success message, this means the copy operation has been successfully scheduled. The success message will be returned after checking the permissions on the source and destination account.

The copy process can be performed with the Shared Access Signature method,

## Configure a Content Delivery Network with Azure Storage Blobs

A Content Delivery Network (CDN) is used to cache static files to different parts of the world. A CDN would be a perfect solution for serving files close to the users. There are way more CDN nodes than data centers, therefore the files in the CDN can be better distributed in an area and reduce the latency for your customers. As a result, files are loaded faster and the user experience is increased.

The CDN cache is perfect for CSS and JavaScript files, documents, images and HTML pages.

Once CDN is enabled and files are hosted in an Azure Storage Account, a configured CDN will store and replicate those files without any management.

To enable CDN for the storage account, follow these steps:

  1. Open your storage account and select Azure CDN under the Blob Service menu.
  2. On the Azure CDN blade, create a new CDN by filling out the form. The difference between the Premium and Standard Pricing tier is that the Premium offers advanced real-time analytics.

<div id="attachment_1077" style="width: 710px" class="wp-caption aligncenter">
  <a href="/wp-content/uploads/2018/04/Create-a-new-CDN.jpg"><img aria-describedby="caption-attachment-1077" loading="lazy" class="wp-image-1077" src="/wp-content/uploads/2018/04/Create-a-new-CDN.jpg" alt="Create a new CDN" width="700" height="279" /></a>
  
  <p id="caption-attachment-1077" class="wp-caption-text">
    Create a new CDN
  </p>
</div>

<span class="fontstyle0">If a file needs to be replaced or removed, you can delete it from the Azure Storage blob container. Remember that the file is being cached in the CDN. It will be removed or updated when the Time-to-Live (TTL) expires. If no cache expiry period is specified, it will be cached in the CDN for seven days. You set the TTL is the web application by using the clientCache element in the web.confg file. Remember when you place that in the web.confg file it affects all folders and subfolders for that application.</span>

## Design blob hierarchies

Containers are flat which means that a container can&#8217;t have a child container inside it. A hierarchy can be created by naming the files similar to a folder structure. A solution would be to prefix all Azure Storage blobs with pictures with pictures/, for example, the file would be named pictures/house.jpg or pictures/tree.jpg. The path to these images would be:

  * https://wolfgangstorageaccount.blob.core.windows.net/myblob/pictures/house.jpg
  * https://wolfgangstorageaccount.blob.core.windows.net/myblob/pictures/tree.jpg

Using the prefix simulates having folders.

## Configure custom domains

The default endpoint for Azure Storage blobs is: StorageAccountName.blob.core.windows.net. Using the default domain can negatively affect SEO. Additionally, it tells that you are hosting your files on Azure. To hide this, you can configure your storage account to use a custom domain. To do that, follow these steps:

  1. Go to your storage account and click on Custom Domain under the Blob Service menu.
  2. Check the Use indirect CNAME validation checkbox. By checking the checkbox, no downtime will incur for your application.
  3. Log on to your DNS provider and add a CName record with the subdomain alias that includes the Asverify domain.
  4. On the Custom domain blade, enter the name of your custom domain, but without the Asverify.
  5. Click Save.
  6. Create another CNAME record that maps your subdomain to your blob service endpoint on your DNS provider&#8217;s website
  7. Now you can delete the Asverify CNAME since it has been verified by Azure already.

<div id="attachment_1078" style="width: 561px" class="wp-caption aligncenter">
  <a href="/wp-content/uploads/2018/04/Add-a-custom-domain-to-your-storage-account.jpg"><img aria-describedby="caption-attachment-1078" loading="lazy" class="size-full wp-image-1078" src="/wp-content/uploads/2018/04/Add-a-custom-domain-to-your-storage-account.jpg" alt="Add a custom domain to your storage account" width="551" height="372" /></a>
  
  <p id="caption-attachment-1078" class="wp-caption-text">
    Add a custom domain to your storage account
  </p>
</div>

## Scale blob storage

Blob storage can be scaled both in terms of storage capacity and performance. Each Azure subscription can have up to 200 storage account, with 500 TB if capacity each. This means that each Azure subscription can have up to 100 PB of data.

A block blob can have 50,000 100 MB blocks with a total size of 4.75 TB. An append blob has a maximum size of 195 GB and a page blob has a maximum size of 8 TB.

In order to scale the performance, there are several features available. For example, you could enable geo-caching for the Azure CDN or implement read access geo-redundant storage and copy your data to multiple data center in different locations.

Azure Storage only charges for disk space used and network bandwidth.

<span class="fontstyle0">Many small files will perform better in Azure Storage than one large file. Blobs use containers for logical grouping, but each blob can be retrieved by diﬀerent compute resources, even if they are in the same container.</span>

## Azure files

Azure files are useful for VMs and cloud services as mounted share. Check out my post <a href="https://www.programmingwithwolfgang.com/design-implement-arm-vm-azure-storage/#CreateAzureStorage" target="_blank" rel="noopener">&#8220;Design and Implement ARM VM Azure Storage&#8221;</a> for an instruction on how to create an Azure file share.

## Implement blob leasing

You can create a lock on a blob for write and delete operations. This lick can be between 15 and 60 seconds or it can be infinite. To write to a blob with an active lease, the client must include the active lease ID with the request.

<span class="fontstyle0">When a client requests a lease, a lease ID is returned. The client can then use this lease ID to renew, change, or release the lease. When the lease is active, the lease ID must be included to write to the blob, set any metadata, add to the blob (through append), copy the blob, or delete the blob. </span>

Use the following code to get the lease ID:

<div id="attachment_1081" style="width: 469px" class="wp-caption aligncenter">
  <a href="/wp-content/uploads/2018/04/Get-the-lease-ID-of-your-block-blob.jpg"><img aria-describedby="caption-attachment-1081" loading="lazy" class="size-full wp-image-1081" src="/wp-content/uploads/2018/04/Get-the-lease-ID-of-your-block-blob.jpg" alt="Get the lease ID of your block blob" width="459" height="141" /></a>
  
  <p id="caption-attachment-1081" class="wp-caption-text">
    Get the lease ID of your block blob
  </p>
</div>

## <span class="fontstyle0">Create connections to files from on-premises or cloud-based Windows or, Linux machines</span>

<span class="fontstyle0">Azure Files can be used to replace on-premise file servers or NAS devices.</span> You can find an instruction on how to connect to Azure Files in my post <a href="https://www.programmingwithwolfgang.com/design-implement-arm-vm-azure-storage/#CreateAzureStorage" target="_blank" rel="noopener">&#8220;Design and Implement ARM VM Azure Storage&#8221;</a>. The instructions are for an Azure VM but you can do also do it with your on-premise machine.

## Shard large datasets

You can use containers to group related blobs that have the same security requirements. The partition key of a blob is the account name + container name + blob name. A single blob can only be served by a single server. If sharding is needed, you need to create multiple blobs.

## Implement Azure File Sync

Azure File Sync (AFS) helps you to automatically upload files from a Windows Server 2012 or 2016 server to the cloud.

Azure File Sync helps organizations to:

  * Cache data in multiple locations for fast, local performance
  * Centralize file services in Azure storage
  * Eliminate local backup

To enable AFS, follow these steps:

  1. Create a Windows server 2012 or 2016 file server and a storage account.
  2. In the Azure portal, click on +Create a resource and search for Azure File Sync. Click on it and click on Create.
  3. Provide a name, subscription, resource group and location and click Create.

<div id="attachment_1188" style="width: 375px" class="wp-caption aligncenter">
  <a href="/wp-content/uploads/2018/04/Deploy-Storage-Sync.jpg"><img aria-describedby="caption-attachment-1188" loading="lazy" class="size-full wp-image-1188" src="/wp-content/uploads/2018/04/Deploy-Storage-Sync.jpg" alt="Deploy Storage Sync" width="365" height="358" /></a>
  
  <p id="caption-attachment-1188" class="wp-caption-text">
    Deploy Storage Sync
  </p>
</div>

<ol start="4">
  <li>
    In your storage account, create a new file share.
  </li>
</ol>

<div id="attachment_1189" style="width: 710px" class="wp-caption aligncenter">
  <a href="/wp-content/uploads/2018/04/Create-a-file-share-in-the-storage-account.jpg"><img aria-describedby="caption-attachment-1189" loading="lazy" class="wp-image-1189" src="/wp-content/uploads/2018/04/Create-a-file-share-in-the-storage-account.jpg" alt="Create a file share in the storage account" width="700" height="462" /></a>
  
  <p id="caption-attachment-1189" class="wp-caption-text">
    Create a file share in the storage account
  </p>
</div>

<ol start="5">
  <li>
    By now the Storage Sync Service should be deployed. Open it and click on +Sync group on the Overview blade.
  </li>
  <li>
    On the Sync group blade, enter a name and select a subscription, a storage account, and a file share and click Create.
  </li>
</ol>

<div id="attachment_1190" style="width: 597px" class="wp-caption aligncenter">
  <a href="/wp-content/uploads/2018/04/Create-a-Sync-group.jpg"><img aria-describedby="caption-attachment-1190" loading="lazy" class="size-full wp-image-1190" src="/wp-content/uploads/2018/04/Create-a-Sync-group.jpg" alt="Create a Sync group" width="587" height="484" /></a>
  
  <p id="caption-attachment-1190" class="wp-caption-text">
    Create a Sync group
  </p>
</div>

### Set up the file server

<ol start="7">
  <li>
    Next, you have to register your server. To do that, connect to your previously created Windows server and download the Azure Storage Sync agent from <a href="https://www.microsoft.com/en-us/download/details.aspx?id=55988&irgwc=1&OCID=AID681541_aff_7593_1211691&tduid=(ir_QTvRPy13FRXOxFBQaey5FwDGUkj3vsTi2XGDXg0)(7593)(1211691)(TnL5HPStwNw-2IiLjGKY4WXhGNZVz0Xy7A)()&irclickid=QTvRPy13FRXOxFBQaey5FwDGUkj3vsTi2XGDXg0" target="_blank" rel="noopener">here</a>.
  </li>
  <li>
    You might have to disable enhanced security.
  </li>
</ol>

<div id="attachment_1191" style="width: 425px" class="wp-caption aligncenter">
  <a href="/wp-content/uploads/2018/04/Disable-enhanced-security.jpg"><img aria-describedby="caption-attachment-1191" loading="lazy" class="size-full wp-image-1191" src="/wp-content/uploads/2018/04/Disable-enhanced-security.jpg" alt="Disable enhanced security" width="415" height="448" /></a>
  
  <p id="caption-attachment-1191" class="wp-caption-text">
    Disable enhanced security
  </p>
</div>

<ol start="9">
  <li>
    After the installation is finished, start the Server Registration if it doesn&#8217;t start automatically. The default path is C:\Program Files\Azure\StorageSyncAgent.
  </li>
  <li>
    If you see a warning that the pre-requisites are missing, you have to install the Azure PowerShell module.
  </li>
</ol>

<div id="attachment_1192" style="width: 710px" class="wp-caption aligncenter">
  <a href="/wp-content/uploads/2018/04/Azure-File-Sync-Azure-PowerShell-Module-missing.jpg"><img aria-describedby="caption-attachment-1192" loading="lazy" class="wp-image-1192" src="/wp-content/uploads/2018/04/Azure-File-Sync-Azure-PowerShell-Module-missing.jpg" alt="Azure File Sync Azure PowerShell Module missing" width="700" height="438" /></a>
  
  <p id="caption-attachment-1192" class="wp-caption-text">
    Azure File Sync Azure PowerShell Module missing
  </p>
</div>

<ol start="11">
  <li>
    To install the Azure PowerShell module, open PowerShell and enter <span class="hljs-pscommand">Install-Module</span><span class="hljs-parameter"> -Name</span> AzureRM<span class="hljs-parameter"> -AllowClobber. For more information see the <a href="https://docs.microsoft.com/en-us/powershell/azure/install-azurerm-ps?view=azurermps-5.7.0" target="_blank" rel="noopener">documentation</a>.</span>
  </li>
</ol>

<div id="attachment_1193" style="width: 710px" class="wp-caption aligncenter">
  <a href="/wp-content/uploads/2018/04/Install-the-Azure-PowerShell-Module.jpg"><img aria-describedby="caption-attachment-1193" loading="lazy" class="wp-image-1193" src="/wp-content/uploads/2018/04/Install-the-Azure-PowerShell-Module.jpg" alt="Install the Azure PowerShell Module" width="700" height="392" /></a>
  
  <p id="caption-attachment-1193" class="wp-caption-text">
    Install the Azure PowerShell module
  </p>
</div>

<ol start="12">
  <li>
    After the Azure PowerShell module is installed, you can sign in and register your server.
  </li>
</ol>

<div id="attachment_1194" style="width: 710px" class="wp-caption aligncenter">
  <a href="/wp-content/uploads/2018/04/Sign-in-and-register-this-server.jpg"><img aria-describedby="caption-attachment-1194" loading="lazy" class="wp-image-1194" src="/wp-content/uploads/2018/04/Sign-in-and-register-this-server.jpg" alt="Sign in and register this server" width="700" height="438" /></a>
  
  <p id="caption-attachment-1194" class="wp-caption-text">
    Sign in and register this server
  </p>
</div>

### Configure the Sync group

<ol start="13">
  <li>
    After the server is set up, go back to your Storage Sync Service and open the previously created Sync group on the Overview blade.
  </li>
  <li>
    On the Sync group blade, click on Add server endpoint and then select the previously registered server. Next, select a path which should be synchronized and then click Create.
  </li>
</ol>

<div id="attachment_1195" style="width: 710px" class="wp-caption aligncenter">
  <a href="/wp-content/uploads/2018/04/Register-a-server-endpoint.jpg"><img aria-describedby="caption-attachment-1195" loading="lazy" class="wp-image-1195" src="/wp-content/uploads/2018/04/Register-a-server-endpoint.jpg" alt="Register a server endpoint" width="700" height="210" /></a>
  
  <p id="caption-attachment-1195" class="wp-caption-text">
    Register a server endpoint
  </p>
</div>

### Test the Azure File Sync

<ol start="15">
  <li>
    To test the file sync, copy some files into the sync folder.
  </li>
</ol>

<div id="attachment_1196" style="width: 710px" class="wp-caption aligncenter">
  <a href="/wp-content/uploads/2018/04/Copy-files-into-the-sync-folder.jpg"><img aria-describedby="caption-attachment-1196" loading="lazy" class="size-full wp-image-1196" src="/wp-content/uploads/2018/04/Copy-files-into-the-sync-folder.jpg" alt="Copy files into the sync folder" width="700" height="146" /></a>
  
  <p id="caption-attachment-1196" class="wp-caption-text">
    Copy files into the sync folder
  </p>
</div>

<ol start="16">
  <li>
    To check if the files were synchronized, go to your file share in the storage account. There should be the files from your sync folder.
  </li>
</ol>

<div id="attachment_1197" style="width: 710px" class="wp-caption aligncenter">
  <a href="/wp-content/uploads/2018/04/The-synchronize-process-was-successful.jpg"><img aria-describedby="caption-attachment-1197" loading="lazy" class="wp-image-1197" src="/wp-content/uploads/2018/04/The-synchronize-process-was-successful.jpg" alt="The synchronize process was successful" width="700" height="266" /></a>
  
  <p id="caption-attachment-1197" class="wp-caption-text">
    The synchronize process was successful
  </p>
</div>

## Conclusion

In this post, I talked about the different types of Azure Storage blobs and when they should be used. Furthermore, I showed how to implement Azure files as a file share for your cloud-based and on-premise machines.

The last section was about implementing Azure File Sync to synchronize files from a server into the cloud.

You can find the example code on <a href="https://github.com/WolfgangOfner/Azure-StorageAccount" target="_blank" rel="noopener">GitHub</a>. Don&#8217;t forget to replace the account name and access key placeholder with the name of your storage account and it&#8217;s access key.

For more information about the 70-532 exam get the <a href="http://amzn.to/2EWNWMF" target="_blank" rel="noopener">Exam Ref book from Microsoft</a> and continue reading my blog posts. I am covering all topics needed to pass the exam. You can find an overview of all posts related to the 70-532 exam <a href="https://www.programmingwithwolfgang.com/prepared-for-the-70-532-exam/" target="_blank" rel="noopener">here</a>.