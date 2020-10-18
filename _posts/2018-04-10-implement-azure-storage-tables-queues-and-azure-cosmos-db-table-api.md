---
title: Implement Azure Storage Tables, Queues, and Azure Cosmos DB Table API
date: 2018-04-10T12:40:08+02:00
author: Wolfgang Ofner
categories: [Cloud]
tags: [70-532, Azure, Certification, Exam, learning]
---
Azure Storage Tables can store tabular data at petabyte scale. Azure Queue storage is used to provide messaging between application components, as they can be de-coupled and scaled individually.

## Azure Storage Tables

Azure Tables are a key-value database solution with rows and columns. Tables store data as a collection of entities where each entity has a property. Azure Tables can have up to 255 properties (columns in relational databases). The maximum entity size (row size in a relational database) is 1 MB.

Azure Tables organize data based on table name. For example, all customers should be stored in the Customers table whereas all products should be stored in the Products table.

The tables store the entities based on a partition key and a row key. All entities stored with the same partition key property are grouped into the same partition and are served by the same partition server. The developer has to choose a good partition key. Having a few partitions will improve scalability, as it will increase the number of partition servers handling a request.

### Azure Storage Tables vs. Azure SQL Database

Microsoft Azure Tables does not enforce any schema for tables. It is the developer&#8217;s responsibility to enforce the schema on the client side. Azure Tables do not have stored procedures, triggers, indexes, constraints, default values and many more SQL Database features. The big advantage of Azure Tables is that you are not charged for compute resources for inserting, updating or retrieving data. You are only charged for the total storage you use. which makes it extremely affordable for large datasets. Azure Tables also scale very well without sacrificing performance

### Using basic CRUD operations

This section will explain how to access table storage programmatically using C#. You can find the code for the following demo on <a href="https://github.com/WolfgangOfner/Azure-TableStorage" target="_blank" rel="noopener">GitHub</a>.

#### Creating a table

  1. Create a new C# console application.
  2. Add the following code to your app.config file:

<div id="attachment_1135" style="width: 568px" class="wp-caption aligncenter">
  <a href="https://www.programmingwithwolfgang.com/wp-content/uploads/2018/04/Setup-the-connection-string-to-your-storage-account.jpg"><img aria-describedby="caption-attachment-1135" loading="lazy" class="size-full wp-image-1135" src="https://www.programmingwithwolfgang.com/wp-content/uploads/2018/04/Setup-the-connection-string-to-your-storage-account.jpg" alt="Setup the connection string to your storage account" width="558" height="87" srcset="https://www.programmingwithwolfgang.com/wp-content/uploads/2018/04/Setup-the-connection-string-to-your-storage-account.jpg 558w, https://www.programmingwithwolfgang.com/wp-content/uploads/2018/04/Setup-the-connection-string-to-your-storage-account-300x47.jpg 300w" sizes="(max-width: 558px) 100vw, 558px" /></a>
  
  <p id="caption-attachment-1135" class="wp-caption-text">
    Setup the connection string to your storage account
  </p>
</div>

Replace the placeholder with your storage account name and storage account key

<ol start="3">
  <li>
    Install the WindowsAzure.Storage NuGet package.
  </li>
  <li>
    In the Main method, retrieve the connection string:
  </li>
</ol>

<div id="attachment_1136" style="width: 710px" class="wp-caption aligncenter">
  <a href="https://www.programmingwithwolfgang.com/wp-content/uploads/2018/04/Retrieve-the-connection-string.jpg"><img aria-describedby="caption-attachment-1136" loading="lazy" class="wp-image-1136" src="https://www.programmingwithwolfgang.com/wp-content/uploads/2018/04/Retrieve-the-connection-string.jpg" alt="Retrieve the connection string" width="700" height="20" srcset="https://www.programmingwithwolfgang.com/wp-content/uploads/2018/04/Retrieve-the-connection-string.jpg 767w, https://www.programmingwithwolfgang.com/wp-content/uploads/2018/04/Retrieve-the-connection-string-300x9.jpg 300w" sizes="(max-width: 700px) 100vw, 700px" /></a>
  
  <p id="caption-attachment-1136" class="wp-caption-text">
    Retrieve the connection string
  </p>
</div>

<ol start="5">
  <li>
    With the following code can you create a table if it doesn&#8217;t exist:
  </li>
</ol>

<div id="attachment_1138" style="width: 433px" class="wp-caption aligncenter">
  <a href="https://www.programmingwithwolfgang.com/wp-content/uploads/2018/04/Create-the-orders-table-if-it-does-not-exist.jpg"><img aria-describedby="caption-attachment-1138" loading="lazy" class="size-full wp-image-1138" src="https://www.programmingwithwolfgang.com/wp-content/uploads/2018/04/Create-the-orders-table-if-it-does-not-exist.jpg" alt="Create the orders table if it does not exist" width="423" height="73" srcset="https://www.programmingwithwolfgang.com/wp-content/uploads/2018/04/Create-the-orders-table-if-it-does-not-exist.jpg 423w, https://www.programmingwithwolfgang.com/wp-content/uploads/2018/04/Create-the-orders-table-if-it-does-not-exist-300x52.jpg 300w" sizes="(max-width: 423px) 100vw, 423px" /></a>
  
  <p id="caption-attachment-1138" class="wp-caption-text">
    Create the orders table if it does not exist
  </p>
</div>

#### Inserting records

The following useful properties for adding entries to a table are provided by the Storage Client Library:

<div class="table-responsive">
  <table class="table table-striped table-bordered table-hover">
    <tr>
      <td>
        Property
      </td>
      
      <td>
        Description
      </td>
    </tr>
    
    <tr>
      <td>
        Partition Key
      </td>
      
      <td>
        The partition key is used to partition data across storage infrastructure
      </td>
    </tr>
    
    <tr>
      <td>
        Row Key:
      </td>
      
      <td>
        The row key is a unique identifier in a partition
      </td>
    </tr>
    
    <tr>
      <td>
        Timestamp
      </td>
      
      <td>
        Contains the time of the last update
      </td>
    </tr>
    
    <tr>
      <td>
        ETag
      </td>
      
      <td>
        The ETag is used internally to provide optimistic concurrency
      </td>
    </tr>
  </table>
</div>

The combination of partition key and row key must be unique within the table. This combination is used for load balancing and scaling, as well as for querying and sorting entities.

To insert data into the previously created table, follow these steps:

  1. Add the following class:

<div id="attachment_1139" style="width: 454px" class="wp-caption aligncenter">
  <a href="https://www.programmingwithwolfgang.com/wp-content/uploads/2018/04/The-OrderEntity-class-will-be-used-to-add-orders-into-the-table.jpg"><img aria-describedby="caption-attachment-1139" loading="lazy" class="size-full wp-image-1139" src="https://www.programmingwithwolfgang.com/wp-content/uploads/2018/04/The-OrderEntity-class-will-be-used-to-add-orders-into-the-table.jpg" alt="The OrderEntity class will be used to add orders into the table" width="444" height="380" srcset="https://www.programmingwithwolfgang.com/wp-content/uploads/2018/04/The-OrderEntity-class-will-be-used-to-add-orders-into-the-table.jpg 444w, https://www.programmingwithwolfgang.com/wp-content/uploads/2018/04/The-OrderEntity-class-will-be-used-to-add-orders-into-the-table-300x257.jpg 300w" sizes="(max-width: 444px) 100vw, 444px" /></a>
  
  <p id="caption-attachment-1139" class="wp-caption-text">
    The OrderEntity class will be used to add orders into the table
  </p>
</div>

<ol start="2">
  <li>
    To add an order to the orders table, use the following code:
  </li>
</ol>

<div id="attachment_1140" style="width: 782px" class="wp-caption aligncenter">
  <a href="https://www.programmingwithwolfgang.com/wp-content/uploads/2018/04/Insert-a-single-record-into-the-table.jpg"><img aria-describedby="caption-attachment-1140" loading="lazy" class="size-full wp-image-1140" src="https://www.programmingwithwolfgang.com/wp-content/uploads/2018/04/Insert-a-single-record-into-the-table.jpg" alt="Insert a single record into the table" width="772" height="239" srcset="https://www.programmingwithwolfgang.com/wp-content/uploads/2018/04/Insert-a-single-record-into-the-table.jpg 772w, https://www.programmingwithwolfgang.com/wp-content/uploads/2018/04/Insert-a-single-record-into-the-table-300x93.jpg 300w, https://www.programmingwithwolfgang.com/wp-content/uploads/2018/04/Insert-a-single-record-into-the-table-768x238.jpg 768w" sizes="(max-width: 772px) 100vw, 772px" /></a>
  
  <p id="caption-attachment-1140" class="wp-caption-text">
    Insert a single record into the table
  </p>
</div>

#### Insert multiple records in a transaction

You can group inserts and other operations into a single batch transaction. You can have up to 100 entities in a batch but the batch size can&#8217;t be greater than 4 MB.

To insert multiple orders in one transaction, add this code to your application:

<div id="attachment_1141" style="width: 780px" class="wp-caption aligncenter">
  <a href="https://www.programmingwithwolfgang.com/wp-content/uploads/2018/04/Insert-multiple-records-into-the-table.jpg"><img aria-describedby="caption-attachment-1141" loading="lazy" class="size-full wp-image-1141" src="https://www.programmingwithwolfgang.com/wp-content/uploads/2018/04/Insert-multiple-records-into-the-table.jpg" alt="Insert multiple records into the table" width="770" height="506" srcset="https://www.programmingwithwolfgang.com/wp-content/uploads/2018/04/Insert-multiple-records-into-the-table.jpg 770w, https://www.programmingwithwolfgang.com/wp-content/uploads/2018/04/Insert-multiple-records-into-the-table-300x197.jpg 300w, https://www.programmingwithwolfgang.com/wp-content/uploads/2018/04/Insert-multiple-records-into-the-table-768x505.jpg 768w" sizes="(max-width: 770px) 100vw, 770px" /></a>
  
  <p id="caption-attachment-1141" class="wp-caption-text">
    Insert multiple records into the table
  </p>
</div>

#### Getting records in a partition

You can select all entities in a partition or a range of entities by partition and row key. Wherever possible, you should try to query with the partition key and row key. Querying entities by other properties does not work well because it launches a scan of the entire table.

Within a table, entities are ordered within the partition key. Within a partition, entities are ordered by the row key. The RowKey property is a string, therefore sorting is handled as a string sort. If you are using a date value for your row key, as I did in the previous example, use the order year month day, for example, 20181220.

The following code gets all records within a partition using the PartitionKey property:

<div id="attachment_1142" style="width: 710px" class="wp-caption aligncenter">
  <a href="https://www.programmingwithwolfgang.com/wp-content/uploads/2018/04/Retrieve-all-records-of-the-table-with-the-partition-key-Wolfgang.jpg"><img aria-describedby="caption-attachment-1142" loading="lazy" class="wp-image-1142" src="https://www.programmingwithwolfgang.com/wp-content/uploads/2018/04/Retrieve-all-records-of-the-table-with-the-partition-key-Wolfgang.jpg" alt="Retrieve all records of the table with the partition key Wolfgang" width="700" height="203" srcset="https://www.programmingwithwolfgang.com/wp-content/uploads/2018/04/Retrieve-all-records-of-the-table-with-the-partition-key-Wolfgang.jpg 770w, https://www.programmingwithwolfgang.com/wp-content/uploads/2018/04/Retrieve-all-records-of-the-table-with-the-partition-key-Wolfgang-300x87.jpg 300w, https://www.programmingwithwolfgang.com/wp-content/uploads/2018/04/Retrieve-all-records-of-the-table-with-the-partition-key-Wolfgang-768x222.jpg 768w" sizes="(max-width: 700px) 100vw, 700px" /></a>
  
  <p id="caption-attachment-1142" class="wp-caption-text">
    Retrieve all records of the table with the partition key Wolfgang
  </p>
</div>

#### Updating records

To update records, you can use the InsertOrReplace() method. It creates a records if it does not exist and updates an existing one, based on the partition key and row key. Use the following code to do that:

<div id="attachment_1143" style="width: 710px" class="wp-caption aligncenter">
  <a href="https://www.programmingwithwolfgang.com/wp-content/uploads/2018/04/Update-a-record-in-the-table.jpg"><img aria-describedby="caption-attachment-1143" loading="lazy" class="wp-image-1143" src="https://www.programmingwithwolfgang.com/wp-content/uploads/2018/04/Update-a-record-in-the-table.jpg" alt="Update a record in the table" width="700" height="209" srcset="https://www.programmingwithwolfgang.com/wp-content/uploads/2018/04/Update-a-record-in-the-table.jpg 766w, https://www.programmingwithwolfgang.com/wp-content/uploads/2018/04/Update-a-record-in-the-table-300x90.jpg 300w" sizes="(max-width: 700px) 100vw, 700px" /></a>
  
  <p id="caption-attachment-1143" class="wp-caption-text">
    Update a record in the table
  </p>
</div>

#### Deleting a record

To delete a record, you have to retrieve it first and then call the Delete() method. Use the following code to do that:

<div id="attachment_1144" style="width: 774px" class="wp-caption aligncenter">
  <a href="https://www.programmingwithwolfgang.com/wp-content/uploads/2018/04/Delete-a-record-from-the-table.jpg"><img aria-describedby="caption-attachment-1144" loading="lazy" class="size-full wp-image-1144" src="https://www.programmingwithwolfgang.com/wp-content/uploads/2018/04/Delete-a-record-from-the-table.jpg" alt="Delete a record from the table" width="764" height="152" srcset="https://www.programmingwithwolfgang.com/wp-content/uploads/2018/04/Delete-a-record-from-the-table.jpg 764w, https://www.programmingwithwolfgang.com/wp-content/uploads/2018/04/Delete-a-record-from-the-table-300x60.jpg 300w" sizes="(max-width: 764px) 100vw, 764px" /></a>
  
  <p id="caption-attachment-1144" class="wp-caption-text">
    Delete a record from the table
  </p>
</div>

### Designing, managing, and scaling table partitions

As previously mentioned, tables are partitioned to allow massive scaling. The partition key is the unit of scale for storage tables. The table services will spread your table to multiple servers and key all rows with the same partition key co-located. Therefore, the partition key is an important grouping for querying and availability.

There are three types of partition keys:

<div class="table-responsive">
  <table class="table table-striped table-bordered table-hover">
    <tr>
      <td>
        Partition Key
      </td>
      
      <td>
        Description
      </td>
    </tr>
    
    <tr>
      <td>
        Single Value
      </td>
      
      <td>
        There is one partition key for the the entire table. This favors a small number of entities. It also makes batch transactions easier since batch transactions need to share a partition key. It does not scale well for large tables since all rows will be on the same partition server
      </td>
    </tr>
    
    <tr>
      <td>
        Multiple Values
      </td>
      
      <td>
        This might place each partition on its own partition server. If the partition size is smaller, it is easier for Azure to load balance partitions. Partitions might get slower as the number of entities increases. This might make further partitioning necessary at some point.
      </td>
    </tr>
    
    <tr>
      <td>
        Unique values
      </td>
      
      <td>
        This is for many small partitions. This is highly scalable, but batch transactions are not possible.
      </td>
    </tr>
  </table>
</div>

For query performance, you should use the partition key and row key always together, if possible. This returns an exact row match. The next best thing is to have an exact partition match with a row range. It is best to avoid scanning the entire table.

## Azure Storage Queues

Azure Storage Queue provides a mechanism for reliable inter-application-messaging to support asynchronous distributed application workflows.

Queues are often used in an ordering or booking application. The customer orders or books something, for example, a plane ticket and gets an email confirmation a couple minutes later. The order gets put into a queue at the end of the ordering process and then gets processed by a separate application. This takes sometimes more, sometimes less time, depending on the workload. After the order is processed, the customer gets an email confirmation.

### Adding messages to a queue

You can find the following demo on <a href="https://github.com/WolfgangOfner/Azure-StorageQueue" target="_blank" rel="noopener">GitHub</a>.

  1. Create a new C# console application.
  2. Add the following code to your app.config file:

<div id="attachment_1135" style="width: 568px" class="wp-caption aligncenter">
  <a href="https://www.programmingwithwolfgang.com/wp-content/uploads/2018/04/Setup-the-connection-string-to-your-storage-account.jpg"><img aria-describedby="caption-attachment-1135" loading="lazy" class="size-full wp-image-1135" src="https://www.programmingwithwolfgang.com/wp-content/uploads/2018/04/Setup-the-connection-string-to-your-storage-account.jpg" alt="Setup the connection string to your storage account" width="558" height="87" srcset="https://www.programmingwithwolfgang.com/wp-content/uploads/2018/04/Setup-the-connection-string-to-your-storage-account.jpg 558w, https://www.programmingwithwolfgang.com/wp-content/uploads/2018/04/Setup-the-connection-string-to-your-storage-account-300x47.jpg 300w" sizes="(max-width: 558px) 100vw, 558px" /></a>
  
  <p id="caption-attachment-1135" class="wp-caption-text">
    Setup the connection string to your storage account
  </p>
</div>

Replace the placeholder with your storage account name and storage account key.

  1. Install the WindowsAzure.Storage NuGet package.
  2. In the Main method, retrieve the connection string:

<div id="attachment_1136" style="width: 710px" class="wp-caption aligncenter">
  <a href="https://www.programmingwithwolfgang.com/wp-content/uploads/2018/04/Retrieve-the-connection-string.jpg"><img aria-describedby="caption-attachment-1136" loading="lazy" class="wp-image-1136" src="https://www.programmingwithwolfgang.com/wp-content/uploads/2018/04/Retrieve-the-connection-string.jpg" alt="Retrieve the connection string" width="700" height="20" srcset="https://www.programmingwithwolfgang.com/wp-content/uploads/2018/04/Retrieve-the-connection-string.jpg 767w, https://www.programmingwithwolfgang.com/wp-content/uploads/2018/04/Retrieve-the-connection-string-300x9.jpg 300w" sizes="(max-width: 700px) 100vw, 700px" /></a>
  
  <p id="caption-attachment-1136" class="wp-caption-text">
    Retrieve the connection string
  </p>
</div>

<ol start="3">
  <li>
    You can create a queue with the following code:
  </li>
</ol>

<div id="attachment_1145" style="width: 427px" class="wp-caption aligncenter">
  <a href="https://www.programmingwithwolfgang.com/wp-content/uploads/2018/04/Create-the-queue-queue-if-it-does-not-exist.jpg"><img aria-describedby="caption-attachment-1145" loading="lazy" class="size-full wp-image-1145" src="https://www.programmingwithwolfgang.com/wp-content/uploads/2018/04/Create-the-queue-queue-if-it-does-not-exist.jpg" alt="Create the queue queue if it does not exist" width="417" height="70" srcset="https://www.programmingwithwolfgang.com/wp-content/uploads/2018/04/Create-the-queue-queue-if-it-does-not-exist.jpg 417w, https://www.programmingwithwolfgang.com/wp-content/uploads/2018/04/Create-the-queue-queue-if-it-does-not-exist-300x50.jpg 300w" sizes="(max-width: 417px) 100vw, 417px" /></a>
  
  <p id="caption-attachment-1145" class="wp-caption-text">
    Create the queue queue if it does not exist
  </p>
</div>

<ol start="4">
  <li>
    After the queue is created, add messages to it:
  </li>
</ol>

<div id="attachment_1146" style="width: 441px" class="wp-caption aligncenter">
  <a href="https://www.programmingwithwolfgang.com/wp-content/uploads/2018/04/Add-three-messages-to-the-queue.jpg"><img aria-describedby="caption-attachment-1146" loading="lazy" class="size-full wp-image-1146" src="https://www.programmingwithwolfgang.com/wp-content/uploads/2018/04/Add-three-messages-to-the-queue.jpg" alt="Add three messages to the queue" width="431" height="53" srcset="https://www.programmingwithwolfgang.com/wp-content/uploads/2018/04/Add-three-messages-to-the-queue.jpg 431w, https://www.programmingwithwolfgang.com/wp-content/uploads/2018/04/Add-three-messages-to-the-queue-300x37.jpg 300w" sizes="(max-width: 431px) 100vw, 431px" /></a>
  
  <p id="caption-attachment-1146" class="wp-caption-text">
    Add three messages to the queue
  </p>
</div>

<ol start="5">
  <li>
    In the Azure Portal, open your storage account and select Queues on the Overview blade. There, you can see your newly created queue and the three added messages.
  </li>
</ol>

[<img loading="lazy" width="774" height="423" class="wp-image-1096" src="https://www.programmingwithwolfgang.com/wp-content/uploads/2018/04/Displaying-the-previously-added-messages-in-the-Azure-Portal.jpg" alt="&quot;Displaying" srcset="https://www.programmingwithwolfgang.com/wp-content/uploads/2018/04/Displaying-the-previously-added-messages-in-the-Azure-Portal.jpg 774w, https://www.programmingwithwolfgang.com/wp-content/uploads/2018/04/Displaying-the-previously-added-messages-in-the-Azure-Portal-300x164.jpg 300w, https://www.programmingwithwolfgang.com/wp-content/uploads/2018/04/Displaying-the-previously-added-messages-in-the-Azure-Portal-768x420.jpg 768w" sizes="(max-width: 774px) 100vw, 774px" />](https://www.programmingwithwolfgang.com/wp-content/uploads/2018/04/Displaying-the-previously-added-messages-in-the-Azure-Portal.jpg)

On the screenshot above, you can see that every message in the queue got a unique ID assigned. This ID is used by the Storage Client Library to identify a message.

The maximum size for a message in a queue is 64 KB, but it is best practice to keep the messages as small as possible and to store any required data for processing in a durable store, such as Azure SQL or Azure Storage tables. The Azure Storage Queue can store messages up to seven days.

### Processing messages

Messages are usually published by a different application from the application that listens for new messages to process them. For simplicity, I use the same application for publishing and listening in this demo. To de-queue a message use the following code:

<div id="attachment_1147" style="width: 400px" class="wp-caption aligncenter">
  <a href="https://www.programmingwithwolfgang.com/wp-content/uploads/2018/04/Dequeue-a-single-message-from-the-queue-which-is-not-older-than-five-minutes.jpg"><img aria-describedby="caption-attachment-1147" loading="lazy" class="size-full wp-image-1147" src="https://www.programmingwithwolfgang.com/wp-content/uploads/2018/04/Dequeue-a-single-message-from-the-queue-which-is-not-older-than-five-minutes.jpg" alt="Dequeue a single message from the queue which is not older than five minutes" width="390" height="105" srcset="https://www.programmingwithwolfgang.com/wp-content/uploads/2018/04/Dequeue-a-single-message-from-the-queue-which-is-not-older-than-five-minutes.jpg 390w, https://www.programmingwithwolfgang.com/wp-content/uploads/2018/04/Dequeue-a-single-message-from-the-queue-which-is-not-older-than-five-minutes-300x81.jpg 300w" sizes="(max-width: 390px) 100vw, 390px" /></a>
  
  <p id="caption-attachment-1147" class="wp-caption-text">
    Dequeue a single message from the queue which is not older than five minutes
  </p>
</div>

The code returns the oldest message, which is not older than five minutes.

### Retrieving a batch of messages

A queue listener can be implemented as single-threaded (processing one message at a time) or multi-threaded (processing messages in a batch on separate threads). You can retrieve up to 32 messages from a queue using GetMessages() to process multiple messages in parallel. To get more than one message with GetMessages(), specify the number of items which should be de-queued:

<div id="attachment_1148" style="width: 438px" class="wp-caption aligncenter">
  <a href="https://www.programmingwithwolfgang.com/wp-content/uploads/2018/04/Dequeue-multiple-messages-from-the-queue.jpg"><img aria-describedby="caption-attachment-1148" loading="lazy" class="size-full wp-image-1148" src="https://www.programmingwithwolfgang.com/wp-content/uploads/2018/04/Dequeue-multiple-messages-from-the-queue.jpg" alt="Dequeue multiple messages from the queue" width="428" height="112" srcset="https://www.programmingwithwolfgang.com/wp-content/uploads/2018/04/Dequeue-multiple-messages-from-the-queue.jpg 428w, https://www.programmingwithwolfgang.com/wp-content/uploads/2018/04/Dequeue-multiple-messages-from-the-queue-300x79.jpg 300w" sizes="(max-width: 428px) 100vw, 428px" /></a>
  
  <p id="caption-attachment-1148" class="wp-caption-text">
    Dequeue multiple messages from the queue
  </p>
</div>

Consider the overhead of message processing before deciding the appropriate number of messages to process in parallel. If significant memory, disk space, or other network resources are used during processing, throttling parallel processing to an acceptable number will be necessary to avoid performance degradation on the compute instance.

### Scaling queues

Each individual queue has a target of approximately 20,000 messages per second. You can partition your application to use multiple queues to increase this throughput.

It is more cost-effective and efficient to pull multiple messages from the queue for processing in parallel on a single compute node. However, this depends on the type of processing and resources required. Scaling out compute nodes to increase processing throughput is usually also required.

You can configure VMs or cloud services to auto-scale by queue. You can specify the average number of messages to be processed per instance, and the auto-scale algorithm will queue to run scale actions to increase or decrease available instances accordingly.

## Choose between Azure Storage Tables and Azure Cosmos DB Table API

Azure Cosmos DB is a cloud-hosted, NoSQL database. A NoSQL database can be key/value stores, table stores, and graph stores. Azure Cosmos DB Table API is a key value store that is very similar to Azure Storage Tables.

The main differences are:

  1. Azure Table Storage only supports a single region with one optional secondary for high availability. Cosmos DB supports over 30 regions.
  2. Azure Table Storage only indexes the partition and the row key. Cosmos DB automatically indexes all properties.
  3. Azure Cosmos DB is much faster, with latency lower than 10 ms on reads and 15 ms on writes at any scale.
  4. Azure Table Storage only supports strong or eventual consistency. Stronger consistency means less overall throughput and concurrent performance while having more up to date data. Eventual consistency allows for high concurrent throughput but you might see older data. Azure Cosmos DB supports five different consistency models and allows those models to be specified at the session level. This means that on user or feature might have a different consistency level than a different user or feature.
  5. Azure Table Storage only charges for the storage you use. This makes it very affordable. Azure Cosmos DB on the other side charges for Request Units (RU) which really is a way for a PaaS product to charge for computer fees. If you need more RUs, you can scale them up. This makes Cosmos DB significantly more expensive than Azure Storage Tables.

## Conclusion

In this post, I talked about how to use Azure Storage Tables and Azure Storage Queues. I showed how to programmatically access both and how to add, retrieve, and remove data. Then, I talked about how to scale Azure Storage Tables and Azure Storage Queues. The last section was a comparison between Azure Storage Tables and Azure Cosmos DB.

You can find the demo code for the Azure Storage Table <a href="https://github.com/WolfgangOfner/Azure-TableStorage" target="_blank" rel="noopener">here</a> and the demo code for the Azure Storage Queues <a href="https://github.com/WolfgangOfner/Azure-StorageQueue" target="_blank" rel="noopener">here</a>.

For more information about the 70-532 exam get the <a href="http://amzn.to/2EWNWMF" target="_blank" rel="noopener">Exam Ref book from Microsoft</a> and continue reading my blog posts. I am covering all topics needed to pass the exam. You can find an overview of all posts related to the 70-532 exam <a href="https://www.programmingwithwolfgang.com/prepared-for-the-70-532-exam/" target="_blank" rel="noopener">here</a>.