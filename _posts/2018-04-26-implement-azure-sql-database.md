---
title: Implement Azure SQL database
date: 2018-04-26T21:52:05+02:00
author: Wolfgang Ofner
categories: [Cloud]
tags: [70-532, Azure, Certification, Exam, learning]
---
Microsoft Azure offers with Azure SQL database a great alternative to an on-premise SQL database. In this post, I will talk about the advantages of having the database in the cloud, how to get data into the cloud and how to use Elastic pools to share resources between several databases. The last section will be about implementing graph functionality in an Azure SQL database.

## Create an Azure SQL database

To create an Azure SQL database in the Azure portal, follow these steps:

  1. In the Azure portal go to SQL databases and click +Add.
  2. On the SQL Database blade provide: 
      * Name
      * Subscription
      * Resource group
      * Source (Blank database, Demo database or a backup)
      * Server
      * Pricing tier
  3. After you entered all information, click on Create.

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2018/04/Create-a-new-SQL-database.jpg"><img loading="lazy" src="/assets/img/posts/2018/04/Create-a-new-SQL-database.jpg" alt="Create a new Azure SQL database" /></a>
  
  <p>
    Create a new Azure SQL database
  </p>
</div>

## Choosing the appropriate database tier and performance level

Azure offers three different pricing tiers to choose from. The major difference between them is in a measurement called database throughput units (DTUs). A DTU is a blended measure of CPU, memory, disk reads, and disk writes. SQL database is a shared resource with other Azure customers, sometimes performance is not stable or predictable. As you go up in performance tiers, you also get better predictability in performance.

The three pricing tiers are:

<div class="table-responsive">
  <table class="table table-striped table-bordered table-hover">
    <tr>
      <td>
        Pricing tier
      </td>
      
      <td>
        Description
      </td>
    </tr>
    
    <tr>
      <td>
        Basic
      </td>
      
      <td>
        The Basic tier is meant for light workloads. I use this tier for testing and at the beginning of a new project.
      </td>
    </tr>
    
    <tr>
      <td>
        Standard
      </td>
      
      <td>
        The Standard tier is used for most production online transaction processing (OLTP) databases. The performance is more predictable than the basic tier. In addition, there are 13 performance levels under this tier, levels S0 to S12.
      </td>
    </tr>
    
    <tr>
      <td>
        Premium
      </td>
      
      <td>
        The Premium tier continues to scale at the same level as the Standard tier. In addition, performance is typically measured in seconds. For instance, the basic tier can handle 16,600 transactions per hour. The standard S2 level can handle 2,570 transactions per minute. The top tier premium of premium can handle 75,000 transactions per second.
      </td>
    </tr>
  </table>
</div>

Each tier has a 99,99 percent up-time SLA, backup and restore capabilities, access to the same tooling, and the same database engine features.

### Analyzing metrics

To review the metrics of your Azure SQL database, follow these steps:

  1. In the Azure portal go to your Azure SQL database.
  2. On the Overview blade, click on the Resource graph.
  3. This opens the Metrics blade. There select the desired metrics, you wish to analyze.

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2018/04/Configure-the-metrics-graph.jpg"><img loading="lazy" src="/assets/img/posts/2018/04/Configure-the-metrics-graph.jpg" alt="Configure the metrics graph" /></a>
  
  <p>
    Configure the metrics graph
  </p>
</div>

Note that there is nothing on the graph because I just created the database and haven&#8217;t used it yet.

## Configure and performing point in time recovery

Azure SQL database does a full backup every week, a differential backup each day, and an incremental log backup every five minutes. The incremental log backup allows for a point in time restore, which means the database can be restored to any specific time of the day. This means that if you accidentally delete a customer&#8217;s table from your database, you will be able to recover it with minimal data loss if you know the time frame to restore from that has the most recent copy.

The further away you get from the last differential backup determines the longer the restore operation takes. When you restore a new database, the service tier stays the same, but the performance level changes to the minimum of that tier.

The retention period of the backup depends on the pricing tier. Basic retains backups for 7 days, Standard and Premium for 35 days. A deleted database can be restored, as long as you are in within the retention period.

### Restore an Azure SQL database

To restore an Azure SQL database, follow these steps:

  1. In the Azure portal, on the Overview blade of your database, click on Restore.
  2. On the Restore blade, select a restore point, either Point-in-time or Long-term backup retention.
  3. If you selected Point-in-time, select a date and time. If you selected Long-term backup retention, select the backup from the drop-down list.
  4. Optionally change the Pricing tier.
  5. Click on OK to restore the backup.

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2018/04/Restore-an-Azure-SQL-database.jpg"><img loading="lazy" src="/assets/img/posts/2018/04/Restore-an-Azure-SQL-database.jpg" alt="Restore an Azure SQL database" /></a>
  
  <p>
    Restore an Azure SQL database
  </p>
</div>

### Restore a deleted Azure SQL database

To restore a deleted Azure SQL database, follow these steps:

  1. In the Azure portal, go to your SQL server and select the Deleted databases blade under the Setting menu.
  2. On the Deleted databases blade, select the database, you want to restore.

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2018/04/Select-a-deleted-Azure-SQL-database-to-restore.jpg"><img loading="lazy" src="/assets/img/posts/2018/04/Select-a-deleted-Azure-SQL-database-to-restore.jpg" alt="Select a deleted Azure SQL database to restore" /></a>
  
  <p>
    Select a deleted Azure SQL database to restore
  </p>
</div>

<ol start="3">
  <li>
    On the  Restore blade, change the database name if desired and click OK to restore the database.
  </li>
</ol>

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2018/04/Restore-the-deleted-Azure-SQL-database.jpg"><img loading="lazy" src="/assets/img/posts/2018/04/Restore-the-deleted-Azure-SQL-database.jpg" alt="Restore the deleted Azure SQL database" /></a>
  
  <p>
    Restore the deleted Azure SQL database
  </p>
</div>

## Enable geo-replication

By default, every Azure SQL database is copied three times across the datacenter. Additionally, you can configure geo-replication. The advantages of geo-replication are:

  * <span class="fontstyle0">You can fail over to a different data center in the event of a natural disaster or other intentionally malicious act.</span>
  * <span class="fontstyle0">Online secondary databases are readable, and they can be used as load balancers for read-only workloads such as reporting.</span>
  * <span class="fontstyle0">With automatic asynchronous replication, after an online secondary database has been seeded, updates to the primary database are automatically copied to the secondary database.</span>

### Create an online secondary database

To enable geo-replication, follow these steps:

  1. Go to your Azure SQL database in the Azure portal and click on Geo-Replication under the Settings menu.
  2. Select the target region.
  3. On the Create secondary blade, enter the server and pricing information.
  4. Click on OK.

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2018/04/Create-an-online-secondary-database.jpg"><img loading="lazy" src="/assets/img/posts/2018/04/Create-an-online-secondary-database.jpg" alt="Create an online secondary database" /></a>
  
  <p>
    Create an online secondary database
  </p>
</div>

## Import and export schema and data

To export the metadata and state data of a SQL server database, you can create a BACPAC file.

### Export a BACPAC file from a SQL database

To create a BACPAC file, follow these steps:

  1. Open SQL Server Management Studio and connect to your database.
  2. Right-click on the database, select Tasks and click on Export Data-tier Application.

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2018/04/Export-your-source-database.jpg"><img loading="lazy" src="/assets/img/posts/2018/04/Export-your-source-database.jpg" alt="Export your source database" /></a>
  
  <p>
    Export your source database
  </p>
</div>

<ol start="3">
  <li>
    In the Export Data-tier Application wizard, you can export the BACPAC file to Azure into a blob storage or to a local storage device.
  </li>
</ol>

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2018/04/Select-a-location-for-the-export-file.jpg"><img loading="lazy" src="/assets/img/posts/2018/04/Select-a-location-for-the-export-file.jpg" alt="Select a location for the export file" /></a>
  
  <p>
    Select a location for the export file
  </p>
</div>

<ol start="4">
  <li>
    After you selected the export destination, click Next and then Finish to export the file.
  </li>
</ol>

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2018/04/Export-successful.jpg"><img loading="lazy" src="/assets/img/posts/2018/04/Export-successful.jpg" alt="Export successful" /></a>
  
  <p>
    Export successful
  </p>
</div>

### Import a BACPAC file into Azure SQL database

To import a BACPAC file into your Azure SQL database, follow these steps:

  1. Open SQL Server Management Studio and connect to your Azure SQL server. You can find the server name in the Azure Portal on your SQL server. Go to Properties under the Settings menu.
  2. If you can&#8217;t connect to your server, you may have to allow your UP address in the firewall. Select Firewalls and virtual network under the Settings menu and enter your IP address or a range of IPs.

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2018/04/Configure-the-firewall-of-your-SQL-server.jpg"><img loading="lazy" src="/assets/img/posts/2018/04/Configure-the-firewall-of-your-SQL-server.jpg" alt="Configure the firewall of your SQL server" /></a>
  
  <p>
    Configure the firewall of your SQL server
  </p>
</div>

<ol start="3">
  <li>
    After you are connected to the server, right-click on the database folder and select Import Data-tier Application.
  </li>
</ol>

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2018/04/Start-the-Import-Data-tier-Application-wizard.jpg"><img loading="lazy" src="/assets/img/posts/2018/04/Start-the-Import-Data-tier-Application-wizard.jpg" alt="Start the Import Data-tier Application wizard" /></a>
  
  <p>
    Start the Import Data-tier Application wizard
  </p>
</div>

<ol start="4">
  <li>
    In the Import Data-tier Application wizard, select the BACPAC file from a local disk or from your Azure storage account.
  </li>
</ol>

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2018/04/Select-your-BACPAC-file-for-the-import.jpg"><img loading="lazy" src="/assets/img/posts/2018/04/Select-your-BACPAC-file-for-the-import.jpg" alt="Select your BACPAC file for the import" /></a>
  
  <p>
    Select your BACPAC file for the import
  </p>
</div>

<ol start="5">
  <li>
    On the Database Settings blade, enter a database name and the pricing tier.
  </li>
</ol>

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2018/04/Enter-a-database-name-and-select-the-pricing-tier.jpg"><img loading="lazy" src="/assets/img/posts/2018/04/Enter-a-database-name-and-select-the-pricing-tier.jpg" alt="Enter a database name and select the pricing tier" /></a>
  
  <p>
    Enter a database name and select the pricing tier
  </p>
</div>

<ol start="6">
  <li>
    Click Next and then Finish.
  </li>
</ol>

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2018/04/The-summary-of-the-import-wizard.jpg"><img loading="lazy" src="/assets/img/posts/2018/04/The-summary-of-the-import-wizard.jpg" alt="The summary of the import wizard" /></a>
  
  <p>
    The summary of the import wizard
  </p>
</div>

<ol start="7">
  <li>
    After the import process is finished, you can see the database in your SQL server in the Azure portal by selecting the SQL databases blade under the Settings menu.
  </li>
</ol>

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2018/04/The-database-got-created-by-the-imported-schema.jpg"><img loading="lazy" src="/assets/img/posts/2018/04/The-database-got-created-by-the-imported-schema.jpg" alt="The database got created by the imported schema" /></a>
  
  <p>
    The database got created by the imported schema
  </p>
</div>

## Scale Azure SQL databases {#ScaleAzureSQLdatabases}

You can scale-up and scale-out your Azure SQL databases.

Scaling-up means to add CPU, memory, and better disk i/o to handle the load. To do that click in the Azure portal on your database on Database size under the monitoring menu and move the slider to the right, or select a higher pricing tier.

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2018/04/Scale-out-your-Azure-SQL-database.jpg"><img loading="lazy" src="/assets/img/posts/2018/04/Scale-out-your-Azure-SQL-database.jpg" alt="Scale-up your Azure SQL database" /></a>
  
  <p>
    Scale-up your Azure SQL database
  </p>
</div>

Scaling-up will give you more DTUs.

Scaling-out means breaking apart a database into smaller pieces. This is called sharding. Methods for sharding can be, for example, by function, by geo-location or by business unit.

Another reason for sharding can be that the database is too large to be stored in a single Azure SQL database or that taking a backup takes too long due to the size of the database.

To increase the performance, a shard map is used. This is usually a table or database that tells the application where data actually is and where to look for it. A shard map also keeps you from rewriting a big part of your application to handle sharding.

Sharding is easily implemented in Azure Table Storage and Azure Cosmos DB but is way more difficult in a relational database. The complexity comes from being transactionally consistent, while having data available and spread throughout several databases.

To help developers, Microsoft released a set of tools called Elastic Database Tools that are compatible with Azure SQL database.  <span class="fontstyle0">This client library can be used in your application to create sharded databases. It has a split-merge tool that will allow you to create new nodes or drop nodes without data loss. It also includes a tool that will keep schema consistent across all the nodes by running scripts on each node individually.</span>

## Managed elastic pools, including DTUs and eDTUs

A single SQL database server can have several databases on it. Those databases can each have their own size and pricing tier. This might work out well if you always know exactly how large each database will and how many DTUs are needed for each one. What happens if you don&#8217;t really know that? What if you want all your databases on one server to share their DTUs? The solution for this are Elastic pools.

Elastic pools enable you to purchase elastic Database Transaction Units (eDTUs) for a pool of multiple databases. The user adds databases to the pool, sets the minimum and maximum eDTUs for each database, and sets the eDTU limit of the pool based on your budget. This means that within the pool, each database is given the ability to auto-scale in a set range.

### Create an Elastic pool

To create an Elastic pool, follow these steps:

  1. Go to your SQL server in the Azure portal and click on +New pool on the Overview blade.
  2. On the Elastic database pool blade, provide a name and select a pricing tier under the Configure pool setting.
  3. On the Resource Configuration & Pricing blade, click on Databases and then +Add databases to add your databases.

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2018/04/Create-a-new-Elastic-pool.jpg"><img loading="lazy" src="/assets/img/posts/2018/04/Create-a-new-Elastic-pool.jpg" alt="Create a new Elastic pool" /></a>
  
  <p>
    Create a new Elastic pool
  </p>
</div>

<ol start="4">
  <li>
    Click OK to create your Elastic pool.
  </li>
</ol>

## Implement Azure SQL Data Sync

SQL Data Sync allows you to <span class="fontstyle0">bi-directionally</span> replicate data between two Azure SQL databases or between an Azure SQL database and an on-premise SQL server.

Azure SQL Data Sync has the following attributes:

<div class="table-responsive">
  <table class="table table-striped table-bordered table-hover">
    <tr>
      <td>
        Attribute
      </td>
      
      <td>
        Description
      </td>
    </tr>
    
    <tr>
      <td>
        Sync Group
      </td>
      
      <td>
        A Sync Group is a group of databases that you want to synchronize using Azure SQL Data Sync.
      </td>
    </tr>
    
    <tr>
      <td>
        Sync Schema
      </td>
      
      <td>
        A Sync Schema is the data you want to synchronize.
      </td>
    </tr>
    
    <tr>
      <td>
        Sync Direction
      </td>
      
      <td>
        The Sync Direction allows you to synchronize data in either one direction or bi-directionally.
      </td>
    </tr>
    
    <tr>
      <td>
        Sync Interval
      </td>
      
      <td>
        Sync Interval controls how often synchronization occurs.
      </td>
    </tr>
    
    <tr>
      <td>
        Conflict Resolution Policy
      </td>
      
      <td>
        A Conflict Resolution Policy determines who wins if data conflicts with one another.
      </td>
    </tr>
  </table>
  
  <p>
    The following screenshot shows how a data sync infrastructure could look like.
  </p>
</div>

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2018/04/Azure-Data-Sync-diagram.jpg"><img loading="lazy" src="/assets/img/posts/2018/04/Azure-Data-Sync-diagram.jpg" alt="Azure Data Sync diagram" /></a>
  
  <p>
    Azure Data Sync diagram (<a href="http://amzn.to/2EWNWMF" target="_blank" rel="noopener">Source</a>)
  </p>
</div>

The hub database must always be an Azure SQL database. A member database can either be an Azure SQL database or an on-premise SQL server.

It is important to note that this is a method of keeping data consistent across multiple databases, it is not an ETL (Extract, Transform and Load) tool. This should not be used to populate a data warehouse or to migrate an on-premise SQL server to the cloud. This can be used to populate a read-only version of the database for reporting, but only if the schema is 100% consistent.

## Implement graph database functionality in Azure SQL database

A graph database is a NoSQL solution and introduces two new vocabulary words: nodes and relationships.

Nodes are entities in relational database terms and a relationship shows that a connection between nodes exists. The relationship in a graph database is hierarchical, where it is flat in a relational database.

A graph is an abstract representation of a set of objects where nodes are linked with relationships in a hierarchy. A graph database is a database with an explicit and enforceable graph structure. Another key difference between a relational database and a graph database is that as the number of nodes increase, the performance cost stays the same. Joining tables will burden the relational database and is a common source of performance issues when scaling. Graph databases don&#8217;t suffer from that issue.

Relational databases are optimized for aggregation, whereas graph databases are optimized for having plenty of connections between nodes.

In Azure SQL database, graph-like capabilities are implemented through T-SQL. You can create graph objects in T-SQL with the following syntax:

<span class="fontstyle0">Create Table Person(ID Integer Primary Key, Name Varchar(100)) As Node;<br /> Create Table kids (Birthdate date) As Edge;</span>

A query can look like this:

<span class="fontstyle0">SELECT Restaurant.name<br /> FROM Person, likes, Restaurant<br /> WHERE MATCH (Person-(likes)->Restaurant)<br /> AND Person.name = &#8216;Wolfgang&#8217;;</span>

This query will give you every restaurant name which is liked by a person named Wolfgang.

## Conclusion

In this post, I showed how to create an Azure SQL database and how to export the schema and metadata from your on-premise database to move it into the cloud. After creating the database in the cloud, I talked about restoring a backup and enabling geo-replication to increase the data security. Next, I talked about leveraging Elastic pools to dynamically share the resources between several databases and by doing so keeping your costs low. The last section talked about implementing graph database functionality with your Azure SQL database.

For more information about the 70-532 exam get the <a href="http://amzn.to/2EWNWMF" target="_blank" rel="noopener">Exam Ref book from Microsoft</a> and continue reading my blog posts. I am covering all topics needed to pass the exam. You can find an overview of all posts related to the 70-532 exam <a href="/prepared-for-the-70-532-exam/" target="_blank" rel="noopener">here</a>.