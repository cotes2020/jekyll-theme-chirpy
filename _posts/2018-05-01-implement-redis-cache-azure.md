---
title: Implement Redis Cache in Azure
date: 2018-05-01T20:10:55+02:00
author: Wolfgang Ofner
categories: [Cloud]
tags: [70-532, Azure, Certification, Exam, learning]
---
Redis Cache is a key-value store NoSQL database. Its implementation is very similar to Azure Table Storage. The main difference is Redis is very high performing by keeping the data in memory most of the time. By default, Redis also doesn&#8217;t persist the data between reboots. The main purpose of keeping Redis Cache in memory is for fast data retrieval and aggregations. Redis is typically used to augment the data store you have already selected. It will make lookups much faster and also helps to reduce the cost of your storage since it reduces the number of queries.

Redis has many uses, but primarily it is a temporary storage location of data that has a longer lifespan. That data needs to be expired when it is out of data and re-populated.

Azure Redis Cache is the Azure product built around Redis and offering it as a Platform as a Service product.

## Choose a Redis Cache tier

To create a Redis Cache, follow these steps:

  1. In the Azure portal select +Create a resource, search for Redis Cache and click Create.
  2. On the New Redis Chache blade, provide a DNS name, subscription, resource group, location, and pricing tier.
  3. Click Create.

<div id="attachment_1239" style="width: 325px" class="wp-caption aligncenter">
  <a href="/assets/img/posts/2018/05/Create-a-new-Redis-cache.png"><img aria-describedby="caption-attachment-1239" loading="lazy" class="size-full wp-image-1239" src="/assets/img/posts/2018/05/Create-a-new-Redis-cache.png" alt="Create a new Redis cache" width="315" height="403" /></a>
  
  <p id="caption-attachment-1239" class="wp-caption-text">
    Create a new Redis Cache
  </p>
</div>

There are three pricing tiers:

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
        The Basic tier is the cheapest and allows up to 53 GB of Redis Cache database size.
      </td>
    </tr>
    
    <tr>
      <td>
        Standard
      </td>
      
      <td>
        The Standard tier has the same storage limit as Basic but includes replication and failover with master/slave replication. This replication is automatic between two nodes
      </td>
    </tr>
    
    <tr>
      <td>
        Premium
      </td>
      
      <td>
        The Premium tier allows a database size of 530 GB and also offers persistence, which means that the data will survive a power outage. It also includes a much better network performance, allowing up to 40,000 client connections.
      </td>
    </tr>
  </table>
</div>

## Implement data persistence

Redis persistence allows you to save data to a disk instead of just memory. Additionally, you can take snapshots of your data for backups. This allows your Redis Cache to survive hardware failure and power outages. Redis persistence is implemented through a relational database model, where data is streamed out to binary into Azure Storage blobs.

To configure the frequency of the snapshots, follow these steps:

  1. In the Azure portal in your Redis Cache, click on Redis data persistence under the Settings menu. Note that you need a premium tier cache to do that.
  2. On the Redis data persistence blade, select the Backup Frequency and select a storage account.
  3. Click OK.

<div id="attachment_1240" style="width: 322px" class="wp-caption aligncenter">
  <a href="/assets/img/posts/2018/05/Configure-data-persistence.jpg"><img aria-describedby="caption-attachment-1240" loading="lazy" class="size-full wp-image-1240" src="/assets/img/posts/2018/05/Configure-data-persistence.jpg" alt="Configure data persistence" width="312" height="337" /></a>
  
  <p id="caption-attachment-1240" class="wp-caption-text">
    Configure data persistence
  </p>
</div>

## Implement security and network isolation

The primary security mechanism is done through access keys. The premium tier offers enhanced security features. This is done primarily through virtual networks (VNET) and allows you to hide your Redis Cache behind your application and not have a public URL that is open to the internet

The VNET is configured at the bottom of the New Redis Cache blade. You can&#8217;t configure it after it has been created. Additionally, you have to use an existing VNET which is in the same data center as your Redis Cache. The Redis Cache must be created in an empty subnet.

## Tune cluster performance

With the premium tier, you can implement a Redis Cluster. This allows you to split the dataset among multiple notes, allowing you to continue operations when a subset of the nodes experiences failure, gives more throughput, and increases memory size as you increase the number of shards. Redis clustering is configured when you create the Azure Redis Cache.

Once the cache is created, Redis distributes the data automatically.

## Integrate Redis caching with ASP.NET session and cache providers

Redis Cache is an excellent place to store session data. To implement this, install the Microsoft.Web.RedisSessionStateProvider NuGet package. Once added to the project, add this line to your web.config file under the providers section:

<add name=&#8221;MySessionStateStore&#8221; type=&#8221;Microsoft.Web.Redis.RedisSessionStateProvider&#8221; host=&#8221;YourHostURL&#8221; accessKey=&#8221;YourAccessKey&#8221; ssl=&#8221;true&#8221; />

Replace the host with the URL of your cache and the accessKey with your key. You can find the keys on the Access Keys blade and the URL on the Overview blade.

## Conclusion

In this short post, I showed how to create a Redis Cache and how to configure data persistence and security using the premium pricing tier.

For more information about the 70-532 exam get the <a href="http://amzn.to/2EWNWMF" target="_blank" rel="noopener">Exam Ref book from Microsoft</a> and continue reading my blog posts. I am covering all topics needed to pass the exam. You can find an overview of all posts related to the 70-532 exam <a href="/prepared-for-the-70-532-exam/" target="_blank" rel="noopener">here</a>.