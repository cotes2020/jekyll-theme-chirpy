
# Alibaba Cloud Elasticsearch



## Overview
Elasticsearch is an open source, distributed, real-time search and analytics engine built on Apache Lucene. It is released under the Apache License and is a popular search engine for enterprises. It provides services based on RESTful APIs and allows you to store, query, and analyze large amounts of datasets in near real time. Elasticsearch is typically used to support complex queries and high-performance applications.

Alibaba Cloud Elasticsearch is a fully managed cloud service that is developed based on open source Elasticsearch.
- This service is fully compatible with the features provided by open source Elasticsearch.
- It is out-of-the-box and supports the pay-as-you-go billing method.
- In addition to Elastic Stack components such as `Elasticsearch, Logstash, Kibana, and Beats`, Alibaba Cloud provides `the X-Pack plug-in` free of charge together with Elastic.
- X-Pack is integrated into Kibana to provide features, such as security, alerting, monitoring, and machine learning.
- It also provides SQL capabilities.

Alibaba Cloud Elasticsearch is widely used in scenarios such as <font color=blue> real-time log analysis and processing, information retrieval, multidimensional data queries, and statistical data analytics </font>.


- providing a low-cost, scenario-based Elasticsearch service on the cloud based on the open source Elastic Stack ecosystem. Alibaba Cloud Elasticsearch originates from but is not limited to this ecosystem. Alibaba Cloud has superior computing and storage capabilities on the cloud and technical expertise in the fields of cluster security and O&M. This enables Alibaba Cloud Elasticsearch to support one-click deployment, auto scaling, intelligent O&M, and various kernel optimization features. Alibaba Cloud Elasticsearch also provides a complete set of solutions such as migration, disaster recovery, backup, and monitoring.

Alibaba Cloud Elasticsearch features high security, performance, and availability and provides powerful search and analytics capabilities. It simplifies cluster deployment and management, reduces resource and O&M costs, ensures data security and reliability, enables upstream and downstream data links, and optimizes read and write performance. Based on these features and optimizations, Alibaba Cloud Elasticsearch allows you to build business applications with ease, such as applications that perform log analysis, exception monitoring, enterprise search, and big data analytics. Alibaba Cloud Elasticsearch enables you to focus on the business applications themselves and add value to your business.

## Components

The Alibaba Cloud Elastic Stack ecosystem contains the following components: Elasticsearch, Kibana, Beats, and Logstash. Elasticsearch is a real-time, distributed search and analytics engine. Kibana provides a visual interface for data analytics. Beats collects data from various machines and systems. Logstash collects, converts, processes, and generates data. Integrated with Kibana, Beats, and Logstash, Alibaba Cloud Elasticsearch can be used for real-time log processing, full-text searches, and data analytics.

### X-Pack

X-Pack is a commercial extension of Elasticsearch. It provides security, alerting, monitoring, graphing, reporting, and machine learning capabilities. When you create an Alibaba Cloud Elasticsearch cluster, the system integrates X-Pack into Kibana to provide free services. The services include authorization and authentication, role-based access control, real-time monitoring, visual reporting, and machine learning. X-Pack facilitates cluster O&M and application development.

### Beats

Beats is a lightweight data collection tool that integrates a variety of single-purpose data shippers. These data shippers collect data from various machines or systems and send the collected data to Logstash or Elasticsearch.

Beats allows you to create the following types of data shippers: Filebeat, Metricbeat, Auditbeat, and Heartbeat. You can create and configure a shipper to collect various types of data from Elastic Compute Service (ECS) instances or Container Service for Kubernetes (ACK) clusters. The data include logs, network data, and container metrics. Beats also allows you to manage your shippers in a centralized manner.

### Logstash

Logstash is a server-side data processing pipeline. It uses input, filter, and output plug-ins to dynamically collect data from a variety of sources, process and convert the data, and then save the data to a specific location.

Alibaba Cloud Logstash is a fully managed service and is fully compatible with open source Logstash. Logstash allows you to quickly deploy pipelines, configure them by using a visual interface, and centrally manage them. It provides multiple types of plug-ins to connect to cloud services, such as Object Storage Service (OSS) and MaxCompute.

### Kibana

Kibana is a flexible data analytics and visualization tool. Multiple users can log on to the Kibana console at the same time. You can use Kibana to search for, view, and manage data in Elasticsearch indexes. When you create an Alibaba Cloud Elasticsearch cluster, the system automatically deploys an independent Kibana node. This node allows you to present diversified data analytics reports and dashboards by using graphs, tables, or maps based on your business requirements.

### Related items

**AliES and its provided plug-ins**

In addition to all the features provided by the open source Elasticsearch kernel, Alibaba Cloud Elasticsearch develops the AliES kernel. This kernel enables Alibaba Cloud Elasticsearch to provide optimizations in multiple aspects, such as thread pools, monitoring metric types, circuit breaking policies, and query and write performance. The kernel also provides a variety of self-developed plug-ins to improve cluster stability, enhance performance, reduce costs, and optimize monitoring and O&M.

**EYou**

EYou is an intelligent O&M system provided by Alibaba Cloud Elasticsearch. This system can detect the health of more than 20 items, such as clusters, nodes, and indexes. EYou simplifies cluster O&M. It observes and records the running statuses of clusters and automatically summarizes cluster diagnostic results. It also detects the possible risks of your clusters. If your clusters are abnormal, the system quickly provides key information and reasonable optimization suggestions.



## Deploy

The Alibaba Cloud Elastic Stack ecosystem contains the following components: `Elasticsearch, Logstash, and Beats`.
- Elasticsearch is a real-time, distributed search and analytics engine.
- Logstash collects, converts, processes, and generates data.
- Beats collects data from various machines and systems.
- These components enable Alibaba Cloud Elasticsearch to be used for real-time log processing, full-text searches, and data analytics.


### Pre

#### specifications and storage capacity

evaluate the total amount of the required resources, such as the disk space, node specifications, number of shards, and size of each shard


**Disk space** evaluation
- The **disk space of an Elasticsearch cluster** is determined by the following factors:
  - Number of `replica shards`: Each primary shard must have at least one replica shard.
  - `Indexing` overheads:
    - In most cases, indexing overheads are 10% greater than those of source data.
    - The overheads of the `_all` parameter are not included.
  - Disk space reserved by the `operating system`: By default, the operating system reserves 5% of disk space for critical processes, system recovery, and disk fragments.
  - `Elasticsearch` overheads: Elasticsearch reserves 20% of disk space for internal operations, such as segment merging and logging.
    - Security threshold overheads: Elasticsearch reserves at least 15% of disk space as the security threshold.

The minimum required disk space is calculated by using the following formula:
```bash
Minimum required disk space =
    Volume of source data
    × (1 + Number of replica shards)
    × Indexing overheads/(1 - Disk space reserved by the operating system)/(1 - Elasticsearch overheads)/(1 - Security threshold overheads)
= Volume of source data × (1 + Number of replica shards) × 1.7
= Volume of source data × 3.4
```


**Node specification** evaluation
- The **performance of an Elasticsearch cluster** is determined by the `specifications of each node` in the cluster:

- Maximum number of nodes per cluster:

```bash
Maximum number of nodes per cluster = Number of vCPUs per node × 5
```

- Maximum volume of data per node:

  - The maximum volume of data that a node in an Elasticsearch cluster can store depends on the scenario.

```bash
# Acceleration or aggregation on data queries:
Maximum volume of data per node = Memory size per node (GiB) × 10

# Log data import or offline analytics:
Maximum volume of data per node = Memory size per node (GiB) × 50

# General scenarios:
Maximum volume of data per node = Memory size per node (GiB) × 30
```


**Shard** evaluation
- The number of shards and the size of each shard determine the **stability and performance of an Elasticsearch cluster**.
- You must properly plan shards for all indexes in an Elasticsearch cluster.
- This prevents numerous shards from affecting cluster performance when it is difficult to define business scenarios.


### step

Procedure:

- Step 1: Create a cluster
  - Create an Alibaba Cloud Elasticsearch V6.7 cluster of the Standard Edition.

- Step 2: Access the cluster
  - Log on to the Kibana console of the cluster to access the cluster after the state of the cluster becomes Active.

- Step 3: Create an index
  - Call a RESTful API to create an index.

- Step 4: Create documents and insert data into the documents
  - Call a RESTful API to create documents and insert data into the documents.

- Step 5: Search for data
  - Call a RESTful API to perform a full-text search or search for data by condition.

- Step 6: (Optional) Delete the index
  - Call a RESTful API to delete the index to save resources if you no longer require the index.

- Step 7: (Optional) Release the cluster
  - Release the cluster if you no longer require the cluster. After a cluster is released, data stored in the cluster cannot be recovered. We recommend that you back up data before you release a cluster.


.
