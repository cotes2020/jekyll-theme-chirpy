
- [Ali - ApsaraDB RDS](#ali---apsaradb-rds)
  - [benefit](#benefit)
  - [RDS editions](#rds-editions)
  - [Features](#features)
    - [Customized database engines](#customized-database-engines)
      - [AliSQL](#alisql)
      - [AliPG](#alipg)
    - [Cost effective and easy to use](#cost-effective-and-easy-to-use)
    - [High performance](#high-performance)
    - [High availability and disaster tolerance](#high-availability-and-disaster-tolerance)
    - [High security](#high-security)

---

# Ali - ApsaraDB RDS

- a stable, reliable, and scalable online database service
- built on top of the `Apsara Distributed File System` and `high-performance SSDs` of Alibaba Cloud.
- ApsaraDB RDS supports the `MySQL, SQL Server, PostgreSQL, and MariaDB TX database engines`.
- ApsaraDB RDS provides a portfolio of solutions for disaster recovery, backup, restoration, monitoring, and migration to facilitate database O&M.


---

## benefit

1. Easy deployment
   1. no need to purchase database server hardware or software. reduces costs.
   2. create an RDS instance of required specifications within a few minutes in the ApsaraDB RDS console by calling an API operation.

2. High compatibility
   1. use ApsaraDB RDS databases in the same way as databases that run native engines.
   2. This relieves the need to acquire new knowledge.
   3. The data migration does not require much workforce. ApsaraDB RDS is compatible with existing programs and tools. You can use Data Transmission Service (DTS) to quickly migrate data to ApsaraDB RDS. You can also use common data import and export tools to migrate data.

3. Easy O&M
   1. Alibaba Cloud is responsible for routine maintenance and management tasks of ApsaraDB RDS instances.
   2. This includes troubleshooting hardware and software issues and installing database patches.
   3. You can add, delete, restart, back up, and restore RDS instances in the ApsaraDB RDS console or by calling API operations.



---


## RDS editions

1. Basic Edition
   1. Data backups are stored as multiple copies on OSS or distributed cloud disks to prevent data loss. This applies to all RDS series.
   2. a `single node without a slave node` as hot backup.
   3. if fault occurs, the restoration time is long.
   4. Choose Basic Edition if you do not require high availability.
   5. database system consists of only one primary RDS instance, and computing is separated from storage. This edition is cost-effective.
   6. scenario: Personal learning, Small-sized websites, Development and test environments for small- and medium-sized enterprises


2. High-availability Edition
   1. adopts the high-availability architecture with `one master node and one slave node`.
      1. database system consists of one primary RDS instance and one secondary RDS instance.
   2. These instances work in the high-availability architecture.
   3. This edition is suitable for more than 80% of the actual business scenarios.
   5. If the master node fails, a switchover occurs within seconds without affecting your applications.
   6. If the slave node fails, a new slave node is automatically generated to ensure high availability.
   7. **Single-zone instance**:
      1. The master and slave nodes are in the same zone but on different physical servers.
      2. All cabinets, air conditioners, electricity, and networks in the zone are redundant to ensure high availability.
   8. **Multi-zone instance**:
      1. The master and slave nodes are in different zones of an area, providing `area-level` disaster tolerance ability.
   9. You can switch between single-zone instances and multi-zone instances.
   10. When the slave node malfunctions, RDS instantly backs up the master node.
   11. When the backup process is about to finish, a global lock is generated and the master node runs in the read-only status for 5 seconds or less.
   6. scenario: Production databases for large and medium-sized enterprises, Databases in industries such as the Internet, Internet of Things (IoT), online retail, logistics, and gaming


3. Cluster Edition
   1. Only **RDS SQL Server 2017** provides the Cluster Edition.
   2. Based on the **AlwaysOn** technology, it provides `one master node`, `one slave node`, and up to seven `read-only nodes` that horizontally scale read capabilities.
      1. The read-only RDS instances are used to increase the read capability.
   3. The slave and read-only nodes synchronize data from the master node.
   4. The Cluster Edition provides the same availability as the High-availability Edition.
   5. Besides, the read-only nodes can be deployed in zones different from those of the master and slave nodes.
   6. scenario: Production databases for large- and medium-sized enterprises, such as online retailers, automobile companies, and ERP providers


4. **Enterprise Edition**
   1. **RDS MySQL 5.7** supports the Enterprise Edition.
   2. the DB system has `one master instance and two slave instances`.
   3. Data is replicated between the master and slave instances through multiple replicas to guarantee data consistency and finance-level reliability.
   4. Data is synchronously replicated from the primary RDS instance to the secondary RDS instances.
   5. The Enterprise Edition can be used by large-sized enterprises to build core production databases.
   6. the master and slave instances are located in three different zones in the same region.
   7. scenario: Important databases in the finance, securities, and insurance industries that require high data security, Important production databases for large-sized enterprises



---


## Features

- ApsaraDB RDS supports a wide range of features, such as `instance management, backup and restoration, log audit, and monitoring and alerting`.
- You can use the instance management feature to create RDS instances and change the specifications of an existing RDS instance.



### Customized database engines

Alibaba Cloud customizes database engines based on the community editions of native `MySQL` and `PostgreSQL` to provide more advanced features.


#### AliSQL
- an independent MySQL branch that is developed by Alibaba Cloud.
- AliSQL provides
  - all the features of the MySQL Community Edition.
  - some similar features that you can find in the MySQL Enterprise Edition.
    - enterprise-grade backup and restoration,
    - thread pool,
    - and parallel query.
  - In addition, AliSQL provides Oracle-compatible features, such as the Sequence engine.
- `ApsaraDB RDS for MySQL with AliSQL` provides
  - all MySQL features
  - and a wide range of advanced features that are developed by Alibaba Cloud.
    - include enterprise-grade security, backup and restoration, monitoring, performance optimization, and read-only instance.



features that are designed to improve functionality, performance, stability, and security, including:


Thread Pool
- This feature uses the **Listener-Worker model** to improve the connection performance of AliSQL.
- It optimizes the concurrency control for different types of operations based on their priorities.
- This allows `ApsaraDB for RDS` to ensure high performance when it processes a large number of concurrent requests.


Statement outline
- This feature uses **optimizer and index hints** to ensure the stability of `ApsaraDB for RDS` when SQL query plans change due to data update, index addition or deletion, or parameter adjustment.


Fast query cache
- developed by Alibaba Cloud based on the native MySQL query cache.
- uses a new design and implementation mechanism to increase the query performance of ApsaraDB for RDS.
- optimizes concurrency control, memory management, and caching.


Binlog in Redo
- synchronously writes binary logs to the redo log file when a transaction is committed.
- This reduces operations on disks and improves database performance.

Faster DDL
- This feature is developed by the `ApsaraDB for RDS` team.
- It fixes defects in the cache maintenance logic that is used to manage data definition language (DDL) operations.
- It also provides the optimized buffer pool management mechanism to reduce competition for locks that are triggered by DDL operations.
- This feature ensures the DDL operation performance of `ApsaraDB for RDS` when it processes a regular number of requests.



---

#### AliPG
Alibaba Cloud offers two PostgreSQL-compatible database services that run AliPG:
- ApsaraDB RDS
- ApsaraDB for MyBase.


AliPG is a unified database engine that is developed by Alibaba Cloud.
- Since the commercial rollout of AliPG in 2015, AliPG has been running stably for years and processed a large volume of workloads within Alibaba Group and for Alibaba Cloud customers.
- AliPG supports the following major PostgreSQL versions: 9.4, 10, 11, 12, 13 and 14.

AliPG has the following advantages over open source PostgreSQL:

Faster speed
- `Image recognition, face recognition, similarity-based retrieval, and similarity-based audience spotting`: Image recognition and vector similarity-based searches are tens of thousands of times faster on AliPG than on open source PostgreSQL
- `Real-time precision marketing (user selection)`: Marketing and user profiling in real time are thousands of times faster on AliPG than on open source PostgreSQL.
- The `GIS-based Mod operator` on AliPG processes mobile objects 50 times faster than the Mod operator on open source PostGIS.



Higher stability
- AliPG uses the Platform as a Service (PaaS) architecture.
- This architecture allows you to transform traditional software from license-based services to subscription-based services.
- You can manage a large amount of metadata, optimize connections, and efficiently isolate resources.
- In addition, each RDS instance supports tens of thousands of schemas.



Higher security
- AliPG is certified based on leading national and international security standards, which empowers enterprises to increase institutional security scores in the financing and listing phases.
- AliPG provides the following security enhancements:
  - Encrypts sensitive data that contains passwords. The sensitive data includes dynamic views, shared memory, the dblink plug-in, historical commands, and audit logs.
  - Fixes the function-related bugs that are found in open source PostgreSQL.
  - Supports fully encrypted databases. For more information, see Fully encrypted databases.
  - Supports the semi-synchronous mode. This mode allows you to specify one of the following protection levels for your RDS instance: maximum protection, maximum availability, and maximum performance. For more information, see Set the protection level of an ApsaraDB RDS for PostgreSQL instance.
  - Supports the failover slot feature. This feature prevents primary/secondary switchovers from affecting the reliability of logical replication. For more information, see Logical Replication Slot Failover.
  - Higher flexibility and controllability. For more information, see What is ApsaraDB for MyBase?
  - AliPG grants you the permissions to manage the operating systems on hosts in dedicated clusters.
  - AliPG allows you to customize overcommit ratios in the development, test, and staging environments. For example, you can specify 128 cores for a host that provides only 64 cores. This way, you can exclusively occupy resources in the production environment to reduce the overall costs.


---


### Cost effective and easy to use

It features cost-effectiveness, flexible billing, on-demand configuration changes, easy deployment, high compatibility, and simple operations and maintenance (O&M).


**Flexible billing**
- ApsaraDB RDS supports the `subscription` and `pay-as-you-go` billing methods.

- short-term use, recommend `pay-as-you-go` billing method.
  - A pay-as-you-go instance is charged per hour based on your actual resource usage.
  - If you no longer need your pay-as-you-go instance, you can release it to reduce costs.

- long-term use, recommend the `subscription` billing method.
  - You can receive larger discounts for longer subscription periods.


**On-demand configuration changes**
- You can purchase an RDS instance with low specifications that can meet your business requirements As the database workloads and data storage increase, you can upgrade the instance specifications.
- If your business scale becomes small again, you can downgrade the instance specifications to reduce costs.


---

### High performance

Parameter optimization
- All parameters that are used in ApsaraDB RDS have been tested and optimized over years of production practices that are conducted by a team of experienced database administrators (DBAs) from Alibaba Cloud.
- These DBAs have continued to optimize each ApsaraDB RDS instance throughout the lifecycle of the instance to ensure that the instance runs at its optimal configuration.


SQL optimization
- ApsaraDB RDS identifies `SQL statements` that are run at low speeds and provides recommendations that help you optimize your business code.


High-end hardware
- All server hardware that is used by ApsaraDB RDS has passed the tests of multiple concerned parties.
- This ensures that ApsaraDB RDS can deliver optimal performance and high stability.


High-speed access
- If an ApsaraDB RDS instance is used with an Elastic Compute Service (ECS) instance that resides in the same region as the RDS instance, these instances can communicate over an internal network to shorten response time and reduce Internet traffic consumption.



---




### High availability and disaster tolerance

Backup and recovery
- RDS supports automatic and manual backups
- set the automatic backup frequency or manually create backups at any time.
- RDS supports data recovery by time or backup set.
- You can restore data of any point in time within the log retention period to a new instance, verify the data, and then transfer the data to the original instance.


Disaster tolerance
- ApsaraDB RDS offers four editions: Basic Edition, High-availability Edition, Cluster Edition, and Enterprise Edition.



---


### High security

DDoS protection
- If DDoS attacks are detected, the security system of RDS enables **traffic cleaning** first.
- If traffic cleaning fails or the attacks reach the blackhole threshold, **blackhole filtering** is triggered.
- Note We recommend that you access RDS instances through the intranet to prevent DDoS attacks.


Access control
- IP addresses can access your RDS instance only after you add them to the whitelists of the RDS instance.
- IP addresses that are not in the whitelists cannot access the RDS instance.
- Each account can only view and operate its own databases.


System security
- RDS is protected by multiple **firewall** layers that block various network attacks to guarantee data security.
- Direct logon to the RDS server is not allowed.
- Only the ports required by certain database services are open.
- The RDS server cannot initiate an external connection. It can only accept access requests.


Professional security team
- Aliabab Cloud security team is responsible for guaranteeing the security of RDS.


---


database engine
- ApsaraDB RDS supports the MySQL, SQL Server, PostgreSQL, and MariaDB TX database engines.

network type
- You can create an RDS instance in the classic network or in a virtual private cloud (VPC).
- A VPC is an isolated virtual network that is deployed on Alibaba Cloud.
- VPCs provide higher security than the classic network.
- We recommend that you create RDS instances in VPCs.

edition
- ApsaraDB RDS provides the
  - Basic Edition,
  - High-availability Edition,
  - Cluster Edition,
  - and Enterprise Edition.



instance family
- ApsaraDB RDS provides the shared, general-purpose, and dedicated instance families.



storage type
- ApsaraDB RDS supports
  - local SSDs,
  - standard SSDs,
  - and enhanced SSDs (ESSDs).





Related services
- Elastic Compute Service (ECS):
  - ECS provides high-performance cloud servers.
  - If an ECS instance and an RDS instance reside in the same region, these instances can communicate over an internal network to ensure the optimal performance of the RDS instance.
  - The use of ECS and ApsaraDB RDS is a typical service access architecture.


- ApsaraDB for Redis:
  - ApsaraDB for Redis is an `in-memory database service` that persists data on disks.
  - You can use `ECS` in combination with `ApsaraDB RDS` and `ApsaraDB for Redis` to process a large number of read requests within a short period of time.

- ApsaraDB for MongoDB:
  - ApsaraDB for MongoDB is a `stable, reliable, and scalable database service` that is compatible with the MongoDB protocol.
  - store structured data in ApsaraDB RDS
  - and unstructured data in ApsaraDB for MongoDB




- MaxCompute:
  - MaxCompute, ODPS
  - a fully hosted data warehousing solution that can process terabytes or petabytes of data at high speeds.
  - MaxCompute provides a complete suite of data import solutions and a variety of distributed computing models.
  - You can use these solutions and models to import data from RDS instances into MaxCompute.
  - Then, you can use MaxCompute to process large amounts of data.



- Data Transmission Service (DTS):
  - DTS is used to migrate data from on-premises databases to RDS instances and migrate data between RDS instances for disaster recovery.

- Object Storage Service (OSS):
  - OSS is a secure, cost-effective, and highly reliable cloud storage solution that allows you to store large amounts of data on the cloud.
