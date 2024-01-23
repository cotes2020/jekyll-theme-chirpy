---
title: GCP - Big Data
date: 2021-01-01 11:11:11 -0400
categories: [01GCP]
tags: [GCP]
toc: true
image:
---

- [Big Data](#big-data)
  - [Google Cloud Big Data Platform](#google-cloud-big-data-platform)
  - [Data](#data)
    - [Ingest](#ingest)
      - [Cloud Pub/Sub](#cloud-pubsub)
    - [Store](#store)
      - [Cloud BigQuery](#cloud-bigquery)
    - [Process](#process)
      - [Cloud Dataproc](#cloud-dataproc)
      - [cloud Dataflow](#cloud-dataflow)
    - [Visualize](#visualize)
  - [Pipieline](#pipieline)
    - [cloud composer](#cloud-composer)
    - [data fustion](#data-fustion)
    - [Cloud Datalab](#cloud-datalab)

---

# Big Data

---


![Screen Shot 2022-08-16 at 23.29.26](https://i.imgur.com/cuDT8ox.jpg)


## Google Cloud Big Data Platform

- help transform the business and user experiences with meaningful data insights.
- an Integrated Serverless Platform.
  - Serverless, no worry about provisioning Compute Instances to run the jobs.
    - The services are fully managed
  - pay only for the resources you consume.
  - The platform is integrated
    - so GCP data services work together to help create custom solutions.

- Apache Hadoop
  - an open source framework for big data.
  - It is based on the <font color=red> MapReduce programming model </font> which Google invented and published.
    - <font color=red> "Map function" </font>
      - runs in parallel with a massive dataset to produce intermediate results.
    - <font color=red> "Reduce function" </font>
      - builds a final result set based on all those intermediate results.
  - The term "Hadoop" is often used informally to encompass Apache Hadoop itself, and related projects such as Apache Spark, Apache Pig, and Apache Hive.


![Screen Shot 2021-02-09 at 11.49.45](https://i.imgur.com/CQv9jnH.png)

![1_OaVwGy1MNotrx3Oi6a9Zeg](https://i.imgur.com/vIj7uX8.png)


![Selection_004](https://i.imgur.com/sYkUkOX.png)



## Data


---

### Ingest

![Screen Shot 2022-08-16 at 23.30.02](https://i.imgur.com/8qhlqlm.png)


#### Cloud Pub/Sub

- Cloud publishers/subscribers

- simple, reliable, scalable foundation for stream analytics.
  - foundation for Dataflow streaming

- <font color=red> Analyzing streaming data </font>

- <font color=red> use for IoT applications </font>

- <font color=red> decoupled systems </font> , and scale independently.
  - offers on-demand scalability to one million messages per second and beyond.

- support many-to-many asynchronous messaging service.
  - Push notifications for cloud-based applications
  - let independent applications send and receive messages.
  - Applications can publish messages in Pub/Sub
  - and one or more subscribers receive them.

- builds on the same technology Google uses internally.
  - connect applications across Google cloud platform
  - push/pull between Compute Engine and App
  - works well with applications built on GCP's Compute Platforms.
  - when analyzing streaming data, Cloud Dataflow is a natural pairing with Pub/Sub.

- Receiving messages doesn't have to be synchronous.
  - That's what makes Pub/Sub great for decoupling systems.
  - It's designed to provide "at least once" delivery at low latency.
    - a small chance some messages might be delivered more than once.
  - keep this in mind when you write your application.

- You just choose the quota you want.

- an important building block for data ingestion in Dataflow
  - for applications where data arrives at high and unpredictable rates,
  - like Internet of Things systems, marketing analytics

- application components make push/pull subsciptions to topics
  - configure subscribers to receive messages on a push or pull basis.
  - get notified when new messages arrive for them
  - or check for new messages at intervals.

- includes supports for offline consumers


---



### Store

![Screen Shot 2022-08-16 at 23.30.22](https://i.imgur.com/VBtAA1Q.jpg)


#### Cloud BigQuery

- if data needs to run more in the way of exploring a vast sea of data.
  - instead of a dynamic pipeline

- fully-managed, petabyte-scale, low-cost <font color=red> data analytics warehouse </font>
  - no infrastructure to manage
  - no cluster maintencance is required
  - focus on analyze data to find meaningful insights by familiar SQL

- do <font color=red> ad-hoc SQL queries on massive data set </font>
  - provide near real-time interactive analysis of massive datasets (hundreds of TBs) using SQL syntax (SQL 2011)

- used by all types of organizations
  - smaller organizations, Big Query's free monthly quotas,
  - bigger organizations like its seamless scale,
    - it's available 99.9 percent service level agreement.

- get data into BigQuery.
  - load it from cloud storage or cloud data store,
  - or stream it into BigQuery at up to 100,000 rows per second.

- process data
  - SQL queries
    - run super-fast SQL queries against multiple terabytes of data in seconds
    - using the processing power of Google's infrastructure.
  - or easily read and write data in BigQuery via Cloud Dataflow, Hadoop, and Spark.


- Google's infrastructure is global and so is BigQuery.
  - can specify the region where the data will be kept.
  - example
  - to keep data in Europe
    - don't have to set up a cluster in Europe.
    - Just specify the EU location where you create your data set.
  - US and Asia locations are also available.

- <font color=red> pay-as-you-go model </font>
  - separates storage and computation with a terabit network in between
  - pay for your data storage separately from queries.
  - pay for queries only when they are actually running.

- have full control over who has access to the data stored in BigQuery,
  - including sharing data sets with people in different projects.
  - If you share data sets that won't impact your cost or performance.
    - People you share with pay for their own queries, not you.

- Long-term storage pricing is an automatic discount for data residing in BigQuery for extended periods of time.
  - data reaches 90 days in BigQuery, auto drop the price of storage.



---


### Process

![Screen Shot 2022-08-16 at 23.30.50](https://i.imgur.com/1nNvpeN.jpg)


#### Cloud Dataproc


> Hadoop jobs Running on-premises
> - requires a capital hardware investment.


Running Hadoop jobs in <font color=red> Cloud Dataproc </font>

- <font color=red> migrate on=permises Hadoop jobs to cloud </font>
  - a fast, easy, managed way to run and manage `Hadoop, MapReduce, Spark, Hive service, and Pig` on Google Cloud Platform.

- Data mining and analysis in <font color=red> datasets of known size </font>

- <font color=red> create clusters in 90 sec or less </font>
  - just need to request a Hadoop cluster.
  - It will be built in 90 seconds or less
    - on top of Compute Engine virtual machines whose number and type you control.

- <font color=red> Scale clusters even when jobs are running </font>
  - need more or less processing power while the cluster is running, scale it up or down.
  - use the default configuration for the Hadoop software in the cluster or customize it.
  - monitor the cluster using Stackdriver.

- <font color=red> save money with preemptible Compute Engine instances </font>
  - <font color=blue> only pay for hardware resources used during the life of the cluster </font>
    - the costs of the Compute Engine instances isn't the only component of the cost of a Dataproc cluster, but it's a significant one.
    - Although the rate for pricing is based on the hour,
      - Cloud Dataproc is billed by the second.
      - billed in one-second clock-time increments, subject to a one minute minimum billing.
    - when done with the cluster, delete it, and billing stops.

  - <font color=blue> more agile use of resources </font> than on-premise hardware assets.

  - let Cloud Dataproc use <font color=blue> preemptible Compute Engine instances </font> for the batch processing.
    - make sure that the jobs can be restarted cleanly, if they're terminated, and you get a significant break in the cost of the instances.
    - preemptible instances were around 80 percent cheaper.


Once the data is in a cluster,
- use Spark and Spark SQL to do <font color=red> data mining </font>

- use MLib, Apache Spark's machine learning libraries to <font color=red> discover patterns through machine learning </font>



---


#### cloud Dataflow

<img alt="pic" src="pic" src="https://i.imgur.com/jcqmkNv.png" width="200">

| term          | cloud Dataproc                    | Cloud Dataflow                                    |
| ------------- | --------------------------------- | ------------------------------------------------- |
| data size     | for known size data set           | unpredictable size or rate                        |
| manage or not | manage your cluster size yourself | a unified programming model and a managed service |
| dataflow      | \                                 | if data shows up in real time                     |


Dataflow
- both a unified programming model and a managed service

- develop and execute a big range of data processing patterns
  - extract, transform, and load batch computation and continuous computation.

- write code once and get batch an streaming
  - Transform-based programming model
  - use Dataflow to build data pipelines.
  - the same pipelines work for both batch and streaming data.

- no need to spin up a cluster or to size instances.

- fully automates the management of whatever processing resources are required.
  - frees you from operational tasks
    - like resource management and performance optimization.

![Screen Shot 2021-02-09 at 12.11.09](https://i.imgur.com/J5huoyB.png)

- example,
  - Dataflow pipeline reads data from a big query table, the Source,
  - processes it in a variety of ways, the Transforms,
  - and writes its output to a cloud storage, the Sink.
  - Some of those transforms you see here are map operations and some are reduce operations.

pipelines
- can build really expressive pipelines.

- Each step in the pipeline is elastically scaled.
  - no need to launch and manage a cluster.
  - the service provides all resources on demand.

- It has automated and optimized worked partitioning built in
  - can dynamically rebalance lagging work.
  - reduces the need to worry about hotkeys.
  - situations where disproportionately large chunks of your input get mapped to the same cluster.

- use cases.
  - a general purpose <font color=red> ETL (extract/transform/load) tool </font>
  - a data analysis engine
    - batch computation or continuous computation using streaming.
    - handy in things like
    - fraud detection and financial services,
    - IoT analytics and manufacturing,
    - healthcare and logistics and click stream,
    - point of sale and segmentation analysis in retail.

  - <font color=red> orchestration </font>
    - create pipeline that coordinates multiple services even external services.
    - can be used in real time applications such as personalizing gaming user experiences.

- integrates with GCP services like CLoud storage, cloud Pub/Sub, BigQuery, and Bigtable
  - Open source Java and Python SDKs

---



### Visualize

![Screen Shot 2022-08-16 at 23.31.05](https://i.imgur.com/6wiJxJw.jpg)

---








## Pipieline



### cloud composer

![Screen Shot 2022-08-16 at 23.31.31](https://i.imgur.com/WCwSReV.jpg)


### data fustion

![Screen Shot 2022-08-16 at 23.31.56](https://i.imgur.com/ZG0BuZa.jpg)





---

### Cloud Datalab

> Scientists have long used lab notebooks to organize their thoughts and explore their data.

- For data science, the lab notebook metaphor works really well
  - because it feels natural to intersperse data analysis with comments about their results.

- A popular environment for hosting those is Project Jupyter.
  - create and maintain web-based notebooks containing Python code
  - and run that code interactively and view the results.

Cloud Datalab

- offers interactive data exploration
  - interactive tool for large-scale data exploration, transformation, analysis, and visulization

- integrated, open sourse
  - build on Jupyter (formerly IPython)

- It's integrated with BigQuery, Compute Engine, and Cloud Storage
  - so access data doesn't run into authentication hassles.
  - analyze data in BigQuery, Compute Engine, and Cloud Storage using python, SQL, and Javascript
  - easily deploy models to BigQuery

- Cloud Datalab takes the management work out of this natural technique.
  - It runs in a Compute Engine virtual machine.

- To get started
  - specify the virtual machine type
  - what GCP region it should run in.
  - When it launches
    - it presents an interactive Python environment
    - it orchestrates multiple GCP services automatically, so can focus on exploring the data.

- only pay for the resources you use.
  - no additional charge for Datalab itself.

- When you're up and running, visualize your data with Google Charts or map plot line and because there's a vibrant interactive Python community, you can learn from published notebooks.

- existing packages for statistics, machine learning, and so on.












.
