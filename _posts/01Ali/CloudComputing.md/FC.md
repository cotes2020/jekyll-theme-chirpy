


---

- [Ali - Function Compute](#ali---function-compute)
- [basic](#basic)
- [use case](#use-case)
  - [Function Compute, Tablestore, and API Gateway](#function-compute-tablestore-and-api-gateway)
  - [Function Compute, Message Queue for Apache RocketMQ and OSS](#function-compute-message-queue-for-apache-rocketmq-and-oss)
  - [Function Compute](#function-compute)
  - [Alibaba Cloud CDN, Function Compute, OSS](#alibaba-cloud-cdn-function-compute-oss)


---

## Ali - Function Compute


---


## basic

- fully-managed, event-driven computing service for serverless applications
- can focus on writing and uploading code, without the need to procure and manage infrastructure resources such as servers. Function Compute prepares computing resources for you, runs code in an elastic and reliable way, and provides features such as log query, performance monitoring, and alert.

With Function Compute, you can quickly create any type of applications and services and only pay for the resources actually consumed when you run your code.


![p96487](https://i.imgur.com/126wddr.png)



- can run many different types of applications, including API gateways, data lake analyses, log stores and backups, web crawlers, and image recognition applications
- With Function Compute, developing is convenient and reliable.
- supports many different programming languages including Java, Python, PHP, and NodeJS.

- provides real-time auto scaling and dynamic load balancing for managing heavy traffic bursts within millisecond timeframes.
- Its compute resources ensure that code is flexible and reliable. Furthermore,
- offers a Pay-As-You-Go option. No fee is incurred if the code doesn’t run. the code run duration is measured in milliseconds.



---

1. Create a service.
2. Create a function, write the code of the function, and then deploy the code to the function.
3. Trigger the function.
4. View the execution logs of the function.
5. View service monitoring data.


![p96542](https://i.imgur.com/ksgzB3Y.png)











---

## use case

Some common business scenarios:
- analysis and management of media assets, such as integrating a range of services that run an elastic and highly available backend video system.
- have a serverless backend that triggers Function Compute code which then renders dynamic and static webpages housed in Alibaba Cloud’s Object Storage Service.
- implementing Function Compute to manage real time IoT message processing and the monitoring of data streams.


---


### Function Compute, Tablestore, and API Gateway


**Customer pain points**
- Burst traffic: Century Mart has a large number of followers. -large number of members to grab coupons at the same time. The burst traffic was so high that its service was interrupted.
- Excessive server demand: Management services of supermarket members are intensive. A large number of servers are used to support regular online shopping promotions, but business is greatly affected during peak hours.
- System overload: Large promotion activities may encounter tedious and complicated work of server management and system overload caused by the fluctuating number of users. Century Mart must find an elastic and stable system architecture to support such activities.


**Solution**
- The serverless architecture features `fast scaling, elasticity, and high availability` and is able to cope with burst traffic.
  - Tablestore replaces traditional relational databases.
  - Function Compute reads and writes data from and to Tablestore and efficiently returns processing results to frontend users.
- The new solution of using `Function Compute, Tablestore, and API Gateway` greatly simplifies O&M compared with the traditional solution of temporarily adding servers.
  - **Function Compute**: supports auto scaling and can dynamically allocate runtime environments based on the number of requests received. Its deployment is simple.
  - **Tablestore**: offers faster access and higher throughput, which eliminates the need to add additional servers.
  - **API Gateway**: allows you to control access and export API documentation in a convenient way.
- Benefits of Alibaba Cloud services and the serverless architecture
  - Increase revenue: reach a new high of 550 million transactions
  - Reduce workload:
    - Function Compute reduces the workload of technical engineers.
    - Function Compute is a fully managed and event-driven computing service. You can write and upload code without the need to manage infrastructure resources such as servers.
    - Function Compute prepares computing resources and provides features such as log query, performance monitoring, and alerting.
    - Migrating all data and business to Alibaba Cloud can greatly reduce the stress and workload of users. If Century Mart does not use Function Compute but only deploy more servers to support large amounts of traffic and business during Double 11, Century Mart will not be able to ensure the normal operation of the activities. Alibaba Cloud resolves the scale-out issues, which greatly improves the data storage capacity.
  - Reduce the dependency on technical engineers: Alibaba Cloud provides cutting edge technologies. Developers in Century Mart do not need to study algorithms, but need only to learn how to use relevant tools of Alibaba Cloud. This reduces research and development investment and costs.


---

### Function Compute, Message Queue for Apache RocketMQ and OSS

**Customer pain points and requirements**
- JL-Tour receives more than ten million pieces of data updates from more than 600,000 hotels every day. These data updates require `high concurrency and feature short validity period`. This poses great pressure on the existing system of JL-Tour in terms of instantaneous concurrent message processing.
- JL-Tour needed an advanced system that can provide the following features and reduce the pressure of concurrent message processing:
  - **Concurrent processing capacity**: The system must concurrently process a maximum of 100,000 messages.
  - **Scalable processing capacity**: The system must automatically scale in or out within milliseconds based on the number of messages to be processed. The cost of use is charged based on the actual resources required.
  - **Support for multiple data sources**, such as OSS and messages.
  - **Support for multiple programming languages**, such as Python, Go, and Java.
  - **O&M monitoring capabilities**: The system allows for rapid deployment and updates, monitors real-time resource usage, analyzes logs, and generates alerts.

**Solution**
- By taking the following advantages of `Function Compute, Message Queue for Apache RocketMQ and OSS`, JL-Tour meets its business requirements:
- Function Compute listens to a variety of data sources.
- It monitors and processes changes in the volume of business, and carries out adaptive scale-out and scale-in operations with efficiency.
- It monitors scale-out operations that are performed within milliseconds. This allowed JL-Tour to achieve linear growth in business capacity.
- Function Compute supports multiple programming languages for easy use.
- Function Compute supports easy deployment and can monitor real-time resource usage, analyze logs, and generate alerts.

**Benefits of Function Compute**
- Business stability: Function Compute automatically scales in or out based on the volume of business. By using Function Compute, JL-Tour can ensure business stability without allocating resources based on spikes in resource usage.
- Simple O&M: Function Compute allows JL-Tour to improve O&M capability by using a variety of tools. Function Compute eliminates the need of scalable resource management and also enables the O&M engineers of JL-Tour to work with higher efficiency.
- Cost efficiency: Function Compute adopts the pay-as-you-go billing method. JL-Tour can select proper specifications to ensure that its resources are efficiently utilized, without the need to pay for idle resources. The overall usage cost of JL-Tour is reduced.



---

### Function Compute


**Customer pain points and requirements**
- To develop real-time collaborative document editing services, the technical team of Shimo Docs has performed in-depth research into the Operational Transformation (OT) algorithm and made several key improvements. This includes the development of a two-dimensional document editing system, on top of the original one-dimensional system. This system eliminates merge conflicts that may occur when multiple users simultaneously edit a piece of text.
- However, Shimo Docs faces the following challenges:
  - Real-time edits by different users is resource intensive and often overloads the servers.
  - When a user types a word on the keyboard, a server saves the edit within a few milliseconds. However, when a large number of users are editing documents on Shimo Docs at the same time, data may be unevenly distributed within a short period.
  - Shimo Docs needed a scalable and highly available service to process conflicts that occur during document editing in real time. The service must have the following benefits:
  - The service is cost-efficient.
  - The service can enable Shimo Docs to smoothly process the surging loads that occur during peak hours to ensure synchronization responses within milliseconds.

**Solution**
- Function Compute:
  - dynamically allocates runtime environments and schedules computing resources in milliseconds based on the number of requests received.
  - minimizes latency when the workload is heavy and maintains high resource utilization when the workload is low.
  - save on costs because it needs only to pay for computing resources that are used when the code is running.
- Based on Function Compute, Shimo Docs uses the Alibaba Cloud serverless architecture to build a real-time document editing service. The logic of real-time document collaboration is implemented as a function. The intelligent scheduling system of Function Compute automatically allocates runtime environments to process the peak load of collaborative document editing. The scalability of Function Compute ensures that applications are stable and reliable during runtime.

**Benefits of Function Compute**
- Cost efficiency: Function Compute enables Shimo Docs to scale out computing resources within milliseconds when surging loads occur during peak hours. Compared with deploying physical servers in data centers, Function Compute improves resource utilization by reducing the waste of idle resources, and saves the server cost.
- Improved efficiency: By using Function Compute, Shimo Docs no longer needed to be concerned about the load balancing of CPU-intensive computing. The pace of project iteration is gradually accelerated, and engineers can focus on working with the product team to continuously expand the business value. Function Compute has helped Shimo Docs improve development efficiency and process stability.


---

### Alibaba Cloud CDN, Function Compute, OSS

**Customer pain points**
- rapid business growth poses a challenge to the original underlying system of Sina Weibo.
  - Traffic surge: high instantaneous peak traffic for a short period of time. The period of each peak interaction event is about 3 hours. Services such as star events and red envelope campaigns often encounter instantaneous peak traffic multiple times higher than normal values.
  - High peak-to-valley ratio: Social media is closely related to work and rest time of people. The average load for Sina Weibo changes over time, and peak traffic can exceed the lows by more than five times.
- Sina Weibo used the following traditional countermeasures to handle traffic surges:
  - Apply for sufficient equipment in advance to ensure redundancy.
  - Downgrade non-core services.

**Solution**
- The public cloud serverless architecture can be used to cope with explosive traffic.
- Function Compute:
  - dynamically allocates runtime environments and schedules computing resources in milliseconds based on the number of requests received.
  - minimizes latency when the workload is heavy and maintains high resource utilization when the workload is low.
  - reduce costs because it needs only to pay for computing resources that are used when the code is running.
  - integrates with Object Storage Service (OSS) and can process images stored in OSS in real time.
  - provides the auto scaling feature, event triggering mechanism, and pay-as-you-go billing method.
- Sina Weibo uses Function Compute to deploy image processing services. Sina Weibo stores images uploaded by users in OSS and defines functions to process the images in a personalized way. When a user requests an image from a client, the request is sent to Function Compute by Alibaba Cloud CDN. The relevant functions are used to download the original image from OSS, convert it into the expected image in real time based on the client type, and then return the final image.

**Benefits of Function Compute**
- Function Compute can scale computing resources in milliseconds to ensure that applications are stable when traffic bursts occur. User experience is not affected by the number of visits.
- Running the image processing service in Function Compute helps Sina Weibo continuously reduce costs. Sina Weibo no longer maintains large numbers of idle servers used to process surge traffic during peak hours. Without managing and maintaining infrastructure, developers can focus on cooperating with product teams and increasing business value.
- As the number of active users of Sina Weibo continues to increase and Sina Weibo continues to expand, Function Compute can automatically allocate more runtime environments to support continuous business growth of Sina Weibo.
.
