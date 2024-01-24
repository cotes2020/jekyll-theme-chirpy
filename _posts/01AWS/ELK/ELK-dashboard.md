


- [Visualizing AWS CloudTrail Events using Kibana](#visualizing-aws-cloudtrail-events-using-kibana)
  - [create chart](#create-chart)
    - [the number of API events logged by CloudTrail on the account in the last hour.](#the-number-of-api-events-logged-by-cloudtrail-on-the-account-in-the-last-hour)
    - [types of errorCode errors](#types-of-errorcode-errors)
  - [AWS Blog solution](#aws-blog-solution)
    - [Creating a CloudTrail trail](#creating-a-cloudtrail-trail)
    - [Creating an Amazon ES domain (Elasticsearch cluster)](#creating-an-amazon-es-domain-elasticsearch-cluster)
    - [Creating an Amazon Cognito user pool and identity pool](#creating-an-amazon-cognito-user-pool-and-identity-pool)
    - [Creating an EC2 instance](#creating-an-ec2-instance)
    - [Configuring a proxy](#configuring-a-proxy)
    - [Creating an SSH tunnel](#creating-an-ssh-tunnel)
    - [Streaming CloudWatch Logs data to Amazon ES](#streaming-cloudwatch-logs-data-to-amazon-es)
    - [Visualizing the CloudTrail events using Kibana](#visualizing-the-cloudtrail-events-using-kibana)



# Visualizing AWS CloudTrail Events using Kibana

visualize [AWS CloudTrail](https://aws.amazon.com/cloudtrail) events, near real time, using Kibana.
- CloudTrail
  - enables governance, compliance, operational auditing, and risk auditing of the AWS account.
  - log, continuously monitor, and retain account activity related to actions across the AWS infrastructure.
- use an [ELK](https://aws.amazon.com/elasticsearch-service/the-elk-stack/) (Elasticsearch, Logstash, Kibana) stack to
  - aggregate logs from all the systems and applications,
  - analyze these logs,
  - create visualizations for application and infrastructure monitoring.
  - faster troubleshooting and security analytics.
- Kibana
  - popular open-source visualization tool designed to work with Elasticsearch.
  - Amazon ES provides an installation of Kibana with every Amazon ES domain.
  - Kibana dashboard: continuously monitor the CloudTrail logs helps simplify operational analysis and troubleshooting compliance issues.


---


## create chart


---


### the number of API events logged by CloudTrail on the account in the last hour.

1. bar chart.
2. For **aggregation** type in the Y-axis, choose **Count**.
3. For **Aggregation** type in the X-axis, choose **Terms**.
4. For **Field**, search for and choose _eventName.keyword_.
5. For **Order by**, choose **Metric**: **Number of Events**.

![pic](https://d2908q01vomqb2.cloudfront.net/972a67c48192728a34979d9a35164c1295401b71/2020/08/11/dashboard-2-1024x525.png)


---

### types of errorCode errors

create a time series graph to check for the different _errorCode_ errors that CloudTrail detects in the AWS account.

1. choose **TSVP**.
4. For **Group by**, choose **Terms**.
5. For **By**, choose **errorCode.keyword**.

The following screenshot shows a graph with the occurrences of ResourceNotFound errors in the last hour.

![pic](https://d2908q01vomqb2.cloudfront.net/972a67c48192728a34979d9a35164c1295401b71/2020/08/11/dashboard-3-1024x525.png)




---

## AWS Blog solution

In this solution:
- replace the Logstash with AWS native solutions to stream CloudTrail events to an [Amazon Elasticsearch](https://aws.amazon.com/elasticsearch-service) (Amazon ES) domain.
- Because the cost of the Amazon ES cluster increases as log data grows, you may want to use cheaper storage tiers within the Amazon ES leveraging the [UltraWarm](https://docs.aws.amazon.com/elasticsearch-service/latest/developerguide/ultrawarm.html) feature.


**Solution Overview**
- got the <font color=red> CloudTrail events </font>
- send the CloudTrail events to <font color=red> Amazon CloudWatch Logs </font>
- CloudWatch Logs trigger <font color=red> Lambda function </font> to send the Trail Events to an Amazon Elasticsearch Index.
- stream the logs to an <font color=red> Amazon ES cluster </font> in near-real time, through a CloudWatch Logs subscription.
- Kibana create the near real-time dashboard
- access the <font color=red> Kibana endpoint </font> to visualize the data.

![pic](https://d2908q01vomqb2.cloudfront.net/972a67c48192728a34979d9a35164c1295401b71/2020/08/11/Cloudtrail_Kibana.png)

**Prerequisites**
- An AWS account
- An IAM user with access to AWS resources used in this solution

**High-level approach**
- CloudTrail is enabled on the AWS account when you create it.
- use the **Event history** page on the CloudTrail console
  - to view, search, download, archive, analyze, and respond
  - to account activity across the AWS infrastructure for the past 90 days.
  - This includes activity made through the [AWS Management Console](https://aws.amazon.com/console), [AWS CLI](https://aws.amazon.com/cli), AWS SDKs, and programmatically.

To implement this visualization solution using Kibana, you complete the following high-level steps:

1. Create a CloudTrail trail for an ongoing record of events in the AWS account.
2. Send the CloudTrail events to an CloudWatch Logs log group.
3. Configure the trail to send events to an Amazon ES domain in near-real time.
4. Create an Amazon ES domain to store the CloudTrail logs, which contain trail events to Amazon ES
5. Visualize the CloudTrail events using Kibana

---

### Creating a CloudTrail trail


1. CloudTrail trail > S3 bucket
   - create a CloudTrail trail: provide the following information:
     - **Trail name** – _myblog-all-events_ (or name of the choice)
     - **S3 bucket for storing logs** – _blog-cloudtrail-events_ (or S3 bucket of the choice)
     - **SSE-KMS encryption** – Use an existing key or create one based on the needs

   - can create up to five trails for a Region.
   - After create a trail, CloudTrail automatically starts logging API calls and related events in the account to the Amazon S3 bucket that you specify.

   - To stop logging, turn off logging for the trail or delete it. 
   - set up a trail that delivers a [single copy of management events in each Region free of charge](https://aws.amazon.com/cloudtrail/pricing/).


2. CloudTrail events > CloudWatch log.
   - [Sending Events to CloudWatch Logs](https://docs.aws.amazon.com/awscloudtrail/latest/userguide/send-cloudtrail-events-to-cloudwatch-logs.html).
   - Specify a log group name, which to use on the CloudWatch console to send the trail events to Amazon ES.
     - We use the name _CloudTrail/MyBlogLogGroup_.

- CloudTrail Insights helps to identify and respond to unusual activity associated with _write_ API calls.
  - By default, AWS CloudTrail trails log all management events, and don’t include data or CloudTrail Insights events.
  - Management events: capture management operations,
  - data events: the resource operations performed on or within a resource


---

### Creating an Amazon ES domain (Elasticsearch cluster)

Placing the Amazon ES domain within a VPC
- provides inherent strong layer of security
- recommended for production clusters.


search the Elasticsearch index using Kibana
- configure an SSH tunnel to access Kibana from outside the VPC.
- can also use an NGINX proxy or client VPN to access Kibana from outside a VPC, along with [Amazon Cognito](https://aws.amazon.com/cognito) authentication.
  1. Create an Amazon Cognito user pool and identity pool.
  2. Create an EC2 instance in a public subnet in the same VPC that the Elasticsearch domain is in.
  3. Use a browser add-on, such as FoxyProxy, to configure a SOCKS proxy.
  4. Create an SSH tunnel from the local machine to the EC2 instance.
  5. Optionally, use the [elasticsearch-in-vpc-only](https://docs.aws.amazon.com/config/latest/developerguide/elasticsearch-in-vpc-only.html) AWS Config Config Rule to determine if Elasticsearch is mistakenly accessible from outside the VPC.

---

### Creating an Amazon Cognito user pool and identity pool

create the Amazon ES production cluster beforehand and modify the access policy,

When creating the Amazon ES domain, complete the following steps:

1. Enter a name for the Amazon ES domain.
2. Select three Availability Zones.
3. Choose the instance types.
4. Choose the number of nodes (a multiple of the selected Availability Zones).
5. Provide the storage requirements.
6. Provide the dedicated primary node (instance type and number).
7. Keep **UltraWarm** unselected.
8. In **Network configuration**:
    1. Select VPC access.
    2. Select the VPC where you want to create the cluster and associated subnets.
    3. Select the security group to use for the Amazon ES domain
9. if want to use fine-grained access control, powered by Open Distro for Elasticsearch.
    - [Fine-Grained Access Control in Amazon Elasticsearch Service](https://docs.aws.amazon.com/elasticsearch-service/latest/developerguide/fgac.html).
10. Enable Amazon Cognito authentication and choose the user pool and identity pool
    - ![pic](https://d2908q01vomqb2.cloudfront.net/972a67c48192728a34979d9a35164c1295401b71/2020/08/11/Cognito_Authorization-1024x463.png)
11. Amazon ES domain access policy: access policy similar to the following and update the placeholders:

        {
          "Version": "2012-10-17",
          "Statement": [
            {
              "Effect": "Allow",
              "Principal": {
                "AWS": "arn:aws:sts::<AWS Account ID>:assumed-role/Cognito_<Cognito Identity Pool Name>Auth_Role/CognitoIdentityCredentials"
              },
              "Action": "es:*",
              "Resource": "arn:aws:es:us-west-2: <AWS Account ID>:domain/<ES Cluster Name>/*"
            }
          ]
        }


12. Select all the encryption options.
13. Confirm all the configurations and create the domain.

---

### Creating an EC2 instance

the SSH tunnel
- create an EC2 instance in the same VPC, where you created the Amazon ES domain. 
- configure the security group rules for the EC2 instance.
  - **Create an EC2 instance and configure security group rules** section in [How can I use an SSH tunnel to access Kibana from outside of a VPC with Amazon Cognito authentication?](https://aws.amazon.com/premiumsupport/knowledge-center/kibana-outside-vpc-ssh-elasticsearch/)


---

### Configuring a proxy

To access the Kibana dashboard
- configure a proxy.
- Configure the SOCKS proxy section in [How can I use an SSH tunnel to access Kibana from outside of a VPC with Amazon Cognito authentication?](https://aws.amazon.com/premiumsupport/knowledge-center/kibana-outside-vpc-ssh-elasticsearch/)
- use an [NGINX proxy](https://aws.amazon.com/premiumsupport/knowledge-center/kibana-outside-vpc-nginx-elasticsearch/) setting or a [Client VPN](https://docs.aws.amazon.com/vpn/latest/clientvpn-user/user-getting-started.html) to establish this secure connection.


---


### Creating an SSH tunnel
After you complete these steps, create an SSH tunnel to access the Kibana dashboard from the local machine (outside VPC).

1. Run the following command from the local machine that you use to access the Kibana dashboard.
   - Replace _mykeypair.pem_ with the key pair for the EC2 instance
   - replace change _public\_dns\_name_ with the public DNS of the _tunnel\_ec2_ EC2 instance.                         
   - `ssh -i "mykeypair.pem" ec2-user@public_dns_name -ND 8157`
2. Enter the Kibana endpoint in the browser.
   - The Amazon Cognito login page for Kibana appears.
3. Use the Amazon Cognito user ID and password to log in to the dashboard.


---


### Streaming CloudWatch Logs data to Amazon ES

- the Amazon ES cluster is ready to use
- configure a CloudWatch Logs log group
  - stream the data it receives to the Amazon ES cluster in near-real time through a CloudWatch Logs subscription.
  - [Streaming CloudWatch Logs Data to Amazon Elasticsearch Service](https://docs.aws.amazon.com/AmazonCloudWatch/latest/logs/CWL_ES_Stream.html).
- For **Log format**, choose **AWS CloudTrail**.
- When the sample testing is complete, start streaming the events to Amazon ES.
- You get a notification when successfully start streaming.
  - ![pic](https://d2908q01vomqb2.cloudfront.net/972a67c48192728a34979d9a35164c1295401b71/2020/08/11/ES-Start-Streaming.png)

- When the subscription filter starts streaming the data to the Amazon ES domain, you get a confirmation message
  - ![pic](https://d2908q01vomqb2.cloudfront.net/972a67c48192728a34979d9a35164c1295401b71/2020/08/11/ES-Start-Streaming-Success-1024x55.png)


- On the Amazon ES console, after a few minutes, activities shows in the **Key performance indicators** section.
  - The following screenshot shows an increase in **Indexing rate**.
  - ![pic](https://d2908q01vomqb2.cloudfront.net/972a67c48192728a34979d9a35164c1295401b71/2020/08/11/Key-Performance-Indicator-1024x526.png)

  - You can also see an increase in the Searchable documents counts.
  - ![pic](https://d2908q01vomqb2.cloudfront.net/972a67c48192728a34979d9a35164c1295401b71/2020/08/11/Searchable_Document-1024x398.png)


---

### Visualizing the CloudTrail events using Kibana


access the Kibana endpoint shown in the Amazon ES cluster overview and create a dashboard.

1. On the Amazon ES console, choose the domain.
2. On the Overview page, copy the Kibana endpoint.
3. In the web browser, choose Use proxy Kibana Proxy for all URLs to enable FoxyProxy.
4. When prompted, enter the Amazon Cognito user name and password to log in to Kibana.
5. On the Add Data to Kibana page
   - choose Use Elasticsearch data,
   - and connect to the Amazon ES index.
   - ![pic](https://d2908q01vomqb2.cloudfront.net/972a67c48192728a34979d9a35164c1295401b71/2020/08/11/Kibana-Fornt-Page.png)
6. When you’re connected, enter _cwl-\*_ as the index pattern.
7. Enter _eventTime_ as the time filter field.

8. now go to the **Discover** tab
   - to add specific fields as filters and search for them. In the following screenshot, I selected fields specific to error events logged in CloudTrail to find the issues.

![pic](https://d2908q01vomqb2.cloudfront.net/972a67c48192728a34979d9a35164c1295401b71/2020/08/11/dashboard-1-1024x487.png)
