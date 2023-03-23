---
title: AWS - VPC - Log and monitor
date: 2020-07-18 11:11:11 -0400
categories: [01AWS, AWSNetwork]
tags: [AWS, Network, VPC]
toc: true
image:
---

- [Log and monitor for Amazon VPC](#log-and-monitor-for-amazon-vpc)
  - [Monitoring NAT gateways using Amazon CloudWatch](#monitoring-nat-gateways-using-amazon-cloudwatch)
  - [VPC Flow Logs](#vpc-flow-logs)
    - [Flow logs basics](#flow-logs-basics)
    - [create a flow log](#create-a-flow-log)
    - [Flow log records](#flow-log-records)
    - [Flow log limitations](#flow-log-limitations)
    - [Flow logs pricing](#flow-logs-pricing)
    - [price](#price)

---

# Log and monitor for Amazon VPC


use automated monitoring tools to watch components in VPC and report when something is wrong:
1. **Flow logs**:
   1. Flow logs capture information about the IP traffic going to and from network interfaces in VPC.
   2. create a flow log for a VPC, subnet, or individual network interface.
   3. Flow log data is published to CloudWatch Logs or Amazon S3
   4. help diagnose overly restrictive or overly permissive security group and network ACL rules.
2. **Monitoring NAT gateways**:
   1. monitor NAT gateway using CloudWatch, which collects information from NAT gateway and creates readable, near real-time metrics.
3. **Traffic Mirroring**
   1. provides deeper insight into the network traffic by allow analyze actual traffic content, including payload.
   2. Traffic Mirroring is targeted for the following types of cases:
      1. Analyzing the actual packets to perform a root-cause analysis on a performance issue
      2. Reverse-engineering a sophisticated network attack
      3. Detecting and stopping insider abuse or compromised workloads

---

## Monitoring NAT gateways using Amazon CloudWatch

monitor NAT gateway using CloudWatch
- collects information from NAT gateway and creates readable, near real-time metrics.
- use this information to monitor and troubleshoot NAT gateway.
- NAT gateway metric data is provided at 1-minute intervals, and statistics are recorded for a period of 15 months.

For more information about Amazon CloudWatch, see the Amazon CloudWatch User Guide. For more information about pricing, see Amazon CloudWatch Pricing.

---

## VPC Flow Logs

> to monitor VPC traffic

> to verify that the configured network access rules are working as expected

### Flow logs basics
VPC Flow Logs
1. <font color=red> captures accepted and rejected traffic flow information </font> that goes to and from all network interfaces in your VPC or in the selected resource.
   - <font color=blue> troubleshoot connectivity and security issues </font>
     - why specific traffic is not reaching an instance
   - <font color=blue> test network access rules </font>
     - diagnose overly restrictive security group rules.
   - <font color=blue> monitor traffic that reaching the instance </font>
     - Determining the `direction of the traffic` to and from the network interfaces
   - <font color=blue> detect and investigate security incidents </font>

2. can <font color=red> create alarms </font>
   - notify if certain types of traffic are detected
   - create metrics to identify trends and patterns.


3. <font color=red> enable flow log </font>
   - Flow logs can be enabled/created at the following levels:
     - VPC.
     - Subnet.
     - Network interface.
   - If create a flow log for a subnet / VPC, each network interface in the VPC or subnet is monitored.


4. Flow log data is recorded as <font color=red> flow log records </font>
   - log events consisting of fields that describe the traffic flow.
   - Flow log data is collected outside of the path of network traffic
     - therefore does not affect network throughput or latency.
     - create or delete flow logs without any risk of impact to network performance.


5. can be published to <font color=red> Amazon CloudWatch Logs or Amazon S3 </font>
   - CloudWatch Logs
     - Flow log data is published / stored to a <font color=blue> log group in CloudWatchLogs </font>
     - each network interface has a unique <font color=blue> log stream </font>
     - Log streams contain flow log records,
     - which are log events that consist of fields that describe the traffic for that network interface.

   - it can take several minutes to begin collecting and publishing data to the chosen destinations.
     - Flow logs <font color=blue> do not capture real-time log streams for network interfaces </font>
   - If you launch more instances into subnet after you've created a flow log for subnet or VPC, a new log stream (for CloudWatch Logs) or log file object (for Amazon S3) is created for each new network interface.
     - This occurs as soon as any network traffic is recorded for that network interface.


6. analyze flow logs with your own applications or with solutions from AWS Marketplace.

---

You cannot:
- can’t enable flow logs for VPC’s that are peered with your VPC unless the peer VPC is in your account.
- can’t tag a flow log.
- <font color=blue> can’t change the configuration </font> of a flow log after it’s been created.
  - need to delete and re-create


Not all traffic is monitored, e.g. the following traffic is excluded:
- Traffic that goes to Route53.
- Traffic generated for Windows license activation.
- Traffic to and from 169.254.169.254 (instance metadata).
- Traffic to and from 169.254.169.123 for the Amazon Time Sync Service.
- DHCP traffic.
- Traffic to the reserved IP address for the default VPC router.

---

### create a flow log

To create a flow log, you specify:
- `The resource` for which to create the flow log
- `The type` of traffic to capture (accepted traffic, rejected traffic, or all traffic)
- `The destinations` to publish the flow log data

![flow-logs-diagram](https://i.imgur.com/sBGBdBd.png)


You can create flow logs for network interfaces that are created by other AWS services, such as:
- Elastic Load Balancing
- Amazon RDS
- Amazon ElastiCache
- Amazon Redshift
- Amazon WorkSpaces
- NAT gateways
- Transit gateways
- Regardless of the type of network interface, you must use the `Amazon EC2 console` or the `Amazon EC2 API` to create a flow log for a network interface.

You can apply tags to flow logs.
- Each tag consists of a key and an optional value, both of which you define.
- Tags can help to organize flow logs

delete
- Deleting a flow log disables the flow log service for the resource, and no new flow log records are created or published to CloudWatch Logs / Amazon S3.
- Deleting the flow log does not delete any existing flow log records or log streams (for CloudWatch Logs) or log file objects (for Amazon S3) for a network interface.
- To delete an existing log stream, use the CloudWatch Logs console.
- To delete existing log file objects, use the Amazon S3 console.
- After you've deleted a flow log, it can take several minutes to stop collecting data.


---

### Flow log records
- A flow log record represents a network flow in VPC.
- By default, each record captures a network internet protocol (IP) traffic flow (characterized by a 5-tuple on a per network interface basis) that occurs within an aggregation interval, also referred to as a capture window.
- By default, the record includes values for the different components of the IP flow, including the source, destination, and protocol.
- When you create a flow log, you can use the default format for the flow log record, or you can specify a custom format.
  - Topics
  - Aggregation interval
  - Default format
  - Custom format
  - Available fields

**Aggregation interval**
- the period of time during which a particular flow is captured and aggregated into a flow log record.
  - By default, the maximum is 10min.
  - when you create a flow log, you can optionally specify a maximum aggregation interval of 1 minute.
  - Flow logs with a maximum aggregation interval of 1 minute produce a higher volume of flow log records than flow logs with a maximum aggregation interval of 10 minutes.
- When a network interface is attached to a Nitro-based instance, the aggregation interval is always 1 minute or less, regardless of the specified maximum aggregation interval.
- After data is captured within an aggregation interval, it takes additional time to process and publish the data to CloudWatch Logs or Amazon S3.
  - around 5 min to publish to CloudWatch Logs,
  - around 10 min to publish to Amazon S3.
  - The flow logs service delivers within this additional time in a best effort manner. In some cases, logs might be delayed beyond the 5 to 10 minutes additional time mentioned previously.

**Default format**
- By default, the log line format for a flow log record is a space-separated string that has the following set of fields in the following order.
  - `<version> <account-id> <interface-id> <srcaddr> <dstaddr> <srcport> <dstport> <protocol> <packets> <bytes> <start> <end> <action> <log-status>`
  - The default format captures only a subset of all of the available fields for a flow log record.
  - To capture all available fields or a different subset of fields, specify a `custom format`.
  - cannot customize or change the default format.


**Custom format**
- specify a custom format for the flow log record.
- This enables you to create flow logs that are specific to needs and to omit fields that are not relevant to you.
- reduce the need for separate processes to extract specific information from published flow logs. You can specify any number of the available flow log fields, but you must specify at least one.


**Available fields**
- The following table describes all of the available fields for a flow log record. The Version column indicates the VPC Flow Logs version in which the field was introduced.

Field	| Description	| Version
---|---|---
version | The VPC Flow Logs version. If you use the default format, the version is 2. If you use a custom format, the version is the highest version among the specified fields. For example, if you only specify fields from version 2, the version is 2. If you specify a mixture of fields from versions 2, 3, and 4, the version is 4. | 2
account-id |  The AWS account ID of the owner of the source network interface for which traffic is recorded. If the network interface is created by an AWS service, for example when creating a VPC endpoint or Network Load Balancer, the record may display unknown for this field. | 2
interface-id |  The ID of the network interface for which the traffic is recorded. | 2
srcaddr |  The source address for incoming traffic, or the IPv4 or IPv6 address of the network interface for outgoing traffic on the network interface. The IPv4 address of the network interface is always its private IPv4 address. See also pkt-srcaddr. | 2
dstaddr |  The destination address for outgoing traffic, or the IPv4 or IPv6 address of the network interface for incoming traffic on the network interface. The IPv4 address of the network interface is always its private IPv4 address. See also pkt-dstaddr. | 2
srcport |  The source port of the traffic. | 2
dstport |  The destination port of the traffic. | 2
protocol |  The IANA protocol number of the traffic. For more information, see Assigned Internet Protocol Numbers. | 2
packets |  The number of packets transferred during the flow | 2bytes | The number of bytes transferred during the flow. | 2
start |  The time, in Unix seconds, when the first packet of the flow was received within the aggregation interval. This might be up to 60 seconds after the packet was transmitted or received on the network interface. | 2
end |  The time, in Unix seconds, when the last packet of the flow was received within the aggregation interval. This might be up to 60 seconds after the packet was transmitted or received on the network interface. | 2
`action` |  The action that is associated with the traffic: <br> ACCEPT: The recorded traffic was permitted by the security groups and network ACLs. <br> REJECT: The recorded traffic was not permitted by the security groups or network ACLs. | 2
log-status |  The logging status of the flow log: <br> OK: Data is logging normally to the chosen destinations. <br> NODATA: There was no network traffic to or from the network interface during the aggregation interval. <br> SKIPDATA: Some flow log records were skipped during the aggregation interval. This may be because of an internal capacity constraint, or an internal error. | 2
`vpc-id` |  The ID of the VPC that contains the network interface for which the traffic is recorded. | 3
subnet-id | The ID of the subnet that contains the network interface for which the traffic is recorded | 3instance-id | The ID of the instance that's associated with network interface for which the traffic is recorded, if the instance is owned by you. Returns a '-' symbol for a requester-managed network interface; for example, the network interface for a NAT gateway. | 3
`tcp-flags` | The bitmask value for the following TCP flags:<br> SYN: 2 <br> SYN-ACK: 18 <br> FIN: 1 <br> RST: 4 <br> ACK is reported only when it's accompanied with SYN. <br> TCP flags can be OR-ed during the aggregation interval. For short connections, the flags might be set on the same line in the flow log record, for example, 19 for SYN-ACK and FIN, and 3 for SYN and FIN. For an example, see TCP flag sequence. | 3
`type` | The type of traffic: IPv4, IPv6, or EFA. For more information about the Elastic Fabric Adapter (EFA), see Elastic Fabric Adapter. | 3
pkt-srcaddr | The packet-level (original) source IP address of the traffic. Use this field with the srcaddr field to distinguish between the IP address of an intermediate layer through which traffic flows, and the original source IP address of the traffic. For example, when traffic flows through a network interface for a NAT gateway, or where the IP address of a pod in Amazon EKS is different from the IP address of the network interface of the instance node on which the pod is running (for communication within a VPC). | 3
pkt-dstaddr | The packet-level (original) destination IP address for the traffic. Use this field with the dstaddr field to distinguish between the IP address of an intermediate layer through which traffic flows, and the final destination IP address of the traffic. For example, when traffic flows through a network interface for a NAT gateway, or where the IP address of a pod in Amazon EKS is different from the IP address of the network interface of the instance node on which the pod is running (for communication within a VPC). | 3
region | The Region that contains the network interface for which traffic is recorded. | 4
az-id | The ID of the Availability Zone that contains the network interface for which traffic is recorded. If the traffic is from a sublocation, the record displays a '-' symbol for this field. | 4
sublocation-type | The type of sublocation that's returned in the sublocation-id field: <br> wavelength <br> outpost <br> localzone <br> If the traffic is not from a sublocation, the record displays a '-' symbol for this field. |4
sublocation-id | The ID of the sublocation that contains the network interface for which traffic is recorded. If the traffic is not from a sublocation, the record displays a '-' symbol for this field | 4

> If a field is not applicable for a specific record, the record displays a '-' symbol for that entry.

### Flow log limitations
- To use flow logs, you need to be aware of the following limitations:
- cannot enable flow logs for network interfaces that are in the EC2-Classic platform.
  - This includes EC2-Classic instances that have been linked to a VPC through ClassicLink.
- can't enable flow logs for VPCs that are peered with VPC unless the peer VPC is in account.
- After you've created a flow log, you cannot change its configuration or the flow log record format. For example, you can't associate a different IAM role with the flow log, or add or remove fields in the flow log record. but delete the flow log and create a new one with the required configuration.
- If network interface has multiple IPv4 addresses and traffic is sent to a secondary private IPv4 address, the flow log displays the primary private IPv4 address in the dstaddr field. To capture the original destination IP address, create a flow log with the pkt-dstaddr field.
- If traffic is sent to a network interface and the destination is not any of the network interface's IP addresses, the flow log displays the primary private IPv4 address in the dstaddr field. To capture the original destination IP address, create a flow log with the pkt-dstaddr field.
- If traffic is sent from a network interface and the source is not any of the network interface's IP addresses, the flow log displays the primary private IPv4 address in the srcaddr field. To capture the original source IP address, create a flow log with the pkt-srcaddr field.
- If traffic is sent to or sent by a network interface, the srcaddr and dstaddr fields in the flow log always display the primary private IPv4 address, regardless of the packet source or destination. To capture the packet source or destination, create a flow log with the pkt-srcaddr and pkt-dstaddr fields.
- When network interface is attached to a Nitro-based instance, the aggregation interval is always 1 minute or less, regardless of the specified maximum aggregation interval.

Flow logs do not capture all IP traffic. The following types of traffic are not logged:
- Traffic generated by instances when they contact the Amazon DNS server. If you use own DNS server, then all traffic to that DNS server is logged.
- Traffic generated by a Windows instance for Amazon Windows license activation.
- Traffic to and from 169.254.169.254 for instance metadata.
- Traffic to and from 169.254.169.123 for the Amazon Time Sync Service.
- DHCP traffic.
- Traffic to the reserved IP address for the default VPC router. For more information, see VPC and subnet sizing.
- Traffic between an endpoint network interface and a Network Load Balancer network interface. For more information, see VPC endpoint services (AWS PrivateLink).

### Flow logs pricing
- `Data ingestion and archival` charges for vended logs apply when you publish flow logs to CloudWatch Logs / S3.
- To track charges from publishing flow logs to S3 buckets, apply `cost allocation tags` to flow log subscriptions.
- To track charges from publishing flow logs to CloudWatch Logs, apply `cost allocation tags` to destination CloudWatch Logs log group.
- Thereafter, AWS cost allocation report will include usage and costs aggregated by these tags. You can apply tags that represent business categories (such as cost centers, application names, or owners) to organize costs.

CloudWatch Container Insights ingests performance events as CloudWatch Logs that automatically create CloudWatch metrics. These performance events are analyzed using CloudWatch Logs Insights queries and are automatically executed as part of some Container Insights automated dashboards (e.g., task/pod, service, node, namespace).

into CloudWatch | price
---|---
Collect (Data Ingestion)	| $0.50 per GB
Store (Archival)	| $0.03 per GB
Analyze (Logs Insights queries)	| $0.005 per GB of data scanned

---

### price

[AWS Simple Monthly Calculator](https://calculator.s3.amazonaws.com/index.html)
[AWS Pricing Calculator](https://calculator.aws/#/createCalculator)

example:

```bash
# detailed monitoring
# The number of metrics sent by EC2 instance as detailed monitoring is dependent on the EC2 instance type
# This example assumes 7 metrics, which covers the most commonly used instance types.
If application runs on 10 EC2 instances 24x7 for a 30-day month, and enable EC2 Detailed Monitoring on all instances:
Total number of metrics: 7 metrics per instance * 10 instances = 70 metrics
Monthly CloudWatch Metrics Charges @$0.30 per custom metric: 70 * $0.30 = $21
Monthly CloudWatch charges = $21 per month
Once you exceed 10,000 total metrics then volume pricing tiers will apply - see metrics pricing table for details.


# monitor with logs
# If you are monitoring HTTP 2xx, 3xx & 4xx response codes using web application access logs 24x7 for one 30-day month, by sending 1GB per day of ingested log data, monitoring for HTTP responses, and archiving the data for one month, charges would be as follows:
# Monthly Ingested Log Charges
Total log data ingested: 1GB * 30 days = 30GB
0 to 5GB: $0
5 to 30GB: $0.50 * 25 = $12.50
# Monthly Monitoring Charges
3 CloudWatch Metrics @$0: 3 * $0 = $0
# Monthly Archived Log Charges (assume log data compresses to 6GB)
0 to 5GB: $0
5GB to 6GB: $0.03 * 1 = $0.03
# Monthly CloudWatch Charges
$12.50 + $0 + $0.03 = $12.53


# monitore VPCs that send 72TB of ingested VPC flow logs to CloudWatch logs per month and archiving the data for one month
# Monthly Log Ingestion Charges
0 to 10TB @$0.50 per GB = 10 * 1,024 * $0.50 = $5,120.00
10TB to 30TB @$0.25 per GB = 20 * 1,024 * $0.25 = $5,120.00
30TB to 50TB @$0.10 per GB = 20 * 1,024 * $0.10 = $2,048.00
50TB to 72TB @$0.05 per GB = 22 * 1024 * $0.05 = $1,126.40
Total Ingestion Charges = $5,120 + $5,120 + $2,048 + $1126.40 = $13,414.40
# Monthly Log Archival Charges (Assume log data compresses to 30TB)
30TB @ $0.03 per GB = 30 * 1024 * 0.03 = $921.6
# Monthly CloudWatch Charges = $13,414.40 + $921.6 = $14,336


# 2.
# monitore VPCs that send 72TB of ingested VPC flow logs directly to S3 per month and archiving the data for one month
# Monthly Log Ingestion Charges
0 to 10TB @$0.25 per GB = 10 * 1,024 * $0.25 = $2,560.00
10TB to 30TB @$0.15 per GB = 20 * 1,024 * $0.15 = $3,072.00
30TB to 50TB @$0.075 per GB = 20 * 1,024 * $0.075 = $1,536.00
50TB to 72TB @$0.05 per GB = 22 * 1024 * $0.05 = $1,126.40
Total Ingestion Charges = $2,560 + $3,072 + $1,536 + $1126.40 = $8,294.40
# Monthly Log Archival Charges (Assume log data compresses to 6.5TB)* *
6.5TB @ $0.023 per GB = 6.5 * 1024 * 0.023 = $153.01
# Monthly Charges = $8,294.40 + $153.01 = $8,447.41
```



---

ref:
- [*VPC Flow Logs](https://docs.aws.amazon.com/vpc/latest/userguide/flow-logs.html)
- [*Traffic Mirroring and VPC Flow Logs](https://docs.aws.amazon.com/vpc/latest/mirroring/flow-log.html)
- [New – VPC Traffic Mirroring – Capture & Inspect Network Traffic](https://aws.amazon.com/blogs/aws/new-vpc-traffic-mirroring/)



.
