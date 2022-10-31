---
title: AWS - boto3 - boto3.client('cloudwatch') CloudWatch
date: 2020-07-18 11:11:11 -0400
categories: [01AWS, boto3]
tags: [AWS]
toc: true
image:
---

- [Amazon CloudWatch `boto3.client('cloudwatch')`](#amazon-cloudwatch-boto3clientcloudwatch)
  - [example](#example)
    - [AWS CloudWatch Log Group Retention to 60](#aws-cloudwatch-log-group-retention-to-60)

---


# Amazon CloudWatch `boto3.client('cloudwatch')`

```py
cloudwatch = boto3.client('cloudwatch')

cloudwatch.get_paginator()  # paginator
for response in paginator.paginate(StateValue='INSUFFICIENT_DATA'):
    response['MetricAlarms']

cloudwatch.put_metric_alarm()
cloudwatch.delete_alarms()

```




```py

# -------------------------- Creating alarms in Amazon CloudWatch --------------------------
cloudwatch = boto3.client('cloudwatch')


# -------------------------- List alarms of insufficient data through the pagination interface --------------------------
# -------------------------- cloudwatch.get_paginator('describe_alarms')
paginator = cloudwatch.get_paginator('describe_alarms')
for response in paginator.paginate(StateValue='INSUFFICIENT_DATA'):
    print(response['MetricAlarms'])


# -------------------------- Create/update alarm for a CloudWatch Metric alarm --------------------------
# -------------------------- put_metric_alarm.
# creates an alarm:
# the alarm state is immediately set to INSUFFICIENT_DATA.
# The alarm is evaluated and its state is set appropriately.
# Any actions associated with the state are then executed.
# update an existing alarm
# its state is left unchanged, but the update completely overwrites the previous configuration of the alarm.
cloudwatch.put_metric_alarm(
    AlarmName='Web_Server_CPU_Utilization',
    ComparisonOperator='GreaterThanThreshold',
    EvaluationPeriods=1,
    MetricName='CPUUtilization',
    Namespace='AWS/EC2',
    Period=60,
    Statistic='Average',
    Threshold=70.0,
    ActionsEnabled=False,
    AlarmDescription='Alarm when server CPU exceeds 70%',
    Dimensions=[
        {
          'Name': 'InstanceId',
          'Value': 'INSTANCE_ID'
        },
    ],
    Unit='Seconds'
)


# -------------------------- Delete an alarm --------------------------
cloudwatch.delete_alarms(
  AlarmNames=['Web_Server_CPU_Utilization'],
)

```


---



## example


### AWS CloudWatch Log Group Retention to 60

pleaze go to the follow link for the original code
ref: [AWS CloudWatch Log Group Retention to 60](https://dev.to/akloya/aws-cloudwatch-log-group-retention-3l47)


CloudWatch organises logs in a log group and when a new log group is created, it’s retention period is set to Never expire by default (be retained forever)


to changing the retention days to 60

```py
import boto3

# set the number of retention days
retention_days = 60

# list the regions you are interested to run this script on
regions=['us-east-1']

for region in regions:
    logclient = boto3.client('logs',region)
    response = logclient.describe_log_groups()
    nextToken = response.get('nextToken', None)
    retention = response['logGroups']

    while (nextToken is not None):
        response = logclient.describe_log_groups(nextToken=nextToken)
        nextToken = response.get('nextToken', None)
        retention = retention + response['logGroups']

    for group in retention:
        if 'retentionInDays' in group.keys():
            print(group['logGroupName'], group['retentionInDays'],region)
        else:
            print("Retention Not found for ",group['logGroupName'],region)
            setretention = logclient.put_retention_policy(
                logGroupName = group['logGroupName'],
                retentionInDays = retention_days
                )
            print(setretention)
```




















.
