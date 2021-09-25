

[toc]


# CLI




---


## iam


```
aws sts get-caller-identity


```




---


## cloudtrail

```
aws cloudtrail lookup-events \
  --lookup-attributes AttributeKey=ResourceName,AttributeValue=ais_endpoint_aggregator \
  --start-time 2021-01-01 \
  --end-time 2021-03-01 \
  --region us-west-2 | > text.json

cat text.txt | grep -e "EventName" -e "EventId" \
             | grep -v -e "AssumeRole"

cat text.txt | grep -e "EventName" \
             | grep -v -e "AssumeRole"




```







---

## ec2

```bash




aws ec2 describe-snapshot-attribute \
    --snapshot-id snap-066877671789bd71b \
    --attribute createVolumePermission
# Possible values:
# productCodes
# createVolumePermission


```



---


## SG

```bash
aws ec2 descruibe-network-interfaces \
    --filters Name=group-id, Values=<group-id> \
    --region <region> \
    --output json

```

---

## s3

```bash

aws s3api list-buckets --query "Buckets[].Name"

aws s3api put-public-access-block \
    --bucket my-bucket \
    --public-access-block-configuration "BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true"
```




---

## SSM

```bash
# list the ssm doc
aws ssm list-documents \
    --filters key="SearchKeyword",Values="AWS" \
    --region us-west-2 > ssmdoc.txt

aws ssm list-documents \
    --filters key=Owner,Values=Private/Public \
    --region us-west-2

aws ssm list-documents \
    --filters key=Owner,Values=Private/Public \
    --region us-west-2

aws ssm start-automation-execution \
    --document-name my-ssm-doc \
    --parameters "BucketName=my-bucket,Role=my-role" \
    --region us-west-2


```
