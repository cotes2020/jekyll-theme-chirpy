



# encrypt the ebs by default


```py
def lambda_handler(event, context):
    # set the region
    region = 'us-east-1'

    # set the client
    client = boto3.client('ec2', region)

    response = client.enable_ebs_encryption_by_default()

    # result =
    print("Default EBS Encryption setup for region", region,": ", response['EbsEncryptionByDefault'])

    return {
        'statusCode': 200,
        'body': json.dumps(event)
    }
```














