



[toc]


- ref
  - [Orchestrating analytics jobs on Amazon EMR Notebooks using Amazon MWAA](https://noise.getoto.net/2021/01/27/orchestrating-analytics-jobs-on-amazon-emr-notebooks-using-amazon-mwaa/)
  - [Introducing Amazon Managed Workflows for Apache Airflow (MWAA)](https://noise.getoto.net/2020/11/24/introducing-amazon-managed-workflows-for-apache-airflow-mwaa/)
  - Post Syndicated from original [link](https://aws.amazon.com/blogs/aws/introducing-amazon-managed-workflows-for-apache-airflow-mwaa/)
  - [Amazon Managed Workflows for Apache Airflow User Guide](https://docs.aws.amazon.com/mwaa/latest/userguide/amazon-mwaa-user-guide.pdf)




---

# Amazon Managed Workflows for Apache Airflow (Amazon MWAA)

---


## Apache Airflow

[Apache Airflow](https://airflow.apache.org/)
- As the volume and complexity of your data processing pipelines increase
- simplify the overall process by decomposing it into a series of smaller tasks and coordinate the execution of these tasks as part of a **workflow**.
- platform created by the community to <font color=red> programmatically author, schedule, and monitor workflows </font>
- manage workflows as scripts,
- monitor them via the user interface (UI),
- and extend their functionality through a set of powerful plugins.


---


## basic

Amazon MWAA

- a fully managed service
- makes it easy to
  - run open-source versions of Apache Airflow on AWS,
  - build workflows to run extract, transform, and load (ETL) jobs and data pipelines.
- avoidmanually installing, maintaining, and scaling Airflow, and handling security, authentication, and authorization for its users

Airflow workflows
- retrieve input from sources like [Amazon Simple Storage Service (S3)](https://aws.amazon.com/s3/) using [Amazon Athena](https://aws.amazon.com/athena) queries,
- perform transformations on [Amazon EMR](https://aws.amazon.com/emr) clusters,
- can use the resulting data to train machine learning models on [Amazon SageMaker](https://aws.amazon.com/sagemaker/).
- Workflows in Airflow are authored as [Directed Acyclic Graphs (DAGs)](https://airflow.apache.org/docs/stable/concepts.html#dags) using the Python programming language.
- Airflow **metrics** can be published as CloudWatch Metrics, and **logs** can be sent to CloudWatch Logs.
- Amazon MWAA provides automatic minor version **upgrades** and patches by default, with an option to designate a maintenance window in which these upgrades are performed.




Benefits of using Amazon MWAA

<font color=red> Setup </font>
- a managed service for Apache Airflow
- build, manage, and maintain Apache Airflow on AWS using services such as Amazon EC2 or Amazon EKS
- sets up Apache Airflow when you create an environment using the same open-source Airflow and user interface available from Apache.
- build workflows to run your extract, transform, and load (ETL) jobs and data pipelines.
- don't need to perform a manual setup or use custom tools to create an environment.
  - not a "branch" of Airflow, nor is it just "compatible with".
  - It is the exact same Apache Airflow that you can download on the own.
- makes it easy for you to build and manage the workflows in the cloud.

<font color=red> Scaling </font>
- use the same familiar Airflow platform with <font color=blue> improved scalability, availability, and security </font>
  - without the operational burden of having to manage the underlying infrastructure.
- uses the Apache Celery Executor to automatically scale workers as needed for the environment.
  - scales capacity up to meet demand
  - and back down to conserve resources and minimize costs.
- monitors the workers in the environment,
  - as demand increases, Amazon MWAA adds additional worker containers.
  - As workers free up, Amazon MWAA removes them.


<font color=red> Security </font>
- Integrated support with AWS Identity and Access Management (IAM), including role-based authentication and authorization for access to the Airflow user interface.
  - Workers assume IAM roles for easy and secure access to AWS services.
  - Workers and Scheduler run in the VPC for access to the resources.
- Amazon MWAA supports accessing the Airflow UI on either a VPC or a public secured endpoint.

<font color=red> Upgrades and patches </font>
- updates and patches Airflow automatically, with scheduled and communicated maintenance windows.
  - manages the provisioning and ongoing maintenance of Apache Airflow
  - automatically applies patches and updates to Apache Airflow in the Amazon MWAA environments.
  - don't need to manage different versions of Airflow using different library versions.
- automatically recovers from failed upgrades and patches.
- Point-point releases available within 7 days
- Minor versions available within 30 days



<font color=red> Monitoring </font>
- integrated with CloudWatch.
- The Apache Airflow logs and performance metrics data for the environment are available in a single location.
- This lets you easily identify workflow errors or task delays.
- Amazon MWAA automatically, if enabled, sends Apache Airflow system metrics and logs to CloudWatch.
- view logs and metrics for multiple environments from a single location
- easily identify task delays or workflow errors without the need for additional third-party tools.

<font color=red> Integration </font>
- easily combine data using any of Apache Airflow’s open source integrations.
- community provides operators (plugins that simplify connections to services) for Apache Airflow to integrate with
  - AWS services
    - such as Amazon S3, Amazon Redshift, Amazon EMR, AWS Batch, and Amazon SageMaker, Amazon Athena, AWS Batch, Amazon CloudWatch, Amazon DynamoDB, AWS DataSync, Amazon EMR, AWS Fargate, Amazon EKS, Amazon Kinesis Data Firehose, AWS Glue, AWS Lambda, Amazon Redshift, Amazon SQS, Amazon SNS, Amazon SageMaker, and Amazon S3
    - integrated with AWS security services to enable secure access to customer data
    - supports single sign-on using the same AWS credentials to access the Apache Airflow UI.
  - as well as hundreds of built-in and community-created operators and sensors
  - services on other cloud platforms.
  - and popular third-party tools
    - such as Apache Hadoop, Presto, Hive, and Spark to perform data processing tasks.
- Amazon MWAA is committed to maintaining compatibility with the Amazon MWAA API,


<font color=red> Containers </font>
- Amazon MWAA offers support for using containers to scale the worker fleet on demand and reduce scheduler outages, through AWS Fargate.
- Operators that execute tasks on Amazon ECS containers, as well as Kubernetes operators that create and run pods on a Kubernetes cluster, are supported.


---

## Amazon MWAA and Airflow workflows

- Apache Airflow manages data through a series of tasks called a <font color=red> workflow </font>
- A workflow comprised of these tasks: a <font color=red> Directed Acyclic Graph (DAG) </font>
  - DAGs describe how to run a workflow and are written in Python
- When a workflow is created, tasks are configured
  - so that some tasks must finish before the next task can start without needing to loop back to a previous task.

- Example,
  - tasks that collect and process data must finish collecting and processing all data before attempting to merge the data.
  - collection of tasks for a media distribution company. There is a task for
  - connecting to each content provider service that media is distributed to,
  - requesting the play count and sales for each title,
  - pulling social media impressions,
  - and then loading that data to a storage location, such as an Amazon S3 bucket.
  - After the data is uploaded, a task to process the data starts and converts the data to another format or modifies specific values.
  - The task to merge the data together starts only after all of the preceding tasks are completed.
    - by tools like AWS Glue or Amazon Athena, or perhaps using Amazon SageMaker to identify similar entries that can combined further.
  - After all tasks are complete, the result is a clean and complete data set ready for
    - analysis, such as with Amazon Redshift, or storage with Amazon DynamoDB.

- If a task fails, the workflow is configured to automatically retry the failed task while the subsequent tasks wait for that task to complete.
  - If a manual restart is required, the workflows starts at the failed task rather than the first task in the workflow.
  - save time and resources by not repeating tasks that had already completed successfully.

---


### Amazon S3

- Amazon MWAA uses an S3 bucket to store DAGs and associated support files.
- must create an S3 bucket before you can create the environment.
- must create the bucket in the same Region where you create the environment.

---


### VPC network configurations


- Required VPC networking components requirements:
  - <font color=red> Two private subnets </font>
    - in two different availability zones within the same Region.
  - also need one of the following:
    1. <font color=red> Two public subnets </font>
       - configured to route the private subnet data to the Internet. (via NAT gateways)
    2. Or <font color=red> VPC endpoint services (AWS PrivateLink) </font>

> If you are unable to provide Internet routing for the two private subnets,
> - VPC endpoint services (AWS PrivateLink) access to the AWS services used by the environment is required.
> - AWS services used: Amazon CloudWatch, CloudWatch Logs, Amazon ECR, Amazon S3, Amazon SQS, AWS Key Management Service


- The Airflow UI in the Amazon MWAA environment is accessible over the internet by users granted access in the IAM policy.
- Amazon MWAA attaches an [Application Load Balancer](https://docs.aws.amazon.com/elasticloadbalancing/latest/application/introduction.html) with an HTTPS endpoint for your web server as part of the Amazon MWAA managed service.


---


### VPC endpoints

- VPC endpoints are highly available VPC components
- enable private connections between your VPC and supported AWS services.
- Traffic between your VPC and the other services remains in your AWS network.


For example:
- use the VPC endpoints to ensure extra security, availability, and Amazon S3 data transfer performance:
- An Amazon S3 [gateway VPC endpoint](https://docs.aws.amazon.com/vpc/latest/userguide/vpce-gateway.html) to establish a private connection between the Amazon MWAA VPC and Amazon S3
- An EMR [interface VPC endpoint](https://docs.aws.amazon.com/vpc/latest/userguide/vpce-interface.html) to securely route traffic directly to Amazon EMR from Amazon MWAA, instead of connecting over the internet


---

## Airflow components

Each environment has an Airflow Scheduler and 1 or more Airflow Workers, managed by auto-scaling, that are linked to the VPC.
- The meta database and web servers are isolated in the service’s account, and there are separate instances of each for each Airflow environment created
  - there is no shared tenancy of any components, even within the same account.
- Web server access can then be exposed through an endpoint within the VPC, or more simply can be exposed through a load balancer to a publicly accessible endpoint, in each case secured by IAM and AWS SSO.


---

## Get started with MWAA

Amazon Managed Workflows for Apache Airflow (MWAA) uses
- the Amazon VPC,
- DAG code and supporting files in the Amazon S3 storage bucket to create an environment.
  - You specify the location of the Amazon S3 bucket, the path to the DAG code, and any custom plugins or dependencies on the Amazon MWAA console when you create an environment.


### Prerequisites
- <font color=red> AWS account </font>
  - An AWS account with permission to use Amazon MWAA and the AWS services and resources used by the environment.
- <font color=red> Amazon S3 bucket </font>
  - An Amazon S3 bucket with versioning enabled.
  - An Amazon S3 bucket is used to store the DAGs and associated files,
  - such as plugins.zip and requirements.txt.
- <font color=red> Amazon VPC </font>
  - The Amazon VPC networking components required by an Amazon MWAA environment.
  - You can use an existing VPC that meets these requirements, or create the VPC and networking components as defined in Create the VPC network.
- <font color=red> Customer master key (CMK) </font>
  - A Customer master key (CMK) for data encryption on the environment.
  - You can choose the default option on the Amazon MWAA console to create an AWS owned CMK when you create an environment.
- <font color=red> Execution role </font>
  - An execution role that allows Amazon MWAA to access AWS resources in the environment.
  - You can choose the default option on the Amazon MWAA console to create an execution role when you create an environment


### 1. Create S3 bucket for Amazon MWAA

- Buckets have configuration properties, including
  - <font color=red> name </font>
    - Avoid including sensitive information in the bucket name.
      - such as account numbers,
    - The bucket name is visible in the URLs that point to the objects in the bucket.
  - <font color=red> geographical Region </font>
  - <font color=red> access settings for the objects in the bucket </font>
    - Amazon MWAA requires that the bucket does not allow public access.
    - You should leave all settings enabled.
  - <font color=red> Bucket Versioning </font>
    - choose Enable.
  - <font color=red> encryption </font>
    - whether to enable server-side encryption for the bucket.
    - If you choose to enable server-side encryption
      - <font color=blue> must use the same key for the S3 bucket and Amazon MWAA environment </font>
  - If you want to <font color=red> enable S3 Object lock </font>
    - can only enable S3 Object lock for bucket when you create it,
    - can't disable it later.
    - Enabling Object lock also enables versioning for the bucket.
    - After you enable Object lock for the bucket, you must configure the Object lock settings before any objects in the bucket are protected.
- Choose Create bucket.


### 2. Create the VPC network
- need the VPC networking components required by an Amazon MWAA environment.
  1. use an existing VPC that meets these requirements,
  2. create the VPC and networking components on the Amazon MWAA console,
  3. use the provided AWS CloudFormation template to create the VPC and other required networking components.

- Amazon MWAA provides private and public networking options for the Apache Airflow web server.
- A <font color=red> public network </font>
  - allows the Airflow UI to be accessed over the Internet by users granted access in the IAM policy.
  - Amazon MWAA attaches an Application Load Balancer with an HTTPS endpoint for the web server as part of the Amazon MWAA managed service.
- A <font color=red> private network </font>
  - limits access to the Airflow UI to users within the VPC.
  - Amazon MWAA attaches a VPC endpoint to the web server.
  - Enabling access to this endpoint requires additional configuration,
    - such as a proxy or Linux Bastion.
  - In addition, you must grant users access in the IAM policy.


### 3. Environment infrastructure

- When create an environment
  - Amazon MWAA
    - creates <font color=red> an Aurora PostgreSQL metadata database and an Fargate container </font>
    - in each of the two private subnets in different availability zones.
  - The Apache Airflow workers on an Amazon MWAA environment
    - use the Celery Executor to queue and distribute tasks to multiple Celery workers from an Apache Airflow platform.
    - The Celery Executor runs in an AWS Fargate container.
      - If a Fargate container in one availability zone fails,
        - Amazon MWAA switches to the other container in a different availability zone to run the Celery Executor,
        - and the Apache Airflow scheduler creates a new task instance in the Amazon Aurora PostgreSQL metadata database.

- When you create an Amazon MWAA environment,
  - it uses the VPC network that you created for Amazon MWAA, and adds the other necessary networking components.
  - it automatically installs the version of Apache Airflow that you specify, including workers, scheduler, and web server.
    - The environment includes a link to access the Apache Airflow UI in the environment.
    - You can create up to 10 environments per account per Region, and each environment can include multiple DAGs.


Amazon MWAA console > Create environment
---

- provide a name for your environment
- select the Apache Airflow version to use.
  -![mwaa-create-environment-1-1024x342](https://i.imgur.com/njWTLzN.png)

- <font color=red> Under DAG code in Amazon S3: </font>
  -![mwaa-dag-code-s3-1012x1024](https://i.imgur.com/lsE8wTe.png)

  - <font color=blue> For S3 bucket </font>
    - choose the bucket that you created for Amazon MWAA
    - Enter the Amazon S3 URI to the bucket.

  - <font color=blue> For DAGs folder </font>
    - choose the DAG folder that you added to the bucket for Amazon MWAA
    - Enter the Amazon S3 URI to the DAG folder in the bucket.

  - (Optional). <font color=blue> For Plugins file </font>
    - The plugins file is a ZIP file <font color=blue> containing the plugins used by my DAGs </font>
    - do one of the following:
      - Choose Browse S3 and select the plugins.zip file that you added to the bucket. You must also select a version from the drop-down menu.
      - Enter the Amazon S3 URI to the plugin.zip file that you added to the bucket.
    - You can create an environment and then add a plugins.zip file later.


  - (Optional) <font color=blue> For Requirements file </font>
    - The requirements file <font color=blue> describes the Python dependencies to run my DAGs </font>
    - do one of the following:
    - Choose Browse S3 and then select the Python requirements.txt that you added to the bucket. Then select a version for the file from the drop-down menu.
    - Enter the Amazon S3 URI to the requirements.txt file in the bucket.
    - You can add a requirements file to your bucket after you create an environment. After you add or update the file you can edit the environment to modify these settings.

  - For plugins and requirements, I can select the [S3 object version](https://docs.aws.amazon.com/AmazonS3/latest/dev/Versioning.html) to use. In case the plugins or the requirements I use create a non-recoverable error in my environment, Amazon MWAA will automatically roll back to the previous working version.


- Configure advanced settings page (Networking)
  - ![mwaa-networking-1-881x1024](https://i.imgur.com/I1CbouT.png)
  - under VPC
    - Each environment runs in a VPC using private subnets in two AZ
    - choose the VPC that was you created for Amazon MWAA.
  - Under Subnets
    - Only private subnets are supported.
    - You can't change the VPC for an environment after you create it.

  - Under <font color=red> Web server access </font>
    - Web server access to the Airflow UI is always protected by a secure login using IAM
      - can have web server access on a public network to login over the Internet,
      - or on a private network in your VPC.
    - <font color=blue> Public Network </font>
      - This creates a public URL to access the Apache Airflow user interface in the environment.
    - <font color=blue> Private Network </font>
      - restrict access to the Apache Airflow UI to be accessible only from within the VPC selected
      - This creates a VPC endpoint that requires additional configuration to allow access, including a Linux Bastion.
    - The VPC endpoint for to access the Apache Airflow UI is listed on the Environment details page after you create the environment.



  - Under <font color=red> Security group </font>
    - Create new security group
    - to have Amazon MWAA create a new security group with inbound and outbound rules based on your Web server access selection.
    - can add one or more existing security groups to fine-tune control of inbound and outbound traffic for the environment.
      - select up to 5 security groups from your account to use for the environment.


  - Under <font color=red> Environment class </font>
  - ![mwaa-environment-class-1024x555](https://i.imgur.com/HhVwUCZ.png)
    - You can increase the environment size later as appropriate.
      - The environment size determines the approximate number of workflows that an environment supports.
    - For Maximum worker count
      - specify the maximum number of workers, up to 25, to run concurrently in the environment.
    - Amazon MWAA automatically handles working scaling up to the maximum worker count.
    - The environment class for the Amazon MWAA environment determines the size of:
        - the AWS-managed <font color=blue> AWS Fargate containers </font>
          - where the Celery Executor runs,
        - and the AWS-managed <font color=blue> Amazon Aurora PostgreSQL metadata database </font>
          - where the Apache Airflow scheduler creates task instances.
    - Each environment includes a scheduler, a web server, and a worker.
      - Workers automatically scale up and down according to the workload.

  - Under <font color=red> Encryption </font>
  - ![mwaa-encryption-1-1024x286](https://i.imgur.com/3rXU8Dr.png)
    - to encrypt your data
    - AWS owned key (by default)
    - or a different AWS KMS key,
      - if you enabled server-side encryption for the S3 bucket you created for Amazon MWAA,
      - you must use the same key for both the S3 bucket and your Amazon MWAA environment.
      - You must also grant permissions for Amazon MWAA to use the key by attaching the policy described in Attach key policy.

  - Under <font color=red> Monitoring, </font>
  - ![mwaa-monitoring-1-1024x715](https://i.imgur.com/a2j0xXb.png)
    - choose whether to enable CloudWatch Metrics.
      - environment performance to CloudWatch Metrics.
      - This is enabled by default, but CloudWatch Metrics can be disabled after launch.
    - For Airflow logging configuration
      - choose whether to enable sending log data to CloudWatch Logs for the following Apache Airflow log categories:
      - Airflow task logs
      - Airflow web server logs
      - Airflow scheduler logs
      - Airflow worker logs
      - Airflow DAG processing logs
    - After you enable a log category, choose the Log level for each as appropriate for your environment.
      - specify the log level and which Airflow components should send their logs to CloudWatch Logs
    - For Airflow configuration options
    - ![mwaa-airflow-configuration-1024x603](https://i.imgur.com/5tdBfTD.png)
      - When you create an environment Apache Airflow is installed using the default configuration options.
      - If you add a custom configuration option, Apache Airflow uses the value from the custom configuration instead of the default.
      - add a customer configuration option
      - Select the configuration option to use a custom value for, then enter the Custom value.

  - Under <font color=red> Tags </font>
    - add any tags as appropriate for your environment.
    - Choose Add new tag, and then enter a Key and optionally, a Value for the key.

  - Under <font color=red> Permissions, </font>
    - ![mwaa-permissions-1024x361](https://i.imgur.com/tz09fDs.png)
    - configure the **permissions** that will be used by environment to <font color=blue> access the DAGs, write logs, and run DAGs accessing other AWS resources </font>
    - choose the role to use as the execution role.
    - To have Amazon MWAA create a role for this environment, choose Create new role.
      - You must have permission to create IAM roles to use this option.
    - If you or someone in your organization created a role to use for Amazon MWAA
    - Choose Create environment.
      - takes about twenty to thirty minutes to create an environment.

---

### 4. Accessing an Amazon MWAA environment

To use Amazon Managed Workflows for Apache Airflow (MWAA), you must use an account, user, or role
with the necessary permissions.

The resources and services used in an Amazon MWAA environment are not accessible to all IAM entities (users, roles, or groups).
- must create a policy that grants your Apache Airflow users permission to access these resources.
- For example
  - grant access to your Apache Airflow development team.
- Amazon MWAA uses these policies to validate whether a user has the permissions needed to perform an action on the AWS console or via the APIs used by an environment.

use the JSON policies in this topic to create a policy for your Apache Airflow users in IAM, and then attach the policy to a user, group, or role in IAM.

Here are the policies available:
- `AmazonMWAAFullConsoleAccess`
  - to configure an environment on the Amazon MWAA console.
- `AmazonMWAAFullApiAccess`
  - if need access to all Amazon MWAA APIs used to manage an environment.
- `AmazonMWAAReadOnlyAccess`
  - if they need to view the resources used by an environment on the Amazon MWAA console.
- `AmazonMWAAWebServerAccess`
  - if they need to access the Apache Airflow UI.
- `AmazonMWAAAirflowCliAccess`
  - to run Apache Airflow CLI commands.


4. Apache Airflow UI access policy: `AmazonMWAAWebServerAccess`
   - A user may need access to the AmazonMWAAWebServerAccess permissions policy if they need to access the Apache Airflow UI.
   - It does not allow the user to view environments on the Amazon MWAA console or use the Amazon MWAA APIs to perform any actions.
   - Specify the Admin, Op, User, Viewer or the Public role in {airflow-role} to customize the level of access for the user of the web token.
   - For more information, see Default Roles in the Apache Airflow reference guide.
   - Note: Amazon MWAA does not support custom Apache Airflow role-based access control (RBAC) roles as of yet.


```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": "airflow:CreateWebLoginToken",
      "Resource": "arn:aws:airflow:{your-region}:{your-account-id}:role/{your-environmentname}/{airflow-role}"
    }
  ]
}
```

---

### 5. Using the Airflow UI

- In the Amazon MWAA console, look for the new environment and click on **Open Airflow UI**.
- A new browser window is created and I am authenticated with a secure login via AWS IAM.

There, I look for a DAG that I put on S3 in the `movie_list_dag.py` file. The DAG is
- downloading the [MovieLens dataset](https://grouplens.org/datasets/movielens/),
- processing the files on S3 using [Amazon Athena](https://aws.amazon.com/athena),
- and loading the result to a Redshift cluster, creating the table if missing.


Here’s the full source code of the DAG:

```py
    from airflow import DAG
    from airflow.operators.python_operator import PythonOperator
    from airflow.operators import HttpSensor, S3KeySensor
    from airflow.contrib.operators.aws_athena_operator import AWSAthenaOperator
    from airflow.utils.dates import days_ago
    from datetime import datetime, timedelta
    from io import StringIO
    from io import BytesIO
    from time import sleep
    import csv
    import requests
    import json
    import boto3
    import zipfile
    import io
    s3_bucket_name = 'my-bucket'
    s3_key='files/'
    redshift_cluster='redshift-cluster-1'
    redshift_db='dev'
    redshift_dbuser='awsuser'
    redshift_table_name='movie_demo'
    test_http='https://grouplens.org/datasets/movielens/latest/'
    download_http='https://files.grouplens.org/datasets/movielens/ml-latest-small.zip'
    athena_db='demo_athena_db'
    athena_results='athena-results/'
    create_athena_movie_table_query="""
    CREATE EXTERNAL TABLE IF NOT EXISTS Demo_Athena_DB.ML_Latest_Small_Movies (
      `movieId` int,
      `title` string,
      `genres` string
    )
    ROW FORMAT SERDE 'org.apache.hadoop.hive.serde2.lazy.LazySimpleSerDe'
    WITH SERDEPROPERTIES (
      'serialization.format' = ',',
      'field.delim' = ','
    ) LOCATION 's3://pinwheeldemo1-pinwheeldagsbucketfeed0594-1bks69fq0utz/files/ml-latest-small/movies.csv/ml-latest-small/'
    TBLPROPERTIES (
      'has_encrypted_data'='false',
      'skip.header.line.count'='1'
    );
    """
    create_athena_ratings_table_query="""
    CREATE EXTERNAL TABLE IF NOT EXISTS Demo_Athena_DB.ML_Latest_Small_Ratings (
      `userId` int,
      `movieId` int,
      `rating` int,
      `timestamp` bigint
    )
    ROW FORMAT SERDE 'org.apache.hadoop.hive.serde2.lazy.LazySimpleSerDe'
    WITH SERDEPROPERTIES (
      'serialization.format' = ',',
      'field.delim' = ','
    ) LOCATION 's3://pinwheeldemo1-pinwheeldagsbucketfeed0594-1bks69fq0utz/files/ml-latest-small/ratings.csv/ml-latest-small/'
    TBLPROPERTIES (
      'has_encrypted_data'='false',
      'skip.header.line.count'='1'
    );
    """
    create_athena_tags_table_query="""
    CREATE EXTERNAL TABLE IF NOT EXISTS Demo_Athena_DB.ML_Latest_Small_Tags (
      `userId` int,
      `movieId` int,
      `tag` int,
      `timestamp` bigint
    )
    ROW FORMAT SERDE 'org.apache.hadoop.hive.serde2.lazy.LazySimpleSerDe'
    WITH SERDEPROPERTIES (
      'serialization.format' = ',',
      'field.delim' = ','
    ) LOCATION 's3://pinwheeldemo1-pinwheeldagsbucketfeed0594-1bks69fq0utz/files/ml-latest-small/tags.csv/ml-latest-small/'
    TBLPROPERTIES (
      'has_encrypted_data'='false',
      'skip.header.line.count'='1'
    );
    """
    join_tables_athena_query="""
    SELECT REPLACE ( m.title , '"' , '' ) as title, r.rating
    FROM demo_athena_db.ML_Latest_Small_Movies m
    INNER JOIN (SELECT rating, movieId FROM demo_athena_db.ML_Latest_Small_Ratings WHERE rating > 4) r on m.movieId = r.movieId
    """
    def download_zip():
        s3c = boto3.client('s3')
        indata = requests.get(download_http)
        n=0
        with zipfile.ZipFile(io.BytesIO(indata.content)) as z:
            zList=z.namelist()
            print(zList)
            for i in zList:
                print(i)
                zfiledata = BytesIO(z.read(i))
                n += 1
                s3c.put_object(Bucket=s3_bucket_name, Key=s3_key+i+'/'+i, Body=zfiledata)
    def clean_up_csv_fn(**kwargs):
        ti = kwargs['task_instance']
        queryId = ti.xcom_pull(key='return_value', task_ids='join_athena_tables' )
        print(queryId)
        athenaKey=athena_results+"join_athena_tables/"+queryId+".csv"
        print(athenaKey)
        cleanKey=athena_results+"join_athena_tables/"+queryId+"_clean.csv"
        s3c = boto3.client('s3')
        obj = s3c.get_object(Bucket=s3_bucket_name, Key=athenaKey)
        infileStr=obj['Body'].read().decode('utf-8')
        outfileStr=infileStr.replace('"e"', '')
        outfile = StringIO(outfileStr)
        s3c.put_object(Bucket=s3_bucket_name, Key=cleanKey, Body=outfile.getvalue())
    def s3_to_redshift(**kwargs):
        ti = kwargs['task_instance']
        queryId = ti.xcom_pull(key='return_value', task_ids='join_athena_tables' )
        print(queryId)
        athenaKey='s3://'+s3_bucket_name+"/"+athena_results+"join_athena_tables/"+queryId+"_clean.csv"
        print(athenaKey)
        sqlQuery="copy "+redshift_table_name+" from '"+athenaKey+"' iam_role 'arn:aws:iam::163919838948:role/myRedshiftRole' CSV IGNOREHEADER 1;"
        print(sqlQuery)
        rsd = boto3.client('redshift-data')
        resp = rsd.execute_statement(
            ClusterIdentifier=redshift_cluster,
            Database=redshift_db,
            DbUser=redshift_dbuser,
            Sql=sqlQuery
        )
        print(resp)
        return "OK"
    def create_redshift_table():
        rsd = boto3.client('redshift-data')
        resp = rsd.execute_statement(
            ClusterIdentifier=redshift_cluster,
            Database=redshift_db,
            DbUser=redshift_dbuser,
            Sql="CREATE TABLE IF NOT EXISTS "+redshift_table_name+" (title	character varying, rating	int);"
        )
        print(resp)
        return "OK"
    DEFAULT_ARGS = {
        'owner': 'airflow',
        'depends_on_past': False,
        'email': ['[email protected]'],
        'email_on_failure': False,
        'email_on_retry': False
    }
    with DAG(
        dag_id='movie-list-dag',
        default_args=DEFAULT_ARGS,
        dagrun_timeout=timedelta(hours=2),
        start_date=days_ago(2),
        schedule_interval='*/10 * * * *',
        tags=['athena','redshift'],
    ) as dag:
        check_s3_for_key = S3KeySensor(
            task_id='check_s3_for_key',
            bucket_key=s3_key,
            wildcard_match=True,
            bucket_name=s3_bucket_name,
            s3_conn_id='aws_default',
            timeout=20,
            poke_interval=5,
            dag=dag
        )
        files_to_s3 = PythonOperator(
            task_id="files_to_s3",
            python_callable=download_zip
        )
        create_athena_movie_table = AWSAthenaOperator(task_id="create_athena_movie_table",query=create_athena_movie_table_query, database=athena_db, output_location='s3://'+s3_bucket_name+"/"+athena_results+'create_athena_movie_table')
        create_athena_ratings_table = AWSAthenaOperator(task_id="create_athena_ratings_table",query=create_athena_ratings_table_query, database=athena_db, output_location='s3://'+s3_bucket_name+"/"+athena_results+'create_athena_ratings_table')
        create_athena_tags_table = AWSAthenaOperator(task_id="create_athena_tags_table",query=create_athena_tags_table_query, database=athena_db, output_location='s3://'+s3_bucket_name+"/"+athena_results+'create_athena_tags_table')
        join_athena_tables = AWSAthenaOperator(task_id="join_athena_tables",query=join_tables_athena_query, database=athena_db, output_location='s3://'+s3_bucket_name+"/"+athena_results+'join_athena_tables')
        create_redshift_table_if_not_exists = PythonOperator(
            task_id="create_redshift_table_if_not_exists",
            python_callable=create_redshift_table
        )
        clean_up_csv = PythonOperator(
            task_id="clean_up_csv",
            python_callable=clean_up_csv_fn,
            provide_context=True
        )
        transfer_to_redshift = PythonOperator(
            task_id="transfer_to_redshift",
            python_callable=s3_to_redshift,
            provide_context=True
        )
        check_s3_for_key >> files_to_s3 >> create_athena_movie_table >> join_athena_tables >> clean_up_csv >> transfer_to_redshift
        files_to_s3 >> create_athena_ratings_table >> join_athena_tables
        files_to_s3 >> create_athena_tags_table >> join_athena_tables
        files_to_s3 >> create_redshift_table_if_not_exists >> transfer_to_redshift
```


- different tasks are created using operators like `PythonOperator`, for generic Python code, or `AWSAthenaOperator`, to use the integration with [Amazon Athena](https://aws.amazon.com/athena).
- To see how those tasks are connected in the workflow, you can see the latest few lines

```py
    check_s3_for_key >> files_to_s3 >> create_athena_movie_table >> join_athena_tables >> clean_up_csv >> transfer_to_redshift
    files_to_s3 >> create_athena_ratings_table >> join_athena_tables
    files_to_s3 >> create_athena_tags_table >> join_athena_tables
    files_to_s3 >> create_redshift_table_if_not_exists >> transfer_to_redshift
```

- The Airflow code is [overloading](https://docs.python.org/3/reference/datamodel.html#emulating-numeric-types) the right shift `>>` operator in Python to create a dependency,
- meaning that the task on the left should be executed first, and the output passed to the task on the right. L
- Each of the four lines above is adding dependencies, and all evaluated together to execute the tasks in the right order.

In the Airflow console
- a **graph view** of the DAG to have a clear representation of how tasks are executed:

![pic](https://d2908q01vomqb2.cloudfront.net/da4b9237bacccdf19c0760cab7aec4a8359010b0/2020/11/17/mwaa-graph-view-1024x466.png)

.
