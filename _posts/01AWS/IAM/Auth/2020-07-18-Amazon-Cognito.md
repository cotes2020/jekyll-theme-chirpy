---
title: AWS - IdenAccessManage - Amazon Cognito
date: 2020-07-18 11:11:11 -0400
categories: [01AWS, IdenAccessManage]
tags: [AWS, IdenAccessManage]
toc: true
image:
---

[toc]

---

# Amazon Cognito

![Cognito](https://i.imgur.com/QqV5Uoe.png)

<font color=red> web identity federation </font>
- allows user to authenticate with a web identity ptovider (google, facebook)
- the user authenticates first with the web id provider and receives and authentication token, and use it to exchanged for temporary AWS credenticals allowing them to assume an IAM role.


Amazon Cognito provide <font color=red> web identity federation </font> with the following features:

1. Simple and Secure <font color=red> User Sign-Up, Sign-In, and get access for guest users </font> to web and mobile apps.

2. provides <font color=red> authentication, authorization, and user management </font> for web and mobile apps .

3. actes as an <font color=blue> identity broker </font> between app and web id providers
   - no additional code
   - no credentials is stored on the devices.

4. <font color=red> Access Control </font>  to AWS resources from your application.
   - define roles and map users to different roles,
   - so application can access only the resources that are authorized for each user.

5. <font color=red> seamless experience </font> across devices with Push Synchronization function.
   - syncs user data for a seamless experience across your devices.
   - use Push Synchronization to send a silent push notification of user data updates to multiple devices associated with a user ID.

6. uses common identity management standards, such as <font color=red> Security Assertion Markup Language (SAML) 2.0 </font>.
   - SAML: open standard for exchanging identity and security information with applications and service providers.
   - Applications and service providers that support SAML enable you to `sign in by using your corporate directory credentials` (user name and password from Microsoft Active Directory)
   - With SAML, you can use single sign-on (SSO) to sign in to all of your SAML-enabled applications by using a single set of credentials.

7. meet multiple security and compliance requirements, including requirements for highly regulated organizations such as healthcare companies and merchants.
   - eligible for use with the `US Health Insurance Portability and Accountability Act (HIPAA)`.
   - for workloads that are compliant with the `Payment Card Industry Data Security Standard(PCI DSS)`; `the American Institute of CPAs (AICPA)`, `Service Organization Control (SOC)`; `the International Organization for Standardization (ISO)` `and International Electrotechnical Commission (IEC)` `standardsISO/IEC 27001`, `ISO/IEC 27017`, and `ISO/IEC 27018`; and `ISO 9001`


---


## <font color=red> Secure and scalable user directory </font>

![scenario-authentication-cup](https://i.imgur.com/R0SLuZ1.png)

![Screen Shot 2020-12-25 at 14.19.01](https://i.imgur.com/UK7uxxF.png)

- The two main components of Amazon Cognito are <font color=red> user pools and identity pools </font>.
  - can use identity pools and user pools separately or together.
  - ![Screen Shot 2020-12-25 at 14.10.07](https://i.imgur.com/nGMQv12.png)
---

### user pools


- <font color=red> secure user directories </font>
  - can scales to hundreds of millions of users.
- <font color=red> provide sign-up and sign-in options </font> for web and mobile app users.
- fully managed service, set up without any worries about standing up server infrastructure.
- user can sign in directly to the user pool, or indirectly via an iderntity provider.


1. Getting Started with User Pools
   - Prerequisite: Sign Up for an AWS Account
   - Step 1. Create a User Pool
     - ![User Pool](https://i.imgur.com/eSKnfun.png)
   - Step 2. Add an App to Enable the Hosted Web UI
     - ![App client](https://i.imgur.com/GewXuXa.png)
     - `callbackURL`: url redirect after user authentication.
     - `Authorization code grant` cognito give back a authorization code which can provide back to the backend authorization process.
     - `Implicit grant`: JWT token
   - Step 3. Add Social Sign-in to a User Pool (Optional)
   - Step 4. Add Sign-in with a SAML Identity Provider to a User Pool (Optional)

---

### Identity pools (federated identities)

- <font color=red> create unique identities </font> for your users
- and assign permissions for users.
- and federate them with identity providers.
- then obtain <font color=blue> temporary, limited-privilege AWS credentials with permissions pre-define </font> to directly access other AWS services or to access resources through Amazon API Gateway.


Identity pool can include:
- Users in an Amazon Cognito user pool
- Users authenticate by external identity providers (Facebook, Google, Apple), or a SAML-based identity provider
- Users authenticated via your own existing authentication process


Amazon Cognito identity pools support the following identity providers:
- Public providers: Login with Amazon (Identity Pools), Facebook (Identity Pools), Google (Identity Pools), Sign in with Apple (Identity Pools).
- Amazon Cognito User Pools
- Open ID Connect Providers (Identity Pools)
- SAML Identity Providers (Identity Pools)
- Developer Authenticated Identities (Identity Pools)

---

#### Identity Pools (Federated Identities) Authentication Flow

**Enhanced (Simplified) Authflow**
- GetId
- GetCredentialsForIdentity

![amazon-cognito-ext-auth-enhanced-flow](https://i.imgur.com/d96X1nI.png)

**Basic (Classic) Authflow**
- GetId
- GetOpenIdToken
- AssumeRoleWithWebIdentity

![amazon-cognito-ext-auth-basic-flow](https://i.imgur.com/yvlAZQW.png)



**Developer Authenticated Identities Authflow**
- When using Developer Authenticated Identities (Identity Pools), the client will use a different authflow that will include code outside of Amazon Cognito to validate the user in your own authentication system.
- Code outside of Amazon Cognito is indicated as such.

Enhanced Authflow
- Login via Developer Provider (code outside of Amazon Cognito)
- Validate the user's login (code outside of Amazon Cognito)
- GetOpenIdTokenForDeveloperIdentity
- GetCredentialsForIdentity

![amazon-cognito-dev-auth-enhanced-flow](https://i.imgur.com/vamWrTi.png)

Basic Authflow
- Login via Developer Provider (code outside of Amazon Cognito)
- Validate the user's login (code outside of Amazon Cognito)
- GetOpenIdTokenForDeveloperIdentity
- AssumeRoleWithWebIdentity

![amazon-cognito-dev-auth-basic-flow](https://i.imgur.com/6OkCmr9.png)


---

## Amazon Cognito Sync

Amazon Cognito Sync is an AWS service and client library that enables <font colore=red> cross-device syncing </font> of application-related user data.
- You can synchronize user profile data across mobile and web without requiring your own backend.
- Amazon Cognito Sync Server & Client


The client libraries cache data locally so your app can read and write data regardless of device connectivity status.
- When the device is online, you can synchronize data, and if you set up `push sync`, notify other devices immediately that an update is available.


step:
1. Sign Up for an AWS Account
2. Set Up an Identity Pool in Amazon Cognito
   - Amazon Cognito Sync requires an Amazon Cognito identity pool to provide user identities. Thus you need to first set up an identity pool before using Amazon Cognito Sync.
3. Store and Sync Data

---


### Synchronizing Data

Amazon Cognito lets you save end user data in datasets containing <font color=red> key-value pairs </font>.
- This data is associated with an Amazon Cognito identity, so that it can be accessed across logins and devices.
- To sync this data between the Amazon Cognito service and an end user’s devices, invoke the synchronize method.
- Each dataset can have a maximum size of 1 MB. You can associate up to 20 datasets with an identity.

The `Amazon Cognito Sync client` creates a <font color=red> local cache </font> for the identity data.
- Your app talks to this local cache when it reads and writes keys.
  - This guarantees that all changes on the device are immediately available on the device,
  - even when offline.
- When the synchronize method is called,
  - <font color=blue> changes from the service </font> are pulled to the device,
  - and <font color=blue> any local changes </font> are pushed to the service.
- At this point the changes are available to other devices to synchronize.


---

### setup Synchronize code

1. <font color=red> Initializing </font> the Amazon Cognito Sync Client
   - first need to create a <font color=blue>  credentials provider </font>
   - The credentials provider acquires temporary AWS credentials to enable your app to access your AWS resources.
   - also need to import the required header files.

2. <font color=red> Understanding Datasets </font>
   - With Amazon Cognito, end user profile data is organized into `datasets`.
   - Each dataset can contain up to 1MB of data in the form of key-value pairs.
   - A dataset is the most granular entity on which you can perform a sync operation.
   - Read and write operations performed on a dataset only affect the local store until the synchronize method is invoked.
   - A dataset is identified by a unique string.
   - You can create a new dataset or open an existing one as shown in the following.

3. delete a dataset
   - first call the method to remove it from local storage,
   - then call the synchronize method to delete the dataset from Amazon Cognito:

4. Reading and Writing Data in Datasets
   - Amazon Cognito datasets function as dictionaries, with values accessible by key.
   - The keys and values of a dataset can be read, added, or modified just as if the dataset were a dictionary.
     - that values written to a dataset only affect the local cached copy of the data until you call the synchronize method.

5. Synchronizing Local Data with the Sync Store
Android
   - The synchronize method compares <font color=blue> local cached data </font> to the <font color=blue> data stored in the Amazon Cognito Sync store </font>
   - Remote changes are pulled from the `Amazon Cognito Sync store`;
   - conflict resolution is invoked if any conflicts occur;
   - and updated values on the device are pushed to the service.



```java
// Android

// to initialize the Amazon Cognito Sync client.
import com.amazonaws.mobileconnectors.cognito.*;
CognitoSyncManager client = new CognitoSyncManager(
    getApplicationContext(),
    Regions.YOUR_REGION,
    credentialsProvider);

// setup Datasets
Dataset dataset = client.openOrCreateDataset("datasetname");

// delete a dataset
dataset.delete();                  // remove it from local storage,
dataset.synchronize(syncCallback); //delete the dataset from Amazon Cognito:

// Reading and Writing Data in Datasets
String value = dataset.get("myKey");
dataset.put("myKey", "my value");

// synchronize a dataset
dataset.synchronize(syncCallback);
// The synchronize method receives an implementation of the SyncCallback interface, discussed below.
// The synchronizeOnConnectivity() method attempts to synchronize when connectivity is available.
// If connectivity is immediately available, synchronizeOnConnectivity() behaves like synchronize().
// Otherwise it monitors for connectivity changes and performs a sync once connectivity is available.
// If synchronizeOnConnectivity()is called multiple times, only the last synchronize request is kept, and only the last callback will fire. If either the dataset or the callback is garbage-collected, this method won't perform a sync, and the callback won't fire.



```


---


### Push Sync

Amazon Cognito automatically tracks the association between identity and devices. Using the push synchronization (push sync) feature
- ensure that every instance of a given identity is notified when identity data changes.
- ensures that whenever the sync store data changes for a particular identity, all devices associated with that identity <font color-red> receive a silent push notification informing them of the change </font>

> Note: Push sync is not supported for JavaScript, Unity, or Xamarin.

---
#### setup push sync

1. Create an Amazon Simple Notification Service (Amazon SNS) App
2. Enable Push Sync in the Amazon Cognito console
   - <kbd>Amazon Cognito console</kbd> > identity pool for which you want to enable push sync > Dashboard > <kbd>Manage Identity Pools</kbd>
   - The <kbd>Federated Identities page</kbd> appears > click <kbd>Push synchronization</kbd> to expand it > <kbd>Service role</kbd> dropdown menu
     - select IAM role that <font color=blue> grants Cognito permission to send an SNS notification </font>
       - configure the IAM roles to have `full SNS access`,
       - or create a new role that `trusts cognito-sync` and has `full SNS access`.
   - Click Create role to create or modify the roles associated with your identity pool in the AWS IAM Console.
   - Select a platform application > Save Changes.
   - Grant SNS Access to Your Application

> Amazon SNS is used to send a silent push notification to all the devies associated with a given user identity whenever data stored in the cloud changed.


---

## AWS AppSync

If new to Amazon Cognito Sync, use AWS AppSync. Like Amazon Cognito Sync, AWS AppSync is a service for synchronizing application data across devices.


It enables user data like app preferences or game state to be synchronized. It also extends these capabilities by allowing multiple users to synchronize and collaborate in real time on shared data.

---


## security

---


### data protection
For data protection purposes, recommend that
- protect AWS account credentials
- set up individual user accounts with AWS IAM.
- given only the permissions necessary for job duties
- Use <font color=red> multi-factor authentication (MFA) </font> with each account.
- Use <font color=red> SSL/TLS </font> to communicate with AWS resources.
- <font color=red> Set up API and user activity logging </font> with AWS CloudTrail.
- Use AWS encryption solutions, along with all default security controls within AWS services.
- Use advanced managed security services such as Amazon Macie, which assists in discovering and securing personal data that is stored in Amazon S3.
- never put sensitive identifying information, such as customers' account numbers, into free-form fields such as a Name field.
  - This includes when you work with Amazon Cognito or other AWS services using the console, API, AWS CLI, or AWS SDKs.
  - Any data that you enter into Amazon Cognito or other services might get picked up for inclusion in diagnostic logs.
  - When you provide a URL to an external server, don't include credentials information in the URL to validate your request to that server.


### use Amazon Resource Names (ARNs)


1. ARNs for Amazon Cognito Federated Identities
   - restrict an IAM user's access to a specific identity pool, using the Amazon Resource Name (ARN) format
   - arn:aws:**cognito-identity**:`REGION:ACCOUNT_ID`:identitypool/`IDENTITY_POOL_ID`


2. ARNs for Amazon Cognito Sync
   - customers can also restrict access by the identity pool ID, identity ID, and dataset name.
   - For APIs that operate on an identity pool, the identity pool ARN format is the same as for Amazon Cognito Federated Identities, except that the service name is cognito-sync instead of cognito-identity:
     - arn:aws:**cognito-sync**:`REGION:ACCOUNT_ID`:identitypool/`IDENTITY_POOL_ID`
   - For APIs that operate on a single identity, such as RegisterDevice, refer to the individual identity by the following ARN format:
     - arn:aws:**cognito-sync**:`REGION:ACCOUNT_ID`:identitypool/`IDENTITY_POOL_ID`/identity/`IDENTITY_ID`
   - For APIs that operate on datasets, such as UpdateRecords and ListRecords, refer to the individual dataset using the following ARN format:
     - arn:aws:**cognito-sync**:`REGION:ACCOUNT_ID`:identitypool/`IDENTITY_POOL_ID`/identity/`IDENTITY_ID`/dataset/`DATASET_NAME`

3. ARNs for Amazon Cognito User Pools
   - restrict an IAM user's access to a specific user pool
   - arn:aws:**cognito-idp**:`REGION:ACCOUNT_ID`:userpool/`USER_POOL_ID`



### Logging in and Monitoring in Amazon Cognito

Monitoring is an important part of maintaining the reliability, availability, and performance of Amazon Cognito and your other AWS solutions.

Amazon Cognito currently supports the following two AWS services
- Amazon CloudWatch Metrics
  - monitor, report and take automatic actions in case of an event in near real time.
  - For example
  - create CloudWatch dashboards on the provided metrics to monitor your Amazon Cognito user pools
  - create CloudWatch alarms on the provided metrics to notify you on breach of a set threshold.

- AWS CloudTrail
  - capture API calls from the Amazon Cognito console and from code calls to the Amazon Cognito API operations.
  - For example
  - when a user authenticates, CloudTrail can record details such as the IP address in the request, who made the request, and when it was made.


### Infrastructure Security in Amazon Cognito

use AWS published API calls to access Amazon Cognito through the network.
- Clients must support Transport Layer Security (TLS) 1.0 or later. We recommend TLS 1.2 or later.
- Clients must also support cipher suites with perfect forward secrecy (PFS) such as Ephemeral Diffie-Hellman (DHE) or Elliptic Curve Ephemeral Diffie-Hellman (ECDHE).
- Most modern systems such as Java 7 and later support these modes.
- requests must be signed using an access key ID and a secret access key that is associated with an IAM principal.
- Or use the AWS Security Token Service (AWS STS) to generate temporary security credentials to sign requests.

### Security Best Practices for Amazon Cognito User Pools

- add <font color=red> multi-factor authentication (MFA) </font> to a user pool to protect the identity of your users. choose to use SMS text messages, or time-based one-time (TOTP) passwords

- user pool <font color=red> advanced security features </font>
  - <font color=red> protections against compromised credentials </font>
    - detect if a user’s credentials (user name and password) have been compromised elsewhere.
    - This can happen when users reuse credentials at more than one site, or when they use passwords that are easy to guess.
  - <font color=red> adaptive authentication </font>
    - use adaptive authentication with its <font color=blue> risk-based model </font> to predict when you might need another authentication factor.
    - configure user pool to block suspicious sign-ins or add second factor authentication in response to an increased risk level.
    - For each sign-in attempt, Amazon Cognito generates a risk score for how likely the sign-in request is to be from a compromised source.
      - This risk score is based on many factors
      - including whether it detects a new device, user location, or IP address.
    - Adaptive Authentication adds MFA based on risk level for users who don't have an MFA type enabled at the user level.
      - When an MFA type is enabled at the user level, those users will always receive the second factor challenge during authentication regardless of how you configured adaptive authentication.

- Amazon Cognito publishes sign-in attempts, their risk levels, and failed challenges to Amazon CloudWatch.









.
