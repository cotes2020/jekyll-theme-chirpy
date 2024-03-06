

- [OSS](#oss)
  - [traditional storage](#traditional-storage)
    - [File storage](#file-storage)
    - [block Storage.](#block-storage)
  - [OSS The Object Storage Service](#oss-the-object-storage-service)
  - [OSS vs traditional storage services](#oss-vs-traditional-storage-services)
  - [Security](#security)
  - [Costs](#costs)
  - [Access](#access)
- [bucket](#bucket)
  - [endpoint](#endpoint)
  - [Access control list/ACL](#access-control-listacl)
  - [storage class](#storage-class)
  - [Regions](#regions)
  - [Objects](#objects)
- [security](#security-1)
  - [Bucket Policy](#bucket-policy)
    - [Hotlink Protection.](#hotlink-protection)
    - [Access Keys](#access-keys)
  - [Server-Side Encryption or SSE](#server-side-encryption-or-sse)


---

## OSS

---

### traditional storage

2 main types of traditional storage available in Alibaba Cloud.
- file storage and block storage.


#### File storage

- also known as networked storage is based on a shared file system.
- This type of storage gives multiple clients the ability to access the same shared data across a network. The interface for this is generally on the client side. The two most popular protocols for access in this type of storage is NFS and SMB.


#### block Storage.

- high performance, low latency block storage service for Alibaba Cloud ECS virtual machines.
- And it supports random or sequential read and write operations.
- Block storage is similar to a physical disc.
  - You can format a block storage device and create a file system on it to meet the data storage needs of your business.
  - File and block storage services are structured data services
- priced based on the end user defined in the capacity required. pay for what you provision.

---

### OSS The Object Storage Service
- a service that enables you to store, back up and archive any amount of non-structured data such as images, videos, documents in the cloud.
- Unlike a structured file service, where you would navigate to a file through its directory structure, files in OSS are uploaded into a container and each file has its own unique address to access it.

- cost effective, highly secure, easily scalable and highly reliable cloud storage solution.
  - store and retrieve any type, any time and from anywhere.
  - use API operations and SDKs provided by Alibaba cloud or OSS migration tools to transfer massive amounts of data into or out of Alibaba Cloud's OSS.

---


### OSS vs traditional storage services
- reliability.
  - OSS offers up to 99.995% service availability to protect against service outages and up to 12 nines of data durability to keep your data safe.
  - It offers automatic scaling without effecting external services.
  - It also offers automatic redundant data backup.
  - And with the optional cross-region replication, it can support automatic failover. So redundancy.

- redundancy
  - There are two types of redundancy available in OSS.
  - Local redundant storage/LRS and zone redundant storage/ZRS.
  - LRS stores the data of each object on multiple devices in the same region which ensures data durability and availability in case hardware failure.
  - ZRS distributes user data across three zones within the same region. Even if one zone becomes unavailable, your data will still be accessible.
  - The integrity of data is periodically checked to discover data damage caused by factors such as hardware failure. OSS reconstructs and repairs damaged data by using redundant data.

---

### Security
- OSS provides enterprise grade multilevel security and denial of service attack protection. It supports multi-user resource isolation and remote disaster recovery.
- It also provides authentication authorization, IP address, blacklist and whitelist support and Resource Access Management/RAM account features. And provides comprehensive logging to help trace malicious access.

---

### Costs
- OSS charges fees based on actual usage. The fees incurred within an hour are billed in the next hour. Fees are calculated based on the formula `fees = actual usage * unit price`. And the term actual usage is based on the volume of storage used, the amount of data transferred and the number of API requests made. There are no upfront costs and uploading data into OSS is free of charge. It's easy to use.

---

### Access
- OSS provides a standard restful API interface, a wide range of SDK client tools and a web based console. You can easily upload, download, retrieve, and manage massive amounts of data for websites and applications in the same way as for regular files in windows.
- There is no limit on the number of files.
- File sizes can be from one bite to a maximum size of 48.8 terabytes for a single file.
- The maximum size however, is dependent on the method use to upload.
- And unlike traditional hardware storage, OSS enables you to easily scale up or expand your storage space as needed.


- It supports streaming upload and download which is suitable for business scenarios.
  - For example, where you need to simultaneously read and write videos and other large files.

- And it offers lifecycle management. You can delete expired by data in batches or transition the data to lower cost archive services.


advantages include
- image processing, which supports format conversion, thumbnails, cropping, watermarks, scaling and other operations.
- Audio video transcoding, which provides high quality, high speed, parallel audio/video transcoding capabilities for audio/video files stored in OSS.
- And Alibaba's content delivery network can be used to speed up the delivery of content stored in OSS.

---


## bucket

Hello, in this session we will focus on the basic concepts of the Object Storage Service. The first thing we will look at is the concept buckets. A bucket is a container for objects stored in OSS. Every object that is uploaded must be placed into a bucket. All objects or files are directly related to their corresponding bucket. OSS lacks the hierarchical structure of directories and subdirectories as in a file system, its file system is flat.

- A bucket can contain an unlimited number of objects, the bucket size is infinite.
- A bucket name must be globally unique within OSS.
  - No two bucket names in the world can be the same.
  - The naming conventions for buckets are as follows.
    - Names can only contain lowercase letters, numbers and hyphens.
    - The name must start and end with a lowercase letter or number
    - must be between three and 63 characters in length.

---

### endpoint

To access a bucket, an `endpoint` for the bucket is created automatically. An endpoint is the `domain name` used to access the bucket. OSS provides external services through `HTTPS, HTTP, and RESTful APIs`. Each region has its own dedicated endpoint. Access to a bucket through an intranet connection uses a different endpoint than when accessing the same bucket through the internet.

For example, shown here, these endpoints for the UK London region for intranet and internet are slightly different.
- The `intranet endpoint`, `oss-eu-west-1-internal.aliyuncs.com` is slightly different from the `internet endpoint` in as much as it has the hyphen internal as part of the address.
- To access the contents of a bucket once it is created, the endpoint address is used to navigate to the bucket but access control must also be set at the bucket level.
- This is achieved by setting the `access control list/ACL`.

---

### Access control list/ACL

3 options to choose from:
- `private`
  - only the owner or authorized users of the bucket can read and write files in the bucket.
- `Public read`
  - only the owner or authorized users of this bucket can write files in the bucket, other users including anonymous users can only read files in the bucket.
- `public read/write`
  - Any users including anonymous users can read and write files in the bucket.

The access control lists settings can be changed after the bucket is created.

---

### storage class
For billing purposes, a bucket must utilize a storage class.
Storage classes control the cost of storage. There are currently four storage classes available.

![Screen Shot 2021-09-16 at 10.03.35 PM](https://i.imgur.com/aIeQ64E.png)

- `Standard`
  - highly reliable, highly available and high performance object storage services that can handle frequent data access.
  - This is the default selection when creating a bucket
  - redundancy: supports local redundant storage and zone redundant storage

- `Infrequent access`.
  - storing objects with long life cycles that do not need to be frequently accessed,
    - an average of once or twice per month.
  - IA storage offers a storage unit price lower than that of standard storage, and is suitable for long term backup of various mobile apps, smart device data and enterprise data.
  - It also supports real time data access.
  - minimum storage period: 30 days
    - delete an object will be charged an early deletion fee
  - retrieving data also incurs fees.
  - redundancy, supports local redundant storage and zone redundant storage.

- Archive.
  - OSS archive storage is suitable for storing objects with long life cycles, at least half a year that are in frequently accessed.
  - Data can be restored in about a minute and then read.
  - This storage option is suitable for data such as archival data, medical images, scientific material, and video footage.
  - minimum storage period: 60 days
    - delete an object will be charged an early deletion fee
  - retrieving data also incurs fees.
  - redundancy, only supports local redundant storage.

- `Cold archive`
  - storing extremely cold data with ultra-long life cycles.
    - suitable for data that must be retained for an extended period of time due to compliance requirements.
  - Data can be restored but it depends on the `retrieval level` selected when cold archived.
  - 3 data retrieval levels,
    - `expedited`, restored within `1 hour`,
    - `standard`, restored within `2 to five 5`,
    - `bulk`, restored within `5 to 11 hours`.
  - minimum storage period: 180 days
    - delete an object will be charged an early deletion fee
  - retrieving data also incurs fees.
  - redundancy, only supports local redundant storage.
  - At the time of this recording, May, 2020, cold archive is currently in preview. The following table shows a comparison of the different storage classes.

Once a bucket is created, the name and the storage class cannot be modified.


### Regions

- Regions. A region represents the physical location of a data center. You can choose the region where OSS will store the buckets you create.
- You may choose a region to optimize latency, minimize costs, or address regulatory requirements.
- Generally, the closer the user is in proximity to a region, the faster the access speed is.
- All objects that are contained in a bucket are stored in the same region.
- A region is chosen when a bucket is created and **can not be changed** once it's created.


### Objects
Objects. Objects, also known as files are the fundamental entities stored in OSS.
- An object is composed of three elements, a key, its data and metadata.
- The key is the unique object's name,
- the data is the file content
- metadata defines the attributes of an object, such as the time created and the object size.
- The lifecycle of an object starts when it is uploaded and ends when it's deleted.
  - During the lifecycle of an object, its contents cannot be changed.

If you want to modify an object, you must upload a new object with the same name as the existing one to replace it.
- Therefore, unlike the file system, OSS does not allow users to modify objects directly.
- OSS provides an Append Upload function, which allows you to continually append data to the end of an object.

- There are some limitations to objects stored in OSS.
- The naming conventions for objects are as follows.
  - must use UTF-8 encoding,
  - must be between one and 1023 characters in length.
  - start with a backslash or a forward slash.

File size limitations depend on how the data was uploaded.
- Using Object mode, the file size cannot exceed five gigabytes,
- using Multipart mode, the file size cannot exceed 48.8 terabytes.
- You can upload or delete up to 100 objects at a time from the console.
- To upload or delete more than 100 objects at a time, you must call an API operation or use an SDK.
- When uploading a file to a bucket, the `access control list` that was set at the bucket level will be selected by default but this can be changed prior to upload and after the file is uploaded
- once an object is deleted, it cannot be restored.


## security

There are three main ways to set security for protecting objects in OSS.

Number one: Access Control. You can use Access Control in the following ways. Access Control Lists or ACLs. With an Access Control List you can define the type of access allowed for a bucket, and the objects that reside within the bucket.

The following settings are available.
- Private: Only the owner or authorized users of the bucket can read and write files in the bucket.
- Public Read: Only the owner or authorized users of this bucket can write files in the bucket. Other users, including anonymous users, can only read files in the bucket.
- Public Read\Write: Any users, including anonymous users, can read and write files in the bucket.


### Bucket Policy
You can use Bucket Policy. This allows you to grant permission on all, or just Specific resources in a bucket to RAM users from your Alibaba Cloud account, other accounts, or anonymous accounts. Conditional access can also be set.
- You can select whether objects can be accessed by HTTP or HTTPS only.
- Every object in OSS is enabled with HTTPS access by default.
- This provides secure uploads and downloads via SSL-encrypted endpoints.

You can also set `whitelist or blacklist IP addresses` to further restrict access to bucket contents.



#### Hotlink Protection.
- Hotlink Protection uses an HTTP Referer whitelist to prevent unauthorized users from accessing your data in OSS.
- The Referer Whitelist specifies the domains are allowed to access OSS resources.

#### Access Keys
Access Keys. An Access Key is composed of a Key Id and a Key Secret. They work in pairs to perform access identity verification.
- OSS verifies the identity of a request sender by using `symmetric encryption`.
- The Access Key Id is used to identify a user,
- and the Access Key Secret is used for the user to encrypt the signature and for OSS to verify the signature.
- In OSS, Access Keys are generated by the following three methods:
  - The bucket owner applies for Access Keys.
  - The bucket owner uses **Resource Access Management** to authorize a third party to apply for Access Keys.
  - the bucket owner uses the **Security Token Service** to authorize a third party to apply for Access Keys.



### Server-Side Encryption or SSE

The second method of security is Server-Side Encryption or SSE. OSS supports server-side encryption for uploaded data when enabled. When you upload data, OSS encrypts and stores the data.
- When you download data, OSS automatically decrypts the data and returns the original data to the user.
- The returned HTTP request header declares that the data has been encrypted on the server.

SSE can be implemented in one of two ways:
- The first is Key Management Services or KMS.
  - This implements Server-Side Encryption with a Customer Master Key, CMK, which is stored in KMS.
  - When uploading an object, you can use a CMK ID stored in KMS to encrypt and decrypt large amounts of data.
  - This method is cost-effective because you do not need to send user data to the KMS server through networks for encryption and decryption.
  - KMS requires activation before it can be used.
- And the second is Advanced Encryption Standard or AES256.
  - This implements server-side encryption with OSS-managed keys.
  - This encryption method is an attribute of objects.
  - In this method, OSS server-side encryption uses AES-256 to encrypt objects with different data keys.
  - Master keys are used to encrypt data,
  - and keys are rotated regularly.
  - This method is suited to encrypt and decrypt multiple objects at the same time.
- And three: Identity Authentication using either Resource Access Management/RAM, or the Security Token Service/STS.
  - We can use these features to make sure that we only grant privileges to specific users or temporary privileges to anonymous users.

The RAM Console
- The resource access management console, manages user identities and permissions to access resources.
- You can manage users by configuring RAM policies.
- For users such as employees, systems, or applications, you can control which resources are accessible.
- RAM applies to scenarios where multiple users in an enterprise must collaboratively manage cloud resources.
- RAM allows you to grant RAM users the minimum permissions.
- In this case, you do not need to share your Alibaba Cloud account and password. This method helps you minimize security risks.


Security Token Service/STS
- STS is a cloud service that provides short-term access control for Alibaba Cloud accounts, or RAM users.
- Through STS, you can issue an access credential with custom time limits and access rights to federated users.
- STS is implemented by `command line, SDKs, or the RAM Console`.
