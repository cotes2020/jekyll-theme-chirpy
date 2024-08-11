---
title: GCP - Cloud Storage
date: 2021-01-01 11:11:11 -0400
categories: [01GCP, Storage]
tags: [GCP]
toc: true
image:
---

- [Cloud Storage](#cloud-storage)
  - [basic](#basic)
  - [Google Cloud Storage](#google-cloud-storage)
  - [cloud storage level](#cloud-storage-level)
    - [buckets](#buckets)
    - [object](#object)
  - [Access](#access)
    - [access control lists - ACLs](#access-control-lists---acls)
    - [Object Versioning](#object-versioning)
    - [lifecycle policy](#lifecycle-policy)
    - [signed URLs](#signed-urls)
    - [storage class](#storage-class)
      - [Autoclass](#autoclass)
      - [Standard](#standard)
        - [Multi-regional storage `99.95 percent`](#multi-regional-storage-9995-percent)
        - [Regional storage `99.9 percent Availability`](#regional-storage-999-percent-availability)
      - [Nearline storage `99 percent Availability`](#nearline-storage-99-percent-availability)
      - [Coldline storage `99 percent Availability`](#coldline-storage-99-percent-availability)
      - [Archive storage](#archive-storage)
  - [bring data into cloud storage.](#bring-data-into-cloud-storage)
  - [Deployment](#deployment)

---

# Cloud Storage

---

## basic

![Screen Shot 2022-08-15 at 00.15.39](https://i.imgur.com/hYLjiss.jpg)

![Screen Shot 2022-08-15 at 00.17.10](https://i.imgur.com/KcFVzIf.jpg)

- <font color=OrangeRed> flat structure </font>


## Google Cloud Storage

![Screen Shot 2021-02-03 at 23.55.15](https://i.imgur.com/q9OtPjX.png)

![Screen Shot 2021-06-29 at 12.46.24 AM](https://i.imgur.com/ZQYA7Ol.png)

![Screen Shot 2021-06-29 at 12.47.20 AM](https://i.imgur.com/lyojjd3.png)

- a fully managed scalable service.
  - don't need to provision capacity ahead of time.

- often the ingestion point for data being moved into the cloud
  - allows world-wide storage and retrieval of any amount of data at any time

- the long-term storage location for data.

- Cloud Storage is comprised of buckets and configure and use to hold the storage objects.
  - Object storage manages data as "objects" rather than a file and folder hierarchy or chunks of a disk.

- Cloud storage can
  - store App Engine logs, cloud data store backups, and objects used by App Engine applications like images.
  - store instant startup scripts, Compute Engine images, and objects used by Compute Engine applications.
  - serving website content,
  - storing data for archival and disaster recovery,
  - distributing large data objects to users via direct download.

- Cloud Storage’s primary use is whenever `binary large-object storage (BLOB)` is needed for online content such as videos and photos, for backup and archived data and for storage of intermediate results in processing workflows.

- Cloud Storage is not:
  - **not a file system**
    - object storage, each of the objects in Cloud Storage has a URL.
    - file storage, manage the data as a hierarchy of folders.
    - block storage, the operating system manages the data as chunks of disk.
  - would not use Cloud Storage as the root file system of the Linux box.

- Cloud Storage encrypts the data on the server side before it is written to disk
  - you don't pay extra for that.
  - by default, data in-transit is encrypted using HTTPS.
- Once they are in Cloud Storage, you can move them onwards to other GCP storage services.


---

## cloud storage level

![Screen Shot 2021-06-29 at 12.59.04 AM](https://i.imgur.com/XxXhkwV.png)

![Screen Shot 2021-06-29 at 1.00.16 AM](https://i.imgur.com/LfMfZkh.png)

---

### buckets

- buckets.
  - globally unique name.
  - specify a geographic location where the bucket and its contents
    - Pick a location that minimizes latency for the users.
  - and you choose a default storage class.

![Screen Shot 2021-02-03 at 22.37.13](https://i.imgur.com/lXaeAvy.png)

- <font color=OrangeRed> The storage objects </font>
  - immutable,
    - do not edit them in place but instead create new versions
    - turn on object versioning on the buckets
      - keeps a history of modifications.
      - it overrides or deletes all of the objects in the bucket.
      - can list the archived versions of an object,
      - restore an object to an older state
      - or permanently delete a version as needed.
    - don't turn on object versioning,
      - new always overrides old.
  - lifecycle management policies.
    - For example
    - tell Cloud Storage to delete objects older than 365 days.
    - tell it to delete objects created before January 1, 2013
    - or keep only the three most recent versions of each object in a bucket that has versioning enabled.

---

### object

- <font color=OrangeRed> object </font>
  - you save to the storage here
  - you keep this arbitrary bunch of bytes I give you and the storage lets you address it with a unique key.
  - Often these unique keys are in the form of URLs which means object storage interacts nicely with Web technologies.
  - make objects and the service stores them with high durability and high availability.

---

## Access

![Screen Shot 2021-06-29 at 1.01.03 AM](https://i.imgur.com/yh0Jtnq.png)

- **Cloud IAM**

- **Roles**:
  - inherited from project to bucket to object.

- **access control lists - ACLs**
  - offer finer control.
  - ACLs define who has access to the buckets and objects as well as what level of access they have.
  - Each ACL consists of two pieces of information,
    - scope
      - defines who can perform the specified actions,
      - for example, a specific user or group of users
    - permission
      - defines what actions can be performed.
      - For example, read or write.

### access control lists - ACLs

![Screenshot 2024-08-07 at 13.40.27](/assets/img/Screenshot%202024-08-07%20at%2013.40.27.png)

![Screen Shot 2021-06-29 at 1.01.52 AM](https://i.imgur.com/ZVXekXM.png)

---

### Object Versioning

![Screen Shot 2021-06-29 at 1.05.15 AM](https://i.imgur.com/lmWwwgY.png)

![Screenshot 2024-08-07 at 13.39.16](/assets/img/Screenshot%202024-08-07%20at%2013.39.16.png)

![Screen Shot 2021-06-29 at 1.05.41 AM](https://i.imgur.com/LA9i41y.png)

---

### lifecycle policy

![Screenshot 2024-08-07 at 13.41.54](/assets/img/Screenshot%202024-08-07%20at%2013.41.54.png)

![Screen Shot 2021-06-29 at 1.06.52 AM](https://i.imgur.com/5cYKp9r.png)

![Screen Shot 2021-06-29 at 1.08.14 AM](https://i.imgur.com/mWf4JUH.png)

![Screen Shot 2021-06-29 at 1.10.17 AM](https://i.imgur.com/BHr6KMB.png)

---

### signed URLs

![Screen Shot 2021-06-29 at 1.02.47 AM](https://i.imgur.com/P3C6TAA.png)

![Screen Shot 2021-06-29 at 1.02.47 AM](https://i.imgur.com/hi3Ufsj.png)

---

### storage class

![Screenshot 2024-08-07 at 13.43.09](/assets/img/Screenshot%202024-08-07%20at%2013.43.09.png)

4 type of storage class:
- Regional
- Multi-regional
- Nearline
- Coldline

![Screen Shot 2021-06-29 at 12.47.53 AM](https://i.imgur.com/xnLKgAP.png)

![Screen Shot 2021-02-03 at 22.38.07](https://i.imgur.com/zQFaWOA.png)

![Screenshot 2024-08-07 at 13.43.50](/assets/img/Screenshot%202024-08-07%20at%2013.43.50.png)

![Screen Shot 2021-06-29 at 1.12.07 AM](https://i.imgur.com/DeMifDh.png)

- Multi-regional and Regional are high-performance object storage
  - multi region: a large geographic areas such as the United States that contains two or more geographic places.
  - Dual region: a specific pair of regions such as Finland and the Netherlands.
  - A region: a specific geographic place such as London
  - Object stored in a multi region or dual region are geo redundant


- Nearline and Coldline are backup and archival storage.
- All of the storage classes are accessed in comparable ways using the cloud storage API and they all offer millisecond access times.


- pricing
  - all storage classes incur a cost per gigabyte of data stored per month
    - Multi-regional having the highest storage price
    - Coldline the lowest storage price.
  - Egress and data transfer charges may also apply.
  - Nearline storage also incurs an access fee per gigabyte of data read
  - Coldline storage incurs a higher fee per gigabyte of data read.

---

#### Autoclass

![Screenshot 2024-08-07 at 13.44.26](/assets/img/Screenshot%202024-08-07%20at%2013.44.26.png)

---

#### Standard

- fast for data that is frequently accessed and are stored for only brief periods of time.

- most expensive storage class,

- but has no minimum storage duration and no retrieval cost.

used in
- **a region**
  - standard storage is appropriate for storing data in the same location as Google, Kubernetes engine clusters or compute engine instances that use the data.
  - Co locating your resources maximizes the performance for data intensive computations and can reduce network charges.
- **used in dual region**
  - still get optimized performance when accessing Google Cloud products that are located in one of the associated regions.
  - But also get improved availability that comes from storing data in geographically separate locations.
- **used in multi region**
  - appropriate for storing data that is `accessed around the world`.
  - Such as serving website content, stream videos, executing interactive workloads, or serving data supporting mobile and gaming applications

##### Multi-regional storage `99.95 percent`
- cost a bit more
- but it's Geo-redundant.
- you pick a broad geographical location like the United States, the European Union, or Asia and cloud storage stores your data in at least two geographic locations separated by at least 160 kilometers.
- appropriate for storing frequently accessed data.
  - For example,
  - website content, interactive workloads,
  - or data that's part of mobile and gaming applications.

##### Regional storage `99.9 percent Availability`
- store data in a specific GCP region:
  - US Central one, Europe West one or Asia East one.
- cheaper than Multi-regional storage
- but it offers less redundancy.
- to store data close to their Compute Engine, virtual machines, or their Kubernetes engine clusters.
  - gives better performance for data-intensive computations.

---

#### Nearline storage `99 percent Availability`

> better choice than standard storage in scenarios were slightly lower availability, authority day, minimum storage duration, and costs for data access are acceptable tradeoffs for lowered at less storage costs.

- low-cost
- highly durable service
- for **storing infrequently accessed data**.
  - For example,
  - data backup, long tailed multimedia content, and data archiving.
  - when plan to read or modify the data once a month or less on average.
  - continuously add files to cloud storage and access those files once a month for analysis

---

####  Coldline storage `99 percent Availability`

> best choice for data that you plan to access -at most- once a year.
> better choice than standard storage or nearline storage. In scenarios where slightly lower availability, a 90 day minimum storage duration and higher costs for data access are acceptable tradeoffs for lowered address storage costs.

- very low cost
- highly durable service
- **storing infrequently accessed data**
  - for data archiving, online backup, and disaster recovery
- due to its slightly lower availability,
- 90-day minimum storage duration,
- costs for data access, and higher per operation costs.
- For example,
  - to archive data or have access to it in case of a disaster recovery event.

---

#### Archive storage

> best choice for data that you plan to access less than once a year.

- the lowest cost
- highly durable storage service
- for data archiving, online backup and disaster recovery.
- data is available within milliseconds, not hours or days.
- Though the typical availability is comparible to nearline and coldline storage. Archive storage also has higher costs for data access and operations as well as a 365 day minimum storage duration.

---

## bring data into cloud storage.

![Screenshot 2024-08-07 at 13.45.53](/assets/img/Screenshot%202024-08-07%20at%2013.45.53.png)

![Screen Shot 2021-02-03 at 23.54.12](https://i.imgur.com/suMHyqx.png)

![Screen Shot 2021-06-29 at 1.09.08 AM](https://i.imgur.com/Vqi3oyV.png)

- bring data into cloud storage.
  - use gsutil
    - the cloud storage command from this cloud SDK.
  - drag and drop in the GCP console in browser.
  - for terabytes or even petabytes of data
    - online storage transfer service
      - schedule and manage batch transfers to cloud storage from another cloud provider from a different cloud storage region or from an HTTPS endpoint.
    - offline transfer appliance
      - a rackable, high-capacity storage server that you lease from Google Cloud.
      - connect it to your network, load it with data, and then ship it to an upload facility where the data is uploaded to cloud storage.
      - securely transfer up to a petabyte of data on a single appliance.

- other ways of getting your data into cloud storage as this storage option is tightly integrated with many of the Google cloud platform products and services.
- For example
  - import and export tables from and to BigQuery as well as Cloud SQL.

---

## Deployment

Lab Setup

create 2 bucket

Bukect1:
- BUCKET_NAME: my_bucket
- Region: us-east4
- how to control access to objects:
  - uncheck Enforce public access prevention on this bucket
  - select Fine-grained.

Create a virtual machine (VM) instance
- Name: first-vm
- Region: us-east4
- Zone: us-east4-b
- Machine type: click e2-micro (2 shared vCPU)
- Firewall: click Allow HTTP traffic.


Create an IAM service account

- Service account name: test-service-account
- Grant this service account access to project page: role as Basic > Editor
- Manage keys: Create new key -> "credentials.json"
- Move the credentials file you created earlier into Cloud Shell


Cloud Shell

```sh
# list all the zones in a given region:
MY_REGION=us-east4
gcloud compute zones list | grep $MY_REGION
# Set default zone
MY_ZONE=us-east4-b
gcloud config set compute/zone $MY_ZONE

# create a second virtual machine
MY_VMNAME=second-vm
gcloud compute instances create $MY_VMNAME \
--machine-type "e2-standard-2" \
--image-project "debian-cloud" \
--image-family "debian-11" \
--subnet "default"

gcloud compute instances list


# create a second service account
gcloud iam service-accounts create test-service-account2 \
  --display-name "test-service-account2"

# grant the second service account the viewer role:
gcloud projects add-iam-policy-binding $GOOGLE_CLOUD_PROJECT \
  --member serviceAccount:test-service-account2@${GOOGLE_CLOUD_PROJECT}.iam.gserviceaccount.com \
  --role roles/viewer
```

- the external IP address of the first VM you created is shown as a link.
  - because this VM's firewall allow HTTP traffic.
  - Click the link. Your browser will present a Connection refused message in a new browser tab. This message occurs because, although there is a firewall port open for HTTP traffic to your VM, no web server is running there.

Download a file to Cloud Shell and copy it to Cloud Storage

```sh
MY_BUCKET_NAME_1=my_bucket
MY_BUCKET_NAME_2=my_bucket2

# create a bucket:
gcloud storage buckets create gs://$MY_BUCKET_NAME_2 --location=us-east4

# Copy a picture of a cat from a Google-provided Cloud Storage bucket to your Cloud Shell:
gcloud storage cp gs://cloud-training/ak8s/cat.jpg cat.jpg
# Copy the file into the first bucket that you created earlier:
gcloud storage cp cat.jpg gs://$MY_BUCKET_NAME_1
# Copy the file from the first bucket into the second bucket:
gcloud storage cp gs://$MY_BUCKET_NAME_1/cat.jpg gs://$MY_BUCKET_NAME_2/cat.jpg

# Set the access control list for a Cloud Storage object

# To get the default access list that's been assigned to cat.jpg
gsutil acl get gs://$MY_BUCKET_NAME_1/cat.jpg  > acl.txt
cat acl.txt
# [
#   {
#     "entity": "project-owners-560255523887",
#     "projectTeam": {
#       "projectNumber": "560255523887",
#       "team": "owners"
#     },
#     "role": "OWNER"
#   },
#   {
#     "entity": "project-editors-560255523887",
#     "projectTeam": {
#       "projectNumber": "560255523887",
#       "team": "editors"
#     },
#     "role": "OWNER"
#   },
#   {
#     "entity": "project-viewers-560255523887",
#     "projectTeam": {
#       "projectNumber": "560255523887",
#       "team": "viewers"
#     },
#     "role": "READER"
#   },
#   {
#     "email": "google12345678_student@qwiklabs.net",
#     "entity": "user-google12345678_student@qwiklabs.net",
#     "role": "OWNER"
#   }
# ]


# change the object to have private access, execute the following command:
gsutil acl set private gs://$MY_BUCKET_NAME_1/cat.jpg

# To verify the new ACL that's been assigned to cat.jpg
gsutil acl get gs://$MY_BUCKET_NAME_1/cat.jpg  > acl-2.txt
cat acl-2.txt
# [
#   {
#     "email": "google12345678_student@qwiklabs.net",
#     "entity": "user-google12345678_student@qwiklabs.net",
#     "role": "OWNER"
#   }
# ]
# Now only the original creator of the object (your lab account) has OWNER access.
```


Authenticate as a service account in Cloud Shell

```sh
gcloud config list
# [component_manager]
# disable_update_check = True
# [compute]
# gce_metadata_read_timeout_sec = 30
# zone = us-east4-b
# [core]
# account = google12345678_student@qwiklabs.net
# disable_usage_reporting = False
# project = qwiklabs-Google Cloud-1aeffbc5d0acb416
# [metrics]
# environment = devshell
# Your active configuration is: [cloudshell-16441]


# change the authenticated user to the first service account
gcloud auth activate-service-account --key-file credentials.json
gcloud config list
# [component_manager]
# disable_update_check = True
# [compute]
# gce_metadata_read_timeout_sec = 30
# zone = us-east4-b
# [core]
# account = test-service-account@qwiklabs-Google Cloud-1aeffbc5d0acb416.iam.gserviceaccount.com
# disable_usage_reporting = False
# project = qwiklabs-Google Cloud-1aeffbc5d0acb416
# [metrics]
# environment = devshell
# Your active configuration is: [cloudshell-16441]


# Because you restricted access to this file
# verify that the current account (test-service-account) cannot access the cat.jpg file
gcloud storage cp gs://$MY_BUCKET_NAME_1/cat.jpg ./cat-copy.jpg
# AccessDeniedException: 403  KiB]


# Verify that the current account (test-service-account) can access the cat.jpg file in the second bucket
gcloud storage cp gs://$MY_BUCKET_NAME_2/cat.jpg ./cat-copy.jpg

# To switch to the lab account, execute the following command
gcloud config set account [USERNAME]
gcloud storage cp gs://$MY_BUCKET_NAME_1/cat.jpg ./copy2-of-cat.jpg


# Make the first Cloud Storage bucket readable by everyone, including unauthenticated users:
gsutil iam ch allUsers:objectViewer gs://$MY_BUCKET_NAME_1


# In the Cloud Shell code editor, select New File.
# Name the file index.html.
# <html><head><title>Cat</title></head>
# <body>
# <h1>Cat</h1>
# <img src="REPLACE_WITH_CAT_URL">
# </body></html>


# SSH to first-vm.
sudo apt-get remove -y --purge man-db
sudo touch /var/lib/man-db/auto-update
sudo apt-get update
sudo apt-get install nginx

# cloud shell
gcloud compute scp index.html first-vm:index.nginx-debian.html --zone=us-east4-b

# copy the HTML file from your home directory to the document root of the nginx web server:
sudo cp index.nginx-debian.html /var/www/html

# Click the link in the External IP column for your first-vm. A new browser tab opens with a webpage that contains the cat image.
```




.
