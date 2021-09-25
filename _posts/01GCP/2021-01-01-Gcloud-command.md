---
title: GCP - Gcloud Commnad
date: 2021-01-01 11:11:11 -0400
categories: [01GCP]
tags: [GCP]
toc: true
image:
---

- [GCP - Gcloud Command](#gcp---gcloud-command)
  - [Network](#network)
  - [firewall](#firewall)
  - [instance](#instance)
  - [VPN](#vpn)
  - [storage](#storage)
  - [Cloud SQL](#cloud-sql)

---


# GCP - Gcloud Command


## Network

PASSWD

```bash
# Creating network
gcloud compute networks create my_NETWORK \
    --subnet-mode=auto/custom
    --bgp-routing-mode=DYNAMIC_ROUTING_MODE \
    --mtu=MTU

gcloud compute networks update my_NETWORK \
    --switch-to-custom-subnet-mode
    --bgp-routing-mode=DYNAMIC_ROUTING_MODE

gcloud compute networks delete my_NETWORK

gcloud compute networks describe my_NETWORK

# List the networks in project
gcloud compute networks list
# NAME             SUBNET_MODE  BGP_ROUTING_MODE  IPV4_RANGE     GATEWAY_IPV4
# custom-network   CUSTOM       REGIONAL
# default          AUTO         REGIONAL
# legacy-network1  LEGACY       REGIONAL          10.240.0.0/16  10.240.0.1





gcloud compute networks subnets create my_SUBNET \
    --network=my_NETWORK \
    --range=PRIMARY_RANGE \
    --region=REGION

gcloud compute networks subnets delete my_SUBNET \
    --region=REGION

# list all subnets in all VPC networks, in all regions:
gcloud compute networks subnets list

gcloud compute networks subnets list \
   --network=NETWORK

gcloud compute networks subnets list \
   --filter="region:( REGION … )"

gcloud compute networks subnets describe my_SUBNET \
   --region=REGION




gcloud compute networks subnets expand-ip-range my_SUBNET \
  --region=REGION \
  --prefix-length=PREFIX_LENGTH

gcloud compute networks subnets update my_SUBNET \
  --region=REGION \
  --add-secondary-ranges=SECONDARY_RANGE_NAME=SECONDARY_RANGE

gcloud compute networks subnets update my_SUBNET \
  --region=REGION \
  --remove-secondary-ranges=SECONDARY_RANGE_NAME



# change MTU
gcloud compute instances stop INSTANCE_NAMES... \
    --zone=ZONE

gcloud compute networks update my_NETWORK \
    --mtu=MTU

gcloud compute instances start INSTANCE_NAMES... \
    --zone=ZONE
```


## firewall

```bash
gcloud compute firewall-rules create privatenet-allow-icmp-ssh-rdp \
          --direction=INGRESS \
          --priority=1000 \
          --network=privatenet \
          --action=ALLOW \
          --rules=icmp,tcp:22,tcp:3389 \
          --source-ranges=0.0.0.0/0


gcloud compute firewall-rules list \
  --sort-by=NETWORK
# NAME                              NETWORK        DIRECTION  PRIORITY  ALLOW                        
# managementnet-allow-icmp-ssh-rdp  managementnet  INGRESS    1000      icmp,tcp:22,tcp:3389
# mynetwork-allow-icmp              mynetwork      INGRESS    1000      icmp
# mynetwork-allow-internal          mynetwork      INGRESS    65534     all                         
# mynetwork-allow-rdp               mynetwork      INGRESS    1000      tcp:3389
# mynetwork-allow-ssh               mynetwork      INGRESS    1000      tcp:22
# privatenet-allow-icmp-ssh-rdp     privatenet     INGRESS    1000      icmp,tcp:22,tcp:3389
```



x
## instance


```bash
gcloud compute instances create privatenet-us-vm \
    --zone=us-central1-c \
    --machine-type=f1-micro \
    --subnet=privatesubnet-us \
    --image-family=debian-10 \
    --image-project=debian-cloud \
    --boot-disk-size=10GB \
    --boot-disk-type=pd-standard \
    --boot-disk-device-name=privatenet-us-vm


gcloud compute instances list \
    --sort-by=ZONE
# NAME                 ZONE            MACHINE_TYPE  PREEMPTIBLE  INTERNAL_IP  EXTERNAL_IP    STATUS
# mynet-eu-vm          europe-west1-c  n1-standard-1              10.132.0.2   34.76.115.41   RUNNING
# managementnet-us-vm  us-central1-c   f1-micro                   10.130.0.2   35.239.68.123  RUNNING
# mynet-us-vm          us-central1-c   n1-standard-1              10.128.0.2   35.202.101.52  RUNNING
# privatenet-us-vm     us-central1-c   f1-micro                   172.16.0.2   34.66.197.202  RUNNING


gcloud compute ssh my-vm \
  --zone us-central1-c \
  --tunnel-through-iap

gcloud auth activate-service-account --key-file credentials.json


```


## VPN


```bash
# Create the vpn-1 gateway and tunnel1to2
gcloud compute target-vpn-gateways create vpn-1 \
  --project=qwiklabs-gcp-04-6d5e14c9499f \
  --region=us-central1 \
  --network=vpn-network-1

gcloud compute forwarding-rules create vpn-1-rule-esp \
  --project=qwiklabs-gcp-04-6d5e14c9499f \
  --region=us-central1 \
  --address=35.184.104.113 \
  --ip-protocol=ESP \
  --target-vpn-gateway=vpn-1

gcloud compute forwarding-rules create vpn-1-rule-udp500 \
  --project=qwiklabs-gcp-04-6d5e14c9499f \
  --region=us-central1 \
  --address=35.184.104.113 \
  --ip-protocol=UDP \
  --ports=500 \
  --target-vpn-gateway=vpn-1

gcloud compute forwarding-rules create vpn-1-rule-udp4500 \
  --project=qwiklabs-gcp-04-6d5e14c9499f \
  --region=us-central1 \
  --address=35.184.104.113 \
  --ip-protocol=UDP \
  --ports=4500 \
  --target-vpn-gateway=vpn-1

gcloud compute vpn-tunnels create tunnel1to2 \
  --project=qwiklabs-gcp-04-6d5e14c9499f \
  --region=us-central1 \
  --peer-address=104.199.49.195 \
  --shared-secret=PASSWD \
  --ike-version=2 \
  --target-vpn-gateway=vpn-1


gcloud compute routes create tunnel1to2 \
  --project=qwiklabs-gcp-04-6d5e14c9499f \
  --network=vpn-network-1 \
  --priority=1000 \
  --destination-range=10.1.3.0/24 \
  --next-hop-vpn-tunnel=tunnel1to2 \
  --next-hop-vpn-tunnel-region=us-central1



gcloud compute routes create tunnel2to1 \
  --project=qwiklabs-gcp-04-6d5e14c9499f \
  --network=vpn-network-2 \
  --priority=1000 \
  --destination-range=10.5.4.0/24 \
  --next-hop-vpn-tunnel=tunnel2to1 \
  --next-hop-vpn-tunnel-region=europe-west1



```


---


## storage


```bash
gsutil cp gs://cloud-training/gcpnet/private/access.svg gs://my-gfdsxcvbnm

# copy the image from bucket
gsutil cp gs://my-gfdsxcvbnm/*.svg .



gsutil acl set private gs://$BUCKET_NAME_1/setup.html



gsutil acl ch -u AllUsers:R gs://$BUCKET_NAME_1/setup.html

gsutil acl get gs://$BUCKET_NAME_1/setup.html  > acl2.txt



gsutil lifecycle set life.json gs://$BUCKET_NAME_1
gsutil lifecycle get gs://$BUCKET_NAME_1
# {
#   "rule":
#   [
#     {
#       "action": {"type": "Delete"},
#       "condition": {"age": 31}
#     }
#   ]
# }


gsutil versioning set on gs://$BUCKET_NAME_1
gsutil versioning get gs://$BUCKET_NAME_1

gsutil ls -a gs://$BUCKET_NAME_1/setup.html



gsutil rsync -r ./firstlevel gs://$BUCKET_NAME_1/firstlevel

export BUCKET_NAME_2=dfghgfdswrxfgc
```








## Cloud SQL


```bash
# Download the Cloud SQL Proxy and make it executable:
wget https://dl.google.com/cloudsql/cloud_sql_proxy.linux.amd64 -O cloud_sql_proxy && chmod +x cloud_sql_proxy

# To activate the proxy connection to your Cloud SQL database and send the process to the background, run the following command:
./cloud_sql_proxy -instances=$SQL_CONNECTION=tcp:3306 &
Listening on 127.0.0.1:3306 for [SQL_CONNECTION_NAME]
Ready for new connections

To find the external IP address of your virtual machine, query its metadata:
# curl -H "Metadata-Flavor: Google" http://169.254.169.254/computeMetadata/v1/instance/network-interfaces/0/access-configs/0/external-ip && echo
```


















.