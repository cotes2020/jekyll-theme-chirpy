


- [Ali ACR (ACR)](#ali-acr-acr)
  - [basic](#basic)
  - [Features](#features)
  - [access control](#access-control)
    - [user access control](#user-access-control)
    - [network access control](#network-access-control)

# Ali ACR (ACR)


## basic

- Alibaba Cloud ACR is a secure platform that allows you to effectively manage and distribute cloud-native artifacts that meet the standards of **Open Container Initiative (OCI)**.
- The artifacts include container images and Helm charts.
- ACR Enterprise Edition provides end-to-end acceleration capabilities to support
  - global image synchronization,
  - distribution of large images at scale,
  - and image building based on multiple code sources.
- The service seamlessly integrates with **Container Service for Kubernetes (ACK)** to help enterprises reduce delivery complexity and provides a one-stop solution for cloud-native applications.



## Features

**Features of ACR Personal Edition**
- Multi-architecture images
  - supports container images that are based on multiple architectures, including Linux, Windows, and ARM.
- Various regions
  - create and delete repositories in different regions based on your business requirements.
  - Each repository has three endpoints, which can be accessed over the `Internet, internal network, and a virtual private cloud (VPC)`.
- Image scanning
  - scan images for security risks and provides detailed information about image layers.
  - After an image is scanned, provides a vulnerability report for the image. The report includes detailed vulnerability information, such as the vulnerability number, the vulnerability severity, and the version in which the vulnerability is fixed.

**Features of ACR Enterprise Edition**
- OCI artifact management
  - can manage multiple types of OCI artifacts, such as container images that are based on multiple architectures (such as Linux, Windows, and ARM), and charts of Helm v2 and Helm v3.
- Multi-dimensional security protection
  - ensures storage and content security by storing cloud-native application artifacts after encryption,
  - supports image scanning to detect vulnerabilities, and generates vulnerability reports from multiple perspectives.
  - ensures secure access by providing `network access control` and `fine-grained operation audit` for container images and Helm charts.
- Accelerated application distribution
  - can synchronize container images across different regions around the world to improve distribution efficiency.
  - supports image distribution in P2P mode to accelerate application deployment and expansion.
- Efficient and secure cloud-native application delivery
  - allows you to create cloud-native application delivery chains that are observable, traceable, and configurable.
  - can automatically deliver applications all over the world upon source code changes in multiple scenarios based on delivery chains and blocking rules.
  - This improves the efficiency and security of cloud-native application delivery.


![Screen Shot 2022-03-03 at 09.10.23](https://i.imgur.com/lIbuvVm.png)





## access control


### user access control

- Access credentials ensure the secure upload and download of container images and Helm charts.
- Access credentials are available in two types:
  - Password: A password is valid permanently. Keep it safe. If the password is lost, you can reset it.
  - Temporary token: A temporary token is valid for an hour. If the temporary token is obtained by using Security Token Service (STS), the temporary token is valid so long as the STS token is valid.


1. Log on to the ACR instance.
Configure access over the Internet or virtual private clouds (VPCs).
1. Use the access credential to log on to the ACR instance: `docker login <Name of the ACR instance>-registry.<Region>.cr.aliyuncs.com`



### network access control

- By default, a newly created ACR instance is disconnected from all networks.
- You must configure access control lists (ACLs) to allow access to the `ACR instance` over **Virtual Private Clouds (VPCs)** or the **public network**.




**Configure access over VPCs**
- If your `Elastic Compute Service (ECS) instances` reside in one or more virtual private clouds (VPCs), you must configure access to your `ACR instance` over the VPCs.
- Then, the ECS instances in the VPCs can connect to the ACR instance.



- After you configure access to the ACR instance over VPCs, the instance occupies an `IP address in each VPC`.
  - You can use the `internal domain nam`e of the instance to access this instance over a VPC: only when the internal domain name is resolved to the IP address occupied by the instance in the VPC.
  - ACR uses `PrivateZone` to automatically configure domain name resolution.

- can select a random vSwitch or vSwitch that has sufficient IP addresses. After the settings are complete, all ECS instances in the VPC can access the ACR instance by using the internal domain name.

- When you configure access to ACR instance over VPCs, ACR automatically creates the service-linked role `AliyunServiceRoleForContainerRegistryAccessCustomerPrivateZone` for `PrivateZone` to resolve the domain names of the ACR instance.

- After the VPC is added, ACR automatically creates a **resolution zone** in `PrivateZone` to resolve the domain name of the ACR instance.
  - You can view the resolution zone in PrivateZone.
  - Log on to the Alibaba Cloud DNS console > PrivateZone > Hosted Zones tab, view the resolution zone.

.
