


# ECS

---


## Elastic Compute Service

The first section we are going look at is ECS Concepts. So what is ECS? The Elastic Compute Service (ECS) is a computing service with flexible processing capacity. It is a high-performance, stable, reliable, and scalable IaaS-level solution used to deploy virtual servers known as Instances. IaaS or Infrastructure as a Service is a concept where Alibaba provides and manages the virtualization, servers, storage, and networking.
- And you as the customer select the operating system, install applications and manage your data.

In the diagram, you can see the different responsibilities when you provision a server on premise and when you provision a server within Alibaba Cloud ECS Instances can easily deploy and manage applications with better stability and security, compared to physical servers on premise.

![Screen Shot 2021-09-17 at 3.43.55 PM](https://i.imgur.com/MozAXcw.png)

- ECS Instances provide resizable compute capacity in the cloud. They are designed to make large scale computing easier. You can create instances with a variety of operating systems.
- Alibaba supports most mainstream Linux and Microsoft Windows Server systems.
- And you can run as many or as few instances as you like.

Why use ECS? Unlike provisioning on-premise machines,
- you do not have to purchase any hardware upfront.
- Instances are delivered within minutes, enabling rapid deployment with little or no wait time.
- You can scale and remove resources based on actual business needs.
- ECS instances provide a host of built-in security solutions, such as Virtual Firewalls, Internal network isolation, public IP Access, Anti-Virus and Denial of Service Attack protection.

The Elastic Compute Service is provided via a virtualization layer that is provisioned within the Data Centres around the world.
- The Data Centres contain thousands of racks and this is where the virtualization technology sits.
- Alibaba uses XEN and KVM Virtualisation to provision its ECS Instances. These instances, in turn, run on top of the `X-Dragon Compute Platform` and the Apsara distributed file system called `Pangu`, which provides the storage system.

ECS comprises the following major components:

- `Instance`: A virtual computing environment that includes basic computing components such as CPU, memory, network bandwidth, and disks.
- `Image`: provides the operating system, initial application data, and pre-installed software for instances.
- `Block Storage`: A block storage device based on the Object Storage Service (OSS) which features high performance and low latency distributed cloud disks.
- `Security Groups`: Used by a logical group of instances located in the same region that have the same security requirements and require access to each other.
- `Network`: A logically isolated private cloud network.

---

### Regions

- physical locations with one or more data centers that are spread all over the world to reduce network latency.
- The region is where Alibaba Cloud Services will launch the Instance you create.
- Choose a region to optimize latency, minimize cost or address regulatory requirements.
- There are specific regions in mainland China and other International regions available,
- Having multiple regions around the world means that you can provision servers closer to your users.


---

### Zones

- Zones refer to physical Data Centres within a region that have independent power supplies and networks.
- Users can separate ECS Instances into different zones in a region to facilitate, for example, ‘High Availability’.
- ECS Instances created in a single region will have private, low latency intranet network connectivity to other zones in the same region.
- However, ECS instances created in different regions, by default will not have private network connectivity.
- The network latency for instances within the same zone, however, is lower than when communicating across zones in the same region.

---


## ECS Instances

- ECS instance is a virtual machine that contains basic computing components such as the CPU, memory, operating system, network bandwidth, and disks.
- Once created, you can customize and modify the configuration of an ECS instance. For example: Add or remove additional Cloud Disks.

ECS instances are categorized into different families,
- based on the business needs to which those families can be applied,
- an instance family also has many instance types based on different CPU and memory configurations.
- An ECS instance defines two basic attributes: the CPU and memory configuration.

The instance types follow a naming convention which depicts the instance family, instance generation and instance size,
- for Example,
  - ecs.g5.large.
  - ecs is a prefix (All ECS instances have this in the name),
  - ‘g’ denotes instance family (in this case general purpose),
  - 5 denotes the instance generation and implies the CPU to RAM ratio, in this case, a ratio of 1 to 4 (this means that for each CPU there is 4 GB RAM), and large denotes the instance size.
- ecs.g5.large is the smallest instance in the general-purpose family and this instance has 2 CPUs, so with a ratio of CPU to RAM of 1 to 4 this instance has 2 CPUs and therefore 8GB of RAM.
- ecs.g5.xlarge is the next in the family tree so it has 4 CPUs and 16GB of RAM.
- ecs.g5.2xlarge is the next in the family tree so it has 8 CPUs and there 32GB of RAM.

3 main types of families:
- X86-Architecture,
- Heterogeneous Computing,
- and ECS Bare Metal Instances.


**X86-Architecture**
- 7 different subtypes as follows:
- `Entry Level (Shared Burstable)`:
  - You can accumulate CPU credits for your burstable instances, and consume those credits to increase the computing power of your workloads when required.
  - Used for Web application servers, Lightweight applications, and development and testing environments.
- `General Purpose`:
  - Used for Websites, application servers, Game servers, Small and medium-sized database systems.
- `Memory Optimised`:
  - Used for data analysis and mining, and other memory-intensive enterprise applications.
- `Big Data`:
  - Used for Enterprises that need to compute, store, and analyze large volumes of data.
- `Local SSD`: Used for Online transaction processing (OLTP) and high-performance databases.
- `High Clock Speed`: Used for on-screen video and telecom data forwarding, High-performance scientific and engineering apps.

**Heterogeneous Computing**
- 2 main subtypes as follows:
- `GPU-based compute-optimized`:
  - Used for Rendering and multimedia encoding and decoding, Machine learning, high-performance computing, and high-performance databases, Other server-high end workloads that require powerful concurrent floating-point compute capabilities.
- `Field-programmable-Gate-Array-based compute-optimized`:
  - Used for Deep learning and reasoning, Genomics research, Financial analysis, Image transcoding, Computational workloads such as real-time video processing and security management.

**ECS Bare Metal Instances**:
- This is a compute service that combines the elasticity of virtual machines and the performance and features of physical machines.
- The virtualization used by ECS Bare Metal Instances is optimized to support common ECS instances and nested virtualization.
- ECS Bare Metal Instances use `virtualization 2.0` to provide business applications with `direct access to the processor and memory resources of the underlying servers without virtualization overheads`.
- These are ideal for applications that need to run in a non-virtualized environment.


---


### ECS Images

Images are a template of a runtime environment for an ECS Instance.

There are 4 main types.

- `Public System Images`:
  - Public images licensed by Alibaba Cloud are highly secure and stable. These public images include most Windows Server and mainstream Linux systems.
  - These images only include standard system environments and you can apply your own customization and configurations based on these images.

- `Marketplace Images`:
  - Alibaba Cloud Marketplace images are classified into the following 2 types.
    - Images provided by Alibaba Cloud
    - and Images provided by Independent Software Vendors and licensed by Alibaba Cloud Marketplace
  - An Alibaba Cloud Marketplace image contains an operating system and pre-installed software.
  - The operating system and pre-installed software are tested and verified by the ISV and Alibaba Cloud to ensure that the images are safe to use.
  - These are suitable for website building, application development, and other personalized use scenarios.

- `Custom Image`:
  - Custom images are created from Instances or system snapshots, or imported from your local device.
  - Only the creator of a custom image can use, share, copy, and delete the image.
  - These custom images can be used to create more instances, saving you the effort of creating a new system from scratch.

- `Shared Image`:
  - A shared image is a custom image that has been shared to other users or accounts.
  - Alibaba Cloud cannot guarantee the security and integrity of the images shared with you. You use them at your own risk and discretion.


---


### ECS Storage

Block Storage
- a high-performance, low latency block storage service. It supports random or sequential read and write operations. Block Storage is similar to a physical disk, you can format a Block Storage device and create a file system on it to meet the data storage needs of your business.

ECS Storage provides architecture-based Cloud disks for your operating system disks and data disks.

Cloud Disks are based on the `Apsara` distributed file system called `“Pangu”`.
- Three redundant copies are stored on different physical servers under different switches in the datacentre.
- This provides high data reliability in the case of a failure.



3 types of Cloud Disk:

- `Ultra Disk`:
  - Cloud disks with high cost-effectiveness, medium random IOPS performance, and high data reliability.
- `Standard SSD`:
  - High-performance disks that feature consistent and high random IOPS performance and high data reliability.
- `Enhanced SSD`:
  - ultra-high performance disks based on the next-generation distributed block storage architecture.
  - Each ESSD can deliver up to 1 million of random IOPS and has low latency.


The 2 functions of Cloud Disks are as follows:

System disk
- As a System disk, by default the system disk has the same life cycle as the ECS instance to which it is mounted, and is released along with the ECS instance. (This auto release function can be changed.)
- Shared access to system disks is not allowed.
- System disk sizes can be 20GB and 500GB.
  - dependent on the operating system being provisioned.
    - Linux and FreeBSD systems default to 20GB.
    - CoreOS systems default to 30GB.
    - Windows systems default to 40GB.


Data disks:
- Data Disks can be created separately or at the same time as an ECS instance.
- Data disks created at the same time as an ECS instances have the same life cycle as the corresponding instance, and are released along with the instance by default.
- And again, this auto-release function can be changed.
- Data disks created separately can be released separately or at the same time as the corresponding ECS instance.
- And like the System disk shared access to a data disk is not allowed.
- sizes can be between 20GB and 32TB
- up to 16 Data Disks can be attached to a single ECS Instance.


Cloud Disks can be mounted to any instance in the same zone, but cannot be mounted to instances across zones.




### ECS Snapshots.

- Snapshots are complete, read-only copies of disk data at certain points in time.


You can use snapshots for the following scenarios:

- Disaster recovery and backup: You can create a snapshot for a disk, and then use the snapshot to create another disk to implement zone- or geo-disaster recovery.

- Environment clone: You can use a system disk snapshot to create a custom image, and then use the custom image to create an ECS instance to clone the environment.

- Data development:
  - Snapshots can provide near-real-time production data for applications
  - such as data mining, report queries, and development and tests.

- Enhanced fault tolerance:
  - roll a disk back to a previous point in time by using a snapshot to reduce the risk of data loss caused by an unexpected occurrence.
  - create snapshots on a regular basis to prevent losses caused by unexpected occurrences.
  - These unexpected occurrences can include, for example,
    - writing incorrect data to disks,
    - accidentally deleting data from a disk,
    - accidentally releasing ECS instances,
    - data errors caused by application errors,
    - and data loss due to hacking attempts.

- before you perform high-risk operations:
  - such as changing operating systems, upgrading applications, and migrating business data.


Snapshots can be created manually or automatically by creating a snapshot policy.
- Up to 64 snapshots can be created per disk and each snapshot is an **incremental copy** of the previous snapshot.
- When the maximum number of snapshots has been reached, the oldest snapshot is deleted as a new one is created.
- Snapshots are charged based on the storage space used and the amount of time they are kept.

---


## Security Groups.


Security groups
- act as virtual firewalls that provide **Stateful Packet Inspection** and **packet filtering** of `network protocol, port and source IP traffic` to allow or deny access.
- You can configure security group rules to control the inbound and outbound traffic of ECS instances in the group.

There are 2 classifications of security groups:
- Basic and Advanced.


Basic security groups support up to 2000 private IP Addresses, inbound and outbound rules can be configured to allow or deny ECS instances in basic security groups access to the Internet or intranet.

Advanced security groups
- new type of security group.
- an advanced security group can contain an unlimited number of private IP addresses.
- can only configure allow rules for inbound and outbound traffic,
- all non-allowed traffic is denied by default.

Security groups have the following characteristics:
- must specify a security group when you create an ECS instance.
- Each ECS instance must belong to at least one security group but can be added to multiple Security Groups at the same time.
- ECS Instances cannot belong to both basic and advanced security groups at the same time
- ECS instances in the same security group can communicate with each other through the internal network.
- ECS instances in different security groups are isolated from each other.
- You can add security group rules to authorize mutual access between two security groups.
- You can configure security group rules only for basic security groups, to authorize mutual access between two security groups.


Default Security Group:
- When you create an ECS instance in a region through the ECS console, a default security group is created if no other security group has been created under the current account in this region.
- The default security group is a basic security group and has the same network type as the ECS instance.


---


## ECS Networking.

Virtual Private Cloud (VPC)
- a logically isolated Virtual Network. It provides VLAN-level isolation and blocks outer network communications, it is a requirement when provisioning an ECS Instance.

VPC offers two major features,
- customize their own network topology, Assign Private IP address ranges, allocate network segments, and Configure VSwitches.

Customers can Integrate existing Datacentres through a `dedicated line (Express Connect)` or a `VPN Gateway` to form a hybrid cloud.

A Virtual Private Network is made up of two main components:
- A Virtual Router (VRouter)
- and one or more Virtual Switches (VSwitch).

A VSwitch
- a basic network device of a VPC network and is used to connect different ECS instances together in a subnet.
- A VPC can have a maximum of 24 VSwitches.

A VRouter
- a hub that connects all of the VSwitches in the VPC and serves as a gateway device that can connect to other networks.


![Screen Shot 2021-09-17 at 4.09.04 PM](https://i.imgur.com/dEsfWmP.png)

- VM1, VM2, and VM 3 can all communicate with each other, irrespective of the fact that they’re in different zones; they are in the same virtual private cloud network.



### IP address

- Each VPC-Connected ECS instance is assigned a private IP address when it is created.
- That address is determined by the VPC and the CIDR block of the vSwitch to which the instance is connected.

A Private IP Address can be used in the following scenarios
- Load balancing
- Communication among ECS instances within an intranet
- Communication between an ECS instance and other cloud products (such as OSS and RDS) or within an intranet.


public IP address
- ECS instances support two public IP address types.
- `NATPublicIP`,
  - which is assigned to a VPC-Connected ECS instance.
  - This type of address can be released only, and cannot be disassociated from the instance.
- `Elastic IP Address (EIP)`.
  - an independent public IP address that you can purchase and use.
  - EIPs can be associated to different ECS instances that reside within VPCs over time to allow public access to the ECS instances.

Their use cases are:
- do not want to retain the public IP address when the instance is released, use a NatPublicIP address
- want to keep a public IP address and associate it to any of your VPC-Connected ECS instances in the same region, use the EIP address
