---
title: Virtualization
date: 2020-11-26 11:11:11 -0400
categories: [00Basic, VMandContainer, VMs]
tags: [Linux, VMs]
math: true
image:
---

- [Virtualization](#virtualization)
  - [traditional way](#traditional-way)
  - [Virtualization](#virtualization-1)
  - [Virtualization component](#virtualization-component)
  - [Comparing Hypervisors](#comparing-hypervisors)
  - [Type of Virtualization](#type-of-virtualization)
    - [full virtualization scenario](#full-virtualization-scenario)
    - [paravirtualization](#paravirtualization)
  - [QEMU](#qemu)
  - [KVM `Kernel-based Virtual Machine`](#kvm-kernel-based-virtual-machine)
  - [QEMU and KVM](#qemu-and-kvm)
  - [QEMU example](#qemu-example)
  - [Libvirt example](#libvirt-example)
  - [virsh example](#virsh-example)
  - [virsh example](#virsh-example-1)
  - [change vm configure](#change-vm-configure)
  - [snapshot and clone](#snapshot-and-clone)
  - [autostart](#autostart)
  - [vm migration](#vm-migration)
  - [container](#container)

---

# Virtualization

---

## traditional way


![Screen Shot 2021-02-11 at 17.21.07](https://i.imgur.com/iUnkAWY.png)

- the default way to deploy an application was on its own physical computer.
  - find some physical space, power, cooling, network connectivity for it
  - and then install an operating system, any software dependencies, and then finally the application itself
- If need more processing power, redundancy, security, or scalability
  - simply add more computers.
  - It was very common for each computer to have a single-purpose.
    - Example: a database, web server, or content delivery.
- Applications were built for a specific operating system even for specific hardware

---

## Virtualization

> Virtualization helped by making it possible to run multiple virtual servers and operating systems on the same physical computer.


![Screen Shot 2021-02-11 at 17.47.31](https://i.imgur.com/26faY8t.png)


virtualization
- takes less time to deploy new solutions
- waste less of the resources on those physical computers
- get some improved portability because virtual machines can be imaged and then moved around.

However
1. the application, all of its dependencies and operating system are still bundled together and it's not very easy to move from a VM from one hypervisor product to another.
2. Every time you start up a VM, it's operating system still takes time to boot up.
3. Running multiple applications within a single VM creates tricky problem,
   1. applications that share dependencies are not isolated from each other
   2. the resource requirements from one application, can starve out other applications of the resources that they need.
   3. a dependency upgrade for one application might cause another to simply stop working.

solve this problem with:
1. rigorous software engineering policies.
   - lock down the dependencies that no application is allowed to make changes,
   - but this leads to new problems because dependencies do need to be upgraded occasionally.
2. add integration tests to ensure that applications work.
   - Integration tests are great, but dependency problems can cause new failure modes that are harder to troubleshoot, and it really slows down development if you have to rely on integration tests to simply just perform basic integrity checks of your application environment.

![Screen Shot 2021-02-11 at 17.49.44](https://i.imgur.com/KVMQ1Yk.png)

3. Now, the VM-centric way to solve this problem is to run a dedicated virtual machine for each application.
   - Each application maintains its own dependencies, and the kernel is isolated.
   - So one application won't affect the performance of another.
   - One you can get as you can see here, is two complete copies of the kernel that are running.
   - issues
     - Scale this approach to hundreds of thousands of applications, and you can quickly see the limitation.
     - trying to do a simple kernel update. So for large systems, dedicated VMs are redundant and wasteful.
     - VMs are also relatively slow to start up because the entire operating system has to boot.


> containers
>
> A more efficient way to resolve the dependency problem
>
> Implement abstraction at the level of the application and its dependencies.
>
> Don't have to virtualize the entire machine or even the entire operating system, but just the user space.
>
> the user space is all the code that resides above the kernel, and includes the applications and their dependencies.

![Screen Shot 2021-02-11 at 17.49.28](https://i.imgur.com/Edp2Re5.png)

---

## Virtualization component


- **Hypervisor**:
  - The software that creates, runs, and manages the VMs.
  - the software layer that breaks the dependencies of an operating system with its underlying hardware, and allow several virtual machines to share that same hardware
  - Several virtualization technologies:
    - VMware,
    - Microsoft Hyper-V products,
    - and Oracle VM VirtualBox.
    - KVM
  - These applications have their own hypervisor software.

- **Host**:
  - The physical system hosting the VMs.
  - It requires more resources than a typical system, such as multiple processors, massive amounts of RAM, fast and abundant hard drive space, and one or more fast network cards.
  - additional resources increase the cost of the host
  - still less expensive than paying for multiple physical systems, less electricity, less cooling, less physical space.

- **Guest**:
  - Operating systems running on the host system are guests or guest machines.
  - Most hypervisors support several different operating systems, including various Microsoft operating systems and various Linux distributions.
  - Additionally, most hypervisors support both 32- bit and 64-bit operating systems.


Host **Elasticity and Scalability**.
- `the ability to resize computing capacity based on the load`.
  - Example
  - VM has increased traffic. You can increase the amount of processing power and memory used by this server relatively easily.
  - it’s relatively easy to decrease the resources when the load decreases.
- Virtualization typically provides the best return on investment (ROI) when an organization has many underutilized servers.
  - Example
  - organization has nine servers with each using only about 20 percent processing power, memory, and disk space.
  - You could convert three physical servers to virtual hosts and run three guest servers on each physical server.
  - Assuming all the servers are similar, this wouldn’t cost any more money for the physical servers.
  - Additionally, three physical servers consume less electricity and require less heating and ventilation to maintain.



In contrast, imagine the organization has nine servers with each using about 80 percent of their processing power, memory, and disk space.
- Although it is possible to convert them all to virtual servers, it requires the purchase of additional hardware.
- The savings from less electricity and less heating and ventilation is offset by the cost of the new servers.



---

## Comparing Hypervisors
Hypervisor virtualization is divided into primarily two different types:

- **Type I hypervisors**
  - run directly on the system hardware.
  - often called `bare-metal hypervisors`, because they don’t need to run within an operating system.
  - Example
  - VMware has `ESX/ESXi` products that are Type I hypervisors.


- **Type II hypervisors**
  - run as software within a host operating system.
  - Example
  - Microsoft `Hyper-V hypervisor` runs within a Microsoft operating system.
  - each guest has a full operating system, including its own kernel.
  - kernel is just the central part or most important part of something.
  - When referring to a computer, the kernel is the central part of the operating system.

![Image21](https://i.imgur.com/aRoAODK.png)


> - When implementing virtualization on a PC, use Type II hypervisor-based virtualization.
> - But virtualization in large-scale data centers, uses Type I virtualization.


---

## Type of Virtualization

- Full virtualization
- Paravirtualization

### full virtualization scenario
- `HVM, hardware virtual machine`
- the hypervisor `creates emulated virtual devices`. 
  - Including a virtual motherboard, virtual processor, virtual RAM, and virtual versions of all of the hardware, a system needs to operate. 
  - For example
  - a virtual network adaptor. 
  - So, the virtual system, gets a device, that looks like a real network adaptor. 
  - But in reality, is simulated, or more precisely, emulated by the hypervisor. 
  - The guest operating sends and receives data through this virtual adaptor. 
  - And then, the emulator translates those requests, to the real hardware. 
- Emulation software can provide almost any kind of device to a virtual machine. 
  - Even memory and a processor. 
  - Though, emulating the activity of processors, can be pretty slow. 
  - To speed up virtual machine,
  - the processor manufacturers, add a set of instructors to their chips, that allow a hypervisor to pass processor instructions directly from a VM to a real processor. 
  - Rather than emulating the processor, while keeping those instructors and their results, separate from other instructors from the host machine and from other guest operating systems. 
- can install any operating system, `OS doesn't need to be modified` to run correctly. 


On Linux systems:
- to emulate hardware, use QEMU, Quick emulator. 
- the software that lets instructors pass through, to supported processor, is KVM, Kernel-based virtual machine. 
- Together, these packages handle the details of presenting a virtual environment for an operating system to use.


**QEMU, Quick emulator**
- `QEMU` can be used by itself to emulate a whole system. 
- if `KVM` is available, and the processor of the host machine supports these commands for hardware virtualization, 
  - QMEU will use KVM for direct access, instead of emulating a processor and memory. 
  - KVM used to be a separate package, but now it's part of QEMU. 



### paravirtualization
- in a paravirtualization machine, 
- the guest os must be modified to know, that it's running as a virtual machine. 
  - These modifications were added to the Linux kernel a while ago. 
    - So, Linux can run either, under a hardware virtualized system, or a paravirtualized system. 
  - But many other os, like Windows, can not. 
  - These modifications `allow a guest to communicate directly to some hardware resources on the host machine`. 
    - Like, storage and network hardware. 
  - This gives a performance boost, 
  - as the hypervisor doesn't have to take input from an emulated adapter and then translate it physical hardware. 

- But, in a paravirtualized machine, the **memory** and **processor**, are still emulated. 
  - `QEMU and KVM`, provide a hardware machine environment, for guest operating systems, 
  - with `some paravirtualized hardware`, to improve speed. 

- to use full paravirtualization for a guest, need to use the `xen hypervisor`. 
  - The Linux kernel has support both hypervisors, and both are widely used in industry. 
  - hypervisors such as `Xen`, `Proxmox` and `ESX`, 
    - run as the bare metal, or the native operating system on a host. 
  - `KVM`
    - runs inside of a Linux installation, as a module within the kernel, 
    - leaving an operating system that can be used for other things.
  - In addition to acting as a hypervisor for guest operating systems. 

- to manage virtual machines
  - can use `QEMU` directly, 
  - or use software that interfaces with a hypervisor, to make creation and manage a more visual experience.
    - Such as `libvirt` package
    - software like `Virtual Machine Manager`. 
      - graphical tool, also part of the libvert ecosystem. 

- Virtual machines give the ability, to scale the processing power, memory, storage, and other aspects of a guest, in order to respond to changes in business needs, or to provision identical nodes, in order to scale up, or out, as your app demands. 

- Many of the features of cloud services are based on this flexibility, though many of the cloud providers have their own tools and systems, to manage, monitor and track resource usage efficiently. 

when we create virtual machines, few options:
- Do we use a dedicated hypervisor, like Xen or Proxmox, or ESX? 
- Do we use containers, either natively, or with software like Docker? 
- Do we take advantage of KVM on a Linux installation? 
- It depends on what you need to do



- when we create virtual machines on a Linux system, we have a few ways to do so. We can create a VM directly with command line, or we can use some management tools that help make the process easier. We'll take a look at those, throughout the rest of the course.



---

## QEMU
- type 2 (i.e runs upon a host OS) hypervisor
- performing `hardware virtualization` such as disk, network, VGA, PCI, USB, serial/parallel ports, etc.
- It is flexible in that it can emulate CPUs via dynamic binary translation (DBT) allowing code written for a given processor to be executed on another (i.e ARM on x86, or PPC on ARM).
- Though QEMU can run on its own and emulate all of the virtual machine’s resources, as all the emulation is performed in software it is extremely slow.
- QEMU（quick emulator)本身并不包含或依赖KVM模块，而是一套由Fabrice Bellard编写的模拟计算机的自由软件。
- QEMU虚拟机是一个纯软件的实现，可以在没有KVM模块的情况下独立运行，但是性能比较低。
- QEMU有整套的虚拟机实现，包括处理器虚拟化、内存虚拟化以及I/O设备的虚拟化。
- QEMU是一个用户空间的进程，需要通过特定的接口才能调用到KVM模块提供的功能。
- 从QEMU角度来看，虚拟机运行期间，QEMU通过KVM模块提供的系统调用接口进行内核设置，由KVM模块负责将虚拟机置于处理器的特殊模式运行。
- QEMU使用了KVM模块的虚拟化功能，为自己的虚拟机提供硬件虚拟化加速以提高虚拟机的性能。

## KVM `Kernel-based Virtual Machine`
- a Linux kernel module.
- 内核驱动模块, 能够让Linux主机成为一个Hypervisor（虚拟机监控器）。
- type 1 hypervisor, a `full virtualization solution for Linux on x86 hardware` containing virtualization extensions (Intel VT or AMD-V)
- full virtualization
  - When a CPU is emulated (vCPU) by the hypervisor,
  - the hypervisor has to translate the instructions meant for the vCPU to the physical CPU.
  - massive performance impact.
  - To overcome this,
  - `modern processors` support **virtualization extensions**, such as `Intel VT-x and AMD-V`.
  - provide the ability for a slice of the physical CPU to be directly mapped to the vCPU.
  - so the instructions for the vCPU can be directly executed on the physical CPU slice.
- 在支持VMX（Virtual Machine Extension）功能的x86处理器中，Linux在原有的用户模式和内核模式中新增加了客户模式，并且客户模式也拥有自己的内核模式和用户模式，虚拟机就是运行在客户模式中。KVM模块的职责就是打开并初始化VMX功能，提供相应的接口以支持虚拟机的运行。


- KVM只模拟CPU和内存，因此一个客户机操作系统可以在宿主机上跑起来，但是你看不到它，无法和它沟通。
  - KVM只是内核模块，用户并没法直接跟内核模块交互，需要借助用户空间的管理工具QEMU。
  - 于是，有人修改了QEMU代码，把他模拟CPU、内存的代码换成KVM，而网卡、显示器等留着，因此QEMU+KVM就成了一个完整的虚拟化平台。

![image1](https://i.imgur.com/DQxrAq8.png)

---


## QEMU and KVM

- QEMU can run independently, but due to the emulation being performed entirely in software it is extremely slow.
- To overcome this, QEMU allows to use KVM as an accelerator so that the `physical CPU virtualization extensions` can be used.
- KVM和QEMU相辅相成，
  - QEMU是个计算机模拟器，而KVM为计算机的模拟提供加速功能。
  - QEMU通过KVM达到了硬件虚拟化的速度，
  - 而KVM则通过QEMU来模拟设备。
  - 对于KVM来说，其匹配的用户空间工具并不仅仅只有QEMU，还有其他的，比如RedHat开发的l`ibvirt、virsh、virt-manager`等，QEMU并不是KVM的唯一选择。

**conclude**:
- QEMU is a type 2 hypervisor that runs within `user space` and performs `virtual hardware emulation`
- KVM is a type 1 hypervisor that runs in `kernel space`, that allows a user space program access to the `hardware virtualization` features of various processors.


---


## QEMU example

```bash
# install
$ apt installl qemu qemu-kvm
# /usr/bin/qeme...

$ man qemu-system


# create a disc for vm
# generate a disc image file
# Raw: 60G even not use, thick provisiones
# QEMU copy on write (QCOW2): write and use
$ qemu-img create -f qcow2 my-image.qcow2 60G
$ qemu-img create -f raw my-image.qcow2 60G


# install os, nedd iso
$ qemu-system-x86_64 -cdom path/toISO my-image.qcow2 -m 2G -enable-kvm

# start
$ qemu-system-x86_64 my-image.qcow2 -m 2G -enable-kvm
$ qemu-system-x86_64 my-image.qcow2 -m 4G -net none -vga qxl -enable-kvm


# tools
apt install hardinfo
# processor: QEMU processor

# check memory
free -h

# check disc
df -h
```


![Screen Shot 2020-11-28 at 22.10.50](https://i.imgur.com/elhp5Kt.png)


---


## Libvirt example

```bash
# install
$ apt install virt-manager

# create and install vm
$ virt-install
$ virt-install --name my-ubunry
               --memory 2014
               --disk size=60, format=QCOW2,
                               path=/var/lib/libvirt/images/my-ubuntu.qcow2 # default
               --cdrom ~/Downloads/ubuntu.iso
$ virt-install --name my-ubunry
               --memory 2014
               --disk size=60, format=QCOW2,
                               path=/var/lib/libvirt/images/my-ubuntu.qcow2 # default
               --cdrom ~/Downloads/ubuntu.iso
```

![Screen Shot 2020-11-28 at 22.27.13](https://i.imgur.com/VloX38m.png)


![Screen Shot 2020-11-25 at 10.53.38](https://i.imgur.com/DZAbrU7.png)





---


## virsh example

```bash

$ virsh --help


$ virsh
virsh $ suspend my-ubuntu
virsh $ resume my-ubuntu
virsh $ shutdown my-ubuntu
virsh $ list -all
virsh $ start my-ubuntu
virsh $ quit

$ virt-viewer my-ubuntu

$ virsh domdisplay my-ubuntu
spice 4.4.4.4:8080

```
 

---


## virsh example

```bash
$ virt-manager

# graphy install of vm
```

---

## change vm configure

```bash

# change vm memory
$ virsh dominfo my-ubuntu
# temperaly
$ virsh setmem my-ubuntu 2048M
# change configure
$ virsh setmem my-ubuntu 1G -config
$ virsh setmaxmem my-ubuntu 4G
$ virsh start my-ubuntu


# change vm storage
$ cd /dev
$ sudo fdish /dev/sda
p # check info
n # create new partition
    p
    1
    w

$ sudo mkfs.ext4 /dev/sda1

$ sudo  mount /dev/sda1 /path/to/mount


#  attach disk
$ sudo qemu-img create -f qcow2 /var/lib/libvirt/images/disk2.qcow2 40G
$ virsh attach-disk my-ubuntu --source /var/lib/libvirt/images/disk2.qcow2 --drive qemu --subdriver qcow2 --target vdb --persistent


# check all disk
$ sudo fdisk -l


#  deattach
$ virsh detach-disk my-ubuntu --source /var/lib/libvirt/images/disk2.qcow2

```


---

## snapshot and clone


## autostart

```bash
$ virsh autostart my-ubuntu
$ virsh autostart my-ubuntu --disable

```


---

## vm migration


![Screen Shot 2020-11-28 at 22.39.31](https://i.imgur.com/ZfKoiM7.png)

---

## container

```bash
$ apt install lxc1
$ ls /usr/share/lxc/templates/

# install container
$ sudo lxc-create -n container1 -t unbuntu

# start container
$ sudo lxc-start -n container1
$ sudo lxc-concole -n container1
$ df -h
$ top
$ sudo lxc-stop -n container1
$ sudo lxc-destroy -n container1
controla+q


$ sudo lxc-attach -n container1
$ sudo lxc-info -n container1

$ sudo lxc-ls
```

















.
