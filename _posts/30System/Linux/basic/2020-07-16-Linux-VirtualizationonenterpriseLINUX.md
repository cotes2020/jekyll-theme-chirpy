---
title: Linux - Virtualization on enterprise LINUX
date: 2020-07-16 11:11:11 -0400
categories: [30System, Basic]
tags: [Linux]
math: true
image:
---

# Deploy systems

[toc]

## Virtualization on enterprise LINUX

KVM provides
- Overcommit of physical resources. (CPU, RAM, Hard drive space)
- agent on guest to communicate with hypervisor. (host can send commands to it directly)
- Disk I/O throttling. (limit the disk resources a VM gets)
- Virtual CPU hot add. (add virtual VPU when the VM is running)
- Nested virtualization. (run hypervisor in VM so can run more VMs on it)

To speed up virtualization KVM supports:
- hardware passthrough
- paravirtualized drivers
- PCI function passthrough


the KVM hypervisor's managed with a `libvert API` and tools like `virt-manager GUI tool` and `virsh command tool`

---

## prepare the host for Virtualization

### install

ref:

https://help.ubuntu.com/community/KVM/Installation
https://bugzilla.redhat.com/show_bug.cgi?id=1753146

#### Pre-installation checklist
Check that your CPU supports hardware virtualization

To run KVM, need a processor supports hardware virtualization.
- Intel and AMD both have developed extensions for their processors, deemed respectively Intel VT-x (code name Vanderpool) and AMD-V (code name Pacifica).

- check processor supports:

```py
egrep -c '(vmx|svm)' /proc/cpuinfo
- 0 CPU doesnt support hardware virtualization.
- 1 or more it does - but still need to make sure that virtualization is enabled in the BIOS.
```

By default, booted into XEN kernel it will not display svm or vmx flag using the grep command.

- To see if it is enabled or not from xen, enter:

`cat /sys/hypervisor/properties/capabilities`

You must see hvm flags in the output.

```py
kvm-ok
# provide an output like this:
INFO: /dev/kvm exists
KVM acceleration can be used

# If you see :
INFO: Your CPU does not support KVM extensions
KVM acceleration can NOT be used
# can still run virtual machines, but it'll be much slower without the KVM extensions.

# NOTE: You may see a message like "KVM acceleration can/can NOT be used". This is misleading and only means if KVM is *currently* available (i.e. "turned on"), *not* if it is supported.
```

Use a 64 bit kernel (if possible)

Running a 64 bit kernel on the host operating system is recommended but not required.
- To serve more than 2GB of RAM for your VMs, you must use a 64-bit kernel (see 32bit_and_64bit). On a 32-bit kernel install, you'll be limited to 2GB RAM at maximum for a given VM.
- Also, a 64-bit system can host both 32-bit and 64-bit guests. A 32-bit system can only host 32-bit guests.

check if processor is 64-bit:

```py
egrep -c ' lm ' /proc/cpuinfo
- 0  CPU is not 64-bit.

- 1 or higher, it is.
Note: lm stands for Long Mode which equates to a 64-bit CPU.
```

check if running kernel is 64-bit

```py
uname -m
#
- x86_64 a running 64-bit kernel.
- i386, i486, i586 or i686 a 32-bit kernel.

Note: x86_64 is synonymous with amd64.
```

#### Installation of KVM

Install Necessary Packages
For the following setup, we will assume that you are deploying KVM on a server, and therefore do not have any X server on the machine.

to install a few packages first:

```shell
# install
$ sudo yum install qemu-kvm libvirt virt-manager libvirt-client


# install by group
$ sudo yum group install "Virtualization Client"

# star libvert service
$ sudo systemctl start libvirtd
# make it persistent
$ sudo systemctl enable libvirtd
$ reboot
```

---

### install CentOS into a guest VM interactively.

use virt-manager

1. application --> system tools --> `Virtual machine manager`
2. login
3. create a new virtual Machine.

warning:
- *VTs not enables*: check the BIOS, make sure it turns on.
- *KVM kernel module didn't get loaded*: make sure install all package, enable `libvirtd` and reboot it.

4. Download the CentOS ISO: `DVD ISO`, save it.
5. file --> new vm --> local install media --> forward
6. Use ISO image: Browse the `DVD ISO` --> forward
7. Memory: 2048. CPUs: 2. --> forward
8. Storage: 9G --> forward
9. set the name of VMs.
10. Network selection: Host device enp***(wired ethernet Devices) wlp*** (wireless Devices) --> finish
11. begin installation of CentOS
  - device selection: Partitioning: automatically config partitioning.
  - root pwd, user creation
  - reboot
  - license.
  - shut down / power off
12. the new VM shows in virt-manager

--

## Kickstart files

 when install Centos, a kickstart file is autpmatically saved in `/root`

to repeat the exact installation:

```shell
$ sudo ls /root
anaconda-ks.cfg  initial-setup-ks.cfg

$ lsee -N initial-setup-ks.cfg   # -N for lighten numbers
```

![Screen Shot 2020-04-05 at 14.19.14](https://i.imgur.com/qyyBSiy.png)

### install with Kickstart

prerequisties:
1. ISO image or network url share of linux installation files.
2. a VM disk image, virtual box...
3. a kickstart files
4. kickstart file delivery

![Screen Shot 2020-04-05 at 14.32.21](https://i.imgur.com/MmGNIyK.png)

5. virtual machine manager check the new VMs.
6. view --> Text consoles: has no graph interface available.
7. light bulb bottom --> controller USB -->
8. below, `Add hardware` --> Add new virtual hardware --> model: `QXL`
9. below, `Add hardware` --> Graphics --> Type: `Spike server`
10. run again
11. view --> Text consoles: now has `Graphical console spice`
12. setup the license.....
13. restart. finish.

## configure VMs to communicate

1. light bulb bottom --> NIC --> network source: enp**

do for each VMs, they will be able to ping each other.

## `virsh`


![Screen Shot 2020-04-05 at 15.00.28](https://i.imgur.com/qn4tcD9.png)

























.
