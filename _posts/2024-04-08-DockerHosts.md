---
layout: post
title:  "Homelab Docker Hosts"
description: >-
  This first dive into my homelab will uncover how I build and run my docker containers. I'll go over everything from setting up the VM to the deploying of a container.
published: true
categories: tech homelab website proxmox linux debian
date: 2024-04-08 20:55:00 +0800
tags: blog tech homelab proxmox docker debian vm howto linux debian12 dockerhost
pin: false
---

{:toc}

## Intro

After some deep thinking (15 minutes), I needed to figure out which base Operating system to use for everything and I eventually landed on using [Debian 12](https://www.debian.org/releases/bookworm/). Why might you ask? Well, I was already using it as a desktop in an attempt to start using linux more. I even did so little research that I was learning on the fly everything that was introduced in version 12.

I also install [Docker](https://docs.docker.com/reference/) through their own documentation since it has never caused me issues since and its easier to follow their instructions every time a new release comes out.

My current setup as of this post has 2 proxmox servers and those split the Docker and Kubernetes aspects in half for my lab. I tried [Docker Swarm](https://docs.docker.com/engine/swarm/) at some point but never liked it or found it practical in the real world. I also wanted to keep my Docker ecosystem simple by design.

## Proxmox

All my docker hosts are installed on a single proxmox host and i'll tend to deploy roughly 3 equal sized VMs to fill all my needs. Yes, I also said VMs. I prefer to use techneques that are more traditional and where the work in this homelab can overlab into more practical use. I will explore [proxmox's CT](https://pve.proxmox.com/wiki/Linux_Container) features at some point in the future.

### Creating Proxmox VM for Docker Hosting

In your Proxmox web interface and under your host, `Right-click -> Create new VM` and start configuring your vm's properties. Each of my Docker hosts are configured with `Start at boot:` and `Start/Shutdown order:` configured. When Proxmox goes down and comes back up, I want my docker hosts to start automatically and in a controlled order. Where critical services come up first followed by least critical. The tags are just to see the VMs purpose in larger menus easier.

![Proxmox Create VM Option](assets/img/20240408/proxmox-create-vm-general.png){: width="972" height="589" .w-75 .normal}

#### OS

As mentioned previously, I intend on using Debian 12.5 for these VMs. I have a directory on my NAS that I use to store ISO's. The Proxmox servers have an SMB/CIFs storage mapping to this directory for access to my ISO library.

![Proxmox Create VM OS](assets/img/20240408/proxmox-create-vm-os.png){: width="972" height="589" .w-75 .normal}

#### System

I deploy all my VMs as q35 with the Qemu agent to collect metrics.

![Proxmox Create VM System](assets/img/20240408/proxmox-create-vm-system.png){: width="972" height="589" .w-75 .normal}

#### Storage

When it comes to storage I chucked some Samsung Evo's into the Proxmox hosts. Each host has a minimum of 2TB of storage and at least 1-2TB of NVME storage. Needless to say, I use the solid states because they're available and the proof is in the pudding. I'll make a post at some point going over the disk metrics of all the hosts I use. For now, I store all VM storage on NVME that is available on the proxmox hosts. Each Host gets 250GB, which for the sake of this homelab and these VMs is plenty.

![Proxmox Create VM Storage](assets/img/20240408/proxmox-create-vm-storage.png){: width="972" height="589" .w-75 .normal}

#### CPU

CPUs are pretty simple. Most applications I run are not requiring multiple threads and in general I never have CPU demands based on my usage in the environment. I do give 2 CPUs in the odd case I can use them, and I just max out the cores for the free Proxmox license. Again, I dont get over-commit issues with CPUs doing this, like most people my bottleneck is memory and disk; in that order.

![Proxmox Create VM CPU](assets/img/20240408/proxmox-create-vm-cpu.png){: width="972" height="589" .w-75 .normal}


#### Memory

Each of the Proxmox hosts has 32GB of ram. I just divided the Ram by machine count (3) and got 10.5GB per vm. I removed a GB for over-commit and just general overhead and arrived at 9GB per vm. A Docker host dedicated to infra, another to game servers and 'stuff' and another true lab or test bed and this is absolutly fine. Also because of the attempt to reduce over-commit I can deploy one-off machines without to much impact either. I think I struck a nice balance for these hosts.

![Proxmox Create VM Memory](assets/img/20240408/proxmox-create-vm-memory.png){: width="972" height="589" .w-75 .normal}

#### Network

This might come as a suprise but I am not here to work on my networking or homelab this area of my home infrastructure. I tend to turn off all firewalls at any layer or other features you might consider production for security or any form of access control. Im a fan of the path of least resistence and if I want to go near networking I'll do it in isolation so dont be to suprised at these level of settings.

![Proxmox Create VM Network](assets/img/20240408/proxmox-create-vm-network.png){: width="972" height="589" .w-75 .normal}

After that we can tick `Start after created` if you want and `Finish`.


## Installing Debian

Once the VM has been created we need to configure the OS. Now, I will go ahead and admit there are a 100 solutions I could use to automate this and get this done in so many different ways. I know, im just not at the "Let me care enough" phase to Ansible this yet. So we get to manually configure Debian, so buckle your seatbeat.

I'll be using that glorious GUI.

![Debian Graphical Install](assets/img/20240408/debian-graphical-install.png){: width="972" height="589" .w-75 .normal}

Your language.

![Debian Graphical Language](assets/img/20240408/debian-graphical-language.png){: width="972" height="589" .w-75 .normal}

Set your location.

![Debian Graphical Location](assets/img/20240408/debian-graphical-location.png){: width="972" height="589" .w-75 .normal}

Set your locals. UTF-8 is pretty standard.

![Debian Graphical Locales](assets/img/20240408/debian-graphical-locales.png){: width="972" height="589" .w-75 .normal}

Configure your keyboard. I do apologize for not getting a screenshot for this. I hope you can managed (just click next).

Once everything is done extracting you can now name your VM.

![Debian Graphical Hostname](assets/img/20240408/debian-graphical-network00.png){: width="972" height="589" .w-75 .normal}

Set domain name if needed.

![Debian Graphical Domain Name](assets/img/20240408/debian-graphical-network01.png){: width="972" height="589" .w-75 .normal}

The next couple of sections didnt make to much sense for me to capture as they are rather straight forward. Next step should have you setting your root user password.

All my servers have a user created called `homelab`. After hitting next you should be presented with disk configuration.

I use Guided - entire disk as LVM. My logic here is using an LVM will make expanding the volume pretty straight forward if I were required to add any extra disks at any point. I just see LVM giving me a tad more flexibility. I dont worry to much about drive encryption since its not needed and can cause a delay in VM startup.

![Debian Graphical LVM](assets/img/20240408/debian-graphical-lvm.png){: width="972" height="589" .w-75 .normal}

If you're following this guide step by step. You should only have 1 disk anyway to select, but if you add extra disks, you can select which one(s) to use for the Operating system install.

I am not to worried about the partitioning as it can create headaches if it fills unevenly.

![Debian Graphical Partition Disks](assets/img/20240408/debian-graphical-partition-disks.png){: width="972" height="589" .w-75 .normal}

Confirm you want to write the changes to the disk and confirm the amount of disk space to use in creating the LVM.

Then, confirm again you want to write disk changes and create the LVM and go grab a coffee.

![Debian Graphical Installing Base System](assets/img/20240408/Installing-base-system.png){: width="972" height="589" .w-75 .normal}

I dont bother with scanning the DVD image for extra media.

![Debian Graphical Media Scan](assets/img/20240408/debian-graphical-media-scan.png){: width="972" height="589" .w-75 .normal}

I will however use a Network Mirror and point it to the closest mirror to me and continue with the OS install.

I do not participate in the package usage survey.

The software I select is pretty straight-forward. Also, apologies for the mouse location, I noticed after posting its horrible. I **DO NOT** install web server tools.

![Debian Graphical Software](assets/img/20240408/debian-graphical-software.png){: width="972" height="589" .w-75 .normal}

Lets install grub on the primary drive.

![Debian Graphical Boot Loader](assets/img/20240408/debian-graphical-boot-loader.png){: width="972" height="589" .w-75 .normal}

![Debian Graphical Boot loader Disk](assets/img/20240408/debian-graphical-boot-loader-disk.png){: width="972" height="589" .w-75 .normal}

If you need another coffee. Now is your chance.

![Debian Graphical Finishing](assets/img/20240408/debian-graphical-finishing.png){: width="972" height="589" .w-75 .normal}

When everything finishes and before clicking `continue` I like to remove the ISO.

![Remove  Media](assets/img/20240408/proxmox-remove-media.png){: width="972" height="589" .w-75 .normal}

[Up Next. Configuring Debian hosts for Docker.](2024-04-08-ConfigureDockerHost.md)