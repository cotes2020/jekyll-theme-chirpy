---
layout: post
title:  "Homelab Configure Docker Hosts"
description: >-
  Our first post showed you how I roll out the virtual machines. Lets configure those machines, and get Docker up and running!
published: true
categories: tech homelab website
date: 2024-04-09 09:00:00 +0800
tags: blog tech homelab
pin: false
---


## Intro

[In the previous article](/posts/DockerHosts) we configured our virtual machines. In this article I hope to go over configuring the operating system. I also realised the last article might have gotten a bit long just decided to end it at the VM configuration.

That should be fine because we can pick up from where we left off and start to configure the operating system. Now, as a quick recap we pretty much "next, next, next" our way through the initial Operating System install. Now we get to configure the operating system. At the momement, I enjoy hand-jamming these configurations and settings since I spent the good part of 15 years work on Windows. I like to reinforce these things to overcome old habits.

Also feel free to substitute vim for nano and not feel judged...

## First ~~contact~~ Login

Picking up where we left off. Our Debian install is so fresh its running on DHCP with no Sudo installed, either way, I need to make a couple of things clear from the start. My network is `192.168.178.0` and is only meant to provide context for when I configure network settings and attach to storage.

Now, if you recall from our Debian setup we installed the QEMU Agent? Well one of the reasons for this is that it helps us with getting this VMs IP address from the Proxmox web interface under `Summary`. If you decided to not install the agent it is okay. You can open a console and log into the host.

### Get IP Address

The quickest way I grab a host IP is `ip addr | grep 192.168.178.` which has the nice bonus of providing the NIC we can use later.

![Proxmox Create VM Storage](assets/img/20240408/debian-cli-ip-addr.png){: width="972" height="589" .w-75 .normal}


```shell
ip addr | grep 192.168.178.
  inet 192.168.178.64/24 brd 192.168.178.255 scope global dynamic enp6s18
```

If you notice the `enp6s18` value. This is the NIC the IP address is assigned to which is useful when we configure the static IP address of our host. Also, our IP address in this instance is `192.168.178.64`.

### SSH Into Host

From any terminal you should be able to ssh into this host because in the last article we clicked the option to install SSH.

![Proxmox Create VM Storage](assets/img/20240408/debian-cli-ssh.png){: width="972" height="589" .w-75 .normal}

```shell
ssh homelab@192.168.178.64
The authenticity of host '192.168.178.64 (192.168.178.64)' can't be established.
ED25519 key fingerprint is SHA256:BdwStOHrLnanQnkn5xlXOeZqjsS3J9sCDbNjrUXxsNQ.
This key is not known by any other names.
Are you sure you want to continue connecting (yes/no/[fingerprint])? yes
Warning: Permanently added '192.168.178.64' (ED25519) to the list of known hosts.
homelab@192.168.178.64's password:
```

### Update the Host
I prefer vim over vi. So lets get this out of the way. Now, I installed with the DVD image which will add the DVD as a source location for apt updates. I tend to just delete this entry, you're free to comment it out.

```shell
su -u # Remember we dont have sudo yet
vi /etc/apt/sources.list
```
Again, comment out or delete any entry that refers to `cdrom` or contains `DVD` in the name.

Update and upgrade as needed.

```shell
apt update && apt upgrade -y
```

### Up-Front Installs

I usually get a couple of the software requirements out of the way while still in root.

```shell
apt install sudo wget curl open-iscsi vim restic smbclient cifs-utils htop -y
```
Are usually good enough for working with files, backups and remote storage connections.

### Set Static IP Address

First thing I like to do before updating is to configure the static IP address, it also gives me a reason to reboot the host to ensure nothing is 'off' during a reboot.

```shell
vim /etc/network/interfaces
```
This will open the file that contains the interface (nic) information. If you remember from getting our local IP address we also got the nic assigned to the primary address. Now if you have 1 nic this is not so much of a problem but is a nice thing to help when working with 2 or more but feel free to substitute `enp6s18` for your NIC. Also, our initial DHCP host IP address was `192.168.178.64` and I will just continue to use this.

To assign a static IP address. I change `iface enp6s18 inet dhcp` to the following. Until we get Pihole running im not to worried about DNS settings right now.

```
iface enp6s18 inet static
  address 192.168.178.64
  network 192.168.178.0
  broadcast 255.255.255.0
  gateway 192.168.178.1
  dns-nameservers 1.1.1.1
```

### Setting up Docker

I will never be able to explain it better than the mainteners of the product. This guide, and the fact im linking it here is enough to get me up and running. Just follow the the 'Installing Using Apt Repository' section.

### Setting up Docker Compose

You followed the above? Good, run `apt install docker-compose`.

### Adding A user to Sudoers

You can add the `homelab` account to the sudoers file with.

```shell
sudo visudo
```
Also because it is a homelab, I just add below the `root` entry the following. Explaining why this is a pretty lazy and bad approach is outside the scope here but I do need to have some due diligence in warning you this is not best practice.

```txt
homelab ALL=(ALL:ALL) ALL
```

It is at this point I just run a reboot.

### FSTAB

My Docker hosts each get a CIFS mount provided by my NAS. This CIFs mount acts as a [restic](https://restic.readthedocs.io/en/latest/010_introduction.html) registry. I will also disable any swap by just commenting out since I dont need swapping when running containers.

#### Creating a CIFs Mount

In `/etc/fstab` add the following, with `192.168.178.101` being my NAS IP and `/timemachine/` the share I use for backups.

```txt
//192.168.178.101/timemachine/docker01 /backups cifs credentials=/home/homelab/.cifs,uid=root,noperm,rw 0 0
```

Now, we run this mount as `root` but store the credentials in the `homelab` home directory. I do this because I ssh and work as `homelab` so if I need to modify that file, its easier to access and adjust. We also adjust its permissions later anyway. Just a bit of an fyi.

As `homelab` user

```txt
touch .cifs
sudo chmod 600 .cifs
sudo vim .cifs
sudo mkdir /backups
```
We also create the `backups` directory which we will mount to in a little bit.\

My .cifs file looks like the following,

```txt
username=Nas_User_account
password=MegaSecurePasswork
domain=Your_FQDN
```

When everything is ready you should be able to run `sudo mount /backups` and it should connect to your CIFs directory via username and password. I will go into how I use restic at some other time but this directory is pretty important to it and should be configured from the start.

## Outro

This pretty much sums up how I get a base-line host going for the purpose of hosting Docker containers. I provide the package sources so updating docker is as easy as `apt update` and `apt upgrade`. I have mapped to my NAS for the purpose of backups, but this is solid enough example for CIFs mounting with a secret file as well as setting a static IP adress. In the next couple of posts I hope to dive a bit more into what I use for backups and how I have things setup there so do stick around!
