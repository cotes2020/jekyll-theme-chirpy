---
title: Linux - Setup NFS with Authentication AUTHGSS
date: 2020-07-16 11:11:11 -0400
categories: [31Linux, CentOS8]
tags: [Linux, Setup, NFS, Authentication]
math: true
image:
---


# Install NFS Server in CentOS 8


[toc]

Assignment under:

LFCE: Advanced Network and System Administration / Configuring and Managing NFS - [LFCEbyPluralsight](https://app.pluralsight.com/library/courses/advanced-network-system-administration-lfce/table-of-contents)

- Installing and Configuring an NFS Server, Adding Exports
- Installing and Configuring an NFS Client
- Mounting Exports and Listing a Remote Server's Exports


## configure and manage NFS, share resource

`exports`: shared resources
- in /etc/exports:
  - /share1 server1.psdemo.local
  - rw/ro: read only
  - async/sync:
    - trade off of the perform and safety
    - server reply the right request when the I/O flush/commit to disc
    - write acknowledgement
  - edelay: delay write to disc. if NFS knows other request in coming soon.
  - root_squash: client request come as root, UID 0, NFS map the UID to anonymous. prevent root access.
  - all_squash:
    - all users map to an anonymous user.
    when resource for public with low security.
  - sec=krb5, krb5i, krb5p
    - authentication settings. default=sys
    - krb5 = user authentication only
    - krb5i = add integrity checking
    - krb5p = add Encryption, most secure.

`exportfs` update and maintains a table of exports in server.

table path: /var/lib/nfs/etab
table holds the run time configuration of exports


host:
- single machine:
  - IP/name: `server1.psdemo.local`
- IP network:
  - CIDR Notation: `192.168.2.0/25`
- range: * ? [a-z]
  - `*.psdemo.local`
  - `server?.psdemo.local` : any single value character.
  - `server[2-9].psdemo.local`


1. runtime mounting:
mount -t nfs -o rw server0.psdemo.local:/share1 /mnt/share1

2. persistent mounting: (keep mount after reboot)
/etc/fstab
server0.psdemo.local:/share1 /mnt/share1 nfs rw 0 0

3. dynamic/on demand mounting:
autofs

4. global server-level/mount-point-level NFS options:
/etc/nfsmount.conf


### runtime mount

server0:

```c

step 1. install nfs server

$ sudo yum install nfs-utils
$ yum info nfs-utils

$ systemctl start nfs-server
$ systemctl enable nfs-server
$ systemctl status nfs-server

$ sudo systemctl enable rpcbind
$ sudo systemctl start rpcbind
$ sudo systemctl status rpcbind


$ sudo firewall-cmd --permanent --zone=public --add-service=nfs
$ sudo firewall-cmd --permanent --zone=public --add-service=rpc-bind
$ sudo firewall-cmd --permanent --add-service=mountd
$ sudo firewall-cmd --reload


step 2. setup DNS to manage naming of system

$ sudo vi /etc/hosts
// modify
127.0.0.1   localhost localhost.localdomain localhost4 localhost4.localdomain4
::1         localhost localhost.localdomain localhost6 localhost6.localdomain6
192.168.1.1     server0.psdemo.local
192.168.1.100   server1.psdemo.local
192.168.2.100   server2.psdemo.local



step 3. define exports for servers

$ sudo mkdir /share1

$ sudo vi /etc/exports
/share1 server1.psdemo.local

// reload the file
$ sudo exportfs -arv
exportfs: No options for /share1 server1.psdemo.local: suggest server1.psdemo.local(sync) to avoid warning
exporting server1.psdemo.local:/share1


// check the rumtime configuration of NFS
$ cat /var/lib/nfs/etab
/share1	server1.psdemo.local(ro,sync,wdelay,hide,nocrossmnt,secure,root_squash,no_all_squash,no_subtree_check,secure_locks,acl,no_pnfs,anonuid=65534,anongid=65534,sec=sys,ro,secure,root_squash,no_all_squash)
// default options


$ sudo vi /etc/exports
/share1 server1.psdemo.local(rw)
$ sudo exportfs -arv              // no warning
exporting server1.psdemo.local:/share1
$ cat /var/lib/nfs/etab
/share1	server1.psdemo.local(rw,sync...)


// no space
/share1 server1.psdemo.local (rw)
$ sudo exportfs -arv
exportfs: No options for /share1 server1.psdemo.local: suggest server1.psdemo.local(sync) to avoid warning
exportfs: No host name given with /share1 (rw), suggest *(rw) to avoid warning
exporting server1.psdemo.local:/share1
exporting *:/share1  // allow anyone to access share1
```

client:

```c


step 1: setup client side

// set the dns setting for each client:
$ sudo vi /etc/hosts
127.0.0.1   localhost localhost.localdomain localhost4 localhost4.localdomain4
::1         localhost localhost.localdomain localhost6 localhost6.localdomain6
192.168.1.1     server0.psdemo.local
192.168.1.100   server1.psdemo.local
192.168.2.100   server2.psdemo.local


// install
$ sudo yum install nfs-utils

$ sudo systemctl status nfs-server
$ sudo systemctl status rpcbind


$ sudo firewall-cmd --permanent --zone=public --add-service=nfs
$ sudo firewall-cmd --permanent --zone=public --add-service=rpc-bind
$ sudo firewall-cmd --permanent --add-service=mountd
$ sudo firewall-cmd --reload

// mount the exports
$ sudo mount -t nfs server0.psdemo.local:/share1 /mnt

// check
$ mount | grep server0
server0.psdemo.local:/share1 on /mnt type nfs4 (rw,relatime,vers=4.2,rsize=524288,wsize=524288,namlen=255,hard,proto=tcp,timeo=600,retrans=2,sec=sys,clientaddr=192.168.1.100,local_lock=none,addr=192.168.1.1)


================================================================

didnot know the exports
ask NFS server, need to enable the rpcbind server on the NFS server

server1:
$ showmount -e server0.psdemo.local // or 192.168.1.1
Export list for 192.168.1.1:
/share1 server1.psdemo.local

$ showmount -e 192.168.1.1
clnt_create: RPC: Unable to receive
# firewall： server systemctl stop firewalld, client success.

server side:
# firewall-cmd --permanent --add-service=nfs
# firewall-cmd --permanent --add-service=rpcbind
# firewall-cmd --permanent --add-service=mountd // this one!!!
# firewall-cmd --reload

================================================================
```

### persistent mount

reboot all lost, rumtime mount

server1:

```c
$ sudo vi /etc/fstab
// add
/dev/mapper/cl-root     /                       xfs     defaults        0 0
UUID=ca2f12da-b776-4a8a-9dd0-308da349067a /boot                   ext4    defaults        1 2
/dev/mapper/cl-swap     swap                    swap    defaults        0 0
server0.psdemo.local:/share1	     /mnt	        nfs	    defaults,rw,_netdev  0 0
// _netdev: not attempt to mount file system until the network device is online, since nfs through network.

$ mount

$ showmount -e server0.psdemo.local
Export list for server0.psdemo.local:
/share1 server?.psdemo.local

$ sudo umount /mnt/

// update configuration file
$ sudo mount -a

// new mount show
$ mount | grep server0
server0.psdemo.local:/share1 on /mnt type nfs4 (rw,relatime,vers=4.2,rsize=524288,wsize=524288,namlen=255,hard,proto=tcp,timeo=600,retrans=2,sec=sys,clientaddr=192.168.1.100,local_lock=none,addr=192.168.1.1,_netdev_)

```

### dynamic monut: autofs

autofs
a daemon that runs on computer that dynamically mount shares/exports/anything/even local file systems

server1:

```c
$ sudo yum install autofs
$ systemctl enable autofs


$ sudo vi /etc/auto.misc
// add
# This is an automounter map and it has the following format key [ -mount-options-separated-by-comma ] location
# Details may be found in the autofs(5) manpage

cd               -fstype=iso9660,ro,nosuid,nodev :/dev/cdrom
share1           -fstype=nfs,rw          server0.psdemo.local:/share1

# the following entries are samples to pique your imagination
#linux          -ro,soft,intr           ftp.example.org:/pub/linux
#boot           -fstype=ext2            :/dev/hda1
#floppy         -fstype=auto            :/dev/fd0
#floppy         -fstype=ext2            :/dev/fd0
#e2floppy       -fstype=ext2            :/dev/fd0
#jaz            -fstype=ext2            :/dev/sdc1
#removable      -fstype=ext2            :/dev/hdd


$ systemctl restart autofs
$ ls /misc  // nothing
$ ls /misc/share1
$ ls /misc
share1      // share1 shows up, on demands

```

### mount unsuccessful

server2:

```c
$ sudo mount -t nfs server0.psdemo.local:/share1 /mnt/
mount.nfs: access denied by server while mounting server0.psdemo.local:/share1
```

server0:

```c
// change
$ vi /etc/exports
/share1 server?.psdemo.local(rw)   // not server1.psdemo.local(rw)

// update to reread the configuration file
$ sudo exportfs -arv
exporting server?.psdemo.local:/share1
```

server2:

```c
$ sudo mount -t nfs server0.psdemo.local:/share1 /mnt/

$ mount | grep server0
server0.psdemo.local:/share1 on /mnt type nfs4 (rw,relatime,vers=4.2,rsize=524288,wsize=524288,namlen=255,hard,proto=tcp,timeo=600,retrans=2,sec=sys,clientaddr=192.168.2.100,local_lock=none,addr=192.168.1.1)
```
