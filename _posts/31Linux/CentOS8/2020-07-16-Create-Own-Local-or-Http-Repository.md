---
title: Linux - Create Own Local or Http Repository in CentOS8
date: 2020-07-16 11:11:11 -0400
categories: [31Linux, CentOS8]
tags: [Linux, CentOS8]
math: true
image:
---



# Create Own Local or Http Repository


[toc]

Assignment under:

LFCE: Advanced Network and System Administration / Advanced Package Management - [LFCEbyPluralsight](https://app.pluralsight.com/library/courses/advanced-network-system-administration-lfce/table-of-contents)


## repository

store software package
- software publisher: Redhat, CentOS
- third party: EPEL(extra packages for enterprise linux), RPMForge
- build you own.

trusted and authenticated


`/etc/yum.repos.d`: all repo configuration file

`/var/repo/dvd`: put custom.repo file


### 1. repository configuration

```c
// yum configuration file
$ more /etc/yum.conf
[main]
cachedir=/var/cache/yum/$basearch/$releasever  //where YUM caches packages locally with performing installations.
keepcache=0                //  after done installation, delete the cache
debuglevel=2
logfile=/var/log/yum.log
exactarch=1
obsoletes=1
gpgcheck=1
plugins=1
installonly_limit=3
# PUT YOUR REPOS HERE OR IN separate files named file.repo in /etc/yum.repos.d


$ cd /etc/yum.repos.d
$ ls
CentOS-AppStream.repo   CentOS-Extras.repo      CentOS-Vault.repo
CentOS-Base.repo        CentOS-fasttrack.repo   epel-modular.repo
CentOS-centosplus.repo  CentOS-HA.repo          epel-playground.repo
CentOS-CR.repo          CentOS-Media.repo       epel.repo
CentOS-Debuginfo.repo   CentOS-PowerTools.repo  epel-testing-modular.repo
CentOS-Devel.repo       CentOS-Sources.repo     epel-testing.repo


$ more CentOS-Base.repo
// check one of it
# CentOS-Base.repo
# The mirror system uses the connecting IP address of the client and the update status of each mirror to pick mirrors that are updated to and geographically close to the client.  You should use this for CentOS updates unless you are manually picking other mirrors.
# If the mirrorlist= does not work for you, as a fall back you can try the remarked out baseurl= line instead.

[BaseOS]
name=CentOS-$releasever - Base
mirrorlist=http://mirrorlist.centos.org/?release=$releasever&arch=$basearch&repo
=BaseOS&infra=$infra
// a copy of the package to download from
#baseurl=http://mirror.centos.org/$contentdir/$releasever/BaseOS/$basearch/os/
gpgcheck=1
// enabled=one. authenticates the remote repository with a digital signature, gpgkey.
gpgkey=file:///etc/pki/rpm-gpg/RPM-GPG-KEY-centosofficial
//local copy of the key provided from the repo.
enabled=1
// enabled = 0: turns this repository off from bing used when use the YUM commands to administer the packages on system. useful if no longer want or need packages from a particular repository.


// see all enabled repo on system
$ yum repolist
Last metadata expiration check: 5:19:07 ago on Sun 26 Apr 2020 02:54:57 PM EDT.
repo id         repo name                                                 status
AppStream       CentOS-8 - AppStream                                      4,830
BaseOS          CentOS-8 - Base                                           1,661
PowerTools      CentOS-8 - PowerTools                                     1,456
*epel           Extra Packages for Enterprise Linux 8 - x86_64            5,352
*epel-modular   Extra Packages for Enterprise Linux Modular 8 - x86_64        0
extras          CentOS-8 - Extras                                            15

$ yum -v repolist
// more version detail info
Repo-id      : AppStream
Repo-name    : CentOS-8 - AppStream
Repo-revision: 8.1.1911
Repo-distro-tags: [cpe:/o:centos:centos:8]:  , 8, C, O, S, e, n, t
Repo-updated : Wed 22 Apr 2020 01:16:06 AM EDT
Repo-pkgs    : 4,830
Repo-size    : 5.6 G
Repo-mirrors : http://mirrorlist.centos.org/?release=8&arch=x86_64&repo=AppStream&infra=stock
Repo-baseurl : http://mirrors.xtom.com/centos/8.1.1911/AppStream/x86_64/os/ (9 more)
Repo-expire  : 172,800 second(s) (last: Sun 26 Apr 2020 02:54:15 PM EDT)
Repo-filename: /etc/yum.repos.d/CentOS-AppStream.repo
// actual location
```

### 2. create own local repository base on DVD of CentOS

```c
install and save network resource
download dvd iso and make a yum repo configuration, use that to install the rpm.

ref:
https://www.itzgeek.com/how-tos/linux/centos-how-tos/create-local-yum-repository-on-centos-7-rhel-7-using-dvd.html
https://www.tecmint.com/setup-local-http-yum-repository-on-centos-7/
https://phoenixnap.com/kb/create-local-yum-repository-centos
============================================


$ yum install createrepo


- make a dir to mount the dvd

$ sudo mkdir -p /var/repo/dvd

// If the CD has been mounted automatically, then ignore this step. Otherwise, mount it manually.
$ sudo mount /dev/sr0 /var/repo/dvd
$ ls /var/repo/dvd      // content of the centos dvd

// If the ISO is present on the file system, mount it to /media/CentOS using the mount command with -o loop option.
$ mount -o loop CentOS-DVD1.iso /var/repo/dvd

$ ls /var/repo/dvd
AppStream  BaseOS  EFI  images  isolinux  media.repo  TRANS.TBL

$ more media.repo
[InstallMedia]
name=CentOS Linux 8
mediaid=None
metadata_expire=-1
gpgcheck=0
cost=500

============================================

- create the repo configuration

$ cd /etc/yum.repos.d
$ cp -v /var/repo/dvd/media.repo /etc/yum.repos.d/local-centos8.repo

// assign file permissions as shown to prevent modification or alteration by other users.
# chmod 644 /etc/yum.repos.d/local-centos8.repo
# ls -l /etc/yum.repos.d/local-centos8.repo


$ sudo vi /etc/yum.repos.d/local-centos8.repo
// add
[Local-Centos8-baseOS]
name=Local-CentOS8-BaseOS
metadata_expire=-1
enabled=1
baseurl=file:///var/repo/dvd/BaseOS/
gpgcheck=0

[Local-Centos8-AppStream]
name=Local-CentOS8-AppStream
metadata_expire=-1
enabled=1
baseurl=file:///var/repo/dvd/AppStream/
gpgcheck=0

//After modifying the repository file with new entries, proceed and clear the DNF / YUM cache as shown.
$ yum clean all

============================================

- disable other repo
$ sudo vi xx.repo
//add
enable=0


- check
$ yum repolist
repo id                 repo name                                           status
AppStream               CentOS-8 - AppStream                                5,402
BaseOS                  CentOS-8 - Base                                     1,661
Local-Centos8-AppStream Local-CentOS8-AppStream                             4,754
Local-Centos8-baseOS    Local-CentOS8-BaseOS                                1,659

$ sudo yum install vsftpd

$ sudo yum install ypserv
================================================================================
 Package         Arch      Version                           Repository    Size
================================================================================
Installing:
 ypserv          x86_64    4.0-6.20170331git5bfba76.el8      Local     171 k
Installing dependencies:
 tokyocabinet    x86_64    1.4.48-10.el8                     Local     486 k


- clean the cache
$ yum clean all
```


### 3. create own HTTP based repository

```c

Step 1: setup Web Server

$ yum list httpd
$ firewall-cmd --zone=public --permanent --add-service=http
$ firewall-cmd --zone=public --permanent --add-service=https
$ firewall-cmd --reload

$ systemctl start httpd
$ systemctl enable httpd
$ systemctl status httpd
// confirm that server is up and running
http://192.168.1.1


Step 2: Create Yum Local Repository

$ yum install createrepo
$ yum install yum-utils
// a better toolbox for managing repositories


- create directory: path of the http repo in file system
// create the necessary directories (yum repositories) that will store packages and any related information.
$ ls /var/www/html/
index.html
$ mkdir -p /var/www/html/custom

// synchronize CentOS YUM repositories to the local directories as shown.
// $ sudo reposync -m --repoid=BaseOS --newest-only --download-metadata -p DOWNLOAD_PATH=/var/www/html/repos/


Step 3: Create a Directory to Store the Repositories

- create repo configuration file: custom repo.

$ vi /etc/yum.repos.d/custom.repo
// change
[http]
name=Local HTTP repository
baseurl=http://192.168.1.1/custom
// default http Document root for HTTP server
enabled=1
gpgcheck=0
// for system want to use this repo
// just take this custom.repo file put it in /etc/yum.repos.d on that servers


Step 4: add own custom built package to repositories

- put package inside
$ cd /var/www/html/custom/
$ sudo yumdownloader ypserv
$ ls
ypserv-4.0-6.20170331git5bfba76.el8.x86_64.rpm



Step 5: Create the New Repository

// create the repo metadata database about the repo
// each time add packages
[server1@server0 custom]$ sudo createrepo .
Directory walk started
Directory walk done - 1 packages
Temporary output repo path: ./.repodata/
Preparing sqlite DBs
Pool started (with 5 workers)
Pool finished

$ sudo yum clean all //clean cache
$ sudo yum makecache // update repo info

```
