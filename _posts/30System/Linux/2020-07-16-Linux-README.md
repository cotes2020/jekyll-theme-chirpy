---
title: Linux - Learning Path
date: 2020-07-16 11:11:11 -0400
categories: [30System, Linux]
tags: [OnePage, Linux]
math: true
image:
---


# Linux - Learning Path

- [Linux - Learning Path](#linux---learning-path)
  - [some stone](#some-stone)
  - [course](#course)
    - [Advanced Linux Networking](#advanced-linux-networking)
    - [Advanced Network and System Administration](#advanced-network-and-system-administration)
    - [LFCE: Network and Host Security](#lfce-network-and-host-security)
    - [CentOS Enterprise Linux User and Group Management](#centos-enterprise-linux-user-and-group-management)

```
    __∧_∧__    ~~~~~
　／(*´O｀)／＼
／|￣∪∪￣|＼／
　|＿＿ ＿|／
```

---

## some stone

Topic | Chapter
---|---|
✔️ | [Build Own RPM package in CentOS 8](https://github.com/ocholuo/system/blob/master/linux/BuildOwnRPMpackageinCentOS8.md)
✔️ | [Create Own Local or Http Repository](https://github.com/ocholuo/system/blob/master/linux/CreateOwnLocalorHttpRepository.md)
✔️ | [Install Kerberos Authentication in Centos 8](https://github.com/ocholuo/system/blob/master/linux/step/InstallKerberosAuthenticationinCentos8.md)
✔️ | [Install NTP in Centos 8](https://github.com/ocholuo/system/blob/master/linux/step/t.HowtoInstallNTPinCentOS8.md)
✔️ | [Install OpenLDAP Directory in CentOS 8](https://github.com/ocholuo/system/blob/master/linux/step/InstallOpenLDAPDirectoryinCentOS8)
✔️ | [Setup NFS with Authentication (AUTH_GSS)](https://github.com/ocholuo/system/blob/master/linux/SetupNFSwithAuthenticationAUTHGSS.md)
✔️ | [Setup NFS with Authentication (AUTH_SYS)](https://github.com/ocholuo/system/blob/master/linux/SetupNFSwithAuthenticationAUTHSYS.md)

---

## course

### Advanced Linux Networking

No. | Work
---|---|
:chestnut: | [Advanced Linux Networking](https://github.com/ocholuo/system/blob/master/linux/note/LFCEbyPluralsight/LFCE.AdvancedLinuxNetworking.md)
✔️ | Virtualbox install |
✔️ | Build CentOS in AWS EC2 |
✔️ | Build GUI and Remote Desktop VNC viewer |
✔️ | yum install the package |
. | yum install gcc gcc-c++ bison flex libpcap-devel qt-devel gtk3-devel rpm-build libtool c-ares-devel qt5-qtbase-devel qt5-qtmultimedia-devel qt5-linguist desktop-file-utils
. | yum -y install epel-release
✔️ | Network setups (Internal network between VMs, vi DNS)|
. | vi /etc/sysconfig/network-scripts/ifcfg-enp0s3
. | vi /etc/resolv.conf
✔️ | Share clipboard in VMs|
✔️ | encapsulate package using tcpdump&Wireshark |
✔️ | Wireshark permission setup
✔️ | setup subnet |
. | PREFIX=24
✔️ | setup NAT |
. | # firewall-config
✔️ | delete the GNOME ... GG :smiley: |
✔️ | backup, clone the VM and reconfigure |
✔️ | operate the ARP |
✔️ | set DNS |
. | etc/recolv.conf
✔️ | Fragment the packets |
. | ping -c 1 -s 1473 192.168.1.1
. | ping -t 4 4.2.2.2
✔️ | reconfigure route |
. | ip route add 192.168.1.1/24 via 192.168.2.1 dev enp0s3
✔️ | setup Virtual LAN using command line |
. | ifcfg-enp0s3.42 BOOTPROTO=none VLAN=yes
✔️ | **check the listening service** |
. | -ss -ltn -t -4
. | netstat -an | head
✔️ | analysis TCP connection package |
✔️ | analysis UDP connection package |
✔️ | manipulate a latency of the network, congestion control |
. | tc qdisc add dev enp0s8 root netem delay 3000ms loss 5%

---

### Advanced Network and System Administration

No. | Work
---|---|
:chestnut: | [Advanced Network and System Administration](https://github.com/ocholuo/system/blob/master/linux/note/LFCEbyPluralsight/LFCE.AdvancedNetworkandSystemAdministration.md)
✔️ | check the init daemons, Starting and Stopping Services |
✔️ | configure the unit file precedence and Automatic Restart |
. | $ sudo systemctl daemon-reload
. | $ systemctl restart httpd
✔️ | **systemd: target**
. | $ ls -lah /etc/systemd/system/multi-user.target.wants | grep httpd
. | $ systemctl target_name
. | $ systemctl get-default graphical
. | $ systemctl set-default graphical
✔️ | **systemd: control group**
. | $ systemd-cgls  *<font color=red> <--currently-running processes on system </font>*
. | $ systemd-cgtop
✔️ | **limit access to system resources with policy**
. | $ systemctl show service -p attribute
. | $ sudo vi /etc/systemd/system/httpd.service
✔️ | **search info though Logs with journalctl**
. | $ sudo journalctl
. | $ sudo journalctl -f // from bottom
. | $ sudo journalctl -o verbose
. | `$ sudo journalctl _UID=1000`
. | $ sudo journalctl --since 19:00:00
✔️ | **operate tftp server and client, xinetd**
. | `echo "whatsup" | cd tee /var/lib/tftpboot/echo.txt`
. | `echo "get echo.txt" | sudo tftp 192.168.1.1`
. | `sudo tftp 192.168.1.1 -c get echo.txt`
✔️ | **investigating system memory**
. | `$ cat /proc/meminfo | sort` *<font color=red> <-- check memoryinfo </font>*
. | $ dd if=/dev/zero of=test1.img bs=1 count=100000 oflag=sync *<font color=red> <-- generate customize load </font>*
. | $ free -m *<font color=red> <-- check memory usage </font>*
. | $ top *<font color=red> <-- dynamic real-time Linux processes </font>*
✔️ | **create c function to isolate a memory hog on system**
. | $ vmstat 1 100 *<font color=red> <-- how much disk I/O generating </font>*
. | $ dstat
✔️ | **memory and disk**
. | $ sudo iotop *<font color=red> <-- find high I/O processes </font>*
. | $ sudo blockdev --getbsz /dev/sda2 *<font color=red> <-- find block size </font>*
✔️ | **check interface buffers and socket queues, find network hogs**
. | $ sudo iptraf-ng *<font color=red> <-- monitor network interface </font>*
. | $ ss -t4 *<font color=red> <-- check queues </font>*
✔️ | **check system realtime status or monitor for specific time**
. | $ vi /etc/sysconfig/sysstat *<font color=red> <-- long-term performance data </font>*
. | $ sudo vi /etc/cron.d/sysstat *<font color=red> <-- create monitor </font>*
. | $ less /var/log/sa/sar23
. | $ sar -u -A
. | $ sar -A -s 21:38:00 -e 22:00:00
. | $ sadf -d | head
. | $ dstat -t --all --tcp *<font color=red> <-- realtime </font>*
✔️ | **package manage: rpm**
. | $ mount | grep /dev/sr0 -> /dev/sr0 on /run/media/server1/CentOS7 type iso9660
. | $ cd /run/media/server1/CentOS7/packages -> mc-4.8.7-8.el7.x86_64.rpm
. | $ rpm -ql mc.rpm -> (tar.gz + spec) *<font color=red> <-- list file in rpm </font>*
. | $ rpm -K mc.rpm *<font color=red> <-- check signiture </font>*
. | $ rpm -ivh mc.rpm *<font color=red> <-- install verbose hashes </font>*
. | $ rpm -Uvh mc *<font color=red> <-- update </font>*
. | $ rpm -e mc *<font color=red> <-- remove </font>*
. | $ rpm -qa mc *<font color=red> <-- query the rpm </font>*
. | $ rpm -qi mc
. | $ rpm -qf /etc/fstab *<font color=red> <-- check rpm from folder </font>*
. | $ rpm2cpio mc-4.8.7-8.el7.src.rpm | cpio -id *<font color=red> <-- extract copy </font>*
✔️ | **install rpm packages from source file**
. | $ ls -R rpmbuild/ *<font color=red> <-- check </font>*
. | $ rpmbuild --rebuild mc.rpm *<font color=red> <-- from source file: rpm </font>*
✔️ | **build ur own rpm from code**
. | $ tar -zxvf hello-1.tar.gz --> (c program, makefile)
. | $ mv hello-1.tar.gz ~/rpmbuild/SOURCES/
. | $ mv hello.spec ~/rpmbuild/SPECS/
. | $ rpmbuild -ba hello.spec
. | $ ls -R rpmbuild/ *<font color=red> <-- got rpm </font>*
. | $ rpmbuild -ivh rpmbuild/RPMS/x86_64/hello.rpm
✔️ | **package & dependency manage: yum (yellowdog, update, modify)**
✔️ | **manage repository configuration**
. | $ more /etc/yum.conf
. | $ yum repolist
✔️ | **create own local repository**
. | $ ls /etc/yum.repos.d
. | $ mount -o loop CentOS-DVD1.iso /var/repo/local
. | $ vi /etc/yum.repos.d/local.repo
. | baseurl=file:///var/repo/local
✔️ | **create own HTTP based repository**
. | $ mkdir -p /var/www/html/custom
. | $ vi /etc/yum.repos.d/custom.repo
. | baseurl=http://192.168.1.1/custom
. | $ sudo cp ypserv.rpm /var/www/html/custom/ *<font color=red> <-- add custom rpm package to repo </font>*
. | $ sudo yum createrepo . *<font color=red> <-- each time add rpm </font>*
. | $ sudo yum clean all
. | $ sudo yum makecache *<font color=red> <-- update repo </font>*
✔️ | **setup runtime NFS server client**
. | $ systemctl start nfs-server
. | $ systemctl start rpcbind
. | $ sudo vi /etc/hosts *<font color=red> <-- DNS, system naming </font>*
. | $ cat /var/lib/nfs/etab *<font color=red> <-- runtime config </font>*
. | $ showmount -e *<font color=red> <-- check </font>*
. | $ sudo vi /etc/exports -> /share1 server1.psdemo.local *<font color=red> <-- setup mount only for server1 </font>*
. | $ sudo exportfs -arv *<font color=red> <-- reread config </font>*
. | $ sudo mount -t nfs server0.psdemo.local:/share1 /mnt *<font color=red> <-- mount </font>*
✔️ | **setup persistent NFS server client**
. | $ sudo vi /etc/fstab *<font color=red> <-- client setup </font>*
. | server0.psdemo.local:/share1	     /mnt	        nfs	    defaults,rw,_netdev  0 0_
. | $ sudo mount -a *<font color=red> <-- reread config </font>*
. | `$ echo "server0.psdemo.local:/share1     /mnt  nfs     defaults,rw,sec=krb5 0 0">>/etc/fstab`
✔️ | **setup dynamic monut: autofs**
. | $ systemctl enable autofs
. | $ sudo vi /etc/auto.misc
. | share1           -fstype=nfs,rw          server0.psdemo.local:/share1
. | $ systemctl restart autofs
. | $ ls /misc/share1 *<font color=red> <-- 只有在点名path后才会出现的mount </font>*
✔️ | **SELinux Monitor**
. | nfsstat, nfsiostat, mountstats
✔️ | **Accidentally delete the snapshot. GG again**

---

### LFCE: Network and Host Security

No. | Work
---|---|
:chestnut: | [LFCE: Network and Host Security](https://github.com/ocholuo/system/blob/master/linux/note/LFCEbyPluralsight/LFCE.LFCE.NetworkandHostSecurity.md)
✔️ | **Configuring Local and Remote Logging with rsyslog**
. | **server side**:
. | $ systemctl status rsyslog
. | $ cat /etc/rsyslog.conf *<font color=red> <-- config file </font>*
. | module(load="imtcp") input(type="imtcp" port="514") *<font color=red> <-- uncommand </font>*
. | $ semanage port -a -t syslogd_port_t -p tcp 514
. | $ firewall-cmd --permanent --add-port 514/tcp
. | $ tail -f /var/log/messages *<font color=red> <-- check log </font>*
. | **client side**:
. | $ cat /etc/rsyslog.conf
. | `*.*  @@server1.demo.local:514` *<font color=red> <-- add line </font>*
. | $ logger "Test Message"
✔️ | **iptables&TCP wrappers**
. | $ iptables -L --line-numbers  # with details
. | $ iptables -I INPUT -s 192.168.2.100 -p tcp -m tcp --dport 22 -j ACCEPT
✔️ | **firewalld**
. | $ firewall-cmd --zone=public --remove-port=514/tcp
. | $ firewall-cmd --zone=public --add-port=514/tcp
. | config file
. | `/usr/lib/firewalld/`, `/etc/firewalld`
. | $ firewall-cmd --runtime-to-permanent
✔️ | **NAT**
. | **server side**:
. | script: `firewall.sh`
. | firewall-cmd --permanent --change-interface=enp0s10 --zone=external
. | firewall-cmd --permanent --change-interface=enp0s8 --zone=internal
. | firewall-cmd --permanent --change-interface=enp0s9 --zone=internal
. | firewall-cmd --permanent --zone=external --add-masquerade
. | firewall-cmd --permanent --direct --add-rule ipv4 filter FORWARD 0 -s 192.168.2.0/25 -j ACCEPT
. | firewall-cmd --permanent --direct --add-rule ipv4 filter FORWARD 0 -s 192.168.3.0/24 -j ACCEPT
. | $ firewall-cmd --zone=external --list-all *<font color=red> <-- masquerade:yes </font>*
. | *<font color=red> <-- --add-forward-port = port=2233(inport): proto=tcp: toport=22(outport): toaddr=192.168.3.100 </font>*
. | $ firewall-cmd --permanent --zone=external --add-forward-port=port=2223:proto=tcp:toport=22:toaddr=192.168.3.100
. | $ firewall-cmd --permanent --zone=external --add-forward-port=port=2222:proto=tcp:toport=22
. | $ firewall-cmd --reload *<font color=red> <-- not runtime, after reload </font>*
. | **mac terminal**:
. | $ ssh server@192.168.1.30 -p 2223  *<font color=red> <-- to serverIP 2223port, forward to other thought 22port</font>*
. | [server@server1 ~]$ exit
. | $ ssh server@192.168.1.30 -p 2222 *<font color=red> <-- to serverIP 2222port, forward to itself thought 22port</font>*
✔️ | **openSSH**
. | $ ssh-keygen
. | $ ssh-copy-id -i id_rsa.pub server2.demo.local
. | $ ssh server@server2.demo.local
. | Enter key password:
. | $ scp file.txt server@server2.demo.local:~/
. | $ ssh server@server2.demo.local ls
. | $ vi `remotecmd.sh` *<font color=red> <-- Automate</font>*
. | remotecmd.sh
. | while read name
. | do
. |   ssh -l server $name -n "ps -aux --noheaders | sort -nrk 3 | head"
. | done < myhosts
✔️ | **SSH tunnel**
. | $ ssh -L 8080:server2.demo.local:80 server2.demo.local
. | $ curl http://localhost:8080
✔️ | **x11**
✔️ | **vnc**
. | $ sudo cp /usr/lib/systemd/system/vncserver@.service /etc/systemd/system/
. | $ sudo vi /etc/systemd/system/vncserver@.service
. | $ vncpasswd
. | $ sudo systemctl start vncserver@:1.service
. | $ vncviewer server2.demo.local:1
. | $ ssh -L 5901:server2.demo.local:5901 server2.demo.local -N
. | $ vncviewer localhost:5901

---

### CentOS Enterprise Linux User and Group Management

No. | Work
---|---|
[:chestnut:](https://app.pluralsight.com/course-player?clipId=5154a368-2082-40ff-991b-af16e2e0f6f7) | [CentOS Enterprise Linux User and Group Management](https://github.com/ocholuo/system/blob/master/linux/note/LFCEbyPluralsight/CentOSEnterpriseLinuxUserandGroupManagement.md)
✔️ | **stop auto create Homedir until login** |
. | $ /etc/login.defs *<font color=red> <-- turn off auto create </font>*
. | $ systemctl start oddjob
. | $ authconfig --enablemkhomrdir --update
✔️ | **operate passwd policy** |
. | $ pwscore
. | /etc/security/pwquality.conf *<font color=red> <-- configure pw role </font>*
✔️ | **limit user access** |
. | $ unlimit -a -u
. | $ /etc/security/limits.conf
. | soft limit <= hard limit
✔️ | **setup access time** |
. | /etc/pam.d/sshd <- account required pam_time.so
. | `etc/lib64/security/time.so <- sshd;*;tux|bob;Wk0800-1800`
✔️ | **setup kerberos** |
. | **1. setup NTP** *<font color=red> <-- sync time </font>*
. | **2. setup reandom generator**
. | **3. setup kerberos**
. | 3.1. server hosts file: `/etc/hosts`
. | 3.1. Server Install the KDC Server
. | 3.2. Serverconfigure the file: *<font color=red> 3 config file </font>* `/var/kerberos/krb5kdc/kadm5.acl` `/var/kerberos/krb5kdc/kdc.conf` `/etc/krb5.conf`
. | 3.3. Server 配置 NFS 共享目录: `/etc/exports`
. | 3.4. Server 初始化 KDC 数据库:
. | $ kdb5_util create -s -r PSDEMO.LOCAL *<font color=red> <-- create server </font>*
. | 3.5. Server setup firewall and start service.
. | 3.6. server 配置 KDC 数据库 & 取得密钥: Add Kerberos principals
. | kadmin.local:  addprinc root/admin
. | kadmin.local:  addprinc -randkey host/server0.psdemo.local
. | kadmin.local:  ktadd host/server0.psdemo.local
✔️ | **setup kerberos for NFS** |
. | 1. Server 配置 NFS 共享目录: /etc/exports
. | 2. NFS client setup:
. | 2.1. hosts file
. | 2.2. install NFS、Kerberos
. | 2.3. 修改 Kerberos 配置文件
. | 2.4. 从 KDC 上取得密钥
. | 2.5. 启动 NFS 加密服务
. | 2.6. 连接 NFS 服务
✔️ | **setup kerberos for ssh**
. | 1. Server:
. | $ vi /etc/ssh/ssh_config -> GSSAPIAuthentication GSSAPIDelegateCredentials yes
. | systemctl reload sshd
. | authconfig --enablekrb5 --update

---













.
