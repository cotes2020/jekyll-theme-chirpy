---
title: Linux - Install OpenLDAP Directory in CentOS8
date: 2020-07-16 11:11:11 -0400
categories: [30System, CentOS8]
tags: [Linux, Install, OpenLDAP, CentOS8]
math: true
image:
---

# Install OpenLDAP Directory in CentOS 8

[toc]

Assignment under:

CentOS Enterprise Linux 7 User and Group Management / Implementing OpenLDAP - [LFCEbyPluralsight](https://app.pluralsight.com/course-player?clipId=5154a368-2082-40ff-991b-af16e2e0f6f7)


## install OpenLDAP

server0:

```c

==================================================================

step 1: check the domain name.

$ hostnamectl set-hostname server0.lab.com
$ hostname
server0.lab.com

$ echo "192.168.1.1 server0.lab.com" >> /etc/hosts
$ ping server0.lab.com  // success

==================================================================

step 2: open the Port

# netstat -ltn
Active Internet connections (only servers)
Proto Recv-Q Send-Q Local Address           Foreign Address         State
tcp        0      0 0.0.0.0:5355            0.0.0.0:*               LISTEN
tcp        0      0 0.0.0.0:111             0.0.0.0:*               LISTEN
tcp        0      0 0.0.0.0:20048           0.0.0.0:*               LISTEN
tcp        0      0 0.0.0.0:38481           0.0.0.0:*               LISTEN
tcp        0      0 192.168.122.1:53        0.0.0.0:*               LISTEN
tcp        0      0 0.0.0.0:22              0.0.0.0:*               LISTEN
tcp        0      0 0.0.0.0:44855           0.0.0.0:*               LISTEN
tcp        0      0 127.0.0.1:631           0.0.0.0:*               LISTEN
tcp        0      0 0.0.0.0:2049            0.0.0.0:*               LISTEN
tcp6       0      0 :::48649                :::*                    LISTEN
tcp6       0      0 :::5355                 :::*                    LISTEN
tcp6       0      0 :::111                  :::*                    LISTEN
tcp6       0      0 :::20048                :::*                    LISTEN
tcp6       0      0 :::80  http             :::*                    LISTEN
tcp6       0      0 :::22  ssh              :::*                    LISTEN
tcp6       0      0 ::1:631                 :::*                    LISTEN
tcp6       0      0 :::443 https            :::*                    LISTEN
tcp6       0      0 :::36287                :::*                    LISTEN
tcp6       0      0 :::2049                 :::*                    LISTEN

# firewall-cmd --permanent --add-service=ldap

==================================================================

step 3: install the OpenLDAP

# sudo yum install -y openldap openldap-clients openldap-servers
# sudo yum install -y migrationtools.noarch
// easy way to take user form passwd file and migrate them to LDAP
// use it to create template
```

### centos8: build it from source

```c
https://tylersguides.com/guides/install-openldap-from-source-on-centos-8/

// Install Required Dependencies and Build Tools
$ yum install cyrus-sasl-devel make libtool autoconf libtool-ltdl-devel openssl-devel libdb-devel tar gcc perl perl-devel wget vim
$ dnf install make tar gcc openssl-devel libtool-ltdl-devel libdb-devel cyrus-sasl-devel

// Create OpenLDAP System Account

// executed as root
// Create the file `/root/.ldap-env` with the following contents:
$ export PATH=/opt/openldap-current/bin:/opt/openldap-current/sbin:/opt/openldap-current/libexec:$PATH
$ export OWNER=ldap:ldap
$ export CONFIG=/opt/openldap-current/etc/openldap/slapd.d

# touch /root/.ldap-env
# source /root/.ldap-env


// download the file
# curl "ftp://ftp.openldap.org/pub/OpenLDAP/openldap-release/openldap-2.4.48.tgz" > openldap-2.4.48.tgz

# tar xf openldap-2.4.48.tgz
# cd openldap-2.4.48

# ./configure --with-cyrus-sasl --with-tls=openssl --enable-overlays=mod \
    --enable-backends=mod --disable-perl --disable-ndb --enable-crypt \
    --enable-modules --enable-dynamic --enable-syslog --enable-debug --enable-local \
    --enable-spasswd --disable-sql --prefix=/opt/openldap-2.4.48

// Compile the source:
# make depend
# make


# cd contrib/slapd-modules/passwd/sha2
# make
# cd ../../../..


// run the test suite to make sure everything went ok
make test > test_results.txt
grep '>>>>>.*failed' test_results.txt

// Installing OpenLDAP on CentOS 8
make install
cd contrib/slapd-modules/passwd/sha2
make DESTDIR=/opt/openldap-2.4.48 install
../../../../libtool --finish /opt/openldap-2.4.48/usr/local/libexec/openldap
cd  /opt/openldap-2.4.48/usr/local/libexec/openldap
mv * /opt/openldap-2.4.48/libexec/openldap


// Create the server’s user and group.
The commands below will create the same user and group as the package manager would if you were to install the OpenLDAP server from the repositories.
groupadd -g 55 ldap
useradd  -g 55 -u 55 -s /sbin/nologin -d /var/lib/ldap -c "OpenLDAP server" ldap


// Fix the permissions on a few directories:
cd /opt/openldap-2.4.48
find . -type d -exec chmod 755 {} \;
cd var/run/
chown $OWNER .


// Create a symbolic link that points to the version of OpenLDAP you are currently using:
ln -s /opt/openldap-2.4.48 /opt/openldap-current


// Create a service unit file
// so systemd can start and stop the server on boot and shutdown.
vi /etc/systemd/system/slapd-current.service.

[Unit]
Description=OpenLDAP Server Daemon
After=syslog.target network-online.target

[Service]
Type=forking
PIDFile=/opt/openldap-current/var/run/slapd.pid
EnvironmentFile=/etc/sysconfig/slapd-current

ExecStart=/opt/openldap-current/libexec/slapd -u ldap -g ldap -h ${SLAPD_URLS} $SLAPD_OPTIONS

[Install]
WantedBy=multi-user.target


// Configure the Server
...
```

## configure OpenLDAP server


## create directory structure


## create groups and users




















.
