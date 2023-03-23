---
title: Linux - CentOS Enterprise User and Group Management
date: 2020-07-16 11:11:11 -0400
categories: [00CodeNote, LinuxNote]
tags: [Linux, LFCE, Pluralsight]
math: true
image:
---


# CentOS Enterprise Linux User and Group Management

[toc]

## PAM

PAM: pluggable authentictaion module

`user login` (SSH, Condole, GUI...) -> `PAM` (Authentication, limits, home directory) -> `access seecion` (monitor the process)


directory to provide service.

```c
//configuration file for the program that use PAM
$ ls /etc/pam.d
atd               gdm-launch-environment  polkit-1          sudo-i
chfn              gdm-password            postlogin         su-l
cups              login                   sshd              vsftpd
fingerprint-auth  other                   sssd-shadowutils  xserver
gdm-autologin     passwd                  su
gdm-fingerprint   password-auth           sudo

// .so file that make up the shared library that PAM is going to use.
ls /lib64/security/
pam_access.so         pam_lastlog.so           pam_shells.so
pam_cap.so            pam_limits.so            pam_ssh_add.so
pam_chroot.so         pam_listfile.so          pam_sss.so
pam_console.so        pam_localuser.so         pam_stress.so


// configuration file for module
$ ls /etc/security/
access.conf       console.perms.d  namespace.d     pwquality.conf.d
chroot.conf       group.conf       namespace.init  sepermit.conf
console.apps      limits.conf      opasswd         time.conf
console.handlers  limits.d         pam_env.conf
console.perms     namespace.conf
pwquality.conf //(configure policy for passwd)

```

---

## Automatic create Home Directory at Login

when batch-creating user for big group, home directory might be unnecessary.

1. stop create user home directory when create user.
2. only create Home Directory when user Login

```c
$ sudo vi /etc/login.defs
# If useradd should create home directories for users by default On RH systems, we do. This option is overridden with the -m flag on useradd command line.
CREATE_HOME     yes
// yes: automatically
// no: wont auto create home directory
```

```c
$ sudo vi /etc/login.defs
CREATE_HOME     no

$ sudo useradd bob
$ ls /home
// bob directory doesnt exists

$ grep bob /etc/passwd
bob:x:1001:1002::/home/bob(in red color):/bin/bash
// bob will have home directory, it need to be created.

// setup passwd
$ sudo passwd bob

$ rpm -qa | grep oddjob
$ systemctl enable oddjobd
$ systemctl start oddjobd
$ systemctl status oddjobd


$ sudo authconfig --enablemkhomedir --update
$ sudo -i
[root] $ cd /etc/pam.d
[root] # grep mkhomedir *
fingerprint-auth:session     optional     pam_oddjob_mkhomedir.so umask=0077
password-auth:session     optional     pam_oddjob_mkhomedir.so umask=0077
system-auth:session     optional     pam_oddjob_mkhomedir.so umask=0077
```

---

## configure password policy

```c
$ cat /etc/pam.d/system-auth
password    requisite    pam_pwquality.so try_first_pass local_users_only
password    sufficient   pam_unix.so sha512 shadow nullok try_first_pass use_authtok
password    sufficient   pam_sss.so use_authtok
password    required     pam_deny.so

============================================

configure password policy

$ less /etc/security/pwquality.conf
// passwd security
# Configuration for systemwide password quality limits
# Defaults:
#
# Number of characters in the new password that must not be present in the
# old password.
# difok = 1  //how many character should be different in the new passwd compare to the old passwd
#
# Minimum acceptable size for the new password (plus one if credits are not disabled which is the default). (See pam_cracklib manual.)
# Cannot be set to lower value than 6.
# minlen = 8
#
# The maximum credit for having digits in the new password. If less than 0 it is the minimum number of digits in the new password.
# dcredit = 0
#
# The maximum credit for having uppercase characters in the new password.
# If less than 0 it is the minimum number of uppercase characters in the new
# password.
# ucredit = 0
#
# Whether to check for the words from the passwd entry GECOS string of the user.
# The check is enabled if the value is not 0.
# gecoscheck = 0

============================================

test passwd quality

$ pwscore
// check passwd security. >100 is good
passwd
Password quality check failed:
 The password is shorter than 8 characters
$ pwscore
passwdjenny
56

```

--

## restrict access to resources

```c

// all restriction in placed
$ ulimit -a
core file size          (blocks, -c) unlimited
data seg size           (kbytes, -d) unlimited
scheduling priority             (-e) 0
file size               (blocks, -f) unlimited
pending signals                 (-i) 8847
max locked memory       (kbytes, -l) 16384
max memory size         (kbytes, -m) unlimited
open files                      (-n) 1024
pipe size            (512 bytes, -p) 8
POSIX message queues     (bytes, -q) 819200
real-time priority              (-r) 0
stack size              (kbytes, -s) 8192
cpu time               (seconds, -t) unlimited
max user processes              (-u) 8847
virtual memory          (kbytes, -v) unlimited
file locks                      (-x) unlimited


// look for number of user process allowed
$ ulimit -u
8847
// change it
$ ulimit -u 10


$ vi test.sh
// add
#!/bin/bash
eho "Test"
$0

$ chmod +x test.sh
$ ./test.sh
// only 10 time

after logout, it will reset.

===============

as admin

$ vi /etc/security/limits.conf
#<domain>      <type>  <item>         <value>
#*               soft    core            0
#*               hard    rss             10000
#@student        hard    nproc           20
#@faculty        soft    nproc           20
#@faculty        hard    nproc           50
#ftp             hard    nproc           0
*                -       maxlogins       4
@users shft nproc 50
@users hard nproc 75

===============

user:

$ ulimit -u
50
$ ulimit -u 70 // good
$ ulimit -u 80
bash: ulimit: max user processes: cannot modify limit: Operation not permitted
```

---

## control access times

```c
$ cd /etc/pam.d
$ ls
atd               gdm-launch-environment  polkit-1          sudo-i
chfn              gdm-password            postlogin         su-l
chsh              gdm-pin                 remote            system-auth
cockpit           gdm-smartcard           runuser           systemd-user
config-util       ksu                     runuser-l         vlock
crond             liveinst                smartcard-auth    vmtoolsd
cups              login                   sshd              vsftpd
fingerprint-auth  other                   sssd-shadowutils  xserver
gdm-autologin     passwd                  su
gdm-fingerprint   password-auth           sudo


// modify for the ssh connection
$ sudo vi sshd

#%PAM-1.0
auth       substack     password-auth
auth       include      postlogin

// add this line
account    required     pam_time.so

account    required     pam_sepermit.so
account    required     pam_nologin.so
account    include      password-auth
password   include      password-auth
# pam_selinux.so close should be the first session rule
session    required     pam_selinux.so close
session    required     pam_loginuid.so
# pam_selinux.so open should only be followed by sessions to be executed in the user context
session    required     pam_selinux.so open env_params
session    required     pam_namespace.so
session    optional     pam_keyinit.so force revoke
session    optional     pam_motd.so
session    include      password-auth
session    include      postlogin


$ cd /etc/security/
$ sudo vi time.conf

# Here is a simple example: running blank on tty* (any ttyXXX device), the users 'you' and 'me' are denied service all of the time
// add
*;*;user1|user2;Wk0800-1800
// for all service, for all terminal
// user1 or user2 able to login during Wk 08:00-18:00
#blank;tty* & !ttyp*;you|me;!Al0000-2400

# Another silly example, user 'root' is denied xsh access from pseudo terminals at the weekend and on mondays.
#xsh;ttyp*;root;!WdMo0000-2400

====================================================

*;*;user1|user2;!Wk0800-1800
// not a Wk 08:00-18:00
not ablt to login again
```














.
