---
title: Network protocol - SSH
date: 2020-07-16 11:11:11 -0400
categories: [00Basic, Network]
tags: [SSH]
math: true
image:
---

- [Network protocol - SSH](#network-protocol---ssh)
  - [basic](#basic)
  - [ssh的安装](#ssh的安装)
  - [SSH Breakdown](#ssh-breakdown)
  - [配置ssh](#配置ssh)
  - [使用ssh在远程后台不中断地跑程序](#使用ssh在远程后台不中断地跑程序)
  - [ssh 服务器端把空闲连接断开](#ssh-服务器端把空闲连接断开)
  - [system administration](#system-administration)
    - [test if ssh is running](#test-if-ssh-is-running)
    - [change the port](#change-the-port)
  - [SSH基本用法](#ssh基本用法)
    - [Changing and Exploring Directories](#changing-and-exploring-directories)
    - [SSH远程登录](#ssh远程登录)
      - [1. 口令登录](#1-口令登录)
      - [2. 公钥登录 `ssh-copy-id user@host`](#2-公钥登录-ssh-copy-id-userhost)
      - [3. authorized\_keys文件](#3-authorized_keys文件)
    - [SSH远程操作](#ssh远程操作)
      - [远程执行命令](#远程执行命令)
      - [SSH数据传输](#ssh数据传输)
  - [Copying Files](#copying-files)
    - [`cp` `copy`](#cp-copy)
    - [`scp` 跨机远程拷贝](#scp-跨机远程拷贝)
    - [`rsync`](#rsync)
  - [`mv` Moving Files, Renaming Files](#mv-moving-files-renaming-files)
  - [compress and decompress](#compress-and-decompress)
    - [`zip`](#zip)
    - [`tar.gz`](#targz)
  - [Backing up](#backing-up)
    - [Databases `mysqldump`](#databases-mysqldump)
    - [Backing up Files and Folders](#backing-up-files-and-folders)
  - [Changing and Setting Owner Group SSH](#changing-and-setting-owner-group-ssh)
  - [Show System Processes with SSH](#show-system-processes-with-ssh)
  - [Get the last 30 lines from Plesk maillog](#get-the-last-30-lines-from-plesk-maillog)
    - [`sshfs` 挂载远程文件系统](#sshfs-挂载远程文件系统)
    - [`ssh -X` X11 图形界面](#ssh--x-x11-图形界面)
    - [`-D` 绑定本地端口](#-d-绑定本地端口)
    - [`-L` 本地端口转发 Local forwarding](#-l-本地端口转发-local-forwarding)
      - [`-R` 远程端口转发 remote forwarding](#-r-远程端口转发-remote-forwarding)
      - [SSH的其他参数](#ssh的其他参数)

---


# Network protocol - SSH

---


## basic


SSH is capable of
- shell access,
- file transfers,
- executing remote commands,
- port forwarding and tunneling,
- creating VPNs,
- forwarding x displays, that's our GUI,
- encrypted proxy browsing (via socks protocol),
- mounting remote directories.

Execute commands
`ssh user@hostname [command]`


Copying files:
- sftp – secure ftp
- `scp fileName user@host:/home/user/fileName`
- Based on sftp, but meant to replace rcp


Authentication Mechanisms
- Password
- Public-key
- Keyboard-interactive
- Multiple passwords required (SecurID, etc.)
- GSSAPI (Generic Security Services Application Program Interface)
- Extensible to provide for Kerberos, NTLM, etc.

**SSH and RDP: Comparison**
- SSH is considered more secure because it does not require additional tools such as a Virtual Private Network (VPN) or Multi-factor authentication (MFA) as RDP does.
- SSH is operated within the terminal, making it hard for non-IT personnel to conduct operations with SSH.




---

- SSH是一种网络协议，用于计算机之间的加密登录。
- 如果一个用户从本地计算机，使用SSH协议登录另一台远程计算机，就可以认为这种登录是安全的，即使被中途截获，密码也不会泄露。
- 最早的时候，互联网通信都是明文通信，一旦被截获，内容就暴露无疑。1995年，芬兰学者Tatu Ylonen设计了SSH协议，将登录信息全部加密，成为互联网安全的一个基本解决方案，迅速在全世界获得推广，目前已经成为Linux系统的标准配置。
- SSH只是一种协议，存在多种实现，既有商业实现，也有开源实现。
- 本文针对的实现是OpenSSH，它是自由软件，应用非常广泛。
- 这里只讨论SSH在Linux Shell中的用法。
- 在Windows系统中使用SSH，会用到另一种软件PuTTY，这需要另文介绍。

- SSH之所以能够保证安全，原因在于它采用了公钥加密。
- 整个过程是这样的：
  1. 远程主机收到用户的登录请求，远程主机把自己的公钥发给用户。
  2. 用户使用这个公钥，将登录密码加密后，发送回来。
  3. 远程主机用自己的私钥，解密登录密码，如果密码正确，就同意用户登录。
- 这个过程本身是安全的，但是实施的时候存在一个风险：*MITM*
  - 如果有人截获了登录请求，然后冒充远程主机，将伪造的公钥发给用户，那么用户很难辨别真伪。
  - 因为不像https协议，SSH协议的公钥是没有证书中心（CA）公证的，也就是说，都是自己签发的。
  - 设想攻击者插在用户与远程主机之间（比如在公共的wifi区域），用伪造的公钥，获取用户的登录密码。再用这个密码登录远程主机，那么SSH的安全机制就荡然无存了。这种风险就是（Man-in-the-middle attack）

---


## ssh的安装
- SSH分`客户端openssh-client`和`openssh-server`
  - 如果只想登陆别的机器的SSH, 只需要安装openssh-client
  - 如果要使本机开放SSH服务, 需要安装openssh-server
- Ubuntu已经安装了ssh client
  - 如果没有则`sudo apt-get install openssh-client`


---

## SSH Breakdown


![SSH-connection-2-1-1536x1536](https://i.imgur.com/IL4kiTQ.png)


1. The client host connects to the server hosts. During the exchange, both swap protocol versions they support.

2. The Server sends
   1. the authentication information and the session parameters to the Client;
   2. this includes the Server’s public key components of public and private key pairs
   3. and a list of encryptions, compressions,
   4. and authentications modes that the Server supports.

3. The client host checks the Server’s `public key` against the client key it holds in the public key library. After identifying the Server, the secret session key is generated.

4. Once the secret session key is both on the server and client host, the encryption and integrity check is enabled during this stage.

5. The client host and the user can now be authenticated to the server host without interception of the transmission between the two. After authentication, the appropriate access t the Server is granted to the user, and the two machines are connected under a secure and encrypted connection. Enabling a secure tunnel for transmissions through the internet gateway.




---

## 配置ssh

```bash
echo -e "\033[31;1m ******************************* \033[0m"
echo -e "\033[31;1m ************安装和配置ssh************ \033[0m"
sudo apt-get install -y openssh-server 1> /dev/null
sudo sed -i 's/UsePAM no/UsePAM yes/g' /etc/ssh/sshd_config
sudo sed -i '8a /etc/init.d/ssh start' /etc/profile
sudo /etc/init.d/ssh start
ps -e | grep ssh

echo -e "\033[31;1m ssh授权 \033[0m"
cd ~/.ssh/
ssh-keygen -t rsa
cat ./id_rsa.pub >> ./authorized_keys

$ ps -e|grep ssh
 2151 ?        00:00:00 ssh-agent

 5313 ?        00:00:00 sshd

ssh-agent表示ssh-client启动，sshd表示ssh-server启动了。

如果缺少sshd，说明ssh服务没有启动或者没有安装。

2.
如果你需要改变 SSH 的配置, 用文本编辑器打开 /etc/ssh/sshd_config 进行编辑.
比如改变登陆的标语, 在配置文件中找到下面这行:
#Banner none
删除 "#" 字符(取消该行的注释), 将 none 替换为包含你期望显示内容的文件地址.
修改后该行应该类似这样:
Banner /etc/issue
在配置文件 /etc/ssh/sshd_config 中你还可以找到端口号, 空闲超时时间等配置项. 配置项大都比较容易理解, 但是保险起见在你修改一些不是很确定的配置项时最好参考下 SSH 的帮助文档.
```

## 使用ssh在远程后台不中断地跑程序
- Linux关闭ssh（关闭终端等）后运行的程序或者服务自动停止，如python3 a.py &。
- 解决：使用`nohup`命令让程序在关闭窗口（切换SSH连接）的时候程序还能继续在后台运行。
- `nohup python3 a.py &`


## ssh 服务器端把空闲连接断开
- ssh连接中断: 服务器端把空闲连接给断开了。只能重新连接

1. 如果使用的是 `iTerm2`，要让ssh空闲不断线： `profiles -> sessions -> When idle, send ASCII code 97 every 60s`
  - 配置的ASCII code是97，所以是一串aaaaaa，一看效果，果然没断。
  - 但是接着用了用，就发现有坑了……
  - 开着vim，过了一段时间再回来时，这一长串aaaaaa……还要手工ESC，u一下，才恢复，而且指不定还会有其它副作用，实在蛋疼……

2. 通过客户端ssh参数配置
  - 其它命令行客户端，通过配置 ServerAliveInterval 来实现，在 `~/.ssh/config` 中加入： `ServerAliveInterval=30`。
  - `ServerAliveInterval=30`: 表示ssh客户端每隔30秒给远程主机发送一个`no-op`包, `no-op`是无任何操作的意思，这样远程主机就不会关闭这个SSH会话。
    - 只需要在当前用户的ssh连接调整就好（注意：是本地发起连接的客户端！并非修改所要连接的远程服务器端)
    - `vim ~/.ssh/config`，
    - 然后新增
    - `Host *
          ServerAliveInterval 60`
  - 60秒就好了，而且基本去连的机器都保持，所以配置了*
  - 如果有需要针对某个机器，可以自行配置为需要的serverHostName。

3. 单次连接
  - 若只是单次连接时需要，可使用-o参数实现：
  - `ssh -o ServerAliveInterval=30 user@host`

---

## system administration


### test if ssh is running

```bash
$ ps aux | grep ssh
root      3278  0.0  0.6  94892  6752 ?        Ss   Apr11   0:00 sshd: ubuntu [priv]
ubuntu    3341  0.0  0.3  94892  3968 ?        S    Apr11   0:00 sshd: ubuntu@pts/0
ubuntu    5795  0.0  0.1  12940  1016 pts/0    S+   00:05   0:00 grep --color=auto ssh
root     20273  0.0  0.5  65508  5944 ?        Ss   Apr11   0:00 /usr/sbin/sshd -D
```

---

### change the port

port 22 open will have a lot of attacks

solutions:
- editing the port, so it doesn't answer on port 22.
- change to something else.


```bash

1. check the port open on the host

// malicious users
$ nmap -T4 -F 3.22.168.152
Starting Nmap 7.80 ( https://nmap.org ) at 2020-04-11 20:29 EDT
Nmap scan report for ec2-3-22-168-152.us-east-2.compute.amazonaws.com (3.22.168.152)
Host is up (0.031s latency).
Not shown: 92 closed ports
PORT     STATE    SERVICE
22/tcp   open     ssh
25/tcp   filtered smtp
135/tcp  filtered msrpc
139/tcp  filtered netbios-ssn
445/tcp  filtered microsoft-ds
5900/tcp open     vnc
6000/tcp open     X11
6001/tcp open     X11:1
// even the os
sudo nmap -O 3.22.168.152
Aggressive OS guesses: Linux 3.11 - 4.1 (94%), Linux 3.2 - 3.8 (94%), Linux 2.6.32 (93%), Linux 2.6.32 - 2.6.39 (92%), Linux 3.13 (90%), Linux 4.4 (90%), Linux 2.6.32 or 3.10 (89%), Linux 3.0 or 3.5 (89%), Linux 4.2 (89%), Linux 3.8 (89%)
No exact OS matches for host (test conditions non-ideal).
Network Distance: 28 hops

// or kali
zenmap: target ip
command: nmap -T4 -F 3.22.168.152
```

2. change the file


```bash
# // command line: change the ssh port to 2237
$ semanage port -a -t ssh_port_t -p tcp, then the 2237.
$ sudo semanage port -l | grep ssh
ssh_port_t                     tcp      2237

# // restart the ssh
$ systemctl restart sshd.service
==== AUTHENTICATING FOR org.freedesktop.systemd1.manage-units ===
Authentication is required to restart 'ssh.service'.
Multiple identities can be used for authentication:
 1.  Ubuntu (ubuntu)
 2.  bob
 3.  lisa
Choose identity to authenticate as (1-3): 2
Password:
==== AUTHENTICATION COMPLETE ===

$ ps aux | grep ssh
root      3278  0.0  0.6  94892  6752 ?        Ss   Apr11   0:00 sshd: ubuntu [priv]
ubuntu    3341  0.0  0.3  94892  3968 ?        S    Apr11   0:00 sshd: ubuntu@pts/0
root      8173  0.0  0.5  65508  5492 ?        Ss   00:56   0:00 /usr/sbin/sshd -D
bob       8179  0.0  0.1  12940  1032 pts/0    S+   00:57   0:00 grep --color=auto ssh



// edit the file
$ vi  /etc/ssh/sshd_config

# Package generated configuration file
# See the sshd_config(5) manpage for details

# What ports, IPs and protocols we listen for
// here!!!!!!!!!!!
// change to other port
// port 2237
Port 22
# Use these options to restrict which interfaces/protocols sshd will bind to
#ListenAddress ::
#ListenAddress 0.0.0.0
Protocol 2
# HostKeys for protocol version 2
HostKey /etc/ssh/ssh_host_rsa_key
HostKey /etc/ssh/ssh_host_dsa_key
HostKey /etc/ssh/ssh_host_ecdsa_key
HostKey /etc/ssh/ssh_host_ed25519_key
#Privilege Separation is turned on for security
UsePrivilegeSeparation yes

# Lifetime and size of ephemeral version 1 server key
KeyRegenerationInterval 3600
ServerKeyBits 1024

# Logging
SyslogFacility AUTH
LogLevel INFO

# Authentication:
LoginGraceTime 120
PermitRootLogin prohibit-password
StrictModes yes

RSAAuthentication yes
PubkeyAuthentication yes
#AuthorizedKeysFile	%h/.ssh/authorized_keys

# Dont read the users ~/.rhosts and ~/.shosts files
IgnoreRhosts yes
# For this to work you will also need host keys in /etc/ssh_known_hosts
RhostsRSAAuthentication no
# similar for protocol version 2
HostbasedAuthentication no
# Uncomment if you dont trust ~/.ssh/known_hosts for RhostsRSAAuthentication
#IgnoreUserKnownHosts yes

# To enable empty passwords, change to yes (NOT RECOMMENDED)
PermitEmptyPasswords no

# Change to yes to enable challenge-response passwords (beware issues with some PAM modules and threads)
ChallengeResponseAuthentication no

# Change to no to disable tunnelled clear text passwords
PasswordAuthentication yes

# Kerberos options
#KerberosAuthentication no
#KerberosGetAFSToken no
#KerberosOrLocalPasswd yes
#KerberosTicketCleanup yes

# GSSAPI options
#GSSAPIAuthentication no
#GSSAPICleanupCredentials yes

X11Forwarding yes
X11DisplayOffset 10
PrintMotd no
PrintLastLog yes
TCPKeepAlive yes
#UseLogin no

#MaxStartups 10:30:60
#Banner /etc/issue.net

# Allow client to pass locale environment variables
AcceptEnv LANG LC_*

Subsystem sftp /usr/lib/openssh/sftp-server

# Set this to 'yes' to enable PAM authentication, account processing,
# and session processing. If this is enabled, PAM authentication will
# be allowed through the ChallengeResponseAuthentication and
# PasswordAuthentication.  Depending on your PAM configuration,
# PAM authentication via ChallengeResponseAuthentication may bypass
# the setting of "PermitRootLogin without-password".
# If you just want the PAM account and session checks to run without
# PAM authentication, then enable this but set PasswordAuthentication
# and ChallengeResponseAuthentication to 'no'.
UsePAM yes



```


---

## SSH基本用法

### Changing and Exploring Directories

SSH terminal can predict what you’re going to type by pressing tab.

```bash
example

to go to /var/www/vhosts/domainname.com/httpdocs

/v`TAB`w`TAB`v`TAB`d`TAB`h`TAB`d`TAB`

This displays as

/var/www/vhosts/domainname.com/httpdocs/

There were two folders the with similar names.

# httpdocs (the one I was after)
# httpsdocs (very similar but with an s in there)

As a result the Tab key autofilled ‘http’ but didn’t know which of the two directories I was autofilling. Pressing d then Tab again confirmed that I wanted httpdocs and not httpsdocs which would have required the letter ‘s’ instead as that’s the first letter after the autofilled item.
```


---


### SSH远程登录

```bash
sshkey.private Login
$ `ssh` `-i` `./sshkey.private` `username@localhost`
```


#### 1. 口令登录


```bash
# 以用户名user，登录远程主机host
$ ssh username@hostname   #ssh pika@192.168.0.111

# 如果本地用户名与远程用户名一致，登录时可以省略用户名。
$ ssh hostname

# 你的登录请求会送进远程主机的SSH默认端口22。用-p 修改端口。
$ ssh -p 2222 user@host   # ssh直接连接远程主机的2222端口

# 第一次登录对方主机，系统会出现下面的提示
$ ssh user@host
The authenticity of host 'host (12.18.429.21)' can't be established.
RSA key fingerprint is 98:2e:d7:e0:de:9f:ac:67:28:c2:42:2d:37:16:58:4d.
Are you sure you want to continue connecting (yes/no)?
# 无法确认host主机的真实性，只知道它的公钥指纹，继续连接？
# 公钥指纹 RSA key fingerprint:
# 指公钥长度较长（采用RSA算法，长达1024位），很难比对，
# 所以对其进行MD5计算变成一个128位的指纹。再进行比较就容易多了。
# 上例中是98:2e:d7:e0:de:9f:ac:67:28:c2:42:2d:37:16:58:4d，

# 问题是，用户怎么知道远程主机的公钥指纹应该是多少？
# 没有好办法，远程主机必须在自己的网站上贴出公钥指纹，以便用户自行核对。

- 假定经过风险衡量以后，用户决定接受这个远程主机的公钥。
    　　Are you sure you want to continue connecting (yes/no)? yes
- 系统会出现一句提示，表示host主机已经得到认可。
    　　Warning: Permanently added 'host,12.18.429.21' (RSA) to the list of known hosts.
- 然后，会要求输入密码。
    　　Password: (enter password)
- 如果密码正确，就可以登录了。
- 当远程主机的公钥被接受以后，它就会被保存在文件 $HOME/.ssh/known_hosts 之中。下次再连接这台主机，系统就会认出它的公钥已经保存在本地了，从而跳过警告部分，直接提示输入密码。
- 每个SSH用户都有自己的known_hosts文件，此外系统也有一个这样的文件，通常是 /etc/ssh/ssh_known_hosts ，保存一些对所有用户都可信赖的远程主机的公钥。

# check you user
$ whoami
ubuntu


```
---

#### 2. 公钥登录 `ssh-copy-id user@host`

```bash
- 使用密码登录，每次都必须输入密码，非常麻烦。
- 好在SSH还提供了公钥登录，省去输入密码的步骤。
- 公钥登录: 用户将自己的公钥储存在远程主机上。登录时，远程主机向用户发送一段随机字符串，用户用自己的私钥加密后，再发回来。远程主机用事先储存的公钥进行解密，如果成功证明用户可信，直接允许登录shell，不再要求密码。

- 寻找主机密钥
  - 在准备添加密钥之前不妨先用以下命令看看是否已经添加了对应主机的密钥了.

$ ssh-keygen -F 10.42.0.47
Host 10.42.0.47 found: line 1 type ECDSA
.............

- 这种方法要求用户必须提供自己的 公钥 。
- 没有现成可以直接用 ssh-keygen 生成一个：
$ ssh-keygen

$ ssh-keygen -t dsa
generating public/private dsa key pair.
enter file in which to save the key (/home/...../.ssh/id_dsa):
enter passphrase (empty for no passphrase):
enter same passphrase again:
your identification has been saved in /home/...../.ssh/id_dsa.
your public key has been saved in /home/...../.ssh/id_dsa.pub.
the key fingerprint is:
15:55:f5:...............:fd XXX@grave
the key's randomart image is:
.....

- 运行上面的命令以后，系统会出现一系列提示，可以一路回车。
- 其中有一个问题是，要不要对私钥设置口令（passphrase），如果担心私钥的安全，这里可以设置一个。
- 运行结束以后，在$HOME/.ssh/目录下，会新生成两个文件：
- id_rsa.pub 公钥 和 id_rsa 私钥。

- 这时再输入下面的命令，将公钥传送到远程主机host上面：
    　　$ ssh-copy-id user@host
- 好了，从此再登录就不需要输入密码了。

- 如果还是不行，就打开远程主机的 /etc/ssh/sshd_config 这个文件，检查下面几行前面"#"注释是否取掉。
    　　RSAAuthentication yes
    　　PubkeyAuthentication yes
    　　AuthorizedKeysFile .ssh/authorized_keys
- 然后，重启远程主机的ssh服务。
    　　// ubuntu系统
    　　service ssh restart
    　　// debian系统
    　　/etc/init.d/ssh restart


- 删除主机密钥
  - 某些情况下, 如主机地址更改或者不再使用某个密钥, 需要删除某个密钥.
$ ssh-keygen -R 10.42.0.47
# 比手动在 ~/.ssh/known_hosts 文件中删除要方便很多.


```

#### 3. authorized_keys文件
```bash
- 远程主机将用户的公钥，保存在登录后的用户主目录的 $HOME/.ssh/authorized_keys 文件中。公钥就是一段字符串，只要把它追加在authorized_keys文件的末尾就行了。
- 这里不使用上面的ssh-copy-id命令，改用下面的命令，解释公钥的保存过程：
    　　$ ssh user@host 'mkdir -p .ssh && cat >> .ssh/authorized_keys' < ~/.ssh/id_rsa.pub
- 分解开来看：
  1. "$ ssh user@host" : 登录远程主机；
  2. 'mkdir .ssh && cat >> .ssh/authorized_keys': 表示登录后在远程shell上执行的命令
    - '$ mkdir -p .ssh' 的作用是，如果用户主目录中的.ssh目录不存在，就创建一个；
    - 'cat >> .ssh/authorized_keys' < ~/.ssh/id_rsa.pub : 将本地的公钥文件 ~/.ssh/id_rsa.pub，重定向追加到远程文件authorized_keys的末尾。
- 写入authorized_keys文件后，公钥登录的设置就完成了。
```

### SSH远程操作

#### 远程执行命令
有时在远程主机执行一条命令并显示到本地, 然后继续本地工作是很方便的. SSH 就能满足这个需求:



```bash
$ `ssh username@host` `command for host PC`

$ ssh pi@10.42.0.47 ls -l
# 枚举远程主机的主目录内容并在本地显示.

$ ssh pi@10.42.0.47 date
pi@10.42.0.47's password:
Thu Aug 7 22:33:51 EAT 2014
```

#### SSH数据传输
- SSH不仅可以用于远程主机登录，还可直接在远程主机上执行操作。

`$ ssh user@host 'mkdir -p .ssh && cat >> .ssh/authorized_keys' < ~/.ssh/id_rsa.pub`

- 单引号, 在远程主机上执行的操作；`'mkdir -p .ssh && cat >> .ssh/authorized_keys'`
- 后面的输入重定向，表示数据通过SSH传向远程主机。`<` ~/.ssh/id_rsa.pub
- 分解开来看：
  1. `$ ssh user@host` : 登录远程主机；
  2. `'mkdir -p .ssh && cat >> .ssh/authorized_keys'`: 表示登录后在远程shell上执行的命令
    - `'$ mkdir -p .ssh'`: 如果用户主目录中的.ssh目录不存在，就创建一个；
    - `'cat >> .ssh/authorized_keys' < ~/.ssh/id_rsa.pub` : 将本地的公钥文件 `~/.ssh/id_rsa.pub`，重定向追加到远程文件`authorized_keys`的末尾。

- 这就是说，SSH可以在用户和远程主机之间，建立命令和数据的传输通道，很多事情都可以通过SSH来完成。

```bash
# 将 $HOME/src/ 目录下面的所有文件，复制到远程主机的 $HOME/src/ 目录。
$ cd && tar czv src | ssh user@host 'tar xz'

# 将远程主机 $HOME/src/ 目录下面的所有文件，复制到用户的当前目录。
$ ssh user@host 'tar cz src' | tar xzv

# 查看远程主机是否运行进程httpd。
$ ssh user@host 'ps ax | grep [h]ttpd'

# 建议使用scp进行远程copy：
```

---

## Copying Files

### `cp` `copy`


```bash

-a: copy the file or directory across retaining the permissions whilst retaining the permissions and ownership:

    cp -a contactus.php ../contact/index.php

copy file in the same dir:

    cp filename.txt new-file-name.txt

copy between directories

    cp filename.txt ../../new-directory/filename-to-copy.txt


copy all files from one directory to another, use *:

    cp images/* ../skin/

```

### `scp` 跨机远程拷贝

- `scp`是secure copy的简写，用于Linux下远程拷贝文件，类似的命令有`cp`
- 不过`cp`只是在本机进行拷贝不能跨服务器，而且`scp`传输是加密的。会稍微影响一下速度。
- 两台主机之间复制文件必需得同时有两台主机的 复制执行帐号 和 操作权限 。

can transfer and zip the file in the same time.



```bash
基础语法:
$ `scp` `source_file_path` `destination_file_path`

scp命令参数
-1 强制scp命令使用协议ssh1
-2 强制scp命令使用协议ssh2
-4 强制scp命令只使用IPv4寻址
-6 强制scp命令只使用IPv6寻址
-B 使用批处理模式（传输过程中不询问传输口令或短语）
-C 允许压缩。（将-C标志传递给ssh，从而打开压缩功能）
-p 留原文件的修改时间，访问时间和访问权限。
-q 不显示传输进度条。
-r 递归复制整个目录。 !!!!!!!!!!!
-v 详细方式显示输出。scp和ssh(1)会显示出整个过程的调试信息。这些信息用于调试连接，验证和配置问题。
-c cipher 以cipher将数据传输进行加密，这个选项将直接传递给ssh。
-F ssh_config 指定一个替代的ssh配置文件，此参数直接传递给ssh。
-i identity_file : 从指定文件中读取传输时使用的密钥文件，此参数直接传递给ssh。
-l limit 限定用户所能使用的带宽，以Kbit/s为单位。
-o ssh_option 如果习惯于使用ssh_config(5)中的参数传递方式，
-P port 注意是大写的P, port是指定数据传输用到的端口号
-S program 指定加密传输时所使用的程序。此程序必须能够理解ssh(1)的选项。
------------------------------------------------------------------------

# 本地复制远程文件
$ scp root@www.test.com:/val/test/test.tar.gz /val/test/test.tar.gz
# 本地复制远程文件到指定目录：（把远程的文件复制到本地）
$ scp root@www.test.com:/val/test/test.tar.gz /val/test/
# 远程复制本地文件
$ scp /val/test.tar.gz root@www.test.com:/val/test.tar.gz
# 远程复制本地文件到指定目录
$ scp /val/test.tar.gz root@www.test.com:/val/

ps: scp复制文件时只指定服务器地址不加路径默认复制到哪里???

# 在两个远程主机之间复制文件, 从一个远程主机复制到另一个远程主机。
$ scp user1@host1:/some/remote/dir/foobar.txt user2@host2:/some/remote/dir/

------------------------------------------------------------------------

-C 允许压缩。
    - 用压缩来加快传输, 节省时间和带宽！
    - 用C选项来启用压缩功能。文件在传输过程中被压缩，在目的主机上被解压缩。

# 开启压缩选项移动了整个文件夹。速度的增长取决于多少文件能被压缩。
$ scp -vrC ~/Downloads root@192.168.1.3:/root/Downloads

------------------------------------------------------------------------

-c cipher 以cipher将数据传输进行加密，这个选项将直接传递给ssh。
  - scp默认使用AES加密
  - 用不同的加密可能会加快转移过程，举例:blowfish和arcfour比AES更快（但是安全不如AES）

$ scp -c blowfish -C ~/local_file.txt username@remotehost:/remote/path/file.txt
# 用blowfish加密并同时压缩
# 可以得到显著的速度上的提升，当然也取决于可用的带宽。

------------------------------------------------------------------------

-i identity_file 特殊标识文件
  - 从指定文件中读取传输时使用的密钥文件，此参数直接传递给ssh。
  - 当使用基于 秘钥认证（无密码）。你将使用特殊的包含私有秘钥的标识文件。
  - 这个选项直接传递到ssh命令并且以同样的方式工作。

$ scp -vCq -i private_key.pem ~/test.txt root@192.168.1.3:/some/path/test.txt

------------------------------------------------------------------------

-l limit 限定用户所能使用的带宽
    - 限制带宽的使用, 不想scp占用所有的带宽
    - 用选项“l”来限制最大传输速度，以Kbit/s为单位

$ scp -vrC -l 400 ~/Downloads root@192.168.1.3:/root/Downloads

------------------------------------------------------------------------

-P port 在远程主机上连接一个不同的端口
  - 大写的P, port是指定数据传输用到的端口号
  - 如果 远程服务器 有ssh守护进程运行在不同的端口上（默认是22），那么需要使用“-P”选项来使用指定的端口。

$ scp -vC -P 2200 ~/test.txt root@192.168.1.3:/some/path/test.txt

------------------------------------------------------------------------

-p 小写 保存文件属性
  - 保存源文件的修改时间，访问时间以及方式。

$ scp -C -p ~/test.txt root@192.168.1.3:/some/path/test.txt

------------------------------------------------------------------------

-q 安静模式 不显示传输进度条
  - scp输出将会减少
  - 不再显示进度表以及警告和诊断信息。

$ scp -vCq ~/test.txt root@192.168.1.3:/some/path/test.txt

------------------------------------------------------------------------

-F ssh_config 使用不同的ssh_config文件
  - 指定一个替代的ssh配置文件，此参数直接传递给ssh。

$ scp -vC -F /home/user/my_ssh_config ~/test.txt root@192.168.1.3:/some/path/test.txt

------------------------------------------------------------------------

-r 递归复制整个目录

# -r: 把远程的目录 复制到 本地
$ scp -r root@www.test.com:/val/test/ /val/test/
# -r: 把本地的目录 复制到 远程主机上
$ scp -r ./ubuntu_env/ root@192.168.0.111:/home/pipi
pika:/media/pika/files/machine_learning/datasets
$ scp -r SocialNetworks/ piting@192.168.0.172:/media/data/pipi/datasets

# 从一个主机往另一个主机复制整个文件夹，需要使用r switch并且指定目录
1 $ scp -v -r ~/Downloads root@192.168.1.3:/root/Downloads
------------------------------------------------------------------------
详细方式显示输出
-v scp和ssh(1)会显示出整个过程的调试信息。这些信息用于调试连接，验证和配置问题。
  - SCP的程序将输出大量关于它在后台做什么的信息。
  - 当程序失败或无法完成请求时。详细的输出将正确的指明该程序哪里出了问题。

$ scp -v  ~/test.txt  root@192.168.1.3:/root/help2356.txt

    Executing: program /usr/bin/ssh host 192.168.1.3, user root, command scp -v -t /root/help2356.txt
    OpenSSH_6.2p2 Ubuntu-6ubuntu0.1, OpenSSL 1.0.1e 11 Feb 2013
    debug1: Reading configuration data /home/enlightened/.ssh/config
    debug1: Reading configuration data /etc/ssh/ssh_config
    debug1: /etc/ssh/ssh_config line 19: Applying options for *
    debug1: Connecting to 192.168.1.3 [192.168.1.3] port 22.
    debug1: Connection established.
    ..... OUTPUT TRUNCATED
------------------------------------------------------------------------
多文件传输
# 多个文件可以像下面那样用空格分隔开
$ scp foo.txt bar.txt username@remotehost:/path/directory/

# 从远程主机复制多个文件到当前目录
$ scp username@host:path\{afile,bfile}
$ scp username@remotehost:/path/directory/ \{foo.txt , bar.txt\} .
$ scp root@192.168.1.3:~/ \{abc.log , cde.txt\} .
------------------------------------------------------------------------
```

### `rsync`

```bash

copy all files including files that begin with . (1 dot) from one directory to another:

    rsync -a ./ ../


To show a progress bar of files being copied:

    rsync --progress /copy/from /copy/to

```

---

## `mv` Moving Files, Renaming Files

```bash

Moving Files

    mv currentpath/file.txt ../newpath

Renaming Files

    mv oldfilename.txt newfilename.txt
```

---

## compress and decompress

### `zip`

```bash

-r: ensures that the file and directories within the parent directory being compressed are also included.


Compressing Files with Zip

    zip -r compressedfile.zip path/directoryname

Decompressing Zip Files

    unzip filename.zip
```

### `tar.gz`

```bash

To create a tar.gz file:

    tar czvf archivename.tar.gz directoryorfile-to-archive/
    tar czvf website-backup-2010-11-31.tar.gz httpdocs/

the Flags czvf:
Compress - Creates the new archive.
Zip - Compresses the file.
File - Implies that we have given the compressed file a name.
Verbose - Prints what the command line is doing, like a progress report.

Decompressing Files with tar.gz

    tar -xzf archivename.tar.gz
    tar -xzf website-backup-2010-11-31.tar.gz

```


---

## Backing up


### Databases `mysqldump`

```bash

To backup a database via ssh:

    mysqldump -u database_username -p database_name > name_of_backup.sql
    mysqldump -u wordpress_bob -p wordpress_blog > wordpress_blog_20101031.sql


To restore and import a database
first create the bank database then assign a user.
Using these details you must replace the database name and user below:

    mysql -u database_username -p database_name < name_of_backup.sql
    mysql -u wordpress_bob -p wordpress_blog < wordpress_blog_2011-03-21.sql


```

### Backing up Files and Folders

To backup files, either use the compressing .tar.gz or .zip methods above.

---

## Changing and Setting Owner Group SSH

chown: change both the owner and group of a directory use the

```bash

chown Owner:Group directoryname/
chown 10000:505 directoryname/

```

---

## Show System Processes with SSH

To see what the system processes in a human readable way, type:

`ps aux --forest`

---

## Get the last 30 lines from Plesk maillog

`tail -n 30 -f /usr/local/psa/var/log/maillog`

---


### `sshfs` 挂载远程文件系统
基于 SSH 的工具: sshfs. 可以在本地直接挂载远程主机的文件系统.
$ `sshfs` -o idmap=user `user@hostname:/home/user` `~/Remotefloder`

```bash
# 将远程主机 pi 用户的主目录 挂载到 本地主目录下的 Pi 文件夹.
$ sshfs -o idmap=user pi@10.42.0.47:/home/pi ~/Pi
```

### `ssh -X` X11 图形界面
假如现在你想要在远程主机运行一个图形界面的程序
用 SSH -X 连接到远程主机即可开启 X11 转发功能.
登录后你可能觉得没什么差别, 但是当你运行一个图形界面程序后就会发现其中的不同的.

```bash
$ ssh -X pi@10.42.0.47
$ pistore
# 如果想在运行图形界面程序的同时做些别的事情
# 只需在命令末尾加一个 & 符号.
$ pistore&
```

---

### `-D` 绑定本地端口
既然SSH可以传送数据，那么可以让那些不加密的网络连接，全部走SSH连接，从而提高安全性。

```bash
# 让8080端口的数据，都通过SSH传向远程主机，命令就这样写：
$ ssh -D 8080 user@host

# SSH会建立一个socket，监听本地的8080端口。
# 一旦有数据传向那个端口，就自动把它转移到SSH连接上面，发往远程主机。
# 8080端口原来是一个不加密端口，现在将变成一个加密端口。
```

### `-L` 本地端口转发 Local forwarding
- 有时，绑定本地端口还不够，还必须指定数据传送的目标主机，从而形成点对点的"端口转发"。
- 为了区别后文的"远程端口转发"，"本地端口转发"（Local forwarding）。

```bash
假定host1本地主机，host2远程主机。种种原因这两台主机之间无法连通。
host3可以同时连通前面两台主机。因此，通过host3，将host1连上host2。

# 在host1执行下面的命令：
$ ssh -L 2121:host2:21 host3

L参数一共接受三个值 "本地端口:目标主机:目标主机端口"
指定SSH绑定本地端口2121，然后指定host3将所有数据，转发到目标主机host2的21端口（假定host2运行FTP端口21）
这样一来，只要连接host1的2121端口，就等于连上了host2的21端口。
$ ftp localhost:2121
"本地端口转发"使得host1和host3之间形成一个数据传输的秘密"SSH隧道"。


$ ssh -L 5900:localhost:5900 host3
# 将本机的5900端口绑定host3的5900端口
里的localhost指的是host3，因为目标主机是相对host3而言的


# 通过host3的端口转发，ssh登录host2。
$ ssh -L 9001:host2:22 host3
# 这时，只要ssh登录本机的9001端口，就相当于登录host2了。
$ ssh -p 9001 localhost

上面的-p参数表示指定登录端口。
出错处理：ssh: Could not resolve hostname 192.168.*.*:***: Name or service not known
解决：指定端口不能直接使用ip:端口号，使用-p参数来解决就可以了。

# 从某主机的80端口开启到本地主机2001端口的隧道
ssh -N -L 2001:localhost:80 somemachine
# 现在可以直接在浏览器中输入http://localhost:2001访问这个网站。

```

#### `-R` 远程端口转发 remote forwarding
"本地端口转发": 绑定本地端口的转发，
"远程端口转发": 绑定远程端口的转发。

```bash
host1与host2之间无法连通，必须借助host3转发。
但是，host3是一台内网机器，它可以连接外网的host1，但是反过来外网的host1连不上内网的host3。
这时，"本地端口转发"就不能用了

解决办法
host3可以连host1，那么就 "从host3上建立与host1的SSH连接" ，然后在host1上使用这条连接就可以了。
在host3执行下面的命令：
$ ssh -R 2121:host2:21 host1

R参数也是接受三个值，分别是"远程主机端口:目标主机:目标主机端口"。
让host1监听它自己的2121端口，然后将所有数据经由host3，转发到host2的21端口。
由于对于host3来说，host1是远程主机，所以这种情况就被称为"远程端口绑定"。

绑定之后，我们在host1就可以连接host2了：
$ ftp localhost:2121
这里必须指出，"远程端口转发"的前提条件是，host1和host3两台主机都有sshD和ssh客户端。
```

#### SSH的其他参数

```bash
1.
N参数，表示只连接远程主机，不打开远程shell；
T参数，表示不为这个连接分配TTY。
# 这两个参数放在一起用，代表这个SSH连接只用来传数据，不执行远程操作。
$ ssh -NT -D 8080 host

2. -f # 表示SSH连接成功后，转入后台运行。
# 这样一来，你就可以在不中断SSH连接的情况下，在本地shell中执行其他操作。
$ ssh -f -D 8080 host
# 要关闭这个后台连接，就只有用kill命令去杀掉进程。


3. 将你的麦克风输出到远程计算机的扬声器
# 这样来自你麦克风端口的声音将在SSH目标计算机的扬声器端口输出，但声音质量很差，会听到很多嘶嘶声。
dd if=/dev/dsp | ssh -c arcfour -C username@host dd of=/dev/dsp


4、cat fileA | diff fileB - # 比较远程和本地文件
$ ssh user@host cat /path/to/remotefile | diff /path/to/localfile –
在比较本地文件和远程文件是否有差异时这个命令很管用。

5、通过SSH挂载目录/文件系统
$ sshfs name@server:/path/to/folder /path/to/mount/point
从http://fuse.sourceforge.net/sshfs.html下载sshfs，它允许你跨网络安全挂载一个目录。

6、通过中间主机建立SSH连接
$ ssh -t reachable_host ssh unreachable_host
Unreachable_host表示从本地网络无法直接访问的主机，但可以从reachable_host所在网络访问，这个命令通过到reachable_host的“隐藏”连接，创建起到unreachable_host的连接。


8、直接连接到只能通过主机B连接的主机A
ssh -t hostA ssh hostB
当然，你要能访问主机A才行。

9、创建到目标主机的持久化连接
ssh -MNf <user>@<host>
在后台创建到目标主机的持久化连接，将这个命令和你~/.ssh/config中的配置结合使用：
Host host
ControlPath ~/.ssh/master-%r@%h:%p
ControlMaster no
所有到目标主机的SSH连接都将使用持久化SSH套接字，如果你使用SSH定期同步文件（使用rsync/sftp/cvs/svn），这个命令将非常有用，因为每次打开一个SSH连接时不会创建新的套接字。

10、通过SSH连接屏幕
ssh -t remote_host screen –r
直接连接到远程屏幕会话（节省了无用的父bash进程）。

11、端口检测（敲门）
knock <host> 3000 4000 5000 && ssh -p <port> user@host && knock <host> 5000 4000 3000
在一个端口上敲一下打开某个服务的端口（如SSH），再敲一下关闭该端口，需要先安装knockd，下面是一个配置文件示例。
[options]
logfile = /var/log/knockd.log
[openSSH]
sequence = 3000,4000,5000
seq_timeout = 5
command = /sbin/iptables -A INPUT -i eth0 -s %IP% -p tcp –dport 22 -j ACCEPT
tcpflags = syn
[closeSSH]
sequence = 5000,4000,3000
seq_timeout = 5
command = /sbin/iptables -D INPUT -i eth0 -s %IP% -p tcp –dport 22 -j ACCEPT
tcpflags = syn


12、删除文本文件中的一行内容，有用的修复
ssh-keygen -R <the_offending_host>
在这种情况下，最好使用专业的工具。

13、通过SSH运行复杂的远程shell命令
ssh host -l user $(<cmd.txt)
更具移植性的版本：
ssh host -l user “`cat cmd.txt`”

14、通过SSH将MySQL数据库复制到新服务器
mysqldump –add-drop-table –extended-insert –force –log-error=error.log -uUSER -pPASS OLD_DB_NAME | ssh -C user@newhost “mysql -uUSER -pPASS NEW_DB_NAME”
通过压缩的SSH隧道Dump一个MySQL数据库，将其作为输入传递给mysql命令，我认为这是迁移数据库到新服务器最快最好的方法。

15、删除文本文件中的一行，修复“SSH主机密钥更改”的警告
sed -i 8d ~/.ssh/known_hosts

16、从一台没有SSH-COPY-ID命令的主机将你的SSH公钥复制到服务器
cat ~/.ssh/id_rsa.pub | ssh user@machine “mkdir ~/.ssh; cat >> ~/.ssh/authorized_keys”
如果你使用Mac OS X或其它没有ssh-copy-id命令的*nix变种，这个命令可以将你的公钥复制到远程主机，因此你照样可以实现无密码SSH登录。

17、实时SSH网络吞吐量测试
yes | pv | ssh $host “cat > /dev/null”
通过SSH连接到主机，显示实时的传输速度，将所有传输数据指向/dev/null，需要先安装pv。
如果是Debian：apt-get install pv
如果是Fedora：yum install pv
（可能需要启用额外的软件仓库）。

18、如果建立一个可以重新连接的远程GNU screen
ssh -t user@some.domain.com /usr/bin/screen –xRR
人们总是喜欢在一个文本终端中打开许多shell，如果会话突然中断，或你按下了“Ctrl-a d”，远程主机上的shell不会受到丝毫影响，你可以重新连接，其它有用的screen命令有“Ctrl-a c”（打开新的shell）和“Ctrl-a a”（在shell之间来回切换），请访问http://aperiodic.net/screen/quick_reference阅读更多关于screen命令的快速参考。

19、继续SCP大文件
rsync –partial –progress –rsh=ssh $file_source $user@$host:$destination_file
它可以恢复失败的rsync命令，当你通过VPN传输大文件，如备份的数据库时这个命令非常有用，需要在两边的主机上安装rsync。
rsync –partial –progress –rsh=ssh $file_source $user@$host:$destination_file local -> remote
或
rsync –partial –progress –rsh=ssh $user@$host:$remote_file $destination_file remote -> local

20、通过SSH W/ WIRESHARK分析流量
ssh root@server.com ‘tshark -f “port !22″ -w -' | wireshark -k -i –'
使用tshark捕捉远程主机上的网络通信，通过SSH连接发送原始pcap数据，并在wireshark中显示，按下Ctrl+C将停止捕捉，但也会关闭wireshark窗口，可以传递一个“-c # ”参数给tshark，让它只捕捉“#”指定的数据包类型，或通过命名管道重定向数据，而不是直接通过SSH传输给wireshark，我建议你过滤数据包，以节约带宽，tshark可以使用tcpdump替代：
ssh root@example.com tcpdump -w – ‘port !22′ | wireshark -k -i –


21、保持SSH会话永久打开
autossh -M50000 -t server.example.com ‘screen -raAd mysession’
打开一个SSH会话后，让其保持永久打开，对于使用笔记本电脑的用户，如果需要在Wi-Fi热点之间切换，可以保证切换后不会丢失连接。


22、更稳定，更快，更强的SSH客户端
ssh -4 -C -c blowfish-cbc
强制使用IPv4，压缩数据流，使用Blowfish加密。


23、使用cstream控制带宽
tar -cj /backup | cstream -t 777k | ssh host ‘tar -xj -C /backup’
使用bzip压缩文件夹，然后以777k bit/s速率向远程主机传输。Cstream还有更多的功能，请访问http://www.cons.org/cracauer/cstream.html#usage了解详情，例如：
echo w00t, i’m 733+ | cstream -b1 -t2


24、一步将SSH公钥传输到另一台机器
ssh-keygen; ssh-copy-id user@host; ssh user@host
这个命令组合允许你无密码SSH登录，注意，如果在本地机器的~/.ssh目录下已经有一个SSH密钥对，ssh-keygen命令生成的新密钥可能会覆盖它们，ssh-copy-id将密钥复制到远程主机，并追加到远程账号的~/.ssh/authorized_keys文件中，使用SSH连接时，如果你没有使用密钥口令，调用ssh user@host后不久就会显示远程shell。


25、将标准输入（stdin）复制到你的X11缓冲区
ssh user@host cat /path/to/some/file | xclip
你是否使用scp将文件复制到工作用电脑上，以便复制其内容到电子邮件中？xclip可以帮到你，它可以将标准输入复制到X11缓冲区，你需要做的就是点击鼠标中键粘贴缓冲区中的内容。
```

总结
- 尽管SCP在安全地传输文件方面是非常有效的，它缺乏文件同步工具必要的功能。它所能做的就是复制粘贴文件从一个位置到另一个位置。
- 一个更强大的工具的 `Rsync` 它不仅具有SCP的所有功能，而且增加了更多的功能用来在2个主机智能同步文件。例如，它可以检查并上传只有修改过的文件，忽略现有的文件等等。

---



















---
