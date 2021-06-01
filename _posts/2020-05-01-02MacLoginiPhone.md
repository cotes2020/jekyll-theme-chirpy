---
layout: post
title: "02-Mac远程登录到iPhone"
date: 2020-05-01 15:50:00.000000000 +09:00
categories: [逆向工程]
tags: [逆向工程, Mac远程登录iPhone]
---

## 使用OpenSSH远程登录

+ 在Mac的终端上，通过一些命令行来完成一些操作。

+ iOS 和 Mac OS X都是基于Darwin(苹果的一个基于Unix的开源系统内核)，所以iOS中同样支持终端的命令行操作。

+ 在逆向工程中，我们经常会通过命令行来操作iPhone。通过Mac远程登录到iPhone的方式建立连接，Mac和iPhone必须是在同一局域网，比如同一wifi下。

  + `SSH`

    + Secure Shell的缩写，意思为“安全外壳协议”，是一种可以为远程登录提供安全保障的协议。
    + 使用SSH，可以把所有传输的数据进行加密，“中间人”攻击方式就不可能实现，能防止DNS欺骗和IP欺骗。

    + SSH协议一共2个版本。

      + SSH-1

      + SSH-2

        > 现在用的比较多的是SSH-2，客户端和服务端版本药保持一致才能通信。

    + 查看SSH版本(查看配置文件的Protocal字段)

      + 客户端: /etc/ssh/ssh_config

        ```
        $ ssh root@192.168.1.20
        $ root# cd /etc/ssh
        $ /etc/ssh root# ls -l // 查找出Protocol 2
        $ /etc/ssh root# cat ssh_config
        ```

      + 服务端: /etc/ssh/sshd_config

  + `openSSH`

    + 是SSH协议的免费开源实现，可以通过openSSH的方式让Mac远程登录到iPhone。

+ 使用openSSH远程登录到iPhone

  + 在iPhone上通过Cydia安装openSSH工具。
  + 在Mac的终端输入ssh 账户名@服务器主机地址，比如ssh root@192.168.1.3，OpenSSH的初始密码是alpine

  ```
  $ ssh root@192.168.3.20
  $ The authenticity of host '192.168.3.20 (192.168.3.20)'   can't be established.
  $ RSA key fingerprint is SHA256:k+VYRtMEEtLL3B6ME7Ud80f+osPbKmMuZewaNK97HYs.
  $ Are you sure you want to continue connecting (yes/no/[fingerprint])? yes
  $ cd /
  $ ls -l
  total 2
  drwxrwxr-x 90 root admin 2880 Apr  7 15:22 Applications/
  drwxrwxr-x  2 root admin   64 Aug  7  2020 Developer/
  drwxrwxr-x 26 root admin  832 Apr  8 01:19 Library/
  drwxr-xr-x  3 root wheel   96 Aug  7  2020 System/
  lrwxr-xr-x  1 root admin   11 Apr  7 14:35 User -> /var/mobile/
  drwxr-xr-x 61 root wheel 1952 Apr  7 14:33 bin/
  drwxr-xr-x  2 root wheel   64 Oct 28  2006 boot/
  drwxrwxr-t  2 root admin   64 Aug  7  2020 cores/
  dr-xr-xr-x  3 root wheel 1368 Apr  7 14:31 dev/
  lrwxr-xr-x  1 root wheel   11 Aug  7  2020 etc -> private/etc/
  drwxr-xr-x  2 root wheel   64 Oct 28  2006 lib/
  drwxr-xr-x  2 root wheel   64 Oct 28  2006 mnt/
  drwxr-xr-x  6 root wheel  192 Aug 30  2020 private/
  drwxr-xr-x 26 root wheel  832 Apr  7 14:33 sbin/
  lrwxr-xr-x  1 root wheel   15 Aug  7  2020 tmp -> private/var/tmp/
  drwxr-xr-x 12 root wheel  384 Apr  7 14:33 usr/
  lrwxr-xr-x  1 root admin   11 Aug  7  2020 var -> private/var/
  $ exit  // 退出root终端
  $ clear // 清掉记录
  ```

+ root、mobile

  + iOS下有2个常用的账户: root、mobile.
  + root是最高权限账户，$HOME是/var/root

  ```
  $ pwd  // 当前文件夹
  /Users/jovins
  $ ssh root@192.168.1.20
  $ root@192.168.3.20's password: -> alpine
  $ root# pwd
  /var/root
  ```

  + Mobile: 普通权限账户，只能操作一些普通文件，不能操作系统级别的文件，$HOME是/var/mobile
  + 登录mobile账户: ssh mobile@服务器主机地址，比如ssh mobile@192.168.1.20。

  ```
  ssh mobile@192.168.1.20
  ```

+ 修改root和mobile用户的登录密码。(登录root账户后，分别通过passwd、passwd mobile完成)

  ```
  $ ssh root@192.168.1.20
  $ root# passwd
  $ New password: 123456
  $ Retype new password: 123456
  // mobile
  $ root# passwd mobile
  $ New password: 123456
  $ Retype new password: 123456
  ```

+  `SSL`
  
  + Secure Sockets Layer的缩写，是为网络通信提供安全及数据完整性的一种安全协议，在传输层对网络连接进行加密。
+ `OpenSSL`
  
  + SSL的开源实现，绝大部分HTTPS请求等价于: HTTP + OpenSSL
+ `SSH`
  
  + Secure Shell的缩写，意思为“安全外壳协议”，是一种可以为远程登录提供安全保障的协议。
+ `OpenSSH`
  
  + OpenSSH的加密就是通过OpenSSL完成的。

## 建立安全连接

+ 在建立安全连接过程中，服务器会提供自己的身份证明。
  + known_hosts : 主要保存服务器端发送的公钥

```
Mac													 iPhone
客户端					发送公钥给客户端	服务器端
/.ssh/known_hosts  <--       公钥/etc/ssh/ssh_host_rsa_key.pub 														  私钥/etc/ssh/ssh_host_rsa_key
```

## SSH的客户端认证方式

+ SSH - 2提供了2种常用的客户端认证方式。

  + 基于密码的客户端认证。

    > 使用账号和密码即可认证，之前输入密码登录root的就是这种方式。

  + 基于密钥的客户端认证

    > 免密码认证，也是最安全的一种认证方式。公钥、私钥

+ SSH-2 默认会优先尝试 "密钥认证"，如果认证失败，才会尝试"密码认证"。

+ SSH - 基于密钥的客户端认证方式

```
客户端														       服务器端
公钥~/.ssh/id_rsa.pub			--->		      ~/.ssh/authorized_keys
私钥~/.ssh/id_rsa
```

> 将公钥内容追加到授权文件尾部进行登录认证

```
$ ssh-keygen // 生成公钥，默认rsa算法，回车即可
$ cd ~/.ssh
$ ls -l 	   // 查看.ssh文件下所有文件
$ ssh-copy-id root@192.168.1.20 // 这样就可以将公钥追加到服务器端authorized_keys，这样登录root账号就可以不用密码了。
```

```
jovins@JovinsdeMacBook-Pro ~ % cd ~/.ssh
jovins@JovinsdeMacBook-Pro .ssh % ls -l
total 24
-rw-------@ 1 jovins  staff  2610  3 17 00:51 id_rsa
-rw-r--r--@ 1 jovins  staff   575  3 17 00:51 id_rsa.pub
-rw-r--r--  1 jovins  staff  1938  4  9 16:05 known_hosts
jovins@JovinsdeMacBook-Pro .ssh % ssh-copy-id root@192.168.3.20
/usr/bin/ssh-copy-id: INFO: Source of key(s) to be installed: "/Users/jovins/.ssh/id_rsa.pub"
/usr/bin/ssh-copy-id: INFO: attempting to log in with the new key(s), to filter out any that are already installed
/usr/bin/ssh-copy-id: INFO: 1 key(s) remain to be installed -- if you are prompted now it is to install the new keys
root@192.168.3.20's password:

Number of key(s) added:        1

Now try logging into the machine, with:   "ssh 'root@192.168.3.20'"
and check to make sure that only the key(s) you wanted were added.

jovins@JovinsdeMacBook-Pro .ssh % ssh root@192.168.3.20
jovinteki-iPhone:~ root#
```

+ 初始密码是: `alpine`
+ 公钥 >> 授权文件
  + 可以使用ssh-copy-id将客户端的公钥内容自动追加到服务器的授权文件尾部，也可以手动操作。
  + 复制客户端的公钥到服务器某路径。
    + scp ~/.ssh/id_rsa.pub root@服务器主机地址:~/.ssh
    + scp是Secure copy的缩写，是基于SSH登录进行安全的远程文件拷贝命令，把一个文件copy到远程另外一台主机上
    + 上面的命令行将客户端的~/.ssh/id_rsa.pub拷贝到服务器的~地址
  + SSH登录服务器
    + ssh root@服务器主机地址
  + 在服务器创建.ssh文件夹
    + mkdir .ssh
  + 追加公钥内容到授权文件尾部
    + cat ~/id_rsa.pub  >> ~/.ssh/authorized_keys
    + $ ~/.ssh root# cat ~/id_rsa.pub  >> authorized_keys
  + 删除公钥
    + rm ~/id_rsa.pub
+ 文件权限问题
  + 如果配置了免密码登录，还是输入密码，需要在服务器端设置文件权限。
    + chmod 755 ~
    + chmod 755 ~/.ssh
    + chmod 644 ~/.ssh/authorized_keys

## 22端口

+ 端口就是设备对外提供服务的窗口，每个端口都有个端口号(范围是0~65535)
+ 有些端口是保留的，已经规定了用短的，比如
  + 21端口提供FTP服务
  + 80端口提供HTTP服务
  + 22端口提供SSH服务(可以查看/etc/ssh/sshd_config的Port字段)
+ iPhone默认是使用22端口进行SSH通信，采用的是TCP协议。

```
Mac    ---  wifi ---> 22 iPhone
           ssh登录
```

## USB进行SSH登录

+ 默认情况下，由于SSH走的是TCP协议，Mac是通过网络连接的方式SSH登录到iPhone，要求iPhone连接WiFi。
+ 为了加快传输速度，也可以通过USB连接的方式进行SSH登录。
  + Mac上有个服务程序usbmuxd(开机自动启动)，可以将Mac的数据通过USB传输到iPhone。
  + /System/Library/PrivateFrameworks/MobileDevice/framework/Resources/usbmuxd

```
 Mac                 usbmuxd                iPhone
SSH登录  10010        USB                    22
```

+ usbmuxd的使用

  + 下载usbmuxd工具包(下载v1.0.8版本，主要用到里面的一个python脚本：tcprelay.py)

    + https://github.com/libimobiledevice/usbmuxd/tags

  + 将iPhone的22端口(SSH端口)映射到哦Mac本地的10010端口

    + cd ~/jovins/usbmuxd-1.0.8/python-client

    + python tcprelay.py -t 22:10010

      > 加上-t参数是为了能够同时支持多个SSH连接。
      >
      > 注意：要想保持端口映射状态，不能终止次命令行(如果要执行其他终端命令行，请重新打开一个终端界面)

    + Control  + C : 关闭连接
    + 不一定非要10010端口，只要不是保留的端口就行。

  + 端口映射完毕后，以后如果想跟iPhone的22端口通信，直接跟Mac本地的10010端口通信就可以了，

    + 打开新的终端界面，SSH登录到Mac本地的10010端口(二选一)

      + ssh root@localhost -p 10010

      + ssh root@127.0.0.1 -p 10010

        > 注意: 这里SSH登录如果没有设置免密码登录时，需要输入密码，默认alpine。

      + localhost是一个域名，指向的IP地址是127.0.0.1，本机虚拟网卡的IP地址。

      + usbmuxd会将Mac本地10010端口的TCP协议数据，通过USB连接转发到iPhone的22端口。

  + 远程拷贝文件也可以直接跟Mac本地的10010端口通信

    + 将Mac上的~/Desktop/1.txt文件，拷贝到iPhone上的~/test路径

      ```
      $ scp -P 10010 ~/Desktop/1.txt root@localhost:~/test
      $ scp -P 10010 /Users/JVTools root@localhost:/usr/bin
    ```
  
    > 注意: scp的端口号参数是大写-P
  
  + 将手机上/System/Library/Caches/com.apple.dyld/所有文件拷贝到电脑桌面
  
  ```
    // scp -r root@手机ip iPhone路径 mac路径
  MacBook-Pro ~ % scp -r root@192.168.3.20:/System/Library/Caches/com.apple.dyld/ /Users/jovins/Desktop
  ```

  + sh脚本文件
  
    + 我们可以将经常执行的一些列终端命令行放到sh脚本文件中(shell)执行脚本文件。
  
    ```
    // 连接10010端口
    // 复制需要执行的终端命令行
    // 如: python /Users/jovins/Jovins/逆向工程/usbmuxd-1.0.8/python-client/tcprelay.py -t 22:10010
    vim usb.sh  // 创建脚本文件
    // 按 i ，然后粘贴已复制的内容进去，按esc -> :wq
    // 这样就创建好执行脚本文件了
    // 执行
    $ sh usb.sh  
    $ bash usb.sh
    $ source usb.sh
    $ . usb.sh
    
    // 登录
    // 复制 ssh root@localhost -p 10010
    $ vim login.sh
    // 粘贴到login.sh中
    $ sh login.sh
    ```
  
    + 可以通过sh、bash、source命令来执行sh脚本文件。
    + sh、bash
      + 当前shell环境会启动一个子进程来执行脚本文件，执行后返回到父进程的shell环境。
      + 执行cd时，在子进程中会进入到cd的目录，但是在父进程中环境并没有改变，也就是说目录没有改变。
  + source
      + 在当前的shell环境下执行脚本文件。
    + 执行cd后会跳转到cd的目录。
      + source可以用一个点"."来代替，比如 ". test.sh"

  + iOS终端的中文乱码问题 

    + 默认情况下，iOS终端不支持中文输入和显示。
  
    + 解决方案: 新建一个~/.inputrc文件，文件内容是
  
    ```
    set convert-meta off  // 不将中文字符转化为转义序列
    set output-meta on 		// 允许想终端输入中文
    set meta-flag on			// 允许向终端输入中文
    set input-meta on 		
    ```
  
    + 如果是想在终端编辑文件内容，可以通过Cydia安装一个VI IMproved