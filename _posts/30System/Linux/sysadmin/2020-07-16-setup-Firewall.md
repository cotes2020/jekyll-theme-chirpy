---
title: Linux - Setup Firewall
date: 2020-07-16 11:11:11 -0400
categories: [30System, Sysadmin]
tags: [Linux, Sysadmin, Setup, Firewall]
math: true
image:
---


# set up firewall

[toc]

---

The two most common software firewalls
- `UFW`: Uncomplicated Firewall.
  - to management the `iptable` in a easier way.
  - LInux原始防火墙工具iptables过于繁琐
    - ubuntu默认提供了基于iptable的防火墙工具ufw。
    - 支持图形界面操作，只需在命令行运行ufw命令即能看到一系列的操作。
- `firewalld`

---

# UFW

ref

https://help.ubuntu.com/community/UFW


UFW
- not only add services and ports, but it looks at IP addresses as well.
- UFW still manages `iptables`.
    - `IPFilter`: packet-filtering software that can be configured for a variety of different platforms.
    - `iptables`: stateful firewall ruleset into Linux systems.


`ufw`相关的文件：

- `/etc/ufw/`：一些ufw的环境设定文件
    - 如 before.rules、after.rules、sysctl.conf、ufw.conf，及 for ip6 的 before6.rule 及 after6.rules。
    - 这些文件一般按照默认的设置进行就ok。

- `/etc/ufw/sysctl.confm`
    - 开启ufw后，`/etc/ufw/sysctl.confm` 会覆盖默认的`/etc/sysctl.conf`文件
    - 若原来的`/etc/sysctl.conf`做了修改，启动ufw后，若`/etc/ufw/sysctl.conf`中有新赋值，则会覆盖`/etc/sysctl.conf`的，否则还以`/etc/sysctl.conf`为准。
    - 可以修改`/etc/default/ufw`中的`“IPT_SYSCTL=”`条目来设置使用哪个 `sysctrl.conf`.

- `/var/lib/ufw/user.rules` :设置的一些防火墙规则
    - 可以直接修改这个文件，不用使用命令来设定。
    - 修改后记得 ufw reload 重启使得新规则生效。


```c

安装

　　$ sudo apt-get install ufw


查看已经定义的ufw规则

    $ sudo ufw status
    Status: inactive
    Status: active


打开/关闭ufw Enable and Disable

    $ sudo ufw enable/disable

    $ sudo ufw enable
    Command may disrupt existing ssh connections. Proceed with operation (y|n) y
    Firewall is active and enabled on system startup
```

**Allow and Deny (specific rules)**

Allow

sudo ufw allow <port>/<optional: protocol>

Deny

sudo ufw deny <port>/<optional: protocol>

Delete Existing Rule

sudo ufw delete deny 80/tcp



```c
外来访问默认允许/拒绝

    $ sudo ufw default allow/deny


允许/拒绝 访问20端口，20后可跟/tcp或/udp，表示tcp或udp封包。

    $ sudo ufw allow/deny 20
    $ sudo ufw allow/deny ［service］


转换日志状态

　　$ sudo ufw logging on|off


设置默认策略 （比如 “mostly open” vs “mostly closed”）

　　$ sudo ufw default allow|deny

```

打开或关闭某个端口

```c

　　sudo ufw allow smtp　
　　sudo ufw allow 25/tcp
允许所有的外部IP访问本机的25/tcp （smtp）端口


　　sudo ufw allow from 192.168.1.100 to any port 25
允许此IP访问所有的本机端口

    ufw allow proto tcp from 10.0.1.0/10 to 本机ip port 25
允许自10.0.1.0/10的tcp封包访问本机的25端口。

　　sudo ufw allow proto udp 192.168.0.1 port 53 to 192.168.0.2 port 53

　　sudo ufw deny smtp
禁止外部访问smtp服务

　　sudo ufw delete allow smtp
删除上面建立的某条规则


ufw allow/deny servicename:ufw从/etc/services中找到对应service的端口，进行过滤。


ufw delete allow/deny 20
删除以前定义的“允许/拒绝访问20端口”的规则

```


## show rules

```c

look at some of the rules, check the status of UFW:

    $ sudo ufw status verbose
    //
    Status: active
    Logging: on (low)
    Default: deny (incoming), allow (outgoing), disabled (routed)
    New profiles: skip


    $ sudo ufw status numbered
    // rules created by ufw
    Status: active
         To                         Action      From
         --                         ------      ----
    [ 1] 53/udp                     ALLOW IN    Anywhere
    [ 2] 22                         ALLOW IN    Anywhere
    [ 3] 53/udp (v6)                ALLOW IN    Anywhere (v6)
    [ 4] 22 (v6)                    ALLOW IN    Anywhere (v6)


    // rules created actually alots
    $ sudo ufw show raw
    $ sudo ufw show raw | grep 22
         722    55582 ACCEPT     all  --  *      *       0.0.0.0/0            0.0.0.0/0            ctstate RELATED,ESTABLISHED
           0        0 ACCEPT     udp  --  *      *       0.0.0.0/0            224.0.0.251          udp dpt:5353
           0        0 ACCEPT     tcp  --  *      *       0.0.0.0/0            0.0.0.0/0            tcp dpt:22
           0        0 ACCEPT     udp  --  *      *       0.0.0.0/0            0.0.0.0/0            udp dpt:22
    Chain PREROUTING (policy ACCEPT 622 packets, 43205 bytes)
    Chain INPUT (policy ACCEPT 622 packets, 43205 bytes)
    Chain PREROUTING (policy ACCEPT 622 packets, 43205 bytes)
           0        0 ACCEPT     tcp      *      *       ::/0                 ::/0                 tcp dpt:22
           0        0 ACCEPT     udp      *      *       ::/0                 ::/0                 udp dpt:22


    $ sudo ufw show raw | grep 22
         229    10988 ufw-after-input  all  --  *      *       0.0.0.0/0            0.0.0.0/0
         224    10736 ufw-after-logging-input  all  --  *      *       0.0.0.0/0            0.0.0.0/0
         224    10736 ufw-reject-input  all  --  *      *       0.0.0.0/0            0.0.0.0/0
         224    10736 ufw-track-input  all  --  *      *       0.0.0.0/0            0.0.0.0/0
         229    10988 ufw-not-local  all  --  *      *       0.0.0.0/0            0.0.0.0/0
           0        0 ACCEPT     udp  --  *      *       0.0.0.0/0            224.0.0.251          udp dpt:5353
         229    10988 ufw-user-input  all  --  *      *       0.0.0.0/0            0.0.0.0/0
         229    10988 RETURN     all  --  *      *       0.0.0.0/0            0.0.0.0/0            ADDRTYPE match dst-type LOCAL
          22     2164 ACCEPT     udp  --  *      *       0.0.0.0/0            0.0.0.0/0            ctstate NEW
           0        0 ACCEPT     tcp  --  *      *       0.0.0.0/0            0.0.0.0/0            tcp dpt:22
           0        0 ACCEPT     udp  --  *      *       0.0.0.0/0            0.0.0.0/0            udp dpt:22
           0        0 ACCEPT     tcp      *      *       ::/0                 ::/0                 tcp dpt:22
           0        0 ACCEPT     udp      *      *       ::/0                 ::/0                 udp dpt:22






```

look at the rules that it already has created.

```c

$ sudo ufw show raw

// what iptables actully manually configured
IPV4 (raw):

Chain INPUT (policy DROP 0 packets, 0 bytes)
    pkts      bytes target     prot opt in     out     source               destination
     314    22701 ufw-before-logging-input  all  --  *      *       0.0.0.0/0            0.0.0.0/0
     314    22701 ufw-before-input  all  --  *      *       0.0.0.0/0            0.0.0.0/0
     140     7041 ufw-after-input  all  --  *      *       0.0.0.0/0            0.0.0.0/0
     136     6841 ufw-after-logging-input  all  --  *      *       0.0.0.0/0            0.0.0.0/0
     136     6841 ufw-reject-input  all  --  *      *       0.0.0.0/0            0.0.0.0/0
     136     6841 ufw-track-input  all  --  *      *       0.0.0.0/0            0.0.0.0/0

Chain FORWARD (policy DROP 0 packets, 0 bytes)
    pkts      bytes target     prot opt in     out     source               destination
       0        0 ufw-before-logging-forward  all  --  *      *       0.0.0.0/0            0.0.0.0/0
       0        0 ufw-before-forward  all  --  *      *       0.0.0.0/0            0.0.0.0/0
       0        0 ufw-after-forward  all  --  *      *       0.0.0.0/0            0.0.0.0/0
       0        0 ufw-after-logging-forward  all  --  *      *       0.0.0.0/0            0.0.0.0/0
       0        0 ufw-reject-forward  all  --  *      *       0.0.0.0/0            0.0.0.0/0
       0        0 ufw-track-forward  all  --  *      *       0.0.0.0/0            0.0.0.0/0

Chain OUTPUT (policy ACCEPT 0 packets, 0 bytes)
    pkts      bytes target     prot opt in     out     source               destination
     132    13843 ufw-before-logging-output  all  --  *      *       0.0.0.0/0            0.0.0.0/0
     132    13843 ufw-before-output  all  --  *      *       0.0.0.0/0            0.0.0.0/0
      12     1156 ufw-after-output  all  --  *      *       0.0.0.0/0            0.0.0.0/0
      12     1156 ufw-after-logging-output  all  --  *      *       0.0.0.0/0            0.0.0.0/0
      12     1156 ufw-reject-output  all  --  *      *       0.0.0.0/0            0.0.0.0/0
      12     1156 ufw-track-output  all  --  *      *       0.0.0.0/0            0.0.0.0/0

Chain ufw-after-forward (1 references)
    pkts      bytes target     prot opt in     out     source               destination

Chain ufw-after-input (1 references)
    pkts      bytes target     prot opt in     out     source               destination
       0        0 ufw-skip-to-policy-input  udp  --  *      *       0.0.0.0/0            0.0.0.0/0            udp dpt:137
       0        0 ufw-skip-to-policy-input  udp  --  *      *       0.0.0.0/0            0.0.0.0/0            udp dpt:138
       0        0 ufw-skip-to-policy-input  tcp  --  *      *       0.0.0.0/0            0.0.0.0/0            tcp dpt:139
       4      200 ufw-skip-to-policy-input  tcp  --  *      *       0.0.0.0/0            0.0.0.0/0            tcp dpt:445
       0        0 ufw-skip-to-policy-input  udp  --  *      *       0.0.0.0/0            0.0.0.0/0            udp dpt:67
       0        0 ufw-skip-to-policy-input  udp  --  *      *       0.0.0.0/0            0.0.0.0/0            udp dpt:68
       0        0 ufw-skip-to-policy-input  all  --  *      *       0.0.0.0/0            0.0.0.0/0            ADDRTYPE match dst-type BROADCAST

Chain ufw-after-logging-forward (1 references)
    pkts      bytes target     prot opt in     out     source               destination
       0        0 LOG        all  --  *      *       0.0.0.0/0            0.0.0.0/0            limit: avg 3/min burst 10 LOG flags 0 level 4 prefix "[UFW BLOCK] "

Chain ufw-after-logging-input (1 references)
    pkts      bytes target     prot opt in     out     source               destination
      49     2294 LOG        all  --  *      *       0.0.0.0/0            0.0.0.0/0            limit: avg 3/min burst 10 LOG flags 0 level 4 prefix "[UFW BLOCK] "

Chain ufw-after-logging-output (1 references)
    pkts      bytes target     prot opt in     out     source               destination

Chain ufw-after-output (1 references)
    pkts      bytes target     prot opt in     out     source               destination

Chain ufw-before-forward (1 references)
    pkts      bytes target     prot opt in     out     source               destination
       0        0 ACCEPT     all  --  *      *       0.0.0.0/0            0.0.0.0/0            ctstate RELATED,ESTABLISHED
       0        0 ACCEPT     icmp --  *      *       0.0.0.0/0            0.0.0.0/0            icmptype 3
       0        0 ACCEPT     icmp --  *      *       0.0.0.0/0            0.0.0.0/0            icmptype 4
       0        0 ACCEPT     icmp --  *      *       0.0.0.0/0            0.0.0.0/0            icmptype 11
       0        0 ACCEPT     icmp --  *      *       0.0.0.0/0            0.0.0.0/0            icmptype 12
       0        0 ACCEPT     icmp --  *      *       0.0.0.0/0            0.0.0.0/0            icmptype 8
       0        0 ufw-user-forward  all  --  *      *       0.0.0.0/0            0.0.0.0/0


```

---

# iptable

## Troubleshooting

```c
1. check connection
ss -lntp  // see port listen
ss -lntp | grep :80


2. iptable status
systemctl status iptables



2. check ip table
iptable -vnL  // check Reject, check order

iptables -I INPUT -p tcp -s 10.0.1.11 --dport 80 -j ACCEPT
service iptables save


vim /etc/systemconf/iptables  // modify the iptable rule
```

---

# firewalld

ref

https://firewalld.org/documentation/utilities/firewall-cmd.html


```c

1. look at firewalld.

$ firewall-cmd --state
running


2. zones
- similar to chains inside iptables.
- The only one that is actually populated is public.

$ firewall-cmd --get -zones.
block dmz drop external home internal public truseted work

// Create new service
firewall-cmd --permanent --new-zone=api
firewall-cmd --permanent --zone=api --add-service=http
firewall-cmd --permanent --zone=api --add-source=10.0.1.11


3. services
$ firewall-cmd  --get-services
// add service
$ firewall-cmd  --zone=public --add-service=http
// Create a new service
firewall-cmd --permanent --new-service=jobsub
firewall-cmd --permanent --service=jobsub --set-description="Job Submission"
firewall-cmd --permanent --service=jobsub --add-port=5671-5677/tcp
firewall-cmd --permanent --add-service=jobsub


4. check rule
firewall-cmd --list-all
// add rule
firewall-cmd --permanent --add-rich-rule='rule family=ipv4 source address=10.0.1.10/24 port port=80 protocol=tcp reject'
firewall-cmd --permanent --add-rich-rule='rule family=ipv4 source address=10.0.1.0/24 port port=8080 protocol=tcp accept'


6. IPSet
// Create IPSet
firewall-cmd --permanent --new-ipset=kiosk --type=hash:ip
firewall-cmd --permanent --ipset=kiosk --add-entry=10.0.1.12
firewall-cmd --permanent --ipset=kiosk --add-entry=192.168.1.0/24
firewall-cmd --permanent --zone=drop --add-source=ipset:kiosk

```

Now within Red Hat-type systems, I can still enable iptables if I want to, if I wanted to get really, really granular with trying to put a firewall into my system. However, with the new firewalld, there is no need; and with firewalld, we're less prone to make mistakes. Firewalls within Linux should always be used to stop services and decrease our attack surface.


## Troubleshooting


```c
1. check connection
ss -lntp  // see port listen


2. check ip table
firewall-cmd --list-all

firewall-cmd --permanent --add-rich-rule='rule family=ipv4 source address=10.0.1.10/24 port port=80 protocol=tcp reject'  // block ip
firewall-cmd --permanent --add-rich-rule='rule family=ipv4 source address=10.0.1.11/24 port port=80 protocol=tcp accept'  // wont work, reject over all.
firewall-cmd -reload



```





.
