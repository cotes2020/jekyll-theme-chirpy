---
layout: post
title: "How to setup loadbalancing with haproxy & keepalived"
date: 2023-07-18 21:19:00 +0800
categories: loadbalancer
tags: haproxy
image:
  path: /assets/img/headers/HAProxy.png
---

### Prerequisites:
- 2 Ubuntu20.04 LoadBalancer node's
- 3 Ubuntu20.04 Kubernetes master node's
- 2 Ubuntu20.04 Kubernetes worker node's

### HAProxy Configurations:

SSH to the nodes which will function as the load balancer and execute the following commands to install HAProxy.
```sh
apt update && apt install -y haproxy
```

*Edit `haproxy.cfg` to connect it to the master nodes, set the correct values for `<loadbalancer-vip>` and `<kube-masterX-ip>` and add an extra entry for each additional master:*
```sh
vim /etc/haproxy/haproxy.cfg
```

```sh
global
    log /dev/log local0
    log /dev/log local1 notice
    daemon
    maxconn 2000
    user haproxy
    group haproxy

defaults
    log     global
    mode    tcp
    option  tcplog
    option  dontlognull
    retries 3
    timeout connect 10s
    timeout client 1m
    timeout server 1m

listen stats
    bind *:8404
    mode http
    stats enable
    stats uri /stats
    stats refresh 10s
    stats show-node
    stats auth admin:adminpass

frontend kubernetes-api
    bind *:6443
    default_backend kubernetes-masters

backend kubernetes-masters
    balance roundrobin
    option tcp-check
	server k8s-master-0 <kube-masterX-ip>:6443 check
	server k8s-master-1 <kube-masterY-ip>:6443 check
    server k8s-master-2 <kube-masterZ-ip>:6443 check
```
*Verify haproxy configuration & restart HAproxy:*

```sh
haproxy -f /etc/haproxy/haproxy.cfg -c
```

```sh
{
systemctl daemon-reload
sudo systemctl enable haproxy 
sudo systemctl start haproxy
sudo systemctl status haproxy
}
```
### Set up high availability with Keepalived

*On both the nodes[master & backup], run the following commands:*

```sh
apt update && apt install -y keepalived && apt install -y libipset13
```
### Keepalived Configurations:

**On Master/Primary node:**

```sh
vim /etc/keepalived/keepalived.conf
```

```sh
# Define the script used to check if haproxy is still working
vrrp_script chk_haproxy {
    script "/usr/bin/killall -0 haproxy"
    interval 2
    weight 2
}

# Configuration for Virtual Interface
vrrp_instance LB_VIP {
    interface ens6
    state MASTER        # set to BACKUP on the peer machine
    priority 301        # set to  300 on the peer machine
    virtual_router_id 51

    authentication {
        auth_type user
        auth_pass UGFzcwo=  # Password for accessing vrrpd. Same on all devices
    }
    unicast_src_ip <lb-master-ip> # IP address of master-lb
    unicast_peer {
        <lb-backup-ip>   # IP address of the backup-lb
   }

    # The virtual ip address shared between the two loadbalancers
    virtual_ipaddress {
        <lb-vip>    # vip 
    }
    # Use the Defined Script to Check whether to initiate a fail over
    track_script {
        chk_haproxy
    }
}
```
*On Backup/Secondary node:*

```sh
vim /etc/keepalived/keepalived.conf
```

```sh
# Define the script used to check if haproxy is still working
vrrp_script chk_haproxy {
    script "/usr/bin/killall -0 haproxy"
    interval 2
    weight 2
}

# Configuration for Virtual Interface
vrrp_instance LB_VIP {
    interface ens6
    state BACKUP        # set to BACKUP on the peer machine
    priority 300        # set to  301 on the peer machine
    virtual_router_id 51

    authentication {
        auth_type user
        auth_pass UGFzcwo=  # Password for accessing vrrpd. Same on all devices
    }
    unicast_src_ip <lb-backup-ip> #IP address of backup-lb
    unicast_peer {
        <lb-master-ip>  #IP address of the master-lb
   }

    # The virtual ip address shared between the two loadbalancers
    virtual_ipaddress {
        <lb-vip>  #vip
    }
    # Use the Defined Script to Check whether to initiate a fail over.
    track_script {
        chk_haproxy
    }
}
```

*Enable and restart keepalived service:*
```sh
{
systemctl enable --now keepalived
systemctl start keepalived
systemctl status keepalived
}
```

### Reference Links:

- [Keepalived](https://keepalived.readthedocs.io/en/latest/introduction.html)

- [Haproxy](https://www.haproxy.org/?ref=linuxhandbook.com)

