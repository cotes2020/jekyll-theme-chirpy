---
title: Docker - basic web site by docker
date: 2020-07-16 11:11:11 -0400
categories: [30System, Docker]
tags: [Docker]
math: true
image:
---


# basic web site by docker

---

```bash

docker run -it --name a-container alpine
docker exec a-container apk add nginx

docker cp a-container:/etc/nginx/conf.d/default.conf .
vim default.conf
# server {
# 	listen 80 default_server;
# 	listen [::]:80 default_server;
#   root /var/www/;
# }

docker cp default.conf a-container:/etc/nginx/conf.d/default.conf


~ $ docker container ls
CONTAINER ID        IMAGE               COMMAND             CREATED             STATUS              PORTS               NAMES
f9f7eb278e81        alpine              "/bin/sh"           17 minutes ago      Up 16 minutes                           web01


~ $ vim index.html
# hello world!!

~ $ docker cp index.html web01:/var/www
~ $ docker exec -dt web01 nginx -g 'pid /tmp/nginx.pid; daemon off;'

~ $ docker inspect web01
[
    {
        "Id": "f9f7eb278e81d0c19f363e970616855d9c5aa9960e3e5ac2fa181592f0e9b12c",
        "Created": "2020-08-19T01:39:03.961208087Z",
        "Path": "/bin/sh",
        "Args": [],
        "State": {
            ..............
                }
    }
]

~ $ docker inspect web01 | grep IP
            "LinkLocalIPv6Address": "",
            "LinkLocalIPv6PrefixLen": 0,
            "SecondaryIPAddresses": null,
            "SecondaryIPv6Addresses": null,
            "GlobalIPv6Address": "",
            "GlobalIPv6PrefixLen": 0,
            "IPAddress": "172.17.0.3",
            "IPPrefixLen": 16,
            "IPv6Gateway": "",
                    "IPAMConfig": null,
                    "IPAddress": "172.17.0.3",
                    "IPPrefixLen": 16,
                    "IPv6Gateway": "",
                    "GlobalIPv6Address": "",
                    "GlobalIPv6PrefixLen": 0,

~ $ curl 172.17.0.3
hello world!!


# publish
~ $ docker commit web01 web-base
sha256:b6290bdfc157161864f4fc2b2628903dfc315655eb040673186399b6a8ae49b0

~ $ docker run -p 80:80 -dt --name web02 web-base
d279a242a2de49701adb56166f6b8a3bb2551aad921c6b32c115449a68c7a39c

~ $ docker inspect web02 | grep IP
            "LinkLocalIPv6Address": "",
            "LinkLocalIPv6PrefixLen": 0,
            "SecondaryIPAddresses": null,
            "SecondaryIPv6Addresses": null,
            "GlobalIPv6Address": "",
            "GlobalIPv6PrefixLen": 0,
            "IPAddress": "172.17.0.2",
            "IPPrefixLen": 16,
            "IPv6Gateway": "",
                    "IPAMConfig": null,
                    "IPAddress": "172.17.0.2",
                    "IPPrefixLen": 16,
                    "IPv6Gateway": "",
                    "GlobalIPv6Address": "",
                    "GlobalIPv6PrefixLen": 0,

~ $ curl 172.17.0.2
curl: (7) Failed to connect to 172.17.0.2 port 80: Connection refused

~ $ docker exec -dt web02 nginx -g 'pid /tmp/nginx.pid; daemon off;'

~ $ curl 172.17.0.2
hello world!!
~ $ curl 172.17.0.3
hello world!!
~ $ curl localhost
hello world!!


# browser: use local ip

```
