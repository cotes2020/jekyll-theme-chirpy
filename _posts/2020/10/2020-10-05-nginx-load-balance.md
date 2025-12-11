---
title: "Nginx 学习笔记：http代理/负载均衡/长连接/健康检查"
date: 2020-10-05
permalink: /2020-10-05-nginx-load-balance/
tags: [网关设计, HTTP协议, 负载均衡]
---

Nginx 是一款轻量级的 Web 服务器、反向代理服务器，由于它的内存占用少，启动极快，高并发能力强，在互联网项目中广泛应用。


在使用中，经常将其用到以下几个方面：

- 反向代理 SPA/SSR 应用
- 负载均衡

## Ubuntu 配置 Nginx


安装 nginx：


```shell
sudo apt-get install nginx
```


重启 nginx 前，需要测试 nginx 配置是否正确：


```shell
sudo nginx -t
```


重新加载 nginx：


```shell
sudo nginx -s reload
```


不特殊指定，默认加载`/etc/conf/nginx.conf` 的配置。


## Nginx 默认配置解读


对于`nginx version: nginx/1.14.0 (Ubuntu)`版本的 nginx，默认配置如下所示：


```yaml
user www-data;
worker_processes auto;
pid /run/nginx.pid;
include /etc/nginx/modules-enabled/*.conf;

events {
	worker_connections 768;
	# multi_accept on;
}

http {

	##
	# Basic Settings
	##
	sendfile on;
	tcp_nopush on;
	tcp_nodelay on;
	keepalive_timeout 65;
	types_hash_max_size 2048;
	# server_tokens off;

	# server_names_hash_bucket_size 64;
	# server_name_in_redirect off;

	include /etc/nginx/mime.types;
	default_type application/octet-stream;

	##
	# SSL Settings
	##

	ssl_protocols TLSv1 TLSv1.1 TLSv1.2; # Dropping SSLv3, ref: POODLE
	ssl_prefer_server_ciphers on;

	##
	# Logging Settings
	##

	access_log /var/log/nginx/access.log;
	error_log /var/log/nginx/error.log;

	##
	# Gzip Settings
	##

	gzip on;

	# gzip_vary on;
	# gzip_proxied any;
	# gzip_comp_level 6;
	# gzip_buffers 16 8k;
	# gzip_http_version 1.1;
	# gzip_types text/plain text/css application/json application/javascript text/xml application/xml application/xml+rss text/javascript;

	##
	# Virtual Host Configs
	##

	include /etc/nginx/conf.d/*.conf;
	include /etc/nginx/sites-enabled/*;
}


#mail {
#	# See sample authentication script at:
#	# <http://wiki.nginx.org/ImapAuthenticateWithApachePhpScript>
#
#	# auth_http localhost/auth.php;
#	# pop3_capabilities "TOP" "USER";
#	# imap_capabilities "IMAP4rev1" "UIDPLUS";
#
#	server {
#		listen     localhost:110;
#		protocol   pop3;
#		proxy      on;
#	}
#
#	server {
#		listen     localhost:143;
#		protocol   imap;
#		proxy      on;
#	}
#}

```


几个重要的点：

- user：代表执行 nginx 的用户，可以换成权限更高的用户
- http/mail：代表 http/邮箱服务的配置
- `include /etc/nginx/conf.d/*.conf`：加载 conf.d 下的所有以.conf 结尾的配置文件

在配置的时候，一般都将对应的配置文件放在`/etc/nginx/conf.d/`。


## 代理 Http(s)服务


[xin-tan.com](http://xin-tan.com/) 是 vuepress 构建的，为了提供给用户更好的浏览体验，用 nginx 做服务器，要求如下：

- 强制 https：监听 80 port，请求转发给 443 port（换协议）
- 配置 SSL 证书：配置私钥文件和证书文件（一般云厂商申请的 ssl 证书，都有对应说明）
- 指定 locaiton：和 SPA 应用一致，路由交给前端 router 管理

代码：[https://github.com/dongyuanxin/blog/blob/master/nginx.conf](https://github.com/dongyuanxin/blog/blob/master/nginx.conf)


```yaml
server {
    listen 80;
    server_name xin-tan.com;
    # nginx最新写法
    return 301 https://$server_name$request_uri;

    location / {
        root /home/ubuntu/data/blog-static;
        index index.html index.htm index.nginx-debian.html;
        try_files $uri $uri/ =404;
    }
}

server {
    listen 443 ssl;
    #填写绑定证书的域名
    server_name xin-tan.com;
    #证书文件名称
    ssl_certificate /tmp/1_bundle.crt;
    #私钥文件名称
    ssl_certificate_key /tmp/2.key;
    ssl_session_timeout 5m;
    #请按照以下协议配置
    ssl_protocols TLSv1 TLSv1.1 TLSv1.2;
    #请按照以下套件配置，配置加密套件，写法遵循 openssl 标准。
    ssl_ciphers ECDHE-RSA-AES128-GCM-SHA256:HIGH:!aNULL:!MD5:!RC4:!DHE;
    ssl_prefer_server_ciphers on;

    location / {
        #网站主页路径。此路径仅供参考，具体请您按照实际目录操作。
        root /home/ubuntu/data/blog-static;
        index index.html index.htm index.nginx-debian.html;
        try_files $uri $uri/ =404;
    }
}
```


## 实现负载均衡


nginx 的优势之一是：可以代理多个服务器。


通俗来说，按照某些规则，将请求分配给对应的服务器。从而实现某些方面（请求量、服务器负载等等）的负载均衡。


### 基本配置


假设针对 [a.com](http://a.com/) 域名的 http 服务，提供负载均衡。并且有多个后端服务，分别运行在本地服务（也可以是同一 vpc 下、或者远程服务器）的不同端口。


假设这些后端服务都是相同的逻辑，并且是无状态服务。


那么首先需要配置反向代理：


```yaml
server {
    listen 80;
    server_name 193.112.241.232;

    location / {
        proxy_pass <http://myservers>;
    }
}
```


这里的 myservers 是 upstream 的 name（往下看就明白了）。


### 负载均衡的各种策略（upstream）


**1、普通轮询**


```text
upstream myservers {
    server 127.0.0.1:4445;
    server 127.0.0.1:4446;
}
```


**2、按权重轮询**


```text
upstream myservers {
    server 127.0.0.1:4445 weight=10;
    server 127.0.0.1:4446 weight=1;
}
```


**3、ip hash：相同的 ip 负载到同一个 upstream server**


```text
upstream myservers{
    ip_hash;
    server 127.0.0.1:4445 weight=1;
    server 127.0.0.1:4446 weight=2;
}
```


**4、一致性 hash**


```text
upstream myservers{
    hash $custom_key consistent;
    server 127.0.0.1:4445 weight=2;
    server 127.0.0.1:4446 weight=1;
}
```


**5、自定义 hash 规则：可以使用 Nginx 变量**


```text
upstream myservers{
    hash $uri;
    server 127.0.0.1:4445
    server 127.0.0.1:4446;
}
```


## 配置 HTTP 长连接


### 为什么要用长连接？

- 不需要每次 tcp 请求都经历握手和挥手的过程
- 提高响应请求响应时间，减少 time-wait 状态的 Socket

### Nginx 配置思路


先来看请求链路过程是：C 端请求 => Nginx 代理 => 上游的 Server。


所以，Nginx 要做到两头都是长连接：

- C 端请求 => Nginx 代理：nginx 扮演 server
- Nginx 代理 => 上游的 Server：naginx 扮演 client

思路和 nginx 在不同过程中的“角色”清除后，剩下就是配置了。


### Nginx 配置


### C 端到 Nginx


默认配置文件中已有：keepalive_timeout。为 0，禁用长连接；不为 0，代表长连接超时关闭时间。


除此之外，还有常用的：keepalive_requests。默认为 100，为一个长连接能接受的最大请求数。


### Nginx 到上游 Server


反向代理配置中，需要支持 keep-alive：


```text
server {
    listen 80;
    server_name 193.112.241.232;

    location / {
        proxy_http_version 1.1;
        proxy_set_header Connection "";
        proxy_pass <http://myservers>;
    }
}

```


到上游 Server 的配置：


```text
upstream myservers {
    server 127.0.0.1:4445;
    server 127.0.0.1:4446;
    keepalive 100;
}

```


## 服务健康检查


### 为什么要心跳检查？


目的是为了 Server 的健康检查。因为 nginx 支持 4 层和 7 层代理，所以支持 tcp 心跳检查和 http 心跳检查。


### nginx 配置


**http 心跳检查**：


```text
upstream myservers {
    server 127.0.0.1:4445;
    server 127.0.0.1:4446;
    # 5s检查一次。检查成功1次，标记server存活；失败5次，标记挂掉。
    check interval=5000 rise=1 fail=5 timeout=5000 type=http;
    # http 心跳包
    check_http_send "HEAD /status HTTP/1.0\\r\\n\\r\\n";
    check_http_expect_alive http_2xx http_3xx;
}
```


**tcp 检查**：


```text
upstream myservers {
    server 127.0.0.1:4445 weight=1;
    server 127.0.0.1:4446 weight=2;
    check interval=5000 rise=1 fail=5 timeout=5000 type=tcp;
}
```


## 参考资料

- [Nginx 中保持长连接的配置 - 运维记录](https://www.cnblogs.com/kevingrace/p/9364404.html)
- [HTTP 的长连接和短连接——Node 上的测试](https://www.cnblogs.com/cswuyg/p/5103909.html)
- 《亿级流量网站架构核心技术》

