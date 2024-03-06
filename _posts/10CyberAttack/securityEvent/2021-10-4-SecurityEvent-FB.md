---
title: Meow's SecurityEvent - 2021 Oct 4 FB outage
date: 2021-10-4 11:11:11 -0400
categories: [10CyberAttack, SecurityEvent]
tags: [SecurityEvent]
toc: true
image:
---

- [2021 Oct 4 FB outage](#2021-oct-4-fb-outage)
  - [Meet BGP](#meet-bgp)
  - [FB timeline](#fb-timeline)
- [facebook炸了却波及到了好多其他网站](#facebook炸了却波及到了好多其他网站)

- ref
  - https://blog.cloudflare.com/october-2021-facebook-outage/
  - https://www.1point3acres.com/bbs/thread-804646-1-1.html


---

# 2021 Oct 4 FB outage

> Update from Facebook
> Facebook has now published a blog post giving some details of what happened internally.
> Externally, we saw the BGP and DNS problems outlined in this post but the problem actually began with a `configuration change` that affected the entire internal backbone.
> That cascaded into Facebook and other properties disappearing and staff internal to Facebook having difficulty getting service going again.

---

## Meet BGP

BGP stands for Border Gateway Protocol.
- mechanism to exchange routing information between `autonomous systems (AS)` on the Internet.
- The big routers that make the Internet work have huge, constantly updated lists of the possible routes that can be used to deliver every network packet to their final destinations.
- Without BGP, the Internet routers wouldn't know what to do, and the Internet wouldn't work.

The Internet is literally a network of networks, and it’s **bound together by BGP.**
- BGP allows `one network (say Facebook)` to advertise its presence to `other networks` that form the Internet.
- As we write Facebook is not advertising its presence, ISPs and other networks can’t find Facebook’s network and so it is unavailable.

The `individual networks` each have an ASN: an Autonomous System Number.
- An Autonomous System (AS) is an individual network with a unified internal routing policy.
- An AS can originate `prefixes` (say that they control a group of IP addresses), as well as `transit prefixes` (say they know how to reach specific groups of IP addresses).
  - Cloudflare's ASN is AS13335.
- Every ASN needs to announce its `prefix route`s to the Internet using BGP;
- otherwise, no one will know how to connect and where to find us.


![image5-10](https://i.imgur.com/d4RFWEL.png)

- six autonomous systems on the Internet
- 2 possible routes that one packet can use to go from Start to End.
  - AS1 → AS2 → AS3 being the fastest,
  - and AS1 → AS6 → AS5 → AS4 → AS3 being the slowest,
  - but that can be used if the first fails.


## FB timeline

- 15:58 UTC - Facebook had stopped announcing the routes to their `DNS prefixes`.
  - That meant that, at least, `Facebook’s DNS servers` were unavailable.
  - Because `Cloudflare’s 1.1.1.1 DNS resolver` could no longer respond to queries asking for the IP address of facebook.com.

```
route-views>show ip bgp 185.89.218.0/23
% Network not in table
route-views>

route-views>show ip bgp 129.134.30.0/23
% Network not in table
route-views>
```

- Meanwhile, other Facebook IP addresses remained routed
  - but weren’t particularly useful since without `DNS Facebook`
  - related services were effectively unavailable:

```
route-views>show ip bgp 129.134.30.0
BGP routing table entry for 129.134.0.0/17, version 1025798334
Paths: (24 available, best #14, table default)
  Not advertised to any peer
  Refresh Epoch 2
  3303 6453 32934
    217.192.89.50 from 217.192.89.50 (138.187.128.158)
      Origin IGP, localpref 100, valid, external
      Community: 3303:1004 3303:1006 3303:3075 6453:3000 6453:3400 6453:3402
      path 7FE1408ED9C8 RPKI State not found
      rx pathid: 0, tx pathid: 0
  Refresh Epoch 1
route-views>
111
```

- keep track of all the BGP updates and announcements we see  in our global network.
  - At our scale, the data we collect gives us a view of how the Internet is connected and where the traffic is meant to flow from and to everywhere on the planet.
  - **BGP UPDATE message**
    - informs a router of any changes you’ve made to a prefix advertisement or entirely withdraws the prefix.
  - We can clearly see this in the number of updates we received from Facebook when checking our `time-series BGP database`.
  - Normally this chart is fairly quiet:
    - Facebook doesn’t make a lot of changes to its network minute to minute.
    - But at around 15:40 UTC we saw a peak of routing changes from Facebook.
    - That’s when the trouble began.


![image4-11](https://i.imgur.com/XMcfrUx.png)


- split this view by routes announcements and withdrawals, we get an even better idea of what happened.
  - Routes were withdrawn, Facebook’s DNS servers went offline
  - With those withdrawals, Facebook and its sites had effectively disconnected themselves from the Internet


- As a direct consequence of this, DNS resolvers all over the world stopped resolving their domain names.


```
➜  ~ dig @1.1.1.1 facebook.com
;; ->>HEADER<<- opcode: QUERY, status: SERVFAIL, id: 31322
;facebook.com.			IN	A

➜  ~ dig @1.1.1.1 whatsapp.com
;; ->>HEADER<<- opcode: QUERY, status: SERVFAIL, id: 31322
;whatsapp.com.			IN	A

➜  ~ dig @8.8.8.8 facebook.com
;; ->>HEADER<<- opcode: QUERY, status: SERVFAIL, id: 31322
;facebook.com.			IN	A

➜  ~ dig @8.8.8.8 whatsapp.com
;; ->>HEADER<<- opcode: QUERY, status: SERVFAIL, id: 31322
;whatsapp.com.			IN	A
```


- DNS also has its routing mechanism.
  - When someone types the https://facebook.com URL in the browser,
  - the DNS resolver, responsible for translating domain names into actual IP
  - it first checks if it has something in its cache and uses it.
  - If not, it tries to grab the answer from the domain nameservers, typically hosted by the entity that owns it.
  - If the nameservers are unreachable or fail to respond because of some other reason, then a `SERVFAIL` is returned, and the browser issues an error to the user.
  - Due to Facebook stopping announcing their `DNS prefix routes` through BGP,
    - everyone else's DNS resolvers had no way to connect to their nameservers.
    - Consequently, 1.1.1.1, 8.8.8.8, and other `major public DNS resolvers` started issuing (and caching) `SERVFAIL` responses.
    - human behavior and application logic kicks in and causes another exponential effect. A tsunami of additional DNS traffic follows.
      - apps won't accept an error for an answer and start retrying, sometimes aggressively,
      - end-users also won't take an error for an answer and start reloading the pages, or killing and relaunching their apps, sometimes also aggressively.


![image6-9](https://i.imgur.com/tJ4CZd6.png)

- Fortunately, 1.1.1.1 was built to be Free, Private, Fast (as the independent DNS monitor DNSPerf can attest), and scalable, and we were able to keep servicing our users with minimal impact.
  - The vast majority of our DNS requests kept resolving in under 10ms.
  - At the same time, a minimal fraction of p95 and p99 percentiles saw increased response times, probably due to expired TTLs having to resort to the Facebook nameservers and timeout.
  - The 10 seconds DNS timeout limit is well known amongst engineers.


- At around 21:00 UTC
  - renewed BGP activity from Facebook's network which peaked at 21:17 UTC.


![unnamed-3-3](https://i.imgur.com/WtqB56s.png)


# facebook炸了却波及到了好多其他网站

- DNS解析器(DNSresolver)
  - 可以把域名(wwwgooglecom)解析成IP地址(eg1422506468)
  - 任何网站背后都是很多ip地址(根据request所在的位置找到对应服务器)
  - 所以你的每一个网络请求都要先通过DNSresolver给你解析出来

- DNS服务器(DNS authoritative nameserver)
- 各大公司serve自己ip的地方
- DNS解析器自己也不知道facebookcom的ip是什么他要去问facebook的nameserver


Process:
1. 从源头开始facebook BGP的configuration出错了
   - 导致各DNS解析器(e.g. 8.8.8.8, 1.1.1.1, etc. 其实大部分人用的都是运营商自己的DNS解析器)无法找到facebook的DNS record。
   - 比如
     - 之前8.8.8.8问fb的nameserver "facebook.com的IP是多少啊？"
     - fb的nameserver说 "x.x.x.x"。
     - 现在8.8.8.8再问，
     - fb的nameserver直接回复 "facebook.com不存在" 或者 "我现在炸了" (NXDOMAIN / SERVFAIL)。

2. DNS是用的UDP传输的
   - 经常会有丢包packet loss或者其他问题比如上述“我炸了请重试”，
   - 所以各个解析器都会设置一些重试retry。
   - 因为找不到facebook的DNS record解析器们就会一直疯狂retry。
   - retry来自于多方面
     - 可以来自DNSresolver
     - 也可以来自浏览器
     - 也可以来自用户手动刷新
     - 又或者是运营商中间的哪个layer
     - example：
       - 每一次facebookcom的request
       - DNSresolver自己retry3次
       - 你用的chrome浏览器retry3次
       - 加载不出来你非常着急自己再点浏览器refresh三次
       - 这乘在一起就是3*3*3=27次
       - 这还不考虑中间某些其他layer
       - 这仅仅是一次对facebook的request被放大了几十倍
       - 这对于很多dnsresolver简直是噩梦
       - 于是纷纷承受不了就炸了
       - 于是别网站的request也变慢
       - 大家刷新的更多了恶性循环
       - 有时候很多小东西经过乘积或指数放大有比预想要大得多且可怕的多的影响
     - 有点像缓存穿透的味道


3. FB的scale太大，全球性
   - 大家越连不上facebook就会总点刷新，解析器们每次失败也会又重试很多次，
   - 这种巨大量的retry很多DNS解析器也承受不住。

4. Retry太多了DNS解析器们受不了会集体爆炸
   - 这样所有网站的DNS解析都会收到影响。
   - 比如某个解析器在这个地区有100个task，各个task逐渐开始炸
     - 比如50个炸了，所有剩余请求都积压到剩的50个了，
     - 这样load增加，速度就会变慢，到最后受不了了剩下的一起全炸了。
   - 当然google的8.8.8.8只是小炸变慢了一点，但是小的DNS resolver就会炸的很严重，
   - 这也是为什么twitter上很多人说上不了网把DNS改成8.8.8.8就好了

5. 这样上任何网站都会变慢甚至无法连接，直到FB和DNS resolver们恢复。

非常有趣的是，这就是典型的蝴蝶效应，虽然看起来是只是facebook的bgp configuration出错了，但是差点炸掉了整个互联网。

假如几个facebook这么大体量的公司同时发生这种"小"错误，这样就能把全世界所有dns解析器炸掉，比如连google的8.8.8.8也无法幸免。整个互联网全都沦陷也是可能只是一瞬间发生的事。
