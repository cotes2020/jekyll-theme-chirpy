---
title: DNS Attack and Securtiy
date: 2020-07-16 11:11:11 -0400
categories: [00Basic, Network]
tags: [Linux, DNS]
math: true
image:
---

- [DNS Attack](#dns-attack)
  - [DNS lookup](#dns-lookup)
  - [DNS spoofing/cache poisoning](#dns-spoofingcache-poisoning)
    - [DNS Cache Poisoning DNS 快取污染](#dns-cache-poisoning-dns-快取污染)
      - [DNS Cache Poisoning and the Birthday Paradox](#dns-cache-poisoning-and-the-birthday-paradox)
    - [Subdomain DNS Cache Poisoning `attack name servers`](#subdomain-dns-cache-poisoning-attack-name-servers)
    - [Client-Side DNS Cache Poisoning Attacks `target client`](#client-side-dns-cache-poisoning-attacks-target-client)
  - [DDoS DNS Attacks](#ddos-dns-attacks)
  - [DNS tunneling](#dns-tunneling)
  - [NXDOMAIN attack](#nxdomain-attack)
  - [Phantom domain attack](#phantom-domain-attack)
  - [Random subdomain attack](#random-subdomain-attack)
  - [Domain lock-up attack](#domain-lock-up-attack)
  - [Botnet-based CPE attack `attack the CPE device`](#botnet-based-cpe-attack-attack-the-cpe-device)
  - [domain name kiting 空头支票 `attack the DNS registration info`](#domain-name-kiting-空头支票-attack-the-dns-registration-info)
  - [Hijacking 劫持](#hijacking-劫持)
    - [DNS hijacking `attack the DNS record of website on nameserver`](#dns-hijacking-attack-the-dns-record-of-website-on-nameserver)
    - [Domain Hijacking `attack the DNS registration info`](#domain-hijacking-attack-the-dns-registration-info)
  - [Pharming and Phishing](#pharming-and-phishing)
    - [DNS pharming 网域嫁接 `attack the client computer`](#dns-pharming-网域嫁接-attack-the-client-computer)
    - [DNS phishing 网络仿冒](#dns-phishing-网络仿冒)
- [DNS securtiy](#dns-securtiy)
  - [DNS Sinkhole DNS沉洞](#dns-sinkhole-dns沉洞)
  - [DNSSEC - Domain Name System  Security Extensions](#dnssec---domain-name-system--security-extensions)
  - [operator of a DNS zone](#operator-of-a-dns-zone)
  - [DNS firewall](#dns-firewall)
  - [DNS resolvers](#dns-resolvers)
- [DNS privacy](#dns-privacy)
  - [DNS over HTTPS](#dns-over-https)


- ref
  - DNS Poisoning (S+ 7th ch9)

---


## DNS Attack

> DNS system was not designed with security in mind and contains several design limitations


DNS:
- resolves host names to IP addresses.
  - responds with the correct IP address and your system connects to the web site using the IP address.
- provides reverse lookups
  - client sends an IP address to a DNS server with a request to resolve it to a name.
- Some applications use this as a rudimentary security mechanism to detect spoofing.


<font color=red> Coumpter domain name - Host File - local DNS (primery DNS + secondary DNS) - public DNS </font>
1. Change the host file - need administrator.
2. Hack the local DNS - redirect it to malicious DNS server.
- White hat: stop unwilling website reading.
- Black hat: steal data information.


DNS is an integral part of most Internet requests, it can be a prime target for attacks. DNS servers is vulnerable to a broad spectrum of attacks, including
- DNS poisoning,
- pharming
- amplification,
- DoS (Denial of Service),
- the interception of private personal information.


Example:
- attacker try to spoof the computer’s identity by using a different name during a session.
- However, the Transmission Control Protocol/ Internet Protocol (TCP/IP) packets in the session include the IP address of the masquerading system and a reverse lookup shows the system’s actual name.
- If the names are different, it shows suspicious activity.
- Reverse lookups are not 100 percent reliable because reverse lookup records are optional on DNS servers.
- However, they are useful when they’re available.


---



### DNS lookup

attacks when DNS lookup for a resource:

Example:
- a person is trying to connect to your FTP server to upload some sensitive data.
- The user types in `ftp.anycomp.com` and presses enter.
- The DNS server closest to the user (defined in your TCP/IP properties) looks through its cache to see whether it knows the address for ftp.anycomp.com.
- If it’s not there,the server works its way through the DNS architecture to find the **authoritative server** for anycomp.com, which must have the correct IP address.
- This response is returned to the client, and FTP-ing begins


Suppose an attacker want that sensitive data.
- DNS poisoning
  - attacker change the cache on the local name server to point to a bogus server instead of the real address for `ftp.anycomp.com`
  - Then the user would connect and upload the documents directly to your server.

one simple mitigation
- restrict the amount of time records can stay in cache before they’re updated.


NOTE
DNS poisoning is of enough importance that an entire extension to DNS was created, way back in 1999.
- The `Domain Name System Security Extensions (DNSSEC)` is a suite of IETF specifications for securing certain kinds of information provided by DNS.
- Dan Kaminsky made DNS vulnerabilities widely known back around 2010, and many service providers are rolling this extension out to ensure that DNS results are cryptographically protected.

setting up DNS required not only a hierarchical design but someone to manage it.
- someone had to be in charge of registering who owned what name and which address ranges went with it.
- someone had to hand out the addresses in the first place.


---


### DNS spoofing/cache poisoning

> DNS server is given information about a server that it thinks is legitimate when it isn’t.

- forged DNS data is introduced into a <font color=red> DNS resolver’s cache</font>, the resolver `returning an incorrect IP address for a domain`.
  - Instead of going to the correct website
  - traffic can be diverted to a malicious machine or anywhere else the attacker desires;
  - often this will be a replica of the original site used for malicious purposes such as distributing malware or collecting login information.

- modify or corrupt DNS results.

- DNS server is used to determine a destination.
  - DNS server is given information about a name server that it thinks is legitimate when it isn’t.
  - redirect users to a website other than the one to which they wanted to go,
  - reroute mail,
  - do any other type of redirection in which data from

- most popular techniques:
  - fast flux


successful DNS poisoning attack:
- modify the IP address associated with google.com and replace it with the IP address of a malicious web site.
- Each time a user queries DNS for the IP address of google.com, the DNS server responds with the IP address of the malicious web site.

Countermeasure:
- Many current DNS servers use **Domain Name System Security Extensions (DNSSEC)**
  - to protect the DNS record
  - prevent DNS poisoning attacks



Types of DNS Poisoning:
1. Intranet DNS Spoofing (Local network).
2. Internet DNS Spoofing (Remote network).
3. Proxy Server DNS Poisoning.
4. DNS Cache Poisoning.


Intranet DNS Spoofing (Local Network)
- you must be connected to the local area network (LAN) and be able to sniff packets.
- It works well against switches with ARP poisoning the router.


Internet DNS Spoofing (Remote Network)
- Send a Trojan to the victim’s machine
- and change her DNS IP address to that of the attacker’s.
- It works across networks.


Proxy Server DNS Poisoning
- Send a Trojan to the victim’s machine and change her proxy server settings in Internet Explorer to that of the attacker’s.
- it works across networks.

---

#### DNS Cache Poisoning DNS 快取污染


attacker trick DNS server to cache a false DNS record,
- Let clients issuing DNS requests to that server
- resolve domains to **attacker-supplied IP addresses**

- 跟 ARP 一樣，DNS 若讀到一組不正確的紀錄，那他就會被影響到，而且還會把這筆不正確的紀錄告訴別人。
  - 然後一段時間內，被影響到的 DNS 伺服器都會給出不正確的資訊。
  - 由於 DNS 是樹狀結構，所以 DNS 快取污染會影響到附近的 DNS 伺服器。

  - DNS uses a 16-bit request identifier to pair queries with answers
  - 正常的狀況下，DNS 伺服器問別人資訊時:
    - 先驗證「這筆資訊是不是合理的」（如使用 DNSSEC）


- **Cache may be poisoned when a name server**:
  - Disregards identifiers
  - Has predictable ids
  - Accepts unsolicited DNS records


- Normally, computer uses a DNS server provided by the organization or ISP.
  - DNS servers are generally deployed in an organization's network to improve resolution response performance by caching previously obtained query results.

- <font color=blue> Poisoning attacks on a single DNS server </font>
  - affect the users serviced directly by the compromised server or indirectly by its downstream server(s) if applicable.

- To perform a cache poisoning attack, the attacker exploits a flaw in the DNS software.

- <font color=blue> If the server does not correctly validate DNS responses </font> to ensure that they are from an authoritative source (using DNSSEC)
  - the server will caching the incorrect entries locally
  - and serve them to other users that make the same request.
  - direct users of a website to another site of the attacker's choosing.


**Step:**

![Screen Shot 2018-11-17 at 09.55.19](https://i.imgur.com/QnyEaA1.png)

1. An attacker launch a **DNS cache poisoning attack** against an ISP DNS server.
   - 攻擊方的網域名稱是 `attacker.here`, IP 1.1.1.1
   - 被攻擊方ISP DNS server是 `target.here`

2. Attacker **rapidly transmits DNS queries to ISP DNS server target**
   - queries an authoritative name server on behalf of Eve. 需要解析 `attacker.here`。
   - 不過被攻擊方並不知道這個網域，所以他要去問別人。
     - 他會和另一台伺服器這樣問: `attacker.here` 的位置是啥？
     - `Attacker.here. IN A`

3. **Eve simultaneously sends a DNS response to her own query**
   - it spoofing the source IP address as originating at the authoritative name server, with the destination IP set to the ISP DNS server target.
   - `attacker.here` 是指向 `ns.target.here`
   - 然後我還知道 `ns.target.here` 是對到 1.1.1.1

4. 假設被攻擊方完全不做檢查
   - The ISP server accepts Eve’s forged response
   - The ISP server caches a DNS entry associating the domain Eve requested with the malicious IP address Eve provided in her forged responses.
5. At this point, any downstream users of that ISP, will be directed to Eve’s malicious web site when they issue DNS requests to resolve the domain name targeted by Eve.
   - 每個人問他 `attacker.here`，他都會回 1.1.1.1（即使他的 IP 根本不是這個）


- For example
  - an attacker spoofs the IP address DNS entries for a target website on a given DNS server, replacing them with the IP address of a server he controls.
  - He then creates files on the server he controls with names matching those on the target server.
  - These files could contain malicious content, such as a computer worm or a computer virus.
  - A user whose computer has referenced the poisoned DNS server would be tricked into accepting content coming from a non-authentic server and unknowingly download malicious content.


There are several obstacles to issue a fake DNS response that will be accepted.
- First, an attacker must issue a response to her own DNS query before the authoritative name server respond.
  - It is easily overcome, however, because if the attacker forces the target name server to query external authoritative name servers, she can expect that her immediate direct response will be received before these external name servers have a chance to perform a lookup and issue a reply.
- Second, each DNS request is given a **16-bit query ID**
  - If the response to a query donot have same ID like request, it will be ignored.
  - If he successfully guesses the random query ID chosen by the ISP DNS server, the response will be cached.
  - This guessing is actually more likely if the attacker `issues a lot of fake requests and responses to the same domain name lookup`. (Birthday Paradox )
  - In 2002, most major DNS software simply used sequential numbers for query IDs, allowing easy prediction and circumvention of this naive authentication. Once this bug was disclosed, most DNS software vendors implement randomization of query IDs.


##### DNS Cache Poisoning and the Birthday Paradox
- increase in fake requests to increase attack success probability is a result of a principle known as the **birthday paradox 生日悖论**:
- the probability of two or more people in a group of 23 sharing the same birthday is greater than 50%.
  - In a group of 23 people, there are actually `(23*22)/2 = 253` pairs of birthdays,
  - and it only takes one matching pair for the birthday paradox to hold.

  - **p(n) = n个人中每人的生日日期都不同的概率**

  	- 假如n > 365，其概率为0
  	- 假设n ≤ 365，则概率为: ![Pasted Graphic](https://i.imgur.com/Fkr1i65.png)
  	- 因为第二个人不能跟第一个人有相同的生日(概率是364/365),第三个人不能跟前两个人生日相同(概率为363/365),依此类推。
  	- 用阶乘可以写成如下形式:￼![Pasted Graphic 1](https://i.imgur.com/StG7Omr.png)

  - **p(n) = n个人中至少2人生日相同的概率**

  	- n≤365￼: ![Pasted Graphic 2](https://i.imgur.com/7hj3TQh.png)
  	- n大于365时概率为1。
  	- n=23发生的概率大约是0.507

- apply the reasoning of the birthday paradox to DNS cache poisoning.
-	- An attacker issuing a fake response, guess a transaction ID
  	- n different 16-bit real IDs XXXXXXXXXXXXXXXX, `probability = n/216`;
-	- hence, she would fail to match one with probability 1 − n/216.
-	- Thus, an attacker issuing `n fake responses` will fail to guess a transaction ID equal to one of n different 16-bit real IDs with a probability ![Screen Shot 2018-11-17 at 11.06.02](https://i.imgur.com/Pog2XkU.png)
- when n = 213, the attacker will have roughly at least a 50% chance that one of her random responses will match a real request.
-	- A DNS cache poisoning attack based on the birthday paradox:
-	- (a) First, an attacker sends `n DNS requests` for the domain she wishes to poison.
-	- (b) The attacker sends `n corresponding replies` for her own request.
-	- If she successfully guesses one of the random query IDs chosen by the ISP DNS server, the response will be cached.
￼
- Hashed passwords are susceptible to birthday attacks.

![Screen Shot 2018-11-17 at 11.10.26](https://i.imgur.com/NSsT5In.png)


---

**防禦方式DNS Cache Poisoning Prevention**
基本上，DNS 快取污染的防禦都是，都是「不要那麼相信對方跟你講的話」，以及「不要理會你沒問的東西」。
1. 使用 HTTPS 的數位簽章來解決掉偽造主機的問題
   - Use random identifiers for queries
   - Always check identifiers
   - Port randomization for DNS requests
2. 目前常見的防禦方式: DNSSEC
   - Challenging because it is still being deployed and requires reciprocity



---


#### Subdomain DNS Cache Poisoning `attack name servers`

- Despite the birthday paradox, the above guessing attack is `extremely limited because of its narrow time frame`.
  - when a correct response to a DNS query is received, that result is cached by the receiving server and stored for the time specified in the time-to-live field.
  - When a name server has a record in its cache, it uses that record rather than issuing a new query to an authoritative name server.

- As a result, the attacker `can only make as many guesses as he can` send in the time between the initial request and the valid reply from the authoritative name server.
  - On each failed guessing attempt, the valid (harmless) response will be cached by the targeted name server
  - so the attacker must wait for that response to expire before trying again.
- Responses may be cached for minutes, hours, or even days
  - this slowdown makes the attack described above almost completely infeasible.


Unfortunately, a new **subdomain DNS cache poisoning attack** in 2008, allows attackers to successfully perform DNS cache poisoning by 2 new techniques:

1. Issues **requests for different nonexistent subdomains** of the target domain.
   - Rather than issuing a request and response for a target domain like `example.com`, only allowing one attempt at a time
   - the attacker issues many requests, each for a different `nonexistent subdomain of the target domain`.
   - example:
   - the attacker might send requests for subdomains `aaaa.example.com, aaab.example.com, aaac.example.com`, and so on.
   - These subdomains don’t actually exist, so the name server for the target domain, example.com, just ignores these requests.
   - Simultaneously, the attacker issues responses for each of these requests, each with a guessed transaction ID.
   - The attacker now has so many chances to correctly guess the response ID
     - no competition from the target domain to worry about
     - it is relatively likely that the attack will be successful.
   - This new attack was shown to be successful against many popular DNS software packages, including **BIND**, the most commonly used system.

1. Include **glue record** that resolves the name server of target domain to attacker-controlled server.
   - Rather than simply reply with an address for each fake subdomain like abcc.example.com,
   - the attacker’s responses `include a glue record that resolves the name server of the target domain, example.com, to an attacker-controlled server`.
   - Using this strategy, on successfully guessing the transaction ID
   - Attacker can control not just one DNS resolution for a nonexistent domain but all resolutions for the entire target domain.



---


#### Client-Side DNS Cache Poisoning Attacks `target client`

![Screen Shot 2018-11-17 at 13.49.19](https://i.imgur.com/kZXcuiU.png)

In addition to attacks on name servers, a similar DNS cache poisoning attack can be conducted against a target client.
￼
A DNS cache poisoning attack against a client:
1. On visiting a malicious web site
   1. the victim views a page containing many images,
   2. each causing a <font color=blue> separate DNS request to a nonexistent subdomain of the domain that is to be poisoned </font>.
      1. An attacker can construct a `malicious web site containing HTML tags(such as image tags)` that automatically issue requests for additional URLs.
      2. These image tags each `issue a request to a different nonexistent subdomain of the domain the attacker wishes to poison`.
2. The malicious web server <font color=blue> sends guessed responses to each of these requests </font>
   1. When the attacker receives indication that the victim has navigated to this page, he can rapidly send DNS replies with poisoned glue records to the client.
3. On a successful guess, the client’s DNS cache will be poisoned.
   1. the client will cache the poisoned DNS entry.


- This type of attack is especially <font color=red> stealthy </font>
  - since it can be initiated just by someone visiting a web site that contains images that trigger the attack.
  - These images will not be found, of course, but the only warning the user has that this is causing a DNS cache poisoning attack is that the browser window may display some icons for missing images.


---


### DDoS DNS Attacks

It’s difficult to take down the Internet. However, a cyberattack in October 2016 effectively did so for millions of users in North America and Europe.
- Specifically, on October 21,
- attackers launched three DDoS attacks during the day at 7:00 a.m., at 11:52 a.m., and at 4:00 p.m.
- These attacks prevented users from accessing a multitude of sites, such as Amazon, CNN, Fox News, Netflix, PayPal, Reddit, Spotify, Twitter, Xbox Live, and more.
- Attackers infected many Internet-connected devices (such as video cameras, video recorders, printers, and baby monitors), with malware called <font color=red> Mirai </font>.
- Mirai forces individual systems to become bots within large botnets.
- On October 21, they sent commands to millions of infected devices directing them to repeatedly send queries to DNS servers.
- These queries overwhelmed the DNS servers and prevented regular users from accessing dozens of web sites.
- These three attacks were launched against DNS servers maintained by `Dyn, Inc.`, an Internet performance management company.
- it is possible to seriously disrupt DNS services, causing Internet access problems for millions of people.


---



### DNS tunneling

> uses other protocols to tunnel through DNS queries and responses.

- Attackers can use SSH, TCP, or HTTP to pass malware or stolen information into DNS queries, undetected by most firewalls.



### NXDOMAIN attack

> attacker inundates a DNS server with requests, asking for records that `do not exist`, to cause a denial-of-service for legitimate traffic.

- a type of DNS flood attack
- This can be accomplished using sophisticated attack tools that can auto-generate unique subdomains for each request.
- NXDOMAIN attacks can also target a recursive resolver with the goal of filling the resolver’s cache with junk requests.




### Phantom domain attack

- similar result to an NXDOMAIN attack on a DNS resolver.
- The attacker sets up a bunch of ‘phantom’ domain servers that either respond to requests very slowly or not at all.
- The resolver is then hit with a flood of requests to these domains and the resolver gets tied up waiting for responses, leading to slow performance and denial-of-service.



### Random subdomain attack

> attacker sends DNS queries for several `random, nonexistent subdomains` of one legitimate site.

- The goal is to create a denial-of-service for the domain’s authoritative nameserver, making it impossible to lookup the website from the nameserver.
- As a side effect, the ISP serving the attacker may also be impacted, as their recursive resolver's cache will be loaded with bad requests.



### Domain lock-up attack

> Attackers setting up special domains and resolvers to create TCP connections with other legitimate resolvers.

- When the targeted resolvers send requests, these domains send back slow streams of random packets, tying up the resolver’s resources.


---


### Botnet-based CPE attack `attack the CPE device`

> These attacks are carried out using CPE devices
> CPE devices (Customer Premise Equipment; hardware from service providers for use by their customers, such as modems, routers, cable boxes, etc.)

- The attackers compromise the CPEs and the devices become part of a botnet, used to perform random subdomain attacks against one site or domain.



---


### domain name kiting 空头支票 `attack the DNS registration info`

- When a new domain name is issued, technically a five-day grace period before you must pay for it.
- Those engaged in kiting can delete the account within the five days and re-register it, allowing them to have accounts that they never have to pay for.


---


### Hijacking 劫持


#### DNS hijacking `attack the DNS record of website on nameserver`

> attacker redirects queries to a different domain name server.

This can be done either with malware or with the unauthorized modification of a DNS server.
- the result is similar to that of DNS spoofing
- its a different attack because it targets the DNS record of the website on the nameserver, rather than a resolver’s cache.

![Screen Shot 2022-05-19 at 11.39.00](https://i.imgur.com/7okrVmH.png)


#### Domain Hijacking `attack the DNS registration info`

> changing the domain registration information for a site without the original registrant’s permission.

- Once hijacked, often the website is replaced by one that looks identical but that records private information (passwords) or spreads malware.
- Attackers often do so with social engineering techniques to gain unauthorized access to the domain owner’s email account.

Example
- Homer
  - sets up a domain: `homersimpson.com`
  - He uses his Gmail to registers it,
- Attackers
  - watch his Facebook page and notice that he often adds simple comments like “Doh!”
  - try to log on to his Gmail with a brute force attempt.
  - try the password of Doh!Doh! and get in.
  - then go to the domain name registrar, and use the Forgot Password feature.
- It sends a link to Homer’s Gmail account to reset the password.
  - resetting the password at the domain name registrar site,
  - the attackers change the domain ownership.
  - delete all the emails tracking what they did.
- Later, Homer notices his web site is completely changed and he no longer has access to it.



---

### Pharming and Phishing


#### DNS pharming 网域嫁接 `attack the client computer`

> Pharming attacks on the client computer

- modify the `hosts file` used on Windows systems.
  - This file is in the `C:\Windows\System32\drivers\etc\`
  - include IP addresses along with host name mappings.

- By default, it doesn’t have anything other than comments on current Windows computers.
  - a mapping might look like this:
  - 127.0.0.1              localhost
  - 13.207.21.200	         google.com
  - The first entry: maps localhost to the loopback IP address of 127.0.0.1.
  - The second entry: maps google.com to the IP address of bing.com (13.207.21.200).

- If a user enters `google.com` into the address bar of a browser, the browser will instead go to `bing.com`
- if the IP address points to a malicious server, this might cause the system to download malware.


manipulates the DNS name resolution process. It either tries to corrupt the DNS server or the DNS client.
- DNS poisoning attack: redirect users to different web sites,
- DNS pharming attack: redirects a user to a different web site.


- An attacker could cause requests for web sites to `resolve to false IP addresses of his own malicious servers`, leading the victim to view or download undesired content, such as malware.
- One of the main uses of pharming is to resolve a domain name to a web site that appears identical to the requested site, but is instead designed for a malicious intent.


---


#### DNS phishing 网络仿冒

- try to grab usernames and passwords, credit card numbers, and other personal information.
- Victims of a combined **pharming and phishing attack** would have no way of distinguishing between the fake and real sites, since all of the information conveyed by the browser indicates that they are visiting a trusted web site.
- Some types of pharming attacks:

  - **email relies on specialized DNS entries (MX records)**
    - attacker can redirect mail intended for certain domains to a malicious server that steals information.
    - Many online services allow password recovery through email, could be identity theft.

  - **with pharming attacks**
    - associate the domain name used for OS updates with a malicious IP address
    - victims automatically download and execute malicious code instead of a needed software patch.
    - In fact, the possibilities of damage from pharming attacks are nearly endless because of the large degree of trust placed on the truthfulness of domain-name resolutions.
    - DNS compromises can have dire consequences for Internet users.



---




## DNS securtiy

> DNS system was not designed with security in mind and contains several design limitations

DNS security
- protecting DNS infrastructure from cyberattacks in order to keep it performing quickly and reliably.
- An effective DNS security strategy incorporates a number of overlapping defenses, including
  - establishing redundant DNS servers,
  - applying security protocols like DNSSEC,
  - and requiring rigorous DNS logging.


---

### DNS Sinkhole DNS沉洞

![Screen Shot 2022-05-24 at 11.33.04](https://i.imgur.com/PDVDChL.png)

- DNS sinkholing prevents devices from connecting to those bad domains in the first place.
  - DNS sinkhole is a DNS server that supplies a false domain name in response to a DNS query.
  - it supply a false IP address redirects the client elsewhere.
  - it redirects the client device and prevents a connection to a bad domain.
  - By configuring the DNS forwarder to return false IP addresses for specific URLs, it prevents connections to bad or unwanted domains. It is sometimes called blackhole DNS.

- a mechanism aimed at protecting users by intercepting DNS request attempting to connect to `known malicious or unwanted domains` and returning a false, or rather controlled IP address.

- The controlled IP address points to a **sinkhole server** defined by the DNS sinkhole administrator.

Because of its direct consequences, **Sinkholing is usually done in special conditions by trusted third parties** with the involvement of law enforcement.

**DNS sinkhole functionalities**

![dns_sinkhole](https://i.imgur.com/5d5PCvi.png)

- it `prevent hosts from connecting/communicating with known malicious destinations` such as a botnet C&C server (link to Infonote).

  - Sinkholing can be done at different levels.
  - Both ISPs and Domain Registrars are known to use sinkholes to help protect their clients by diverting requests to malicious or unwanted domain names onto controlled IP addresses.
  - System administrators can also set up an internal **DNS sinkhole server** within their organisations infrastructure.
  - A user (with administrative privileges) can also modify the `host file` on their machine and obtain the same result.
  - There are many lists (both open-source and commercial) of known malicious domains that a sinkhole administrator can use to populate the DNS Sinkhole.


- it can be used to `collect event logs`, but in such cases the Sinkhole administrator must ensure that all logging is done within their legal boundaries and that there is no breach of privacy.
  - it can identify compromised hosts by analysing the sinkhole logs and identifying hosts that are trying to connect to known malicious domains.
  - For example if the logs show that one particular machine is continuously attempting to connect to a C&C server, but the request is being redirected because of the sinkhole, then there is a good chance that this particular machine is infected with a bot.

- <font color=blue> Blocking drive-by downloads </font>
  - DNS sinkhole redirects user access to a legitimate website that an attacker has secretly inserted with a malicious hidden link, which forces the client to download and execute malicious code without their knowledge.

- <font color=blue> Blocking C&C channels </font>
  - When a user tries to connect to a C&C server, a referrer can be popped up, which indicates a direct connection to the domain.
  - This is a good indicator that tells the user is being compromised and the bot is attempting to contact the controller for further malicious commands.




**Architecture**

![062414_2128_DNSSinkhole1](https://i.imgur.com/aeTSiJt.png)

The figure illustrates the DNS flows that occur when an attacker compromises a user and this infected user tries to contact a botnet.

1. The DNS sinkhole bypasses the DNS request and provides the response that is configured by the DNS sinkhole administrator.
   1. It doesn’t allow the domain to be resolved by the domain’s authoritative owner.
   2. Instead, the DNS sinkhole intercepts the DNS request and responds with an authoritative answer configured by the organization.

2. when the malware on the infected machine attempts to initiate a connection to a system hosted on a URL with a known malicious domain configured in the DNS sinkhole.
   1. But the request is not passed to the malicious URL.
   2. Instead, it is sent to the sinkhole which in turn responds with an IP of the local host, forcing the client to connect to itself instead of the malicious IP.
   3. The client is unable to contact the malicious site and the command and control connection with the botnet is never established.
   4. The botmaster will be unaware that the compromise has occurred.

3. After this step, the preparation, detection and partial containment are finished.
   1. Containment is partial because the compromised computer may still attempt to attack internal computers.
   2. Therefore, additional analysis and eradication steps should be carried out by the corresponding teams.

![Screen Shot 2022-05-24 at 11.35.28](https://i.imgur.com/i8NYPxY.jpg)

![dns_sinkhole_com_flow_01](https://i.imgur.com/2J8n3vV.png)


**Limitations of DNS sinkholing**

- To block malware or its traffic by using a DNS sinkhole, it is required by the malware to use the organization’s DNS server itself.
  - Malware with its own hardcoded DNS server and IP address cannot be detected by the DNS sinkholing mechanism.
  - this drawback can be mitigated by: `using perimeter firewalls configured to block all other outbound DNS queries rather than the organization’s DNS servers.`

- A DNS sinkhole cannot prevent malware from being executed and also being spread to other computers.
- by using a DNS sinkhole, malware cannot be removed from an infected machine.

- A DNS sinkhole will be put in with the indicators of the malware, and these indicators should be analyzed beforehand.
  - Also, the malicious IP information gathered from open sources that are to be given into the DNS sinkhole may contain false positives.
  - The sources may contain a URL that is not malicious, and hence it will result in an unwanted restriction to legitimate websites.

- A DNS sinkhole should be isolated from the external network
  - so that the attacker cannot be informed of the fact that their C&C traffic has been mitigated.
  - Otherwise, it results in a reverse effect where attackers may manipulate the entries in the DNS sinkhole and use them for malicious purposes.

- DNS records should be implemented with time-to-live (TTL) settings with short values,
  - or it may result in users caching the old data for a longer period.


**Implementation**


- https://knowledgebase.paloaltonetworks.com/KCSArticleDetail?id=kA10g000000ClGECA0
- https://docs.paloaltonetworks.com/pan-os/9-1/pan-os-admin/threat-prevention/use-dns-queries-to-identify-infected-hosts-on-the-network/dns-sinkholing
- https://www.netacea.com/glossary/dns-sinkhole/





ref:
- https://www.enisa.europa.eu/topics/csirts-in-europe/glossary/dns-sinkhole
- https://www.sans.org/white-papers/33523/


---


### DNSSEC - Domain Name System  Security Extensions

- a suite of extensions to DNS
  - security specifications for security DNS,
  - provides validation for DNS responses.

- involves <font color=red> many security features </font>
  - like **digitally signed DNS responses**.
  - mitigate the risk of DNS attacks: like DNS poisoning.
  - Provides **cryptographic authenticity of responses** using `Resource Record Signatures (RRSIG)`
  - and authenticated denial of existence using `Next-Secure (NSEC) and Hashed-NSEC records (NSEC3)`.

- Guarantees:
  - Authenticity of DNS answer origin
  - Integrity of reply
  - Authenticity of denial of the existence

- DNSSEC protects against attacks by `digitally signing data to help ensure its validity`
  - adds a digital signature to each record and provides integrity.
  - to ensure a secure lookup, the `signing must happen at every level` in the DNS lookup process.
    - signing DNS replies at each step of the way
  - Uses `public-key cryptography` to sign responses
    - typically use trust anchors, and entries in the OS to bootstrap the process
    - This is also related to DNS resolution.
      - When the DNS resolution process is sent in clear text, vulnerable to packet sniffing.
      - Therefore, DNS resolution should also be secured/encrypted.
  - If a DNS server receives a DNSSEC-enabled response with digitally signed records, the DNS server knows that the response is valid.


- BEST supports the deployment of DNSSEC at the organization: TLS


**signing process**
- similar to someone signing a legal document with a pen; that person signs with a unique signature that no one else can create, and a court expert can look at that signature and verify that the document was signed by that person.
- These digital signatures ensure that data has not been tampered with.
- DNSSEC implements a **hierarchical digital signing policy** across all layers of DNS.
  - For example
  - a ‘google.com’ lookup,
  - a root DNS server would sign a key for the `.COM nameserver`,
  - and the .COM nameserver would then sign a key for `google.com’s authoritative nameserver`.

- DNSSEC is designed to be backwards-compatible to ensure that traditional DNS lookups still resolve correctly, albeit without the added security.
- DNSSEC is meant to work with other security measures like SSL/TLS as part of a holistic Internet security strategy.

- DNSSEC creates a parent-child train of trust that travels all the way up to the root zone.
  - This chain of trust cannot be compromised at any layer of DNS, or else the request will become open to an on-path attack.

- To close the chain of trust, the root zone itself needs to be validated (proven to be free of tampering or fraud), and this is actually done using human intervention. **Root Zone Signing Ceremony**, selected individuals from around the world meet to sign the root DNSKEY RRset in a public and audited way.




---

### operator of a DNS zone

- an operator of a DNS zone can take further measures to secure their servers.
- Over-provisioning infrastructure is one simple strategy to overcome DDoS attacks.
- Simply put, if your nameservers can handle several multiples more traffic than you expect, it is harder for a volume-based attack to overwhelm your server.
- Organizations can accomplish this by increasing their DNS server's total traffic capacity, by establishing multiple redundant DNS servers, and by using load balancing to route DNS requests to healthy servers when one begins to perform poorly.





---

### DNS firewall

- sits between a `user’s recursive resolver` and the `authoritative nameserver of the website or service` they are trying to reach.

provide a number of security and performance services for DNS servers.
- provide **rate limiting services** to shut down attackers trying to overwhelm the server.
  - If the server does experience downtime as the result of an attack or for any other reason, the DNS firewall can keep the operator’s site or service up by serving DNS responses from cache.

- provide **performance solutions** such as
  - faster DNS lookups
  - reduced bandwidth costs for the DNS operator.


---


### DNS resolvers

- DNS as a security tool

- **DNS resolvers** can also be configured to provide security solutions for their end users (people browsing the Internet).

- Some DNS resolvers provide features such as `content filtering`, block sites known to distribute malware and spam, and botnet protection, blocks communication with known botnets.

  - Many of these secured DNS resolvers are free to use
  - user can switch to one of these recursive DNS services by changing a single setting in their local router.


---


## DNS privacy

- DNS queries are not encrypted.
- Even if users use a DNS resolver like 1.1.1.1 that does not track their activities, DNS queries travel over the Internet in plaintext.

This lack of privacy has an impact on security and human rights;
- means anyone who intercepts the query can see which websites the user is visiting.
- easier for governments to censor the Internet and for attackers to stalk users' online behavior.

`DNS over TLS` and `DNS over HTTPS` are two standards for encrypting DNS queries in order to prevent external parties from being able to read them.


---


### DNS over HTTPS

Normally when you request a domain name such as google.com
- you send out an `unencrypted DNS request` to find the IP Address the domain resolves to (so the machine can connect to the domain.)
- A relatively modern replacement for **traditional DNS** is **DNS over HTTPS (DoH)**.
  - DoH encrypts `DNS queries`, and sends the requests out as regular HTTPS traffic to DoH resolvers.


Using DoH
- a fairly unusual choice for the Denonia authors
- provides two disadvantages here:
  - AWS cannot see the dns lookups for the malicious domain, reducing the likelihood of triggering a detection
  - Some Lambda environments may be unable to perform DNS lookups, depending on VPC settings.





---
