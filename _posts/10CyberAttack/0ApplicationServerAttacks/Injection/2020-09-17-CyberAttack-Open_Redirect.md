---
title: Meow's CyberAttack - Application/Server Attacks - Injection - Open redirect
date: 2020-09-17 11:11:11 -0400
categories: [10CyberAttack, Injection]
tags: [CyberAttack, Injection]
toc: true
image:
---

- [Meow's CyberAttack - Application/Server Attacks - Injection - Open redirect](#meows-cyberattack---applicationserver-attacks---injection---open-redirect)
	- [Open redirect](#open-redirect)

book: S+ 7th ch9

---

# Meow's CyberAttack - Application/Server Attacks - Injection - Open redirect

---

## Open redirect

- a security flaw in an app or a web page that causes it to <font color=LightSlateBlue> fail to properly authenticate URLs </font>.

- When apps and web pages have requests for URLs, they are supposed to verify that `those URLs are part of the intended page’s domain`.

- Open redirect is a failure in that process that makes it possible for attackers to steer users to malicious third-party websites.

- Sites or apps that fail to authenticate URLs can become a vector for malicious redirects to <font color=OrangeRed> convincing fake sites for identity theft or sites that install malware </font>.

- Normally, redirection is a technique for shifting users to a different web page than the URL they requested. Webmasters use redirection for valid reasons, such as dealing with resources that are no longer available or have been moved to a different location. Web users often encounter redirection when they visit the Web site of a company whose name has been changed or which has been acquired by another company.

- The `Heartbleed` vulnerability, originally reported to be enabled by convert redirects, was eventually discovered to be the result of the less serious -- but still irresponsible -- enabling of open redirect.
