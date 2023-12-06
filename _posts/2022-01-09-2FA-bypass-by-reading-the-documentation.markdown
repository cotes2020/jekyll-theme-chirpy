---
layout:	post
title:	"2FA bypass by reading the documentation"
date:	2022-01-09
medium_url: https://noob3xploiter.medium.com/2fa-bypass-by-reading-the-documentation-3260a372d8a8
categories: [Hacking, Bug Bounty]
tags: [Hacking, Bug Bounty, Web]
---

  This is a fairly simple and short writeup, but i think is worth sharing, so lets get started.

This program is private so i will be redacting most of the information from it.

Like any other website, my program has a 2fa implemented, and their implementation is pretty good too. So i started reading the documentation. Most of the api functions requires api key for authorization

![](/img/1*bI2wElWlBszamW-a2mOhtg.png)

And this api key can be only obtained in the web client after logging in which require a 2fa verification. However, while reading other api functions, i found one odd api method.

![](/img/1*3JSThgMJ1k7pXQ_i0fgxMQ.png)

Unlike the other api methods, it doesnt use the api key for authorization. Instead, it uses a basic authentication stated by the -u and only requires the email and the password . After trying it out myself, the request succeeds without the 2fa verification.

This is fixed now and is accepted as low since it requires knowing the credentials of the target and only one api method is vulnerable but still interesting for me nonetheless.

![](/img/1*gUFHILgVEGZgeMPioeiFwA.png)!
[](/img/1*irhoxiweHLqo-Qv4rO-mag.jpeg)

Thanks for reading, Join the Bounty Hunter Discord Server: <https://discord.gg/bugbounty>