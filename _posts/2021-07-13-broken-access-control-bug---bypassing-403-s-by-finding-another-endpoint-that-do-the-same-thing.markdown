---
layout:	post
title:	"Broken Access control bug : Bypassing 403’s by finding another endpoint that do the same thing."
date:	2021-07-13
medium_url: https://noob3xploiter.medium.com/broken-access-control-bug-bypassing-403s-by-finding-another-endpoint-that-do-the-same-thing-b5238386ce58
categories: [Hacking, Bug Bounty]
tags: [Hacking, Web, Bug Bounty]
---

I found a really interesting bug in my private program and i want to share it through this writeup. Lets get started.

I was testing all the functionalities of this website and found this one interesting request when editing the residents

![](/img/1*PW9fo2Blnni7Zgl2jhwTMg.png)

We can see in the response that there is an interesting parameter called moved in. I tried including it in the request and setting it to true hoping it would change the value of that parameter and it works

![](/img/1*FhogQ8nvFAW9W-dDoBpHEA.png)

So now, we can move in/move out residents if we have an update permission. Normally, that wouldnt be a bug, but in this program, editing residents and moving in/moving out are on different permissions. I tried it again but this time, i remove the move in/move out permission

![](/img/1*890KVYhO5ryJW5ZacmgVTw.png)

And it still works. It still allows me to move in/move out residents if we have update permission. Normally, moving in/moving out users is done in a different endpoint, if we tried it out, it will not work, because we have no permission to move in/move out as expected.

![](/img/1*wsDgZHPUrodfM6nrfFPQaQ.png)

So this is a neat little bypass.

Thanks for reading.

Join the discord server: https://discord.gg/bugbounty

  