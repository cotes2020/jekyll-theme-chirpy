---
layout:	post
title:	"Bookwyrm Server Side Request Forgery"
date:	2022-02-14
medium_url: https://noob3xploiter.medium.com/bookwyrm-server-side-request-forgery-b1462829d68e
categories: [Hacking, Code Review]
tags: [Hacking, Django, Code Review, Web]
---

  While reading the code of bookwyrm, i encounter this endpoint

![](/img/1*GnNMAvVLVgvcTObyL2jS7w.png)This endpoint calls the function views.upload\_cover.

![](/img/1*53FS9Ng7EEpzeidX_BJCCQ.png)You can see that it accepts a post request from the decorator above, and it requires authentication, which is not a problem since bookwyrm allow self registration. In the function, you can see that it takes a post parameter cover-url, and pass it through the function set\_cover\_from\_url, then save the the cover of the book based on the returned value of set\_cover\_from\_url.

![](/img/1*k-97ATIN11ipYgDQr7WNvw.png)In set\_cover\_from\_url, it pass the url argument to the function get\_image. Then, it generates a random image\_name and return the image name and the content from the get\_image function.

![](/img/1*VOGMxY8zlPvr6kHy8zo16g.png)In theget\_image function, it pass our url argument to requests.get and return the response. Making it vulnerable to ssrf.

So to test it out, i hosted my own bookworm instance and tested out the bug.

![](/img/1*rtlmM6TaeleqUQHbHKGYFA.png)I put `http://172.18.0.1/`` in the cover-url which is the internal ip of my vps since my bookwyrm instance is hosted with docker. As you see, it responded with a redirect, just like what we expected

Now when we visit the cover of book 2, we will see this.

![](/img/1*UWas9O4pvBn4WF-u4Pwagg.png)I host a wordpress instace on my vps and the response show a wordpress instance so this verifies that our ssrf works.

I tried achieving lfi with it, unfortunately, by default, requests.get of python doesnt accept the file:// scheme.

This is the end of the writeup, thanks for reading. Thank you to the maintainer of bookwyrm for being cooperative and responsive.

Join The Bounty Hunter Discord Server: <https://discord.gg/bugbounty>  
Add me on Twitter: [https://twitter.com/tomorrowisnew\_\_](https://twitter.com/tomorrowisnew__)

  