---
layout:	post
title:	"Hacking the Tenda AC10–1200 Router Part 2: Strcpy Buffer Overflow"
date:	2021-08-07
medium_url: https://noob3xploiter.medium.com/hacking-the-tenda-ac10-1200-router-part-2-strcpy-buffer-overflow-92cd88e1d503
categories: [Hacking, Iot]
tags: [Hacking, Iot, Reverse Engineering, Binary Exploitation]
---

  Hi. This would be another series of writeup where we will try to hack the tenda ac10 1200 and try to get a cve. This writeup is fairly short so lets get started

While looking through the functions of tenda, i found this one interesting function saveParentControlInfo

![](/img/1*gKdC4sjzgiaNLodutRXLIw.png)

What made this function interesting is this.

![](/img/1*SduIxapOUbYdH2PrJX52xA.png)

We can see here that it get the value of the post parameter urls using websGetVar then save its value to the variable `var_3bc`. If we follow this variable, we will see this.

![](/img/1*slAbQ7RwQTIBmJD9_ebTxA.png)

We can see that it is used as an argument to strcpy. Now as we all know, strcpy is vulnerable to buffer overflow. So lets try it our, if we send a a long string in the urls parameter, the server should crash. But first, we have to find out the vulnerable endpoint.

Looking at the cross references to saveParentControlInfo, i saw a cross references to formDefineTendDa

![](/img/1*bnKxX_mDSVsetW1SqzkTzw.png)

That means, our vulnerable endpoint is saveParentControlInfo, so now, we can test it out.

I tried it out in burpsuite and saw this,

![](/img/1*IXhhV1eDaMI1yPqgo5ZSJQ.png)

“errCode”:1 , that is not what were expecting, lets find out why that happened. After reversing the function once again, i found the errCode: 1

![](/img/1*1mGYtw6aXMlabdTGTa-hVg.png)

We can see the error code string there. Now lets see what causes it to jump there.

![](/img/1*jtnZunIsPzXSVue_cEjEBw.png)

Here, we can see that it checks if the `var_3b4` is equals to zero, if it is, it will jump to the errCode. if we follow this `var_3b4`,

![](/img/1*fUGrdEJcR5bvhmLQfTu-1A.png)

We can see that it is the output of websGetVar with time parameter. In our last attempt, we didnt send a time parameter so it is equal to null which caused the jump to the errCode. Lets try it again but this time, lets provide a time parameter

![](/img/1*9gc9Mb8QwZGY_Cvlbkm21Q.png)

No response, lets try it again

![](/img/1*jpYZp9tOs88cj1jg3H8PIw.png)

Failed to connect. That means we successfully crashed the web server. We can confirm it even more by looking at the emulation

![](/img/1*uUftddhYij_ZUU9pL-_U_w.png)

So we have a buffer overflow confirmed.

Sadly, we cant overwrite the program pointer since this is a heap overflow. If we go back to the vulnerable strcpy

![](/img/1*sNfOkiWIoXgb4aUXYXUo2A.png)

We can see that in the first argument ***$a0***, it uses the variable `var_3d8 + 0x50`. If we trace back this `var_3d8`, we can see that it is the output of malloc

![](/img/1*zgjy2DFaMUtJyMENlE8UNQ.png)

meaning, it is pointing to the heap, not the stack, thats why we cant overwrite the program pointer with what we want. However, with heap overflow, we can overwrite the other data in the heap. But i will end the writeup now.

Other parameter is also vulnerable like deviceId and time but i didnt talked about them since it is already reported and is already a cve [CVE-2020–13393](https://joel-malwarebenchmark.github.io/blog/2020/04/28/cve-2020-13393-Tenda-vulnerability/). This one is not a cve yet tho so this is the one that i focused in this writeup.

I tried contacting tenda but they didnt responded so i decided to publish this writeup now.

Thanks for reading.

Join the discord server: <https://discord.gg/bugbounty>

  