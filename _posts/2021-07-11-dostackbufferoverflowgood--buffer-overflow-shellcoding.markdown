---
layout:	post
title:	"dostackbufferoverflowgood: buffer overflow shellcoding"
date:	2021-07-11
medium_url: https://noob3xploiter.medium.com/dostackbufferoverflowgood-buffer-overflow-shellcoding-411c4369a5ca
categories: [Hacking, CTF]
tags: [Hacking, CTF, Reverse Engineering, Binary Exploitation]
---

  Hi. This is the third writeup on my buffer overflow series. In this writeup, we will exploit buffer overflow and achieve remote code execution with shellcodes. Here, we will slightly use a debugger sometimes. Lets get started

First, we load up the binary file in ida and inspect strings in that function. One of the strings that i found is 31337

![](/img/1*DcIg_o9Sn1ir9v31mzNRsA.png)

I guessed that this is the port the program will run on. Lets try it out.

![](/img/1*3hs5I9aoxA3uSHGVSaJwIA.png)

Looks like our guess was right. So now we know the port this program run on. I also found this interesting string

![](/img/1*YQr_vwsVdnHxsCORGVAL0w.png)

Lets look at where this string is referenced to using cross references. I found that this string is referenced in a function called \_doResponse

![](/img/1*AmN63MJZAtCbEsvqkwAWZQ.png)

We can see that it use sprintf to format the string and the user input. Now as we know, sprintf is vulnerable to buffer overflow. So this is probably the vulnerable function. We can see that it use the format string Hello %s!!!\n and use the user input clientName and store it to the variable response. We can see that the varibale response is **0x94** bytes below the ebp. **0x94** is 148 in decimal. We want to overwrite the return address is and the return address is 4 bytes above the ebp

![](/img/1*2m-RujUvYe9S02nGoERRJg.png)

So in theory, if we sent **0x94**+**0x4** bytes of data, we can overwrite the return address and control the eip. However, theres more, we can see that in the sprintf function, we use the string Hello %s!!!\n to format. This string is also used in the sprintf so we have to take care of them too. The word before our input is Hello and we can see that takes up 6 bytes. So in theory, if we sent **0x94**+**0x8**–**0x6** bytes of data, we can overwrite the return address and control the eip. I made a python script to test it out

![](/img/1*u7uzuTb61gkzZEpgziWwYw.png)

We can see that we are right, we successfully controlled the eip. If we make our buffer longer, we can see that our input is also stored in the esp

![](/img/1*3qY_0jLZRxln-pRvvJ38VQ.png)

So what we need to do is find a jmp esp instruction and point our eip in there, then the program execution will be redirected to esp, then we will store our shellcode in the esp to be executed and we will get a remote code execution.

So first, finding the jmp esp, the opcode of jmp esp is ff e4

![](/img/1*jBi9KPE7X9rB38Nr5AjBtA.png)

So lets find it in ida.

![](/img/1*2EYSawjpI6tVf8Dpd0Knag.png)

We can see that we got two hits. Both of these works so i will just choose the first one. Lets take note of its address and put it in our exploit. And also, lets add some nops(\x90) and \xcc opcode to test if it will actually redirect to our esp. \xcc is an opcode in x86 called int or interrupt. It will what that will do is it will pause our debugger. Lets try it out

![](/img/1*CBgzZBoqQ9sbtGxcdGoGzw.png)

This is what our exploit code looks like now. Lets run in and debug it to see if we will hit the \xcc interrupt opcode

![](/img/1*AjV95Na6OKbEocRsv90twA.png)

We can see that it does. Now all we need to do is replace \xcc with our shellcode. I will use a shellcode i found in exploit-db <https://www.exploit-db.com/shellcodes/48116>. So i quickly modified my exploit

![](/img/1*6g4cEjAr4WFPXcX02O9Nqw.png)

Now, the moment of truth, we will run this and we should see a calculator popup.

![](/img/1*VmBSb5OYDKnIVkuKM2Ublg.gif)And it works.

There you have it, buffer overflow writeup with shellcode. Thanks for reading

  