---
layout:	post
title:	"Csaw18 — get it : Buffer overflow to redirect the return flow"
date:	2021-07-11
medium_url: https://noob3xploiter.medium.com/csaw18-get-it-buffer-overflow-to-redirect-the-return-flow-7d9fb5f25e96
categories: [Hacking, CTF]
tags: [Hacking, CTF, Reverse Engineering, Binary Exploitation]
---


  In this writeup, we will be solving the csaw18 get it challenge. Here, we will overwrite the return address to redirect the flow of the program to any function we want. Lets start

First i opened up the binary file in ida and go to the main function

![](/img/1*oFyL9FsN-EzUpOV0EyOd7g.png)

We can see that it simply get our input using gets and take var\_20 as an argument. We know that gets is vulnerable to buffer overflow. Looking at the functions, we see an interesting function called give\_shell

![](/img/1*-kO9WKjh3a3QmHCJFnXaCg.png)

What it does is it simply give us a shell. So what we want to do is, we want to overwrite the return address in the stack in main function and set it to the address of give\_shell to control the flow of the program.

Since our program is a 64 bit program, that means that the return address is 8 bytes above the ebp. And our vulnerable buffer var\_20 is 0x20 bytes below ebp. With those information in mind, we can calculate the bytes needed to overwrite the return address and control our program

![](/img/1*04oP2wiLFwo8EZR3yAnxfA.png)

We need 40 bytes of data to overwrite the return address so lets write our exploit

![](/img/1*wRmnU7jMccTO0YdpKfV5ig.png)

We can see in our exploit script that we first put 40 bytes of A then the returnaddr variable. returnaddr is the little endian 64 bit address format of the function give\_shell. So lets try it out

![](/img/1*BxA5fvaf7X5caM-Aqf0Gfg.png)

And it works, thanks for reading

  