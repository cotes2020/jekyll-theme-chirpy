---
layout:	post
title:	"Pwnable.kr — bof : Basic buffer overflow"
date:	2021-07-11
medium_url: https://noob3xploiter.medium.com/pwnable-kr-bof-basic-buffer-overflow-cf9579f4c2f3
categories: [Hacking, CTF]
tags: [Hacking, CTF, Reverse Engineering, Binary Exploitation]
---

  This is my writeup of the bof challenge from pwnable.kr . In this writeup, we will not rely on debuggers and we will not read the source code. We will just simply reverse the binary because we can learn more from it. Lets get started

I got the file and opened it up in ida.

![](/img/1*HidceHDw4ymAEv9GQpU2zg.png)

We can see that what the main function does is that it calls the function func with the argument 0xDEADBEEF.

![](/img/1*6Nj0nMB9G9Aj-T4tZ4UZIg.png)

In the func function, we can see that it get the user input using the function gets and store the user input to the variable var\_2c. Then, the argument is compared to the hex value 0xCAFEBABE, if it is equal, it will call the system with /bin/sh as the argument giving us a shell, if not, it will puts Nah..

Commonly, there’s no way we can change the argument(arg\_0), however, we can see that it uses the function gets to get our user input, gets is known to be vulnerable to buffer overflow, so what we can do is supply a long enough input to overwrite the value of arg\_0

If we view the variables, we can see that our vulnerable buffer(var\_2c) is 0x2c or 44 bytes below the ebp and arg\_0 is 8 bytes above the ebp. To computer how many bytes we need to overwrite the arg\_0, we will use python

![](/img/1*Tf_PwI02zF3ZXwplnLnYeg.png)

We can see that the bytes between is var\_2c and arg\_0 is 52 bytes. So in theory, if we send 52 chars and 0xcafebabe, we can overwrite the arg\_0 with 0xcafebabe and the compare will be true and we will get a shell. Lets try it out

![](/img/1*pCpz-l7E2A65SSn8PmwWGA.png)

And it works. We are successful. For those who dont know, our program is little endian so we have to format 0xcafebabe to little endian format. Also, you can see that i used cat, it is because without it, the program will just stop and we will not recieve the shell. You can learn more about it in here [https://www.youtube.com/watch?v=yH8kzOkA\_vw](https://www.youtube.com/watch?v=yH8kzOkA_vw)

Thats the end of my writeup, thanks for reading

  