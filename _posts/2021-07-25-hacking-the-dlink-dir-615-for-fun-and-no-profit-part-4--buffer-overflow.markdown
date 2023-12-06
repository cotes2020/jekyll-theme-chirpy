---
layout:	post
title:	"Hacking the dlink DIR-615 for fun and no profit Part 4: Buffer Overflow"
date:	2021-07-25
medium_url: https://noob3xploiter.medium.com/hacking-the-dlink-dir-615-for-fun-and-no-profit-part-4-buffer-overflow-f278ecfdb3c4
categories: [Hacking, Iot]
tags: [Hacking, Iot, Reverse Engineering, Binary Exploitation]
---

  Hi. This is my 4th writeup in the hacking the dlink dir 615 series where im trying to get my first cve. Lets get stared.

While looking again on all functions that accept user input, i found this interesting function. `sub_40e148`

![](/img/1*ynv0ouPckfxW6RYE-W0Xug.png)

What this function does is it get the value of the post parameter `ping_ipaddr` using `get_cgi` then store it to a global variable ping

![](/img/1*oPCBxPNO6-UQYWu4ZFbwVg.png)

Then i looked through all the cross references to the global variable ping and found a function called `return_ping_result`

![](/img/1*7lM11ej2oYr1LWeuQrPXcw.png)

There, it loads the value of the global varibale ping to ***$v0***, then use ***$v0*** as an argument to `get_ping_app_stat`. So i checked the `get_ping_app_stat`.

![](/img/1*Ot7wCQwEkHLTkayjA0Eo2A.png)

You can see that it moves our argument(ping) to register ***$s0***, then pass it as an argument to `parse_special_char`, then pass it again as an argument to `_system`. You might be thinking that it is vulnerable to command injection but its not. Believe me, this guy tried it <https://limbenjamin.com/articles/dlink-routers-ping-function.html>

Its not vulnerable to command injection because of the `parse_special_char`. This `parse_special_char` function is an external function. So i have to find which library contains this function.

I used grep to find which libraries and programs have the string `parse_special_char`. And found `libproject.so`

![](/img/1*mwDOacUYrqQQTe4yocwfHw.png)

So i opened up libproject.so in binary ninja and start reversing it. This `parse_special_char` is complicated but i’ll try to explain it as much as i can

![](/img/1*F15s_28Amx4xGWdsOF1zLg.png)

First, it store our input to to the register ***$s2***.

![](/img/1*GdcQKe7lt8KyBXL8sqcYjA.png)

Then, it get the length of our input using strlen and store it to ***$s1***

![](/img/1*3N1SNuK_H8dfldIotMFUcg.png)

Then, it copy our input to a variable i called `input_copy` using `strcpy`

![](/img/1*heJav5qelmWVEvxWPAXvSQ.png)

Then it store, a variable called `var_270` to register ***$a0***

![](/img/1*bKZ7JdGlv_F3nXtZsrm9Ww.png)

Then, it load ***$s1***(output of strlen) to ***$a1***. Then it also store some strings to various registers . Then it store the variable input\_copy to ***$a2***

![](/img/1*ZMC3ZWIyvNVVNT3v6-n7WA.png)

Then, it load the byte in ***$a2***(input_copy) to ***$v1***. Then, it moved ***$v1*** to ***$v0***. Then it checks if ***$v0*** is equals to the strings in the registers ***$t0***, ***$a3***, ***$t1***, ***$t2***. These registers contains the string “\””, “\\”, “`”, “$” respectively

If it does, it will jump here

![](/img/1*Nx2daejchF70-YNqptuRrQ.png)

Here, it store the loaded byte in input\_copy(***$a2***) to ***$v0***. Then, it set the loaded byte in ***$a0***(var\_270) to the string ‘\\’. Then, it deduct 1 to ***$a1*** which is the output of strlen. Then, it add 1 to ***$a0*** meaning, it will load the next byte in var\_270. Then, it set the loaded byte in ***$a0***(var\_270) to the loaded byte in ***$v0***(input\_copy). Then it add 1 into ***$a2***(input\_copy). Meaning, it will load the next byte in the variable input\_copy. Then, it checks if ***$a1***(output of strlen) is equals to zero. If it does, it will jump to the end of the function. If not, it will loop back again. But before that, it add 1 into ***$a0***(var\_270) meaning, it will load the next byte in the variable var\_270

If the ***$v0*** is not equals to the strings in the registers, it will jump here instead

![](/img/1*d3kv646G4WMjS6WClUmyrw.png)

Like before, it deduct 1 to the output of strlen. Then, it set the byte loaded in ***$a0***(which is var\_270) to ***$v1***(which is the loaded byte in input\_copy). Then, it add 1 in ***$a2***(which is input copy), meaning, it will load the next byte in input\_copy. Then, it check again if ***$a1***(output of strlen) is equals to zero. Before that, it adds 1 to ***$a0***(which is var\_270), meaning, it will load the next byte in the var\_270 variable. Then it loop back again until ***$a1***(output of strlen) is equals to zero.

You can see that it just copy bytes blindly from input\_copy to var\_270 even if we are copying bytes larger than the buffer size of var\_270 can handle, meaning we have a buffer overflow vulnerability here. So if we provide an input with lenght larger than var\_270 can handle, we can overflow the buffer and overwrite the other variables in the stack. Now, we just have to calculate the bytes between var\_270 and the saved return address.

var\_270 is stored in the offset $sp+0x18 and the saved return address is stored in the offset $sp+0x280. ***$sp*** is -0x288

![](/img/1*sO1Kqe_MOgcIXGAfhRDk3g.png)

If we calculate it -0x288+0x280 is equals to -8 so the saved return address is at stack offset 8. var\_270 is -0x288+0x18 is -624 so it is in the offset 624. 624–8 is equals to 616, meaing, there has 616 bytes between var\_270 and saved return address. Lets try it out. But first, we have to find a way to call the function return\_ping\_result.

In the directory, tools\_vct.asp, there has a function called CmoGetStatus(“ping\_result”);. What it does is it call the function return\_ping result. So now, we have all we need to exploit this.

![](/img/1*V74IDXnI8YjYNfcd55UCrg.png)

I made a python exploit script to test it out.

![](/img/1*rYH3_1LjYv30pRV3kLRwVw.png)

What it does is first, it login with the default creds. Then, it make a post request to ping\_response.cgi with the vulnerable parameter ping\_ipaddr containing our exploit. Then, it make a get request to tools\_vct.asp to call the function return\_ping\_result. In the exploit, i put there 616 bytes of A’s and 4 bytes of B’s. Now with this, we should be able to overwrite the return address and the program counter to 42424242 or BBBB. Lets try it.

![](/img/1*iSplcoN53ytVri_f2Dzz0A.png)

It works. We managed to overwrite the return address and epc(program counter) to 42424242. Now, we have a buffer overflow

I am not good at exploiting buffer overflows. I dont know what and how to do it so i havent escalated this buffer overflow yet, but im trying. If you have experience on mips bof and you think you can help me, please reach me out. I need help in this. But right now, we have a bof that can result to dos.

This is the end of my writeup. After googling, i found out that this is not a cve yet so i requested a cve for it. If it gets accepted, this will be my first cve. I have high hopes for this. This bug is already fixed in the newest version of the firmware so youre safe as long as you update your router’s firmware.

Update: It is now confirmed and got my first cve with this. <https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2021-37388>.

Thanks for reading.

Join the discord server: <https://discord.gg/bugbounty>

You can reach me out here if you think you can help me with this bof:

Discord: sonics the name speeds the game#5389

twitter: [https://twitter.com/tomorrowisnew\_](https://twitter.com/tomorrowisnew_)

  