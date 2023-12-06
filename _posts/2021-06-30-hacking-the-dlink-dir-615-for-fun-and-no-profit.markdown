---
layout:	post
title:	"Hacking the dlink DIR-615 for fun and no profit"
date:	2021-06-30
medium_url: https://noob3xploiter.medium.com/hacking-the-dlink-dir-615-for-fun-and-no-profit-a2f1689f9920
categories: [Hacking, Iot]
tags: [Hacking, Iot, Reverse Engineering, Binary Exploitation]
---

  Hello . In this writeup, i will show you how i found a potential remote code execution (CVE-2019–13561) in the dlink dir-615 firmware.

I Started up by downloading the firmware in the official site of dlink <https://www.dlink.com.ph/dir-615/> . I downloaded the second firmware instead of the first since the first is the latest and is probably more secured. The file is a rar file, i simply unrar it and ran binwalk on it. And i managed to get the linux file system

![](/img/1*Lw9M0ouKQi6pBlPFQ9npJg.png)

binwalkMy next goal is finding out what web server this firmware is using to reverse engineer it, so i used the command tree | grep httpto find out potential http server binary files and i found one called httpd.

![](/img/1*0nSTmLSG-hVkckw6CtZcJg.png)

finding httpdAnd i found this file in the /sbin directory. Now time for reverse engineering

After running through the functions of httpd, i found an interesting function called ***return\_online\_firmware\_check***. What makes it interesting is that it has a call to the function ***\_system****.*

![](/img/1*_sjmw8xp-28vy-rZpAGmIA.png)

system callThe ***\_system*** is like a mixture of ***sprintf*** and ***system***. It mixes string formatting and system into a one function. Upon looking more into the disassembly

![](/img/1*y9pQ57iPHkoTXCkCystx7A.png)

We can see that the ***\_system*** takes 3 arguments, in the register ***$a0***, ***$a1*** and ***$a2***. We can see that in the first argument(***$a0***), it takes a string that is equal to fwqd -i %s -u %s* , *this is the code that will be executed by the ***\_system***. We can see in the string that it has two %s and like i said, ***\_system*** is like ***sprintf*** and system combined into one function so this string will be used for string formatting

![](/img/1*LMIaOexTGt0GDCgo68lN_A.png)

We can see that the value of the second argument of ***\_system***(***$a1***) is equals to the value of ***$v0***. The ***$v0*** register holds the return value of a function call. The last function call made is from ***nvram\_safe\_get*** so we can assume that the value ***$a1*** is equal to the return value of the ***nvram\_safe\_get***. I am not 100% sure what the ***nvram\_safe\_get*** do, it is an external function so icant reverse it and searching it on google is not giving me any useful result. But I will just assume that it gets the value of the variable given to it. We can see that before the ***nvram\_safe\_get*** call, the string ***wan\_eth*** is moved to the first argument(***$a0***) so that probably means this will get the value of the variable ***wan\_eth***. So now, we can assume that the value of the second argument(***$a1***) of the ***\_system ***call*** ***is the value of the variable ***wan\_eth ****from the nvram*.

![](/img/1*Luo3klWcFylEYwoCgf1_fQ.png)

Next up, the third argument(***$a2***), we can see that it takes the value of the register ***$s0***. Going back, we can see that the value of the ***$s0*** is equal to the value of the ***$v0***. I already explained it above so i will just summarize it in this argument. So here, the value of the third argument(***$a2***) is equal to the returned value of the function ***nvram\_safe\_get*** with an argument, ***check\_fw\_url***. So this probably means that the third argument(***$a2***) is equal to the value of the variable ***check\_fw\_url ****from nvram*.

With those all parameters we have, we can now assume that the ***\_system*** call is equal to \_system(“fwqd -i %s -u %s”,nvram\_safe\_get(“wan\_eth”),nvram\_safe\_get(“check\_fw\_url”)).

The value of the ***wan\_eth*** is constant, it is the value of the wan address of the router and we cant control it, however, we might be able to control the value of ***check\_fw\_url***. If we can change the value of the ***check\_fw\_url*** to ;reboot; and we can achieve code injection since it will get passed to ***\_system***. However, i cant figure out where we can change the value of ***check\_fw\_url***. If only i have the dir-615 router with me, i can try to find out where i can modify the value of this variable. But until then, this remote code execution bug is still a theory…………. **OR IS IT**

You see, even if we cant confirm the code injection bug, i found a cve similar to it. After searching the string ***check\_fw\_url*** in google, it gives me a link to CVE-2019–13561

![](/img/1*d4aSORNVndZe05g-gnO8Cg.png)

Here, we can see that he found a code injection in dlink dir-655 in the ***online\_firmware\_check.cgi***, in the ***check\_fw\_url***. If you can remember, the function that we are reversing is called ***return\_online\_firmware\_check***

![](/img/1*43qVkwvxJD3ENacitS5TOA.png)

And the vulnerable variable that we are inspecting is the ***check\_fw\_url*** which matches in his cve. I guess the ***online\_firmware\_check.cgi*** is just a simple cgi script that calls the function ***return\_online\_firmware\_check***. Again, i cant confirm it since i dont have the dir-615 with me.

And there you have it, my writeup of a potential code injection vulnerability in the dlink dir-615/analysis of the cve 2019–13561. I found this bug in the 2009 version of the firmware of dir-615 so as long as your firmware is updated, you should be safe from it. Thanks for reading.

PS. I assumed that what ***nvram\_safe\_get*** is doing is getting the value of a variable from nvram because it is close to the function ***nvram\_get*** which get the value of the string you provide to it from the nvram. You can see it in this writeup. <https://elongl.github.io/exploitation/2021/05/30/pwning-home-router.html> . Thanks again for reading.


  