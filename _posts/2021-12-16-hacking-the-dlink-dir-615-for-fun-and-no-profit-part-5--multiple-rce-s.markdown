---
layout:	post
title:	"Hacking the dlink DIR-615 for fun and no profit Part 5: Multiple RCE’s"
date:	2021-12-16
medium_url: https://noob3xploiter.medium.com/hacking-the-dlink-dir-615-for-fun-and-no-profit-part-5-multiple-rces-d508f58e2471
categories: [Hacking, Iot]
tags: [Hacking, Iot, Reverse Engineering, Binary Exploitation]
---


  Its been a while since i last did some iot hacking and i missed it. So i decided to try it again with my trusty target, dlink dir-615. And in this writeup, i will show you multiple bugs that i found

### CVE-2020–10216

First bug on the list is a remote code execution.

![](/img/1*GN1PVeEzmX0QfTk2JnjGeQ.png)Here, it get the value of the parameter date and store it to the **$s0** register.

![](/img/1*rIJ_zN5nAmjLqCnY0JJ9FA.png)And down below, this $s0 register is used as an argument to `_system` in the format string date -s %s making it vulnerable to remote code execution. The vulnerability exists in `system_time.cgi`

![](/img/1*60a3wWqm26Dj3lJz0SmAXA.png)Now lets try it in burp suite, i tried sending the payload $(reboot), and

![](/img/1*ZJF8q6uv3k8NQKEdF0v3cg.png)And after sending it, my emulation rebooted, just like what we expected

![](/img/1*pACdpeedKbfusj_fFB1G1A.png)

### CVE-2019–9122 & CVE-2020–10214

Next bugs are two bugs in the same parameter. One is an rce (CVE-2019–9122) and the other one is a buffer overflow (CVE-2020–10214). While reversing, i found this parameter called `ntp_server`.

![](/img/1*4zdNY9G3QG70DIoJQm7Ddw.png)Here, it gets the value of the parameter and passed it to sprintf as an argument.

![](/img/1*HqWqePed2EH47YFMac7t0A.png)Then, the result of the sprintf is used into `_system` making it vulnerable to rce. This bug exist in `ntp_sync.cgi`

![](/img/1*RSceuQz3iF1MOwk1S1pTpg.png)Now lets try to replicate it

![](/img/1*9nXt43bEwoiSjDT8UgGV8g.png)After sending a request with the payload $(reboot), the emulation rebooted as we excpeted

![](/img/1*Yq5GE4zdYMJsjL-GwoKvQg.png)If you remember, our input is passed an argument to sprintf and sprintf is widely known as a cause of buffer overflow due to the lack of length check. So, if we supplied a very long input, it should cause a buffer overflow overwriting the return address and crashing the program. Now to replicate it

![](/img/1*mU5tDIoCOCIeJGsHsq6RMw.png)I sent a very long string of A’s, looking at the emulation

![](/img/1*jIgm0-yo1d4yLQ0WrkOKPw.png)We can see that we overwritten the return address **$ra**, with 414141 which is the equivalent of A in hex.

This is the end of the writeup, even though i didnt found any new bugs and get a new cve, we still found some pretty interesting bugs.

Join The bug hunting discord server: <https://discord.gg/bugbounty>

  