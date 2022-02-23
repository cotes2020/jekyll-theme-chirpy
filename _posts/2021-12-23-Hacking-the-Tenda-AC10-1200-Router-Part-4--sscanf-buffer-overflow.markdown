---
layout:	post
title:	"Hacking the Tenda AC10–1200 Router Part 4: sscanf buffer overflow"
date:	2021-12-23
medium_url: https://noob3xploiter.medium.com/hacking-the-tenda-ac10-1200-router-part-4-sscanf-buffer-overflow-75ae0e06abb6
categories: [Hacking, Iot]
tags: [Hacking, Iot, Reverse Engineering, Binary Exploitation]
---

  In this writeup, i will show you a sscanf buffer overflow that i found in tenda ac10–1200. I tried reporting it but no response, so i decided to publish it to raise awareness on other people.

While reversing the firmware, i found the function a vulnerable function called **setSmartPowerManagement,**

![](/img/1*FYJurJhh2SfBC4tqisfsVQ.png)Here, it gets the value of the time parameter, and store it to the variable **var\_154\_1**. This variable is then used in ***sscanf*** which is known to cause buffer overflows

![](/img/1*duSICXwHoeIJZgpEs3i_kg.png)The ***sscanf*** accept our input in the time variable, matches it with the format in **$a1**, and store the values in the variables **var\_14c**, **var\_144**, **var\_13c**, and **var\_134**. These variables are just 8 bytes so if we send an input with longer than 8 bytes with the correct format, we can overflow past these variables. For the format, websGetVar’s second parameter contains the default value of the parameter if none is given, we can use that as a reference

![](/img/1*vSF3XIFaBIqpoSqMPjDUWA.png)Now that we know the format, we can now test the bof.

![](/img/1*ug9G8Zs7_I6ZqoSfVH2f1A.png)**setSmartPowerManagement** is defined with **PowerSaveSet**, that means our vulnerable endpoint is **/goform/PowerSaveSet**.

![](/img/1*LC-qMSMq8irLk88S7vTukA.png)After sending the request, it didnt responded, thats a good indication that our exploit worked. If we looked at the emulation, it shows a SIGSEGV which means we are successful at crashing the server.

![](/img/1*i6M8jVZiSrPRlkskLD323Q.png)While debugging this, i cant find a way to overwrite the program counter, the websDone at the end of the function is crashing the program before it even reach the return.

![](/img/1*Vw9bl1BbWtipCcHuJKHbQA.png)![](/img/1*oeAT6L2FrhDTeB6O-1mvhg.png)But, we still have a dos here. So thats nice.

This is the end of the writeup, i tried reaching out to tenda alot of times before but no response as always, so i decided to publish this bug now. Thanks for reading

  