---
layout:	post
title:	"Game Hacking with Python and cheat engine"
date:	2021-11-11
medium_url: https://noob3xploiter.medium.com/game-hacking-with-python-and-cheat-engine-5000369e27b9
categories: [Hacking, Game Hacking]
tags: [Hacking, Game Hacking, Reverse Engineering, Game Hacking]
---


  Hi. In this write up, i will be showing you, how to hack games by editing the memory with cheat engine and will also write a program in python that will automatically edit the memory and do the hack for us. While i was learning game hacking, i noticed that there very little resource about game hacking with python and more on c++. But im a big python fan so i learnt it myself with the help of a really good module in python. Lets get started

### Finding the Memory Address using cheat engine

In this demo, we will be hacking a game called ultrakill. What we will try to do is modify the health and give ourselve alot of health. So, we start up by booting ultrakill and attaching cheat engine into it.

So we start up the game and we can see that on start, we have 200 health

![](/img/1*_VBg8ma972n-14HBQEJV-Q.png)Now what we need to do is find the memory address responsible for this health. In cheat engine, we will scan for the int value 200

![](/img/1*ymEpjKop7vzvA8Jr6xXL2w.png)You can see that it gave us 3,802 memory address with the value 200. We need to narrow it down more. So in game, we will modify our health by using the shotgun explosive in front of us

![](/img/1*te77_QVUeHn-4ouGUCyoeg.png)Now we are on 165 health. Now we will scan the lists of address we found before and we will look if any of those value changed to 165. We can do that by using the next scan.

![](/img/1*awjQqTh5S1Tgyqz2D8LB_w.png)Now we are only down to 1 address. Double click it to add it to our address list. We can confirm that this is the memory address of health by changing it, if we change it, our health in the game should change too.

![](/img/1*SK7OPjJXQ64dC97-dbIuDQ.png)And we can see that it does, i changed the value of the address to 300 and my health in the game become 300 too. Noice

### Finding static pointers that point to the health address

Now even though we have found the health address, it is not static. If we reload or restart the game, this address will be invalid. So what we need to do is find static address that points to the memory of the health. We can do that by right clicking the memory address of the health and pointer scan for this address. The default setting is already good so just click ok. It may take a while.

![](/img/1*-IzzOM7Gu8-Z1h5PaEIVJg.png)We can see that we found alot. We have to narrow this down. To do this, we have to restart the game and find the health address by following the first part of this writeup.

Now i already done it and i got the address 0x173AC32475C

![](/img/1*aCMnL1g9SzPkEGuok8a1JA.png)Now what we need to do next is go back to the pointer scan window, click on pointer scan, then rescan memory. Paste the new address and click OK.

![](/img/1*Llno7ZQbefYUtF5WsY16_Q.png)Now we have 14958. It is less than our previous scan. You can either repeat the proccess again and try to get the pointer paths lower or you can proceed to the next part. I will not scan anymore for this example

Now for the next step, we have to guess. Which of these address do actually point to the address. So we will double click random pointers. Just a note, ignore the threadstacks. I picked 4 values

![](/img/1*ttZwd5X4j34wsF5lyoO5TA.png)Now we will test if these pointers do actually point to the health. We can do that by restarting the game again.

![](/img/1*gEnVDtob4lwTm5Xq4RF2yA.png)After restarting the game, we can see that their values still match our in game health so that must mean they all actually points to the health address. Now you can pick any of those addresses but i will pick the first one. We can further confirm that this is pointer point to the health by modifying its value

![](/img/1*U4DU5f9ukpkoY0psUGNOnw.png)Now we are 100% sure that this pointer point to the health address in the game. If we double click it, we can see that it is a multilevel pointer

![](/img/1*MzKcSOFb-POF6TQ8qlFE9w.png)

### Python Scripting

Now in this part, we will be accessing this pointer and we will access the memory address it points to and modify it using python. We will be using pymeow for this writeup since from all the libraries i tested, this is the only one that worked. You can download pymeow here <https://github.com/qb-0/PyMeow>

So we will make a new script and we will start by importing the library

After that, we can start coding. We will follow the cheatsheet of pymeow for <https://github.com/qb-0/PyMeow/blob/master/cheatsheet.txt>

There is 2 way to get the process, `process_by_name`, and `process_by_pid`. We will be using `process_by_name` since it is easier.

![](/img/1*xg42HKZNO4eK7oyx7FAkIA.png)In there we passed the name of the program ULTRAKILL.exe.

Now, we have to get the base address of the pointer.

![](/img/1*jLDuNw4wiySHvRO4yLHY7w.png)In our case, the base address is “mono-2.0-bdwgc.dll”+004A1820. So, first, we have to get the address of mono-2.0-bdwgc.dll and add 0x004A1820 to it to get the base address. In pymeow, Process objects have a dictionary called modules that contains all the modules and those modules are also dictionaries and they have a parameter called baseaddr.

![](/img/1*tqkhkMlTLlTprFWV4jJx2g.png)Now what we want to do is get the base address of mono-2.0-bdwgc.dll and we can access it with process[“modules”][“mono-2.0-bdwgc.dll”][“baseaddr”], then we will add 0x017C58C0 to it to get the base address of the pointer

![](/img/1*YDB7wRw2QgrOBq-qBWzCZw.png)We can confirm that this is the right address in cheat engine

![](/img/1*TV9pf3uH_u2rsXdztiVZWw.png)Now we have to deal with offsets. First, i will explain how offsets works and how we will deal with them.

![](/img/1*R0ERdHcI5N8AIhiSSWS9pQ.png)Here, we can see that it has 7 offsets. We can see that below, the first address is 0x7ff920af1820. Then, the first offset came which is 140. What it does is it adds 0x140 bytes to the first address which is 0x7ff920af1820 and then it will get the value in that address with offset. Now that address points to another pointer and again, we will add 0x1F0 bytes into that pointer until we get to the last part which is the actual memory address of the user health. Now, lets implement it in our code.

pymeow has a method `pointer_chain` but its not working for me so we will implement our own function instead.

![](/img/1*Zu9Xhv6I5LyasNQKT_eVRQ.png)I came up with a function that i called `read_offsets`. It takes 3 arguments, the first one is the proc, which is the process, the second is the base address, the third is the array of offsets. In line 4, it will get the pointer in the base address. In line 6, it makes a new variable called `current_pointer` from the basepoint variable. Then it will loop through every offsets except for the last character, what it does is, it will read the pointer in the `current_pointer+offset` and set the value of `current_pointer` to it for the next iteration. Then in the last part, it will return the final pointer.

We use `read_int64` since we are working on a 64 bit program and memory address in 64 bit programs are 8 bytes. Now lets try it if it works

![](/img/1*qGmb0A-1pwMRo8gxG2IEkQ.png)I mapped out the offsets into an array in my code. Now lets run it.

![](/img/1*iv3P0RXP3NTplFPidT71PQ.png)We can see that it works, it obtained the value 1337 which is the value we set the health earlier in in cheat engine. Now lets try modifying that address using `write_int`.

![](/img/1*TeUQInQDOkpGjErKO-ckHg.png)Now if we get back to the game,

![](/img/1*NhSzdhLWg1b8j3XF2eNHuA.png)We can see that it is successful and our health is now 200.

This is the end of my writeup. I hope this resource will help for future hackers that are interested on game hacking with python. Thanks for reading

Follow me on twitter: [https://twitter.com/tomorrowisnew\__](https://twitter.com/tomorrowisnew__)

Join the discord server: <https://discord.gg/bugbounty>

  