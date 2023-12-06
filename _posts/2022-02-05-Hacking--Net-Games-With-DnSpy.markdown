---
layout:	post
title:	"Hacking .Net Games With DnSpy"
date:	2022-02-05
medium_url: https://noob3xploiter.medium.com/hacking-net-games-with-dnspy-73e1441f81c1
categories: [Hacking, Game Hacking]
tags: [Hacking, Game Hacking, Reverse Engineering, Game Hacking]
---

### Introduction

Unlike native games that is made with c++/c, games made with c# are easier since c# is not compiled and the metadata are not completely lost. In this writeup, i will show you how to hack Unity Games compiled with mono, or any other games that is made with .net framework.

For this writeup, i will be hacking the game ultrakill again. The tool we will be mainly using is Dnspy. <https://github.com/dnSpy/dnSpy>

### Hacking

So to start of, we need to open up the dlls of ultrakill to dnspy, these dll’s can be found in ULTRAKILL\_Data\Managed

![](/img/1*bhI11zPql_CeblXdyAKDWg.png)

Now, we have full access to the code of the game. After some reversing, i found out that the player class is called NewMovement

![](/img/1*OpU7pF5jFRJnsPXPmUYReA.png)On Unity, every class has a function called Start() which is called when a new object of the class is made. In the start of NewMovement, we can see various variables being initialized

![](/img/1*atXU6hw1wTu4Yb_VqpzBTw.png)Now what we can do is add a new line at the end with this.hp = 2000 to give ourself 2000 hp at startup. To do that, right click, and select edit method.

![](/img/1*-E90gimf-ZWmkk0n-s0rMw.png)Now hit compile and save the module. Now when we open up ultrakill. We can see that we will get a 2000 hp on startup.

![](/img/1*NSsHaU4fhGaTs-0O-k0Rkw.png)After, looking through the code more, i found an interesting function called GetHurt which i assume is the function responsible for reducing our health.

![](/img/1*T-bSsmD0InYV92G8-5Vtlw.png)It accepts a damage argument and reduce our hp with the given damage. Now at the start of the function, we can overwrite this first parameter to never reduce our hp. Again, right click the function, and select edit method.

![](/img/1*eiFQqe_LIu2dQ1uFchkZkg.png)I added damage = 0 at the end of the function. Now we should never take damage anymore.

![](/img/1*8BTY4vajSkyTx01npG_tyQ.gif)So thats how you can use dnSpy for game hacking. You can be more creative and achieve more things but this is how its generally done. Thanks for reading.

  