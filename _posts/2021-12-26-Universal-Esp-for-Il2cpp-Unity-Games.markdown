---
layout:	post
title:	"Universal Esp for Il2cpp Unity Games"
date:	2021-12-26
medium_url: https://noob3xploiter.medium.com/universal-esp-for-il2cpp-unity-games-7ba57c8f8605
categories: [Hacking, Game Hacking]
tags: [Hacking, Game Hacking, Reverse Engineering, Game Hacking]
---

  In this writeup i will be showing you how to make an esp on any unity games that is il2cpp compiled. This writeup is inspired by <https://github.com/ethanedits/Universal-Unity-ESP> . Unity can be compiled with either il2cpp or mono. In mono, we can do mono injection for making our hacks, however, you cant do the same on il2cpp games. Instead, we will be treating il2cpp games as native applications. Lets get start

### Setup

Before we start, we need to setup some things first. For this example, i will be using a slightly modified FPS Microgame by unity. By slightly modified, i added a static variable in the EnemyTurret class that points to the instance of an EnemyTurret making it easier for us to get its address. You can download my modified game here <https://noobexploiter.itch.io/slightly-modified>

Next, we need to dump our il2cpp game. For that we will be using [il2cppdumper](https://github.com/Perfare/Il2CppDumper). After dumping the GameAssembly.dll, it will provide multiple dll’s. We will open these dll’s in [dnspy](https://github.com/dnSpy/dnSpy#:~:text=dnSpy%20is%20a%20debugger%20and,have%20any%20source%20code%20available.).

![](/img/1*a3FZGF57jS8S-ZXxkpQQ9g.png)Next, for making our esp lines, we will be using, imgui, there is already a template for hooking directx11 with imgui implementation, which we will use in this tutorial. <https://github.com/rdbo/ImGui-DirectX-11-Kiero-Hook>. We can now start

### Coding

Inside the main.cpp, we will be making a new function called MainHack. We will put all our code in there. Then, we will call MainHack between `ImGui::Begin(“ImGui Window”); and ImGui::End();`

![](/img/1*R1KFW0EUAzyfV2ecpVeLXg.png)

Like i said, i added a static variable in EnemyTurret class that points to the EnemyTurret Object. You can follow this tutorial to get the static variable. <https://guidedhacking.com/threads/how-to-get-the-address-of-a-static-variable-in-unity-games.16246/>

![](/img/1*zq94o-TIulD1cLfJjlkt7A.png)Just a note, instead of using `0x5C`, use `0xB8` since my game is 64 bit(i forgot to compile it as 32 bit) so it is twice as big.

### Getting the position

The EnemyTurret class Inherits from MonoBehaviour and MonoBehaviour inherits Behaviour and Behaviour inherits Component and the Component class has a function called transform which will return the Transform object of our EnemyTurret Object

![](/img/1*ZE768gj6x10HbConjjaXMQ.png)The tranform class holds the position of a gameobject. We will first make our transform function and call transform to our EnemyTurret object. If you dont know how to call functions, follow this guide [https://www.unknowncheats.me/wiki/Calling\_Functions\_From\_Injected\_Library\_Using\_Function\_Pointers\_in\_C%2B%2B](https://www.unknowncheats.me/wiki/Calling_Functions_From_Injected_Library_Using_Function_Pointers_in_C%2B%2B).

For the address of the function, we will be using the RVA, the RVA is the offset of the function from the GameAssembly.dll

![](/img/1*sWThGqy1ApV8XNNw6nueJA.png)Now we have the transform object of our turret. The Transform class in unity has a property called position which is a vector 3 that holds the position of a game object. Properties on C# can be called just like a normal function call.

![](/img/1*LppSULQZi-XbUZ_pe_ir6w.png)

But first, we need to make our own Vector3 struct. Vector 3 is just 3 float, called x,y and z.

![](/img/1*uY4ZaEKujRAFA_VeVfpZwA.png)

Now we can write our own position function and call it with the transform of our turret.

![](/img/1*mgG9KqvG_zAmYmUIHUt6TQ.png)

### Camera And WorldToScreen

In making esp, WorldToScreen is a function that transform the position of an object in a 3d world to screen coordinates. Hopefully for us, Unity has a built in worldtoscreen function in the Camera class called [WorldToScreenPoint](https://docs.unity3d.com/ScriptReference/Camera.WorldToScreenPoint.html).

We need to get the current camera first. In unity, the camera class has a static property called current that returns the current camera in use.

![](/img/1*Xa2hon2kvTTcfuXhjfRXIw.png)We can call this, like a normal function call.

![](/img/1*d4sPiBQ2Oc1a66Xdp4yhRg.png)Now that we have camera object. We can now call the WorldToScreenPoint

![](/img/1*WysH3itFuWa-f5e5-y4MlA.png)WorldToScreenPoint accept a Vector3 as an argument which is the position of our target EnemyTurret Object.

![](/img/1*nqTTh2zFXU5DIZJHIIZBHw.png)This is what our code looks like now.

### Esp with ImGui

Now that we know where to draw, we will now begin to draw. We will be using the function AddLine of ImGui to draw a line from the bottom of the screen, to the position of the Enemy turret

In ImGui, it only accepts ImVec2 for its coordinate so we will make our ImVec2 variables first. Then we will call the AddLine Function

![](/img/1*Kw4FSvT39WOueNCfAgXl6A.png)So lets build it, compile it, and………

![](/img/1*ivcOoHpg7LTzA24Ab7cNcg.gif)Its not working. The x position of the esp line is correct. However, the y position is wrong. After some googling, i found this <https://forum.unity.com/threads/worldtoscreenpoint-doesnt-work-on-y-screen-axis.34161/>. Here, he said to subtract the Screen height to the y position. So lets do the same.

![](/img/1*w77ker-uDxhPBi7wGfFXBg.png)Build it again, inject it, and now its working fine.

![](/img/1*hScEdJX3ikpY6XSm_7OTFg.gif)This is the end of the writeup, thanks for reading.

  