---
layout:	post
title:	"Il2cppDumper Manually Finding CodeRegistration and MetadataRegistration"
date:	2022-05-15
categories: [Hacking, Game Hacking]
image: /img/guide1.PNG
tags: [Hacking, Reverse Engineering, Game Hacking, ida]
---


# INTRODUCTION
While trying to dump the game [marvel strike force](https://play.google.com/store/apps/details?id=com.foxnextgames.m3&hl=en&gl=US). I encountered this error     
![](/img/ida1.PNG)
I have no idea what to do, so i do a google search and found this thread in unknowncheats    
<https://www.unknowncheats.me/forum/general-programming-and-reversing/355005-il2cppdumper-tutorial-finding-coderegistration-metadataregistration.html>    
I tried it myself, but i just cant get it to work with the patter creation part. Scrolling down, i found this comment
![](/img/ida2.PNG)
This is really interesting so lets check it out.

# Reversing Our Own Game
So i made a blank game, from unity, for android, and compiled it with unity. And then, i opened it up in ida. I went to the strings window, and searched for `global-metadata.dat`    
I looked for cross references and found one       
![](/img/ida3.PNG)      
It is referenced in `il2cpp::vm::GlobalMetadata::Initialize`. This is the function responsible for parsing the global metadata file. Lets look for cross references again on this function and found one cross reference again.          
![](/img/ida4.PNG)     
It is referenced in `il2cpp::vm::MetadataCache::Initialize`. Again, it is not what we are looking for, so lets check for a cross references into this function again.      
![](/img/ida5.PNG)       
It has one cross reference. This time, in `il2cpp::vm::Runtime::Init`      
![](/img/ida6.PNG)       
This cross reference is interesting. Above the cross reference, we can see that, it calls a function, pointed out by the address in `g_CodegenRegistration_ptr`    
If we double cliked that address, we can see that it points out to the function `g_CodegenRegistration`.      
![](/img/ida7.PNG)       
And `g_CodegenRegistration` points to `s_Il2CppCodeGenRegistration`      
![](/img/ida8.PNG)       
And in there, we can indeed see, what we are looking for.      
![](/img/ida9.PNG)       
The `metadataRegistration` and `codeRegistration`. They are passed as an argument to the function `il2cpp_codegen_register`. `codeRegistration` as the first argument and `metadataRegistration` as the second argument. Now lets try it in marvel super war.   

# Reversing Marvel Super War
Note, use the 32 bit version of the game. Idk why but i cant get it to work in 64 bit.       
Once again, we searched for `global-metadata.dat` in the strings window, and searched for cross references on it.      
![](/img/ida10.PNG)       
We found one cross references. We can assume that this function is the `il2cpp::vm::GlobalMetadata::Initialize`.        
Then, we looked for cross references in that function and found one. 
![](/img/ida11.PNG)      
We can again assume that this is `il2cpp::vm::MetadataCache::Initialize`.
Look again for cross references and we found one      
![](/img/ida12.PNG)      
We can assume that this function is `il2cpp::vm::Runtime::Init`.       
![](/img/ida13.PNG)         
Above our cross reference, we can see that it made a function call in the address stored in the R0 register. We can assume that this points to `s_Il2CppCodeGenRegistration`. The address of `g_CodegenRegistration` is in `off_3460F40 - 0xB31598`. Visiting that address, it points to a function, which, we can assume is the `s_Il2CppCodeGenRegistration`       
![](/img/ida14.PNG)      
Visiting this function         
![](/img/ida15.PNG)       
It does look similar on `s_Il2CppCodeGenRegistration` function. In there, we can assume that the address stored in R0 is the  `codeRegistration` and R1 is the `metadataRegistration`. Their address are `unk_341F640` and `unk_341FBD4` which are `0x341F640` and `0x341FBD4` in hex. Now lets try dumping the game with these offsets.       
![](/img/ida16.PNG)      
It succeed.      

# Alternative Route
Alternatively, if in any chance, we cant find the global-metadata.dat string, eg its  encrypted, we can also start from the function `il2cpp_init`.      
![](/img/ida17.PNG)       
You can see a function call to `sub_B31484` which is the `il2cpp::vm::Runtime::Init`. Scroll down until you see `4.0`, and you will know that you are close.     
![](/img/ida18.PNG)      

# Conclusion
There you have it. This is how you can get the `CodeRegistration` and `metadataRegistration` manually. Thanks for reading.      