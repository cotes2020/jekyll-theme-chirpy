---
layout:	post
title:	"Hacking Mono Games With Frida"
date:	2022-04-24
categories: [Hacking, Game Hacking]
image: /img/Frida1.png
tags: [Hacking, Game Hacking, Mono, Frida]
---


# INTRODUCTION
Recently, i've been practicing game hacking frida. Frida is an excellent tool for hacking. It allows us to make scripts in javascript, and is very flexible, a script, we made can be run on any architecture. Making it great for android game hacking. But there is one problem i encountered often. Games are compiled in mono and i dont know how to hack mono games and there is almost no resource about it except for the documentation. Mono games works differently than native games. So, i decided to make a writeup about it after hours of reading the documentation and reading the source code. Lets get started.

# Game Plan
Before we start, make sure you have an ide and frida installed.
The game plan on hacking mono games is simple. Here is a diagram i made summarizing it.    
![](/img/Frida1.png)        
First, we will get the mono library which we will use. Then, we will need to set our thread to the root domain. So when we later compile functions using `mono_compile_method` (we will talk about this later.), it will work. Next, we will get all the assemblies. Assemblies are the inidividual dll's that are loaded by the mono library.      
![](/img/frida2.PNG)        
Then, we will get the class that we need from the assembly. Then, we will get the function in the class, and get its address. Then, we can do whatever we want. Lets get started

# Coding
For this writeup, i will be hacking ultrakill once again since i already worked with it before and know its classes and functions already.
First, we will get the mono library. In ultrakill, the mono library is named `mono-2.0-bdwgc.dll`. We can get it using `getModuleByName` which will give us a module object.
```js
var hMono = Process.getModuleByName("mono-2.0-bdwgc.dll")
```
Next, like i said in the flowchart, we need to set our thread to the root domain. The function for getting the root domain is `mono_get_root_domain` and the function for setting our thread is `mono_thread_attach`. We can get the address of these functions using the `getExportByName`. Since these are functions, we will make a new [NativeFunction](https://frida.re/docs/javascript-api/#nativefunction) out of them.
```js
var hMono = Process.getModuleByName("mono-2.0-bdwgc.dll") //Hook the mono module

//Domain Stuffs
var mono_get_root_domain = new NativeFunction(hMono.getExportByName("mono_get_root_domain"), 'pointer', [])
var mono_thread_attach = new NativeFunction(hMono.getExportByName("mono_thread_attach"), 'pointer', ['pointer'])
mono_thread_attach(mono_get_root_domain())
```
Next, to get the assemblies, we will use the function `mono_assembly_foreach` to loop through each assemblies, and check get the assembly if it is named `GameAssembly`, which is the dll that contains all the logics for unity. Then, we will need to convert this Assembly object to image object, since we cant really do anything with an Assembly object. According to the documentation, `mono_assembly_foreach`, first argument should be a function, which will be called, on every iteration of assemblies found. The Assembly object will get passed as the first argument to the function. The second argument of `mono_assembly_foreach` is useless and can be ignored.
```js
var hMono = Process.getModuleByName("mono-2.0-bdwgc.dll") //Hook the mono module

//Domain Stuffs
var mono_get_root_domain = new NativeFunction(hMono.getExportByName("mono_get_root_domain"), 'pointer', [])
var mono_thread_attach = new NativeFunction(hMono.getExportByName("mono_thread_attach"), 'pointer', ['pointer'])
mono_thread_attach(mono_get_root_domain())

//Assemblies stuff
var mono_assembly_foreach = new NativeFunction(hMono.getExportByName("mono_assembly_foreach"), 'void', ['pointer', 'pointer']) //List all assembly
var mono_assembly_get_name  = new NativeFunction(hMono.getExportByName("mono_assembly_get_image"), 'pointer', ['pointer']) //Get image from assembly
var AssemblyCsharpAssembly
function GetAssemblyCsharpCallback(MonoAssemblyObject, user_data){ //Function to be called on every assemblies found
    var MonoAssemblyImageObject = mono_assembly_get_name(MonoAssemblyObject)
    var ImageName = mono_image_get_name(MonoAssemblyImageObject)
    if(ImageName.readUtf8String() == "Assembly-CSharp"){ //Check if it is the Assembly-CSharp
        console.log("AssemblyCsharp Found. Assembly object at :" + MonoAssemblyImageObject)
        AssemblyCsharpAssembly = MonoAssemblyImageObject
    }
}

mono_assembly_foreach(new NativeCallback(GetAssemblyCsharpCallback, 'void', ['pointer', 'pointer']), ptr(0))
```
Here, i made a new function, `GetAssemblyCsharpCallback`, the first argument is the Mono Assembly Object. It will get the name of the Assembly using `mono_assembly_get_name`, and if it is equals to `Assembly-CSharp`, set our AssemblyCsharpAssembly variable to the image of it.
AssemblyCsharp contains all the game logic. Next, we are gonna get our target class on it, the `NewMovement` class.    
![](/img/frida3.PNG)     
This is the class of our player. We will use `mono_class_from_name` to get our Mono class object
```js
var hMono = Process.getModuleByName("mono-2.0-bdwgc.dll") //Hook the mono module

//Domain Stuffs
var mono_get_root_domain = new NativeFunction(hMono.getExportByName("mono_get_root_domain"), 'pointer', [])
var mono_thread_attach = new NativeFunction(hMono.getExportByName("mono_thread_attach"), 'pointer', ['pointer'])
mono_thread_attach(mono_get_root_domain())

//Assemblies stuff
var mono_assembly_foreach = new NativeFunction(hMono.getExportByName("mono_assembly_foreach"), 'void', ['pointer', 'pointer']) //List all assembly
var mono_assembly_get_name  = new NativeFunction(hMono.getExportByName("mono_assembly_get_image"), 'pointer', ['pointer']) //Get image from assembly
var AssemblyCsharpAssembly
function GetAssemblyCsharpCallback(MonoAssemblyObject, user_data){
    var MonoAssemblyImageObject = mono_assembly_get_name(MonoAssemblyObject)
    var ImageName = mono_image_get_name(MonoAssemblyImageObject)
    if(ImageName.readUtf8String() == "Assembly-CSharp"){
        console.log("AssemblyCsharp Found. Assembly object at :" + MonoAssemblyImageObject)
        AssemblyCsharpAssembly = MonoAssemblyImageObject
    }
}

mono_assembly_foreach(new NativeCallback(GetAssemblyCsharpCallback, 'void', ['pointer', 'pointer']), ptr(0))

//Class Stuff
var mono_class_from_name = new NativeFunction(hMono.getExportByName("mono_class_from_name"), 'pointer', ['pointer', 'pointer', 'pointer'])

var NewMovementClass = mono_class_from_name(ptr(AssemblyCsharpAssembly), Memory.allocUtf8String(""), Memory.allocUtf8String("NewMovement"))
```
The mono_class_from_name takes 3 parameter, the first is the Assembly image, the next is the namespace, and the last is the name of the class. It accepts the second and third parameter as `const char*`, so we will use `Memory.allocUtf8String`, to alloc a const char to memory, and return a pointer to it.
Now, we will get the Update method. we can do that with the function `mono_class_get_method_from_name`. It will return a mono method object, then we will pass it to `mono_compile_method` to get it's address
```js
var hMono = Process.getModuleByName("mono-2.0-bdwgc.dll") //Hook the mono module

//Domain Stuffs
var mono_get_root_domain = new NativeFunction(hMono.getExportByName("mono_get_root_domain"), 'pointer', [])
var mono_thread_attach = new NativeFunction(hMono.getExportByName("mono_thread_attach"), 'pointer', ['pointer'])
mono_thread_attach(mono_get_root_domain())

//Assemblies stuff
var mono_assembly_foreach = new NativeFunction(hMono.getExportByName("mono_assembly_foreach"), 'void', ['pointer', 'pointer']) //List all assembly
var mono_assembly_get_name  = new NativeFunction(hMono.getExportByName("mono_assembly_get_image"), 'pointer', ['pointer']) //Get image from assembly
var AssemblyCsharpAssembly
function GetAssemblyCsharpCallback(MonoAssemblyObject, user_data){
    var MonoAssemblyImageObject = mono_assembly_get_name(MonoAssemblyObject)
    var ImageName = mono_image_get_name(MonoAssemblyImageObject)
    if(ImageName.readUtf8String() == "Assembly-CSharp"){
        console.log("AssemblyCsharp Found. Assembly object at :" + MonoAssemblyImageObject)
        AssemblyCsharpAssembly = MonoAssemblyImageObject
    }
}

mono_assembly_foreach(new NativeCallback(GetAssemblyCsharpCallback, 'void', ['pointer', 'pointer']), ptr(0))

//Class Stuff
var mono_class_from_name = new NativeFunction(hMono.getExportByName("mono_class_from_name"), 'pointer', ['pointer', 'pointer', 'pointer'])

var NewMovementClass = mono_class_from_name(ptr(AssemblyCsharpAssembly), Memory.allocUtf8String(""), Memory.allocUtf8String("NewMovement"))

//Method Stuff
var mono_class_get_method_from_name  = new NativeFunction(hMono.getExportByName("mono_class_get_method_from_name"), 'pointer', ['pointer', 'pointer', 'int'])
var mono_compile_method = new NativeFunction(hMono.getExportByName("mono_compile_method"), 'pointer', ['pointer'])

//Get Update Method address
var NewMovementUpdateMethod = mono_class_get_method_from_name(NewMovementClass, Memory.allocUtf8String("Update"), 0)
var NewMovement_Update = mono_compile_method(NewMovementUpdateMethod)
```
`mono_class_get_method_from_name` accepts 3 parameter, the first is the Mono Class object, the second is the name of the function, and the third is the number of parameters of the function. Then, we passed the Mono Method Object to  mono_compile_method. Now that we have those, we can start making a hack

# Making the hack
Now that we know how to get the address of the functions, we can now make our hack. To demonstrate the power of frida, i will be showing you three of the most important aspect of game hacking, memory manipulation, function hooking, and function calling

## Function Hooking.
First, we will hook the update method to get our player object. Class methods always pass the object instance as the first variable, so we can get our NewMovement instance in the first argument of the Update method. We can do that in frida, using [Interceptor](https://frida.re/docs/javascript-api/#interceptor)
```js
...

var LocalPlayer
Interceptor.attach(NewMovement_Update, {
  onEnter(args) {
    if(!LocalPlayer || LocalPlayer.toString() != args[0].toString()){
      console.log("LocalPlayer Found: " + args[0].toString())
      LocalPlayer = args[0]
    }
  }
});
```
Now we have the LocalPlayer instance

## Memory Manipulation
Now that we have the local player. Now, we will manipulate its data. Specifically, the hp field. To get the offset of a field. We will use `mono_class_get_field_from_name` to get the Mono field object and `mono_field_get_offset` to get its offset
```js
...

//Field Stuff
var mono_class_get_field_from_name = new NativeFunction(hMono.getExportByName("mono_class_get_field_from_name"), 'pointer', ['pointer', 'pointer'])
var mono_field_get_offset = new NativeFunction(hMono.getExportByName("mono_field_get_offset"), 'int', ['pointer'])

function ManipulateHealth(Player){
  var ClassHpField = mono_class_get_field_from_name(NewMovementClass, Memory.allocUtf8String("hp"))
  var HpOffset = mono_field_get_offset(ClassHpField)
  console.log("Health: " + Player.add(HpOffset).readInt())
  Player.add(HpOffset).writeInt(200)
}

var LocalPlayer
Interceptor.attach(NewMovement_Update, {
  onEnter(args) {
    if(!LocalPlayer || LocalPlayer.toString() != args[0].toString()){
      console.log("LocalPlayer Found: " + args[0].toString())
      LocalPlayer = args[0]
      ManipulateHealth(args[0])
    }
  }
});
```

## Function Calling
Now for function calling, lets call the jump function.
To get the address of it, we will use again `mono_class_get_method_from_name` and `mono_compile_method`. For this example, we will get rid of the ManipulateHealth method and instead, call the SuperCharge function.     
![](/img/frida5.PNG)       
```js
...
function SuperCharge(Player){
    var SuperChargeMethod = mono_class_get_method_from_name(NewMovementClass, Memory.allocUtf8String("SuperCharge"), 0)
    var SuperChargeAddress = mono_compile_method(SuperChargeMethod)
    var SuperChargeFunction = new NativeFunction(SuperChargeAddress, 'void', ['pointer'])
    SuperChargeFunction(Player)
}

var LocalPlayer
Interceptor.attach(NewMovement_Update, {
  onEnter(args) {
    if(!LocalPlayer || LocalPlayer.toString() != args[0].toString()){
      console.log("LocalPlayer Found: " + args[0].toString())
      LocalPlayer = args[0]
      SuperCharge(args[0])
    }
  }
});
```    
![](/img/frida6.gif)       

# Conclusion
As you can see, frida is a powerful tool. The hardest part of this is learning and understanding the documentation. I hope this writeup will be a guide to future hackers that are also finding problems in hacking mono games. Thanks for reading.
