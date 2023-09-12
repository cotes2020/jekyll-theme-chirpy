---
title: iOS 静态库制作、静态库的源码调试
date: 2023-08-02 17:03:25
categories: [iOS]
tags: [SDK]
---

## 静态库和动态库的存在形式和区别

### 静态库

* `.a` 和 `.framework`
* 会被完整地复制到每个调用它的程序中，被多个程序使用就有多份冗余拷贝


### 动态库

* `.tbd`（这是从 Xcode 7 开始的后缀名，Xcode 7 之前是 `.dylib` ） 和 `.framework`
* 在程序运行时由系统动态加载到内存中，系统只加载一份，多个程序共用，节省内存


<br>

## CPU 架构介绍

* CPU 执行计算任务时都需要遵从一定的规范，程序在被执行前都需要先翻译为 CPU 可以理解的语言。这种规范或语言就是`指令集架构（ISA，Instruction Set Architecture）`。
* 目前市面上的 CPU 分类主要分有`两大阵营`，一个是 `Intel、AMD 为首的复杂指令集 CPU`；另一个是`以 IBM、ARM 为首的精简指令集 CPU`。
* `Intel、AMD 的 CPU 是 x86 架构`的；而 `IBM 公司的 CPU 是 PowerPC 架构`；`ARM、苹果 公司是 ARM 架构`。x86、ARM v8、MIPS 都是指令集的代号。

### arm 处理器
arm处理器，因为其低功耗和小尺寸而闻名，几乎所有的手机处理器都基于arm，其在嵌入式系统中的应用非常广泛，它的性能在同等功耗产品中也很出色。

<br>

### iPhone 、iPad 指令集架构

苹果A7处理器（A系列的第一代64位处理器）支持两个不同的指令集：32位ARM指令集（armv6｜armv7｜armv7s）和64位ARM指令集（arm64），i386｜x86_64 是PC处理器的指令集，i386是针对intel通用微处理器32架构的。x86_64是针对x86架构的64位处理器。当使用iOS模拟器的时候会用到 i386｜x86_64，iOS模拟器没有arm指令集，因为模拟器运行在Mac电脑上，使用的是Mac的CPU指令集（在苹果的 M 系列电脑端的 arm 处理器出来后，模拟器使用的指令集也是 arm ？还未查询验证）。


<br>

`armv7` / `armv7s` / `arm64` 都是 ARM 处理器的指令集，一般用于移动设备
`i386` 是针对 intel 通用微处理器 `32` 位处理器
`x86_64` 是针对 x86 架构的 `64` 位处理器，`兼容 32 位`


arm64 ：5S+
armv7s：5 ~ 5C
armv7 ：3GS ~ 4S `（静态库只要支持了armv7，就可以跑在armv7s的架构上）`


模拟器 32 位处理器测试需要 i386 架构，即 5S 之前的设备，不包含 5S
模拟器 64 位处理器测试需要 x86_64 架构，即 5S 之后的设备，包含 5S（电脑非苹果芯片）
真机 32 位处理器需要 armv7，或者 armv7s 架构，
真机 64 位处理器需要 arm64 架构。


<br>

## Xcode 中设置

### 设置架构

主要有以下三个：`Architectures`、`Build Active Architecture Only`、`Excluded Architectures`

<br>

#### Architectures

* 控制编译器生成的二进制文件所支持的 CPU 架构。通常用默认的`Standard Architectures(arm64)`通用架构即可。

<br>

#### Build Active Architecture Only

* 控制编译器是否只编译当前活动的 CPU 架构。如果开启该选项，编译器只会编译当前选中的 CPU 架构，可以加快编译速度，但是生成的二进制文件只能在当前架构的设备上运行。如果关闭该选项，则编译器会同时编译所有支持的 CPU 架构，生成通用的二进制文件。
* `建议把 Release 模式都设置为 NO`。


<br>

#### Excluded Architectures

* 用于排除某些不需要支持的 CPU 架构。如果某些架构不需要被支持，可以在这个选项中将其排除，编译器会忽略这些架构，不会为其生成二进制文件。
* `一般不用设置`。


<br>

### 设置 Other linker flags 参数

`当静态库使用了 Category 时，需要把 Other linker flags 参数设置为` `－ObjC`，否则会报错

有以下三个参数可选：

* `－ObjC`：加了这个参数后，链接器就会把静态库中所有的Objective-C类和分类都加载到最后的可执行文件中

* `－all_load`：会让链接器把所有找到的目标文件都加载到可执行文件中，但是千万不要随便使用这个参数！假如你使用了不止一个静态库文件，然后又使用了这个参数，那么你很有可能会遇到ld: duplicate symbol错误，因为不同的库文件里面可能会有相同的目标文件，所以建议在遇到-ObjC失效的情况下使用-force_load参数。

* `-force_load`：所做的事情跟-all_load其实是一样的，但是-force_load需要指定要进行全部加载的库文件的路径，这样的话，你就只是完全加载了一个库文件，不影响其余库文件的按需加载

<br>

## 以下用 Xcode 14.3.1 演示打包静态库

<br>

### 创建 .a 静态库

`command + shift + N` -> `iOS` -> `Framework & Library` -> `Static Library`

<br>

#### 写完代码后

1. 公开头文件
    * 点击 `TARGETS` 下的项目 -> `Build Phases` -> `Copy Files`，添加需要公开的头文件

2. 设置 Xcode
    * 设置 `Build Active Architecture Only` 下的 `Debug` 和 `Release` 都为 `NO`
    * 如果`涉及 C++ 混编`，需要修改 `Build Settings` -> `Compile Sources As` 为 `Objective-C++`，否则在导入静态库的项目中混编 C++ 代码时，编译器报错
    * 如果想去掉打出来的 .a 库文件名前面自动添加的`lib`，可以设置取消，-> `Build Settings` -> `Executable Prefix` -> `删掉 lib`
    * `Build Settings` -> `Generate Debug Symbols`，`如果此项设置为 Yes，那么打出来的静态库会暴露源码`，`在调用库方法的地方打断点，单步调试进入就能看到方法源码`。有时为了调试方便，可以设置为 Yes，一般两个都设为 NO，或者只把 `Release 设为 NO`，设为 NO 打出来的库体积也更小。

3. 选择制作 Release 或 Debug 版本的库
    * 选择 项目 -> `Edit Scheme` -> `Run` -> `Info` -> `Build Configuration`，选择 `Debug` 或者 `Release`

4. 打包库
    * 选择某个模拟器，编译运行，打出来的是支持模拟器指令集的 .a 静态库
    * 选择 `Any iOS Device (arm64)`，打出来的是支持arm64架构真机使用的 .a 静态库

5. 查看打包好的 .a 静态库
    * 选择 `Product` -> `Show Build Folder in Finder` -> `Products` -> `Debug-iphoneos/Release-iphoneos`，在目录下即可看到 xxxx.a 文件和相应的 .h 头文件

6. 使用
    * `如果打包的 .a 库内部有依赖某个 .framework 库，那么打出来的 .a 库不包含所依赖的 .framework 库`；如果 .a 库内部依赖了其他的 .a 库，那么其他的 .a 库会一同被打包进新的 .a 库中。`使用时需要把头文件、 .a 库、所依赖的 .framework 库一起导入新工程使用`；如果没有依赖库，直接把头文件和 .a 文件拖入新工程即可使用




<br>

### 创建 .framework 静态库

`command + shift + N` -> `iOS` -> `Framework & Library` -> `Framework`

<br>

#### 写完代码后

1. 公开头文件
    * 选择 `TARGETS` 下的静态库项目 - `Build Phases` - `Headers`，把需要公开的头文件从 `Project` 拖入 `Public` 中

2. 设置 Xcode
    * 设置 `General` 下支持的平台；支持的最低 iOS 版本
    * 设置 `Build Active Architecture Only` 下的 `Debug` 和 `Release` 都为 `NO`
    * 设置 `Mach-O Type` 选择 `Static Library`，否则打出来的 .framework 是动态库
    * 如果`涉及 C++ 混编`，需要修改 `Build Settings` -> `Compile Sources As` 为 `Objective-C++`，否则在导入静态库的项目中混编 C++ 代码时，编译器报错
    * `Build Settings` -> `Generate Debug Symbols`，`如果此项设置为 Yes，那么打出来的静态库会暴露源码`，`在调用库方法的地方打断点，单步调试进入就能看到方法源码`。有时为了调试方便，可以设置为 Yes，一般两个都设为 NO，或者只把 `Release 设为 NO`，设为 NO 打出来的库体积也更小。

3. 选择制作 Release 或 Debug 版本的库
    * 点击 项目 -> `Edit Scheme` -> `Run` -> `Info` -> `Build Configuration`，选择 `Debug` 或者 `Release`

4. 打包库
    * 选择某个模拟器，编译运行，打出来的是支持模拟器指令集的 .framework 静态库
    * 选择 `Any iOS Device (arm64)`，打出来的是支持arm64真机使用的 .framework 静态库

5. 查看打包好的 .framework 静态库
    * 选择 `Product` -> `Show Build Folder in Finder` -> `Products` -> `Debug-iphoneos/Release-iphoneos`，在目录下即可看到 xxxx.framework 静态库文件

6. 使用
    * `如果打包的 xxxx.framework 库内部有依赖其他某个 .framework 库，那么打出来的库不包含所依赖的其他 .framework 库`，使用时需要把 xxxx.framework 库和所依赖的 .framework 库一同导入新工程使用；如果 xxxx.framework 依赖某个 .a 库，那么打包出的 xxxx.framework 库就已经包含 .a 库，使用时直接导入 xxxx.framework 到新工程即可。

<br>

### 把 .framework 静态库变成 .a 库

xxx.framework 只是个文件夹，进入 xxx.framework，找到同名的 xxx 无后缀的文件，直接改名成 other.a，选择添加该扩展名到文件末尾，再把 Headers 文件夹中的文件拷贝出来，头文件和 .a 库文件一起使用即可。


<br>

### 查看静态库所支持的指令集

* `lipo -info xxxx.a`  
* `lipo -info xxxx.framework/xxxx`

<br>

### 合并真机和模拟器版本的静态库
#### 合并.a

* `lipo -create Debug-iphoneos/xxxx.a Debug-iphonesimulator/xxxx.a -output xxxx.a`

<br>

#### 合并.framework

* `lipo -create ../xxxx.framework/xxxx ../xxxx.framework/xxxx -output ../xxxx`
* 合并后用生成的 xxxx 文件替换模拟器或者真机的 xxxx.framework 内的同名文件，这样库就能在真机和模拟器上跑

<br>

#### framework合并脚本：

1. 在 `TARGETS` -> `Build Phases` 中点 `+` 加号，选择 `New Run Script Phase`，会添加一项 `Run Script`
2. 将下面的脚本代码粘贴至提示 `Type a script or drag...`输入框内

    ``` zsh
    if [ "${ACTION}" = "build" ]
    then
    INSTALL_DIR=${SRCROOT}/Products/${PROJECT_NAME}.framework
    DEVICE_DIR=${BUILD_ROOT}/${CONFIGURATION}-iphoneos/${PROJECT_NAME}.framework
    SIMULATOR_DIR=${BUILD_ROOT}/${CONFIGURATION}-iphonesimulator/${PROJECT_NAME}.framework
    if [ -d "${INSTALL_DIR}" ]
    then
    rm -rf "${INSTALL_DIR}"
    fi
    mkdir -p "${INSTALL_DIR}"
    cp -R "${DEVICE_DIR}/" "${INSTALL_DIR}/"
    #ditto "${DEVICE_DIR}/Headers" "${INSTALL_DIR}/Headers"
    lipo -create "${DEVICE_DIR}/${PROJECT_NAME}" "${SIMULATOR_DIR}/${PROJECT_NAME}" -output "${INSTALL_DIR}/${PROJECT_NAME}"
    open "${DEVICE_DIR}"
    open "${SRCROOT}/Products"
    fi
    ```
3. 测试脚本，在 `Debug` 或 `Release` 环境下分别选择真机和模拟器运行，运行成功后会自动打开工程根目录下的 `Products` 目录，查看 framework 所支持的指令集


<br>

#### 合并好坏：

* 好：开发过程中更方便，真机和模拟器上都能运行调试
* 坏：合并后静态库大小会变大，因此很多第三方的静态库是区分调试和发布版本的

<br>





## 静态库结合库源码调试

静态库制作完成后，如果使用的人调用报错，为了方便调试，有两种方法：

### 方法一：
打包静态库时，设置 `Build Settings` -> `Generate Debug Symbols`，选择 `Debug` 或者 `Release` 为 `Yes`，这样打出来的静态库，调用库方法崩溃时，会定位到崩溃方法的源码处。

<br>

### 方法二：
把调用静态库的工程和库源码工程相关联

1. 先运行静态库源码工程，生成静态库 xxx.a 或 xxx.framework 文件

2. 关闭静态库源码工程，将库项目文件拖入到使用静态库的工程中
![StaticLibrary_1](/assets/img/StaticLibrary_1.png)

3. 再把 原先的库移除，添加静态库工程里的库
![StaticLibrary_2](/assets/img/StaticLibrary_2.png)
![StaticLibrary_3](/assets/img/StaticLibrary_3.png)

4. 在有被调用的库源码方法里打个断点，运行项目，如果能卡在断点处，就表明关联正常，就可以修改代静态库源码进行调试了。
![StaticLibrary_4](/assets/img/StaticLibrary_4.png)

<br>

## 小结
`打包静态库时，.a 静态库可以被打包进依赖它的静态库中`；`.framework 静态库无法被打包进依赖它的静态库中`，`使用时需要把依赖的 .framework 库一起拷贝到新工程中使用。`

<br>

>  本文由以下几篇文章抄抄改改而来
{: .prompt-info }

* [SDK系列-iOS FrameWork制作概述](https://www.jianshu.com/p/e263aec947ff)
* [SDK系列-FrameWork的制作(1)](https://www.jianshu.com/p/7a88c39f048a)
* [SDK系列-FrameWork制作(2)](https://www.jianshu.com/p/115ba9be4da1)
* [处理器架构介绍](https://marlous.github.io/2019/03/01/处理器、处理器架构与指令集关系/)
