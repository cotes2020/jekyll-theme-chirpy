---
layout: post
title: "05-动态库缓存"
date: 2020-05-01 22:31:00.000000000 +09:00
categories: [逆向工程]
tags: [逆向工程, dyld]
---

## 动态库共享缓存(dyld shared cache)

+ 问题: 找出UIKit的`Mach-O`文件

+ 从iOS13.1开始，为了提高性能，绝大部分的系统动态库文件都打包存放到一个缓存文件中(`dyld shared cache`)

+ 缓存路径,

  + 手机(/System/Library/Caches/com.apple.dyld/dyld_shared_cache_arm)

+ dyld_shared_cache_arm`X`的`X`代表ARM处理器指令集架构

  + v6

  ```
  iPhone、iPhone3G、iPod Touch、iPod Touch2
  ```

  + v7

  ```
  iPhone3GS、iPhone4、iPhone4s、iPad、iPad2、iPad3、iPad mini
  iPod Touch3G、iPod Touch4、iPod Touch5
  ```

  + v7s

  ```
  iPhone5、iPhone5C、iPad4
  ```

  + arm64

  ```
  iPhone5S、iPhone6、iPhone6 Plus、iPhone6S、iPhone6S Plus
  iPhoneSE、iPhone7、iPhone7 Plus、iPhone8、iPhone8 Plus
  iPhoneX
  iPad5、iPad Air、 iPad Air2、iPad Pro、iPad Pro2
  iPad mini with Retina display、iPad mini3、iPad mini4
  iPod Touch6
  ```

+ 所有指令集原则上都是向下兼容的

+ 动态库共享缓存有一个非常明显的好处是节省内存

+ 现在的ida、Hopper反编译工具都可以识别动态库共享缓存

## 动态库加载

+ 在Mac\iOS中，是使用/usr/lib/dyld程序来加载动态库
+ dyld
  + `Dynamic link editor` 动态连接编辑器
  + `dynamic loader` 动态加载器

## 抽取动态库

+ 从动态库共享缓存`dyld_shared_cache_arm64`抽取动态库。

+ [源码](https://opensource.apple.com/traballs/dyld/)，在网站上面下载好的`dyld`源码，从该项目获取dsc_extractor编译文件。

+ 将#if 0前面的代码删除（包括#if 0），把最后面的#endif也删掉。

+ 编译`dsc_extractor.cpp`

  ```
  $ cland++ dsc_extractor.cpp // 文件名是 a.o
  $ clang++ -o dsc_extractor dsc_extractor.cpp // 文件是dsc_extractor.o
  ```

+ 将编译好的`dsc_extractor.o`复制到`dyld_shared_cache_arm64`共同下

  ```
  $ cd // dyld_shared_cache_arm64目录下
  // 开始抽取动态库
  $ ./dsc_extractor dyld_shared_cache_arm64 arm64
  ```

## Mach-O

+ Mach-O是Mach oject的缩写，是Mac/iOS上用于存储程序、库的标准格式
+ 属于Mach-O格式的文件类型有

```
#define MH_OBJECT 			0x1
#define MH_EXECUTE 	 		0x2
#define MH_FVMELIB 			0x3
#define MH_CORE					0x4
#define MH_PRELOAD			0x5
#define MH_DYLIB				0x6
#define MH_DYLINKER			0x7
#define MH_BUNDLE				0x8	
#define MH_DYLIB_STUB		0x9
#define MH_DSYM					0xa
#define MH_KEXT_BUNDLE 	0xB
```

+ 可以在xnu源码中，查看到`Mach-O`格式的详细定义[源码](https://opensource.apple.com/traballs/xnu)

  + `EXTERNAL_HEADERS/mach-o/fat.h`
  + `EXTERNAL_HEADERS/mach-o/load.h`

+ 常见的Mach-O文件类型

  + MH_OBJECT
    + 目标文件(.o)
    + 静态文件(.a)，静态库骑士就是N个.o文件合并在一起的
  + MH_EXECUTE: 可执行文件
  + MH_DYLIB: 动态库文件
    + .dylib
    + .framework/xx
  + MH_DYLINKER: 动态链接编辑器
    + /usr/lib/dyld
  + MH_DSYM: 存储着二进制文件符号信息的文件
    + .dSYM/Contents/Resources/DWARF/xx(常用语分析App的崩溃信息)

+ `Universal Binary`(通用二进制文件)

  + 同时使用与多种架构的二进制文件
  + 包含了多种不同架构的独立的二进制文件

  > 因为需要存储多种 架构的代码，通用二进制文件通常比单一平台二进制的进程要大
  >
  > 由于两种架构有共同的一些资源，所以并不会达到单一版本的两本只多
  >
  > 由于执行过程中，只调用一部分代码，运行起来也不需要额外的内存
  >
  > 因为文件比原来的要大，也被称为 “胖二进制文件”(Fat Binary)

+ 查看Mach-O文件的架构

  + 命令行工具

    + `file`: 查看Mach-O文件类型

      > file 文件路径

    + `otool`: 查看Mach-O特定部分和段的内容

      > 查看依赖动态库: otool -L 文件路径
      >
      > 查看头文件信息: otool -h 文件路径

    + `lipo`: 常用语多架构的Mach-O文件的处理

      > 查看架构信息: lipo -info 文件路径
      >
      > 导出某种特定架构: lipo 文件路径 -thin 架构类型 -ouput 输出路径
      >
      > 合并多种架构: lipo 文件路径1 文件路径2 -output 输出路径

    + GUI工具

      + [MachOView](https://github.com/gdbinit/MachOView)

  ```
  $ lipo -info Test  // file Test
  // 把单一的armv7架构抽出来
  $ lipo Test -thin armv7 -output Test_armv7 
  // 创建一个
  $ lipo -create Test_arm64 Test_armv7 -output Test
  ```

+ Mach-O的基本结构

  + [官方描述](https://developer.apple.com/library/content/documentation/DeveloperTools/Conceptual/MachOTopics/0-Introduction/introduction.html)
  + 一个Mach-O文件包含3个主要区域
    + Header: 文件类型、目标架构类型等
    + Load commands: 描述文件在虚拟内存中的逻辑结构、布局
    + Raw segment data: 在Load commands中定义的Segment的原始数据
  
+ dyld和Mach-O

  + dyld是MH_DYLINKER类型，用于加载以下类型的Mach-O文件
    + MH_EXECUTE、MH_DYLIB、MH_BUNDLE
  + APP的可执行文件、动态库都是有dyld负责加载的，但不能加载自己。 
