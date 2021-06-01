---
layout: post
title: "04-逆向App"
date: 2020-05-01 18:02:00.000000000 +09:00
categories: [逆向工程]
tags: [逆向工程, 逆向App]
---

## 1.逆向App的思路

+ 界面分析

  + 借助工具分析App.

  + `Cycript`、`Reveal`

+ 代码分析

  + 对`Mach-O`文件的静态分析。
  + `MachOView`、`class-dump`、`Hopper Disassembler`、ida等。

+ 动态调试

  + 对运行中的App进行代码调试。
  + `Debugserver`、`LLDB`

+ 代码编写

  + 注入代码到App中
  + 必要时还可能需要重新签名、打包ipa.

## 2.class-dump

+ 顾名思义，它的作用就是把Mach-O文件的class信息给dump出来(把类信息给导出来)，生成对应的.h文件。
+ [官方地址](http://stevenygard.com/projects/class-dump/)
+ 下载完工具包后将class-dump文件复制到/usr/local/bin目录下，这样在终端就能识别class-dump命令了
+ 常用格式
  + `OC`
    + `class-dump -H Mach-O文件路径 -o 头文件存放目录`
  + `Swift`
    + `class-dump -S -s -H Mach-O文件路径 -o 头文件存放目录`