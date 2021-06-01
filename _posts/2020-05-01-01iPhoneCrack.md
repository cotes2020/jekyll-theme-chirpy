---
layout: post
title: "01-iPhone 手机越狱教程"
date: 2020-05-01 14:20:00.000000000 +09:00
categories: [逆向工程]
tags: [逆向工程, 越狱, Jailbreak, checkra1n, unc0ver]
---

## 越狱(Jailbreak)

iOS 越狱（iOS Jailbreaking）是获取[iOS设备](https://zh.wikipedia.org/wiki/IOS设备列表)的[Root权限](https://zh.wikipedia.org/wiki/超级用户)的技术手段。

## 越狱类别

+ 完美越狱

  > 完美越狱（Untethered Jailbreak），指 iOS 设备重启后，仍保留完整越狱状态。

+ 不完美越狱

  > 不完美越狱（Tethered Jailbreak），指的是，当处于此状态的iOS设备开机重启后，之前进行的越狱程序就会失效，用户将失去 Root 权限，需要将设备连接电脑来使用等越狱软件进行引导开机以后，才可再次使用越狱程序。否则设备将无法正常引导。

+ 半不完美越狱

  > 半不完美越狱（Semi-tethered Jailbreak），指设备在重启后，将丢失越狱状态，并恢复成未越狱状态。如果想要恢复越狱环境，必须连接计算机并在越狱工具的引导下引导来恢复越狱状态。

+ 半完美越狱

  > 半完美越狱（Semi-untethered Jailbreak），指设备在重启后，将丢失越狱状态；而若想要再恢复越狱环境，只需在设备上进行某些操作即可恢复越狱。

## 越狱工具

+ [checkra1n](https://checkra.in/)

  > checkra1n是一个社区项目，旨在基于“ checkm8” bootrom漏洞向所有人提供高质量的半不完美越狱。

+ [unc0ver](https://unc0ver.dev/)

  > unc0ver是越狱工具，这意味着您可以自由地对iOS设备执行任何您想做的事情。 unc0ver允许您更改您想要的内容并在您的权限范围内操作，它可以释放iDevice的真正威力。
  >
  > unc0ver 目前可支持 iOS11.0 - iOS14.3 的 iOS 设备越狱。

## 查看是否可以越狱

+ [Can I Jailbreak](https://canijailbreak.com/)

## 爱思助手越狱

+  下载爱思助手，直接越狱。
+ 刷机越狱 -> 一键越狱，等待安装越狱App，我的手机系统是13.6.1，爱思助手给我安装的是Qdyssey，这个软件我没办法越狱，就尝试下另一种方法越狱了。

## unc0ver越狱

+ 安装Altstore，通过Alstore来安装un0ver。
+ https://altstore.io/
+ https://unc0ver.dev/

> 注意: mac新系统的邮件偏好设置那里已经没有`邮件插件`功能了，所以mac系统是macOS Big Sur 11.0.1之后的就通过安装Alstore来安装un0ver了。

+ 通过Windows安装 Alstore。

  + 手机连接电脑，通过上面网址下载windows Alstore安装，如果电脑没有iCloud时，点击AltSever时会提示安装iCloud，下载安装成功后右下角找到Alstore -> Install Alstore。

+ 通过Alstore安装un0ver。

  + 在手机safari那里输入https://unc0ver.dev/ 网址下载un0ver，点击下载好的un0ver，通过Alstore打开安装。

  + 安装成功就可以通过un0ver越狱了，首次打开un0ver会显示未受信任，需要打开设置->通用->描述文件信任即可。

  + 打开un0ver，点击Jailbreak开始越狱。

  + 第一次越狱会重启，再次打开unc0ver；

    再点击点击Jailbreak按钮即可完成；

    桌面出现 Cydia 意味着越狱成功。

    清理越狱方法

    打开unc0ver-左上角设置；

    把全部按钮关闭，仅开启Restore RootFS按钮；

    再点击右上角 Done；

    再点击 Restore RootFS 即可清理；

## 越狱成功安装的插件

+ 1.Filza File Manager
  + 可以在iPhone上自由访问iOS文件系统
+ 2.Apple File Conuit 2
  + 可以访问整个iOS设备的文件系统
+ 3.AppSync Unified
  + 可以绕过系统验证，随意安装、运行破解的ipa安装包。

+ 4.Mac提高工作效率的工具。
  + Alfred
    + 便捷搜索
    + 工作流
  + XtraFinder
    + 增强型Finder
  + iTerm2
    + 完爆Terminal的命令行工具
  + Go2Shell
    + 从Fider快速定位到命令行工具