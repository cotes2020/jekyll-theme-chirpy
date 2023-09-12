---
title: 如何下载 App Store 上旧版软件的 ipa 包
date: 2023-03-14 11:03:25
categories: [iOS]
tags: [零散]
---

## 前言
有些 App 更新至最新版后，可能会发现不喜欢 UI 风格、操作方式、功能等，这时可以通过一些工具安装旧版本。本文介绍的方法，所需的工具都有官网地址，抓包非破解。

<br>

## 需要的工具
* 一、Windows 系统，也可在 macOS 上安装虚拟机，再装 Windows 系统
* 二、Windows 版的 Fiddler 抓包工具，，官网地址：[Fiddler Classic](https://www.telerik.com/fiddler/fiddler-classic)
* 三、Windows 版的 iTunes，一定要带有 App Store 的版本，本文是用的是苹果官方提供的最后一个带 App Store 的版本 v12.6.5.3，官网地址：[64 位](https://secure-appldnld.apple.com/itunes12/091-87819-20180912-69177170-B085-11E8-B6AB-C1D03409AD2A6/iTunes64Setup.exe)、[32 位](https://secure-appldnld.apple.com/itunes12/091-87820-20180912-69177170-B085-11E8-B6AB-C1D03409AD2A5/iTunesSetup.exe)

<br>

## 实现原理
* 用 iTunes 登录自己的 Apple ID，支持国区或外区的 Apple ID
* 通过抓包软件查找 App 的历史版本对应的 ID
* 通过抓包软件修改下载请求，将请求中最新版本 ID 改成想要的历史版本 ID 再执行下载
* 执行下载刚开始时可能看不到进度，过十几秒应该就会显示进度

<br>

## 具体操作步骤
1、 安装完 Fiddler 后，运行软件安装证书，菜单栏依次选择 `Tools -> Options... -> HTTPS -> YES` 选项卡中勾选 `Decrypt HTTPS traffic`，如果有弹窗点 Trust Root Certificate，再勾选最后一项 `Check for certificate revocation -> OK`
2、 重启 Fiddler，在软件下方的黑色文本框内输入 `bpu MZBuy.woa` 然后回车，针对 URL 断点调试 
3、 打开 iTunes，登录 Apple ID，搜索想要的软件，点下载，再回到 Fiddler，找到图标是红色的 URL `https://p50-buy.itunes.apple.com/WebObjects/MZBuy.woa/wa/buyProduct`，选中后点击右边的 `Inspectors -> TextView`
4、 `appExtVrsId` 的值对应的是当前最新的版本号，如果要下载旧版，就要替换这个值，然后放行 URL 
5、 第一遍可以直接点击绿色按钮 `Run to Completion` 放行，然后点击黄色按钮 `Response body is encoded，Click to decode.` 

![download_ipa_1](/assets/img/download_ipa_1.png)


6、 然后点击响应体的 `TextView`，查看数组 `softwareVersionExternalIdentifiers`，这一串数值的最后一个就是最新版对应的版本号，往前倒推，结合 AppStore 去查看自己想要哪个版本，复制对应版本 ID 

![download_ipa_2](/assets/img/download_ipa_2.png)


7、 如果想要 3 年前的版本，就需要网上找对应的 ID，或者倒序往前一个个试？
8、 把下载的最新版文件删掉，再打断点执行执行一次下载，这时先把版本号修改成我们要用的版本号（就是第 4 步里的 appExtVrsId），然后再放行下载，over！

## 以下两个版本号都是 TikTok v21.1.0

844024073
843972181

