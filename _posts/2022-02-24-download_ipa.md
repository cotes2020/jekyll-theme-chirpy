---
title: 获取已上架的最新版 ipa 安装包的方式
date: 2022-02-24 09:12:52
categories: [iOS]
tags: [零散]
---

# 本文系转载，[原文地址](https://blog.csdn.net/lwb102063/article/details/110739441)

## 获取IPA包的的方式

* 一、使用 `Apple Configurator 2`
* 二、使用 `爱思助手`

<br>

### 一、利用 Apple Configurator 2 获取 IPA 包

之前我们可以借助PP助手来获取越狱或者非越狱后的IPA安装包，但现在PP助手已经凉凉了，不过我们还是有其他的方式可以获取到IPA包的---`Apple Configurator 2`；这款应用我们可以直接在Apple Store上进行下载，https://apps.apple.com/cn/app/apple-configurator-2/id1037126344?mt=12，下面说一下如何获取官方的IPA包：

以下步骤为转载内容：[原文地址](https://www.jianshu.com/p/95440d5ae795)

<br>

1、首先下载一个 `Apple Configurator 2`
![Apple_Configurator_2_1](/assets/img/Apple_Configurator_2_1.png)

<br>

2、连接手机，打开 `Apple Configurator 2`，在 `所有设备` 中找到手机，双击进入手机信息界面
![Apple_Configurator_2_2](/assets/img/Apple_Configurator_2_2.png)

<br>

3、选择左侧应用标签，并在上方点击 `添加` 按钮
![Apple_Configurator_2_3](/assets/img/Apple_Configurator_2_3.png)

<br>
                                               
4、选择 app 下载
![Apple_Configurator_2_4](/assets/img/Apple_Configurator_2_4.png)

<br>

5、登录 App Store ，搜索app名字来下载
![Apple_Configurator_2_5](/assets/img/Apple_Configurator_2_5.png)

<br>

6、下载完成后，会提示手机上已经存在该app，此时，停留到当前弹框状态
![Apple_Configurator_2_6](/assets/img/Apple_Configurator_2_6.png)

<br>

7、此时打开路径，
`~/Library/Group Containers/K36BKF7T3D.group.com.apple.configurator/Library/Caches/Assets/TemporaryItems/MobileApps`
![Apple_Configurator_2_7](/assets/img/Apple_Configurator_2_7.png)

<br>

就可以看到下载的缓存文件。

<br>

#### 问题记录：

在使用过程中可能会出现点击“添加APP”的时候提示没有对应的服务，导致无法添加APP。这个时候点击上方的“账户”，注销Apple Id账号重新登录。

<br>

### 二、使用 爱思助手

爱思助手的官网地址：https://www.i4.cn/ 下载以后，点击 `应用游戏` 标签，如下图：
![ai_si_zhu_shou_1](/assets/img/ai_si_zhu_shou_1.png)

<br>

选择应用后点击安装，在上图的右上角，我们可以查看进度，安装完后，我们可以打开目录，里面就会看到下载下来的 ipa 包了
