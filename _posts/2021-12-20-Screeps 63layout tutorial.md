---
layout: post
title: Screeps 63抠位置自动布局 使用教程
date: 2021-12-20 22:06:00 +0800
Author: Sokranotes
tags: [recording, Screeps]
comments: true
categories: recording
toc: true
typora-root-url: ..

---

# Screeps 63抠位置自动布局 使用教程

## 获取63抠位置自动布局

1. 加群 Screeps>_编程游戏小组：291065849

   ![image-20221029104752098](/assets/img/2021-12-20-Screeps 63layout tutorial/image-20221029104752098.png)

2. 从群文件中下载`63超级抠位置自动布局 傻瓜版.7z`并解压7z文件

## 将代码及库导入到游戏中

1. 打开游戏进入到目标房间

   <img src="/assets/img/2021-12-20-Screeps 63layout tutorial/image-20221029204808844.png" alt="image-20221029204808844"/>

2. 在Script选项卡下新增名为`algo_wasm_priorityqueue`的二进制模块

   ![image-20221029205100173](/assets/img/2021-12-20-Screeps 63layout tutorial/image-20221029205100173.png)

3. 新增完成后上传二进制文件![image-20221029205353299](/assets/img/2021-12-20-Screeps 63layout tutorial/image-20221029205353299.png)

4. 找到解压后的7z文件夹并上传`algo_wasm_priorityqueue.wasm`![image-20221029205504419](/assets/img/2021-12-20-Screeps 63layout tutorial/image-20221029205504419.png)

5. 将main模块中的内容替换为`63超级抠位置自动布局 傻瓜版.js`的内容并保存（**注意保存好自己的代码**）![image-20221029205917955](/assets/img/2021-12-20-Screeps 63layout tutorial/image-20221029205917955.png)

6. 依次在房间两个能量矿上插两个flag，分别为pa和pb，在矿物矿上插flag pm，在控制器上插flag pc

7. 再插一个flag，名字为p，插下去之后注意抓紧时间截图。效果图如下：

   根据显示效果可以再适当做一些调整，不一定要完全按照显示的布局。如有bug可以在群里反馈给63。

   ![image-20221029210202718](/assets/img/2021-12-20-Screeps 63layout tutorial/image-20221029210202718.png)

   ![image-20221029210335302](/assets/img/2021-12-20-Screeps 63layout tutorial/image-20221029210335302.png)

   ![image-20221029210342339](/assets/img/2021-12-20-Screeps 63layout tutorial/image-20221029210342339.png)

## 图例（应该猜也能猜到吧）

![image-20221029210513758](/assets/img/2021-12-20-Screeps 63layout tutorial/image-20221029210513758.png)

## 鸣谢

63（[Screeps:6g3y](https://screeps.com/a/#!/profile/6g3y)）

