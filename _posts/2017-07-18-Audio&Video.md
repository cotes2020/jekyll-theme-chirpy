---
title: 音视频
date: 2017-07-18 16:16:20
categories: iOS 
tags: 音视频
---

# AVPlayer
* 创建初始化 AVPlayer 实例，然后添加到 AVPlayerLayer 层上，再把这个播放层添加到控制器视图的层上，调 play 方法

<br>

# AVPlayerViewController
* 用来替代 MPMoviePlayerViewController(iOS 9 开始被废弃) 播放音视频
* 创建初始化 AVPlayerViewController 实例，设置 .player 属性，present 控制器
* 可通过代码设置 present 成功后直接开始播放，也可让用户手动点击播放
* 可监听缓存进度、播放进度，设置播放倍速、音量，或者总时长，设置从某个时间点开始播放等
* 画中画功能只支持 iOS 9+ 的 iPad 设备，不支持 iPhone，不过越狱的 iPhone 可以支持


<br>

# 引用
[音频掌柜 - AVAudioSession](https://www.jianshu.com/p/3e0a399380df)
[音视频播放 - AVPlayer](https://www.cnblogs.com/QianChia/p/5771172.html)
[在 iOS 上捕获视频](https://objccn.io/issue-23-1/)
[音视频业务](https://www.samirchen.com/ios-index/)
[码农人生](https://msching.github.io/blog/categories/)

<br>
<br>
<br>

