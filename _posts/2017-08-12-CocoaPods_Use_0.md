---
title: CocoaPods 的使用
date: 2017-08-12 22:00:13
categories: [iOS]
tags: [CocoaPods]
---

* [Podfile 文件格式](https://guides.cocoapods.org/using/the-podfile.html) 
* [pod install vs. pod update](https://guides.cocoapods.org/using/pod-install-vs-update.html)
* [pod install 和 pod update 的区别](https://www.jianshu.com/p/002306a40dc7)

<br>

## 注意点
* 如果有挂梯子的话就不用换成淘宝的源。
* `Podfile.lock` 文件要加入版本控制。
* 如果你想添加或者更新某一个库，用以下命令：`pod update AFNetworking`

<br>

## 例子
```
platform :ios, '8.0'
use_frameworks!

target 'MyApp' do
    
#    使用最新的 2.x.x 版本
#    pod 'AFNetworking', '~> 2.0'

#    使用最新的 2.5.x 版本
#    pod 'AFNetworking', '~> 2.5.1'

#    使用指定的 2.5.2 版本
#    pod 'AFNetworking', '= 2.5.2'

#    使用最新的版本
    pod 'AFNetworking'
   
end
```
