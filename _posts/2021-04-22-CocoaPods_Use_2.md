---
title: CocoaPods 使用：创建公有 pod 库
date: 2021-04-22 20:56:40
categories: iOS
tags: CocoaPods
---

## 准备工作
在创建公开的 pod 库之前，请确保已经有了 `github` 账号。此篇文章介绍如何将代码开源并放到 pod 库中，供别人使用，这种方式属于打造公共（Public repo）仓库，任何人都可以搜索到你的库并使用。

<br>

## 创建 Repository 并完善项目
1. 在 Github 上创建一个 Repository，名为 `SSSYPerson`。
2. 将该代码仓库 `clone` 到本地。
3. 编写测试代码，这里只写了 SSSYPerson.h 和 .m 文件。
4. 执行命令 `pod spec create SSSYPerson`，创建 `SSSYPerson.podspec` 代码库描述文件。
5. 按照文件规范和实际情况修改 `SSSYPerson.podspec` 文件，或者在 Github 上面找一个开源项目参考即可。
6. 注意：`每次更新 pod 库的版本，都要修改 SSSYPerson.podspec 文件里的版本号`

![cocoapods_use_6](/assets/img/cocoapods_use_6.jpg)

最终修改完如下：

```
Pod::Spec.new do |spec|
  spec.name         = "SSSYPerson"
  spec.version      = "0.0.1"
  spec.summary      = "Test CocoaPods."
  spec.description  = <<-DESC
  测试使用 CocoaPods 创建公开库
                   DESC

  spec.homepage     = "https://github.com/cfap/SSSYPerson"
  spec.license      = "MIT"
  spec.author             = { "" => "" }
  spec.platform     = :ios, "10.0"
  spec.source       = { :git => "https://github.com/cfap/SSSYPerson.git", :tag => spec.version }
  spec.source_files  = "Classes", "Classes/**/*.{h,m}"
  spec.exclude_files = "Classes/Exclude"
end
```

<br>

## 检测项目和文件配置的合法性
检查该 .podspec 文件合法性，执行命令
```
pod lib lint SSSYPerson.podspec
```
–verbose 可以输出更加详细的内容

```
pod lib lint SSSYPerson.podspec --verbose
```

`pod lib lint *.podspec` 是只从本地验证你的 pod 能否通过验证。

`pod spec lint *.podspec` 是从本地和远程验证你的 pod 能否通过验证。

如果正确，可以看到下图所示内容：
![cocoapods_use_7](/assets/img/cocoapods_use_7.jpg)

<br>

## 添加项目到 Github Repository
将本地代码 push 到 Github 上的 SSSYPerson 仓库
```
git add .
git commit -m "Init"
git push
```

给仓库打一个 Tag，即版本号，这里第一个版本就用 `0.0.1`，和 `SSSYPerson.podspec` 文件里设置的一样，如图所示:
![cocoapods_use_8](/assets/img/cocoapods_use_8.jpg)

<br>

## 将代码库推送到 CocoaPods
1. 注册 trunk，格式为 `pod trunk register 你的邮箱 ‘用户名’ –description=’简单描述’`，示例:`pod trunk register xxxxx@gmail.com 'TestName' --description='test test'`
2. 打开邮箱，点击邮件里的激活链接
3. 激活成功后，检查注册信息，执行命令如下：`pod trunk me`
4. 把代码库添加到 CocoaPods，执行命令 `pod trunk push SSSYPerson.podspec`

成功的效果图:
![cocoapods_use_9](/assets/img/cocoapods_use_9.jpg)

<br>

## 验证使用
把库提交到 CocoaPods 上之后，可以查看下[CocoaPods 项目的提交记录](https://github.com/CocoaPods/Specs/commits/master)，刷新查看是否有你刚提交的记录
![cocoapods_use_10](/assets/img/cocoapods_use_10.jpg)

<br>

然后在终端检查是否可以搜索到
```
pod search SSSYPerson
```
<br>

如果搜索不到，就先删除索引缓存，再搜索

```
rm ~/Library/Caches/CocoaPods/search_index.json
pod setup
pod repo update
pod search SSSYPerson
```

<br>

在项目中可以使用该 pod 代码库，先修改 `Podfile` 文件

```
pod 'SSSYPerson', '~> 0.0.1'
```
然后在你的测试项目中，执行 `pod install` 即可。

<br>

## 删除 pod 库
如果想删除自己提交到 pod 上的公开库，可以用命令`pod trunk delete SSSYPerson 0.0.1`，一定要指明库名、具体的版本号：

```
$ pod trunk delete {podname} {version}
WARNING: It is generally considered bad behavior to remove versions of a Pod that others are depending on!
Please consider using the `deprecate` command instead.
Are you sure you want to delete this Pod version?
> yes
```

<br>

## 关于 Tag
有时候 tag 打错了并已提交，需要先删除本地 tag，再删远程 tag，然后再添加 tag
```
git tag -d 0.0.1 // 先删除本地 tag
git push origin --delete tag 0.0.1  // 再删除远程 tag，然后重新打 tag
```
