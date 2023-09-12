---
title: CocoaPods 使用：创建私有 pod 库
date: 2021-04-28 11:29:09
categories: iOS
tags: CocoaPods
---

## 使用场景

有些项目或公司内，不希望把一些核心的公用代码开源，但是这些代码基本很稳定很成熟，可以做成组件给到其他人或者组内使用。就可以使用 Cocoapods 来创建自己的私有仓库，让大家共享代码，也是组件化的一种方案。

由于现实原因，国内访问 Github 和 CocoaPods 网站比较蛋疼，这里就用国内的 gitee 来做自私有库，效果和放在 Github 上一样，速度还快。如果放在 Github 上，操作方式一样。

以下就讲述如何创建自己的私有仓库以及如何使用私有仓库。

<br>

## 创建仓库
我们需要两个仓库，一个是代码仓库，另一个是 pod 源的仓库。我们在 Podfile 文件里，如果没写明 pod 源地址，默认就是 `https://github.com/CocoaPods/Specs.git`，当我们使用了私有库时，就需要指明私有库的具体源地址，如下
```
source   'https://github.com/CocoaPods/Specs.git' # 公有库源地址
source   'https://gitee.com/extras/Specs.git'     # 私有库源地址

use_frameworks!

platform :ios, '9.0'

target 'SSSKit_Example' do
    pod 'SSSKit'
    pod 'AFNetworking'
end

```

<br>

### 创建私有 pod 源仓库
在码云（https://gitee.com）上创建私有 pod 源仓库，用来存放私有代码库的详细描述信息 .podspec 文件，如下图：
![cocoapods_use_15](/assets/img/cocoapods_use_15.jpg)

<br>

用 `pod repo` 查看当前本地已存在的索引库
``` zsh
tem@temdeMacBook-Pro Documents % pod repo                                                  

cocoapods
- Type: git (remotes/origin/master)
- URL:  https://github.com/CocoaPods/Specs.git
- Path: /Users/tem/.cocoapods/repos/cocoapods

trunk
- Type: CDN
- URL:  https://cdn.cocoapods.org/
- Path: /Users/tem/.cocoapods/repos/trunk

2 repos
```

<br>

将上一步创建的库 clone 到本地，添加刚创建的私有 pod 仓库源
```
pod repo add SWSpecs https://gitee.com/extrass/SWSpecs.git
```

<br>

再次用 `pod repo` 查看本地已存在的索引库

``` zsh
tem@temdeMacBook-Pro Documents % pod repo                                                  

cocoapods
- Type: git (remotes/origin/master)
- URL:  https://github.com/CocoaPods/Specs.git
- Path: /Users/tem/.cocoapods/repos/cocoapods

SWSpecs
- Type: git (master)
- URL:  https://gitee.com/extrass/SWSpecs.git
- Path: /Users/tem/.cocoapods/repos/SWSpecs

trunk
- Type: CDN
- URL:  https://cdn.cocoapods.org/
- Path: /Users/tem/.cocoapods/repos/trunk

3 repos
```
           
<br>

### 创建一个用来存放项目基础组件代码的仓库 ABCDKit 
远程创建代码库
![cocoapods_use_18](/assets/img/cocoapods_use_18.jpg)

<br>

本地快速创建 ABCDKit 目录及 ABCDKit.podspec 文件
``` zsh
pod lib create ABCDKit
```
![cocoapods_use_16](/assets/img/cocoapods_use_16.jpg)

<br>

填写以上信息后 Xcode 会自动打开测试工程，在测试模板工程文件夹下，我们可以看到如下：  
![cocoapods_use_17](/assets/img/cocoapods_use_17.jpg)

<br>

修改 ABCDKit.podspec 文件
![cocoapods_use_22](/assets/img/cocoapods_use_22.jpg)

<br>

验证 ABCDKit.podspec 文件有效性 `pod lib lint --allow-warnings`
![cocoapods_use_20](/assets/img/cocoapods_use_20.jpg)

<br>

将本地代码库关联远程库，并提交 git 改变
``` zsh
git remote add origin https://gitee.com/extrass/ABCDKit.git（关联远程库）
git pull origin master --allow-unrelated-histories（拉取远程库）
git status
git add .
git commit -m 
git push origin master（推送到远程仓库 master 分支）
```

<br>

创建版本号 tag

``` zsh
git tag '0.1.0' (要与 ABCDKit.podspec 文件中的 version 值保持一致)
git push --tags (将 tag 提交到远程)
pod spec lint --allow-warnings（验证有效性，注意 Username 与 Password 要填写正确）
```

<br>

### 将 podspec 文件提交到本地的私有索引库

``` zsh
pod repo
pod repo push swspecs ABCDKit.podspec
```

注意：提交后，依然会验证 podspec 文件，验证通过后 会自动上传到在远程 spec 索引库，可以看看在第二步创建的 SWSpecs 远程私有索引库，是不是多了一个 ABCDKit/0.1.0 文件！

<br>

### 更新私有代码库及其版本号

修改代码后，也要修改 ABCDKit.podspec 文件里的版本号
``` zsh
git status
git add .
git commit -m '新增内容'
git push origin master (提交到远程)
git tag '0.1.1'（和 ABCDKit.podspec 文件里的版本号要一致）
git push --tags
pod repo push swspecs ABCDKit.podspec
```

<br>

### 在基础组件 ABCDKit 内部生成子库
当我们执行完上面的步骤后，发现主工程的 Pods 中并没有按文件夹（One，Two）进行不同类的划分，同时当我们仅仅想引入一个 One 库时，连同 Two 也引入进项目中，冗余代码量增加。这是可以修改 ABCDKit.podspec 文件，再更新库的版本。
![cocoapods_use_21](/assets/img/cocoapods_use_21.jpg)    

至此生成子库

<br>

## 主工程使用生成的私有框架

```
source   'https://github.com/CocoaPods/Specs.git' # 公有库地址
source   'https://gitee.com/extrass/swspecs.git'     # 私有库地址

use_frameworks!

platform :ios, '9.0'

target 'ABCDKit_Example' do
#  pod 'ABCDKit'
#  pod 'ABCDKit/Two'
  pod 'ABCDKit/One'
  pod 'MJRefresh'
end
```
