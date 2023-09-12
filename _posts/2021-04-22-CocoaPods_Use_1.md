---
title: CocoaPods 使用：创建本地 pod 库
date: 2021-04-22 17:06:49
categories: iOS
tags: CocoaPods
---

<br>

## 使用 CocoaPods 管理代码的必要性
项目到了一定规模, 代码组织和结构显得尤为重要。

重构项目结构，可以从分离代码开始。代码分离，可以按功能划分，把常用、稳定且和业务无关的代码封装成组件，抽离出来。

分离代码, 常用的有几种方式：

* 放到不同的文件夹, 管理和组织代码。（源码可见）
* 打包成静态库 `.a` 或者 `.framework` 提供给项目使用。（只能调 API，看不到源码实现）
* 使用工具管理，如 `CocoaPods`（统一管理，源码可见）

以下介绍下使用 `CocoaPods` 来管理自己的本地代码。

<br>

## 创建工程
1. 用 Xcode 创建一个测试工程 `TestPods`
2. 在工程 `TestPods` 目录下面创建 `LocalLib` 目录，用来放置分离的代码，也可以将其放到其他目录。
3. 在 `LocalLib` 下面，pod 代码库名称为 `LibOne`，代码放在 `Codes` 目录下

![cocoapods_use_0](/assets/img/cocoapods_use_0.jpg)


<br>

## 创建库的 .podspec 描述文件
在终端进入 `LocalLib` 目录下，用命令行创建 podspec 文件，创建完成后, 会生成 `LibOne.podspec` 文件，如上图所示。

```
pod spec create LibOne
```

<br>

## 修改 podspec 文件
修改 `LibOne.podspec` 文件，主要修改几个关键地方：

1. spec.description
2. spec.license
3. spec.source
4. spec.source_files
5. spec.exclude_files

![cocoapods_use_1](/assets/img/cocoapods_use_1.jpg)
![cocoapods_use_2](/assets/img/cocoapods_use_2.jpg)
![cocoapods_use_3](/assets/img/cocoapods_use_3.jpg)


配置好相关描述信息，不要包含 ‘Example’ 的字样，不然新版的 CocoaPods 执行 pod install 时候会报错误和警告。

这样，工程就可以使用本地 pod 库了。

## 工程使用本地 pod 库
将 TestPods 改为 CocoaPods 项目，在 TestPods 目录，执行命令

``` zsh
pod init
```

<br>

然后会生成 Podfile 文件，再修改 Podfile 文件


```
platform :ios, '14.4'

# 测试用 CocoaPods 制作本地库，此处定义的名称必须以小写字母开头？
def localRepos

  # 如果存放本地库的 LocalLib 文件夹是放在和 Podfile 文件同一层级的地方，以下写法都可以
  # pod 'LibOne', :path => 'LocalLib'
  # pod 'LibOne', :path => 'LocalLib/'
  # pod 'LibOne', :path => './LocalLib'
  pod 'LibOne', :path => './LocalLib/'

end

target 'TestPods' do
  use_frameworks!

  localRepos

end
```

关键是指明 pod 库的位置，路径一定要正确，否则无法找到该库。

在 TestPods 目录下，执行命令，即可安装 pod 本地库。

``` zsh
pod install
```

<br>

安装成功后的提示语如下：
![cocoapods_use_4](/assets/img/cocoapods_use_4.jpg)


如果报错，一般都是 pod 库的配置文件(.podspec)里面写的不符合要求。根据报错信息，加以修改即可。

<br>
<br>

Xcode 打开工程
![cocoapods_use_5](/assets/img/cocoapods_use_5.jpg)


