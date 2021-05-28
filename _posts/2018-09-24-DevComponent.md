---
layout: post
title: "组件化开发之开发小组件"
date: 2018-09-24 22:45:00.000000000 +09:00
categories: [Summary]
tags: [Summary, 组件化]
---

## 前言

概念:
 将一个单一工程的项目，分解成为各个独立的组件，然后按照某种方式，任意组织成一个拥有完整业务逻辑的项目。

产生原因:
 如果是单一工程，业务比较少，人数比较少，一般的开发模式没有任何问题。但是一个项目发展慢慢庞大是，业务主线增多，开发人员增多，就会暴露出一系列问题。如: 耦合比较严重、编译速度慢、测试不独立、无法使用自己擅长的设计模式等等。

组件化可达到的效果:

1. 组件独立。独立编写、独立编译、独立运行、独立测试。
2. 资源重用。功能代码可以重复使用。
3. 高效迭代。增删模块。
4. 可配合二进制化, 最大化的提高项目编译速度。

## 1.创建远程私有索引库

在coding代码托管那里创建一个远程私有索引库。例如名称为MMSpecs.

## 2.克隆远程索引库到本地

```
pod repo add MMSpecs + url
// 注释: url 是远程私有索引库的HTTPS或者SSH的url  
// MMSpecs 是本地索引库的名称
```

有关pod repo 命令:

```
// 删除本地索引库 删除后可以克隆远程索引库。
pod repo remove MMSpecs 
```

## 3.创建一个组件模板库

例如创建一个基础组件

```
pod lib create MMBase 
// 命令行需添入 
// What is your email? 随便填一个邮件名就可以
// What language do you want to use?? [ Swift / ObjC ] 选择的语言:Objc
// Would you like to include a demo application with your library? [ Yes / No ] 是否需要一个测试Demo: Yes
// Which testing frameworks will you use? [ Specta / Kiwi / None ] 选None
// Would you like to do view based testing? [ Yes / No ] 选No
// What is your class prefix? 文件前缀 如: MM
```

## 4.基础组件

将基础组件的相关代码拖入Base文件中Classes文件下，删掉ReplaceMe.m文件。

## 5.修改描述文件Base.podspec

修改描述文件Base.podspec，描述文件需要修改的内容如下:

```swift
// 组件描述信息，内容随意
s.summary          = 'Base.'
// 组件描述信息，内容写在中间那里，内容随意
s.description      = <<-DESC
Base.基础组件，主要包含基本的配置、分类、工具类等。
                       DESC
// 远程代码仓库的首页。如https://coding.net/u/evenCoder/p/Base
s.homepage         = 'https://coding.net/u/evenCoder/p/Base'
// 远程代码仓库的HTTPS或SSH
s.source           = { :git => 'https://git.coding.net/evenCoder/Base.git', :tag => s.version.to_s }
// 组件如果不需要分层的话不需要修改
s.source_files = 'Base/Classes/**/*'
// 组件分层需要这样修改，先注释掉  #s.source_files = 'Base/Classes/**/*'
 s.subspec 'Bases' do |b|
  b.source_files = 'Base/Classes/Bases/**/*'
  end
  s.subspec 'Category' do |c|
  c.source_files = 'Base/Classes/Category/**/*'
  end
  s.subspec 'Network' do |n|
  n.source_files = 'Base/Classes/Network/**/*'
  n.dependency 'AFNetworking'
  n.dependency 'SDWebImage'
  end
  s.subspec 'Tool' do |t|
  t.source_files = 'Base/Classes/Tool/**/*'
  end
// 加入组件中用其他资源在 ../Base/Assets路径下，假如有图片。需要修改
s.resource_bundles = {
     'Base' => ['Base/Assets/*']
}
// 而且加载图片写法要这样写: 
NSBundle *currentBundle = [NSBundle bundleForClass:[self class]];
NSString *imagePath = [currentBundle pathForResource:@"tabbar_bg@2x.png" ofType:nil inDirectory:@"Base.bundle"];
UIImage *image = [UIImage imageWithContentsOfFile:imagePath];
    self.backgroundImage = image;
// 依赖系统的动态库的SQLite3
s.library = 'sqlite3'
// 依赖第三方框架
s.dependency 'SDWebImage'
```

> 注: 还有一些地方要注意的，如s.resource_bundles那里，假如Assets里面有xib文件，它的加载方式也要改。

## 6.创建远程仓库Base

注意几个url:

```bash
// homepage 
https://coding.net/u/evenCoder/p/Base 
// HTTPS
https://git.coding.net/evenCoder/Base.git
// SSH 这个要生成公钥才可以用
git@git.coding.net:evenCoder/Base.git
```

## 7.提交代码

将Base路径下的代码全提交

```bash
git add .      // 将代码提交到暂缓区
git commit -m '初始化版本'    // 将代码提交到本地仓库
git remote    // 查看本地仓库是否关联到远程仓库
git remote add origin https://git.coding.net/evenCoder/Base.git     // 关联远程仓库
git push origin master   // 将代码提交到远程仓库
git tag   // 查看是否有打标签
git tag -a '0.1.0' -m '打标签'    // 标签0.1.0要跟描述文件的版本号一致
git push --tags  // 提交到远程仓库
```

## 8.验证spec描述文件

```
pod lib lint   // 本地验证
pod spec lint   // 远程验证
// 注释:假如这部分验证有错误，那么可能是描述文件写法可能有误，需要重新修改。修改成功后重新提交代码，但这步要注意，之前有打过的本地标签和远程标签需要删除，完成这步后再提交代码、验证描述文件
git tag -d 0.1.0   // 删除本地标签
git push origin :0.1.0  // 删除远程标签
```

> 注意: 验证描述文件时有个地方要注意就是，当组件中依赖库s.dependency是自己创建的私有库时，那么在验证描述文件时有警告信息，但是没关系，尽管将描述文件提交。

## 9.将描述文件提交到远程私有索引库

前提是本地私有索引库跟远程私有索引库关联

```
pod repo push MMSpecs Base.podspec
```

## 10.将小组件pod下来进行测试

假如你工程有用到第三方框架和自己的私有组件时，那么你的Podfile文件需要假如两个source

```swift
source ‘https://github.com/CocoaPods/Specs.git’   // Github
source ‘https://git.coding.net/evenCoder/MMSpecs.git’  // 私有的
platform :ios, '9.0'
target 'MMFM' do
    use_frameworks!

    pod ‘MMBase/Base’
    pod ‘MMBase/Category’
    pod ‘MMBase/Network’
    pod ‘MMBase/Tool’
end
```

