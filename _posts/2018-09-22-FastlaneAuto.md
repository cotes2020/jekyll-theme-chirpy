---
layout: post
title: "fastlane自动化开发组件"
date: 2018-09-22 23:31:00.000000000 +09:00
categories: [Summary]
tags: [Summary, Fastlane]
---

> 一、什么是自动化:
> 通过简单的一条命令, 去自动执行一组固定操作.
> 二、自动化使用场景：
> 测试、打包上传审核、分发等.

## 自动化实现方案

1. fastlane Fastlane是一个ruby脚本集合.
2. 使用概念 Action机制: Action是Fastlane自动化流程中的最小执行单元，体现在Fastfile脚本中的一个个命令。比如：cocoapods, git_add等等，而这些命令背后都对应一个用Ruby编写的脚本。 [目前所有的Action](https://link.juejin.im?target=https%3A%2F%2Fdocs.fastlane.tools%2Factions%2FActions%2F) [源码链接](<https://docs.fastlane.tools/actions/>) 常用action:

```
produce 创建可用于 iTunes Connect 和 Apple Developer Portal 的 iOS app
cert 自动创建和维护 iOS 代码签名证书
sigh 创建、更新、下载和修复 provisioning profiles
snapshot 自动将 App 屏幕截图本地化到每种设备上
frameit 将屏幕截图适配到适当的设备屏幕大小
gym 创建和打包 iOS app
deliver 上传屏幕截图、元数据和 App 到 App 商店
PEM 自动创建和更新 Push 通知的 profile
```

## 安装fastlane

```bash
// 注意要ruby版本最新
sudo gem install -n /usr/local/bin fastlane
// 更新ruby
brew update
brew install ruby
// 查看版本
fastlane --version
```

## 自动化实现

下列是手动设置和自动化设置的步骤:

```
// 需要手动设置创建的步骤
1. pod lib create XXX。
2. 将代码拖入预定的位置。
3. 关联远程代码仓库。(git remote add origin  + url)
4. 修改描述文件。

// 自动化实现的步骤
1. pod install       // 主要目的是将组件代码给项目有关联
2. 将代码提交到本地仓库。
3. 将代码提交到远程仓库。
4. 检查标签(存在就删除标签)
5. 打标签。
6. 验证描述文件。
7. 提交到私有索引库。
```

初始化fastlane

```
// 里面需要填苹果账号的相关信息，但这步对组件自动化没什么影响，可有可无
fastlane init
```

创建fastlane文件

```
// 文件的路径是：首先在项目文件下创建个文件夹fastlane(如果在fastlane init 这步已经创建，就不用创建), 然后cd到fastlane目录下执行下一步。
touch Fastfile
```

## 在Fastfile文件中描述航道

```ruby
desc ‘ManagerLib 使用这个航道，可以快速的对自己的私有库进行升级维护’
lane :ManagerLib do |options|

tagName = options[:tag]
targetName = options[:target]

# 具体这个航道上面执行的哪些Actions

# 1. pod install
cocoapods(
clean: true,
podfile: “./Example/Podfile”
)

# 2. git add .
git_add(path: “.”)

# 3. git commit -m ‘XXX’
git_commit(path: “.”, message: “版本升级维护”)

# 4. git push origin master
push_to_git_remote

# 验证tag是否存在，如果存在应该删除本地和远程的tag
if git_tag_exists(tag: tagName)
  UI.message(“发现tag:#{tagName} 已经存在，即将执行删除操作 🚀")
  remove_tag(tag: tagName)
end

# 5. git tag 标签名称
add_git_tag(
tag: tagName
)

# 6. git push —-tags
push_git_tags

# 7. pod spec lint
pod_lib_lint(allow_warnings: true)

# 8.pod repo push xxx xxx.podspec
pod_push(path: “#{targetName}.podspec”, repo: “MMSpecs”, allow_warnings: true)

end
```

> 注意:上诉的内容主要用ruby on rail写的，每一步的意思都有解释的，很容易理解。大多的航道Action都可以在[Action](https://link.juejin.im?target=https%3A%2F%2Fdocs.fastlane.tools%2Factions%2FActions%2F)这里找到，但是有一条航道remove_tag(tag: tagName)是自己创建的，下面一步是介绍remove_tag(tag: tagName)。

## 自定义Action

原因: 有些action,并没有人提供; 那么我们可以自己自定来满足我们的需求. 

示例: 在制作私有库的过程中, 如果上一个标签已经存在, 再次创建则会报错 解决方案: 先判断标签是否存在, 如果存在, 则删除标签(本地/远程) 自定义action实现位置remove_tag：

```ruby
# 验证tag是否存在，如果存在应该删除本地和远程的tag
if git_tag_exists(tag: tagName)
  UI.message(“发现tag:#{tagName} 已经存在，即将执行删除操作 🚀")
  remove_tag(tag: tagName)
end
```

接下来主要是实现remove_tag的Action了

```ruby
fastlane new_action
// 命令行需要写action名称，如remove_tag
```

> 注:执行到上述命令后，会在fastlane文件夹中生成一个actions文件夹，里面有个remove_tag.rb文件，打开修改里面的内容。主要是自定义航道。

## remove_tag.rb 需要用ruby编写的脚本

```ruby
module Fastlane
  module Actions
    module SharedValues
      REMOVE_TAG_CUSTOM_VALUE = :REMOVE_TAG_CUSTOM_VALUE
    end

    class RemoveTagAction < Action
      def self.run(params)
      
      tagName = params[:tag]
      isRemoveLocalTag = params[:rL]
      isRemoveRemoteTag = params[:rR]
      
      # 定义一个数据 用来存储所有需要执行的命令
      cmds = []

      # 删除本地标签
      # git tag -d 标签名
      if isRemoveLocalTag
        cmds << "git tag -d #{tagName} "
      end
    
      # 删除远程标签
      # git push origin :标签名
      if isRemoveRemoteTag
        cmds << " git push origin :#{tagName}"
      end
  
      # 执行数组里面所有的命令
      result = Actions.sh(cmds.join('&'));
      return result
  
      end

      def self.description
        "非常牛逼"
      end

      def self.details
        # Optional:
        # this is your chance to provide a more detailed description of this action
        "这个Action是用来删除本地和远程仓库的标签tag"
      end

      def self.available_options
        # Define all options your action supports. 

        # Below a few examples
        [
            FastlaneCore::ConfigItem.new(key: :tag,
                                         description: "需要被删除的标签名称",
                                         optional: false,
                                         is_string: true
                                         ),
            FastlaneCore::ConfigItem.new(key: :rL,
                                         description: "是否需要删除本地标签",
                                         optional: true,
                                         is_string: false,
                                         default_value: true
                                         ),
            FastlaneCore::ConfigItem.new(key: :rR,
                                         description: "是否需要删除远程标签",
                                         optional: true,
                                         is_string: false,
                                         default_value: true
                                         )
        ]
      end

      def self.output

      end

      def self.return_value
        nil
      end

      def self.authors
        # So no one will ever forget your contribution to fastlane :) You are awesome btw!
        ["黄进文-evencoder@163.com"]
      end

      def self.is_supported?(platform)
        # you can do things like
        # 
        #  true
        # 
        #  platform == :ios
        # 
        #  [:ios, :mac].include?(platform)
        # 

        platform == :ios
      end
    end
  end
end
```

## 验证自定义action

验证

```
// cd 到项目目录下(不是fastlane文件夹目录)，执行
fastlane action remove_tag  
// 可能会有红色英文提示，主要是提示不要用文本编辑打开编辑，那个不是错误来的
// 验证没错误的话证明remove_tag这个action就可以用了
```

remove_tag.rb这个action使用的位置

```ruby
// 发现提交的代码版本跟已经存在于本地和远程的tag相同时，需要执行的action进行删除
// 验证tag是否存在，如果存在应该删除本地和远程的tag
if git_tag_exists(tag: tagName)
  UI.message(“发现tag:#{tagName} 已经存在，即将执行删除操作 🚀")
  remove_tag(tag: tagName)
end
```

执行自动化的fastlane航道.

```ruby
// 命令每执行的步骤都提示解释的
fastlane ManagerLib tag:0.1.0 target:MMBase
```