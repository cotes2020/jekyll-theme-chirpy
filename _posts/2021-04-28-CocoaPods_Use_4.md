---
title: CocoaPods 使用：podspec 文件参数详解
date: 2021-04-28 17:31:17
categories: [iOS]
tags: [CocoaPods]
---

> 【[转自这里](https://www.jianshu.com/p/5d24c03f24d3)】【[另](https://www.jianshu.com/p/6e0988a01db9)】

<br>

## 一、podspec 文件讲解

podspec 是一个描述 pod 库版本文件，一个标准的 .podspec 文件可以通过 `pod spec create xxx` 命令生成，生成的文件名为 `xxx.podspec`

<br>

### 简单示例：
 
```
Pod::Spec.new do |spec|
     spec.name         = 'Reachability'
     spec.version      = '3.1.0'
     spec.license      = { :type => 'BSD' }
     spec.homepage     = 'https://github.com/tonymillion/Reachability'
     spec.authors      = { 'Tony Million' => 'tonymillion@gmail.com' }
     spec.summary      = 'ARC and GCD Compatible Reachability Class for iOS and OS X.'
     spec.source       = { :git => 'https://github.com/tonymillion/Reachability.git', :tag => 'v3.1.0' }
     spec.source_files = 'Reachability.{h,m}'
     spec.framework    = 'SystemConfiguration'
 end
```

<br>

### 详细示例：
```
 Pod::Spec.new do |spec|
     spec.name          = 'Reachability'
     spec.version       = '3.1.0'
     spec.license       = { :type => 'BSD' }
     spec.homepage      = 'https://github.com/tonymillion/Reachability'
     spec.authors       = { 'Tony Million' => 'tonymillion@gmail.com' }
     spec.summary       = 'ARC and GCD Compatible Reachability Class for iOS and OS X.'
     spec.source        = { :git => 'https://github.com/tonymillion/Reachability.git', :t            ag => 'v3.1.0’ }    
     spec.module_name   = 'Rich'
     spec.swift_version = '4.0'
     spec.ios.deployment_target  = '9.0'
     spec.osx.deployment_target  = '10.10'
     spec.source_files       = 'Reachability/common/*.swift'
     spec.ios.source_files   = 'Reachability/ios/*.swift', 'Reachability/extensions/*.swift'
     spec.osx.source_files   = 'Reachability/osx/*.swift'
     spec.framework      = 'SystemConfiguration'
     spec.ios.framework  = 'UIKit'
     spec.osx.framework  = 'AppKit'
     spec.dependency 'SomeOtherPod'
 end
```

<br>

## 二、描述库的特定版本信息
```
Pod::Spec.new do |spec|
    //库名称
    spec.name          = ‘Reachability’

    //库版本号，这也是我们podfile文件指定的版本号。 每次发布版本都需要打tag标签（名称就是版本号）
    spec.version       = ‘3.1.0'

    //许可证，除非源代码包含了LICENSE.*或者LICENCE.*文件，否则必须指定许可证文件。文件扩展名可以没有，或者是.txt,.md,.markdown
    spec.license       = { :type => 'BSD’ }
    or spec.license = ‘MIT'
    or spec.license = { :type => 'MIT', :file => 'MIT-LICENSE.txt’ }
    or spec.license = { :type => 'MIT', :text => <<-LICENSE
                        Copyright 2012
                         Permission is granted to...
                         LICENSE
                      }

    //pod主页
    spec.homepage      = 'https://github.com/tonymillion/Reachability’      

    //pod库维护者的名车和邮箱
    spec.authors       = { 'Tony Million' => 'tonymillion@gmail.com’ } 
    or spec.author = 'Darth Vader'
    or spec.authors = 'Darth Vader', 'Wookiee'
    or spec.authors = { 'Darth Vader' => 'darthvader@darkside.com',
                    'Wookiee'     => 'wookiee@aggrrttaaggrrt.com' }     

    //指定多媒体地址，如果是推特发布版本会有通知
    spec.social_media_url = 'https: //twitter.com/cocoapods 
    or spec.social_media_url = 'https: //groups.google.com/forum/#!forum/cocoapods'
    
    //pod简介
    spec.summary = 'ARC and GCD Compatible Reachability Class for iOS and OS X.'
        
    //详细描述
    spec.description = <<-DESC                     Computes the meaning of life.
                 Features:
                 1. Is self aware
                 ...
                 42. Likes candies.
               DESC
    
    //获取库的地址
    a. Git：git地址，tag:值以v开头，支持子模块
        spec.source        = { :git => 'https://github.com/tonymillion/Reachability.git', :tag => 'v3.1.0’ }
        spec.source = { :git => 'https://github.com/typhoon-framework/Typhoon.git',
                :tag => "v#{spec.version}", :submodules => true }
    b. Svn:svn地址
        spec.source = { :svn => 'https://svn.code.sf.net/p/polyclipping/code', :tag => ‘4.8.8‘ }
    c. Hg:Mercurial
        spec.source = { :hg => 'https://bitbucket.org/dcutting/hyperbek', :revision => "#{s.version}" }
    
    // Pod 屏幕截图，支持单个或者数组，主要适用于UI类的pod库。cocoapods推荐使用gif
    spec.screenshot  = 'https://dl.dropbox.com/u/378729/MBProgressHUD/1.png'
    or spec.screenshots = [ 'https://dl.dropbox.com/u/378729/MBProgressHUD/1.png',
                            'https://dl.dropbox.com/u/378729/MBProgressHUD/2.png' ]
    
    // 说明文档地址
    spec.documentation_url  =  'https://www.example.com/docs.html’
    
    // pod下载完成之后，执行的命令。可以创建，删除，修改任何下载的文件。该命令在pod清理之前和pod创建之前执行。
    spec.prepare_command = 'ruby build_files.rb'
    or spec.prepare_command = <<-CMD                        sed -i 's/MyNameSpacedHeader/Header/g' ./**/*.h
                    sed -i 's/MyNameOtherSpacedHeader/OtherHeader/g' ./**/*.h
             CMD
            
    //module name
    spec.module_name   = ‘Rich'
    //支持的swift版本
    spec.swift_version = ‘4.0'
    // 支持的Cocoapods版本
    spec.cocoapods_version = ‘>=0.36’
    
    // 是否使用静态库。如果podfile指明了use_frameworks!命令，但是pod仓库需要使用静态库则需要设置
    spec.static_framework = true
    
    // 库是否废弃
    spec.deprecated = true

    //  废弃的pod名称
    spec.deprecated_in_favor_of = 'NewMoreAwesomePod'

    // pod支持的平台，如果没有设置意味着支持所有平台,使用deployment_target支持选择多个平台
    spec.platform = :osx,  ’10.8'
    or spec.platform = :ios
    or spec.platform = :osx         
            
    // 可以指定多个不同平台
    spec.ios.deployment_target =  ‘6.0'
    or .osx.deployment_target = '10.8'
```

<br>

## 三、配置 pod库 工程环境变量

1. dependency： 私有库依赖的三方库

    ```
    spec.dependency 'AFNetworking', '~> 1.0'
    ```
2. requires_arc: 指定私有库 文件是否 是ARC.默认是true,表示所有的 source_files是arc文件
```
spec.requires_arc = true
//指定除了Arc文件下的是arc，其余的全还mrc，会添加-fno-objc-arc 编辑标记
spec.requires_arc = false       spec.requires_arc = 'Classes/Arc'
spec.requires_arc = ['Classes/*ARC.m', 'Classes/ARC.mm']
```
注意：spec.requires_arc 指定的路径表示是arc文件，不被指定才会被标记 -fno-objc-arc

3. frameworks: pod库使用的系统库
```
spec.ios.framework = 'CFNetwork'
spec.frameworks = 'QuartzCore', 'CoreData'
```

4. weak_frameworks: 如果在高版本的 OS 中调用新增的功能，还要保持低版本 OS 能够运行，就要使用 weak_framwoks 如果引用的某些类或者接口在低版本中并不支持，可以再运行时判断。
```
spec.weak_framework = 'Twitter'
```

5. libraries 使用的静态库 比如 libz、sqlite3.0 等,多个用逗号分开
```
spec.ios.library = ‘xml2'
spec.libraries = 'xml2', 'z'
```

6. compiler_flags: 传递个编译器的标记列表
```
spec.compiler_flags = '-DOS_OBJECT_USE_OBJC = 0' ， '-Wno-format'
```

7. pod_target_xcconfig: flag 添加到私有库的 target Xcconfig 文件中，只会配置当前私有库
```
//user_target_xcconfig  会影响所有target的 Xcconfig，他会污染用户配置，所以不推荐使用，可能会导致冲突
spec.pod_target_xcconfig  =  {  'OTHER_LDFLAGS'  =>  '-lObjC'  }
spec.user_target_xcconfig  =  {  'MY_SUBSPEC'  =>  'YES'  }
```
注意：尽量使用 pod_target_xcconfig，只会影响你编译的 pod

8. prefix_header_file: 默认是 true 时 cocoapos 会生成默认前缀 .pch 文件
```
//自定义前缀文件
spec.prefix_header_file = false
spec.prefix_header_file = 'iphone/include/prefix.pch'
```

9. prefix_header_contents 向pod项目的前缀文件中添加内容
```
spec.prefix_header_contents  =  '#import <UIKit / UIKit.h>'
spec.prefix_header_contents  =  '#import <UIKit / UIKit.h>' , '#import <Foundation / Foundat            ion.h>'
```

10. module_name

11. header_dir

12. header_mappings_dir

<br>

## 四、文件操作：podspec 文件必须在根仓库文件中。文件路径也是相对于根仓库位置的

模式1：*（检测文件名）
* 匹配所有文件
c* 匹配以c开头的文件
*c 匹配以c结尾的文件
c 匹配以包括c的文件

模式2：**
递归匹配目录

模式3：？
匹配任何一个字符

模式4：[set]
匹配任何在set中的字符

模式5：{p, q}
匹配p，或q

模式6：遗弃下一个元字符
"JSONKit.?" #=> ["JSONKit.h", "JSONKit.m"]
".[a-z][a-z]" #=> ["CHANGELOG.md", "README.md"]
".[^m]" #=> ["JSONKit.h"]
".{h,m}" #=> ["JSONKit.h", "JSONKit.m"]
"*" #=> ["CHANGELOG.md", "JSONKit.h", "JSONKit.m", "README.md"]

1. source_files：pod文件路径
```
spec.souce_files = 'Classes/**/*.{h,m}’
spec.source_files = 'Classes/**/*.{h,m}', 'More_Classes/**/*.{h,m}'
```

2. public_header_files：公共头文件，这些头文件将暴露给用户的项目。如果不设置，所有source_files的头文件将被暴露
```
spec.public_header_files  =  'Headers/Public/*.h'
```
      
3. private_header_files.和public_header_files相反，指定不暴露的头文件

4. vendored_frameworks：使用的三方framework
```
spec.ios.vendored_frameworks = 'MyPod/Frameworks/MyFramework.framework’ //指定三方库的路径
spec.vendored_frameworks = 'MyFramework.framework', ‘TheirFramework.framework'
```
 
5. vendored libraries：三方静态库 指明具体路径
```
spec.ios.vendored_library = 'Libraries/libProj4.a’
spec.vendored_libraries = 'libProj4.a', 'libJavaScriptCore.a'
```

6. resource_bundles：资源文件
```
s.ios.resource_bundle = { 'MapBox' => 'MapView/Map/Resources/*.png' }
s.resource_bundles = {   'XBPodSDK' => ['XBPodSDK/Assets/**']}
```
 
7. exclude_files:被排除的文件
```
spec.ios.exclude_files = 'Classes/osx’
spec.exclude_files = 'Classes/**/unused.{h,m}'
```
 
 <br>
 
## 五、Subspecs 私有库模块

subspec：

```
//简单：
pec 'Twitter' do |sp|
    sp.source_files = 'Classes/Twitter' //指定子模块路径
end

subspec 'Pinboard' do |sp|
    sp.source_files = 'Classes/Pinboard'
end
    
//复杂：
Pod::Spec.new do |s|
    s.name = 'RestKit'

    s.subspec 'Core' do |cs|
        cs.dependency 'RestKit/ObjectMapping'
        cs.dependency 'RestKit/Network'
        cs.dependency 'RestKit/CoreData'
    end

    s.subspec 'ObjectMapping' do |os|
    end
    
end
```
