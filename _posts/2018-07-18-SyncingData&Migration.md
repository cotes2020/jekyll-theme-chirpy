---
title: 数据同步、迁移
date: 2018-07-18 11:32:59
categories: [iOS]
tags: [数据存储]
---

## 要求
* 无论有没有网络连接，每一台设备都能够访问完整的数据集。
* 网络可能连接不稳定，数据同步时发起的请求数量要尽可能少。
* 数据更改必须基于最新的数据，任何人都不应该在不知晓其他人修改的情况下覆盖那些改动。

<br>

# iCloud 和 Core Data


<br>
<br>
<br>
<br>
<br>
<br>

## 数据迁移

<br>

## 添加数据前尽量考虑完全
在处理任何数据持久性问题时最重要的事情之一就是先仔细思考你的模型，在最开始创建模型的时候尽量考虑完全。添加空属性或者空实体也比以后进行迁移时候创建好的多，因为迁移很容易出现错误，而未使用的数据就不会了。


<br>

## 轻量迁移相对于自定义迁移来说非常快速。能够处理简单的，系统能推断出来的迁移方式，比如：

* 删除实体、属性 或者 关系。
* 使用 renamingIdentifier 重新命名实体、属性 或者关系。
* 新添加一个 Optional 的属性。
* 新添加一个 Required 属性，但是必须有默认值。
* 把一个 Optional 属性改成带有默认值的 Required 属性。
* 把一个 非Option 的属性改成 Optional属性。
* 改变实体结构。
* 新添加父实体，把属性向父类移动或者将父类属性往子类中移。
* 把 对一 关系改成 对多 关系。
* 改变关系，从 non-ordered to-many 到 ordered to-many。


``` objc
// 轻量级迁移，模型映射交给 Core Data 去自动推断识别
NSDictionary *options = @{
                          NSSQLitePragmasOption: @{@"journal_mode": @"DELETE"}, // 关闭数据库日志记录模式
                          NSMigratePersistentStoresAutomaticallyOption :@YES,
                          NSInferMappingModelAutomaticallyOption:@YES };
    
NSError *error = nil;
[self.coordinator addPersistentStoreWithType:NSSQLiteStoreType
                               configuration:nil
                                         URL:[self storeURL]
                                     options:options
                                       error:&error];
```
<br>

## 复杂的迁移
* 手动创建 Mapping Model 适用于更加复杂的数据迁移

<br>

# 引用参考
* [同步案例学习](https://objccn.io/issue-10-4/)
* [自定义 Core Data 迁移](https://objccn.io/issue-4-7/)
* [Core Data 数据迁移指南](https://www.jianshu.com/p/b3b764fc5191)
* [一个完整的 Core Data 应用](https://objccn.io/issue-4-2/)
* [iCloud 和 Core Data](https://objccn.io/issue-10-2/)
* [Design for Core Data in iCloud](https://developer.apple.com/library/archive/documentation/General/Conceptual/iCloudDesignGuide/Chapters/DesignForCoreDataIniCloud.html)
* [Core Data Model Versioning and Data Migration Programming Guide](https://developer.apple.com/library/archive/documentation/Cocoa/Conceptual/CoreDataVersioning/Articles/vmLightweightMigration.html)
