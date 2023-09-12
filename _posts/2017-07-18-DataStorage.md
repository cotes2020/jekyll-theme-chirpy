---
title: 数据存储
date: 2017-07-18 16:04:52
categories: [iOS]
tags: [数据存储]
---

## 数据存储方式
* plist 属性列表
* NSKeyedArchiver
* Keychain（保存帐号、凭证等少量数据）
* Core Data
* SQLite 3

<br>

### plist 属性列表

``` objc
// 把 key 和 value 存入 /Library/Preferences/xxxxx.plist 属性列表文件中，xxxxx 是 bundle id 
[[NSUserDefaults standardUserDefaults] setInteger:5 forKey:@"age"];
NSInteger age = [[NSUserDefaults standardUserDefaults] integerForKey:@"age"];


// ======================================================================================


#pragma mark 存储在自定义的 plist 文件中，可以存储 字典 和 数组
- (void)writeDict {
    NSMutableDictionary *dict = [NSMutableDictionary dictionary];
    [dict setObject:@"张三" forKey:@"name"];
    [dict setObject:@(18) forKey:@"age"];
    
    // 获取应用沙盒的根路径
    NSString *home = NSHomeDirectory();
    NSString *documents = [home stringByAppendingPathComponent:@"Documents"];
    // 属性列表的默认拓展名是 plist
    NSString *path = [documents stringByAppendingPathComponent:@"dict.plist"];
    
    [dict writeToFile:path atomically:YES];
}

- (void)readDict {
    // 获取应用沙盒的根路径
    NSString *home = NSHomeDirectory();
    NSString *documents = [home stringByAppendingPathComponent:@"Documents"];
    // 属性列表的默认拓展名是plist
    NSString *path = [documents stringByAppendingPathComponent:@"dict.plist"];
    
    NSDictionary *dict = [NSDictionary dictionaryWithContentsOfFile:path];
    
    NSLog(@"%@", dict);
}
```

<br>

### NSKeyedArchiver

``` objc
// 归档（待归档的类需要实现 NSCoding 协议的两个方法，否则运行会报错）
NSData *data = [NSKeyedArchiver archivedDataWithRootObject:self.person];
[data writeToFile:path atomically:YES];
    
// 解档
Person *p = [NSKeyedUnarchiver unarchiveObjectWithData:data];
```

<br>

### Keychain
* 相当于是一个系统自带的数据库，可存储帐号密码等小量的数据（增、删、改、查）
* 存储在 Keychain 中的数据，在删除 APP 后数据还存在设备中，下次安装 APP 还能获取之前存的数据
* 可以将数据归档为 NSData 后存入 Keychain ；取出 NSData 后解档
* 可以使用第三方库 [SAMKeychain](https://github.com/soffes/SAMKeychain)

<br>

### Core Data
* 可以使用第三方库 [MagicalRecord](https://github.com/magicalpanda/MagicalRecord)
* [Core Data入门](https://www.cnblogs.com/mjios/archive/2013/02/26/2932999.html)


``` objc
// OC 语言，底层是基于 SQLite 3

- (void)useCoreData {
    
    // ================================ 创建上下文 ========================================================
    
    // 多个 Data Model 文件时用以下方法获取，注意 Model.xcdatamodel 文件的扩展名要写 "momd" 
    // NSURL *companyURL = [[NSBundle mainBundle] URLForResource:"Model" withExtension:@"momd"]; 
    // NSManagedObjectModel *model = [[NSManagedObjectModel alloc] initWithContentsOfURL:companyURL]; 
    
    // 单个文件时加载模型文件
    NSManagedObjectModel *model = [NSManagedObjectModel mergedModelFromBundles:nil];
    
    // 传入模型对象，初始化 NSPersistentStoreCoordinator
    NSPersistentStoreCoordinator *psc = [[NSPersistentStoreCoordinator alloc] initWithManagedObjectModel: model];
    
    // 构建 SQLite 数据库文件的路径
    NSString *docs = [NSSearchPathForDirectoriesInDomains(NSDocumentDirectory, NSUserDomainMask, YES) lastObject];
    NSURL *url = [NSURL fileURLWithPath:[docs stringByAppendingPathComponent:@"person.data"]];
    
    // 添加持久化存储库，这里使用 SQLite 作为存储库
    NSError *error = nil;
    NSPersistentStore *store = [psc addPersistentStoreWithType:NSSQLiteStoreType configuration:nil URL:url options:nil error:&error];
    if (store == nil) { // 直接抛异常
        [NSException raise:@"添加数据库错误" format:@"%@", [error localizedDescription]];
    }
    
    // 初始化上下文，设置 persistentStoreCoordinator 属性
    NSManagedObjectContext *context = [[NSManagedObjectContext alloc] initWithConcurrencyType:NSMainQueueConcurrencyType];
    
    context.persistentStoreCoordinator = psc;
    
    // 用完之后，记得要 release
    // [context release];
    
    
    
    
    
    
    // ================================ 新增 ========================================================
    
    
    // 传入上下文，创建一个 Person 实体对象
    NSManagedObject *person = [NSEntityDescription insertNewObjectForEntityForName:@"Person" inManagedObjectContext:context];
    // 设置Person的简单属性
    [person setValue:@"Jack" forKey:@"name"];
    [person setValue:[NSNumber numberWithInt:27] forKey:@"age"];
    
    // 传入上下文，创建一个 Card 实体对象
    NSManagedObject *book = [NSEntityDescription insertNewObjectForEntityForName:@"Book" inManagedObjectContext:context];
    [book setValue:@20.9 forKey:@"price"];
    
    // 设置 Person 和 Card 之间的关联关系
    [person setValue:book forKey:@"book"];
    
    // 利用上下文对象，将数据同步到持久化存储库
    error = nil;
    BOOL success = [context save:&error];
    if (!success) {
        [NSException raise:@"访问数据库错误" format:@"%@", [error localizedDescription]];
    }
    // 如果是想做更新操作：只要在更改了实体对象的属性后调用 [context save:&error]，就能将更改的数据同步到数据库
    
    
    
    
    
    // ================================ 查询 ========================================================
    
    
    // 初始化一个查询请求
    NSFetchRequest *request = [[NSFetchRequest alloc] init];
    
    // 设置要查询的实体
    request.entity = [NSEntityDescription entityForName:@"Person" inManagedObjectContext:context];
    
    // 设置排序（按照age降序）
    NSSortDescriptor *sort = [NSSortDescriptor sortDescriptorWithKey:@"age" ascending:NO];
    request.sortDescriptors = [NSArray arrayWithObject:sort];
    
    // 设置条件过滤(搜索name中包含字符串"Jack"的记录，注意：设置条件过滤时，数据库SQL语句中的%要用*来代替，所以%Jack%应该写成*Jack*)
    NSPredicate *predicate = [NSPredicate predicateWithFormat:@"name like %@", @"*Jack*"];
    request.predicate = predicate;
    
    // 执行请求
    error = nil;
    NSArray *objs = [context executeFetchRequest:request error:&error];
    if (error) {
        [NSException raise:@"查询错误" format:@"%@", [error localizedDescription]];
    }
    
    // 遍历数据
    for (NSManagedObject *obj in objs) {
        NSLog(@"name=%@", [obj valueForKey:@"name"]);
    }
    
    
    
    // ================================ 删除 ========================================================
    
    
    
    // 传入需要删除的实体对象
    [context deleteObject:objs.firstObject];
    
    // 将结果同步到数据库
    error = nil;
    [context save:&error];
    if (error) {
        [NSException raise:@"删除错误" format:@"%@", [error localizedDescription]];
    } else {
        // 遍历数据
        for (NSManagedObject *obj in objs) {
            NSLog(@"name=%@", [obj valueForKey:@"name"]);
        }
    }
}



// 创建数据库中的表对应的类：
// 选中模型文件 -> Editor -> Create NSManagedObject Subclass... -> 选择需要创建的类。。。


// 如果创建了对应的类，那么往数据库中添加数据的写法如下：
Person *person = [NSEntityDescription insertNewObjectForEntityForName:@"Person" inManagedObjectContext:context];
person.name = @"张三";
person.age = @(18);
    
Book *book = [NSEntityDescription insertNewObjectForEntityForName:@"Book" inManagedObjectContext:context];
book.price = @(33.0);
    
person.book = book;
    
// 最后调用保存数据
[context save&error];

```

![coredata](/assets/img/coredata.png)


<br>

### SQLite 3
* [SQLite 教程](https://www.runoob.com/sqlite/sqlite-tutorial.html)
* 可以使用第三方库 [fmdb](https://github.com/ccgus/fmdb)