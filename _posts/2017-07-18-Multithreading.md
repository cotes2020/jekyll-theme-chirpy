---
title: 多线程概述
date: 2017-07-18 16:03:25
categories: iOS
tags: 多线程
---

## 一堆博客
* [起底多线程同步锁(iOS)](https://blog.51cto.com/u_16124099/6326657)
* [看一眼，你就会用GCD](https://www.jianshu.com/p/a28c5bbd5b4a?nomobile=yes)
* [iOS多线程：『GCD』详尽总结](https://www.jianshu.com/p/2d57c72016c6)
* [GCD高级用法](https://www.jianshu.com/p/805dd3c16869)

<br>

## 进程和线程
* 进程：在早期面向进程设计的计算机结构中，进程是程序的基本执行实体；在当代面向线程设计的计算机结构中，进程是线程的容器。
* 线程：是程序执行的最小单元，在多线程系统中，通常是在一个进程中包括多个线程。

<br>

##  四种多线程技术:
1. pthread ：是一套 C 语言的 API，跨平台（Unix / Linux / Windows），很少用
2. NSThread ：OC 语言实现的 API，需要开发者手动管理线程生命周期
3. GCD（Grand Central Dispatch）：是一套 C 语言的 API，由系统管理线程生命周期
4. NSOperation/NSOperationQueue ：底层基于 GCD， OC 语言实现的 API，由系统管理线程生命周期

操作队列(NSOperation/NSOperationQueue)是并发编程的首选工具
在iOS开发中，所有UI的更新工作，都必须在主线程执行！

<br>

## 队列：
* 串行和并行针对的是队列；同步和异步针对的是线程。
* 主队列：是一个串行队列
* 全局队列：是一个并行队列
* 同步串行：`不开`新线程，顺序执行（死锁问题看后面）
* 同步并行：`不开`新线程，顺序执行
* 异步串行：如果串行队列是主队列，则`不开`新线程，在主线程执行；否则会新建`一个`子线程顺序执行
* 异步并行：会新建`多个`子线程，无序执行


<br>


``` objc
dispatch_queue_t q = dispatch_queue_create("gcddemo", DISPATCH_QUEUE_SERIAL);
    
// 非ARC开发时，别忘记release
dispatch_release(q);
```

<br>

``` objc
[self.myQueue addOperation:op];
[op4 addDependency:op3]; // 添加依赖
```


<br>

``` objc
// 在 iOS 中，GCD 和 NSOperation 新建线程时，会自动在新线程中创建自动释放池
// 但是使用 NSThread 的时候，系统不会自动创建自动释放池
// 用 NSThread 实现多线程时，若涉及对象分配，需手动添加 autoreleasepool ，否则会内存泄露
@autoreleasepool {
    NSLog(@"%@", [NSThread currentThread]);
    // [NSThread sleepForTimeInterval:2.0f];    休眠2秒
        
    // wait = YES 表示主线程会等待 selector 方法执行完毕再继续往下执行
    // wait = NO 表示 selector 方法会追加到主队列的队尾，最后执行
    [self performSelectorOnMainThread:@selector(setImage:) withObject:[UIImage imageNamed:imageName] waitUntilDone:NO];
}
```

<br>



## GCD 公开的有5个不同的队列：
一个运行在主线程中的主队列，3 个不同优先级的后台队列，以及一个优先级更低的后台队列（用于 I/O）


<br>

实际运用中，一般可以这样来写，常见的网络请求数据多线程执行模型：

``` objc
dispatch_async(dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0), ^{
　　// 子线程中开始网络请求数据
　　// 更新数据模型
　　dispatch_async(dispatch_get_main_queue(), ^{
　　　　// 在主线程中更新UI代码
　　});
});
```
程序的后台运行和UI更新代码紧凑，代码逻辑一目了然。



<br>

## 单例的实现步骤:
* 重写`allocWithZone`方法，`allocWithZone`方法是对象分配内存空间时，最终会调用的方法，重写该方法，保证只会分配一个内存空间
* 提供`sharedXXX`类方法，便于其他类访问


``` objc
+ (id)allocWithZone:(struct _NSZone *)zone {
    static Ticket *instance;
    static dispatch_once_t onceToken;
    //dispatch_once是线程安全的,能够做到在多线程的环境下Block中的代码只会被执行一次
    dispatch_once(&onceToken, ^{
        instance = [super allocWithZone:zone];
    });  
    return instance;
}
```


<br>


## 关于多线程死锁


``` objc
// 案例与分析

    
// 死锁一： 主线程中 同步+主队列
// 同步操作会阻塞当前线程，而串行队列，有任务来，会将任务加到队尾，并遵循 FIFO 原则。
// 现在 ”任务1” 就会被加到主队列最后，也就是在主队列 “任务2” 之后，也就出现互相等待
// 的局面，造成死锁。
dispatch_sync(dispatch_get_main_queue(), ^{
    NSLog(@"任务1");
});
NSLog(@"任务2");
    
    
// 死锁二： (异步串行)串行队列中 嵌套 同步+串行（同一个串行队列）
dispatch_queue_t queue = dispatch_queue_create("", DISPATCH_QUEUE_SERIAL);
dispatch_async(queue, ^{
    dispatch_sync(queue, ^{
        NSLog(@"任务");
    });
});
    
    
// 死锁三： (同步串行)串行队列中 嵌套 同步+串行（同一个串行队列）
dispatch_queue_t queue = dispatch_queue_create("", DISPATCH_QUEUE_SERIAL);
dispatch_sync(queue, ^{
    dispatch_sync(queue, ^{
        NSLog(@"任务");
    });
});
    
    
// 不死锁一： 子线程中 同步+主队列
dispatch_async(dispatch_get_global_queue(0, 0), ^{
    dispatch_sync(dispatch_get_main_queue(), ^{
        NSLog(@"任务");
    });
});
    
    
// 不死锁二： (异步串行)串行队列中 嵌套 同步+串行（不同的串行队列）
dispatch_queue_t queue = dispatch_queue_create(nil, DISPATCH_QUEUE_SERIAL);
dispatch_queue_t queue2 = dispatch_queue_create(nil, DISPATCH_QUEUE_SERIAL);
dispatch_async(queue, ^{
    NSLog(@"queue===%@", [NSThread currentThread]);
    dispatch_sync(queue2, ^{
        NSLog(@"queue2===%@", [NSThread currentThread]);
        NSLog(@"任务");
    });
});
    
    
// 不死锁三： (同步串行)串行队列中 嵌套 同步+串行（不同的串行队列）
dispatch_queue_t queue = dispatch_queue_create(nil, DISPATCH_QUEUE_SERIAL);
dispatch_queue_t queue2 = dispatch_queue_create(nil, DISPATCH_QUEUE_SERIAL);
dispatch_sync(queue, ^{
    NSLog(@"queue===%@", [NSThread currentThread]);
    dispatch_sync(queue2, ^{
        NSLog(@"queue2===%@", [NSThread currentThread]);
        NSLog(@"任务");
    });
});

```

## 死锁小结
* 主队列也是一个串行队列，主队列中的任务放在主线程中执行
* 在 串行队列(包括同步、异步) 中嵌套执行 同步串行(同一个串行队列) 任务 => 死锁
* 避免死锁方法：异步并行 嵌套 异步串行 => 实现多线程并发执行，然后从 子线程 回到 主线程


<br>

## 自动释放池的工作原理
1. 标记为`autorelease`的对象在出了作用域范围后，会被添加到最近一次创建的自动释放池中
2. 当自动释放池被销毁或耗尽时，会向自动释放池中的所有对象发送`release`消息
3. 每个线程都需要有`@autoreleasepool`，否则可能会出现内存泄漏，使用`NSThread`多线程技术，不会为后台线程创建自动释放池
