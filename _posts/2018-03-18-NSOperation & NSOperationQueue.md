---
title: 多线程：NSOperation、NSOperationQueue
date: 2018-03-18 16:47:26
categories: iOS
tags: 多线程
---

## NSOperation、NSOperationQueue 是一套多线程解决方案，通常是配合使用

<br>

## 特点
1. 底层是基于 GCD 的封装，OC 语言实现，面向对象，现在 Swift 也支持
2. 可获取和设置操作的各种状态，其内部是通过 KVO 来实现
3. 可设置队列的最大并发数
4. 各操作间可设置依赖关系

<br>

## 用 NSOperation、NSOperationQueue 实现多线程的使用步骤
1. 创建操作：先将需要执行的操作封装到一个 NSOperation 对象中。
2. 创建队列：创建 NSOperationQueue 对象。
3. 将操作加入到队列中：将 NSOperation 对象添加到 NSOperationQueue 对象中。

这样 NSOperationQueue 中的 NSOperation 就会自动在多个线程执行，不用调用 start 方法。

<br>

## 基本使用
NSOperation 是个抽象类，开发中只使用它的子类来封装操作。

1. 使用子类 NSInvocationOperation
2. 使用子类 NSBlockOperation
3. 自定义继承自 NSOperation 的子类

<br>

## 使用子类 NSInvocationOperation

``` objc
- (void)useInvocationOperation {

    // 1.创建 NSInvocationOperation 对象
    NSInvocationOperation *op = [[NSInvocationOperation alloc] initWithTarget:self selector:@selector(task1) object:nil];

    // 2.需要手动调用 start 方法开始执行操作
    [op start];
}

/**
 * 任务1，在主线程执行，并没有新建线程
 */
- (void)task1 {
    for (int i = 0; i < 2; i++) {
        [NSThread sleepForTimeInterval:2]; // 模拟耗时操作
        NSLog(@"1---%@", [NSThread currentThread]); // 打印当前线程
    }
}
```

<br>

## 使用子类 NSBlockOperation
``` objc
- (void)useBlockOperation {

    // 1.创建 NSBlockOperation 对象
    NSBlockOperation *op = [NSBlockOperation blockOperationWithBlock:^{
        for (int i = 0; i < 2; i++) {
            // 模拟耗时操作
            [NSThread sleepForTimeInterval:2]; 
            // 只封装一个block时，在主线程中执行；有调用 addExecutionBlock: 不一定在主线程执行
            NSLog(@"1---%@", [NSThread currentThread]); 
        }
    }];

    // 2.添加额外操作，如果有调用 addExecutionBlock: 添加多个操作，则会异步并发执行所有操作
    // 新建线程的数量由系统决定
    [op addExecutionBlock:^{
        for (int i = 0; i < 2; i++) {
            [NSThread sleepForTimeInterval:2]; // 模拟耗时操作
            NSLog(@"2---%@", [NSThread currentThread]); // 打印当前线程
        }
    }];
    
    [op addExecutionBlock:^{
        for (int i = 0; i < 2; i++) {
            [NSThread sleepForTimeInterval:2]; // 模拟耗时操作
            NSLog(@"3---%@", [NSThread currentThread]); // 打印当前线程
        }
    }];

    // 3.需要手动调用 start 方法开始执行操作
    [op start];
}
```


<br>

## 使用自定义继承自 NSOperation 的子类
自定义的子类需要重写 main 方法。我们不需要管理操作的状态属性 isExecuting 和 isFinished。当 main 方法执行完返回，这个操作就结束了。


<br>
<br>

## 队列的使用

``` objc
// 主队列- 添加到主队列中的操作，都会放到主线程中执行。
NSOperationQueue *queue = [NSOperationQueue mainQueue];

// 自定义队列- 添加到这种队列中的操作，就会自动放到子线程中执行。
NSOperationQueue *queue = [[NSOperationQueue alloc] init];
```


<br>

``` objc
/**
 * 使用 addOperation: 将操作加入到操作队列中
 */
- (void)addOperationToQueue {

    // 1.创建队列
    NSOperationQueue *queue = [[NSOperationQueue alloc] init];

    // 设置最大并发操作数，注意：不是最大线程数
    // queue.maxConcurrentOperationCount = 1;

    // 2.创建操作
    // 使用 NSInvocationOperation 创建操作1
    NSInvocationOperation *op1 = [[NSInvocationOperation alloc] initWithTarget:self selector:@selector(task1) object:nil];

    // 使用 NSInvocationOperation 创建操作2
    NSInvocationOperation *op2 = [[NSInvocationOperation alloc] initWithTarget:self selector:@selector(task2) object:nil];

    // 使用 NSBlockOperation 创建操作3
    NSBlockOperation *op3 = [NSBlockOperation blockOperationWithBlock:^{
        for (int i = 0; i < 2; i++) {
            [NSThread sleepForTimeInterval:2]; // 模拟耗时操作
            NSLog(@"3---%@", [NSThread currentThread]); // 打印当前线程
        }
    }];
    
    [op3 addExecutionBlock:^{
        for (int i = 0; i < 2; i++) {
            [NSThread sleepForTimeInterval:2]; // 模拟耗时操作
            NSLog(@"4---%@", [NSThread currentThread]); // 打印当前线程
        }
    }];

    // 执行顺序的依据：先依赖关系，再优先级。如果想让 op2 执行完后再执行 op1，要设置依赖关系，不能设置优先级
    // 3.设置优先级，op1 会比 op2 先开始执行，但并不是 op1 执行完后才开始执行 op2
    op2.queuePriority = NSOperationQueuePriorityVeryLow;
    op1.queuePriority = NSOperationQueuePriorityVeryHigh;

    // 4.添加依赖，就算 op1 的优先级比 op2 高，op1 还是会在 op2 执行完后才会被执行
    [op1 addDependency:op2]; 
    
    // 5.使用 addOperation: 添加所有操作到队列中
    [queue addOperation:op1];
    [queue addOperation:op2];
    [queue addOperation:op3];
    
    
    // 如果 queue.maxConcurrentOperationCount = 1，就是异步串行，
    // 只会另开一条子线程，但是为什么在 iOS 11 模拟器上测试会开 2 条子线程？？？
}
```

<br>

> 注意
{: .prompt-danger }

* **NSOperationQueue 中的操作是异步执行，当 maxConcurrentOperationCount 等于 1 时是异步串行；大于 1 时是异步并行。**
* **这里 maxConcurrentOperationCount 控制的不是并发线程的数量，而是一个队列中同时能并发执行的最大操作数。**


<br>
<br>
#### NSOperation、NSOperationQueue 线程间的通信
在 iOS 开发过程中，需要在主线程刷新 UI，把一些耗时的操作放在其他线程，当完成耗时操作时，要回到主线程，那么就用到了线程之间的通讯。

``` objc
/**
 * 线程间通信
 */
- (void)communication {

    // 1.创建队列
    NSOperationQueue *queue = [[NSOperationQueue alloc]init];

    // 2.添加操作
    [queue addOperationWithBlock:^{
        // 异步进行耗时操作
        。。。
        
        // 3.回到主线程更新 UI
        [[NSOperationQueue mainQueue] addOperationWithBlock:^{
            。。。
        }];
    }];
}
```


<br>

## NSOperation、NSOperationQueue 线程同步和线程安全
线程同步：可理解为线程 A 执行到某个地方时要依靠线程 B 的某个结果，于是停下来等 B 执行完，再基于 B 的执行结果继续操作。

线程安全：若有多个线程同时执行某段代码（更改变量），一般都需要考虑线程同步，否则可能影响线程安全。


<br>

## NSOperation 常用属性和方法

``` objc
// 取消操作，实质是标记 isCancelled 状态。
- (void)cancel; 

// 判断操作是否已经结束。
- (BOOL)isFinished; 

// 判断操作是否已经标记为取消。
- (BOOL)isCancelled; 

// 判断操作是否正在在运行。
- (BOOL)isExecuting; 

// 判断操作是否处于准备就绪状态，这个值和操作的依赖关系相关。
- (BOOL)isReady;

// 阻塞当前线程，直到该操作结束。可用于线程执行顺序的同步。
- (void)waitUntilFinished;

// 会在当前操作执行完毕时执行 completionBlock。
- (void)setCompletionBlock:(void (^)(void))block;

// 添加依赖，使当前操作依赖于操作 op 的完成。
- (void)addDependency:(NSOperation *)op;

// 移除依赖，取消当前操作对操作 op 的依赖。
- (void)removeDependency:(NSOperation *)op;

// 在当前操作开始执行之前完成执行的所有操作对象数组。
@property (readonly, copy) NSArray<NSOperation *> *dependencies; 
```


<br>

## NSOperationQueue 常用属性和方法

``` objc
// 可以取消队列的所有未执行操作，正在执行的操作不会被取消
- (void)cancelAllOperations; 

// 判断队列是否处于暂停状态。 YES 为暂停状态，NO 为恢复状态。
- (BOOL)isSuspended; 

// 可设置操作的暂停和恢复，YES 代表暂停队列，NO 代表恢复队列。
- (void)setSuspended:(BOOL)b;

// 阻塞当前线程，直到队列中的操作全部执行完毕。
- (void)waitUntilAllOperationsAreFinished; 

// 向队列中添加一个 NSBlockOperation 类型操作对象。
- (void)addOperationWithBlock:(void (^)(void))block; 

// 向队列中添加操作数组，wait 标志是否阻塞当前线程直到所有操作结束
- (void)addOperations:(NSArray *)ops waitUntilFinished:(BOOL)wait; 

// 当前在队列中的操作数组（某个操作执行结束后会自动从这个数组清除）。
- (NSArray *)operations;

// 当前队列中的操作数。
- (NSUInteger)operationCount;

// 获取当前队列，如果当前线程不是在 NSOperationQueue 上运行则返回 nil。
+ (id)currentQueue;

// 获取主队列。
+ (id)mainQueue;
```

<br>

> 注意
{: .prompt-danger }

* **这里的暂停和取消（包括操作的取消和队列的取消）并不代表可以将当前的操作立即取消，而是当当前的操作执行完毕之后不再执行新的操作。**
* **暂停和取消的区别就在于：暂停之后可以恢复继续执行；而取消操作之后不行。**
