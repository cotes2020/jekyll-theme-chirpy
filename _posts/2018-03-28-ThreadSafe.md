---
title: 线程安全中的各种锁
date: 2018-03-28 17:50:49
categories: iOS
tags: 多线程
---

* [起底多线程同步锁(iOS)](https://blog.51cto.com/u_16124099/6326657)
* [不再安全的 OSSpinLock](https://blog.ibireme.com/2016/01/16/spinlock_is_unsafe_in_ios/)
* [探讨iOS开发中各种锁](https://blog.csdn.net/qq_30513483/article/details/53814482)
* [iOS 开发中的八种锁](https://www.jianshu.com/p/2d59ecd5e81d)

<br>

> 注意
{: .prompt-danger }
除非开发者能保证访问锁的线程全部都处于同一优先级，否则 iOS 系统中所有类型的自旋锁都别再用。

<br>

## iOS 加锁的几种方式
### 1、synchronized
``` objc
// 性能最差，敲以下代码时 Xcode 没有提示，是否可以理解为 Apple 不建议使用这种加锁方式 ？
@synchronized (对象，一般用self) {
    // ......
}
```

<br>

### 2、NSLock
``` objc
[self.lock lock];
    // ......
[self.lock unlock];
```

<br>

### 3、NSCondition
```  objc
// NSCondition 实现了`NSLocking`协议，同样具有锁的功能，与`NSLock`一样可以加锁、解锁

NS_CLASS_AVAILABLE(10_5, 2_0)
@interface NSCondition : NSObject <NSLocking> {

/*
阻塞线程，直到其他线程调用该对象的 signal 方法或 broadcast 方法来唤醒
唤醒后该线程从阻塞态变为就绪态，交由系统进行线程调度
调用 wait 方法时内部会自动执行 unlock 方法释放锁，并阻塞线程
*/
- (void)wait;

// 同上，只是该方法是在 limit 到达时唤醒线程
- (BOOL)waitUntilDate:(NSDate *)limit;

// 随机唤醒一个在当前 NSCondition 对象上阻塞的一个线程，使其从阻塞态进入就绪态
- (void)signal;

// 唤醒在当前 NSCondition 对象上阻塞的所有线程
- (void)broadcast;

// 设置名称
@property (nullable, copy) NSString *name;

@end




// ===============================使用事例==================================




// 取钱
- (void)draw:(id)money {
    // 设置消费者取钱20次
    NSUInteger count = 0;
    
    while (count < 20) {
        // 首先使用condition上锁，如果其他线程已经上锁则阻塞
        [self.condition lock];
        
        // 判断是否有钱
        if (self.haveMoney) {
            // 有钱则进行取钱的操作，并设置haveMoney为NO
            self.balance -= [money doubleValue];
            self.haveMoney = NO;
            count += 1;
            NSLog(@"%@ draw money %lf %lf", [[NSThread currentThread] name], [money doubleValue], self.balance);
            
            // 取钱操作完成后唤醒其他在此condition上等待的所有线程
            [self.condition broadcast];
        } else {
            // 如果没有钱则在此condition上等待，并阻塞
            [self.condition wait];
            // 如果阻塞的线程被唤醒后会继续执行代码
            NSLog(@"%@ wake up", [[NSThread currentThread] name]);
        }
        
        // 释放锁
        [self.condition unlock];
    }
}

// 存钱
- (void)deposite:(id)money {
    // 创建了三个取钱线程，每个取钱20次，则存钱60次
    NSUInteger count = 0;
    
    while (count < 60) {
    
        // 上锁，如果其他线程上锁了则阻塞
        [self.condition lock];
        
        // 判断如果没有钱则进行存钱操作
        if (!self.haveMoney) {
            // 进行存钱操作，并设置 haveMoney 为 YES
            self.balance += [money doubleValue];
            self.haveMoney = YES;
            count += 1;
            NSLog(@"Deposite money %lf %lf", [money doubleValue], self.balance);
            
            // 唤醒其他所有在condition上等待的线程
            [self.condition broadcast];
        } else {
            // 如果有钱则等待
            [self.condition wait];
            NSLog(@"Deposite Thread wake up");
        }
        
        // 释放锁
        [self.condition unlock];
    }
}

- (void)useNSCondition {
    
    Account *account = [[Account alloc] init];
    account.accountNumber = @"1603121434";
    account.balance = 0;
    
    // 消费者线程1，每次取1000元
    NSThread *thread1 = [[NSThread alloc] initWithTarget:account selector:@selector(draw:) object:@(1000)];
    [thread1 setName:@"consumer1"];
    
    // 消费者线程2，每次取1000元
    NSThread *thread2 = [[NSThread alloc] initWithTarget:account selector:@selector(draw:) object:@(1000)];
    [thread2 setName:@"consumer2"];
    
    // 生产者线程3，每次存1000元
    NSThread *thread3 = [[NSThread alloc] initWithTarget:account selector:@selector(deposite:) object:@(1000)];
    [thread3 setName:@"productor"];
    
    [thread1 start];
    [thread2 start];
    [thread3 start];
}
```

<br>

### 4、NSConditionLock
``` objc
// NSConditionLock 条件锁
// 其中的 condition 参数，可以理解为一个条件标识，只有传入的 condition 和锁的标识相同才会加锁成功，否则阻塞线程。

NSConditionLock *cLock = [[NSConditionLock alloc] initWithCondition:0];

// 任务1
dispatch_async(dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0), ^{
    if ([cLock tryLockWhenCondition:0]) {
        NSLog(@"加锁成功---任务1");
       [cLock unlockWithCondition:1];
    } else {
        NSLog(@"加锁失败---任务1");
    }
});

// 任务2
dispatch_async(dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0), ^{
    [cLock lockWhenCondition:3];
    NSLog(@"加锁成功---任务2");
    [cLock unlockWithCondition:2];
});

// 任务3
dispatch_async(dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0), ^{
    [cLock lockWhenCondition:1];
    NSLog(@"加锁成功---任务3");
    [cLock unlockWithCondition:3];
});

打印顺序: 任务1  任务3  任务2


我们在初始化 NSConditionLock 对象时，给了他的标示为 0
执行 tryLockWhenCondition: 时，我们传入的条件标示也是 0，所以 任务1 加锁成功
执行 unlockWithCondition: 时，这时候会把 condition 由 0 修改为 1
接着会走到 任务3，然后 任务3 又将 condition 修改为 3，最后走 任务2 的流程
从上面的结果我们可以发现，NSConditionLock 还可以实现任务之间的依赖。
```

<br>

### 5、NSRecursiveLock 递归锁
**递归锁可以被同一个线程多次获取而不会导致死锁。但是所有其他线程都无法访问由锁保护的代码。**

``` objc
// 此处如果用 NSLock 会死锁
// NSLock *rLock = [NSLock new];

NSRecursiveLock *rLock = [NSRecursiveLock new];
dispatch_async(dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0), ^{
    static void (^RecursiveBlock)(NSInteger);
    RecursiveBlock = ^(NSInteger value) {
        [rLock lock];
        if (value > 0) {
            NSLog(@"线程%d", value);
            RecursiveBlock(value - 1);
        }
        [rLock unlock];
    };
    RecursiveBlock(4);
});


死锁情况：
如果用 NSLock，在线程中 RecursiveMethod 是递归调用，每次进入 block 时，都会加一次锁，
而从第二次开始，由于锁已经被使用，所以它需要等待锁被解除，这就导致死锁，线程被阻塞。
需要将 NSLock 替换为 NSRecursiveLock。
```

<br>

### 6、pthread_mutex 互斥锁

``` objc
// ==================================================================

pthread_mutex_t _mutex;
pthread_cond_t _cond;

// 用法类似 NSCondition
pthread_cond_broadcast(&_cond);
pthread_cond_signal(&_cond);
pthread_cond_wait(&_cond, &_mutex);
    
    
// 普通锁
pthread_mutex_init(&_mutex, NULL));

// 递归锁
pthread_mutexattr_t attr;
pthread_mutexattr_init (&attr);
pthread_mutexattr_settype (&attr, PTHREAD_MUTEX_RECURSIVE);
pthread_mutex_init (&_mutex, &attr);
pthread_mutexattr_destroy (&attr);
   
   
   
#define PTHREAD_MUTEX_NORMAL		0
#define PTHREAD_MUTEX_ERRORCHECK	1
#define PTHREAD_MUTEX_RECURSIVE		2
#define PTHREAD_MUTEX_DEFAULT		PTHREAD_MUTEX_NORMAL
    
    
    
    
// 在 dealloc 方法中需要销毁
pthread_mutex_destroy(&_mutex)




// ==================================================================




static pthread_mutex_t pLock;
pthread_mutex_init(&pLock, NULL);

// 任务1
dispatch_async(dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0), ^{
    NSLog(@"任务1 准备上锁");
    pthread_mutex_lock(&pLock);
    sleep(3);
    NSLog(@"任务1");
    pthread_mutex_unlock(&pLock);
});

// 任务2
dispatch_async(dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0), ^{
    NSLog(@"任务2 准备上锁");
    pthread_mutex_lock(&pLock);
    NSLog(@"任务2");
    pthread_mutex_unlock(&pLock);
});


// pthread_mutex 中有个 pthread_mutex_trylock(&pLock) 和 OSSpinLockTry(&oslock) 的
// 区别在于，前者可以加锁时返回的是 0，否则返回一个错误提示码；后者返回的 YES 和 NO
```


<br>

### 7、信号量机制
``` objc
// 信号量机制
// 初始化信号量，值要 >= 0，如果 < 0，则返回 NULL
dispatch_semaphore_t semaphore = dispatch_semaphore_create(1);

// 可以是具体的等待时间，或者 DISPATCH_TIME_FOREVER
dispatch_time_t timeout = dispatch_time(DISPATCH_TIME_NOW, 3.0f * NSEC_PER_SEC);

// 线程1
dispatch_async(dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0), ^{
    NSLog(@"线程1 等待ing");
    
    // 信号等待: 如果 semaphore = 0，则阻塞当前线程；在 > 0 时 semaphore - 1 并返回
    dispatch_semaphore_wait(semaphore, timeout);
    
    NSLog(@"线程1 正在执行");
    
    // 信号通知: semaphore + 1 并返回
    dispatch_semaphore_signal(semaphore);
    NSLog(@"线程1 发送信号");
});

// 线程2
dispatch_async(dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0), ^{
    NSLog(@"线程2 等待ing");
    dispatch_semaphore_wait(semaphore, timeout);
    NSLog(@"线程2 正在执行 ");
    dispatch_semaphore_signal(semaphore);
    NSLog(@"线程2 发送信号");
});
```

<br>

### 8、pthread_rwlock_t 读写锁

``` objc
- (void)useRWLock
{
    // pthread_rwlock_t 读写锁
    // 当读写锁被一个线程以读模式占用时，写操作的其他线程会被阻塞，读操作的其他线程还可继续进行。
    // 当读写锁被一个线程以写模式占用时，写操作的其他线程会被阻塞，读操作的其他线程也被阻塞。
    
    // 初始化
    pthread_rwlock_t rwLock = PTHREAD_RWLOCK_INITIALIZER;
    // 读模式
    pthread_rwlock_wrlock(&rwLock);
    // 写模式
    pthread_rwlock_rdlock(&rwLock);
    // 读模式或者写模式的解锁
    pthread_rwlock_unlock(&rwLock);
    
    
    dispatch_async(dispatch_get_global_queue(0, 0), ^{
        [self readBookWithTag:1];
    });
    
    dispatch_async(dispatch_get_global_queue(0, 0), ^{
        [self readBookWithTag:2];
    });
    
    dispatch_async(dispatch_get_global_queue(0, 0), ^{
        [self writeBook:3];
    });
    
    dispatch_async(dispatch_get_global_queue(0, 0), ^{
        [self writeBook:4];
    });
    
    dispatch_async(dispatch_get_global_queue(0, 0), ^{
        [self readBookWithTag:5];
    });
}

- (void)readBookWithTag:(NSInteger )tag {
    pthread_rwlock_rdlock(&rwLock);
    NSLog(@"start read ---- %ld",tag);
    
    // 读...
    
    NSLog(@"end   read ---- %ld",tag);
    pthread_rwlock_unlock(&rwLock);
}

- (void)writeBook:(NSInteger)tag {
    pthread_rwlock_wrlock(&rwLock);
    NSLog(@"start wirte ---- %ld",tag);
    
    // 写...
    
    NSLog(@"end   wirte ---- %ld",tag);
    pthread_rwlock_unlock(&rwLock);
}
```


<br>

### 8、os_unfair_lock 苹果官方建议的用来替代 OSSpinLock 的锁

``` objc
// 创建锁，并初始化
os_unfair_lock_t unfairLock;
unfairLock = &(OS_UNFAIR_LOCK_INIT);

// 加锁
os_unfair_lock_lock(unfairLock);

// 解锁
os_unfair_lock_unlock(unfairLock);
```

<br>

### 9、OSSpinLock 自旋锁（性能最好，但某些场景会有问题，已经被苹果废弃）
* **自旋锁的特点是当有其他线程加锁后，当前线程会循环等待，并不会进入睡眠，所以性能最好。**
* **若几个线程的优先级不同，则可能出现[优先级翻转](https://baike.baidu.com/item/%E4%BC%98%E5%85%88%E7%BA%A7%E7%BF%BB%E8%BD%AC/4945202?fr=aladdin)的问题。所以 OSSpinLock 已经被苹果废弃。**
* **除非开发者能保证访问锁的线程优先级都相同，否则 iOS 系统中所有类型的自旋锁都别再用。**

``` objc
- (void)testOSSpinLock
{
    // 主线程中
    __block OSSpinLock spinlock = OS_SPINLOCK_INIT;
    
    // 任务1
    dispatch_async(dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0), ^{
        OSSpinLockLock(&spinlock);
        [self threadMethod1];
        sleep(3);
        OSSpinLockUnlock(&spinlock);
    });
    
    for (NSUInteger i = 0; i < 10; i++) {
        // 任务2
        dispatch_async(dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0), ^{
            // 睡一秒，保证 任务1 先加锁
            sleep(1);
            
            // 如果 任务1 加锁成功，则 任务2 的线程会循环等待，并不会进入睡眠，这是自旋锁的特点
            OSSpinLockLock(&spinlock);
            [self threadMethod2];
            OSSpinLockUnlock(&spinlock);
        });
    }
}
```
