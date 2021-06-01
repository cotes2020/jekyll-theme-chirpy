---
layout: post
title: "Swift defer 的正确使用"
date: 2019-08-11 22:45:00.000000000 +09:00
categories: [Swift]
tags: [Swift, defer]
---

Swift 里的 `defer` 大家应该都很熟悉了，`defer` 所声明的 block 会在当前代码执行退出后被调用。正因为它提供了一种延时调用的方式，所以一般会被用来做资源释放或者销毁，这在某个函数有多个返回出口的时候特别有用。比如下面的通过 `FileHandle` 打开文件进行操作的方法：

```swift
func operateOnFile(descriptor: Int32) {
    let fileHandle = FileHandle(fileDescriptor: descriptor)
    
    let data = fileHandle.readDataToEndOfFile()

    if /* onlyRead */ {
        fileHandle.closeFile()
        return
    }
    
    let shouldWrite = /* 是否需要写文件 */
    guard shouldWrite else {
        fileHandle.closeFile()
        return
    }
    
    fileHandle.seekToEndOfFile()
    fileHandle.write(someData)
    fileHandle.closeFile()
}
```

在不同的地方都需要调用 `fileHandle.closeFile()` 来关闭文件，这里更好的做法是用 `defer` 来统一处理。这不仅可以让我们就近在资源申请的地方就声明释放，也减少了未来添加代码时忘记释放资源的可能性：

```swift
func operateOnFile(descriptor: Int32) {
    let fileHandle = FileHandle(fileDescriptor: descriptor)
    defer { fileHandle.closeFile() }
    let data = fileHandle.readDataToEndOfFile()

    if /* onlyRead */ { return }
    
    let shouldWrite = /* 是否需要写文件 */
    guard shouldWrite else { return }
    
    fileHandle.seekToEndOfFile()
    fileHandle.write(someData)
}
```

## defer 的作用域

对于线程安全的保证，我选择使用了 `NSLock` 来完成。简单说，会有一些类似这样的方法：

```swift
let lock = NSLock()
let tasks: [ID: Task] = [:]

func remove(_ id: ID) {
    lock.lock()
    defer { lock.unlock() }
    tasks[id] = nil
}
```

对于 `tasks` 的操作可能发生在不同线程中，用 `lock()` 来获取锁，并保证当前线程独占，然后在操作完成后使用 `unlock()` 释放资源。这是很典型的 `defer` 的使用方式。

但是后来出现了一种情况，即调用 `remove` 方法之前，我们在同一线程的 caller 中获取过这个锁了，比如：

```swift
func doSomethingThenRemove() {
    lock.lock()
    defer { lock.unlock() }
    
    // 操作 `tasks`
    // ...
    
    // 最后，移除 `task`
    remove(123)
}
```

这样做显然在 `remove` 中造成了死锁 (deadlock)：`remove` 里的 `lock()` 在等待 `doSomethingThenRemove`中做 `unlock()` 操作，而这个 `unlock` 被 `remove` 阻塞了，永远不可能达到。

解决的方法大概有三种：

1. 换用 `NSRecursiveLock`：[`NSRecursiveLock`](https://developer.apple.com/documentation/foundation/nsrecursivelock) 可以在同一个线程获取多次，而不造成死锁的问题。
2. 在调用 `remove` 之前先 `unlock`。
3. 为 `remove` 传入按照条件，避免在其中加锁。

1 和 2 都会造成额外的性能损失，虽然在一般情况下这样的加锁性能微乎其微，但是使用方案 3 似乎也并不很麻烦。于是我很开心地把 `remove` 改成了这样：

```swift
func remove(_ id: ID, acquireLock: Bool) {
    if acquireLock {
        lock.lock()
        defer { lock.unlock() }
    }
    tasks[id] = nil
}
```

很好，现在调用 `remove(123, acquireLock: false)` 不再会死锁了。但是很快我发现，在 `acquireLock` 为 `true` 的时候锁也失效了。再仔细阅读 Swift Programming Language 关于 `defer` 的描述：

> A `defer` statement is used for executing code just before transferring program control outside of **the scope that the defer statement appears in**.

所以，上面的代码其实相当于：

```swift
func remove(_ id: ID, acquireLock: Bool) {
    if acquireLock {
        lock.lock()
        lock.unlock()
    }
    tasks[id] = nil
}
```

以前很单纯地认为 `defer` 是在函数退出的时候调用，并没有注意其实是**当前 scope 退出的时候**调用这个事实，造成了这个错误。在 `if`，`guard`，`for`，`try` 这些语句中使用 `defer` 时，应该要特别注意这一点。

## defer和闭包

另一个比较有意思的事实是，虽然 `defer` 后面跟了一个闭包，但是它更多地像是一个语法糖，和我们所熟知的闭包特性不一样，并不会持有里面的值。比如：

```swift
func foo() {
    var number = 1
    defer { print("Statement 2: \(number)") }
    number = 100
    print("Statement 1: \(number)")
}
```

将会输出：

```
Statement 1: 100
Statement 2: 100
```

在 `defer` 中如果要依赖某个变量值时，需要自行进行复制：

```swift
func foo() {
    var number = 1
    var closureNumber = number
    defer { print("Statement 2: \(closureNumber)") }
    number = 100
    print("Statement 1: \(number)")
}

// Statement 1: 100
// Statement 2: 1
```

## defer的执行时机

`defer` 的执行时机紧接在离开作用域之后，但是是在其他语句之前。这个特性为 `defer` 带来了一些很“微妙”的使用方式。比如从 `0` 开始的自增：

```swift
class Foo {
    var num = 0
    func foo() -> Int {
        defer { num += 1 }
        return num
    }
    
    // 没有 `defer` 的话我们可能要这么写
    // func foo() -> Int {
    //    num += 1
    //    return num - 1
    // }
}

let f = Foo()
f.foo() // 0
f.foo() // 1
f.num   // 2
```

输出结果 `foo()` 返回了 `+1` 之前的 `num`，而 `f.num` 则是 `defer` 中经过 `+1` 之后的结果。不使用 `defer`的话，我们其实很难达到这种“在返回后进行操作”的效果。

虽然很特殊，**但是强烈不建议在 defer 中执行这类 side effect**。

> This means that a `defer` statement can be used, for example, to perform manual resource management such as closing file descriptors, and to perform actions that need to happen even if an error is thrown.

从语言设计上来说，`defer` 的目的就是进行资源清理和避免重复的返回前需要执行的代码，而不是用来以取巧地实现某些功能。这样做只会让代码可读性降低。