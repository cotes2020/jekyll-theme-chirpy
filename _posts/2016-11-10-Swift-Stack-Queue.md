---
layout: post
title: "栈和队列"
date: 2016-11-10 22:55:00.000000000 +09:00
categories: [算法]
tags: [算法, 栈, 队列]
---

在Swift中没有设的栈和队列，很多扩展库中使用Generic Type来实现栈和队列。正规的做法是用链表来实现，这样可以保证加入或者删除的时间复杂度是O(1)。然而，我觉得最实用的实现方法是使用数组，因为Swift没有现成的链表，而数组又有很多的API可以使用，非常方便。

## 栈和队列的基本概念

对于栈来说，我们需要了解以下几点：

- 栈是**后进先出**的结构。你可以理解成有好几个盘子要垒成一叠，哪个盘子最后叠上去，下次使用的时候它就最先被抽出去。
- 在iOS开发中，如果你要在你的App中添加撤销操作（比如删除图片，恢复删除图片），那么栈是首选数据结构
- 无论在面试还是写App中，只关注栈的这几个基本操作：push, pop, isEmpty, peek, size。

```swift
// 栈是后进先出的结构，好比如叠盘子。
protocol StackProtocol {
    // 持有的元素类型
    associatedtype Element
    // 是否为空
    var isEmpty: Bool { get }
    // 栈的大小
    var size: Int { get }
    // 栈顶元素
    var peek: Element? { get }
    // 进栈
    mutating func push(_ newElement: Element)
    // 出栈
    mutating func pop() -> Element?
}

struct Stack: StackProtocol {
    
    private var stack = [Element]()
    typealias Element = Int
    var isEmpty: Bool { return stack.isEmpty }
    var size: Int { return stack.count }
    var peek: Element? { return stack.last }
    
    mutating func push(_ newElement: Element) {
        stack.append(newElement)
    }
    
    mutating func pop() -> Element? {
        return stack.popLast()
    }
}
```

测试

```swift
// 栈(Stack)是一种后入先出(Last in First Out)的数据结构，仅限定在栈顶进行插入或者删除操作。
// 栈结构的实际应用主要有数制转换、括号匹配、表达式求值等等。
func stackTest() {
    
    // 通过协议创建
    var stack = Stack()
    for i in 1..<10 {
        stack.push(i)
    }
    for _ in 0..<stack.size {
        print(stack.pop()!)
    }
    //    直接创建
    //    var stack = Stacks<Any>()
    //    stack.capacity = 20
    //    for i in 1..<10 {
    //        stack.push(i as Any)
    //    }
    //    for _ in 0..<stack.count {
    //        print(stack.pop() as! Int)
    //    }
}
```

对于队列来说，我们需要了解以下几点：

- 队列是**先进先出**的结构。这个正好就像现实生活中排队买票，谁先来排队，谁先买到票。
- iOS开发中多线程的GCD和NSOperationQueue就是基于队列实现的。
- 关于队列我们只关注这几个操作：enqueue, dequeue, isEmpty, peek, size。

```swift
// 队列是先进先出的结构，比如排队买票
// 这里只关注enqueue、dequeue、isEmpty、size、peek
protocol QueueProtocol {
    // 持有的元素类型
    associatedtype Element
    // 是否为空
    var isEmpty: Bool { get }
    // 队列的大小
    var size: Int { get }
    // 队首元素
    var peek: Element? { get }
    // 入队
    mutating func enqueue(_ newElement: Element)
    // 出队
    mutating func dequeue() -> Element?
}

struct Queue: QueueProtocol {
    
    typealias Element = Int
    private var queue = [Element]()
    private var copy = [Element]()
    
    var isEmpty: Bool {
        return queue.isEmpty && copy.isEmpty
    }
    
    var size: Int {
        return queue.count + copy.count
    }
    
    var peek: Element? {
        return copy.isEmpty ? queue.first : copy.last
    }
    
    mutating func enqueue(_ newElement: Int) {
        queue.append(newElement)
    }
    
    mutating func dequeue() -> Element? {
        
        if copy.isEmpty {
            copy = queue.reversed() // 反转
            queue.removeAll()
        }
        return copy.popLast()
    }
}
```

测试

```swift
// 队列: 先进先出
func queueTest() {
    var queue = Queue()
    for i in 1..<10 {
        queue.enqueue(i)
    }
    for _ in 0..<queue.size {
        print(queue.dequeue()!)
    }
}
```

## 转换

处理栈和队列问题，最经典的一个思路就是使用两个栈/队列来解决问题。也就是说在原栈/队列的基础上，我们用一个协助栈/队列来帮助我们简化算法，这是一种空间换时间的思路。比如

:::tip

用队列实现栈

:::

```swift
// MARK: 队列实现栈
struct MyStack {
    var queueA: Queue
    var queueB: Queue
    init() {
        queueA = Queue()
        queueB = Queue()
    }
    
    var isEmpty: Bool {
        return queueA.isEmpty && queueB.isEmpty
    }
    
    var peek: Any? {
        
        mutating get {
            shift()
            let peekObj = queueA.peek
            queueB.enqueue(queueA.dequeue()!)
            swap()
            return peekObj
        }
    }
    
    var size: Int {
        return queueA.size
    }
    
    mutating func push(object: Any) {
        queueA.enqueue(object as! Int)
    }
    
    mutating func pop() -> Any? {
        shift()
        let popObj = queueA.dequeue()
        swap()
        return popObj
    }
    
    mutating private func shift() {
        while queueA.size != 1 {
            queueB.enqueue(queueA.dequeue()!)
        }
    }
    
    mutating private func swap() {
        (queueA, queueB) = (queueB, queueA)
    }
}
```

:::tip

用栈来实现队列

:::

```swift
// 用栈实现队列
struct MyQueue {
    
    var stackA: Stack
    var stackB: Stack
    init() {
        stackA = Stack()
        stackB = Stack()
    }
    
    var isEmpty: Bool {
        return stackA.isEmpty && stackB.isEmpty
    }
    
    var peek: Any? {
        mutating get {
            shift()
            return stackB.peek
        }
    }
    
    var size: Int {
        return stackA.size + stackB.size
    }
    
    mutating func enqueue(object: Any) {
        stackA.push(object as! Stack.Element)
    }
    
    mutating func dequeue() -> Any {
        shift()
        return stackB.pop()!
    }
 
    mutating fileprivate func shift() {
        if stackB.isEmpty {
            while !stackA.isEmpty {
                stackB.push(stackA.pop()!)
            }
        }
    }
}
```

上面两种实现方法都是使用两个相同的数据结构，然后将元素由其中一个转向另一个，从而形成一种完全不同的数据。

## 实战

下面是Facebook一道真实的面试题。

> Given an absolute path for a file (Unix-style), simplify it.
>  For example,
>  **path** = "/home/", => "/home"
>  **path** = "/a/./b/../../c/", => "/c"

这道题目一看，这不就是我们平常在terminal里面敲的cd啊pwd之类的吗，好熟悉啊。

根据常识，我们知道以下规则：

-  **.** 代表当前路径。比如 /a/. 实际上就是 /a，无论输入多少个 **.** 都返回当前目录
-  **..**代表上一级目录。比如 /a/b/.. 实际上就是 /a，也就是说先进入a目录，再进入其下的b目录，再返回b目录的上一层，也就是a目录。

然后针对以上信息，我们可以得出以下思路：

1. 首先输入是个 String，代表路径。输出要求也是 String, 同样代表路径。
2. 我们可以把 input 根据 “/” 符号去拆分，比如 "/a/b/./../d/" 就拆成了一个String数组["a", "b", ".", "..", "d"]
3. 创立一个栈然后遍历拆分后的 String 数组，对于一般 String ，直接加入到栈中，对于 ".." 那我们就对栈做pop操作，其他情况不错处理

思路有了，代码也就有了

```swift
// let path: String = "/home/" // "/a/./b/../../c/"
// print(simplifyPath(path: path))
func simplifyPath(path: String) -> String {
    
    // 用数组来实现栈功能
    var pathStack = [String]()
    // 拆分原路径
    let paths = path.components(separatedBy: "/")
    for p in paths {
        // 对于"."直接跳过
        guard p != "." else {
            continue
        }
        // 对于“.."使用pop操作
        if p == ".." {
            if pathStack.count > 0 {
                pathStack.removeLast()
            }
        } else if p != "" { // 对于数组中空字符串的特殊操作
            pathStack.append(p)
        }
    }
    // 将栈中的内容转化为优化后的新路径
    let res = pathStack.reduce("") { total, dir in "\(total)/\(dir)"}
    // 注意空路径的结果是"/"
    return res.isEmpty ? "/" : res
}
```

## 总结

在Swift中，栈和队列是比较特殊的数据结构，个人认为最实用的实现方法是利用数组。虽然它们本身比较抽象，却是很多复杂数据结构和iOS开发中的功能模块的基础。这也是一个工程师进阶之路理应熟练掌握的两种数据结构。

[源码地址](<https://github.com/Jovins/Algorithm>)