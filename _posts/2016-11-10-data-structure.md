---
layout: post
title: "数据结构"
date: 2016-11-10 20:22:00.000000000 +09:00
categories: [算法]
tags: [算法基础, 数据结构, 字典, 字符串, 数组]
---

数组是最基本的数据结构。在Swift中，以前Objective-C时代中将NSMutableArray和NSArray分开的做法，被统一到了唯一的数据结构—Array。虽然看上去就一种数据结构，其实它的实现有三种:

- contiguousArray: 效率最高，元素分配在连续的元素上。如果数组是只类型(栈上操作)，则Swift会自动调用Array的这种实现；如果注重效率，那么推荐声明这种高磊鑫个，尤其是在大量元素的类时，这样的效果会更好。

- .Array: 会自动桥接到Objective-C中国的NSArray上，如果是值类型，则其性能与ContiguousArray无差别。
- ArraySlice: 它不是一个新的数组，只是一个片段，在内存上与原数组享用同一个区域。

下面是数组的一些最基本的使用:

```swift
var nums = [3, 4, 1, 6, 5, 8]
// 对原数组进行升序排序
// nums.sort()
// 对原数组进行降序排序
nums.sort(by: >)
print(nums)
```

不要小看这些简单的操作: 数组可以依靠它们显示更多的数据结构。虽然不想Java有现成的队列和栈，但完全可以用数组配合最简单的操作实现这些数据结构，下面就是用数组实现栈的示例代码:

```swift
//    直接创建
var stack = Stacks<Any>()
stack.capacity = 20
for i in 1..<10 {
		stack.push(i as Any)
}
for _ in 0..<stack.count {
		print(stack.pop() as! Int)
}
```

```swift
struct Stacks<T> {
    
    // 声明一个泛型数组，用于储存栈中的元素
    private var elements = [T]()
    
    // 返回栈结构中元素的个数
    public var count: Int {
        return elements.count
    }
    
    // 获取或者设置栈的存储容量
    public var capacity: Int {
        get {
            return elements.capacity
        }
        set {
            elements.reserveCapacity(newValue) // 用于为原数组预留空间，防止数组在增加或者删除元素时反复申请内存空间或者创建新的数组
        }
    }
    
    // 初始化
    public init(){}
    
    // 使用push方法执行入栈操作
    public mutating func push(_ element: T) {
        // 判断是否已经满
        if count == capacity {
            fatalError("栈已满，不能再执行入栈")
        }
        self.elements.append(element)
    }
    
    // 使用pop方法执行出栈操作
    @discardableResult
    public mutating func pop() -> T? {
        // 是否为空
        if count == 0 {
            fatalError("栈已经空，不能再执行出栈操作")
        }
        return elements.popLast() // 返回并且删除最后一个元素
    }
    
    // 返回栈顶元素
    public func peek() -> T? {
        return elements.last
    }
    
    // 清空所有元素
    public mutating func clearAll() {
        elements.removeAll()
    }
    
    // 判断是否为空
    public mutating func isEmpty() -> Bool {
        return elements.isEmpty
    }
    
    // 判断栈是否满了
    public mutating func isFull() -> Bool {
        if count == 0 {
            return false
        } else {
            return count == elements.capacity
        }
    }
}
```

最后要特别强调一个操作: reserveCapacity(), 它用于为原数组预留空间，防止数组在增加或者删除元素时反复申请内存空间或者是创建新数组，特别适用于创建和removeAll()时进行调用，对于整段代码可以起到提高性能的作用。

## 字典和集合

字典和集合(指的是HashSet)经常被适用的原因在于，查找数据的时间复杂度为O(1)。一般字典和集合要求他们的Key都必须遵守Hashable协议，Cocoa中的基本数据类型都满足这一点；自定义的class需要实现Hashable，而又因为Hashable是对Equable的扩展，所以还要重载 === 运算符。

下面是关于字典和集合的一些实用操作:

```swift
let primeNums: Set = [3, 5, 7, 11, 13]
let oddNums: Set = [1, 3, 5, 7, 9]

// 交集、并集、差集
let primeAndOddNum = primeNums.intersection(oddNums)
print(primeAndOddNum)
let primeOrOddNum = primeNums.union(oddNums)
print(primeOrOddNum)
let primeNotOddNum = primeNums.subtracting(oddNums)
print(primeNotOddNum)
```

还有字典和集合在实际开发中经常与数组配合使用，下面有道算法题:

**给出一个整型数组和一个目标值，判断数组中是否有两个数之和等于目标值**

这道题是经典的"2Sum"，即已经有一个数组记为nums，也有一个目标值记为target，最后返回一个Bool值。

最粗暴的方法就是每次选中一个数，然后遍历整个数组，判断是否有另一个数是两者之和为target。这种方法的时间复杂度为O(n^2)。

采用集合可以优化时间复杂度，即在变量数组的过程中，用集合每次保存当前值。假如集合中已经有了一个数等于目标值减去当前值，则证明在之前的遍历中一定有一个数与当前值之和等于目标值。这种方法的时间复杂度为O(n)，代码如下：

```swift
// let nums: [Int] = [1, 2, 3, 4, 5]
// print(twoSum(nums, 7))
func twoSum(_ nums:[Int], _ target: Int) -> Bool {
    var set = Set<Int>()
    for num in nums {
        if set.contains(target - num) {
            return true
        }
        set.insert(num)
    }
    return false
}
```

如果把这道题稍微改一下，则变为:

**给定一个整型数组中有且只有两个数之和等于目标值，求这两个数组中的序号**

解决思路与上道题目基本类似，但是为了方便得到序列号，这里使用字典，而此方法的时间复杂度依然是O(n)。代码如下:

```swift
// 给定一个整型数组中有且只有两个数之和等于目标值，求这两个数在数组中的序号
// let nums: [Int] = [2, 5, 8, 3, 14]
// print(twoSum2(nums, 16))
func twoSum2(_ nums:[Int], _ target: Int) -> [Int] {
    var dic = [Int: Int]()
    for (i, num) in nums.enumerated() {
        if let index = dic[target - num] {
            return [index, i]
        } else {
            dic[num] = i
        }
    }
    fatalError("不存在")
}
```

## 字符串

字符串在算法实战中及其常见。在Swift中，字符串不同于其他语言(Objective-C)，他是值类型，而非引用类型。首先列举一下字符串的通常用法。

```swift
// 访问字符串中的单个字符，时间复杂度为O(1)
let str = "Hello World"
let char = str[str.index(str.startIndex, offsetBy: 3)]
print(char)
```

关于字符串，下面先看一道以前Google的面试题目:

**给出一个字符串，要求将其按照单词顺序进行反转**

这个题目看起来有两个问题:

- 每个单词长度不一样
- 空格需要特殊处理

```swift
// 字符串反转(思想: 元组)
// let str = "abcdef" -> fedcba
// var chars = Array(str)
// reserses(&chars, 0, chars.count - 1)
func reserses<T>(_ chars: inout [T], _ start: Int,_ end: Int) {
    var start = start, end = end
    while start <= end {
        swaps(&chars, start, end)
        start += 1
        end -= 1
    }
}

func swaps<T>(_ chars: inout [T], _ s: Int, _ e: Int) {
    (chars[s], chars[e]) = (chars[e], chars[s])
}
```

有了这个方法，就可以实行下面两种字符串的反转:

- 整个字符串反转，"Love You Forever" -> "reverof ouY  evoL"
- 每个单词作为一个字符串单独反转，"reverof ouY  evoL" -> "Forever You Love"

整体思路就有了，然后就可以解决这道题了。

```swift
// 给出一个字符串，要求将其按照单词顺序进行反转。比如"the sky is blue" -> "blue is sky the"
// let str = "I Love You"
// let rts = reverseWords(str) as! String
// print(rts)
func reverseWords(_ str: String?) -> String? {
    guard let str = str else {
        return nil
    }
    var chars = Array(str), start = 0
    reserses(&chars, start, chars.count - 1) // 将字符串反转
    for i in 0..<chars.count {  // 将里面每个单词再反转
        if i == chars.count - 1 || chars[i + 1] == " " {
            reserses(&chars, start, i)
            start = i + 2
        }
    }
    return String(chars)
}
```

时间复杂度还是O(n)，但整体思路和代码简单很多。

## 总结

在Swift中，数组、字符串、集合以及字典是最基本的数据结构，但是围绕这些数据结构的问题层出不穷。而在日常生活开发中，他们使用起来也非常高效(栈上运行)和安全(无须顾虑现成问题)，因为它们都是值类型。

[源码地址](<https://github.com/Jovins/Algorithm>)  