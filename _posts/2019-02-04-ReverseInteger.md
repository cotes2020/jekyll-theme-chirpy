---
layout: post
title: "整数反转"
date: 2019-02-04 22:32:00.000000000 +09:00
categories: [算法]
tags: [算法, 整数反转]
---

> 给出一个 32 位的有符号整数，你需要将这个整数中每位上的数字进行反转。

**示例 1:**

```
输入: 123
输出: 321
```

 **示例 2:**

```
输入: -123
输出: -321
```

**示例 3:**

```
输入: 120
输出: 21
```

**注意:**

假设我们的环境只能存储得下 32 位的有符号整数，则其数值范围为 [−231,  231 − 1]。请根据这个假设，如果反转后整数溢出那么就返回 0。

**方法**

弹出和推入数字 & 溢出前进行检查

**思路**

我们可以一次构建反转整数的一位数字。在这样做的时候，我们可以预先检查向原整数附加另一位数字是否会导致溢出。

**算法**

反转整数的方法可以与反转字符串进行类比。

我们想重复“弹出”  x  的最后一位数字，并将它“推入”到 rev 的后面。最后，rev 将与 x 相反。

要在没有辅助堆栈 / 数组的帮助下 “弹出” 和 “推入” 数字，我们可以使用数学方法。

```cpp
//pop operation:
pop = x % 10;
x /= 10;

//push operation:
temp = rev * 10 + pop;
rev = temp;
```

但是，这种方法很危险，因为当 temp = rev * 10 + pop 时会导致溢出。

幸运的是，事先检查这个语句是否会导致溢出很容易。

为了便于解释，我们假设 rev 是正数。

1. 如果 temp = rev * 10 + pop 导致溢出，那么一定有 rev >= INTMAX / 10。
2. 如果 rev > INTMAX / 10，那么 temp = rev * 10 + pop 一定会溢出。
3. 如果 rev == INTMAX / 10，那么只要 pop > 7，temp = rev * 10 + pop 就会溢出。

当 rev 为负时可以应用类似的逻辑。

**复杂度分析**

- 时间复杂度：O(log(x)) ，x 中大约有 O(log10(x))  位数字。
- 空间复杂度：O(1)。

**代码实现**

```swift
class ReverseInteger {
    
    func reverse(_ x: Int) -> Int {
        
        var result = 0
        var x = x
        while x != 0 {
            // 判断是否溢出
            if result > Int(Int32.max) / 10 ||  result < Int(Int32.min) / 10 {
                return 0
            }
            result = result * 10 + x % 10
            x = x / 10
        }
        return result
    }
}
```

[源码地址](<https://github.com/Jovins/Algorithm>)