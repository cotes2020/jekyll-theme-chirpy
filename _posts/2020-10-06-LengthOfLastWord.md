---
layout: post
title: "最后一个单词的长度"
date: 2020-10-06 22:32:00.000000000 +09:00
categories: [算法]
tags: [算法, 最后一个单词的长度]
---

> 给定一个仅包含大小写字母和空格 `' '` 的字符串，返回其最后一个单词的长度。
>
> 如果不存在最后一个单词，请返回 0 。

**说明**

一个单词是指由字母组成，但不包含任何空格的字符串。

**示例:**

```
输入: "Hello World"
输出: 5
```

注意: 大多数人容易忽略考虑末尾是空格的情况。

**思路**

向后迭代字符串，也就是先将字符串反转，然后逐个遍历字符串，利用空格作为判断条件。

**时间复杂度O(n)**

**空间复杂度为O(n)**

**代码实现**

```swift
func lengthOfLastWord(_ s: String) -> Int {
     
    var len = 0
    let sChars = Array(s)
    guard sChars.count != 0 else {
        return 0
    }
    for i in (0...sChars.count - 1).reversed() {

        if len == 0 {

            if sChars[i] == " " {   // 判断末尾是否有空格
                continue
            } else {
                len += 1
            }
        } else {

            if sChars[i] == " " {
                break
            } else {
                len += 1
            }
        }
    }
    return len
}
```

[源码地址](https://github.com/Jovins/Algorithm)