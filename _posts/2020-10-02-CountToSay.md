---
layout: post
title: "报数"
date: 2020-10-02 21:12:00.000000000 +09:00
categories: [算法]
tags: [算法, 报数]
---

> 报数序列是一个整数序列，按照其中的整数的顺序进行报数，得到下一个数。其前五项如下：

```
1.     1
2.     11
3.     21
4.     1211
5.     111221
```

1 被读作  "one 1"  ("一个一") , 即 11。
11 被读作 "two 1s" ("两个一"）, 即 21。
21 被读作 "one 2",  "one 1" （"一个二" ,  "一个一") , 即 1211。

给定一个正整数 n（1 ≤ n ≤ 30），输出报数序列的第 n 项。

注意：整数顺序将表示为一个字符串。

**示例 1:**

```
输入: 1
输出: "1"
```

**示例 2:**

```
输入: 4
输出: "1211"
```

**规律**

- 1
- 2 描述的是1，是一个1，也就是11
- 3 描述的是11，是两个1，也就是21
- 4 描述的是21，是一个2一个1，也就是12-11
- 5 描述的是1211, 是一个1，一个2，两个1，也就是11-12-21
- 6 描述的是111221，是三个1，两个2，一个1，也就是31-22-11
- 7 描述的是312211，是一个3一个1两个2两个1，也即是13-11-22-21
- 以此类推

## 暴力法

由于给出的条件是: 给定一个正整数 n（1 ≤ n ≤ 30，把所有的可能性列出来再匹配。

**代码**

```swift
// 暴力法
func countAndSay(_ n: Int) -> String {

    switch n {
    case 1:
        return "1"
    case 2:
        return "11"
    case 3:
        return "21"
    case 4:
        return "1211"
    case 5:
        return "111221"
    case 6:
        return "312211"
    case 7:
        return "13112221"
    case 8:
        return "1113213211"
    case 9:
        return "31131211131221"
    case 10:
        return "13211311123113112211"
    case 11:
        return "11131221133112132113212221"
    case 12:
        return "3113112221232112111312211312113211"
    case 13:
        return "1321132132111213122112311311222113111221131221"
    case 14:
        return "11131221131211131231121113112221121321132132211331222113112211"
    default:
        return "等等，不推荐"
    }
}
```

## 递归法

**代码**

```swift
func countAndSay(_ n: Int) -> String {
        
    if n == 1 {
        return "1"
    }
    let lastStr = countAndSay(n - 1)
    var charStr = Array(lastStr)
    var count = 1
    var res = ""
    for i in 0..<charStr.count - 1 {

        if charStr[i] == charStr[i + 1] {
            count += 1
            continue
        } else {
            res += String(count) + String(charStr[i])
            count = 1
        }
    }
    return res + String(count) + String(charStr[charStr.count - 1])
}
```

[源码地址](https://github.com/Jovins/Algorithm)