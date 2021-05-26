---
layout: post
title: "最长公共前缀"
date: 2019-04-22 22:35:00.000000000 +09:00
categories: [算法]
tags: [算法, 最长公共前缀]
---

> 编写一个函数来查找字符串数组中的最长公共前缀。
>
> 如果不存在公共前缀，返回空字符串 `""`。

**示例 1:**

```
输入: ["flower","flow","flight"]
输出: "fl"
```

**示例 2:**

```
输入: ["dog","racecar","car"]
输出: ""
解释: 输入不存在公共前缀。
```

**说明:**

所有输入只包含小写字母 `a-z` 。

**思路**

选择其中一个字符串作为对照标准，与后面的比较获得最长公共前缀res，再用res继续与后面的字符串比较，最后获得最终结果。

**代码实现**

```swift
func longestCommonPrefix(_ strs: [String]) -> String {
     
      // 选择第一个字符串作为对照标准，遍历数组
      // 将数组中单个字符一一对比，只要不同就可以返回
      var res = [Character]()
      // 如果为空 返回”“
      guard let first = strs.first else {
          return ""
      }
      let firstArray = Array(first) // 字符串转数组
      let strArray = strs.map { Array($0) } // 字符串转字符数组
      var index = 0
      while index < first.count {

          res.append(firstArray[index])
          for str in strArray {

              if index >= str.count {

                  return String(res.dropLast()) // 去掉最后那个字符
              }
              if str[index] != res[index] {

                  return String(res.dropLast())
              }
          }
          index += 1
      }
      return String(res)
  }
```

[源码地址](https://github.com/Jovins/Algorithm)