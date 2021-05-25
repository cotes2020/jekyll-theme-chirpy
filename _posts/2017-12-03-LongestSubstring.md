---
layout: post
title: "无重复字符的最长子串"
date: 2017-12-03 23:05:00.000000000 +09:00
categories: [算法]
tags: [算法]
---

> **给定一个字符串，请你找出其中不含有重复字符的 最长子串的长度**.

**示例 1:**

```
输入: "abcabcbb"
输出: 3 
解释: 因为无重复字符的最长子串是 "abc"，所以其长度为 3。
```

**示例 2:**

```
输入: "bbbbb"
输出: 1
解释: 因为无重复字符的最长子串是 "b"，所以其长度为 1。
```

**示例 3:**

```
输入: "pwwkew"
输出: 3
解释: 因为无重复字符的最长子串是 "wke"，所以其长度为 3。
     请注意，你的答案必须是 子串 的长度，"pwke" 是一个子序列，不是子串。
```

**思路**

逐个检查所有的子字符串，看它是否不含有重复的字符。

**代码实现**

```swift
class LengthOfLongestSubstring {
    
    func lengthOfLongestSubstring(_ s: String) -> Int {
        
        var longest = 0
        var index = 0           // 起点
        var set = Set<Character>()
        let sChars = Array(s)
        for (i, char) in sChars.enumerated() {
            
            if set.contains(char) {
  
                longest = max(longest, i - index)
                while sChars[index] != char {
                    set.remove(sChars[index])
                    index += 1
                }
                index += 1
            } else {
                
                set.insert(char)
            }
        }
        return max(longest, sChars.count - index)
    }
}
```

[源码地址](<https://github.com/Jovins/Algorithm>)