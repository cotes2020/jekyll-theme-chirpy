---
layout: post
title: "搜索插入位置"
date: 2020-08-05 20:05:00.000000000 +09:00
categories: [算法]
tags: [算法, 搜索插入位置]
---

> 给定一个排序数组和一个目标值，在数组中找到目标值，并返回其索引。如果目标值不存在于数组中，返回它将会被按顺序插入的位置。
>
> 你可以假设数组中无重复元素。

**示例 1:**

```
输入: [1,3,5,6], 5
输出: 2
```

**示例 2:**

```
输入: [1,3,5,6], 2
输出: 1
```

**示例 3:**

```
输入: [1,3,5,6], 7
输出: 4
```

**示例 4:**

```
输入: [1,3,5,6], 0
输出: 0
```

**循环搜索**

```swift
func searchInsert(_ nums: [Int], _ target: Int) -> Int {
        
    guard nums.count > 0 else {
        return 0
    }
    for i in 0..<nums.count {

        if nums[i] >= target {
            return i
        }
    }
    return nums.count
}
```

**二分查找**

```swift
func searchInsert(_ nums: [Int], _ target: Int) -> Int {
     
    var pre = 0, mid = 0, las = nums.count
    while pre < las {

        mid = pre + (las - pre) / 2
        if nums[mid] < target {

            pre = mid + 1
        } else {
            las = mid
        }
    }
   return pre
}
```

[源码地址](https://github.com/Jovins/Algorithm)