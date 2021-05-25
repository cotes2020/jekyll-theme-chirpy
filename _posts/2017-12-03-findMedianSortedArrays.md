---
layout: post
title: "寻找两个有序数组的中位数"
date: 2017-12-03 23:30:00.000000000 +09:00
categories: [算法]
tags: [算法]
---

> 给定两个大小为 m 和 n 的有序数组 `nums1` 和 `nums2`。
>
> 请你找出这两个有序数组的中位数，并且要求算法的时间复杂度为 O(log(m + n))。
>
> 你可以假设 `nums1` 和 `nums2` 不会同时为空

**示例 1:**

```
nums1 = [1, 3]
nums2 = [2]

则中位数是 2.0
```

**示例 2:**

```
nums1 = [1, 2]
nums2 = [3, 4]

则中位数是 (2 + 3)/2 = 2.5
```

## 什么是中位数

所谓中位数就是一组数据从小到大排列中间的那个数字。但是有的时候一组数据个数是偶数的话就是中间两个数字相加除以2。比如:

`odd : [1,| 2 |,3]`，`2` 就是这个数组的中位数，左右两边都只要 1 位；

`even: [1,| 2, 3 |,4]`，`(2+3)/2` 就是这个数组的中位数，左右两边 1 位；

## 中位数的作用

为了解决这个问题，我们需要理解 “中位数的作用是什么”。在统计中，中位数被用来：

> 将一个集合划分为两个长度相等的子集，其中一个子集中的元素总是大于另一个子集中的元素。

## 思路

有两个数组：

```
num1: [a1,a2,a3,...am]
nums2: [b1,b2,b3,...bn]
[nums1[m1],nums2[n1] | nums1[m2], nums2[n2]]
```

只要保证左右两边 **个数** 相同，中位数就在 `|` 这个边界旁边产生。

如何找边界值，我们可以用二分法，我们先确定 `num1` 取 `m1` 个数的左半边，那么 `num2` 取 `m2 = (m+n+1)/2 - m1` 的左半边，找到合适的 `m1`，就用二分法找。

当 `[ [a1],[b1,b2,b3] | [a2,..an],[b4,...bn] ]`

我们只需要比较 `b3` 和 `a2` 的关系的大小，就可以知道这种分法是不是准确的！

例如: 

```
nums1 = [-1,1,3,5,7,9]
nums2 =[2,4,6,8,10,12,14,16]
```

当 `m1 = 4,m2 = 3`,它的中位数就是`median = (num1[m1] + num2[m2])/2`

**复杂度分析**

- 时间复杂度：O(log(min(m,n))
  首先，查找的区间是 [0, m]。 而该区间的长度在每次循环之后都会减少为原来的一半。 所以，我们只需要执行log(m)次循环。由于我们在每次循环中进行常量次数的操作，所以时间复杂度为 O(log(m)。 由于 m ≤ n，所以时间复杂度是 O(log(min(m,n))。
- 空间复杂度：O(1)， 我们只需要恒定的内存来存储 99 个局部变量， 所以空间复杂度为 O(1)。

**代码实现**

```swift
func findMedianSortedArrays(_ nums1: [Int], _ nums2: [Int]) -> Double {
        
    let m = nums1.count
    let n = nums2.count
    if m > n {
        return findMedianSortedArrays(nums2, nums1)
    }
    let k = (m + n + 1) / 2
    var left = 0
    var right = m
    // num1数组右移，num2数组左移,找出中间位置
    while left < right {
        let m1 = left + (right - left) / 2
        let m2 = k - m1
        if nums1[m1] < nums2[m2 - 1] {
            left = m1 + 1
        } else {
            right = m1
        }
    }
    let m1 = left
    let m2 = k - left
    let re1: Double = Double(max(m1 <= 0 ? Int.min : nums1[m1 - 1], m2 <= 0 ? Int.min : nums2[m2 - 1]))
    if (m + n) % 2 == 1 {
        return re1
    }
    let re2: Double = Double(min(m1 >= m ? Int.max : nums1[m1], m2 >= n ? Int.max : nums2[m2]))
    return (re1 + re2) / 2
}
```

[源码地址](https://github.com/Jovins/Algorithm)