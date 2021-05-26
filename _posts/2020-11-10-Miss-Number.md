---
layout: post
title: "缺失数字"
date: 2020-11-10 21:55:00.000000000 +09:00
categories: [算法]
tags: [算法, 缺失数字]
---

> 给定一个包含 `0, 1, 2, ..., n` 中 *n* 个数的序列，找出 0 .. *n* 中没有出现在序列中的那个数。

**示例 1:**

```
输入: [3,0,1]
输出: 2
```

**示例 2:**

```
输入: [9,6,4,2,3,5,7,0,1]
输出: 8
```

**说明:**
你的算法应具有线性时间复杂度。你能否仅使用额外常数空间来实现?

![](/assets/images/al-missNumber-01.png)

## 异或法

> 只出现一次的数字: 给定一个**非空**整数数组，除了某个元素只出现一次以外，其余每个元素均出现两次。找出那个只出现了一次的元素。

如果我们补充一个完整的数组和原数组进行组合，那所求解的问题就变成了 **只出现一次的数字**。

将少了一个数的数组与 0 到 n 之间完整的那个数组进行异或处理，因为相同的数字异或会变为了 0 ，那么全部数字异或后，剩下的就是少了的那个数字。

**如图**

![](/assets/images/al-missNumber-02.png)

**代码实现**

```swift
func missingNumber(_ nums: [Int]) -> Int {
        
    var res = nums.count
    for i in 0..<nums.count {

        res ^= nums[i]
        res ^= i
    }
    return res
}
```

## 求和法

**思路**

- 求出 0 到 n 之间所有的数字之和
- 遍历数组计算出原始数组中数字的累积和
- 两和相减，差值就是丢失的那个数字

**图解**

![](/assets/images/al-missNumber-03.gif)

**代码实现**

```swift
func missingNumber(_ nums: [Int]) -> Int {
    
    // 先求出0...n的总和
    var res = (1 + nums.count) * nums.count / 2
    for num in nums {

        res -= num
    }
    return res
}
```

## 二分法

将数组进行排序后，利用二分查找的方法来找到缺少的数字，注意搜索的范围为 0 到 n 。

- 首先对数组进行排序
- 用元素值和下标值之间做对比，如果元素值大于下标值，则说明缺失的数字在左边，此时将 right 赋为 mid ，反之则将 left 赋为 mid + 1 。

注：由于一开始进行了排序操作，因此使用二分法的性能是不如上面两种方法。

**代码实现**

```swift
func missingNumber(_ nums: [Int]) -> Int {
        
    let res = nums.sorted(by: <)
    var left = 0
    var right = res.count
    while left < right {

        let mid = (left + right) / 2
        if mid < res[mid] {

            right = mid
        } else {

            left = mid + 1
        }
    }
    return left
}
```

[源码地址](<https://github.com/Jovins/Algorithm>)