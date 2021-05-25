---
layout: post
title: "两个数之和"
date: 2017-12-02 21:05:00.000000000 +09:00
categories: [算法]
tags: [算法, 两个数之和]
---

> 给一个整型数组和一个目标值，判断数组中是否有两个数字之和等于目标值

这道题是传说中经典的2Sum，我们已经有一个数组记为nums，也有一个目标值记为target，最后要返回一个Bool值。

最粗暴的方法就是每次选中一个数，然后遍历整个数组，判断是否有另一个数使两者之和为target。这种做法时间复杂度为O(n^2)。
 采用集合可以优化时间复杂度。在遍历数组的过程中，用集合每次保存当前值。假如集合中已经有了**目标值减去当前值**，则证明在之前的遍历中一定有一个数与当前值之和等于目标值。这种做法时间复杂度为O(n)，代码如下。

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

> 给定一个整数数组 `nums` 和一个目标值 `target`，请你在该数组中找出和为目标值的那 **两个** 整数，并返回他们的数组下标。

你可以假设每种输入只会对应一个答案。但是，你不能重复利用这个数组中同样的元素。

**示例:**

```
给定 nums = [2, 7, 11, 15], target = 9

因为 nums[0] + nums[1] = 2 + 7 = 9
所以返回 [0, 1]
```

实现:

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

[源码地址](<https://github.com/Jovins/Algorithm>)