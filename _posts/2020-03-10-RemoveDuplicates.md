---
layout: post
title: "删除排序数组中的重复项"
date: 2020-03-10 22:08:00.000000000 +09:00
categories: [算法]
tags: [算法, 删除排序数组中的重复项]
---

> 给定一个排序数组，你需要在**原地**删除重复出现的元素，使得每个元素只出现一次，返回移除后数组的新长度。
>
> 不要使用额外的数组空间，你必须在**原地修改输入数组**并在使用 O(1) 额外空间的条件下完成。

**示例 1:**

```
给定数组 nums = [1,1,2], 

函数应该返回新的长度 2, 并且原数组 nums 的前两个元素被修改为 1, 2。 

你不需要考虑数组中超出新长度后面的元素。
```

**示例 2:**

```
给定 nums = [0,0,1,1,1,2,2,3,3,4],

函数应该返回新的长度 5, 并且原数组 nums 的前五个元素被修改为 0, 1, 2, 3, 4。

你不需要考虑数组中超出新长度后面的元素。
```

**说明:**

为什么返回数值是整数，但输出的答案是数组呢?

请注意，输入数组是以**“引用”**方式传递的，这意味着在函数里修改输入数组对于调用者是可见的。

你可以想象内部操作如下:

```
// nums 是以“引用”方式传递的。也就是说，不对实参做任何拷贝
int len = removeDuplicates(nums);

// 在函数里修改输入数组对于调用者是可见的。
// 根据你的函数返回的长度, 它会打印出数组中该长度范围内的所有元素。
for (int i = 0; i < len; i++) {
    print(nums[i]);
}
```

**解题思路**

使用快慢指针来记录遍历的坐标。

+ 开始时这两个指针都指向第一个数字
+ 如果两个指针指的数字相同，则快指针向前走一步
+ 如果不同，则两个指针都向前走一步
+ 当快指针走完整个数组后，慢指针当前的坐标加1就是数组中不同数字的个数

**复杂度分析**

- 时间复杂度：O(n)，假设数组的长度是 n，那么 i 和 j 分别最多遍历 n 步。
- 空间复杂度：O(1)

**代码实现**

```swift
func removeDuplicates(_ nums: inout [Int]) -> Int {
        
    guard nums.count > 0 else {
        return 0
    }
    var i = 0
    for j in 1..<nums.count {

        if nums[i] != nums[j] {

            i += 1
            nums[i] = nums[j]
        }
    }
    return i + 1
}
```

**方法二**

```swift
func removeDuplicates(_ nums: inout [Int]) -> Int {
        
    guard nums.count > 0 else {
        return 0
    }
    var index = 0
    for num in nums where num != nums[index] {

        index += 1
        nums[index] = num
    }
    return index + 1
}
```

[源码地址](https://github.com/Jovins/Algorithm)

