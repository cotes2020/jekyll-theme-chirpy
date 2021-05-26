---
layout: post
title: "接雨水"
date: 2020-10-04 22:24:00.000000000 +09:00
categories: [算法]
tags: [算法, 接雨水]
---

> 给定 *n* 个非负整数表示每个宽度为 1 的柱子的高度图，计算按此排列的柱子，下雨之后能接多少雨水。

![](/assets/images/al-trapRainWater-01.png)

上面是由数组 [0,1,0,2,1,0,1,3,2,1,2,1] 表示的高度图，在这种情况下，可以接 6 个单位的雨水（蓝色部分表示雨水）。 

**示例:**

```
输入: [0,1,0,2,1,0,1,3,2,1,2,1]
输出: 6
```

**思路**

这道题真正**难点**在于: 在一个位置能容下的雨水量等于它左右两边柱子最大高度的最小值减去它的高度.比如下图所示:

![](/assets/images/al-trapRainWater-02.png)

## 暴力法

**直观想法**

直接按问题描述进行。对于数组中的每个元素，我们找出下雨后水能达到的最高位置，等于两边最大高度的较小值减去当前高度的值。

**算法**

+ 初始化res = 0
+ 从当前数据向左向右扫描数组，找出最大的数
  + 初始化max_left = 0 和max_right = 0
  + 从当前元素向左扫面并且更新
    + max_left = max(max_left, height[i])
  + 从当前元素向右扫描并且更新
    + max_right = max(max_right, height[i])
  + 将min(max_left, max_right) - height[i]累加到res中

**复杂性分析**

- 时间复杂度： O(n^2)。数组中的每个元素都需要向左向右扫描。
- 空间复杂度 O(1)  的额外空间。 

**代码实现**(会超出时间限制, 不可取)

```swift
func trap(_ height: [Int]) -> Int {
        
        guard height.count > 1 else {
            return 0
        }
        
        var res = 0
        for i in 1..<height.count - 1 {
            
            var max_left = 0
            var max_right = 0
            for j in (0...i).reversed() {
                max_left = max(max_left, height[j])
            }
            for k in i..<height.count {
                max_right = max(max_right, height[k])
            }
            res += min(max_left, max_right) - height[i]
        }
        return res
    }
}
```

##动态编程法

**思路**

在暴力方法中，我们仅仅为了找到最大值每次都要向左和向右扫描一次。但是我们可以提前存储这个值。因此，可以通过动态编程解决。

这个概念可以见下图解释：

![](/assets/images/al-trapRainWater-03.png)

**算法**

- 找到数组中从下标 i 到最左端最高的条形块高度 left_max。
- 找到数组中从下标 i 到最右端最高的条形块高度 right_max。
- 扫描数组 height 并更新答案：
- 累加min(max_left[i], max_right[i]) - height[i]到res上

**复杂性分析**

- 时间复杂度：O(n)。
  - 存储最大高度数组，需要两次遍历，每次 O(n) 。
  - 最终使用存储的数据更新\text{ans}ans ，O(n)。
- 空间复杂度：O(n)  额外空间。
  - 和方法 1 相比使用了额外的 O(n) 空间用来放置 left_max和 right_max 数组。 

**代码实现**

```swift
func trap(_ height: [Int]) -> Int {
        
    guard height.count > 1 else {
        return 0
    }
    let count = height.count
    var res = 0
    var lefts: [Int] = Array(repeating: 0, count: count)
    var rights: [Int] = Array(repeating: 0, count: count)
    lefts[0] = height[0]
    for i in 1..<count {

        lefts[i] = max(height[i], lefts[i - 1])
    }
    rights[count - 1] = height[count - 1]
    for j in (0...count - 2).reversed() {
        rights[j] = max(height[j], rights[j + 1])
    }
    for k in 1..<count - 1 {
        res += min(lefts[k], rights[k]) - height[k]
    }
    return res
}
```

## 栈的应用

**思路**

我们可以不用像方法 2 那样存储最大高度，而是用栈来跟踪可能储水的最长的条形块。使用栈就可以在一次遍历内完成计算。

我们在遍历数组时维护一个栈。如果当前的条形块小于或等于栈顶的条形块，我们将条形块的索引入栈，意思是当前的条形块被栈中的前一个条形块界定。如果我们发现一个条形块长于栈顶，我们可以确定栈顶的条形块被当前条形块和栈的前一个条形块界定，因此我们可以弹出栈顶元素并且累加答案到 res 。

**算法**

+ 使用栈来存储条形块的索引下标
+ 遍历数组
  + 当栈非空且height[current] > height[stack.top]
    + 意味着栈中元素可以被弹出。弹出栈顶元素 top。
    + 计算当前元素和栈顶元素的距离，准备进行填充操作 distance = current - stack.top() - 1
    + 找出界定高度bound_height = min(height[current], height[stack.top]) - height[top]
    + 往res中累加积水量res = distance * bound_height
  + 将当前的索引下标
  + 将current移动到下一个位置

**复杂性分析**

+ 时间复杂度 O(n)。
  + 单次遍历 O(n) ，每个条形块最多访问两次（由于栈的弹入和弹出），并且弹入和弹出栈都是 O(1) 的。
+ 空间复杂度：O(n) 。 栈最多在阶梯型或平坦型条形块结构中占用 O(n) 的空间。 

**代码实现**

```swift
func trap(_ height: [Int]) -> Int {
        
    guard height.count > 1 else {
        return 0
    }
    var res = 0
    var cur = 0
    var stack = [Int]()
    while cur < height.count {

        // 非空肯定有值
        while !stack.isEmpty && height[cur] > height[stack.last!] {

            let top = stack.last!   // 记录top索引
            let _ = stack.popLast() // 栈顶出栈
            if stack.isEmpty {
                break
            }
            let distance = cur - stack.last! - 1
            let bound_height = min(height[cur], height[stack.last!]) - height[top]
            res += distance * bound_height
        }
        stack.append(cur)
        cur += 1
    }
    return res
}
```

## 双指针

**思路**

和方法 2 相比，我们不从左和从右分开计算，我们想办法一次完成遍历。 从动态编程方法的示意图中我们注意到，只要 right_max[i] > left_max[i]（元素 0 到元素 6），积水高度将由 left_max 决定，类似地 left_max[i] > right_max[i]（元素 8 到元素 11）。 所以我们可以认为如果一端有更高的条形块（例如右端），积水的高度依赖于当前方向的高度（从左到右）。当我们发现另一侧（右侧）的条形块高度不是最高的，我们则开始从相反的方向遍历（从右到左）。 我们必须在遍历时维护 left_max 和 right_max ，但是我们现在可以使用两个指针交替进行，实现 1 次遍历即可完成。

**算法**

+ 初始化left指针为 0并且right 指针为size - 1
+ while left < right
  + if height[left] < height[right] 
    + if height[left] >= left_max，更新left_max
    + else 累加left_max - height[left]到res
    + left = left + 1
  + else 
    + if height[right] >= right_max，更新right_max
    + else 累加right_max - height[right]到res
    + right = right - 1

**复杂性分析**

- 时间复杂度：O(n)。单次遍历的时间O(n)。
- 空间复杂度：O(1) 的额外空间。left, right, left_max和right_max 只需要常数的空间。

**代码实现**

```swift
// 效率最高
func trap(_ height: [Int]) -> Int {
        
    guard height.count > 1 else {
        return 0
    }
    var left = 0, right = height.count - 1, res = 0
    var left_max = 0, right_max = 0
    while left < right {

        if height[left] < height[right] {

            height[left] >= left_max ? (left_max = height[left]) : (res += left_max - height[left])
            left += 1
        } else {

            height[right] >= right_max ? (right_max = height[right]) : (res += right_max - height[right])
            right -= 1
        }
    }
    return res
}
```

[源码地址](https://github.com/Jovins/Algorithm)