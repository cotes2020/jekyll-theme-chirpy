---
title: "剑指Offer JavaScript-查找专题"
date: 2019-06-23
permalink: /2019-06-23-find-min-num/
categories: ["开源技术课程", "JavaScript版·剑指offer刷题笔记"]
---
## 旋转数组最小的数字


### 1. 题目描述


把一个数组最开始的若干个元素搬到数组的末尾，我们称之为数组的旋转。


输入一个递增排序的数组的一个旋转，输出旋转数组的最小元素。例如数组{3,4,5,1,2}为{1,2,3,4,5}的一个旋转，该数组的最小值为 1。


### 2. 解题思路


最简单的肯定是从头到尾遍历，复杂度是 $O(N)$。**这种方法没有利用“旋转数组”的特性**。


借助二分查找的思想，时间复杂度可以降低到 $O(log(N))$。


可以通过以下方法确定最小值元素的位置，然后移动指针，缩小范围：

- 中间指针对应的元素 ≥ 左侧元素, 那么中间元素位于原递增数组中, 最小值在右侧
- 中间指针对应的元素 ≤ 右侧元素, 那么中间元素位于被移动的递增数组中，最小值在左侧

特殊情况，如果三者相等，那么无法判断最小值元素的位置，就退化为普通遍历即可。


### 3. 代码


先上一段二分查找和实现思路：


```typescript
/**
 * 二分查找
 * @param {Array} arr
 * @param {*} elem
 */
function binarySearch(arr, elem) {
    let left = 0,
        right = arr.length - 1,
        mid = -1;

    while (left <= right) {
        // 注意是≤：考虑只剩1个元素的情况
        mid = Math.floor((left + right) / 2);

        if (arr[mid] === elem) {
            return true;
        }

        if (elem < arr[mid]) {
            right = mid - 1;
        } else {
            left = mid + 1;
        }
    }

    return false;
}

/**
 * 测试代码
 */
console.log(binarySearch([1, 2], 2));
console.log(binarySearch([1, 2], -1));
console.log(binarySearch([1, 2, 10], 2));
```


借助二分查找的思想，写出本题代码：


```typescript
/**
 * 在arr[left, right]中顺序查找最小值
 * @param {Array} arr
 * @param {Number} left
 * @param {Number} right
 */
function orderSearchMin(arr, left, right) {
    let min = arr[left];

    for (let i = left + 1; i <= right; ++i) {
        arr[i] < min && (min = arr[i]);
    }

    return min;
}

/**
 * 在旋转数组arr中用二分法查找最小值
 * @param {Array} arr
 */

function binSearchMin(arr) {
    if (!Array.isArray(arr) || !arr.length) {
        throw Error("Empty Array");
    }

    let left = 0,
        right = arr.length - 1,
        mid = null;

    while (left < right) {
        if (right === 1 + left) {
            return arr[right];
        }

        mid = Math.floor((left + right) / 2);

        if (arr[mid] === arr[left] && arr[mid] === arr[right]) {
            // 无法判断最小值位置
            return orderSearchMin(arr, left, right);
        }

        if (arr[mid] >= arr[left]) {
            // 最小值在右边
            left = mid;
        } else if (arr[mid] <= arr[right]) {
            // 最小值在左边
            right = mid;
        }
    }

    return arr[right];
}

/**
 * 测试代码
 */

console.log(binSearchMin([3, 4, 5, 1, 2]));
console.log(binSearchMin([2, 3, 4, 5, 1]));
console.log(binSearchMin([2, 2, 2, 1, 1, 2]));
console.log(binSearchMin([1]));
```


## 数字在排序数组中出现的次数


### 1. 题目


统计一个数字在排序数组中出现的次数。


### 2. 思路解析


题目说是排序数组，所以可以使用“二分查找”的思想。


一种思路是查找到指定数字，然后向前向后遍历，复杂度是 O(N)。


另一种是不需要遍历所有的数字，只需要找到数字在数组中的左右边界即可，做差即可得到出现次数。


### 3. 代码实现


```typescript
/**
 * 寻找指定数字的左 / 右边界
 *
 * @param {Array} nums
 * @param {*} target
 * @param {String} mode left | right 寻找左 | 右边界
 */
function findBoundary(nums, target, mode) {
    let left = 0,
        right = nums.length - 1;

    while (left < right) {
        let mid = (right + left) >> 1;

        if (nums[mid] > target) {
            right = mid - 1;
        } else if (nums[mid] < target) {
            left = mid + 1;
        } else if (mode === "left") {
            // nums[mid] === target
            // 如果下标是0或者前一个元素不等于target
            // 那么mid就是左边界
            if (mid === 0 || nums[mid - 1] !== target) {
                return mid;
            }
            // 否则，继续在左部分遍历
            right = mid - 1;
        } else if (mode === "right") {
            // nums[mid] === target
            // 如果下标是最后一位 或者 后一个元素不等于target
            // 那么mid就是右边界
            if (mid === nums.length - 1 || nums[mid + 1] !== target) {
                return mid;
            }
            // 否则，继续在右部分遍历
            left = mid + 1;
        }
    }

    // left === right
    if (nums[left] === target) {
        return left;
    }

    return -1;
}

/**
 * 寻找指定数字的出现次数
 *
 * @param {Array} nums
 * @param {*} target
 */
function getTotalTimes(nums, target) {
    const length = nums.length;
    if (!length) {
        return 0;
    }

    return (
        findBoundary(nums, target, "right") -
        findBoundary(nums, target, "left") +
        1
    );
}

/**
 * 以下是测试代码
 */

const nums = [1, 2, 3, 3, 3, 4, 5];
console.log(getTotalTimes(nums, 3));
```


