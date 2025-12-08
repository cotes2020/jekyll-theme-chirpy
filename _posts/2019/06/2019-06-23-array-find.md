---
title: "剑指Offer JavaScript-数组专题"
date: 2019-06-23
permalink: /2019-06-23-array-find/
categories: ["开源技术课程", "JavaScript版·剑指offer刷题笔记"]
---
## 二维数组中的查找


### 1. 题目描述


题目：在一个二维数组中，每一行都按照从左到右递增的顺序排序，每一列都按照从上到下递增的顺序排序。请完成一个函数，输入这样的一个二维数组和一个整数，判断数组中是否含有该整数


### 2. 解题思路


时间复杂度是 $O(N)$，空间复杂度是$O(1)$


**利用数组的排序性质**：如果要查找的元素小于当前元素，那么一定不在当前元素左边的列；如果要查找的元素大于当前元素，那么一定在当前元素下面的行。


### 3. 代码


```javascript
/**
 * 题目答案
 * @param {Array} arr
 * @param {Number} elem
 */

function findElem(arr, elem) {
    let row = arr.length - 1,
        col = arr[0].length - 1;
    let i = 0,
        j = col;

    while (i <= row && j >= 0) {
        if (arr[i][j] === elem) {
            return true;
        }

        if (elem > arr[i][j]) {
            ++i;
        } else {
            --j;
        }
    }

    return false;
}

/**
 * 以下是测试代码
 */

const arr = [
    [1, 2, 8, 9],
    [2, 4, 9, 12],
    [4, 7, 10, 13],
    [6, 8, 11, 15],
];

console.log(findElem(arr, 8));
console.log(findElem(arr, 1));
console.log(findElem(arr, 145));

```


## 数组顺序调整


### 1. 题目描述


输入一个整数数组，实现一个函数来调整该数组中数字的顺序，使得所有奇数位于数组的前半部分，所有偶数位于数组的后半部分。


### 2. 思路描述


这题进一步抽象就是满足一定条件的元素都移动到数组的前面，不满足的移动到后面。所以，需要有一个参数用来传递**判断函数**。


最优解法就是数组两头分别有一个指针，然后向中间靠拢。符合条件，就一直向中间移动；不符合条件，就停下来指针，交换两个元素；然后继续移动，直到两个指针相遇。


### 3. 代码实现


函数`change`运用了设计模式中的“[桥接模式](https://dongyuanxin.github.io/2018-12-16-flyweight-pattern/)”，判断条件由用户自己定义。


```javascript
/**
 * 交换数组元素
 * @param {Array} arr
 * @param {Number} i
 * @param {Number} j
 */
const swap = (arr, i, j) => ([arr[i], arr[j]] = [arr[j], arr[i]]);

/**
 * 将符合compareFn要求的数据排在前半部分，不符合要求的排在后半部分
 * @param {Array} brr
 * @param {Function} compareFn
 * @return {Array}
 */
function change(brr, compareFn) {
    const arr = [...brr],
        length = brr.length;
    let i = 0,
        j = arr.length - 1;
    while (i < j) {
        while (i < length && compareFn(arr[i])) ++i;
        while (j >= 0 && !compareFn(arr[j])) --j;

        if (i < j) {
            swap(arr, i, j);
            ++i;
            --j;
        }
    }
    return arr;
}

/**
 * 测试代码
 */

const isOdd = (num) => (num & 1) === 1;
console.log(change([1, 2, 3, 4], isOdd));

```


## 把数组排成最小的数


### 1. 题目描述


输入一个正整数数组，把数组里所有数字拼接起来排成一个数，打印能拼接出的所有数字中最小的一个。例如输入数组{3，32，321}，则打印出这三个数字能排成的最小数字为 321323。


### 2. 思路分析


因为涉及拼接，所以可以将其看做字符串，同时规避了大数溢出的问题，而且字符串的比较规则和数字相同。


借助自定义排序，可以快速比较两个数的大小。比如只看{3, 32}这两个数字。它们可以拼接成 332 和 323，按照题目要求，这里应该取 323。也就是说，此处自定义函数应该返回-1。


### 3. 代码实现


```javascript
/**
 *
 * @param {Array} numbers
 */
function printMinNumber(numbers) {
    numbers.sort((x, y) => {
        const s1 = x + "" + y,
            s2 = y + "" + x;

        if (s1 < s2) return -1;
        if (s1 > s2) return 1;
        return 0;
    });

    console.log(numbers.join(""));
}

/**
 * 测试代码
 */

printMinNumber([3, 32, 321]);

```


## 数组中的逆序对


### 1. 题目描述


输入一个数组,求出这个数组中的逆序对的总数。


例如在数组{7，5，6，4}中，一共存在 5 个逆序对，分别是(7,6), (7, 5), (7,4), (6,4), (5,4)。


### 2. 思路分析


暴力法的时间复杂度是 O(N^2)。利用归并排序的思路，可以将时间复杂度降低到 O(NlogN)。


比如对于 7、5、6、4 来说，会被分成 5、7 和 4、6 两组。


准备两个指针指向两组最后元素，当左边数组指针的对应元素小于右边指针对应元素，结果可以加上从左指针到右指针之间的元素个数（都是逆序的）。


依次移动指针，直到达到边界。


### 3. 代码实现


代码最后输出了数组，经过归并，数组已经是有序的了。


```javascript
/**
 *
 * @param {Array} arr
 * @param {Number} start
 * @param {Number} end
 * @return {Number}
 */
function findInversePairNum(arr, start, end) {
    if (start === end) {
        return 0;
    }

    const copy = new Array(end - start + 1);
    const length = (end - start) >> 1;
    const leftNum = findInversePairNum(arr, start, start + length);
    const rightNum = findInversePairNum(arr, start + length + 1, end);

    let i = start + length, // 左子数组的最后一个下标
        j = end, // 右子数组的最后一个下标
        count = leftNum + rightNum,
        copyIndex = end - start; // copy数组中的最后一个下标

    // 可以参考数据集合：[2, 3, 1, 4]
    for (; i >= start && j >= start + length + 1; ) {
        if (arr[i] > arr[j]) {
            copy[copyIndex--] = arr[i--];
            count += j - start - length;
        } else {
            copy[copyIndex--] = arr[j--];
        }
    }

    for (; i >= start; --i) {
        copy[copyIndex--] = arr[i];
    }

    for (; j >= start + length + 1; --j) {
        copy[copyIndex--] = arr[j];
    }

    // 将排序号的数据放到原数组中
    for (i = 0; i < end - start + 1; ++i) {
        arr[i + start] = copy[i];
    }

    // clear
    copy.length = 0;

    return count;
}

/**
 * 测试代码
 */

const arr = [7, 5, 6, 4];
console.log(findInversePairNum(arr, 0, arr.length - 1)); // output: 5
console.log(arr); // output: [4, 5, 6, 7]

```


