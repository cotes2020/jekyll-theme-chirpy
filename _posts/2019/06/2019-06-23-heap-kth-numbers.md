---
title: "剑指Offer JavaScript-堆专题"
date: 2019-06-23
permalink: /2019-06-23-heap-kth-numbers/
categories: ["A开源技术课程", "JavaScript版·剑指offer刷题笔记"]
---
## 最小的k个数


### 1. 题目描述


输入 n 个整数，找出其中最小的 k 个数。例如输入 4、5、1、6、2、7、3、8 这 8 个数字，则最小的 4 个数字是 1、2、3、4。


### 2. 思路分析


这里创建一个容量为 k 的最大堆。遍历给定数据集合，每次和堆顶元素进行比较，如果小于堆顶元素，则弹出堆顶元素，然后将当前元素放入堆。


由于堆大小为 k，所以弹出、推入操作复杂度为：O(logK)。因为有 n 个，总体复杂度为：O(nLogK)。


对比快排 partition 的思路，这种思路优点如下：

1. 不会变动原数组
2. 适合处理海量数据，尤其对于不是一次性读取的数据

### 3. 代码实现


请先执行：`yarn add heap` 或者 `npm install heap`


代码如下；


```typescript
const Heap = require("heap");

function compare(a, b) {
    if (a < b) {
        return 1;
    }
    if (a > b) {
        return -1;
    }
    return 0;
}

function getKthNumbers(nums = [], k) {
    if (k <= 0) {
        return null;
    }

    const heap = new Heap(compare);
    for (let num of nums) {
        if (heap.size() < k) {
            heap.push(num);
        } else {
            const top = heap.pop();
            if (num <= top) {
                heap.push(num);
            } else {
                heap.push(top);
            }
        }
    }

    return heap.toArray();
}

/**
 * 以下是测试代码
 */

console.log(getKthNumbers([4, 5, 1, 6, 2, 7, 3, 8], 4)); // output: [ 4, 3, 1, 2 ]
console.log(getKthNumbers([10, 2], 1)); // output: [ 2 ]
```


