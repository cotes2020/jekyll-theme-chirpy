---
title: "剑指Offer JavaScript-位运算专题"
date: 2019-06-23
permalink: /2019-06-23-bit-number-of-one/
categories: ["开源技术课程", "JavaScript版·剑指offer刷题笔记"]
---
## 二进制中1的个数


### 1. 题目


请实现一个函数，输入一个整数，输出该数二进制表示中 1 的个数。例如把 9 表示成二进制是 1001，有 2 位是 1。因此如果输入 9，该函数输出 2。


### 2. 思路


注意到，如果要判断一个二进制数指定位数是否为 1，比如这个二进制数是 1011。那么只需要构造除了这个位为 1，其他位为 0 的二进制即可，这个例子是 0100。


两者进行`&`运算，如果结果为 0，那么指定位数不为 1；否则为 1。


现在事情就简单了，只要准备数字`1`，每次与原数进行`&`操作，然后左移`1`；
重复前面的步骤，就能逐步比较出每一位是不是`1`。


### 3. 代码实现


```typescript
/**
 * @param {Number} n
 */
function numberOf1(n) {
    let count = 0,
        flag = 1;

    while (flag) {
        if (flag & n) {
            ++count;
        }

        flag = flag << 1;
    }

    return count;
}

/**
 * 测试代码
 */

console.log(numberOf1(3));

```


**注意**：有更好的实现思路，请见“02-二进制中 1 的个数进阶版”。


## 二进制中1的个数进阶版


### 1. 优化做法


有个不错的规律，对于一个整数`n`，运算结果`n & (n - 1)`可以消除而今中从右向左出现的第一个`1`。比如二进制数`011`，减去 1 是`010`，做与运算的结果就是`010`。


利用这个性质，可以逐步剔除原数二进制中的`1`。每次剔除，统计量`count`都加 1；直到所有的`1`都被移除，原数变成`0`。


```typescript
/**
 * @param {Number} n
 */
function numberOf1(n) {
    let count = 0;

    while (n) {
        ++count;
        n = n & (n - 1);
    }

    return count;
}

/**
 * 测试代码
 */

console.log(numberOf1(3));

```


### 2. 如何判断 2 的整次方


如果一个数是 2 的整次方，那么只有一个二进制位为 1。所以，`n & (n - 1)`如果不是 1，说明二进制表示中有多个 1，那么就不是 2 的整次方；否则，就是得。


```typescript
/**
 * 判断是否是2的整次方
 * @param {Number} n
 */
function is2Power(n) {
    if (n <= 0) {
        throw new Error("Unvalid param");
    }

    return !(n & (n - 1));
}

console.log(is2Power(128));

```


### 3. 求多少个不同的二进制位


题目：输入两个整数 m 和 n，计算需要改变 m 的二进制表示中的多少位才能得到 n。翻译过来就是：m 和 n 二进制位上有多少个不同的数。


思路：

1. m 和 n 进行异或操作，不同的位都变成了 1
2. 利用前面的思路统计 1 的个数

```typescript
/**
 * 求解二进制表示中有多少位不相同
 * @param {Number} a
 * @param {Number} b
 */
function getDiffBytes(a, b) {
    let count = 0,
        n = a ^ b;

    while (n) {
        ++count;
        n = n & (n - 1);
    }

    return count;
}

/**
 * 测试代码
 */

console.log(getDiffBytes(1, 1));
console.log(getDiffBytes(3, 1));

```


## 数组中只出现一次的数字


### 1. 题目描述


一个整型数组中，除了 2 个数字之外，其他数字都出现了 2 次。要求找出来这 2 个数字，时间复杂度 O(N)，空间复杂度 O(1)


### 2. 思路分析


因为空间复杂度限制，所以没法用哈希表。


如果只有 1 个数字出现 1 次，那么可以使用“异或”运算，最后的结果就是这个数字。


但题目中有 2 个数字，要考虑分组问题。将这两个数字分到 2 组中，然后再每组内分别异或：

1. 全部异或，最终结果是 2 个数字异或结果
2. 找到结果中第一个 1 出现的位数
3. 按照此位是不是 1，将原数据分成 2 组
4. 组内分别异或

### 3. 代码实现


```typescript
/**
 * 找到num二进制表示中第一个1的位
 *
 * @param {Number} num
 */
function findFirstBitIsOne(num) {
    let indexBit = 0,
        flag = 1;
    while (flag && (flag & num) === 0) {
        ++indexBit;
        flag = flag << 1;
    }
    return indexBit;
}

/**
 * 判断num的第index二进制位是否为1
 *
 * @param {Number} num
 * @param {Number} index
 */
function checkIndexBitIsOne(num, index) {
    num = num >> index;
    return !!(num & 1);
}

/**
 * 主函数
 *
 * @param {Array} nums
 */
function findNumsAppearOnce(nums) {
    if (!nums) {
        return null;
    }

    let orResult = 0;
    for (let num of nums) {
        orResult ^= num;
    }

    let indexOfOne = findFirstBitIsOne(orResult);
    let num1 = 0,
        num2 = 0;
    for (let num of nums) {
        if (checkIndexBitIsOne(num, indexOfOne)) {
            num1 ^= num;
        } else {
            num2 ^= num;
        }
    }

    return [num1, num2];
}

/**
 * 测试
 */

console.log(findNumsAppearOnce([2, 4, 3, 6, 3, 2, 5, 5]));

```


### 4. 拓展阅读


在实现的过程中遇到一个好玩的问题：


```shell
$ 1 << 32 # 1

$ 1 << 31 # -2147483648
$ -2147483648 << 1 # 0
```


同样是 1 移动了 32 位，但是结果不同。这是因为在位移操作中，原数和位移数都是 32 位有符号位表示。


为了防止越界，js 会“自作聪明”地帮你把位移数做运算：`shiftNum & 0x1f`。


所以，`1 << 32` 就相当于 `1 << (32 & 0x1f)`，即：`1 << 0`。


参考：[ECMA 官方定义](https://www.ecma-international.org/ecma-262/5.1/#sec-11.7.1)


