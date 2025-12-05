---
title: "剑指Offer JavaScript-字符串专题"
date: 2019-06-23
permalink: /2019-06-23-str-replace-empty/
---
## 替换空格


### 1. 题目描述


请实现一个函数，把字符串中的每个空格替换成"%20"。


例如输入“We are happy.”，则输出“We%20are%20happy.”。


### 2. 解题思路


一种是正则表达式：直接使用正则表达式全局替换，这种方法取巧一些。


另一种是先计算出来替换后的字符串长度，然后逐个填写字符。这种方法的时间复杂度是$O(N)$。


### 3. 代码


```typescript
/**
 * 用正则表达式替换
 * @param {String} str
 */

function repalceEmpty1(str) {
    const re = / /g;
    return str.replace(re, "%20");
}

/**
 * 将空格替换为 %20
 * @param {String} arr
 */
function repalceEmpty2(str) {
    str = str.split("");

    let count = 0,
        i = 0,
        j = 0;
    for (let i = 0; i < str.length; ++i) {
        str[i] === " " && ++count;
    }

    let length = str.length + count * 2; // 新的字符串的长度：%20比空格长度多2
    let result = new Array(length);

    while (i < result.length) {
        if (str[j] === " ") {
            result[i++] = "%";
            result[i++] = "2";
            result[i++] = "0";
            j++;
        } else {
            result[i++] = str[j++];
        }
    }

    return result.join("");
}

/**
 * 测试代码
 */

console.log(repalceEmpty1("We are  happy"));
console.log(repalceEmpty2("We are  happy"));

```


## 字符串全排列


### 1. 题目描述


输入一个字符串，打印出该字符串中字符的所有排列。例如输入字符串 abc，则打印出由字符 a、b、c 所能排列出来的所有字符串 abc、acb、bac、bca、cab 和 cba。


### 2. 思路分析


把集合看成 2 个部分，第一部分是第一个元素，第二部分是后面剩余元素。所有字符都要与当前集合的第一个元素交换，交换后的元素是固定的，也就是一种情况。


每次交换，都继续处理后面剩余元素，它们又可以分成 2 部分，和之前讲述的一样。就这样一直递归下去，直到最后一个元素，那么就排出了其中一种情况。所有情况放在一起，就是全排列的结果。


### 3. 代码实现


```typescript
/**
 * 交换数组指定坐标的2个元素
 * @param {Array} arr
 * @param {Number} i
 * @param {Number} j
 */
function swap(arr, i, j) {
    [arr[i], arr[j]] = [arr[j], arr[i]];
}

/**
 * 检测arr[start, end)中, 是否有和arr[end]相等的元素
 * @param {Array} arr
 * @param {Number} start
 * @param {Number} end
 */
function check(arr, start, end) {
    for (let i = start; i < end; ++i) {
        if (arr[end] === arr[i]) {
            return false;
        }
    }
    return true;
}

/**
 * 全排列
 * @param {Array} arr 元素集合
 * @param {Number} n 起始位置
 */
function perm(arr = [], n = 0) {
    const length = arr.length;
    if (length === n) {
        console.log(arr.join(" "));
        return;
    }

    for (let i = n; i < length; ++i) {
        if (check(arr, n, i)) {
            swap(arr, n, i);
            perm(arr, n + 1);
            swap(arr, n, i);
        }
    }
}

/**
 * 测试代码
 */
perm(["a", "b", "c"], 0);
console.log("*".repeat(10));
perm(["a", "b", "b"], 0);

```


## 翻转单词顺序


### 1. 题目描述


输入一个英文句子，翻转句子中单词的顺序，但单词内字符的顺序不变。


为简单起见，标点符号和普通字母一样处理。例如输入字符串"I am a student."，则输出"student. a am I"。


### 2. 思路分析


进行 2 次不同层次的翻转。第一个层次的翻转，是对整体字符串进行翻转。第二个层次的翻转，是对翻转后字符串中的单词进行翻转。


### 3. 代码实现


**注意**：因为 js 按位重写字符，所以第一次整体字符串翻转后的每个字符，都放入了数组中。


```typescript
/**
 * @param {String} sentence
 */
function reverseSentence(sentence) {
    // 第一次翻转：每个字符
    const chars = sentence.split("").reverse();
    let result = "",
        last = []; // 保存上一个空格到当前空格之间的所有字符

    chars.forEach((ch) => {
        // 遇到空格，说明之前的字符组成了单词
        // 进行第二次翻转：单词
        if (ch === " ") {
            result += last.reverse().join("");
            last.length = 0; // 清空上一个单词
        }

        last.push(ch);
    });

    result += last.reverse().join("");
    return result;
}

/**
 * 测试代码，输出：
 * student.a am I
 */
console.log(reverseSentence("I am a student."));

```


## 实现 `atoi` 函数


### 1. 题目描述


请你来实现一个  atoi  函数，使其能将字符串转换成整数。


首先，该函数会根据需要丢弃无用的开头空格字符，直到寻找到第一个非空格的字符为止。


当我们寻找到的第一个非空字符为正或者负号时，则将该符号与之后面尽可能多的连续数字组合起来，作为该整数的正负号；假如第一个非空字符是数字，则直接将其与之后连续的数字字符组合起来，形成整数。


该字符串除了有效的整数部分之后也可能会存在多余的字符，这些字符可以被忽略，它们对于函数不应该造成影响。


注意：假如该字符串中的第一个非空格字符不是一个有效整数字符、字符串为空或字符串仅包含空白字符时，则你的函数不需要进行转换。


在任何情况下，若函数不能进行有效的转换时，请返回 0。


说明：


假设我们的环境只能存储 32 位大小的有符号整数，那么其数值范围为  [−2^31,  2^31 − 1]。如果数值超过这个范围，qing 返回  INT_MAX (2^31 − 1) 或  INT_MIN (−2^31) 。


题目来自 [LeetCode](https://leetcode-cn.com/problems/string-to-integer-atoi)，可以直接前往这个网址查看题目各种情况下要求的输出。


### 2. 思路分析


这种题目主要就是考察细心，要主动处理所有情况。所以一步步来即可：

1. 找出第一个非空字符，判断是不是符号或者数字
2. 如果是符号，那么判断正负号
3. 如果符号后面跟的不是数字，那么就是非法的，返回 0
4. 确定连续数字字符的起始边界
5. 计算数字字符的代表的数字大小，并且判断是否越界
6. 返回结果的时候注意符号

### 3. 代码实现


代码通过了 leetcode 的测试，成绩还不错，如下图：


![1.png](https://static.godbmw.com/img/2019-06-23-str-atoi/1.png)


代码如下：


```typescript
const MIN_INT_ABS = Math.pow(2, 31);
const MAX_INT = MIN_INT_ABS - 1;

/**
 * 判断char是否是符号
 * @param {String} char
 */
function isSymbol(char) {
    return char === "-" || char === "+";
}

/**
 * 判断char是否是数字
 * @param {String} char
 */
function isNumber(char) {
    return char >= "0" && char <= "9";
}

/**
 * 模拟atoi(str)
 * @param {String} str
 */
function myAtoi(str) {
    const length = str.length;

    // 找出第一个非空字符，判断是不是符号或者数字
    let firstNotEmptyIndex = 0;
    for (
        ;
        firstNotEmptyIndex < length && str[firstNotEmptyIndex] === " ";
        ++firstNotEmptyIndex
    ) {}
    if (
        !isSymbol(str[firstNotEmptyIndex]) &&
        !isNumber(str[firstNotEmptyIndex])
    ) {
        return 0;
    }

    // 如果是符号，那么判断正负号
    let positive = true,
        firstNumberIndex = firstNotEmptyIndex;
    if (isSymbol(str[firstNotEmptyIndex])) {
        positive = str[firstNotEmptyIndex] === "+";
        firstNumberIndex += 1;
    }

    // 如果符号后面跟的不是数字，那么就是非法的，返回0
    if (!isNumber(str[firstNumberIndex])) {
        return 0;
    }

    // 确定连续数字字符的起始边界
    let endNumberIndex = firstNumberIndex;
    while (endNumberIndex < length && isNumber(str[endNumberIndex + 1])) {
        ++endNumberIndex;
    }

    // 计算数字字符的代表的数字大小
    // 并且判断是否越界
    let result = 0;
    for (let i = firstNumberIndex; i <= endNumberIndex; ++i) {
        result = result * 10 + (str[i] - "0");
        if (positive && result > MAX_INT) {
            return MAX_INT;
        }
        if (!positive && result > MIN_INT_ABS) {
            return -1 * MIN_INT_ABS;
        }
    }

    // 返回的时候注意符号
    return positive ? result : -1 * result;
}

/**
 * 以下是测试代码
 */

console.log(myAtoi(" +1.123sfsdfsd")); // 1
console.log(myAtoi(" -42")); // -42
console.log(myAtoi("words and 987")); // 0
console.log(myAtoi("-91283472332")); // -2147483648

```


