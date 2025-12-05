---
title: "剑指Offer JavaScript-栈和队列专题"
date: 2019-06-23
permalink: /2019-06-23-stack-queue-exchange/
---
## 用两个栈实现队列


### 1. 题目描述


用两个栈实现一个队列。队列的声明如下:


请实现它的两个函数`appendTail`和`deleteHead`，分别完成在队列尾部插入结点和在队列头部删除结点的功能。


### 2. 解题思路


一个栈用来存储插入队列数据，一个栈用来从队列中取出数据。


从第一个栈向第二个栈转移数据的过程中：数据的性质已经从后入先出变成了先入先出。


### 3. 代码


```typescript
class Queue {
    constructor() {
        this.stack1 = [];
        this.stack2 = [];
    }

    appendTail(value) {
        // 新插入队列的数据都放在 stack1
        this.stack1.splice(0, 0, value);
    }

    deleteHead() {
        // 将要取出的值都从stack2中取
        // 如果stack2为空，那么将 stack1 中的元素都转移过来
        // 此时，stack2中的元素顺序已经被改变了，满足队列的条件
        if (this.stack2.length === 0) {
            let length = this.stack1.length;
            for (let i = 0; i < length; ++i) {
                this.stack2.splice(0, 0, this.stack1.shift());
            }
        }

        return this.stack2.length === 0 ? null : this.stack2.shift();
    }
}

/**
 * 测试代码
 */

let queue = new Queue();
queue.appendTail(1);
queue.appendTail(2);
queue.appendTail(3);

console.log(queue.deleteHead());
queue.appendTail(1);

console.log(queue.deleteHead());
console.log(queue.deleteHead());
console.log(queue.deleteHead());

```


## 包含min函数的栈


### 1. 题目描述


定义栈的数据结构，请在该类型中实现一个能够得到栈的最小元素的 min 函数。在该栈中，调用 min、push 及 pop 的时间复杂度都是 O（1）。


### 2. 思路分析


有关栈的题目，可以考虑使用“辅助栈”，即利用空间换时间的方法。


这道题就是借助“辅助栈”来实现。当有新元素被 push 进普通栈的时候，**程序比较新元素和辅助栈中的原有元素，选出最小的元素，将其放入辅助栈**。


根据栈的特点和操作思路，辅助栈顶的元素就是最小元素。并且辅助栈的元素和普通栈的元素是“一一对应”的。


### 3. 代码实现


```typescript
/**
 * 包含Min函数的栈
 */
class MinStack {
    constructor() {
        this.stack = []; // 数据栈
        this.minStack = []; // 辅助栈
    }

    push(item) {
        const minLength = this.minStack.length;
        this.stack.push(item);

        if (minLength === 0) {
            // 初始情况: 直接放入
            this.minStack.push(item);
        } else {
            if (item < this.minStack[minLength - 1]) {
                // 新元素 ＜ 辅助栈的最小元素: 将新元素放入
                this.minStack.push(item);
            } else {
                // 否则,为了保持2个栈的对应关系，放入辅助栈最小元素
                this.minStack.push(this.minStack[minLength - 1]);
            }
        }
    }

    pop() {
        if (this.stack.length === 0) {
            return null;
        }

        this.stack.pop();
        return this.minStack.pop();
    }

    min() {
        const minLength = this.minStack.length;
        if (minLength === 0) {
            return null;
        }

        return this.minStack[minLength - 1];
    }
}

/**
 * 以下是测试代码
 */

const minStack = new MinStack();

minStack.push(3);
minStack.push(4);
minStack.push(2);
minStack.push(1);
console.log(minStack.minStack, minStack.min()); // output: [ 3, 3, 2, 1 ] 1

minStack.pop();
minStack.pop();
minStack.push(0);
console.log(minStack.minStack, minStack.min()); // output: [ 3, 3, 0 ] 0

```


## 栈的压入弹出序列


### 1. 题目描述


输入两个整数序列，第一个序列表示栈的压入顺序，请判断第二个序列是否为该栈的弹出顺序。假设压入栈的所有数字均不相等。


例如序列 1、2、3、4、5 是某栈的压栈序列，序列 4、5、3、2、1 是该压栈序列对应的一个弹出序列，但 4、3、5、1、2 就不可能是该压栈序列的弹出序列。


### 2. 思路分析


栈的题目还是借助“辅助栈”。大体思路如下：

1. 将入栈序列的元素依次入辅助栈
2. 检查辅助栈顶元素和弹栈序列栈顶元素是否一致：
- 元素一致，弹出辅助栈元素，弹栈序列指针后移
- 不一致，回到第一步

需要注意的是，过程中的边界条件检查（多试试几种情况）。除此之外，由于 js 不提供指针运算，所以用标记下标的方法代替指针。


### 3. 代码实现


```typescript
/**
 * 获得栈顶元素
 * @param {Array} stack
 */
function getStackTop(stack) {
    if (!Array.isArray(stack)) {
        return null;
    }

    if (!stack.length) {
        return null;
    }

    return stack[stack.length - 1];
}

/**
 * 第二个参数是否是该栈的弹出顺序
 * @param {Array} pushOrder
 * @param {Array} popOrder
 * @return {Boolean}
 */
function check(pushOrder, popOrder) {
    if (
        !pushOrder.length ||
        !popOrder.length ||
        pushOrder.length !== popOrder.length
    ) {
        return false;
    }

    const stack = []; // 辅助栈
    let i = 0,
        j = 0; // i: 压入序列指针; j: 弹出序列指针

    while (j < popOrder.length) {
        for (
            ;
            i < pushOrder.length && popOrder[j] !== getStackTop(stack);
            ++i
        ) {
            stack.push(pushOrder[i]);
        }

        if (popOrder[j] !== getStackTop(stack)) {
            return false;
        }

        stack.pop();
        ++j;
    }

    return true;
}

/**
 * 以下是测试代码
 */

console.log(check([1, 2, 3, 4], [4, 3, 2, 1]));

console.log(check([1, 2, 3, 4, 5], [4, 5, 3, 2, 1]));

console.log(check([1, 2, 3, 4, 5], [4, 3, 5, 1, 2]));

```


