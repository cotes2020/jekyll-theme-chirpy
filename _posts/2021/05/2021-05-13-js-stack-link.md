---
title: "线性栈和链表栈"
url: "2021-05-13-js-stack-link"
date: 2021-05-13
---

## 介绍


如果用数组实现个简单的栈，那么弹出和插入复杂度都是O(N)，因为会涉及到数组的动态扩容。


这里可以使用「线性栈」和「链表栈」:

- **链表栈（推荐）：实现和链表一样**，只是限制了链表操作，并且也没有扩容问题
- **线性栈：利用指针移动来实现，避免自动阔缩数组**。当容量不够，自动阔缩，复杂度是O(N)。

	但是由于读写是O(1),平均复杂度是O(1)。


## 线性栈实现


这里借助了指针来实现线性栈。从而避免自动阔缩数组。


```javascript
class LinearStack {
    constructor(capcity = 16) {
        this.capcity = capcity; // 栈大小
        this.container = new Array(capcity); // 内部容器
        this.count = 0; // 栈中目前的元素个数
    }

    push(data) {
        if (this.count === this.capcity) {
            // 栈满
            return false;
        }
        // 将数据入栈，并且更新指针
        this.container[this.count] = data;
        ++this.count;
        return true;
    }

    pop() {
        if (this.count === 0) {
            return null;
        }

        const data = this.container[this.count - 1];
        --this.count;
        return data;
    }

    /**
     * 动态调节栈大小
     */
    changeCapcity(capcity = 20) {
        if (capcity <= this.capcity) {
            return false;
        }
        this.capcity = capcity;

        const newContainer = new Array(capcity);
        this.container.forEach((data, index) => (newContainer[index] = data));
        this.container = newContainer;
        return true;
    }
}

```


使用效果：


```javascript
const stack = new LinearStack(2);
stack.push(1);
stack.push(2);
stack.push(3);
// 输出：
// [ 1, 2 ]
// 2
// 1
// null
console.log(stack.container);
console.log(stack.pop());
console.log(stack.pop());
console.log(stack.pop());

// 输出：
// [ 1, 2, 3 ]
stack.push(1);
stack.push(2);
stack.changeCapcity(3);
stack.push(3);
console.log(stack.container);

```


