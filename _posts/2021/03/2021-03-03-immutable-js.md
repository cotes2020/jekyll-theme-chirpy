---
title: "认识 ImmutableJS 与不可变数据实现原理"
date: 2021-03-03
permalink: /2021-03-03-immutable-js/
categories: ["B源码精读", "Others"]
tags: [不可变数据设计]
---

## 什么是 Immutable


Immutable 代表着不可变数据类型。它一旦被创建，不可被更改。


它不是 js 特有，是一种编程理念。在 js 中，一个常见的落地实现是 immutable.js 库。


## 可变数据类型与不可变数据类型


举个例子，在 js 中，默认都是可变数据类型：


```typescript
const obj = { name: "dongyuanxin", age: 22 };
const obj2 = obj;

obj2.age = 23;
console.log(obj.age)
```


上面代码输出 23。因为 obj2 是 obj 的浅拷贝，那么改变 obj2 上的属性，就会影响 obj 上的属性。


那什么是不可变数据类型呢？假设使用 immutable.js 来实现 Immutable。


```typescript
const { Map } = require("immutable");

const obj = Map({ name: "dongyuanxin", age: 22 });
const obj2 = obj.set("age", 23);

console.log(obj.get("age"));
console.log(obj2.get("age"));
```


上面代码输出 22、23。和上一个例子不同的是，这里的 obj2 不再是 obj 的浅拷贝。


## 不可变数据类型是如何实现的


熟悉 js 的，可能立即可以想到：借助“深拷贝”实现 Immutable。而深拷贝的实现，可以手动实现（经典面试题），也可以使用 lodash 的 deepClone。


**但是，深拷贝实现 Immutable 的缺点是什么呢？** 对于深拷贝，需要遍历所有元素，并且对值进行复制，对性能影响非常大（这就是为什么 js 中默认复杂对象是浅拷贝）。


有没有一种方法，既可以满足不可变的特性，又在实现上能性能最优？**Immutable 的实现原理是：Persistent Data Structure（持久化数据结构）。**


对于 Persistent Data Structure，它实现了 2 个功能：

- 对于旧数据创建新数据，旧数据可读不可写
- 使用“Structural Sharing（结构共享）”，避免复制所有节点

## 结构共享过程


Immutable 实现和深拷贝最大的区别，就是避免复制了所有节点造成的性能损耗。


当改变对象中的某个属性的时候，它只改变这个属性的值和属性的上层属性。反应在多叉树上，就是只改变变化节点以及其上祖先节点，简单来说，就是变化节点到根节点路径上的所有节点。


如下图所示：


![TB1zzi_KXXXXXctXFXXbrb8OVXX-613-575.gif](http://img.alicdn.com/tps/i2/TB1zzi_KXXXXXctXFXXbrb8OVXX-613-575.gif)


可以看到，除了变化节点 => 根节点路径上的涉及节点，其他节点都是无需拷贝，原地复用的。


从图中还可以看到，最终新生成的树（类似修改属性后返回的新对象），和之前的树（旧对象），是有一些节点（属性）是复用的。


## 深入路径复制


假设代码如下：


```typescript
const { Map } = require('immutable')

const map1 = Map({
    obj1: {
        name: 'pony',
        age: 43
    },
    obj2: {
        name: 'vic',
        age: 22
    }
})

const map2 = map1.setIn(['obj2', 'age'], 23)
console.log(map2.get('obj2'))
```


由于更新了obj2.age 的值，因此从age节点向上到根节点的路径上所有节点，都要复制一遍。


其他节点保持不变，不需要浪费时间空间开辟节点，拷贝值。


![Untitled.png](https://raw.githubusercontent.com/dongyuanxin/static/main/blog/imgs/2021-03-03-immutable-js/6a711d4a22861dd11c0bf50520867607.png)


## 其他实现方法（写时复制、胖节点）


[bookmark](https://zh.wikipedia.org/wiki/%E5%8F%AF%E6%8C%81%E4%B9%85%E5%8C%96%E6%95%B0%E6%8D%AE%E7%BB%93%E6%9E%84)


![Untitled.png](https://raw.githubusercontent.com/dongyuanxin/static/main/blog/imgs/2021-03-03-immutable-js/a952ba70a33b81bd0d678bbe9f8bab9d.png)


## 参考链接

- [原理、应用](https://zhuanlan.zhihu.com/p/20295971)：一定要看，涉及到原理图、immutable的优势缺点

