---
title: "JS里原型、原型链以及instanceof和new实现"
date: 2019-03-27
permalink: /2019-03-27-javascript-second-prototype/
---
## JS 对象的 `__proto__`

- 所有的引用类型（数组、对象、函数），都有一个`__proto__`属性，~~属性值是一个普通的对象~~
- 所有的函数，都有一个 prototype 属性，属性值也是一个普通的对象
- 所有的引用类型（数组、对象、函数），`__proto__`属性值指向它的构造函数的 prototype 属性值

**注**：ES6 的箭头函数没有`prototype`属性，但是有`__proto__`属性。


```javascript
const obj = {};
// 引用类型的 __proto__ 属性值指向它的构造函数的 prototype 属性值
console.log(obj.__proto__ === Object.prototype); // output: true

```


## 原型


> 题目：如何理解 JS 中的原型？


```javascript
// 构造函数
function Foo(name, age) {
    this.name = name;
}
Foo.prototype.alertName = function() {
    alert(this.name);
};
// 创建示例
var f = new Foo("zhangsan");
f.printName = function() {
    console.log(this.name);
};
// 测试
f.printName();
f.alertName();

```


但是执行`alertName`时发生了什么？这里再记住一个重点 **当试图得到一个对象的某个属性时，如果这个对象本身没有这个属性，那么会去它的****`__proto__`****（即它的构造函数的****`prototype`****）中寻找**，因此`f.alertName`就会找到`Foo.prototype.alertName`。


## 原型链


> 题目：如何 JS 中的原型链？


以上一题为基础，如果调用`f.toString()`。

1. `f`试图从`__proto__`中寻找（即`Foo.prototype`），还是没找到`toString()`方法。
2. 继续向上找，从`f.__proto__.__proto__`中寻找（即`Foo.prototype.__proto__`中）。**因为****`Foo.prototype`****就是一个普通对象，因此****`Foo.prototype.__proto__ = Object.prototype`**
3. 最终对应到了`Object.prototype.toString`
这是对深度遍历的过程，寻找的依据就是一个链式结构，所以叫做“原型链”。

## instanceof 实现


`instanceof`是通过原型链来进行判断的，所以只要不断地通过访问`__proto__`，就可以拿到构造函数的原型`prototype`。直到`null`停止。


```javascript
/**
 * 判断left是不是right类型的对象
 * @param {*} left
 * @param {*} right
 * @return {Boolean}
 */
function instanceof2(left, right) {
    let prototype = right.prototype;
    // 沿着left的原型链, 看看是否有何prototype相等的节点
    left = left.__proto__;
    while (1) {
        if (left === null || left === undefined) {
            return false;
        }
        if (left === prototype) {
            return true;
        }
        left = left.__proto__;
    }
}
/**
 * 测试代码
 */
console.log(instanceof2([], Array)); // output: true
function Test() {}
let test = new Test();
console.log(instanceof2(test, Test)); // output: true
```


## new 操作符实现


做之前，得懂：

1. apply的用法
2. `__proto__`和函数`prototype`属性的关系

实现方法：


```typescript
function mynew(Func, ...args) {
    // 1.创建一个新对象
    const obj = {}
    // 2.新对象原型指向构造函数原型对象
    obj.__proto__ = Func.prototype
    // 3.将构建函数的this指向新对象
    let result = Func.apply(obj, args)
    // 4.根据返回值判断
    return result instanceof Object ? result : obj
}
```


