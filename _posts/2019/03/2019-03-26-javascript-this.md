---
title: "ES5和ES6的this"
date: 2019-03-26
permalink: /2019-03-26-javascript-this/
---

### 普通函数和箭头函数的 this


还是一道经典题目，下面的这段代码的输出是什么？（为了方便解释，输出放在了注释中）


```javascript
function fn() {
    console.log(this); // 1. {a: 100}
    var arr = [1, 2, 3];

    (function () {
        console.log(this); // 2. Window
    })();

    // 普通 JS
    arr.map(function (item) {
        console.log(this); // 3. Window
        return item + 1;
    });
    // 箭头函数
    let brr = arr.map((item) => {
        console.log("es6", this); // 4. {a: 100}
        return item + 1;
    });
}
fn.call({ a: 100 });

```


其实诀窍很简单，常见的基本是 3 种情况：es5 普通函数、es6 的箭头函数以及通过`bind`改变过上下文返回的新函数。


① **es5 普通函数**：

- 函数被直接调用，上下文一定是`window`
- 函数作为对象属性被调用，例如：`obj.foo()`，上下文就是对象本身`obj`
- 通过`new`调用，`this`绑定在返回的实例上

② **es6 箭头函数**： 它本身没有`this`，会沿着作用域向上寻找，直到`global` / `window`。请看下面的这段代码：


```javascript
function run() {
    const inner = () => {
        return () => {
            console.log(this.a);
        };
    };

    inner()();
}

run.bind({ a: 1 })(); // Output: 1

```


③ **bind 绑定上下文返回的新函数**：就是被第一个 bind 绑定的上下文，而且 bind 对“箭头函数”无效。请看下面的这段代码：


```javascript
function run() {
    console.log(this.a);
}

run.bind({ a: 1 })(); // output: 1

// 多次bind，上下文由第一个bind的上下文决定
run.bind({ a: 2 }).bind({ a: 1 })(); // output: 2

```


最后，再说说这几种方法的优先级：new > bind > 对象调用 > 直接调用


至此，这道题目的输出就说可以解释明白了。


