---
title: "JS里的基础类型"
url: "2019-03-26-javascript-datatype"
date: 2019-03-26
---

## 原始数据类型和判断方法


> 题目：JS 中的原始数据类型？


ECMAScript 中定义了 7 种原始类型：

- Boolean
- String
- Number
- Null
- Undefined
- Symbol（新定义）
- BigInt（新定义）

**注意**：原始类型不包含 Object 和 Function


> 题目：常用的判断方法？


在进行判断的时候有`typeof`、`instanceof`。对于数组的判断，使用`Array.isArray()`：

- typeof：
	- typeof 基本都可以正确判断数据类型
	- `typeof null`和`typeof [1, 2, 3]`均返回"object"
	- ES6 新增：`typeof Symbol()`返回"symbol"
- instanceof：
	- 专门用于实例和构造函数对应

		```javascript
		function Obj(value) {
		    this.value = value;
		}
		let obj = new Obj("test");
		console.log(obj instanceof Obj); // output: true
		
		```

	- 判断是否是数组：`[1, 2, 3] instanceof Array`
- Array.isArray()：ES6 新增，用来判断是否是'Array'。`Array.isArray({})`返回`false`。

## 原始类型转化


当我们对一个“对象”进行数学运算操作时候，会涉及到对象 => 基础数据类型的转化问题。


事实上，当一个对象执行例如加法操作的时候，如果它是原始类型，那么就不需要转换。否则，将遵循以下规则：

1. 调用实例的`valueOf()`方法，如果有返回的是基础类型，停止下面的过程；否则继续
2. 调用实例的`toString()`方法，如果有返回的是基础类型，停止下面的过程；否则继续
3. 都没返回原始类型，就会报错

请看下面的测试代码：


```javascript
let a = {
    toString: function () {
        return "a";
    },
};

let b = {
    valueOf: function () {
        return 100;
    },
    toString: function () {
        return "b";
    },
};

let c = Object.create(null); // 创建一个空对象

console.log(a + "123"); // output: a123
console.log(b + 1); // output: 101
console.log(c + "123"); // 报错
```


除了`valueOf`和`toString`，es6 还提供了`Symbol.toPrimitive`供对象向原始类型转化，并且**它的优先级最高**！！稍微改造下上面的代码：


```javascript
let b = {
    valueOf: function () {
        return 100;
    },
    toString: function () {
        return "b";
    },
    [Symbol.toPrimitive]: function () {
        return 10000;
    },
};

console.log(b + 1); // output: 10001
```


最后，其实关于`instanceof`判断是否是某个对象的实例，es6 也提供了`Symbol.hasInstance`接口，代码如下：


```javascript
class Even {
    static [Symbol.hasInstance](num) {
        return Number(num) % 2 === 0;
    }
}

const Odd = {
    [Symbol.hasInstance](num) {
        return Number(num) % 2 !== 0;
    },
};

console.log(1 instanceof Even); // output: false
console.log(1 instanceof Odd); // output: true
```


