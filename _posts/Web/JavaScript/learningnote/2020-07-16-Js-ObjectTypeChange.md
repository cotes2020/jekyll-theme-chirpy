---
title: JavaScript 类型转换
# author: Grace JyL
date: 2020-07-16 11:11:11 -0400
description:
excerpt_separator:
categories: [Web, JavaScriptNote]
tags: [Web, JavaScriptNote]
math: true
# pin: true
toc: true
# image: /assets/img/sample/devices-mockup.png
---

# JavaScript 类型转换

[toc]

## JavaScript 类型

6 种不同的数据类型：
- string
- number
- boolean
- object [array, date]
- function
- symbol

3 种对象类型：
- Object
- Date
- Array

2 个不包含任何值的数据类型：
- null
- undefined


### `typeof`

```js
typeof "John"                 // 返回 string
typeof 3.14                   // 返回 number
typeof NaN                    // 返回 number
typeof false                  // 返回 boolean
typeof [1,2,3,4]              // 返回 object
typeof {name:'John', age:34}  // 返回 object
typeof new Date()             // 返回 object
typeof function () {}         // 返回 function
typeof myCar                  // 返回 undefined (如果 myCar 没有声明)
typeof null                   // 返回 object
```

---

### `constructor`

返回所有 JavaScript 变量的构造函数。

```js
"John".constructor                 // 返回函数 String()  { [native code] }
(3.14).constructor                 // 返回函数 Number()  { [native code] }
false.constructor                  // 返回函数 Boolean() { [native code] }
[1,2,3,4].constructor              // 返回函数 Array()   { [native code] }
{name:'John', age:34}.constructor  // 返回函数 Object()  { [native code] }
new Date().constructor             // 返回函数 Date()    { [native code] }
function () {}.constructor         // 返回函数 Function(){ [native code] }
```

---

## JavaScript 类型转换

1. 转换为字符串


```js
String(100 + 23)       // 将数字表达式转换为字符串并返回
(100 + 23).toString()  // Number 方法 toString()
// Number 方法
toString()
toExponential()	// 把对象的值转换为指数计数法。
toFixed()	    // 把数字转换为字符串，结果的小数点后有指定位数的数字。
toPrecision()	// 把数字格式化为指定的长度。


String(true)         // 将布尔值转换为字符串。
// Boolean 方法 toString() 也有相同的效果。
false.toString()     // 返回 "false"
true.toString()      // 返回 "true"


// Date() 返回字符串。
Date()
```

2. 转换为数字

```js
Number("3.14")    // 返回 3.14
Number(" ")       // 返回 0
Number("")        // 返回 0
Number(false)     // 返回 0
Number(true)      // 返回 1
d = new Date();
Number(d)          // 返回 1404568027739


// 一元运算符 +
将变量转换为数字：
var y = "5";      // y 是一个字符串
var x = + y;      // x 是一个数字
如果变量不能转换，它仍然会是一个数字，但值为 NaN (不是一个数字):
var y = "John";   // y 是一个字符串
var x = + y;      // x 是一个数字 (NaN)
```
