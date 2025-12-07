---
title: "《JavaScript高级程序设计》读书笔记"
date: "2019-05-28"
permalink: /2019-05-28-professional-js-notes/
---

## 第一章 JavaScript 简介

### 1.2 js 实现

1. 一个完成的 js 实现由 3 个不同部分组成：核心（ECMAScript）、文档对象模型（DOM）、浏览器对象模型（BOM）
2. 常见的 Web 浏览器知识 ECMAScript 实现的**宿主环境**之一，其他环境包括 Node、Adobe Flash
3. DOM：是针对 XML 经过扩展用于 HTML 的程序编程 API
4. BOM：控制浏览器显示的页面以外部分

## 第二章 在 HTML 中使用 JavaScript

### 2.1 `<script>`标签

下面两个属性可以控制 script 加载，它们不能严格保证执行顺序：

1. async：不阻塞页面，下载并且执行脚本
2. defer：脚本延迟到文档被完全解析和显示后再执行。

script 脚本中不要嵌入出现`"</script>"` 字符串，会被错误识别为结束标签。正确写法是：`"<\/script>"`。

如果 script 标签中既有代码内容，并且也引入了外部脚本（src 属性）。浏览器只会执行外部脚本。

### 2.2 可扩展超文本标记语言 XHTML

XHTML 编写比 HTML 更严格，例如 `>` 等符号都需要转义。为了保证 js 正常运行，用 `CDATA` 来包裹。

下面代码在不兼容 xml 的浏览器可以平稳退化：

```html
<script type="text/javascript">
  //<![CDATA[
  function compare(a, b) {
    if (a < b) {
      console.log("a is less than b");
    }
  }
  //]]>
</script>
```

### 2.4 `<noscript>` 元素

对于不支持 js 的浏览器，此标签可平稳退化。

```html
<body>
  <noscript>
    <p>请启用JavaScript</p>
  </noscript>
</body>
```

## 第三章 基本概念

### 3.1 语法

针对 ES3 的不确定行为，ES5 增加了严格模式，它是“编译指示”，用来告知 Js 引擎切换到严格模式，需要在代码顶部添加：`"use strict";`

### 3.2 关键字和保留字

保留字是之后可能被用作关键字的标识符。比如`super`，在 es6 中被用在了子类的构造函数中。

### 3.3 变量

`var` 声明的变量存在声明提升，如下：

```javascript
var a = 1;

function test() {
  console.log(a);
  var a = 2;
  a = 3;
}

test();
```

由于变量声明提升以及函数作用域，相当于以下代码：

```javascript
var a = 1;

function test() {
  var a = undefined;
  console.log(a);
  a = 2;
  a = 3;
}

test();
```

因此，输出结果是 `undefined`

### 3.4 类型

#### 3.4.1 null 和 undefined

`null` 和 `undefined` 不相同，区别如下：

- null：空对象指针，`typeof null` 返回 `"object"`，常用于定义空变量
- undefined：未定义，变量只声明时，默认赋值`undefined`

#### 3.4.2 8 和 16 进制

`number`类型：

- 8 进制：0 开头，例如 070
- 16 进制：0x 开头，例如 0x1f
- 科学计数法：1ex，例如 1e2 = 100

所有 8 和 16 进制值在运算时，都会被转化为 10 进制。

#### 3.4.3 特殊数字

`Number.MIN_VALUE` 和 `Number.MAX_VALUE` 分别返回最小值和最大值。超出范围的会被转化为 `Infinity` 。

不合法的数，比如 1/0 ，会返回 `NaN`，需要用 `isNaN` 判断。对于对象，`isNaN` 先调用 `valueOf` ，再掉用 `toString` 。

```javascript
const validNum = {
  valueOf: function() {
    return 1;
  },
  toString: function() {
    return "str";
  }
};

console.log(isNaN(validNum)); // output: false
```

#### 3.4.4 字符串转数字

`parseInt` 应该在第二个参数指明进制。

#### 3.4.5 字符串

字符串变量的值是不可变的，当改变值时，会销毁之前的字符串，然后用包含新值的字符串填充变量。

调用数值的 `toString` 方法，给定参数代表进制。

特殊编码：

- `\xnn`：以 16 进制代码 nn 表示字符
- `\unnnn`：以 16 进制代码 nnnn 表示 Unicode 字符

```javascript
console.log("\x41"); // A
console.log("\u03a3"); // Σ
```

#### 3.4.6 Object 类型

Object 实例都有以下属性：

- constructor: 指向创建对象的函数
- hasOwnProperty
- obj1.isPrototypeOf(obj2): obj1 是不是在 obj2 的原型链上
- propertyIsEnumerable(propName): propName 能否用 for-in 枚举

关于 `isPrototypeOf`:

```javascript
function Demo() {}
var o = {};

var demo = new Demo();
console.log(o.isPrototypeOf(demo)); // output: false

// 将o放在demo实例的原型链上
demo.__proto__ = Demo.prototype = o;
console.log(o.isPrototypeOf(demo)); // output: true
```

对于 BOM、DOM 等宿主环境提供的对象，可能并不继承 Object，不具有以上通性。

### 3.5 操作符

1、**位操作**

- `~`: 按位非。`~110 => 001`
- `&`: 按位与。
- `|`: 按位或。
- `^`: 异或操作。位数相同返回 0，不同返回 1。
- `<<`: 左移
- `>>`: 默认情况，有符号右移，保留符号位（符合正常逻辑）
- `>>>`: 无符号右移，在移动时候忽略符号位。

正因为移动时候忽略符号位，因此例如 -64 = 111111..11100000，负数的补码会被当做正数的二进制码。

2、**布尔操作**

一般直接使用 `!!` 进行转化。

3、**逗号操作符**

```javascript
var num1 = 1,
  num2 = 2,
  num3 = 3; // 多变量声明

var num = (3, 2, 1); // 从右边开始解析，返回 1
console.log(num); // output: 1
```

### 3.6 语句

#### 3.6.5 for-in 语句

精准迭代，枚举对象属性。但是效率很低，而且输出的属性名的顺序不确定。

在执行前，需要检测对象是否为 `null` 或者 `undefined`，否则 es3 会报错。

#### 3.6.6 label 语句

与`break` 和 `continue` 联合使用，主要用于多层嵌套循环的流程控制。

配合 `break`，直接跳出指定的 `label` ：

```javascript
var num = 0;
outermost: for (var i = 0; i < 10; ++i) {
  for (var j = 0; j < 10; ++j) {
    if (i === 5 && j === 5) {
      // i, j为5的时候，结束循环
      break outermost;
    }
    ++num;
  }
}
console.log(num); // 55
```

配合 `continue`，直接跳出指定的 `label` ：

```javascript
var num = 0;
outermost: for (var i = 0; i < 10; ++i) {
  for (var j = 0; j < 10; ++j) {
    if (i === 5 && j === 5) {
      continue outermost;
    }
    ++num;
  }
}
console.log(num); // 95
```

开启调试后会发现，当 i 和 j 为 5 的时候，跳到了 outermost，并且保持了 i 和 j 的变量值。

外层循环导致 i 变为 6，j 清零。

#### 3.6.8 with 语句

设置代码作用域到指定对象中，会导致性能下降。

```javascript
const obj = {
  a: 1
};

with (obj) {
  console.log(a); // 1
}
```

### 3.7 函数

`arguments` 是类数组对象，严格模式下不能重写或者重新定义其中的值。

`arguments.callee` 指向函数自身，用于编写递归函数。

**注意**：js 的函数没有重载。ts 可以重载，但是也只是多类型声明，不符合传统意义的函数重载。

## 第四章 变量、作用域和内存问题

### 4.1 基本类型和引用类型的值

#### 4.1.2 赋值

复制函数：

```javascript
var obj1 = new Object(); // obj1 保存的是副本，不过这个副本是指向实例的一个指针
var obj2 = obj1;
```

ECMAScript 中所有函数的参数都是按值传递，对于复杂类型，副本就是指向它的指针。

#### 4.1.4 检测类型

基本数据类型：`typeof`；对象类型检测：`instanceof`

### 4.2 执行环境和作用域

延长作用域链的情景：

1. `try-catch`中的`catch`：作用域链前端新增错误对象
2. `with`：作用域链前端新增指定对象
3. 函数闭包

### 4.3 垃圾回收(GC)

#### 4.3.1 标记清除和引用计数

浏览器的实现有两种：

1. **标记清除**：所有变量打标记；去掉环境中变量的标记，以及被环境中变量引用变量的标记；之后，清除还有标记的变量。
2. **引用计数**：跟踪每个变量引用次数，被引用的变量就加 1；如果此变量又取了另一个变量，减 1。

```javascript
const value = 1; // 引用0
const copy = value; // 引用+1
const obj = {
  copy // 引用 + 1
};
obj.copy = null; // 引用 -1
// 最后，引用次数为1
```

引用计数无法处理“循环引用”的情况，例如：

```javascript
function problem() {
  const obja = {},
    objb = {};

  obja.prop = objb; // objb的引用次数和obja的引用次数都+1
  objb.prop = obja; // objb的引用次数和obja的引用次数再+1
  // obja 和 obj2 的引用次数均是2
  // 变量永远不会被清除，造成内存泄漏
}
```

#### 4.3.3 性能优化

在**优化性能问题**上，IE6 根据固定的内存分配量来触发 gc。但是如果脚本中声明了很多变量，并且都没有被释放，那么一直会达到触发标准，gc 会高频率触发，效率低下。

es7 做出了改进：临界值是动态计算的。如果一次垃圾回收的内存量低于 15%，那么临界值会翻倍；如果高于 85%，重置临界值。

#### 4.3.4 管理内存

解除引用：不使用的变量，设置为`null`。

解除引用不意味变量内存回收，而是让其脱离执行环境，方便下次 gc 回收。

## 5. 引用类型

ECMAScript 是面向对象语言，但不是传统的面向对象。提供构造函数，专门对接传统对象编程。

### 5.1 Object 类型

`new Object()` 和 `{}` 声明等效。

### 5.2 Array 类型

创建有`Array`和 `[]`2 种方式。

`length` 是可读写的，置 0 可以清空数组。

#### 5.2.1 数组检测

请用 `Array.isArray` 检测数组。`instanceof` 不适用于网页包含多个框架，2 个运行环境，从一个向另一个传入数组构造函数，严格意义上并不相等。

```html
<script>
  const { frames } = window;
  const length = frames.length;
  xArray = frames[length - 1].Array;
  const arr = new Array();
  console.log(arr instanceof xArray); // false
</script>
```

#### 5.2.3 栈和队列

- 栈：`push` && `pop`
- 队列：`push` && `shift`

#### 5.2.6 操作方法

concat：参数会被自动展开

```javascript
const colors = [1];
const colors2 = colors.concat(2, [3, 4]); // [1, 2, 3, 4]
```

slice(star, end): 切片，返回新数组。

splice(start, count, ...items):

- 删除：不需要第三个参数
- 插入：第二参数置 0
- 替换：第二个和第三个参数要用

### 5.3 Date 类型

Date.now() 和 new Date().gewNow() 等价。

Date.parse(string): 返回 string 代表的日期的毫秒数。`年/月/日`，请不要使用`-`连接！

Date 实例可以直接比较大小，因为`valueOf`返回毫秒数。

### 5.4 RegExp 类型

不推荐 `new RegExp(string)` 来声明正则，因为 string 是字符串，元字符需要双重转义。比如`\n`，就是`\\n`。

每个实例拥有以下属性：

- global：g
- ignoreCase: i
- multiline: m
- **lastIndex**: 搜索下一匹配项的字符位置
- **source**: 正则的字符串表示

### 5.5 Function 类型

代码求值时，js 引擎会将声明函数提升到源码顶部。

`arguments`上重要属性：

- length：参数长度
- callee: 函数自身引用

函数上重要属性：

- caller: 调用此函数的函数引用。全局访问返回 null
- length：函数希望接受的参数个数（不算默认参数）

```javascript
function outer() {
  inner();
}
function inner(a, b = 1) {
  console.log(arguments.callee.caller === outer);
}

outer(); // true
inner.length; // 2 - 1 = 1
```

函数 prototype 属性无法枚举，不能用 for-in 枚举

- 可以使用 `Object.getOwnPropertyNames` ，返回一个由指定对象的所有自身属性的属性名（包括不可枚举属性但不包括 Symbol 值作为名称的属性）组成的数组。
- 可以使用 `Reflect.ownKeys`，返回包括所有自身属性的属性名的数组

### 5.6 基本包装类型

num.toFixed(位数)：自动舍入，返回字符串。

num.toExponential(位数)：转化为科学计数法，返回字符串。

String.fromCharCode(...charcodes): 将字符编码转化为字符串。

String.charCodeAt(index): 将 index 的字符转化为字符编码。

### 5.7 单体内置对象

随机整数生成：

```javascript
// [start, end]
function randomInt(start, end) {
  const times = end - start + 1;
  return Math.floor(Math.random() * times + start);
}
```

## 第六章 面向对象的程序设计

ECMA-62 对象定义：无序属性集合，其属性可以包括基本值、对象和函数。

### 6.1 理解对象

ECMA 有 2 种属性：数据属性和访问器属性。它们可以通过 `Object.getOwnPropertyDescriptor` 来读取。

**1.数据属性**

通过 `Object.defineProperty(对象, 属性名, {属性: 值})` 来修改，可修改的属性是：configurable(是否可通过`delete`删除)、enumerable(能否 for-in 循环)、writable(能否修改)、value。

可以多次调用 api 修改上述属性，除了将 `configurable` 设置为 false。

**2.访问器属性**

访问器属性不包含数据值，也是通过 `Object.defineProperty(对象, 属性名, {属性: 值})` 来修改。

可修改的属性是：configurable、enumerable、get、set。其中，只指定 get 不指定 set，那么就是不可写；反过来，不能读。

### 6.2 创建对象

#### 6.2.1 理解原型对象

原型模式中，实例的 `__proto__` 指向构造函数的 `prototype`，因此，`构造函数.prototype.isPrototypeOf(实例)`返回 true。

因为原型链有下端“屏蔽”上端的机制，可以通过逐步 `delete` 来暴露上端属性。

#### 6.2.2 原型与 `in` 操作符

如果对象可以访问给定属性，那么 `in` 返回 true。

```javascript
function Person() {}
Person.prototype.name = "student";
const person = new Person();
console.log("name" in person); // output
```

检测 `prototype` 是否位于 原型链上，而不位于实例上。

```javascript
function hasPropertyInPrototype(object, prototype) {
  // hasOwnProperty 是否位于实例上
  return prototype in object && !object.hasOwnProperty(prototype);
}
```

#### 6.2.3 自定义原型

(构造)函数的`constructor` 属性是自身，所以重写`prototype`的时候，需要注意：

```javascript
function Person() {}
Person.prototype = {
  name: "dongyuanxin"
};
Object.defineProperty(Person.prototype, "constructor", {
  enumerable: false, // Person.prototype.constructor 是不可枚举的
  value: Person
});
```

#### 6.2.4 动态原型

为了对应 OO 编程习惯，prototype 上属性在访问时动态创建：

```javascript
function Person() {
  this.name = "person";

  if (typeof this.sayHello !== "function") {
    Person.prototype.sayHello = function() {
      console.log(`Hello, I'm ${this.name}`);
    };
  }
}
```

#### 6.2.6 稳妥构造函数

经常使用，尤其是在对原生对象做拓展时候，而且不能影响原有原型链。

```javascript
function PowerDate() {
  const date = new Date();
  date.format = () => {
    const year = addZeroStr(date.getFullYear()),
      month = addZeroStr(date.getMonth() + 1),
      day = addZeroStr(date.getDate()),
      hour = addZeroStr(date.getHours()),
      minute = addZeroStr(date.getMinutes()),
      second = addZeroStr(date.getSeconds());
    return `${year}/${month}/${day} ${hour}:${minute}:${second}`;
  };
  // 可以new调用，因为return重置了返回值
  return date;
}
```

### 6.3 继承

- 接口继承：继承方法签名
- 实现继承：继承实际方法

常见四种方法：[JavaScript 基础知识梳理-下](https://github.com/dongyuanxin/blog/blob/master/%E5%89%8D%E7%AB%AF%E7%9F%A5%E8%AF%86%E4%BD%93%E7%B3%BB/js/JavaScript%E5%9F%BA%E7%A1%80%E7%9F%A5%E8%AF%86%E6%A2%B3%E7%90%86-%E4%B8%8B.md)

## 第七章 函数表达式

### 7.2 闭包

闭包是指：有权访问另一个函数作用域中的变量的函数。作用域得到了延长。

一个经典问题：

```javascript
function createFunction() {
  var result = new Array();

  for (var i = 0; i < 10; ++i) {
    result[i] = function() {
      return i;
    };
  }

  return result;
}
```

调用 result 中的函数，返回值均是 10。这是因为 `var` 不是块级作用域，闭包声明造成了内函数可以访问 `createFunction` 的作用域，并且在结束函数后，变量`i`的生命被延长了下来。例如，当调用 `result[0]` 的时候，就会访问并且返回 `createFunction` 中的 变量`i`的值。

如果将 `var` 换成 `let`，则不存在这个问题。虽然变量`i`生命被延长，也属于 `createFunction`作用域，但是`let`本身是“**块级作用域**”。也就是说，闭包中返回的`i`是当前循环下的`i`，没有发生污染。

### 7.3 模仿块级作用域

下面写法内存占用低，标记清除的`gc`在函数运行完，检测到不被使用，会立即销毁作用域链。

```javascript
(function() {
  // ...
})();
```

### 7.4 私有变量

利用闭包，可以很巧妙地实现静态私有变量、私有函数方法等。

```javascript
(function() {
  var name = ""; // 静态私有变量

  return {
    name() {
      return name + "123";
    }
  };
})();
```

## 第八章 BOM

### 8.1 window 对象

双重角色：js 访问浏览器的 api + ECMAScript 规定的 global 对象。

#### 8.1.1 全局作用域

定义在全局的变量不能被 delete, 定义在 window 上的属性可以被 delete。

#### 8.1.2 窗口关系及框架

对于 window 的`frames`，为了保证兼容性，请使用：`top.frames`。因为`top`是绝对的。

除了`top`外，还有`parent`，在没有任何框架情况下，`top === window`。

最后，还有`self`。在 sw 中，常用 self 访问 window 上的 api。

#### 8.1.3 窗口位置

跨浏览器取得窗口左边、上边的位置：

```javascript
let leftPos =
  typeof window.screenLeft === "number" ? window.screenLeft : window.screenX;
let topPos =
  typeof window.screenTop === "number" ? window.screenTop : window.screenY;
```

此外，还有`window.moveTo(x, y)` 和 `window.moveBy(offsetX, offsetY)`两个方法移动位置。但是默认是禁用的。

#### 8.1.4 窗口大小

窗口大小无法确定，但是可以跨浏览器获得页面视图大小：

```javascript
let pageWidth = window.innerWidth,
  pageHeight = window.innerHeight;

if (typeof pageWidth !== "number") {
  if (document.compatMode === "CSS1Compat") {
    // 是否是标准模式
    pageWidth = document.documentElement.clientWidth;
    pageHeight = document.documentElement.clientHeight;
  } else {
    // 是否是混杂模式
    pageWidth = document.body.clientWidth;
    pageHeight = document.body.clientHeight;
  }
}
```

此外，还有`window.resizeTo(width, height)` 和 `window.resize(offsetWidth, offsetHeight)`调整大小。但是默认是禁用的。

#### 8.1.5 导航和打开窗口

`window.open(href, windowName, paramsString)`: 最后一个参数形如 `height=400,width=10`。

这里有同域限制，并且返回的指针指向新开窗口，可以使用以上被禁用的方法。

对于一些浏览器插件，会禁用弹出，兼容代码如下：

```javascript
let blocked = false;
try {
  let wroxWin = window.open("http://baidu.com", "_blank");
  if (!wroxWin) {
    // 打开失败
    blocked = true;
  }
} catch (error) {
  // 插件禁止后，会报错
  blocked = true;
}
```

#### 8.1.7 系统对话框

它们是浏览器决定的，是同步和模态的。显示的时候，会终止代码执行。

### 8.2 location 对象

location.href(最常用) 和 window.location 本质都是调用 location.assign()。

除此之外，修改 location 上的其他属性，也可以改变当前加载的页面，比如 `location.hash='#setion'`

以上方法，会在浏览器中生成新的历史记录。使用`location.replace()`方法，不会在浏览器中生成历史记录。

location.reload(true)：强制重新加载。

### 8.3 navigator 对象

#### 8.3.1 检测插件

`navigator.plugins` 存放插件信息：

```javascript
// 通用检测方法
function hasPlugin(name = "") {
  name = name.toLocaleLowerCase();
  for (var i = 0; i < navigator.plugins.length; ++i) {
    if (navigator.plugins[i].name.toLocaleLowerCase().indexOf(name) > -1) {
      return true;
    }
  }
  return false;
}
```

但由于 IE 浏览器的兼容，最好针对不同浏览器封装不同的插件检测方法。

#### 8.3.2 注册处理程序

google 支持 register​Protocol​Handler 自定义协议。比如打开`https://www.baidu.com`的控制台，在其中输入：

```javascript
// 理论上是这样，但是效果不好
navigator.registerProtocolHandler(
  "web+baidu",
  "https://www.baidu.com/s?wd=%s",
  "Baidu handler"
);
```

### 8.5 history 对象

history.go(): 任意跳转。数字代表前后跳转，字符串会自动找寻历史中最近的位置跳转。

history.length: 保存历史记录的数量。
