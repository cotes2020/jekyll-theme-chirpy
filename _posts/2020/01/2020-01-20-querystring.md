---
title: "Node.js querystring模块 — URL工具"
date: 2020-01-20
permalink: /2020-01-20-querystring/
---
querystring 是专门用来解析和格式化 URL 的查询字符串 URL 的工具。

- 序列化和解析查询字符串
- 不同语言的兼容处理
- 百分比编码的原理

## 序列化和解析查询字符串


形如`w=%D6%D0%CE%C4&foo=bar`的字符串，就符合查询字符串的格式。querystring 提供了两种 api，一类用于序列化（编码），简单来说就是`json => url查询字符串`；另一类用于解析（解码），简单来说就是 `url查询字符串 => json`。

- 序列化 API：`querystring.encode()` 和 `querystring.stringify()`，两者完全一样
- 解析 API：`querystring.decode()` 和 `querystring.parse()`，两者完全一样

```typescript
const querystring = require("querystring");
const params = {
    foo: "bar",
    baz: ["qux", "quux"],
    corge: ""
};
// output: foo=bar&baz=qux&baz=quux&corge=
console.log(querystring.encode(params));
```


## 不同语言的兼容处理


由于不同的语言中，百分比的编码规则有区别。例如对字符串 `心 谭` 来说:

- 在 java 中，空格换成+号，结果是`%E5%BF%83+%E8%B0%AD`
- 在 js 中，空格换成字节码，结果是`%E5%BF%83%20%E8%B0%AD`

在 querystring.encode() 和 querystring.decode() 接口中，可以使用特殊的百分比编解码函数。


```typescript
const params = {
    key: "原文地址: http://dongyuanxin.github.io/#"
};
querystring.stringify(params, {
    encodeURIComponent: encodeURI // 覆盖默认的百分比编码函数
});
querystring.parse("dongyuanxin.github.io%2F%23", {
    decodeURIComponent: decodeURI // 覆盖默认的百分比解码函数
});
```


## 百分比编码的原理


前面有提到百分比编码，也就是 js 中常用的 `encodeURIComponent` 函数。它的编码规则如下：

- 部分字符不编码：`abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.!~*'()`
- 其他字符：获得 16 进制的字节码（大写），每个字节前添加`%`号

代码如下：


```typescript
/**
 * 将单个字符按编码要求处理成百分比字节码
 * @param {string} ch
 */
function chToHex(ch) {
    const buf = Buffer.from(ch, "utf8");
    let hex = "";
    for (let i = 0; i < buf.byteLength; ++i) {
        hex = hex + "%" + buf[i].toString(16).toLocaleUpperCase();
    }
    return hex;
}
const chs =
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.!~*'()";
/**
 * @param {string} str
 * @return {string}
 */
function myEncodeURIComponent(str) {
    if (str === null || !str.trim().length) {
        return str;
    }
    let ans = "";
    const length = str.length;
    for (let i = 0; i < length; ++i) {
        if (chs.includes(str[i])) {
            ans += str[i];
        } else {
            ans += chToHex(str[i]);
        }
    }
    return ans;
}
```


对于`原文地址：心谭博客`这段字符串的编码结果是：


```text
%E5%8E%9F%E6%96%87%E5%9C%B0%E5%9D%80%EF%BC%9A%E5%BF%83%E8%B0%AD%E5%8D%9A%E5%AE%A2
```


## 参考链接

- [Nodejs v12 文档](http://nodejs.cn/api/querystring.html)
- [java 代码实现 encodeURIComponent 和 decodeURIComponent，解决空格转义为加号的问题。](https://blog.csdn.net/KokJuis/article/details/84140514)

