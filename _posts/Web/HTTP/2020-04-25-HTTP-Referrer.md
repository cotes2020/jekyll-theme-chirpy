---
title: HTTP Referer
# author: Grace JyL
date: 2020-04-25 11:11:11 -0400
description:
excerpt_separator:
categories: [Web, HTTP]
tags: [Web, HTTP]
math: true
# pin: true
toc: true
# image: /assets/img/sample/devices-mockup.png
---

- [HTTP Referer](#http-referer)
  - [Referer 的含义](#referer-的含义)
  - [Referer 的发生场景](#referer-的发生场景)
  - [Referer 的作用](#referer-的作用)
  - [`rel`属性](#rel属性)
  - [Referrer Policy 的值](#referrer-policy-的值)
  - [Referrer Policy 的用法](#referrer-policy-的用法)
  - [退出页面重定向](#退出页面重定向)

---


# HTTP Referer

---

HTTP 请求的头信息里面，`Referer` 是一个常见字段，提供访问来源的信息。

## Referer 的含义

引荐人（referrer），谁引荐了你？对于公司来说，这是很有用的信息。
- 互联网也是一样，你不会无缘无故访问一个网页，总是有人告诉你。服务器也想知道，你的"引荐人"是谁？
- HTTP 协议在请求（request）的头信息里面，设计了一个`Referer`字段，给出"引荐网页"的 URL。
- `Referer` logging is used to allow websites and web servers to identify where people are visiting them from, for promotional or statistical purposes.
- The default behaviour of `referer` leaking puts websites at risk of privacy and security breaches.

![bg2019060402](https://i.imgur.com/cmsOpvw.jpg)

这个字段是可选的。客户端发送请求的时候，自主决定是否加上该字段。

Referer的正确拼写是Referrer，但是标准定案只能将错就错。

## Referer 的发生场景
浏览器向服务器请求资源的时候，用户在地址栏输入网址，或者选中浏览器书签，就不发送Referer字段。

主要是以下三种场景，会发送Referer字段。
1. 用户点击网页上的链接。
2. 用户发送表单。
3. 网页加载静态资源，比如加载图片、脚本、样式。

```html
<!-- 加载图片 -->
<img alt="pic" src="foo.jpg">
<!-- 加载脚本 -->
<script src="foo.js"></script>
<!-- 加载样式 -->
<!-- <link href="foo.css" rel="stylesheet"> -->
```

- 浏览器都会将当前网址作为Referer字段，放在 HTTP 请求的头信息发送。
- 浏览器的 `JavaScript` 引擎提供`document.referrer`属性，可以查看当前页面的引荐来源。注意，这里采用的是正确拼写。

## Referer 的作用

`Referer`字段实际上告诉了服务器，用户在访问当前资源之前的位置。可以用来用户跟踪。
- 一个典型的应用是，有些网站不允许图片外链，只有自家的网站才能显示图片，外部网站加载图片就会报错。它的实现就是基于`Referer`字段，如果该字段的网址是自家网址，就放行。

由于涉及隐私，很多时候不适合发送Referer字段。
- 例子，不适合暴露 URL。
- 一个是`功能 URL`，有的 URL 不要登录，可以访问，就能直接完成密码重置、邮件退订等功能。
- 一个是`内网 URL`，不希望外部用户知道内网有这样的地址。`Referer`可能把这些 URL 暴露出去

此外，还有一种特殊情况，需要定制`Referer`字段。
- 比如社交网站上，用户在对话中提到某个网址。这时，不希望暴露用户所在的原始网址，但是可以暴露社交网站的域名，让对方知道，是我贡献了你的流量。

## `rel`属性

浏览器提供一系列手段，允许`改变默认的Referer`行为。
- 用户: 可以改变浏览器本身的全局设置，也可以安装浏览器扩展。
- 开发者: `rel="noreferrer"` 属性是最简单的一种方法。
  - `<a>、<area>和<form>`三个标签可以使用这个属性
  - 一旦使用，该元素就不会发送Referer字段。

```html
<a href="..." rel="noreferrer" target="_blank"> xxx </a>

# 上面链接点击产生的 HTTP 请求，不会带有Referer字段。
# rel="noreferrer"采用的是正确的拼写。
```

## Referrer Policy 的值
`rel`属性只能定制单个元素的`Referer`行为，而且选择比较少，只能发送或不发送。
`W3C` 为此制定了更强大的 Referrer Policy。

Referrer Policy 可以设定8个值。

1. `no-referrer`
不发送Referer字段。

2. `no-referrer-when-downgrade`
如果从 HTTPS 网址链接到 HTTP 网址，不发送Referer字段
其他情况发送（包括 HTTP 网址链接到 HTTP 网址）。
这是浏览器的默认行为。

3. `same-origin`
链接到同源网址（协议+域名+端口 都相同）时发送，否则不发送。
注意，https://foo.com 链接到 https://foo.com 也属于跨域。

4. `origin`
`Referer`字段一律只发送源信息（协议+域名+端口），不管是否跨域。

5. `strict-origin`
如果从 HTTPS 网址链接到 HTTP 网址，不发送Referer字段，其他情况只发送源信息。

6. `origin-when-cross-origin`
同源时，发送完整的`Referer`字段
跨域时，发送源信息。

7. `strict-origin-when-cross-origin`
同源时，发送完整的Referer字段；
跨域时，如果 HTTPS 网址链接到 HTTP 网址，不发送Referer字段，否则发送源信息。

8. `unsafe-url`
`Referer`字段包含源信息、路径和查询字符串，不包含锚点、用户名和密码。

## Referrer Policy 的用法
Referrer Policy 有多种使用方法。

1. `HTTP 头信息` : 服务器发送网页的时候，通过 HTTP 头信息的Referrer-Policy告诉浏览器。
`Referrer-Policy: origin`

2. `<meta>标签` : 使用`<meta>`标签，在网页头部设置。
`<meta name="referrer" content="origin">`

3. `referrerpolicy属性`
`<a>、<area>、<img>、<iframe>和<link>`标签，可以设置referrerpolicy 属性。

`<a href="..." referrerpolicy="origin" target="_blank">xxx</a>`

## 退出页面重定向
还有一种比较老式的技巧，但是非常有效，可以隐藏掉原始网址，谷歌和 Facebook 都在使用这种方法。
- 链接的时候，不要直接跳转，而是通过一个`重定向网址`，就像下面这样。

```html
<a href="/exit.php?url=http%3A%2F%2Fexample.com"> Example.com </a>
上面网址中，先跳转到/exit.php，然后再跳转到目标网址。这时，Referer字段就不会包含原始网址。
```























.
