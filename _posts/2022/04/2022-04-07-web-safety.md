---
title: "常见 Web 攻击手段和防范策略"
date: 2022-04-07
permalink: /2022-04-07-web-safety/
tags: [安全设计]
---
## CSRF：跨站请求伪造


### 攻击流程


一个典型的 CSRF 攻击有着如下的流程：

- 受害者登录 [a.com](http://a.com/)，并保留了登录凭证（Cookie）。
- 攻击者引诱受害者访问了 [b.com](http://b.com/)。
- [b.com](http://b.com/) 向 [a.com](http://a.com/) 发送了一个请求：[a.com/act=xx。浏览器会默认携带](http://a.com/act=xx%E3%80%82%E6%B5%8F%E8%A7%88%E5%99%A8%E4%BC%9A%E9%BB%98%E8%AE%A4%E6%90%BA%E5%B8%A6) [a.com](http://a.com/) 的 Cookie。
- [a.com](http://a.com/) 接收到请求后，对请求进行验证，并确认是受害者的凭证，误以为是受害者自己发送的请求。
- [a.com](http://a.com/) 以受害者的名义执行了 act=xx。
- 攻击完成，攻击者在受害者不知情的情况下，冒充受害者，让 [a.com](http://a.com/) 执行了自己定义的操作。

下面这张图就展示了 CSRF 跨站请求伪造的过程：


![007S8ZIlgy1gixko2nir0j30u10ddmzn.jpg](https://raw.githubusercontent.com/dongyuanxin/static/main/blog/imgs/2022-04-07-web-safety/007S8ZIlgy1gixko2nir0j30u10ddmzn.jpg)


### 攻击分析


可以看到，csrf 能成功的原因在于：借助浏览器 cookie 的机制，非法使用了用户的 cookie。


特点如下：

- 发生在第三方域名
- 攻击者只能使用cookie，不能获取/更改cookie

### 防范策略

1. 禁用或限制跨域请求

在服务端，采用「白名单」的思路，返回的`Access-Control-Allow-Origin` 字段只包括合法的域名。

1. 对cookie开启samesite。新版 Chrome 默认开启。
2. 表单中新增cookies hashing，服务端进行校验。

在表单中新增伪随机数的demo：


```html
<form method="POST" action="transfer.php">
　　<input type="text" name="toBankId">
　　<input type="text" name="money">
　　<input type="hidden" name="hash" value="这里是前端表单提交前，构造加密的cookie信息">
　　<input type="submit" name="submit" value="Submit">
</form>
```


服务端校验时，会检查表单提交的字段是否和加密cookie中的字段信息一样。

1. token 验证。

客户端提交表单的时候，应该带上token。服务端会检查token是否合法、是否过期。


如何下发token呢？一个是在提交前，请求后端接口获取合法token；另一个是在渲染页面的时候（SSR），将token从session中读取出来，并且将其放到页面的表单中。第一种方法更好，服务端是无状态的，方便横向扩缩。

1. 验证码。

有很多验证码的生成方式，比如前端绘制验证码图片，并且将验证码的正确信息放到cookie中，进行表单提交。后端校验提交表单中用户输入的验证码与cookie中的验证码是否匹配。


还有一种是目前比较常用的手机验证码。


## XSS：跨站脚本攻击


Cross Site Scrit 跨站脚本攻击（为与 CSS 区别，所以在安全领域叫 XSS）。


### 攻击原理


代码被恶意注入到页面中（例如评论），然后其他用户在访问本站页面时，浏览器执行了代码逻辑。


> 这个和 CSRF 相比，攻击的发起是在本站点，并且通过脚本，攻击者能操作用户的各种信息。危害更大。


### 防范策略

- 浏览器自带 X-XSS-Protection
	- 介绍：为 0 是禁止 xss 过滤，为 1 是启用 xss 过滤。
	- 缺点：兼容性、不能完全杜绝 xss
- 特殊字符转义
	- 介绍：例如`<`变为`&lt;`
- 标签过滤
	- 介绍：采用白名单机制，仅对安全 html 标签进行渲染，其他不给渲染
- 内容安全策略（CSP）
	- 介绍：[Content Security Policy 入门教程](http://www.ruanyifeng.com/blog/2016/09/csp.html)

## SQL 注入攻击


### 攻击原理


例如做一个系统的登录界面，输入用户名和密码，提交之后，后端直接拿到数据就拼接 SQL 语句去查询数据库。如果在输入时进行了恶意的 SQL 拼装，那么最后生成的 SQL 就会有问题。


比如后端拼接的 SQL 字符串是：


```sql
SELECT * FROM user WHERE username = 'user' AND password = 'pwd';
```


如果不做任何防护，直接拼接前端的字符，就会出现问题。比如前端传来的`user`字段是以`'#`结尾，`password`随意：


```sql
SELECT * FROM user WHERE username = 'user'#'AND password = 'pwd';
```


**密码验证部分直接被注释掉了**。


### 防范策略

- 特殊字符转义
- 借助成熟的 ORM 库，避免直接拼接 SQL 语句

## DDoS


### 攻击原理


攻击者在短时间内发起大量请求，利用协议的缺点，耗尽服务器的资源，导致网站无法响应正常的访问。


### 防范策略

1. 借助云厂商 CDN：静态流量的资源还得自己掏钱
2. IP 黑/白名单：`nginx` 和 `apache` 都可以设置
3. HTTP 请求信息：根据 UserAgent 等字段的信息
4. 降级处理：阮一峰老师的网站被 ddos 的时候就有个备份页面
5. 限频限流：云厂商都提供阈值设置
6. 缓存优化：针对接口
7. 其他：弹性 ip、免费的 DNSpod、国内外分流、高防 ip 等等

> 和其他攻击不同，这个最难防范，需要做好「降级」。因为难以区分正常流量和攻击流量，Github照样被打挂。


## 中间人攻击


### 攻击原理


它也被称为浏览器劫持、web 劫持。中间人（攻击者）可以往网站添加一些第三方厂商的 dom 元素，或者重定向到另外的钓鱼站。


### 防范策略

- 切换成 `https` 协议
- 采用更安全的 SSL 证书

## SSRF：服务器端请求伪造


SSRF(Server-Side Request Forgery:服务器端请求伪造) 是一种由攻击者构造形成由服务端发起请求的一个安全漏洞。


### 攻击原理


正常来说，由于网络分区、防火墙等手段，攻击者无法攻击内网。但是某些对外的接口可能有执行危险行为，例如探测服务、执行脚本等。攻击者可以通过调用这些接口，通过调用接口，以内网服务的身份实施攻击。


假设当前提供一个接口，支持用户绑定邮箱。接口会在绑定之前发送邮件判断 smtp 服务是否可用、账号密码是否正确。例如国外云厂商的一些邮箱服务。作为攻击者可能调用此接口，以接口提供者的身份和服务能力，对 smtp 服务发起探测攻击。


### 防范策略


不断完善代码，发现问题即时修复。没啥好办法。


## 点击劫持


### 攻击原理


点击劫持是一种视觉欺骗的攻击手段。攻击者通过 `iframe` 嵌套嵌入被攻击网页，诱导用户点击。如果用户之前登陆过被攻击网页，那么浏览器可能保存了信息，因此可以以用户的身份实现操作。


![e6c9d24egy1h11mcijjtyj20hf0diwfr.jpg](https://raw.githubusercontent.com/dongyuanxin/static/main/blog/imgs/2022-04-07-web-safety/e6c9d24egy1h11mcijjtyj20hf0diwfr.jpg)


上面这个是一个简单的例子。攻击页面包括两层，用户看到的，以为自己正在操作的实际上是在下层，真正操作的是上面的透明的一层。利用的是css的`z-index`和`opacity`属性。


### 防范策略

- 设置 `X-FRAME-OPTIONS` 响应头
	- DENY，表示页面不允许通过 iframe 的方式展示
	- SAMEORIGIN，表示页面可以在相同域名下通过 iframe 的方式展示
	- ALLOW-FROM，表示页面可以在指定来源的 iframe 中展示
- JS 代码防御。相较于方法1，兼容性更好。

```html
<!-- 当攻击者通过 iframe 加载页面的时候，直接不显示所有内容 -->
<head>
  <style id="click-jack">
    html {
      display: none !important;
    }
  </style>
</head>
<body>
  <script>
    if (self == top) {
      var style = document.getElementById('click-jack')
      document.body.removeChild(style)
    } else {
      top.location = self.location
    }
  </script>
</body>
```


## 参考

- [前端安全系列（二）：如何防止 CSRF 攻击？](https://tech.meituan.com/2018/10/11/fe-security-csrf.html)
- [CORS 和 CSRF 修炼宝典](https://xie.infoq.cn/article/1299660678cda463e988251b9)
- [Web安全实践：点击劫持](https://zhuanlan.zhihu.com/p/53197562)
- [掘金-前端面试之道-安全防范知识点](https://juejin.cn/book/6844733763675488269/section/6844733763776151565)

