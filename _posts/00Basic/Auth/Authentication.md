---
title: Basic - Authentication
# author: Grace JyL
date: 2020-09-23 11:11:11 -0400
description:
excerpt_separator:
categories: [00Basic, Authentication]
tags: [Basic, Authentication]
math: true
# pin: true
toc: true
# image: /assets/img/sample/devices-mockup.png
---

- [Basic - Authentication](#basic---authentication)
  - [Different Authentication](#different-authentication)
  - [Different use case](#different-use-case)
    - [Cookies](#cookies)
    - [Tokens](#tokens)
    - [Signatures](#signatures)
- [HTTP authentication](#http-authentication)
  - [basic](#basic)
  - [Authentication Type/schemes](#authentication-typeschemes)
  - [`Basic authentication scheme`](#basic-authentication-scheme)
    - [step:](#step)
    - [drawbacks](#drawbacks)
    - [Security of basic authentication](#security-of-basic-authentication)
  - [Proxy authentication](#proxy-authentication)
  - [Authentication of cross-origin images](#authentication-of-cross-origin-images)
- [Session-based authentication](#session-based-authentication)
  - [step](#step-1)
- [Token-based authentication](#token-based-authentication)
  - [characteristics of token](#characteristics-of-token)
    - [token and password](#token-and-password)
  - [token based authentication strategies](#token-based-authentication-strategies)
    - [JWT, JSON web tokens](#jwt-json-web-tokens)
    - [SAML](#saml)
    - [OAuth - Open Authorization 开放授权](#oauth---open-authorization-开放授权)
      - [OAuth 2.0 术语表](#oauth-20-术语表)
      - [Scope 授权范围](#scope-授权范围)
      - [OAuth 2.0 配置](#oauth-20-配置)
      - [authorization grant](#authorization-grant)
        - [授权码 `AUTHORIZATION_CODE` -\> response\_type=code](#授权码-authorization_code---response_typecode)
        - [隐藏式 `implicit` -\> response\_type=token](#隐藏式-implicit---response_typetoken)
        - [密码式 `password` -\> grant\_type=password](#密码式-password---grant_typepassword)
        - [第四种方式:凭证式 `client credentials` -\> token](#第四种方式凭证式-client-credentials---token)
        - [令牌的使用](#令牌的使用)
        - [**更新令牌** `refresh_token` -\> token](#更新令牌-refresh_token---token)
      - [example](#example)
        - [OAuth2 Proxy](#oauth2-proxy)
        - [oauth2 proxy with Github](#oauth2-proxy-with-github)
        - [基于k8s部署的nginx服务 通过ingress和oauth2 proxy对接gitlab](#基于k8s部署的nginx服务-通过ingress和oauth2-proxy对接gitlab)
        - [在Gitlab配置**OpenID应用**](#在gitlab配置openid应用)
        - [生成**Cookie密钥**](#生成cookie密钥)
        - [部署**oauth2-proxy**](#部署oauth2-proxy)
        - [创建测试应用并配置Ingress](#创建测试应用并配置ingress)
        - [测试外部认证](#测试外部认证)
        - [流程分析](#流程分析)
    - [OpenID](#openid)
    - [OpenID Connect(OIDC) 协议](#openid-connectoidc-协议)
      - [OAuth2 vs OIDC](#oauth2-vs-oidc)
      - [basic](#basic-1)
      - [OIDC的好处](#oidc的好处)
      - [OIDC相关的协议](#oidc相关的协议)
      - [OIDC核心规范](#oidc核心规范)
      - [协议流程](#协议流程)
        - [声明（Claim）](#声明claim)
        - [ID Token](#id-token)
      - [授权](#授权)
        - [Authorization code 授权码方式](#authorization-code-授权码方式)
          - [授权步骤](#授权步骤)
          - [身份验证请求](#身份验证请求)
          - [授权响应](#授权响应)
          - [获取Token](#获取token)
          - [验证Token](#验证token)
          - [获取用户信息 UserInfo](#获取用户信息-userinfo)
        - [Implicit 隐式授权](#implicit-隐式授权)
          - [授权步骤](#授权步骤-1)
          - [授权请求](#授权请求)
          - [授权响应](#授权响应-1)
        - [混合授权](#混合授权)
          - [授权步骤](#授权步骤-2)
          - [授权请求](#授权请求-1)
          - [授权响应](#授权响应-2)
      - [example](#example-1)
        - [通过 OIDC 协议实现 SSO 单点登录](#通过-oidc-协议实现-sso-单点登录)
        - [创建自己的用户目录](#创建自己的用户目录)
        - [架设自己的 OIDC Provider](#架设自己的-oidc-provider)
        - [在 OIDC Provider 申请 Client](#在-oidc-provider-申请-client)
        - [修改配置文件](#修改配置文件)
        - [启动 node-oidc-provider](#启动-node-oidc-provider)
        - [编写第一个应用](#编写第一个应用)
        - [编写第二个应用](#编写第二个应用)
        - [向 OIDC Provider 发起登录请求](#向-oidc-provider-发起登录请求)
        - [Web App 从 OIDC Provider 获取用户信息](#web-app-从-oidc-provider-获取用户信息)
        - [登录第二个 Web App](#登录第二个-web-app)
        - [登录态管理](#登录态管理)
- [compare](#compare)





---


# Basic - Authentication

认证（Authentication）:通过认证以确定用户身份，认证可以理解为用户登录过程。
授权（Authorization）:给用户分配可权限，以确定用户可访问的资源范围。授权的前提是要确认用户身份，即先认证，再授权。

各种应用都需要做用户验证。最简单的方式是在本地维护一个数据库，存放用户账户和证书等数据。这种方式对于业务来说可能会不太友好：

注册和账户创建过程本来就很无聊。对于很多电商网站来说，它们会允许非登陆用户添加购物车，然后让用户下单时再注册。乏味的注册流程可能会导致很多用户放弃购买。
对于那些提供多个应用的企业来说，让各个应用维护各自的用户数据库，不管从管理还是安全层面来说，都是一个很大的负担。
对于这个问题，更好的方案是将用户认证和授权这些事情交给专门的identity provider（idp）服务来处理。

google、facebook、twitter这些大厂，就为它们的注册用户提供了这类idp服务。一个网站可以通过使用这类idp服务来极大简化用户的注册和登录流程。



---


## Different Authentication

1. HTTP authentication: `username and passwd` are sent in each request
2. Session based authentication: `session id` are sent in each request
3. Token based authentication: `token` are sent in each request


## Different use case

- to support a **web application** only
  - either `cookies` or `tokens` are fine
  - for cookies think about XSRF,
  - for JWT take care of XSS.

- to support both a **web application and mobile client**
  - go with an API that supports `token-based authentication`.

- If building **APIs** that communicate with each other
  - go with `request signing`.




```yaml
#  OAuth2.0
https://b.com/oauth/authorize?
  response_type=code # 告知了授权服务端用授权码来响应
  &client_id=your_client_id
  # scope=read
  &scope=profile%20contacts # 客户端请求能够访问该用户公共主页和联系人的用户许可
  &redirect_uri=CALLBACK_URL


# OpenID Connect 认证请求 URI
https://accounts.google.com/o/oauth2/v2/auth?
   response_type=code
   &client_id=your_client_id
   &scope=openid%20contacts
   &redirect_uri=https%3A//oauth2.example.com/code

```




---


### Cookies

Cookies
- When a server receives an `HTTP request` in the response, it can send a `Set-Cookie` header.
- The browser puts it into a cookie jar, and the cookie will be sent along with every request made to the same origin in the `Cookie HTTP header`.

To use cookies for authentication, few key principles must follow.

1. Always use `HttpOnly` cookies
   - To mitigate the possibility of XSS attacks
   - use the `HttpOnly` flag when setting cookies.
   - This way cookies won't show up in `document.cookies`.

2. Always use `signed` cookies
   - With signed cookies, a server can tell if a cookie was modified by the client.


Cookies can be observed in Chrome
- how a server set cookies:

![illustration of Chrome cookie set for web authentication purposes](https://i.imgur.com/VubG0Xs.png)

- Later on, all the requests use the cookies set for the given domain:

![web authentication method illustration Chrome cookie usage](https://i.imgur.com/cEM2xQw.png)


The cons:
1. Need to make extra effort to mitigate `CSRF attacks`
2. Incompatibility with REST - as it introduces a state into a stateless protocol


---

### Tokens



---

### Signatures

> cookies or tokens,
> if the transport layer for whatever reason gets exposed, credentials are easy to access, the attacker can act like the real user.

**sign each request**.
- A possible way to solve
  - at least when we are talking about APIs and not the browser

When a consumer of an API makes a request `it has to sign it`
- meaning it has to create a hash from the entire request using a private key.

For that hash calculation you may use:
- HTTP method
- Path of the request
- HTTP headers
- Checksum of the HTTP payload
- and a private key to create the hash

To make it work, both the consumer of the API and the provider have to have the same private key.
- Once have the signature, have to add it to the request, either in query strings or HTTP headers.
- Also, a date should be added as well, so you can define an expiration date.

AWS Request Signing: Flow of a Web Authentication Method:

![aws_request_signing_flow_of_a_web_authentication](https://i.imgur.com/fLUPjQr.png)

go through all these steps, even if the transport layer gets compromised, an attacker can only read your traffic, won't be able to act as a user, as the attacker will not be able to sign requests
- as the private key is not in his/her possession.
- Most AWS services are using this kind of authentication.

`node-http-signature` deals with HTTP Request Signing and worth checking out.

The cons:
- Cannot use in the browser / client, only between APIs
- One-Time Passwords
  - One-Time passwords algorithms
  - generate a one-time password with a shared secret and either the current time or a counter:
  - `Time-based One-time Password Algorithm`, based on the current time,
  - `HMAC-based One-time Password Algorithm`, based on a counter.

These methods are used in applications that leverage two-factor authentication:
- a user enters the username and password then both the server and the client generates a one-time password.

In Node.js, implementing this using notp is relatively easy.

Cons:
- with the shared-secret (if stolen) user tokens can be emulated
- because clients can be stolen / go wrong every real-time application have methods to bypass this, like an email reset that adds additional attack vectors to the application


---


# HTTP authentication

## basic

server:
- `WWW-Authenticate` and `Proxy-Authenticate` headers
- The `WWW-Authenticate` and `Proxy-Authenticate` response headers **define the authentication method that should be used** to gain access to a resource.
- must specify which authentication scheme is used, so the client knows how to provide the credentials.

- The syntax for these headers:

  ```html
  WWW-Authenticate: <type> realm=<realm>
  Proxy-Authenticate: <type> realm=<realm>
  ```

  - `<type>` is the **authentication scheme**
    - `Basic` : the most common scheme and introduced below

  - `realm` : describe the protected area or to indicate the scope of protection.
    - a message like "Access to the staging site" or similar
    - so that the user knows to which space they are trying to get access to.


client:
- `Authorization` and `Proxy-Authorization` request headers
- contain the credentials to authenticate a user agent with a (proxy) server
- `<type>` is needed again
- `credentials`: be encoded or encrypted depending on which authentication scheme is used.

  ```html
  Authorization: <type> 12345678
  Proxy-Authorization: <type> 123456
  ```

---

## Authentication Type/schemes
The general HTTP authentication framework is used by several authentication schemes.
- Schemes can differ in security strength and in their availability in client or server software.
- there are other schemes offered by host services, such as Amazon AWS.

| Schemes            | Note                                                                                              |
| ------------------ | ------------------------------------------------------------------------------------------------- |
| `Basic`            | RFC 7617, **base64-encoded** credentials.                                                         |
| `Bearer`           | See RFC 6750, bearer tokens to access OAuth 2.0-protected resources                               |
| `Digest`           | See RFC 7616, only md5 hashing is supported in Firefox, see bug 472823 for SHA encryption support |
| `HOBA`             | See RFC 7486, Section 3, HTTP Origin-Bound Authentication, digital-signature-based                |
| `Mutual`           | See RFC 8120                                                                                      |
| `AWS4-HMAC-SHA256` | See AWS docs                                                                                      |

---


## `Basic authentication scheme`

![Basic authentication](https://i.imgur.com/W4lnwIN.png)

- transmits credentials as user `ID/password` pairs, encoded using `base64`.

- the simplest possible way to enforce access control
  - as it `doesn't require cookies, sessions or anything else`.


- the client
  - provide username and password when making a request.
  - has to send the `Authorization` header along with every request it makes.
- the exchange must happen over an **HTTPS (TLS) connection** to be secure.

- The username and password are not encrypted, but constructed this way:
  1. username and password are concatenated into a single string: `username:password`
  2. this string is encoded with `Base64`
  3. the Basic keyword is put before this encoded value

---

### step:

![HTTPAuth](https://i.imgur.com/HMO7vyi.png)

1. client access come protected URL: `https://some.url`

2. server check the request has `Authorization header` with `valid usrname and passwd`

    ```
    HTTTP/1.1 401 Unauthorized
    Date: Sat, 16 May 2020 16:50:53 GMT
    WWW-Authenticate: Basic realm="MyApp"
    ```

   - The server responds to a client with a `401 (Unauthorized) response status` and provides information on how to authorize with a WWW-Authenticate response header containing at least one challenge.
     - `401 Unauthorized`: invalid, authentication is impossible for this user.
     - `200 OK`: exist and valid
     - `403 forbidden`: valid credentials that are inadequate to access a given resource
     - `407 (Proxy Authentication Required)`: authentication is impossible for this user.
   - realm:
     - protection space:
     - group of pages use the same credential.
     - browser can cache the calid credentials for given realm and use them in future
   - "free text":
     - server is responsible for defininf realms and do the authentication


3. browser notice the `WWW-Authenticate` header in response:
   - show the windoW
   - presents the alert for credentials


4. user submit username and passwd.


5. browser encode it with  `base64` and sends in the next request
   - Browsers use `utf-8` encoding for usernames and passwords.
   - `base64("username:passwd")`
   - send a `Authorization` request header with the credentials.


6. server do step 2 again


Example

```
curl --header "Authorization: Basic am9objpzZWNyZXQ=" my-website.com
```

The same can be observed in Chrome

![google_chrome_basic_web_authentication_method-1448359567226](https://i.imgur.com/V1x3yw7.png)

Implementing in `Node.js`

```js
import basicAuth from 'basic-auth';

export default function auth(req, res, next) {
  const {name, pass} = basicAuth(req) || {};

  if (!name || !pass) return unauthorized(res);
  if (name === 'john' && pass === 'secret') return next();
  return unauthorized(res);
};

function unauthorized(res) {
  res.set('WWW-Authenticate', 'Basic realm=Authorization Required');
  return res.send(401);
};
```

---

### drawbacks
1. the **username and password are sent with every request**
   - not secure unless used with TLS/HTTPS.
     - anyone can eavesdrop and decode the credentials.
   - potentially exposing them
   - even sent via a secure connection connected to SSL/TLS, if a website uses weak encryption, or an attacker can break it, the usernames and passwords will be exposed immediately
2. **no way to log out** the user using Basic auth
3. expiration of credentials is not trivial
   - have to ask the user to change password to do so

---

### Security of basic authentication

the user ID and password are passed over the network as clear text
- base64 encoded, but is a reversible encoding
- the `basic authentication scheme` is not secure.
- HTTPS/TLS should be used for basic authentication.
  - Without additional security enhancements
  - basic authentication should not be used to protect sensitive or valuable information.

1. Restricting access with `Apache` and basic authentication

   - To password-protect a directory on an Apache server
     - need a `.htaccess` and a `.htpasswd` file.
   - `.htpasswd file`:
     - each line consists of a username and a password
     - separated by a colon (:).
     - the passwords are hashed (MD5-based hashing)
     - can name the `.htpasswd file` differently
       - but keep in mind this file shouldn't be accessible to anyone.
       - Apache usually configured to prevent access to `.ht* files`


      ```
      .htaccess file:

      AuthType Basic
      AuthName "Access to the staging site"
      AuthUserFile /path/to/.htpasswd
      Require valid-user
      ```

      ```
      .htpasswd file :

      aladdin:$apr1$ZjTqBB3f$IF9gdYAGlMrs2fuINjHsz.
      user2:$apr1$O04r.y2H$/vEkesPhVInBByJUkXitA/
      ```

2. Restricting access with `nginx` and basic authentication
   1. a `location` going to protect
   2. the `auth_basic directive`: provides the name to the password-protected area.
   3. The `auth_basic_user_file directive` : points to a `.htpasswd file` containing the encrypted user credentials, just like Apache

      ```
      location /status {
          auth_basic           "Access to the staging site";
          auth_basic_user_file /etc/apache2/.htpasswd;
      }
      ```

3. Avoid access using `credentials in the URL`
   - Many clients can avoid the login prompt by using an `encoded URL` containing the credentials
   - `https://username:password@www.example.com/`
   - The use of these URLs is deprecated.
     - In Chrome, the `username:password@` part in URLs is even stripped out for security reasons.
     - In Firefox, it is checked if the site actually requires authentication and if not,
       - Firefox will warn the user with a prompt
       - "You are about to log in to the site “www.example.com” with the username “username”, but the website does not require authentication. This may be an attempt to trick you."



---

## Proxy authentication

The same challenge and response mechanism can be used for proxy authentication.

- As both resource authentication and proxy authentication can coexist
- but different set of `headers` and `status codes` is needed.

1. the challenging status code is `407 (Proxy Authentication Required)`
   - the `Proxy-Authenticate` response header
     - contains at least one challenge applicable to the proxy
     - used for providing the credentials to the proxy server.



## Authentication of cross-origin images
security hole recently been fixed by browsers is `authentication of cross-site images`.
- From Firefox 59
- **image resources loaded from different origins to the current document** are no longer able to trigger `HTTP authentication dialogs` (bug 1423146),
- preventing user credentials being stolen if attackers were able to embed an arbitrary image into a third-party page.



---


# Session-based authentication

![session-base](https://i.imgur.com/9PNfrgQ.png)

client sends the **session id** in all the request, and server uses it to identify the user.

- a stateful authentication method
- as the server need it to be implemented

---

## step

1. client: sent the login request

2. server: `creates and stores the session data` in the **server memory** after the user logs in
   - some random unique identifier to identify the user

3. client: the `session id` also stores in a **cookie on the user browser**.
   - store in cookie if cokkies enabled
   - or somewhere else, e.g. in `local/session storage`
   - The `session Id` is sent on subsequent requests to the server

4. server: compares it with the `stored session data` and proceeds to process the requested action.

5. server:
   - when user logout, the session is destroyed
   - cookie removed and session removed from the server
   - same session ID can not be reused

```
1. client
http://someurl/login (username:passwd)

2. server
session ID: 123456

3. client stored it

4. silent send request with session id
http://someurl/login (session id:123456)

5. server check the session id
- 200 ok
- 401 unauthorized
```

![Screen Shot 2020-09-24 at 00.36.44](https://i.imgur.com/RWalJf8.png)

![Screen Shot 2020-09-24 at 00.36.54](https://i.imgur.com/3nF9a8H.png)

![Screen Shot 2020-09-24 at 00.37.27](https://i.imgur.com/NBesRsm.png)

![Screen Shot 2020-09-24 at 00.37.40](https://i.imgur.com/k4Ajzfo.png)

![1_Hg1gUTXN5E3Nrku0jWCRow](https://i.imgur.com/8w0haoq.png)


---






# Token-based authentication

![token-based](https://i.imgur.com/HKxHiDr.png)

**token**

- the `user state is stored inside the token` **on the client side**.

- token is a normal `URL-safe string`
  - can be pass in header, body, or URL
  - most of the time sent by HTTP headers, not cookies
    - not only for browser, but also app.
  - the preferred mode of authentication for `RESTful APIs`.

- token is `self-contained`
  - contains session info and user info as well
  - carries the data

- anyone can view the contained

- as a standard

- use when interact between multiple untrsuted parties:
  - bank - app - user

- have lifetime

- can grant access to only a subset of data
  - one `token` for one `API`
  - just give specific right: make transaction, just search ...


token has 3 part: `header.payload.signature`
- `header`:
  - string generated using `base64(tokenMetadata)`
  - token's Metadata
    ```
    {
      "typ":"jwt",
      "alg":"HS256",
    }
    ```

- `payload`:
  - string generated using `base64(ourDatas)`
  - ourDatas: data that want to embed in the token (aka JWT Claims)
  - these are called `claims`
  - 3 type of `claims`
    - <font color=blue> Registered claims </font>
    - ![Registered claims](https://i.imgur.com/KHBHULL.png)
    - <font color=blue> Public claims </font>
      - claims to defines and use for our own data
      - e.g. userid, email...
    - <font color=blue> Private claims </font>
      - names without meaning except the consumer and producer of tokens
    ```
    {
      "userid":"123",
      "email":"234",
    }
    ```

- `signature`:
  - string generated by hashing the header+payload with a secret
  - `HMACSHA256(header + ',' + payload, 'secret')`
  - `secret`: held at server and used to generate and verify tokens

![1_PDry-Wb8JRquwnikIbJOJQ](https://i.imgur.com/jtgGEVO.png)


![Screen Shot 2020-09-24 at 00.42.22](https://i.imgur.com/c4WAwrS.png)

1. client send credentials to generate a token
2. server validate the credentials:
   - `422`: unprocessable entity
   - `200`: ok + `token in body or header`
     - the user data is encrypted into a `JWT (JSON Web Token)` with a `secret`
     - and then sent back to the client.
3. The `JWT` stored on the client-side in `localStorage/cookie` and sent as a `header` for every subsequent request.
4. The server validates the `JWT` before proceeding to send a response to the client.
   - `401`: unauthorized
   - `200`: ok + `token in body or header`

```
headers:{
"Authorization": "Bearer ${JWT_TOKEN}"
}
```


---


## characteristics of token

1. random string
2. server does not store it (stateless)
3. has an expiry, then token is useless
4. normally sighed with a secret so to identify any tampering and thus can be trusted by the server
5. normally sent in the authorization header
6. can be `Opaque` or `Self-contained`
   - `Opaque`
     - random string, no meaning
     - can only be verfied by the authorization server
     - just like session ids
   - `Self-contained`
     - token has the data and can be reviewd by the clients
     - e.g. JWT tokens


### token and password

token and password 的作用是一样的，都可以进入系统，但是有三点差异。
- 令牌是短期的，到期会自动失效，用户自己无法修改。
  - 密码一般长期有效，用户不修改，就不会发生变化。
- 令牌可以被数据所有者撤销，会立即失效。以
  - 可以随时取消令牌。密码一般不允许被他人撤销。
- 令牌有权限范围 scope
  - 比如只能进小区的二号门。
  - 对于网络服务来说，只读令牌就比读写令牌更安全。
  - 密码一般是完整权限。

- 注意，只要知道了令牌，就能进入系统。
  - 系统一般不会再次确认身份，所以令牌必须保密，泄漏令牌与泄漏密码的后果是一样的。
  - 这也是为什么令牌的有效期，一般都设置得很短的原因。

**OAuth 2.0 的优点**
- 令牌既可以让第三方应用获得权限，同时又随时可控，不会危及系统安全。
- OAuth 2.0 对于如何颁发令牌的细节，规定得非常详细。具体来说，一共分成四种授权类型(authorization grant)，即四种颁发令牌的方式，适用于不同的互联网场景。



---

## token based authentication strategies

emaple of token based authentication strategies

- SWT, simple web tokenss

- JWT, JSON web tokens

- SAML, security assertions markup language

- **OAuth**, open authorization

- **OAuth2.0**
  - 向第三方系统提供授权（访问自身）服务的协议规范。
  - 通过向第三方系统提供Token，以便在不向第三方系统提供自身密码的情况下，授权第三方系统访问自身的部分服务。
  - 有两种 OAuth 2.0 授权流程最为常见：
    - 服务端应用程序的`授权码流程`
    - 和 基于浏览器的应用程序的`隐式流程`。
  - OpenID Connect 是 OAuth 2.0 协议之上的标识层，以使 OAuth 适用于认证的用例。

- **OpenID、OpenID Connect(OIDC)**
  - 提供第三方认证的协议规范(单点登录SSO)，即一个认证服务和多个业务应用。
  - 用户在认证中心登录，业务应用通过认证中心接口获取用户身份信息
  - 典型场景为企业内部Web系统集成单点登录，典型的有CAS。


<font color=red> OAuth 解决了代理授权的问题，但是它没有提供一个认证用户身份的标准方法 </font>
- OAuth 2.0 用于**授权**
- OpenID Connect 用于**认证**



---


### JWT, JSON web tokens

- form of token based authentication

- based on an Open Standard

- JWT (JSON Web Token) is everywhere

- JWT consists of three parts:
  - `Header`, containing the type of the token and the hashing algorithm
  - `Payload`, containing the claims
  - `Signature`, which can be calculated as follows if you chose HMAC SHA256:
    - `HMACSHA256( base64UrlEncode(header) + "." + base64UrlEncode(payload), secret)`

Adding JWT to Koa applications:

```js
var koa = require('koa');
var jwt = require('koa-jwt');
var app = koa();

app.use(
  jwt(
    {secret: 'very-secret'}
  )
);

// Protected middleware
app.use(
  function *(){
    // content of the token will be available on this.state.user
    this.body = {
      secret: '42'
    };
});
```

Example usage - (to check out the validity/content of the token, you can use `jwt.io`):

```baSH
curl --header "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4gRG9lIiwiYWRtaW4iOnRydWV9.TJVA95OrM7E2cBab30RMHrHDcEfxjoYZgeFONFh7HgQ" my-website.com
```


tokens can be observed in Chrome

![google_chrome_json_web_token_as_a_web_authentication](https://i.imgur.com/f6P96PY.png)

Google Chrome JSON Web Token as a web authentication method

when writing APIs for native mobile applications or SPAs, JWT can be a good fit.
to use JWT in the browser have to stored in either `LocalStorage` or `SessionStorage`, can lead to XSS attacks.

The cons:
- Need to make extra effort to mitigate XSS attacks



---

### SAML

> SAML
> Security Assertion 断言 Markup Language

![Saml-Authentication-Final](/assets/img/Saml-Authentication-Final.png)

![Pasted Graphic](/assets/img/Pasted%20Graphic.png)

- Security Assertion Markup Language (SAML)

- This `single sign-on (SSO) login standard` has significant advantages over logging in using a username/password, to use this information to log users in to other applications, such as web-based applications, one way of doing this is by using **SAML**.

- a standard for logging users into applications based on their sessions in another context.

- an open standard that defines a XML-based framework for `exchanging authentication and authorization information` between an `identity provider (IdP)` and a `service provider (SP)`, to enable web-based single sign-on (SSO) and identity federation.

- An XML-based markup language, much like HTML.
  - HTML: defining web page elements
  - SAML: It uses tags, defines security authorization.
  - used to exchange authentication and authorization information between identity providers and service providers.

- SAML commly used for
  - federated identity management across mulyiple organizations.
  - federation / web browser single sign-on implementations.
  - Allows an application to securely authenticate a user by receiving credentials from a web domain.
  - (TACACS+, RADIUS, Kerberos cannot do this)!!!


![Pasted Graphic 1](/assets/img/Pasted%20Graphic%201.jpg)



- An SSO solution used for web-based application.

- Example:

- A secure web portal accessible to user by username and password, use SAML to support authentication.
  - `Portal`: service provider, request an authentication assertion
  - `back-end networks`: function as an identity provider and issue an authentication assertion

- SAML在单点登录中大有用处：
	- 在SAML协议中，一旦用户身份被主网站（身份鉴别服务器，Identity Provider，IDP）认证过后，该用户再去访问其他在主站注册过的应用（服务提供者，Service Providers，SP）时，都可以直接登录，而不用再输入身份和口令。


- 用户登录SP，SP向IDP发起请求来确认用户身份为例子
	- 比如SP是Google的Apps，IDP是一所大学的身份服务器，Alice是该大学的一名学生。

![3297585-50f9c9530cef962d](/assets/img/3297585-50f9c9530cef962d.png)

---

### OAuth - Open Authorization 开放授权

- ref:
  - [https://luvletter.cn/blog/使用oauth2-proxy保护你的应用/](https://luvletter.cn/blog/使用oauth2-proxy保护你的应用/)
  - [https://www.ruanyifeng.com/blog/2019/04/oauth_design.html](https://www.ruanyifeng.com/blog/2019/04/oauth_design.html)
  - https://oauth.net/2/
  - https://kubernetes.github.io/ingress-nginx/examples/auth/oauth-external-auth/
  - https://oauth2-proxy.github.io/oauth2-proxy/docs/
  - [https://energygreek.github.io/2020/07/23/oauth2/](https://energygreek.github.io/2020/07/23/oauth2/)


> OAuth 引入了一个授权层，用来分离两种不同的角色:**客户端** 和 **资源所有者**
> `资源所有者`同意以后，`资源服务器`可以向`客户端`颁发令牌。`客户端`通过令牌，去请求数据。



**代理授权**
- 代理授权是一种允许第三方应用访问用户数据的方法。
- 有两种`代理授权`的方式：
  - 一是你将账号密码提供给第三方应用，以便它们可以代表你来登陆账号并且访问数据；
  - 二是你通过 OAuth 授权第三方应用访问你的数据，而无需提供密码。（我相信我们都不会选择交出我们的密码！）


**OAuth**

- 开放授权

- 一个用于代理授权的标准协议。
  - 允许应用程序在不提供用户密码的情况下访问该用户的数据。

- 一种 <font color=red> 授权机制 </font>

- **数据的所有者**告诉系统，同意**授权第三方应用**进入系统，获取这些数据。
  - allow users to share their private resources to a third party
    - allow some app log you in using twitter
      - expose your twitter info to an external app using `twitter's OAuth server`
    - authorizing you app's frontend from your API
      - your `custom OAuth server`: where user of your website get authorized using `OAuth`


- **系统**从而产生一个短期的`进入令牌token`，用来代替密码，供**第三方应用**使用。

- 2 version:
  - OAuth1.0,
  - OAuth12.0(active, not backward compatible),
  - OAuth12.1


**OAuth 2.0**
- OAuth 2.0是用于授权的 <font color=red> 行业标准协议 </font>。
- OAuth 2.0致力于
  - 简化客户端开发人员的工作
  - 同时为Web应用程序，桌面应用程序，移动电话和客厅设备提供特定的授权流程。

- OAuth 2.0是目前最流行的授权机制，用来授权第三方应用，获取用户数据。

- oauth2 是依赖第三方的认证方式，

**应用场景**
- 很多情况下，许多应用程序不提供内置的身份验证或开箱即用的访问控制。
- 由于这些应用程序处理的敏感数据，这可能是一个主要问题，通常有必要提供某种类型的安全性。
- 基于k8s部署的一些服务，并没有自身的访问认证控制机制。
- 例如
  - 部署一个用于公司内部使用的web应用
    - 不想做基于统一账号SSO的认证功能的开发
    - 但是又想在用户访问时加上一层认证功能。
    - 这类情况的解决思路一般是在访问入口，例如Ingress上添加一层访问认证
      - 可以借助于basic auth实现此功能，但basic auth存在过于简单、账号权限不好控制、需要手动维护等诸多问题。
    - 另外一种相对更为成功的解决办法是使Ingress通过OAuth对接到能够提供支持oauth认证的外部服务，例如github、gitlab。这种方式没有对应用程序的代码侵入，仅仅在应用入口添加了配置。

- 玩游戏的时候弹出QQ登录，微信登录。
  - `游戏运营商`并不需要用户注册
  - `游戏运营商`直接从QQ或者微信那里获取用户的**OPENID**
  - `游戏运营商`存储并通过**OPENID**来识别用户
  - `有资质的游戏运营商`还可以通过玩家的openid来获取用户的信息， 例如用户的手机号，网名，年龄等信息。
    - 有资质这个是有QQ和微信来决定的，游戏运营商需要先去腾讯那里注册认证。腾讯愿意给游戏运营商分享多少信息是腾讯说了算。
  - 以QQ登录为例
    - 玩家登录游戏时
    - `游戏运营商`先让用户访问QQ的auth2服务器，并带上游戏运营商的id。
    - 待QQ认证后会回调`游戏运营商注册的回调接口 一般为oauth/callback`，带上用户的openid。 这样游戏运营商就知道是谁登录了。
    - 如果游戏运营商需要更多用户资料时，例如注册，游戏运营商可以通过QQ的查询接口，密钥以及用户的openid 去查询, 拉取到用户信息。
    - 如果资料不全， 再让玩家补充，例如输入身份证号。这应该国家不准腾讯向别人分享的，必须要用户自己输入。。



**grant types**
- token response in all grant types is normally accompanied by an `expiry date` and a `refresh token` (to refresh the token when expired)

![Screen Shot 2020-09-24 at 01.37.09](https://i.imgur.com/kBixqIO.png)

![Screen Shot 2020-09-24 at 01.39.33](https://i.imgur.com/RhXzwxG.png)

---

#### OAuth 2.0 术语表

- 资源所有者（Resource Owner）：拥有客户端应用程序想要访问的数据的用户。
- 客户端（Client）：想要访问用户数据的的`应用程序`
- 授权服务端（Authorization Server）：通过用户许可，授权客户端访问用户数据的授权服务端。
- 资源服务端（Resource Server）：存储客户端要访问的数据的系统。在某些情况下，资源服务端和授权服务端是同一个服务端。
- 访问令牌：访问令牌是客户端可用于访问资源服务端上用户授权的数据的唯一密钥。

![Screen Shot 2022-07-10 at 02.08.16](https://i.imgur.com/28XqHEN.png)

授权密钥（Authorization Key）或者权限（Grant）可以是授权码或者令牌的类型。

授权的流程 **用户在不提供密码的情况下，允许第三方应用访问用户数据**
- 用户通过点击按钮启动整个授权流程。这个按钮通常类似于“谷歌登陆“、”Facebook 登陆“或者通过其他的应用登陆。
- 然后客户端将用户`重定向`到授权服务端。在重定向的过程中，客户端将类似客户 ID、重定向 URI 的信息发送给授权服务端。
- 授权服务端处理用户认证，并显示授权许可窗口，然后从用户方获得授权许可。如果你通过谷歌登陆，你必须向谷歌，而不是客户端，提供登陆证书——例如向 accounts.google.com 提供登陆证书。
- 如果用户授权许可，则授权服务端将用户`重定向`到客户端，同时发送授权密钥（授权码或令牌）。
- 客户端向资源服务端发送包含授权密钥的请求，要求资源服务端返回用户数据。
- 资源服务端验证授权密钥，并向客户端返回它所请求的数据。


但与此同时，有一些问题出现了：
- 我们如何限制客户端只访问资源服务端上的部分数据？
- 如果我们只希望客户端读取数据，而没有权限写入数据呢？

这些问题将我们引导至 OAuth 技术术语中另一部分很重要的概念：**授权范围（Scope）**。




---


#### Scope 授权范围

在 OAuth 2.0 中，授权范围用于限制应用程序访问某用户的数据。这是通过发布仅限于用户授权范围的权限来实现的。

- 当客户端向授权服务端发起权限请求时，它同时随之发送一个**授权范围列表**。
- 授权客户端根据这个列表生成一个授权许可窗口，并通过用户授权许可。
- 如果用户同意了其授权告知，授权客户端将发布一个令牌或者授权码，该令牌或授权码仅限于用户授权的范围。

举个例子，如果我授权了某客户端应用访问我的谷歌通讯录，则授权服务端向该客户端发布的令牌不能用于删除我的联系人，或者查看我的谷歌日历事件——因为它仅限于读取谷歌通讯录的范围。


---


#### OAuth 2.0 配置

当发起授权权限的请求时，客户端将一些配置数据作为查询参数发送给授权服务端。

这些基本的查询参数包括：
- 响应类型（response_type）：希望从授权服务端获得的响应类型
- 授权范围（scope）：客户端希望访问的授权范围列表。授权服务端将使用这个列表为用户产生同意授权许可窗口。
- 用户 ID（client_id）：由授权服务在为 OAuth 设置客户端时提供。此 ID 可帮助授权服务端确定正在发送 OAuth 流程的客户端。
- 重定向通用资源标识符（redirect_uri）：用于告知授权服务器当 OAuth 流程完成后重定向的地址
- 客户密码（client_secret）：由授权服务提供，根据 OAuth 流程，这个参数可能需要也可能不需要。



---


#### authorization grant

**OAuth 的核心就是向第三方应用颁发令牌**

- 由于互联网有多种场景, 本标准定义了获得令牌的四种授权方式(authorization grant)。

- OAuth 2.0 规定了四种获得令牌的流程 向第三方应用颁发令牌。
  - 授权码(authorization-code)
  - 隐藏式(implicit)
  - 密码式(password):
  - 客户端凭证(client credentials)

注意，不管哪一种授权方式，第三方应用申请令牌之前，都必须先到系统备案，说明自己的身份，然后会拿到两个身份识别码:客户端 ID(client ID)和客户端密钥(client secret)。这是为了防止令牌被滥用，没有备案过的第三方应用，是不会拿到令牌的。

两种最常用的 OAuth2.0 流程是：
- 基于服务器的应用程序所使用的授权码流程，
- 以及 纯 JavaScript 单页应用所使用的隐式流程。




```bash
# A 网站提供一个链接，要求用户跳转到 B 网站，授权用户数据给 A 网站使用。
https://b.com/oauth/authorize?
  response_type=code&  # 授权码
  response_type=token& # 隐藏式
  client_id=CLIENT_ID&
  redirect_uri=CALLBACK_URL&
  scope=read

https://oauth.b.com/token?
  grant_type=password&  # 密码式
  username=USERNAME&
  password=PASSWORD&
  client_id=CLIENT_ID

https://oauth.b.com/token?
  grant_type=client_credentials& # 凭证式
  client_id=CLIENT_ID&
  client_secret=CLIENT_SECRET

# 用户跳转到 B 网站，登录, 同意给予 A 网站授权。
# B 网站就会跳回`redirect_uri`参数指定的跳转网址，并且把令牌作为 URL 参数，传给 A 网站。
https://a.com/callback?code=AUTHORIZATION_CODE  # 授权码
https://a.com/callback#token=ACCESS_TOKEN       # 隐藏式
```


```yaml
# Request
POST /token HTTP/1.1
Host: oauth2.googleapis.com
Content-length: 261
content-type: application/x-www-form-urlencoded
user-agent: google-oauth-playground
code=4%2F0AdQt8qgHXK5ctUWLQk19w4AzrpQzmqoLyzIkajEzTv94IFLcAQVwPVQbBezzVCWDXirlNw
&redirect_uri=https%3A%2F%2Fdevelopers.google.com%2Foauthplayground
&client_id=407408718192.apps.googleusercontent.com
&client_secret=************
&scope=
&grant_type=authorization_code

# Response
HTTP/1.1 200 OK
Content-length: 1273
X-xss-protection: 0
X-content-type-options: nosniff
Transfer-encoding: chunked
Expires: Mon, 01 Jan 1990 00:00:00 GMT
Vary: Origin, X-Origin, Referer
Server: scaffolding on HTTPServer2
-content-encoding: gzip
Pragma: no-cache
Cache-control: no-cache, no-store, max-age=0, must-revalidate
Date: Sun, 10 Jul 2022 20:38:52 GMT
X-frame-options: SAMEORIGIN
Alt-svc: h3=":443"; ma=2592000,h3-29=":443"; ma=2592000,h3-Q050=":443"; ma=2592000,h3-Q046=":443"; ma=2592000,h3-Q043=":443"; ma=2592000,quic=":443"; ma=2592000; v="46,43"
Content-type: application/json; charset=utf-8
{
  "access_token": "abcd",
  "id_token": "eyJhbGciOiJSUzI1NiIsImtpZCI6IjFiZDY4NWY1ZThmYzYyZDc1ODcwNWMxZWIwZThhNzUyNGM0NzU5NzUiLCJ0eXAiOiJKV1QifQ.eyJpc3MiOiJodHRwczovL2FjY291bnRzLmdvb2dsZS5jb20iLCJhenAiOiI0MDc0MDg3MTgxOTIuYXBwcy5nb29nbGV1c2VyY29udGVudC5jb20iLCJhdWQiOiI0MDc0MDg3MTgxOTIuYXBwcy5nb29nbGV1c2VyY29udGVudC5jb20iLCJzdWIiOiIxMDcwNDMxOTE2NjcyNTM2ODc0NjAiLCJlbWFpbCI6ImxncmFjZXllQGhvdG1haWwuY29tIiwiZW1haWxfdmVyaWZpZWQiOnRydWUsImF0X2hhc2giOiIyS1lkaHhKUHVQZlNQZFp4Rk9aV2FBIiwiaWF0IjoxNjU3NDg1NTMyLCJleHAiOjE2NTc0ODkxMzJ9.m675UQKWgX_0eBUNC94sU7FDJHqauWyVQW0XnvYkCz4_AKkuUKxyS7d4VMB4KCSWhUhylBx1ilq5XsdqFlugksEHP6hgRgTf-5M1PIKbo0HEFJWhoFGIZDu907hcQl8eE5mCBk9nr3SuuJpbDLFVy9jaY96qTRrCvVXINOC6mXPjU7mohB0Rg3DgHkCbLvCHbfmPIR72_DuGVmtdQWUrpnQICRGJcdX3PY-wgGoOa9U6qqEJFK9bGcSG-0sE9rnF_iR_piX9jVYFnplxslkuKeGBu4xsQpFenVEFlOEhDw6QAFdvmm6idlpjnXE9j7QwpRxuQ3uX-kM2YOWJCasjdQ",
  "expires_in": 3599,
  "token_type": "Bearer",
  "scope": "https://www.googleapis.com/auth/userinfo.email openid",
  "refresh_token": "abcd"
}
```

---

##### 授权码 `AUTHORIZATION_CODE` -> response_type=code

<font color=red>第三方应用先申请一个授权码，然后再用该码获取令牌</font>

![Screen Shot 2022-07-10 at 02.19.09](https://i.imgur.com/6GVVn55.png)

> 最常用的流程，安全性也最高，是理想的 OAuth 流程。
> 它被认为是非常安全的，因为它同时使用前端途径（浏览器）和后端途径（服务器）来实现 OAuth2.0 机制。
> 它适用于那些有后端的 Web 应用。


![Screen Shot 2022-07-10 at 02.25.27](https://i.imgur.com/Zb6Ow8D.png)

- **授权码**通过前端传送
- **令牌**则是储存在后端，而且所有与资源服务器的通信都在后端完成。

- 将response_type设置成授权码, 因为这样做能使 OAuth 流程非常安全。
- 这样的前后端分离，可以避免令牌泄漏。

  - 访问令牌是唯一能用于访问资源服务端上的数据的东西，而不是授权码。
    - 访问令牌是我们不希望任何人能访问的秘密信息。
    - 如果客户端直接请求访问令牌，并将其存储在浏览器里，它可能会被盗，因为浏览器并不是完全安全的。
    - 任何人都能看见网页的代码，或者使用开发工具来获取访问令牌。

  - 未了避免将访问令牌暴露在浏览器中
    - 客户端的前端从授权服务端获得授权码，然后发送这个授权码到客户端的后端。
    - 现在，为了用授权码交换访问令牌，我们需要一个叫做客户密码（client_secret）的东西。
    - 这个客户密码只有客户端的后端知道，然后后端向授权服务端发送一个 POST 请求，其中包含了授权码和客户密码
    - 这个请求可能如下所示：
    - 授权服务端会验证客户密码和授权码，然后返回一个访问令牌。
    - 后端程序存储了这个访问令牌并且可能使用此令牌来访问资源服务端。
    - 这样一来，浏览器就无法读取访问令牌了。

      ```yaml
      POST /token HTTP/1.1
      Host: oauth2.googleapis.com
      Content-Type: application/x-www-form-urlencoded
      code=4/W7q7P51a-iMsCeLvIaQc6bYrgtp9
      &client_id=your_client_id
      &client_secret=your_client_secret_only_known_by_server
      &redirect_uri=https%3A//oauth2.example.com/code
      ```

![pi](https://www.wangbase.com/blogimg/asset/201904/bg2019040905.jpg)


```bash
# 客户端通过将用户重定向到授权服务端来发起一个授权流程，
# A 网站提供一个链接，用户点击后就会跳转到 B 网站，授权用户数据给 A 网站使用。
https://b.com/oauth/authorize?
  response_type=code& # 告知了授权服务端用授权码来响应
  client_id=CLIENT_ID&
  redirect_uri=CALLBACK_URL&
  # scope=read
  scope=profile%20contacts& # 客户端请求能够访问该用户公共主页和联系人的用户许可

# 用户跳转后，B 网站会要求用户登录，
# 登录后询问是否同意给予 A 网站授权。
# 表示同意，这时 B 网站就会跳回指定的网址 https://a.com/callback
# 跳转时，会传回一个授权码 ?code=AUTHORIZATION_CODE
# 这个请求的结果是授权码，客户端可以使用该授权码来交换访问令牌。
https://a.com/callback?code=AUTHORIZATION_CODE


# A 网站拿到授权码以后，在后端，向 B 网站请求令牌。
https://b.com/oauth/token?
  client_id=CLIENT_ID&
  client_secret=CLIENT_SECRET&
  grant_type=authorization_code&
  code=AUTHORIZATION_CODE&
  redirect_uri=CALLBACK_URL


# B 网站收到请求以后，就会颁发令牌。
# 向`redirect_uri`指定的网址，发送一段 JSON 数据。
{
  "access_token":"ACCESS_TOKEN",
  "token_type":"bearer",
  "expires_in":2592000,
  "refresh_token":"REFRESH_TOKEN",
  "scope":"read",
  "uid":100101,
  "info":{...}
}
# `access_token`字段就是令牌，A 网站在后端拿到了。
```



1. A 网站提供一个链接，用户点击后就会跳转到 B 网站，授权用户数据给 A 网站使用。
   - 下面就是 A 网站跳转 B 网站的一个示意链接。
   - `https://b.com/oauth/authorize?response_type=code&client_id=CLIENT_ID&redirect_uri=CALLBACK_URL&scope=read`
   - 上面 URL 中:
     - `response_type`参数表示要求返回授权码(`code`)，
     - `client_id`参数让 B 知道是谁在请求，
     - `redirect_uri`参数是 B 接受或拒绝请求后的跳转网址，
     - `scope`参数表示要求的授权范围(这里是只读)。

2. 用户跳转后，B 网站会要求用户登录，然后询问是否同意给予 A 网站授权。
   - 用户表示同意，这时 B 网站就会跳回`redirect_uri`参数指定的网址。
   - 跳转时，会传回一个授权码，就像下面这样。
   - `https://a.com/callback?code=AUTHORIZATION_CODE`
   - 上面 URL 中，`code`参数就是授权码。


3. A 网站拿到授权码以后，就可以在后端，向 B 网站请求令牌。
   - `https://b.com/oauth/token?client_id=CLIENT_ID&client_secret=CLIENT_SECRET&grant_type=authorization_code&code=AUTHORIZATION_CODE&redirect_uri=CALLBACK_URL`
   - 上面 URL 中
     - `client_id`参数和`client_secret`参数用来让 B 确认 A 的身份(`client_secret`参数是保密的，因此只能在后端发请求)
     - `grant_type`参数的值是`AUTHORIZATION_CODE`，表示采用的授权方式是授权码
     - `code`参数是上一步拿到的授权码
     - `redirect_uri`参数是令牌颁发后的回调网址。


4. B 网站收到请求以后，就会颁发令牌。
   - 具体做法是向`redirect_uri`指定的网址，发送一段 JSON 数据。

    ```json
    {
      "access_token":"ACCESS_TOKEN",
      "token_type":"bearer",
      "expires_in":2592000,
      "refresh_token":"REFRESH_TOKEN",
      "scope":"read",
      "uid":100101,
      "info":{...}
    }
    ```

    - 上面 JSON 数据中，`access_token`字段就是令牌，A 网站在后端拿到了。


---



##### 隐藏式 `implicit` -> response_type=token

> 有些 Web 应用是纯前端应用，没有后端。
> 这时就不能用上面的方式了，必须将令牌储存在前端。
> **RFC 6749 就规定了第二种方式，允许直接向前端颁发令牌**
> 这种方式没有授权码这个中间步骤，所以称为(授权码)"隐藏式"(implicit)

![Screen Shot 2022-07-10 at 02.31.30](https://i.imgur.com/1F8avbz.png)

- 客户端将浏览器重定向到授权服务端 URI，并将response_type设置成token，以启动授权流程。
- 授权服务端处理用户的登录和授权许可。
- 请求的返回结果是访问令牌，客户端可以通过这个令牌访问资源服务端。

隐式流程被认为不那么安全，因为浏览器负责管理访问令牌，因此令牌有可能被盗。尽管如此，它仍然被单页应用广泛使用。


```bash
# A 网站提供一个链接，要求用户跳转到 B 网站，授权用户数据给 A 网站使用。
https://b.com/oauth/authorize?
  response_type=token&   # `response_type`参数为`token`，表示要求直接返回令牌。
  client_id=CLIENT_ID&
  redirect_uri=CALLBACK_URL&
  scope=read

# 用户跳转到 B 网站，登录, 同意给予 A 网站授权。
# B 网站就会跳回`redirect_uri`参数指定的跳转网址，并且把令牌作为 URL 参数，传给 A 网站。
https://a.com/callback#token=ACCESS_TOKEN # `token`参数就是令牌

# A 网站因此直接在前端拿到令牌。
```


- 注意，令牌的位置是 URL 锚点(fragment)，而不是查询字符串(querystring)，这是因为 OAuth 2.0 允许跳转网址是 HTTP 协议，因此存在"中间人攻击"的风险，而浏览器跳转时，锚点不会发到服务器，就减少了泄漏令牌的风险。

![pi](https://www.wangbase.com/blogimg/asset/201904/bg2019040906.jpg)

- 这种方式把令牌直接传给前端，是很不安全的。因此，只能用于一些安全要求不高的场景
- 并且令牌的有效期必须非常短，通常就是会话期间(session)有效，浏览器关掉，令牌就失效了。

---

##### 密码式 `password` -> grant_type=password


**如果你高度信任某个应用，RFC 6749 也允许用户把用户名和密码，直接告诉该应用。该应用就使用你的密码，申请令牌，这种方式称为"密码式"(password)。**


```bash
# A 网站要求用户提供 B 网站的用户名和密码。
# 拿到以后，A 就直接向 B 请求令牌。
https://oauth.b.com/token?
  grant_type=password&   # 授权方式 密码式
  username=USERNAME&     # 用户名和密码
  password=PASSWORD&
  client_id=CLIENT_ID

# B 网站验证身份通过后，直接给出令牌。
# 注意，这时不需要跳转，而是把令牌放在 JSON 数据里面，作为 HTTP 回应

# A 因此拿到令牌
```

- 这种方式需要用户给出自己的用户名/密码，显然风险很大
- 因此只适用于其他授权方式都无法采用的情况，而且必须是用户高度信任的应用。

---

##### 第四种方式:凭证式 `client credentials` -> token

**最后一种方式是凭证式(client credentials)，适用于没有前端的命令行应用，即在命令行下请求令牌。**

```bash
# A 应用在命令行向 B 发出请求。
https://oauth.b.com/token?
  grant_type=client_credentials&
  client_id=CLIENT_ID&         # 用来让 B 确认 A 的身份。
  client_secret=CLIENT_SECRET

# B 网站验证通过以后，直接返回令牌。
```

- 这种方式给出的令牌，是针对第三方应用的，而不是针对用户的，即有可能多个用户共享同一个令牌。

---

##### 令牌的使用

**令牌的使用**
- A 网站拿到令牌以后，就可以向 B 网站的 API 请求数据了。
- 每个发到 API 的请求，都必须带有令牌。
- 具体做法是在请求的头信息，加上一个`Authorization`字段，令牌就放在这个字段里面。

```
curl -H "Authorization: Bearer ACCESS_TOKEN" "https://api.b.com"
```

上面命令中，`ACCESS_TOKEN`就是拿到的令牌。

---

##### **更新令牌** `refresh_token` -> token
- 令牌的有效期到了，如果让用户重新走一遍上面的流程，再申请一个新的令牌，很可能体验不好，而且也没有必要。
- OAuth 2.0 允许用户自动更新令牌。

```bash
# B 网站颁发令牌的时候，一次性颁发两个令牌
# 一个用于获取数据，另一个用于获取新的令牌(refresh token 字段)。
# 令牌到期前，用户使用 refresh token 发一个请求，去更新令牌。

https://b.com/oauth/token?
  grant_type=refresh_token&
  client_id=CLIENT_ID&
  client_secret=CLIENT_SECRET&
  refresh_token=REFRESH_TOKEN

# B 网站验证通过以后，就会颁发新的令牌。
```


---

#### example

---

##### OAuth2 Proxy

- 一个使用go编写的反向代理和静态文件服务器
- 使用提供程序(Google，GitHub和其他提供商)提供身份验证，以通过电子邮件，域或组验证帐户。

![Screen Shot 2022-03-23 at 11.13.30](https://i.imgur.com/G7dpktE.png)

---

##### oauth2 proxy with Github

1. 先去github -> developer 创建oauth应用， 输入自己的回调地址。 当用户被github认证后，会调用这个地址
2. 在服务端配置，利用一个开源 oauth2_proxy 工具, 项目地址:`https://github.com/oauth2-proxy/oauth2-proxy`
3. 配置 nginx



**oauth2-proxy.cfg**

```yaml
auth_logging = true
# auth_logging_format = "{{.Client}} - {{.Username}} [{{.Timestamp}}] [{{.Status}}] {{.Message}}"
# pass HTTP Basic Auth, X-Forwarded-User and X-Forwarded-Email information to upstream
pass_basic_auth = true
# pass_user_headers = true
# pass the request Host Header to upstream
# when disabled the upstream Host is used as the Host Header
pass_host_header = true


# 可以通过验证的邮箱域名
# Email Domains to allow authentication for (this authorizes any email on this domain)
# for more granular authorization use `authenticated_emails_file`
# To authorize any email addresses use "*"
# email_domains = [
#     "yourcompany.com"
# ]
email_domains=["*"]

# callback的域名
allowlist_domains = [".example.com"]
cookie_domains = ["example.com"]
skip_auth_preflight = false


# Cookie Settings
# Name     - the cookie name
# Secret   - the seed string for secure cookies; should be 16, 24, or 32 bytes
#            for use with an AES cipher when cookie_refresh or pass_access_token
#            is set
# Domain   - (optional) cookie domain to force cookies to (ie: .yourcompany.com)
# Expire   - (duration) expire timeframe for cookie
# Refresh  - (duration) refresh the cookie when duration has elapsed after cookie was initially set.
#            Should be less than cookie_expire; set to 0 to disable.
#            On refresh, OAuth token is re-validated.
#            (ie: 1h means tokens are refreshed on request 1hr+ after it was set)
# Secure   - secure cookies are only sent by the browser of a HTTPS connection (recommended)
# HttpOnly - httponly cookies are not readable by javascript (recommended)
# cookie_name = "_oauth2_proxy"
# cookie加密密钥
cookie_secret = "beautyfly"
cookie_domains = "beautyflying.cn"
cookie_expire = "168h"
# cookie_refresh = ""
cookie_secure = false
# cookie_httponly = true



http_address="0.0.0.0:4180"
# 与GitHub callback URL一致
# The OAuth Client ID, Secret
redirect_url="https://example.com/oauth2/callback"
provider="github"
# 刚刚创建的GitHub OAuth Apps里有
client_id = "cef54714c84e3b0c2248"
client_secret = "a96d3d94771273b5295202d03c0c2d3ca7f625dc"
# Pass OAuth Access token to upstream via "X-Forwarded-Access-Token"
pass_access_token = false
# Authenticated Email Addresses File (one email per line)
# authenticated_emails_file = ""
# Htpasswd File (optional)
# Additionally authenticate against a htpasswd file. Entries must be created with "htpasswd -s" for SHA encryption
# enabling exposes a username/login signin form
# htpasswd_file = ""
# Templates
# optional directory with custom sign_in.html and error.html
# custom_templates_dir = ""
# skip SSL checking for HTTPS requests
# ssl_insecure_skip_verify = false


# 限制登录用户
github_users=["J2ephyr"]
```


可以将oauth2 配置成服务

```bash
[Unit]
Description = OAuth2 proxy for www blog

[Service]
Type=simple
ExecStart=/usr/bin/oauth2_proxy -config /etc/oauth2-proxy.cfg
[Install]
WantedBy=multi-user.target
```


nginx 配置

```
location /oauth2/ {
        proxy_pass       http://127.0.0.1:4180;
        proxy_set_header Host                    $host;
        proxy_set_header X-Real-IP               $remote_addr;
        proxy_set_header X-Scheme                $scheme;
        proxy_set_header X-Auth-Request-Redirect $request_uri;
    # or, if you are handling multiple domains:
    # proxy_set_header X-Auth-Request-Redirect $scheme://$host$request_uri;
}

location = /oauth2/auth {
proxy_pass       http://127.0.0.1:4180;
proxy_set_header Host             $host;
proxy_set_header X-Real-IP        $remote_addr;
proxy_set_header X-Scheme         $scheme;
# nginx auth_request includes headers but not body
proxy_set_header Content-Length   "";
proxy_pass_request_body           off;
}

location / {

auth_request /oauth2/auth;
error_page 401 = /oauth2/sign_in;

# pass information via X-User and X-Email headers to backend,
# requires running with --set-xauthrequest flag
auth_request_set $user   $upstream_http_x_auth_request_user;
auth_request_set $email  $upstream_http_x_auth_request_email;
proxy_set_header X-User  $user;
proxy_set_header X-Email $email;

# if you enabled --pass-access-token, this will pass the token to the backend
auth_request_set $token  $upstream_http_x_auth_request_access_token;
proxy_set_header X-Access-Token $token;

# if you enabled --cookie-refresh, this is needed for it to work with auth_request
auth_request_set $auth_cookie $upstream_http_set_cookie;
add_header Set-Cookie $auth_cookie;

# When using the --set-authorization-header flag, some provider's cookies can exceed the 4kb
# limit and so the OAuth2 Proxy splits these into multiple parts.
# Nginx normally only copies the first `Set-Cookie` header from the auth_request to the response,
# so if your cookies are larger than 4kb, you will need to extract additional cookies manually.
auth_request_set $auth_cookie_name_upstream_1 $upstream_cookie_auth_cookie_name_1;

# Extract the Cookie attributes from the first Set-Cookie header and append them
# to the second part ($upstream_cookie_* variables only contain the raw cookie content)
if ($auth_cookie ~* "(; .*)") {
    set $auth_cookie_name_0 $auth_cookie;
    set $auth_cookie_name_1 "auth_cookie_name_1=$auth_cookie_name_upstream_1$1";
}

# Send both Set-Cookie headers now if there was a second part
if ($auth_cookie_name_upstream_1) {
    add_header Set-Cookie $auth_cookie_name_0;
    add_header Set-Cookie $auth_cookie_name_1;
}

    root   /usr/share/nginx/html/blog;
    index  index.html index.htm;
}

#error_page  404              /404.html;

# redirect server error pages to the static page /50x.html
#
error_page   500 502 503 504  /50x.html;
location = /50x.html {
    root   /usr/share/nginx/html;
}
```

**upstreams.ymal**

需要被保护的应用服务代理配置

```yaml
upstreams:
  - id: example
    path: /example
    url: http://app.example.com
```

- 访问https://example.com/example
- 就会被直接代理到 http://app.example.com/example





---

##### 基于k8s部署的nginx服务 通过ingress和oauth2 proxy对接gitlab

- 基于k8s部署的nginx服务
- 通过ingress和oauth2 proxy对接gitlab
- 实现对应用没有代码侵入的外部认证。



实验环境:
- k8s 1.15.0
- Ingress nginx 0.25.0
- gitlab 13.7.4


##### 在Gitlab配置**OpenID应用**

- 登录到Gitlab—>管理中心—>应用，创建一个应用
  - 参数:
    - **Authorization callback URL** 回调URL:
      - 指GitLab在用户通过身份验证后应将其发送到的端点
      - 填入oauth2-proxy的callback地址
      - 对于oauth2-proxy应该是`https://<应用域名>/oauth2/callback`
    - 范围:
      - 应用程序对GitLab用户配置文件的访问级别。
      - 对于大多数应用程序，选择openid，profile和email即可。
  - 创建完应用后，会生成`一对ID和密钥`，这个在后面会用到。

---

##### 生成**Cookie密钥**
- 生成**Cookie密钥**
  - 该Cookie密钥作为`种子字符串`以产生安全的cookie。
  - 使用base64编码，可利用以下的python脚`本生成字符串。

```py
import secrets
import base64
print(base64.b64encode(base64.b64encode(secrets.token_bytes(16))))
```

---


##### 部署**oauth2-proxy**

- 部署**oauth2-proxy**
  - 在k8s中部署 oauth-proxy，资源清单oauth2-gitlab.yaml 和 相关参数

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  labels:
    k8s-app: oauth2-proxy
  name: oauth2-proxy
  namespace: kube-system
spec:
  replicas: 1
  selector:
    matchLabels:
      k8s-app: oauth2-proxy
  template:
    metadata:
      labels:
        k8s-app: oauth2-proxy
    spec:
      containers:
      - name: oauth2-proxy
        image: quay.io/oauth2-proxy/oauth2-proxy:latest
        imagePullPolicy: IfNotPresent
        ports:
        - containerPort: 4180
          protocol: TCP
        args:
        # OAuth提供者
        - --provider=gitlab
        # 上游端点的http网址
        - --upstream=file:///dev/null
        # 对具有指定域的电子邮件进行身份验证，可以多次给出，使用*验证任何电子邮件
        - --email-domain=*
        # 监听的地址
        - --http-address=0.0.0.0:4180
        # 设置安全(仅HTTPS)cookie标志
        - --cookie-secure=false
        # OAuth重定向URL
        - --redirect-url=https://nginx-test.ssgeek.com/oauth2/callback
        # 跳过登录页面直接进入下一步
        - --skip-provider-button=false
        # 设置X-Auth-Request-User，X-Auth-Request-Email和X-Auth-Request-Preferred-Username响应头(在Nginx auth_request模式下有用)。与结合使用时--pass-access-token，会将X-Auth-Request-Access-Token添加到响应标头中
        - --set-xauthrequest=true
        # 跳过OPTIONS请求的身份验证
        - --skip-auth-preflight=false
        # 绕过OIDC端点发现
        - --skip-oidc-discovery
        # OpenID Connect发行者url，这里是gitlab的url
        - --oidc-issuer-url=https://gitlab.ssgeek.com
        # 认证url
        - --login-url=https://gitlab.ssgeek.com/oauth/authorize
        # token url
        - --redeem-url=https://gitlab.ssgeek.com/oauth/token
        # 用于令牌验证的url
        - --oidc-jwks-url=https://gitlab.ssgeek.com/oauth/discovery/keys
        env:
        - name: OAUTH2_PROXY_CLIENT_ID
          value: '85945b7195ab109377183837b9221bd299bc64b31fe272304a1c777e8e241d83'
        - name: OAUTH2_PROXY_CLIENT_SECRET
          value: '2f9782928b493686f387d18db9138e92607448cef045c81319967cc3e5ce4ba1'
         # 安全cookie的种子字符串，可通过python脚本生成
        - name: OAUTH2_PROXY_COOKIE_SECRET
          value: 'VGlYNVBVOGw4UFgyRURzbERxVTRiZz09'

---
apiVersion: v1
kind: Service
metadata:
  labels:
    k8s-app: oauth2-proxy
  name: oauth2-proxy
  namespace: kube-system
spec:
  type: NodePort
  ports:
  - name: http
    port: 4180
    protocol: TCP
    targetPort: 4180
    nodePort: 30020
  selector:
    k8s-app: oauth2-procy

```



- 应用上面的资源清单，创建deployment和service

```bash
$ kubectl apply -f oauth2-gitlab.yaml
$ kubectl -n kube-system get pods -l k8s-app=oauth2-proxy
NAME                           READY   STATUS    RESTARTS   AGE
oauth2-proxy-884695869-bkwns   1/1     Running   0          113
```

- 通过nodeport单独暴露了oauth2-proxy应用，可以访问检查以确保浏览器可以正常打开


![Screen Shot 2022-03-23 at 11.22.30](https://i.imgur.com/h7fkMq7.png)


---

##### 创建测试应用并配置Ingress

资源清单文件`nginx.yaml`如下，其中为该nginx应用配置了https证书

```yaml
apiVersion: extensions/v1beta1
kind: Deployment
metadata:
  name: nginx
  namespace: kube-system
spec:
  replicas: 1
  selector:
    matchLabels:
      app: nginx
  template:
    metadata:
      labels:
        app: nginx
    spec:
      containers:
      - image: nginx:1.15
        imagePullPolicy: IfNotPresent
        name: nginx

---
apiVersion: v1
kind: Service
metadata:
  name: nginx
  namespace: kube-system
spec:
  selector:
    app: nginx
  ports:
  - name: nginx
    port: 80
    targetPort: 80

---
apiVersion: extensions/v1beta1
kind: Ingress
metadata:
  name: nginx
  namespace: kube-system
  annotations:
    kubernetes.io/ingress.class: "nginx"
    nginx.ingress.kubernetes.io/rewrite-target: /
    # 指定外部认证url
    nginx.ingress.kubernetes.io/auth-url: "https://$host/oauth2/auth"
    # 指定外部认证重定向的地址
    nginx.ingress.kubernetes.io/auth-signin: "https://$host/oauth2/start?rd=$escaped_request_uri"
    nginx.ingress.kubernetes.io/force-ssl-redirect: "true"
    nginx.ingress.kubernetes.io/secure-backends: "true"
    nginx.ingress.kubernetes.io/ssl-passthrough: "true"
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
spec:
  tls:
  - hosts:
    - nginx-test.ssgeek.com
    secretName: nginx-test
  rules:
    - host: nginx-test.ssgeek.com
      http:
        paths:
        - path: /
          backend:
            serviceName: nginx
            servicePort: 80

---
apiVersion: extensions/v1beta1
kind: Ingress
metadata:
  annotations:
    kubernetes.io/ingress.class: "nginx"
    # 将nginx应用的访问请求跳转到oauth2-proxy组件url
    nginx.ingress.kubernetes.io/rewrite-target: "/oauth2"
    nginx.ingress.kubernetes.io/force-ssl-redirect: "true"
    nginx.ingress.kubernetes.io/secure-backends: "true"
    nginx.ingress.kubernetes.io/ssl-passthrough: "true"
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
  name: nginx-oauth2
  namespace: kube-system
spec:
  tls:
  - hosts:
    - nginx-test.ssgeek.com
    secretName: nginx-test
  rules:
  - host: nginx-test.ssgeek.com
    http:
      paths:
      - path: /oauth2
        backend:
          serviceName: oauth2-proxy
          servicePort: 4180
```


- 应用上面的资源清单，创建相应资源

```bash
$ kubectl apply -f other/nginx.yaml
deployment.extensions/nginx unchanged
service/nginx unchanged
ingress.extensions/nginx unchanged
ingress.extensions/nginx-oauth2 unchanged


$ kubectl -n kube-system get po,svc,ing |grep nginx
pod/nginx-5ddcc6cb74-rnjlx                    1/1     Running   0          3m
   80/TCP                      3m
ingress.extensions/nginx               nginx-test.ssgeek.com             80, 443   3m
ingress.extensions/nginx-oauth2        nginx-test.ssgeek.com             80, 443   3m
```



##### 测试外部认证

通过访问上面部署的nginx应用，在浏览器中进行测试，会被重定向到Gitlab登录页面；

输入账号，正确登录后，会被重定向回nginx应用。

![05zwoff77t](https://i.imgur.com/Xb90yDQ.gif)

---

##### 流程分析

在请求登录外部认证的过程中查看oauth2-proxy的日志如下

```bash

# 访问nginx应用的时候，Ingress nginx controller会向定义的 auth-url 发起认证
# 该认证由Ingress nginx controller发起，所以Ingress nginx controller对应的pod必须能够访问 auth-url。

# 如果认证没有通过，Ingress nginx controller将客户端重定向到 auth-signin。
# auth-signin 是目标应用的 oauth2登录页面 即 oauth2-proxy。

# 客户端被重定向到oauth2登录页面后，自动进入Gitlab的登录页面，
# 用户登录Gitlab后，Gitlab再将客户端重定向到在Gitlab中配置的 应用 回调地址。

# 客户端访问 回调地址 后，oauth2_proxy在客户端设置cookie，并将客户端重定向到最初的访问地址。


172.16.1.110:49976 - -                [2021/01/23 17:28:23] nginx-test.ssgeek.com GET - "/oauth2/auth"                          HTTP/1.1 "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_6) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0.2 Safari/605.1.15" 401 21  0.000

172.16.1.110:9991 - -                 [2021/01/23 17:28:23] nginx-test.ssgeek.com GET - "/oauth2/start?rd=%2F"                  HTTP/1.1 "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_6) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0.2 Safari/605.1.15" 302 341 0.000

172.16.1.110:9991 - admin@example.com [2021/01/23 17:28:32] [AuthSuccess] Authenticated via OAuth2: Session{email:admin@example.com user:root PreferredUsername: token:true id_token:true created:2021-01-23 17:28:32.440915913 +0000 UTC m=+2248.944621207 expires:2021-01-23 17:30:32 +0000 UTC refresh_token:true}

# 带有cookie的客户端再次访问目标应用时，通过了auth-url的认证，成功访问到目标服务即nginx应用。
172.16.1.110:9991 - -                 [2021/01/23 17:28:32] nginx-test.ssgeek.com GET - "/oauth2/callback?code=abcd&state=abcd" HTTP/1.1 "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_6) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0.2 Safari/605.1.15" 302 24 0.381
172.16.1.110:5610 - admin@example.com [2021/01/23 17:28:32] nginx-test.ssgeek.com GET - "/oauth2/auth"                          HTTP/1.1 "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_6) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0.2 Safari/605.1.15" 202 0 0.000
```



---


### OpenID


OpenID vs OpenID Connect

- `OpenID`:由OpenID基金会维护的第三方认证规范，存在如下缺点:
  - 以URI为用户唯一标识，用户难以记忆
  - 第三方应用必须是网站，没有提供API，不支持移动应用
  - 不支持健壮的加密和签名

- `OpenID Connect`:基于OAuth2.0实现的用户认证规范。相对OpenID提供了如下增强特性。
  - 提供可扩展性，运行人们通过任何OpenID Connect Provider进行身份验证，而不是仅限于Google、Facebook等主流IDP。
  - 电子邮件作为用户标识，便于用户记忆。
  - 允许客户端动态注册，减轻管理员显示注册设备和网站的工作量。


一个授权服务如何为第三方提供认证服务？
- OpenID Connect对OAuth2.0接口进行了扩展，通过在协议中扩展身份认证所需的`id_token`字段，增加UserInfo Endpoint接口，向第三方应用提供身份认证服务。
- 用户执行OAuth2.0的授权流程后，第三方应用获得Access Token和附加的`id_token`，`id_token`包含基本的用户身份信息，可用于身份认证。
- 如果需要更详细用户信息，第三方应用通过Access Token，从认证服务UserInfo Endpoint接口获取用户信息。
第三方应用可以把`id_token`和UserInfo信息作为认证用户的用户信息。




---






### OpenID Connect(OIDC) 协议

<font color=red> OAuth2与资源访问和共享有关，而OIDC与用户身份验证有关。 </font>

OpenID Connect 1.0 is a simple identity layer on top of the OAuth 2.0 protocol. It allows Clients to verify the identity of the End-User based on the authentication performed by an Authorization Server, as well as to obtain basic profile information about the End-User in an interoperable and REST-like manner.

OpenID Connect allows clients of all types, including Web-based, mobile, and JavaScript clients, to request and receive information about authenticated sessions and end-users. The specification suite is extensible, allowing participants to use optional features such as encryption of identity data, discovery of OpenID Providers, and session management, when it makes sense for them.


#### OAuth2 vs OIDC

![Screen Shot 2022-07-10 at 02.36.39](https://i.imgur.com/MlRVQnB.png)

- 一个`OAuth2`上层的简单身份层协议。
  - OIDC 是在 OAuth2.0 协议之上的标识层。
  - 它拓展了 OAuth2.0，使得**认证**方式标准化。

OIDC在2014年发行。虽然它不是第一个idp标准，但从可用性、简单性方面来说，它可能是最好的。OIDC从SAML和OpenID 1.0/2.0中做了大量借鉴。

- OAuth
  - 不会立即提供用户身份，而是会提供用于**授权**的`访问令牌`。
  - OAuth2.0 通过 `Access Token` 作为向 第三方应用 **授权** 访问自身资源的凭证。
  - OAuth2.0 使用 `Access Token` 来授权三方应用访问受保护的信息。

- OIDC
  - <font color=red> OIDC 对 OAuth2.0 进行协议进行了扩展 </font>
    - OIDC 遵循 oAuth2.0 协议流程，并在这个基础上提供了id token来解决三方应用的用户**身份认证**问题。

---

#### basic

- 一种安全认证机制，

- `第三方应用` 连接到 `身份认证服务器（Identify Service）` 获取用户信息，并把这些信息以安全可靠的方式返回给 `第三方应用`

- 它允许客户端验证用户的身份并获取基本的用户配置信息。


- OIDC 使**客户端**能够通过**认证**来识别用户
  - 认证在授权服务端执行。
    - 在向授权服务端发起用户登录和授权告知的请求时，定义一个名叫openid的授权范围。
    - 在告知授权服务器需要使用 OIDC 时，openid是必须存在的范围

  - 通过扩展的`id_token`字段，提供用户基础身份信息
    - ID 令牌是 JWT (一个编码令牌)，它由三部分组成：头部，有效负载和签名。
    - `id_token`使用`JWT（JSON Web Token）`格式进行封装,作为信息返回，通过符合OAuth2的流程来获取
    - 提供自包含性、防篡改机制，可以安全的传递给第三方应用程序并容易被验证。

  - OIDC 将用户身份认证信息以id token的方式给到三房应用。
    - 三方应用在验证完id token的正确性后，进一步通过oAuth2授权流程获得的a ccess token 读取更多的用户信息。
    - 通过 Access Token 从认证服务的 UserInfo Endpoint 接口获取更详细的用户信息。



- 它规定了其他应用，例如你开发的应用 A（XX 邮件系统），应用 B（XX 聊天系统），应用 C（XX 文档系统），如何到你的中央数据表中取出用户数据，
- 约定了交互方式、安全规范等，确保了你的用户能够在访问所有应用时，只需登录一遍，而不是反反复复地输入密码，而且遵循这些规范，你的用户认证环节会很安全。




用户目录
- 系统的总用户表就像一本书一样，书的封皮上写着“所有用户”四个字。
- 打开第一页，就是目录，里面列满了用户的名字，翻到对应的页码就能看到这个人的邮箱，手机号，生日信息等等。
- 无论你开发多少个应用，要确保你有一份这些应用所有用户信息的 truth source。
- 所有的注册、认证、注销都要到你的用户目录中进行增加、查询、删除操作。
- 你要做的就是创建一个中央数据表，专门用于存储用户信息，不论这个用户是来自 A 应用、B 应用还是 C 应用。



OIDC Provider
- 经常见到一些网站的登录页面上有「使用 Github 登录」、「使用 Google 登录」这样的按钮。
- 要想集成这样的功能，你要先去 Github 那里注册一个 OAuth App，填写一些资料，然后 Github 分配给你一对 id 和 key。
- 此时 Github 扮演的角色就是 OIDC Provider，你要做的就是把 Github 的这种角色的行为，搬到你自己的服务器来。

在 Github 上面搜索 OIDC Provider 会有很多结果：
- JS：https://github.com/panva/node-oidc-provider
- Golang：https://github.com/dexidp/dex
- Python：https://github.com/juanifioren/django-oidc-provider



####  OIDC的好处

- OIDC使得身份认证可以作为一个服务存在

- OIDC可以很方便的实现SSO（跨顶级域）

- OIDC兼容OAuth2，可以使用Access Token控制受保护的API资源

- OIDC可以兼容众多的IDP（身份提供商）作为OIDC的OP来使用

- OIDC的一些敏感接口均强制要求TLS，除此之外，得益于JWT,JWS,JWE家族的安全机制，使得一些敏感信息可以进行数字签名、加密和验证，进一步确保整个认证过程中的安全保障



OIDC大获成功的秘诀：
- 容易处理的id token。
  - OpenID Connect使用JWT来给应用传递用户的身份信息。
  - JWT以其高安全性（防止token被伪造和篡改）、跨语言、支持过期、自包含等特性而著称，非常适合作为token来使用。
- 基于oAuth2.0协议。
  - id token是经过oAuth2.0流程来获取的，这个流程即支持web应用，也支持原生app。
- 简单。
  - OpenID Connect足够简单。但同时也提供了大量的功能和安全选项以满足企业级业务需求。


其目的是为您提供多个站点的登录名。
- 每次需要使用OIDC登录网站时，都会被重定向到登录的OpenID网站，然后再回到该网站。
- 例如，如果选择使用Google帐户登录Auth0，这就使用了OIDC。
  - 成功通过Google身份验证并授权Auth0访问您的信息后，
  - Google会将有关用户和执行的身份验证的信息发送回Auth0。
  - 此信息在JWT中返回，包含`id_token`或者Access Token。

- JWT包含Claims
  - 它们是有关实体(通常是用户)的Claims(例如名称或电子邮件地址)和其他元数据。

- OIDC规范定义了一组标准的权利要求。
  - 这组标准声明包括姓名，电子邮件，性别，出生日期等。
  - 但是，如果要获取有关用户的信息，并且当前没有最能反映此信息的标准声明，则可以创建自定义声明并将其添加到令牌中。


---

#### OIDC相关的协议

OIDC本身是有多个规范构成，其中最主要的是一个核心的规范，多个可选支持的规范来提供扩展支持

主要包含：

- Core：必选。定义OIDC的核心功能，在OAuth 2.0之上构建身份认证，以及如何使用Claims来传递用户的信息。
- Discovery：可选。发现服务，使客户端可以动态的获取OIDC服务相关的元数据描述信息
- Dynamic Client Registration：可选。动态注册服务，使客户端可以动态的注册到OIDC的OP
- Session Management：可选。Session管理，用于规范OIDC服务如何管理Session信息
- Form Post Response Mode：可选。针对OAuth2的扩展，OAuth2回传信息给客户端是通过URL的querystring和fragment这两种方式，这个扩展标准提供了一基于form表单的形式把数据post给客户端的机制。

基础协议：
- OAuth2.0 Core：https://tools.ietf.org/html/rfc6749
- OAuth2.0 Bearer：https://tools.ietf.org/html/rfc6750
- OAuth2.0 Assertions：https://tools.ietf.org/html/rfc7521
- OAuth2.0 JWT Profile：https://tools.ietf.org/html/rfc7523
- OAuth2.0 Responses：可选。针对OAuth2的扩展，提供几个新的response_type。
- JWT(JSON Web Token)：https://tools.ietf.org/html/rfc7519
- JWS（JSON Web Signature）：https://tools.ietf.org/html/rfc7515
- JWE(JSON Web Encryption)：https://tools.ietf.org/html/rfc7516
- JWK(JSON Web Key)：https://tools.ietf.org/html/rfc7517
- JWA(JSON Web Algorithms)：https://tools.ietf.org/html/rfc7518
- WebFinger：https://tools.ietf.org/html/rfc7033

![Screen Shot 2022-07-14 at 12.12.34](https://i.imgur.com/3tXZL3r.png)

OIDC不是什么新技术，它主要是借鉴OpenId的身份标识，OAuth2的授权和JWT包装数据的方式，组合使用这些技术就是现在的OIDC。

---

#### OIDC核心规范

较OAuth2，OIDC有一些不同的概念:
- **OpenID Provider(OP)**，实现OIDC的`OAuth2授权服务器`, Authorization Server
- **Relying Party(RP)**，使用OIDC的`OAuth2客户端`, client
- **End-User(EU)**，用户
- **id_token**，JWT格式的授权Claims
- **UserInfo Endpoint**，用户信息接口，通过`id_token`访问时返回用户信息，此端点必须为HTTPS


#### 协议流程

从理论上来讲，OIDC协议遵循以下步骤:
- RP发送认证请求到OP
- OP验证End-User并颁发**授权**
- OP用`id_token`(通常是Access Token)进行响应
- RP携带Access Token发送请求到UserInfo Endpoint
- UserInfo Endpoint返回End-User的Claims


1. RP发送认证请求到OP
   1. 客户端发起的用于 OpenID Connect
   2. 认证请求 URI 会是如下的形式：

   ```yaml
   https://accounts.google.com/o/oauth2/v2/auth?
    response_type=code
    &client_id=your_client_id
    &scope=openid%20contacts
    &redirect_uri=https%3A//oauth2.example.com/code
   ```

2. OP验证End-User并颁发**授权**
   1. OP用`id_token`(通常是Access Token)进行响应
   2. 该请求的返回结果
      1. 是客户端可以用来交换`访问令牌和 ID 令牌`的**授权码**。
      2. 如果 OAuth 流程是隐式的，那么授权服务端将直接返回`访问令牌和 ID 令牌`。

3. 在获得了 ID 令牌后，客户端可以将其解码，并且得到被编码在有效负载中的用户信息，
   1. RP携带Access Token发送请求到UserInfo Endpoint
   2. UserInfo Endpoint返回End-User的Claims 声明
   3. 如以下例子所示：

  ```yaml
  {
    "iss": "https://accounts.google.com",
    "sub": "10965150351106250715113082368",
    "email": "johndoe@example.com",
    "iat": 1516239022,
    "exp": 1516242922
  }
  ```

```
+--------+                                   +--------+
|        |                                   |        |
|        |---------(1) AuthN Request-------->|        |
|        |                                   |        |
|        |  +--------+                       |        |
|        |  |        |                       |        |
|        |  |  End-  |<--(2) AuthN & AuthZ-->|        |
|        |  |  User  |                       |        |
|   RP   |  |        |                       |   OP   |
|        |  +--------+                       |        |
|        |                                   |        |
|        |<--------(3) AuthN Response--------|        |
|        |                                   |        |
|        |---------(4) UserInfo Request----->|        |
|        |                                   |        |
|        |<--------(5) UserInfo Response-----|        |
|        |                                   |        |
+--------+                                   +--------+

AuthN=Authentication，表示认证；

AuthZ=Authorization，代表授权。


RP发往OP的请求，是属于Authentication类型的请求，
虽然在OIDC中是复用OAuth2的Authorization请求通道，但是用途是不一样的，

OIDC的AuthN请求中scope参数必须要有一个值为的openid的参数, 用来区分这是一个OIDC的Authentication请求，而不是OAuth2的Authorization请求。
```

---


##### 声明（Claim）

ID 令牌的有效负载包括了一些被称作声明的域。

基本的声明有：
- `iss`：令牌发布者
- `sub`：用户的唯一标识符
- `email`：用户的邮箱
- `iat`：用 Unix 时间表示的令牌发布时间
- `exp`：Unix 时间表示的令牌到期时间

然而，声明不仅限于上述这些域。
- 由授权服务器对声明进行编码。客户端可以用这些信息来认证用户。
- 如果客户端需要更多的用户信息，客户端可以指定标准的 OpenID Connect 范围，来告知授权服务端将所需信息包括在 ID 令牌的有效负载中。
  - 这些范围包括个人主页（profile）、邮箱（email）、地址（address）和电话（phone）。

---

##### ID Token

**ID Token**
- OIDC 对 OAuth2 进行的主要扩展(用户用户身份验证)就是 `id_token`
- `id_token` 的概念类似身份证，只不过是JWT的形式，并由OP签发。

- 其中包含`授权服务器`对`用户` **验证的Claims** 和 **其它请求的Claims**

- `id_token` 可能包含其它 Claims，任何未知的Claims都必须忽略。
  - `id_token` 必须使用JWS进行签名，并分别使用JWS和JWE进行可选的签名和加密，从而提供身份验证、完整性、不可抵赖性和可选的机密性。
  - 如果对`id_token`进行了加密，则必须先对其签名，结果是一个嵌套的JWT。
  - `id_token`不能使用`nonce`作为alg值，除非所使用的**响应类型**没有从Authorization Endpoint返回任何`id_token`(如Authorization Code Flow)，并且客户端在注册时显示请求使用`nonce`


id token具有如下属性：
- 说明是哪位用户，也叫做主题（sub）
- 说明token由谁签发的（iss）
- 是否是为某一个特殊的用户生成的（aud）
- 可能会包含一个随机数（nonce）
- 认证时间（auth_time），以及认证强度（acr）
- 签发时间（iat）和过期时间（exp）
- 可能包含额外的请求细节，比如名字和email地址等
- 是否包含数字签名，token的接收方可以验证这个签名
- 可以被加密

一个id token样例如下：
```yaml
{
  "iss"       : "https://openid.c2id.com",
  "sub"       : "alice",
  "aud"       : "client-12345",
  "nonce"     : "n-0S6_WzA2Mj",
  "exp"       : 1311281970,
  "iat"       : 1311280970,
  "auth_time" : 1311280969,
  "acr"       : "c2id.loa.hisec",
}

# id token的头部，包含签名等信息，则会被编码成base64格式，下面是一个例子：
eyJhbGciOiJSUzI1NiIsImtpZCI6IjFlOWdkazcifQ.ewogImlzcyI6ICJodHRw Oi8vc2VydmVyLmV4YW1wbGUuY29tIiwKICJzdWIiOiAiMjQ4Mjg5NzYxMDAxIiw KICJhdWQiOiAiczZCaGRSa3F0MyIsCiAibm9uY2UiOiAibi0wUzZfV3pBMk1qIi wKICJleHAiOiAxMzExMjgxOTcwLAogImlhdCI6IDEzMTEyODA5NzAKfQ.ggW8hZ 1EuVLuxNuuIJKX_V8a_OMXzR0EHR9R6jgdqrOOF4daGU96Sr_P6qJp6IcmD3HP9 9Obi1PRs-cwh3LO-p146waJ8IhehcwL7F09JdijmBqkvPeB2T9CJNqeGpe-gccM g4vfKjkM8FcGvnzZUN4_KSP0aAp1tOJ1zZwgjxqGByKHiOtX7TpdQyHE5lcMiKP XfEIQILVq0pc_E2DzL7emopWoaoZTF_m0_N0YzFC6g6EJbOEoRoSK5hoDalrcvR YLSrQAZZKflyuVCyixEoV9GfNQC3_osjzw2PAithfubEEBLuVVk4XUVrWOLrLl0
nx7RkKU8NXNHq-rvKMzqg

# ID Token必须使用JWS进行签名和JWE加密，从而提供认证的完整性、不可否认性以及可选的保密性。
```




- ID Token是JWS（JSON Web Signature）格式的字符串
  - JWS字符串有三部分组成，分别为JWS Protected Header、JWS Payload、JWS Signature
  - 三部分内容分别Base64编码后通过点(.)拼接，拼接公式如下：

    ```yaml
          BASE64URL(UTF8(JWS Protected Header)) || '.' ||
          BASE64URL(JWS Payload) || '.' ||
          BASE64URL(JWS Signature)
    ```

**JWS Signature**
- JWS可以通过JWS Signature来校验数据的完整性，但不提供机密性。

**JWS Payload**是ID Token的内容部分，是一个JSON对象，包括如下字段：
- 在`id_token`中，以下Clams适用于使用OIDC的所有OAuth2:
- `iss`: 必须
  - （Issuer Identifier）ID Token颁发者的标识符，一般为认证服务器的URL。
  - 发行机构Issuer，大小写敏感的URL，不能包含query参数.
- `sub`: 必须
  - （Subject Identifier）认证用户（End User）标识符，全局唯一。
  - 用户身份Subject，Issuer为End-User分配的唯一标识符，大小写敏感不超过255 ASCII自符
- `aud`: 必须
  - （Audience(s)）ID Token的受众
  - 特别的身份Audience，必须包含OAuth2第三方应用的client_id，大小写敏感的字符串/数组
- `exp`: 必须
  - （Expiration time）Token过期时间。
  - iat到期时间Expire，参数要求当前时间在该时间之前，通常可以时钟偏差几分钟，unix时间戳
- `iat`: 必须
  - （Issued At Time）JWT生成时间。
  - unix时间戳
- `auth_time`:
  - （Authentication Time）用户认证发送时间。
  - End-User验证时间，unix时间戳。
  - 当发出max_age或auth_time Claims时, 必须
- `nonce`:
  - 随机数，防重放攻击。
  - 用于将Client session和`id_token`关联，减轻重放攻击，大小写敏感字符串
- `acr`: 可选
  - Authentication Context Class Reference 表示一个认证上下文引用值，用以标识认证上下文类。
  - 0 End-User不符合ISO/IEC 28115 level 1，不应该授权对任何货币价值的资源访问。大小写敏感的字符串。
- `amr`: 可选
  - Authentication Methods References 一组认证方法。
  - JSON字符串数组，身份验证的表示符，如可能使用了密码和OTP身份验证方式
- `azp`: 可选
  - Authorized party，被授权方。
  - 结合aud使用，只有在被认证的一方和受众（aud）不一致时才使用此值，一般情况下很少使用。
  - 如果存在必须包含OAuth2的Client ID，仅当`id_token`有单个Audience且与授权方不同时，才需要此Claim


---

#### 授权

由于OIDC基于OAuth2，所以OIDC的认证流程主要是由OAuth2的几种授权流程延伸而来的，

`身份验证`遵循以下三种方式；
- Authorization code Flow 授权码方式 (`response_type = code`)
- Implicit Flow 隐式方式 (`response_type = id_token token / id_token`)
- Hybrid Flow 混合方式：混合Authorization Code Flow + Implici Flow。


OAuth2中还有基于Resource Owner Password Credentials Grant和Client Credentials Grant的方式来获取Access Token，为什么OIDC没有扩展这些方式呢？
- Resource Owner Password Credentials Grant是需要用户提供账号密码给RP的，账号密码给到RP，还需要什么ID Token
- Client Credentials Grant这种方式根本就不需要用户参与，更谈不上用户身份认证。这也能反映授权和认证的差异，以及只使用OAuth2来做身份认证的事情是远远不够的，也是不合适的。

下表是三种方式的特征:

| 属性                         | 授权码 | 隐式 | 混合 |
| ---------------------------- | ------ | ---- | ---- |
| Token从authorization端点返回 | no     | yes  | no   |
| Token从token端点返回         | yes    | no   | no   |
| Token未显示给浏览器          | yes    | no   | no   |
| 能够验证客户端               | yes    | no   | yes  |
| 可以刷新Token                | yes    | no   | yes  |
| 一次交流                     | no     | yes  | no   |
| 服务器到服务器               | yes    | no   | no   |

response_type对应的身份验证方式:

| response_type         | 方式   |
| --------------------- | ------ |
| code                  | 授权码 |
| id_token              | 隐式   |
| id_token token        | 隐式   |
| code + id_token       | 混合   |
| code + token          | 混合   |
| code + id_token token | 混合   |


除了由OAuth2定义的“response_type”之外，所有code均在 OAuth2多种响应类型编码实践。

注意OAuth2为隐式类型定义token的响应类型，但OIDC不会使用此响应类型，因为不会返回`id_token`。




---


##### Authorization code 授权码方式

以下是 OIDC 授权码模式的交互模式，你的应用和 OP 之间要通过这样的交互方式来获取用户信息。

- 使用授权码方式时，所有Token从Token端点返回。
- 授权码将授权code返回给客户端，然后客户端可以将其直接交换为`id_token`和Access Token。
- 这样的好处是不会向User-Agent及可能访问User-Agent的其它恶意应用公开任何Token。
- 授权服务器还可以在交换Access Token的授权code之前对客户端进行身份验证。
- 授权code适用于可以安全的维护其自身和授权服务器之间的客户端机密的客户端。


![2020032710565066](https://i.imgur.com/D5E1e5p.png)

OIDC Provider 对外暴露一些接口

- 授权接口 Authorization Endpoint
  - 每次调用这个接口，就像是对 OIDC Provider 喊话：我要登录，如第一步所示。
  - 然后 OIDC Provider 会检查当前用户在 OIDC Provider 的登录状态，
    - 如果是未登录状态，OIDC Provider 会弹出一个登录框，与终端用户确认身份，登录成功后会将一个临时授权码（一个随机字符串）发到你的应用（业务回调地址）；
    - 如果是已登录状态，OIDC Provider 会将浏览器直接重定向到你的应用（业务回调地址），并携带临时授权码（一个随机字符串）。如第二、三步所示。

- token 接口 Token Endpoint
  - 每次调用这个接口，就像是对 OIDC Provider 说：这是我的授权码，给我换一个 access_token。如第四、五步所示。

- 用户信息接口 UserInfo Endpoint
  - 每次调用这个接口，就像是对 OIDC Provider 说：这是我的 access_token，给我换一下用户信息。到此用户信息获取完毕。

为什么这么麻烦？直接返回用户信息不行吗？
- 因为安全，
- code 的有效期一般只有十分钟，而且一次使用过后作废。
- OIDC 协议授权码模式中，只有 code 的传输经过了用户的浏览器，一旦泄露，攻击者很难抢在应用服务器拿这个 code 换 token 之前，先去 OP 使用这个 code 换掉 token。
- 而如果 access_token 的传输经过浏览器，一般 access_token 的有效期都是一个小时左右，攻击者可以利用 access_token 获取用户的信息，而应用服务器和 OP 也很难察觉到，更不必说去手动撤退了。
- 如果直接传输用户信息，那安全性就更低了。
- 一句话：避免让攻击者偷走用户信息。


---

###### 授权步骤

1. `Client (RP)`
   1. 准备一个包含 **所需请求参数** 的 **身份验证请求**
   2. 请求发送到`Authorization Server (OP)`, 向授权服务器（Authorization Server）请求认证。
2. `Authorization Server (OP)`
   1. 对`用户(EU)`进行身份验证
   2. 获得 用户 同意/或授权, 用户确认给Client授权并确认。
   3. 使用 授权码 将 `用户` 发送回`Client (RP)`, URL中携带授权代码。
3. `Client (RP)`
   1. 使用令牌端点上的授权码来请求响应。通过授权代码，向Token Endpoint发送请求。
4. `Token Endpoint`
   1. 收到, 响应
   2. 响应Body中包含 `id_token` 和 `Access Token` (ID令牌和访问令牌)
5. `Client (RP)`
   1. 校验 `id_token`，从中提取用户的身份标识（Subject Identifier）。


---

###### 身份验证请求

Authorization Server (OP)的authorization端点需要支持GET和POST方法，
- GET采用Query String序列化，
- POST采用Form序列化。

OIDC采用OAuth2的授权码流程参数:
- `response_type`: 必须，同OAuth2
- `scope`: 必须，OIDC必须包含openid的scope参数
- `client_id`: 必须，同OAuth2
- `redirect_uri`: 必须，同OAuth2
- `state`，可选，同OAuth2


如:

```yaml
HTTP/1.1 302 Found
Location: https://openid.c2id.com/login?
          response_type=code
          &scope=openid
          &client_id=s6BhdRkqt3
          &state=af0ifjsldkj
          &redirect_uri=https%3A%2F%2Fclient.example.org%2Fcb

GET /authorize?
    response_type=code
    &scope=openid%20profile%20email
    &client_id=s6BhdRkqt3
    &state=af0ifjsldkj
    &redirect_uri=https%3A%2F%2Fclient.example.org%2Fcb HTTP/1.1
Host: server.example.com
```

---

###### 授权响应

OP收到验证请求后，需要对请求参数做严格的验证:

1. 验证OAuth2的相关参数
2. 验证`scope`是否有openid参数，如果没有则为OAuth2请求
3. 验证所有必须的参数是否都存在
4. 如果sub是被要求了，必须尽在由子值标识的最终用户与活动session通过身份验证的情况下积极响应。不得使用不用用户的`id_token`或Access Token响应，即使这些用户与授权服务器由活动session。如果支持claims，则可以使用id_token_hint发出请求。


验证通过后引导EU进行身份认证并同意授权。完成后，会重定向到RP指定的回调地址，并携带`code`和`state`相关参数:

```yaml
HTTP/1.1 302 Found
Location: https://client.example.org/cb?
          code=SplxlOBeZQQYbYS6WxSbIA
          &state=af0ifjsldkj
```


---

###### 获取Token

RP使用上一步获得的code请求token端点，然后就可以获得响应Token
- 其中除了OAuth2规定的数据外，还会附加一个 `id_token` 的字段，

如:

```yaml
POST /token HTTP/1.1
Host: openid.c2id.com
Content-Type: application/x-www-form-urlencoded
Authorization: Basic czZCaGRSa3F0MzpnWDFmQmF0M2JW

grant_type=authorization_code
 &code=SplxlOBeZQQYbYS6WxSbIA
 &redirect_uri=https%3A%2F%2Fclient.example.org%2Fcb
```

成功后，OP会返回带有 `id_token` 的JSON数据:

```yaml
  HTTP/1.1 200 OK
  Content-Type: application/json
  Cache-Control: no-store
  Pragma: no-cache

  {
   "access_token": "SlAV32hkKG",
   "token_type": "Bearer",
   "refresh_token": "8xLOxBtZp8",
   "expires_in": 3600,
   "id_token": "eyJhbGciOiJSUzI1NiIsImtpZCI6IjFlOWdkazcifQ.ewogImlzc
     yI6ICJodHRwOi8vc2VydmVyLmV4YW1wbGUuY29tIiwKICJzdWIiOiAiMjQ4Mjg5
     NzYxMDAxIiwKICJhdWQiOiAiczZCaGRSa3F0MyIsCiAibm9uY2UiOiAibi0wUzZ
     fV3pBMk1qIiwKICJleHAiOiAxMzExMjgxOTcwLAogImlhdCI6IDEzMTEyODA5Nz
     AKfQ.ggW8hZ1EuVLuxNuuIJKX_V8a_OMXzR0EHR9R6jgdqrOOF4daGU96Sr_P6q
     Jp6IcmD3HP99Obi1PRs-cwh3LO-p146waJ8IhehcwL7F09JdijmBqkvPeB2T9CJ
     NqeGpe-gccMg4vfKjkM8FcGvnzZUN4_KSP0aAp1tOJ1zZwgjxqGByKHiOtX7Tpd
     QyHE5lcMiKPXfEIQILVq0pc_E2DzL7emopWoaoZTF_m0_N0YzFC6g6EJbOEoRoS
     K5hoDalrcvRYLSrQAZZKflyuVCyixEoV9GfNQC3_osjzw2PAithfubEEBLuVVk4
     XUVrWOLrLl0nx7RkKU8NXNHq-rvKMzqg"
  }
```

在拿到这些信息后，需要对id_token及access_token进行验证。验证成功就可以通过UserInfo端点获取用户信息了。


---


###### 验证Token

授权服务器必须验证Token的有效性:
- 根据RFC6749
- 验证`id_token`规则
- 验证Access Token规则

---


###### 获取用户信息 UserInfo
Client (RP)可以通过GET或POST请求通过 `UserInfo Endpoint` 获取用户信息。

```yaml
GET /userinfo HTTP/1.1
Host: openid.c2id.com
Authorization: Bearer SlAV32hkKG

# 请求成功:
{
   "sub"                     : "alice",
   "email"                   : "alice@wonderland.net",
   "email_verified"          : true,
   "name"                    : "Alice Adams",
   "given_name"              : "Alice",
   "family_name"             : "Adams",
   "phone_number"            : "+359 (99) 100200305",
   "profile"                 : "https://c2id.com/users/alice",
   "https://c2id.com/groups" : [ "audit", "admin" ]
}
```


---



##### Implicit 隐式授权

隐式授权
- 所有Token都从授权端点返回。
- 主要由浏览器中使用脚本语言实现的客户机使用。
- 访问Token和`id_token`直接返回给客户端，授权服务器不执行客户端身份验证。

---

###### 授权步骤

1. `Client (RP)`
   1. 携带 **认证参数发送请求** 到Authorization Server (OP)

2. `Authorization Server (OP)`
   1. 验证用户 并得到 用户批准
   2. 携带用户相关信息 + `id_token/Access Token` 返回到Client (RP)

3. `Client (RP)`验证 `id_token` 和检索用户标识符

---

###### 授权请求

- `response_type`: 必须，`id_token token`或`id_token`。无Access Token使用`id_token`
- `redirect_uri`: 必须，OP处登记的重定向地址
- `nonce`: 必须，隐式授权必须

```yaml
GET /authorize?
    response_type=id_token%20token
    &client_id=s6BhdRkqt3
    &redirect_uri=https%3A%2F%2Fclient.example.org%2Fcb
    &scope=openid%20profile
    &state=af0ifjsldkj
    &nonce=n-0S6_WzA2Mj
HTTP/1.1
  Host: server.example.com
```

---

###### 授权响应

- `access_token`: 如果response_type是id_token可以不反回
- `token_type`: 固定为Bearer，
- `id_token`: 必须，`id_token`
- `state`
- `expires_in`，可选，Access Token到期时间(s)

之后就可以拿着`id_token`

```yaml
HTTP/1.1 302 Found
  Location: https://client.example.org/cb#
    access_token=SlAV32hkKG
    &token_type=bearer
    &id_token=eyJ0```ZXso
    &expires_in=3600
    &state=af0ifjsldkj
```






---

##### 混合授权

是上面两种模式的混合。

可选response_type有:code id_token，code token，code id_token token。


###### 授权步骤

1. 客户端
   1. 准备一个包含所需请求参数的身份验证请求。
   2. 将请求发送到授权服务器。
2. 授权服务器
   1. 对最终用户进行身份验证。
   2. 获得最终用户同意/授权。
   3. 使用授权码以及一个或多个其他参数（根据响应类型）将最终用户发送回客户端。
3. 客户端
   1. 使用令牌端点上的授权码来请求响应。
   2. 收到响应，该响应在响应主体中包含ID令牌和访问令牌。
   3. 验证ID令牌并检索最终用户的主题标识符。


---

###### 授权请求

```yaml
GET /authorize?
    response_type=code%20id_token
    &client_id=s6BhdRkqt3
    &redirect_uri=https%3A%2F%2Fclient.example.org%2Fcb
    &scope=openid%20profile%20email
    &nonce=n-0S6_WzA2Mj
    &state=af0ifjsldkj

HTTP/1.1
  Host: server.example.com
```

---

###### 授权响应

```yaml
HTTP/1.1 302 Found
  Location: https://client.example.org/cb#
    code=SplxlOBeZQQYbYS6WxSbIA
    &id_token=eyJ0```ZXso
    &state=af0ifjsldkj
```


---


#### example

##### 通过 OIDC 协议实现 SSO 单点登录

SSO

- 例子:
  - 假设有一所大学，内部有两个系统，一个是邮箱系统，一个是课表查询系统。
  - 现在想实现这样的效果：在邮箱系统中登录一遍，然后此时进入课表系统的网站，无需再次登录，课表网站系统直接跳转到个人课表页面，反之亦然。

- Single Sign On
- 流行的企业业务整合的解决方案之一
- SSO 的定义是在多个应用系统中，用户只需要登录一次就可以访问所有相互信任的应用系统。

- 单点登录的意义在于能够在`不同的系统中` **统一账号、统一登录**。
  - 用户不必在每个系统中都进行注册、登录，只需要使用一个统一的账号，登录一次，就可以访问所有系统。


##### 创建自己的用户目录


##### 架设自己的 OIDC Provider

OIDC Provider

- 本文使用 JS 语言的 node-oidc-provider。

示例代码 Github

- 可以在 Github 找到本文示例代码：https://github.com/Authing/implement-oidc-sso-demo.git

```bash
# 创建文件夹，用于存放代码：
$ mkdir demo
$ cd demo

# 克隆仓库
# 将 https://github.com/panva/node-oidc-provider.git 仓库 clone 到本地
$ git clone https://github.com/panva/node-oidc-provider.git

# 安装依赖
$ cd node-oidc-provider
$ npm install
```


##### 在 OIDC Provider 申请 Client
- Github 会分配给你一对 id 和 key
  - 这一步其实就是你在 Github 申请了一个 Client。
  - 在 Github 上填写应用信息，然后提交，会发送一个 HTTP 请求到 Github 服务器。
  - Github 服务器会生成一对 id 和 key，还会把它们与你的应用信息存储到 Github 自己的数据库里。

- 如何向我们自己的服务器上的 OIDC Provider 申请一对这样的 id 和 key 呢
  - 以 node-oidc-provider 举例，
  - 最快的获得一个 Client 的方法就是将 OIDC Client 所需的元数据直接写入 node-oidc-provider 的配置文件里面。
  - 将 OIDC Client 所需的元数据直接写入到配置文件，可以理解成，我们在自己的数据库里手动插入了一条数据，为自己指定了一对 id 和 key 还有其他的一些 OIDC Client 信息。



##### 修改配置文件

进入 node-oidc-provider 项目下的 example 文件夹：

```yaml
$ cd ./example

# 编辑 ./support/configuration.js
# 更改第 16 行的 clients 配置，
# 为自己指定了一个 client_id 和一个 client_secret，
# 其中的 grant_types 为授权模式，authorization_code 授权码模式，
# redirect_uris 数组是允许的业务回调地址，需要填写 Web App 应用的地址, OIDC Provider 会将临时授权码发送到这个地址，以便后续换取 token。

module.exports = {
  clients: [{
      client_id: '1',
      client_secret: '1',
      grant_types: [
        'refresh_token',
        'authorization_code'
      ],
      redirect_uris: [
        'http://baidu.com',
        'http://localhost:8080/app1.html',
        'http://localhost:8080/app2.html'
      ],
    },
  ],
```





##### 启动 node-oidc-provider

在 node-oidc-provider/example 文件夹下，运行以下命令来启动我们的 OP：

```bash
$ node express.js
```
到现在，我们的准备工作已经完成了





##### 编写第一个应用

我们创建一个 app1.html 文件来编写第一个应用 demo，在 demo/app 目录下创建：

```js
$ touch app1.html

// 并写入以下内容：
<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>第一个应用</title>
  </head>

  <body>
    <a href="http://localhost:3000/auth?client_id=1
      &redirect_uri=http://localhost:8080/app1.html
      &scope=openid profile
      &response_type=code
      &state=455356436">
      登录
    </a>

  </body>
</html>
```





##### 编写第二个应用

我们创建一个 app2.html 文件来编写第二个应用 demo，注意 redirect_uri 的变化，在 demo/app 目录下创建：

```js
$ touch app2.html
// 并写入以下内容：

<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>第二个应用</title>
  </head>
  <body>
    <a href="http://localhost:3000/auth?client_id=1
      &redirect_uri=http://localhost:8080/app2.html
      &scope=openid profile
      &response_type=code
      &state=455356436">
      登录
    </a>
  </body>
</html>
```



##### 向 OIDC Provider 发起登录请求

现在我们启动一个 web 服务器，推荐使用 http-server

```bash
$ npm install -g http-server # 安装 http-server
$ cd demo/app
$ http-server .
```


1. 我们访问第一个应用：http://localhost:8080/app1.html
2. 点击「登录」，也就是访问 OIDC Provider 的授权接口。
3. 然后我们来到了 OIDC Provider 交互环节，OIDC Provider 发现用户没有登录，要求用户先登录。
4. node-oidc-provider demo 会放通任意用户名 + 密码
   1. 但是你在真正实施单点登录时，你必须使用你的用户目录即中央数据表中的用户数据来鉴权用户
   2. 相关的代码可能会涉及到数据库适配器，自定义用户查询逻辑，这些在 node-oidc-provider 包的相关配置中需要自行插入。

5. 点击「登录」，转到确权页面，这个页面会显示你的应用需要获取那些用户权限，本例中请求用户授权获取他的基础资料。

6. 点击「继续」，完成在 OP 的登录，之后 OP 会将浏览器重定向到预先设置的业务回调地址，所以我们又回到了 app1.html。

7. 在 url query 中有一个 code 参数，这个参数就是临时授权码。
   1. code 最终对应一条用户信息，接下来看我们如何获取用户信息。


##### Web App 从 OIDC Provider 获取用户信息

code 可以直接发送到后端，然后在后端使用 code 换取 access_token。


```bash
# 用 curl 命令来发送 HTTP 请求：

$ curl --location \
  --request POST 'http://localhost:3000/token' \
  --header 'Content-Type: application/x-www-form-urlencoded' \
  --data-urlencode 'client_id=1' \
  --data-urlencode 'client_secret=1' \
  --data-urlencode 'redirect_uri=http://localhost:8080/app2.html' \
  --data-urlencode 'code=QL10pBYMjVSw5B3Ir3_KdmgVPCLFOMfQHOcclKd2tj1' \
  --data-urlencode 'grant_type=authorization_code'


# 获取到 access_token 之后，我们可以使用 access_token 访问 OP 上面的资源
# 主要用于获取用户信息，即你的应用从你的用户目录中读取一条用户信息。

# 你可以使用 curl 来发送 HTTP 请求：
$ curl --location \
  --request POST 'http://localhost:3000/me' \
  --header 'Content-Type: application/x-www-form-urlencoded' \
  --data-urlencode 'access_token=I6WB2g0Rq9G307pPVTDhN5vKuyC9eWjrGjxsO2j6jm-'
```


用 postman 演示如何通过 code 换取 access_token。


到此，App 1 的登录已经完成，接下来，让我们看进入 App 2 是怎样的情形。





##### 登录第二个 Web App



1. 打开第二个应用，http://localhost:8080/app2.html

2. 然后点击「登录」。



3. 用户已经在 App 1 登录时与 OP 建立了会话，User ←→ OP 已经是登录状态
   1. 所以 OP 检查到之后，没有再让用户输入登录凭证，而是直接将用户重定向回业务地址，并返回了授权码 code。

3. 同样，App 2 使用 code 换 access_token

```bash
# curl 命令代码：
$ curl --location \
  --request POST 'http://localhost:3000/token' \
  --header 'Content-Type: application/x-www-form-urlencoded' \
  --data-urlencode 'client_id=1' \
  --data-urlencode 'client_secret=1' \
  --data-urlencode 'redirect_uri=http://localhost:8080/app2.html' \
  --data-urlencode 'code=QL10pBYMjVSw5B3Ir3_KdmgVPCLFOMfQHOcclKd2tj1' \
  --data-urlencode 'grant_type=authorization_code'

# 再使用 access_token 换用户信息，可以看到，是同一个用户。

# curl 命令代码：
$ curl --location \
  --request POST 'http://localhost:3000/me' \
  --header 'Content-Type: application/x-www-form-urlencoded' \
  --data-urlencode 'access_token=I6WB2g0Rq9G307pPVTDhN5vKuyC9eWjrGjxsO2j6jm-'
```


到此，我们实现了 App 1 与 App 2 之间的账号打通与单点登录。



##### 登录态管理

**单点登录**
- 实现了两个应用之间账号的统一，而且在 App 1 中登录时输入一次密码，在 App 2 中登录，无需再次让用户输入密码进行登录，可以直接返回授权码到业务地址然后完成后续的用户信息获取。



退出问题

**只退出 App 1 而不退出 App 2**

![20200327105659964](https://i.imgur.com/8lM2DWf.png)

这个问题实质上是登录态的管理问题。我们应该管理三个会话：
- User ←→ App 1、
- User ←→ App 2、
- User ←→ OP。

当 OP 给 App 1 返回 code 时，App 1 的后端在完成用户信息获取后，应该与浏览器建立会话，也就是说 App 1 与用户需要自己保持一套自己的登录状态，方式上可以通过 App 1 自签的 JWT Token 或 App 1 的 cookie-session。

对于 App 2，也是同样的做法。

当用户在 App 1 退出时，App 1 只需清理掉自己的登录状态就完成了退出，而用户访问 App 2 时，仍然和 App 2 存在会话，因此用户在 App 2 是登录状态。



**同时退出 App 1 和 App 2**

![20200327105700129](https://i.imgur.com/xua0Iz0.png)

单点登出，即用户只需退出一次，就能在所有的应用中退出，变成未登录状态。

1. 在 OIDC Provider 进行登出。
2. 因为用户和 App 1 , App 2 之间的会话同样依然保持，所以用户在 App 1 和 App 2 的状态仍然是登录态。

所以，有没有什么办法在用户从 OIDC Provider 登出之后，App 1 和 App 2 的会话也被切断呢？我们可以通过 `OIDC Session Mangement` 来解决这个问题。

简单来说，App 1 的前端需要轮询 OP
- 不断询问 OP：用户在你那还登录着吗？
- 如果答案是否定的，App 1 主动将用户踢下线，并将会话释放掉，让用户重新登录，
- App 2 也是同样的操作。


当用户在 OP 登出后，App 1、App 2 轮询 OP 时会收到用户已经从 OP 登出的响应，接下来，应该释放掉自己的会话状态，并将用户踢出系统，重新登录。

OIDC Session Management
- 这部分的核心就是两个 iframe
- 一个是我们自己应用中写的（以下叫做 RP iframe），用于不断发送 PostMessage 给 OP iframe，
- OP iframe 负责查询用户登录状态，并返回给 RP iframe。


首先打开 node-oidc-provider 的 sessionManangement 功能，编辑 ./support/configuration.js 文件，在 42 行附近，进行以下修改：

```yaml
features: {
  sessionManagement: {
    enabled: true,
    keepHeaders: false,
  },
},
```


然后和 app1.html、app2.html 平级新建一个 rp.html 文件，并加入以下内容：

```java
<script>
  var stat = 'unchanged';
  var url = new URL(window.parent.location);
  // 这里的 '1' 是我们的 client_id，之前在 node-oidc-provider 中填写的
  var mes = '1' + ' ' + url.searchParams.get('session_state');
  console.log('mes: ')
  console.log(mes)
  function check_session() {
    var targetOrigin = 'http://localhost:3000';
    var win = window.parent.document.getElementById('op').contentWindow;
    win.postMessage(mes, targetOrigin);
  }

  function setTimer() {
    check_session();
    timerID = setInterval('check_session()', 3 * 1000);
  }

  window.addEventListener('message', receiveMessage, false);
  setTimer()
  function receiveMessage(e) {
    console.log(e.data);
    var targetOrigin = 'http://localhost:3000';
    if (e.origin !== targetOrigin) {
      return;
    }
    stat = e.data;
    if (stat == 'changed') {
      console.log('should log out now!!');
    }
  }
</script>
```


在 app1.html 和 app2.html 中加入两个 iframe 标签：

<iframe src="rp.html" hidden></iframe>
<iframe src="http://localhost:3000/session/check" id="op" hidden></iframe>


使用 Ctrl + C 关闭我们的 node-oidc-provider 和 http-server，然后再次启动。访问 app1.html，打开浏览器控制台，会得到以下信息，这意味着，用户当前处于未登录状态，应该进行 App 自身会话的销毁等操作



然后我们点击「登录」，在 OP 完成登录之后，回调到 app1.html，此时用户变成了登录状态，注意地址栏多了一个参数：session_state，这个参数就是我们上文用于在代码中向 OP iframe 轮询时需要携带的参数。



现在我们试一试单点登出，对于 node-oidc-provider 包提供的 OIDC Provider，只需要前端访问 localhost:3000/session/end



收到来自 OP 的登出成功信息



我们转到 app1.html 看一下，此时控制台输出，用户已经登出，现在要执行会话销毁等操作了。



不想维护 App 1 与用户的登录状态、App 2 与用户的登录状态

如果不各自维护 App 1、App 2 与用户的登录状态，那么无法实现只退出 App 1 而不退出 App 2 这样的需求。所有的登录状态将会完全依赖用户与 OP 之间的登录状态，在效果上是：用户在 OP 一次登录，之后访问所有的应用，都不必再输入密码，实现单点登录；用户在 OP 登出，则在所有应用登出，实现单点登出。

使用 Authing 解决单点登录

以上就是一个完整的单点登录系统的轮廓，我们需要维护一份全体用户目录，进行用户注册、登录；我们需要自己搭建一个 OIDC Provider，并申请一个 OIDC Client；我们需要使用 code 换 token，token 换用户信息；我们需要在自己的应用中不断轮询 OP 的登录状态。

读到这里，你可能会觉得实现一套完整的单点登录系统十分繁琐，不仅要对 OIDC 协议非常熟悉，还要自己架设 OIDC Provider，并且需要自行处理应用、用户、OP 之间登录状态。有没有开箱即用的登录服务呢？Authing 能够提供云上的 OP，云上的用户目录和直观的控制台，能够轻松管理所有用户、完成对 OP 的配置。





Authing 对开发者十分友好，提供丰富的 SDK，进行快速集成。



如果你不想关心登录的细节，将 Authing 集成到你的系统必定能够大幅提升开发效率，能够将更多的精力集中到核心业务上。





---

# compare

usually session-based for web browser, token-based for app

Scalability
1. Session based authentication:
   - Because the sessions are stored in the server`s memory
   - scaling becomes an issue when there is a huge number of users using the system at once.
2. Token based authentication:
   - no issue with scaling
   - because token is stored on the client side.


Multiple Device
1. Session based authentication:
   - Cookies normally work on a single domain or subdomains and they are normally disabled by browser if they work cross-domain (3rd party cookies).
   - It poses issues when APIs are served from a different domain to mobile and web devices.
2. Token based authentication:
   - no issue with cookies as the JWT is included in the request header.



- JWT
  - the size is much bigger comparing with the session id stored in cookie
  - because JWT contains more user information.
- Care must be taken to ensure only the necessary information is included in JWT
- and sensitive information should be omitted to prevent XSS security attacks.



---


ref:
- [Session vs Token Based Authentication](https://medium.com/@sherryhsu/session-vs-token-based-authentication-11a6c5ac45e4)
- [HTTP authentication](https://developer.mozilla.org/en-US/docs/Web/HTTP/Authentication)
- [Session vs Token-Based Authentication](https://medium.com/@allwinraju/session-vs-token-based-authentication-b1f862dd7ed8)
- [Difference between cookies, session and tokens](https://www.youtube.com/watch?v=44c1t_cKylo&ab_channel=ValentinDespa)
- [Authentication Types Ethical Hackers Academy || Cyber Security News](https://www.linkedin.com/posts/ethical-hackers-academy_authentication-types-ethical-hackers-academy-activity-6710268783136796672-g5Fl)

- https://openid.net/specs/openid-connect-core-1_0.html
- https://www.jianshu.com/p/be7cc032a4e9
- https://demo.c2id.com/oidc-client/
- https://deepzz.com/post/what-is-oidc-protocol.html
