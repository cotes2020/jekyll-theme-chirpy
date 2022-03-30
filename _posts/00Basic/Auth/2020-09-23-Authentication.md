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
  - [different auth](#different-auth)
    - [Cookies](#cookies)
    - [Signatures](#signatures)
- [Authentication](#authentication)
  - [HTTP authentication](#http-authentication)
    - [basic](#basic)
    - [Authentication Type/schemes](#authentication-typeschemes)
    - [`Basic authentication scheme`](#basic-authentication-scheme)
    - [step:](#step)
    - [drawbacks](#drawbacks)
    - [Security of basic authentication](#security-of-basic-authentication)
    - [Proxy authentication](#proxy-authentication)
    - [Authentication of cross-origin images](#authentication-of-cross-origin-images)
  - [session-based authentication](#session-based-authentication)
    - [step](#step-1)
  - [Token-based authentication](#token-based-authentication)
    - [characteristics of token](#characteristics-of-token)
      - [token and password](#token-and-password)
    - [token based authentication strategies](#token-based-authentication-strategies)
      - [JWT, JSON web tokens](#jwt-json-web-tokens)
      - [OAuth - Open Authorization](#oauth---open-authorization)
        - [authorization grant](#authorization-grant)
          - [第一种授权方式：授权码 `AUTHORIZATION_CODE` -> token](#第一种授权方式授权码-authorization_code---token)
          - [第二种方式：隐藏式 `implicit` -> token](#第二种方式隐藏式-implicit---token)
          - [第三种方式：密码式 `password` -> token](#第三种方式密码式-password---token)
          - [第四种方式：凭证式 `client credentials` -> token](#第四种方式凭证式-client-credentials---token)
          - [令牌的使用](#令牌的使用)
          - [**更新令牌** `refresh_token` -> token](#更新令牌-refresh_token---token)
        - [OAuth2 Proxy](#oauth2-proxy)
        - [oauth2 proxy with Github](#oauth2-proxy-with-github)
          - [oauth2-proxy.cfg](#oauth2-proxycfg)
          - [upstreams.ymal](#upstreamsymal)
        - [基于k8s部署的nginx服务 通过ingress和oauth2 proxy对接gitlab](#基于k8s部署的nginx服务-通过ingress和oauth2-proxy对接gitlab)
          - [在Gitlab配置**OpenID应用**](#在gitlab配置openid应用)
          - [生成**Cookie密钥**](#生成cookie密钥)
          - [部署**oauth2-proxy**](#部署oauth2-proxy)
          - [创建测试应用并配置Ingress](#创建测试应用并配置ingress)
          - [测试外部认证](#测试外部认证)
          - [流程分析](#流程分析)
- [compare](#compare)



- [Web Authentication Methods Explained](https://blog.risingstack.com/web-authentication-methods-explained/)





---


# Basic - Authentication

---

## different auth

- to support a web application only
  - either `cookies` or `tokens` are fine
  - for cookies think about XSRF,
  - for JWT take care of XSS.

- to support both a web application and a mobile client
  - go with an API that supports `token-based authentication`.

- If building APIs that communicate with each other
  - go with `request signing`.

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






2. Tokens



---

### Signatures

> cookies or tokens,
> if the transport layer for whatever reason gets exposed, credentials are easy to acces, the attacker can act like the real user.

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


# Authentication

1. HTTP authentication: `username and passwd` are sent in each request
2. Session based authentication: `session id` are sent in each request
3. Token based authentication: `token` are sent in each request


---


## HTTP authentication

### basic

server:
- `WWW-Authenticate` and `Proxy-Authenticate` headers
- The `WWW-Authenticate` and `Proxy-Authenticate` response headers **define the authentication method that should be used** to gain access to a resource.
- must specify which authentication scheme is used, so the client knows how to provide the credentials.

The syntax for these headers:

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
- `Authorization` and `Proxy-Authorization` equest headers
- contain the credentials to authenticate a user agent with a (proxy) server
- `<type>` is needed again
- `credentials`: be encoded or encrypted depending on which authentication scheme is used.

```html
Authorization: <type> 12345678
Proxy-Authorization: <type> 123456
```

---

### Authentication Type/schemes
The general HTTP authentication framework is used by several authentication schemes.
- Schemes can differ in security strength and in their availability in client or server software.
- there are other schemes offered by host services, such as Amazon AWS.

Schemes | Note
---|---
`Basic` | RFC 7617, **base64-encoded** credentials.
`Bearer` | See RFC 6750, bearer tokens to access OAuth 2.0-protected resources
`Digest` | See RFC 7616, only md5 hashing is supported in Firefox, see bug 472823 for SHA encryption support
`HOBA` | See RFC 7486, Section 3, HTTP Origin-Bound Authentication, digital-signature-based
`Mutual` | See RFC 8120
`AWS4-HMAC-SHA256` | See AWS docs

---


### `Basic authentication scheme`

![Basic authentication](https://i.imgur.com/W4lnwIN.png)


- transmits credentials as user ID/password pairs, encoded using `base64`.

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
     - `200 OK`: exist and vaild
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

### Proxy authentication

The same challenge and response mechanism can be used for proxy authentication.

- As both resource authentication and proxy authentication can coexist
- but different set of `headers` and `status codes` is needed.

1. the challenging status code is `407 (Proxy Authentication Required)`
   - the `Proxy-Authenticate` response header
     - contains at least one challenge applicable to the proxy
     - used for providing the credentials to the proxy server.


### Authentication of cross-origin images
security hole recently been fixed by browsers is `authentication of cross-site images`.
- From Firefox 59
- **image resources loaded from different origins to the current document** are no longer able to trigger `HTTP authentication dialogs` (bug 1423146),
- preventing user credentials being stolen if attackers were able to embed an arbitrary image into a third-party page.


---

## session-based authentication

![session-base](https://i.imgur.com/9PNfrgQ.png)

client sends the **seesion id** in all the request, and server uses it to identify the user.

- a stateful authentication method
- as the server need it to be implemented

---

### step

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

4. slient send request with session id
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


## Token-based authentication

![token-based](https://i.imgur.com/HKxHiDr.png)

token

- the `user state is stored inside the token` **on the client side**.

- token is a normal URL-safe string
  - can be pass in header, body, or URL
  - most of the time sent by HTTP headers, not cookies
    - not only for browser, but also app.
  - the preferred mode of authentication for `RESTful APIs`.

- token is `self-contained`
  - containes session infor and user info as well
  - carries the data

- anyone can view the contained

- as a standard

- use when interact between multible untrsuted parties:
  - bank - app - user

- have lifetime

- can grant access to only a subset of data
  - one `token` for one `API`
  - just give specific right: make transaction, just search ....


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
    - Registered claims
    - ![Registered claims](https://i.imgur.com/KHBHULL.png)
    - Public claims
      - claims to defines and use for our own data
      - e.g. userid, email...
    - Private claims
      - names withour meaning except the consumer and producer of tokens
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

![Screen Shot 2020-09-24 at 00.42.22](https://i.imgur.com/c4WAwrS.png)

![1_PDry-Wb8JRquwnikIbJOJQ](https://i.imgur.com/jtgGEVO.png)


---


### characteristics of token

1. random string
2. server does not store it (stateless)
3. has an expiry, then token is useless
4. normally sighed with a secret so to identify any tampering and thus can be trusted by the server
5. normally sent in the authorization header
6. can be `Opaque` or `Self-contained`
   - `Opaque`
     - random string, no meaning
     - can only be verfied by the autorization server
     - just like session ids
   - `Self-contained`
     - token has the data and can be reviewd by the clients
     - e.g. JWT tokens


#### token and password

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
- OAuth 2.0 对于如何颁发令牌的细节，规定得非常详细。具体来说，一共分成四种授权类型（authorization grant），即四种颁发令牌的方式，适用于不同的互联网场景。

---

### token based authentication strategies

emaple of token based authentication strategies
- SWT, simple web tokenss
- JWT, JSON web tokens
- OAuth, open authorization
- SAML, security assertions markup language
- OpenID



---


#### JWT, JSON web tokens
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

app.use( jwt({secret: 'very-secret'}) );

// Protected middleware
app.use(function *(){
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





#### OAuth - Open Authorization

- ref:
  - [https://luvletter.cn/blog/使用oauth2-proxy保护你的应用/](https://luvletter.cn/blog/使用oauth2-proxy保护你的应用/)
  - [http://www.ruanyifeng.com/blog/2019/04/oauth_design.html](http://www.ruanyifeng.com/blog/2019/04/oauth_design.html)
  - https://oauth.net/2/
  - https://kubernetes.github.io/ingress-nginx/examples/auth/oauth-external-auth/
  - https://oauth2-proxy.github.io/oauth2-proxy/docs/
  - [https://energygreek.github.io/2020/07/23/oauth2/](https://energygreek.github.io/2020/07/23/oauth2/)


> OAuth 引入了一个授权层，用来分离两种不同的角色：客户端和资源所有者。
> 资源所有者同意以后，资源服务器可以向客户端颁发令牌。客户端通过令牌，去请求数据。


**OAuth**
- OAuth是一种 <font color=red> 授权机制 </font>
- **数据的所有者**告诉系统，同意**授权第三方应用**进入系统，获取这些数据。
- **系统**从而产生一个短期的`进入令牌token`，用来代替密码，供**第三方应用**使用。

- allow users to share their private resources to a third party
  - allow some app log you in using twitter
    - expose your twitter info to an external app using `twitter's OAuth server`
  - authorizing you app's frontend from your API
    - your `custom OAuth server`: where user of your website get authorized using `OAuth`

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

##### authorization grant

**OAuth 的核心就是向第三方应用颁发令牌**
- 由于互联网有多种场景, 本标准定义了获得令牌的四种授权方式（authorization grant）。
- OAuth 2.0 规定了四种获得令牌的流程 向第三方应用颁发令牌。
  - 授权码（authorization-code）
  - 隐藏式（implicit）
  - 密码式（password）：
  - 客户端凭证（client credentials）

注意，不管哪一种授权方式，第三方应用申请令牌之前，都必须先到系统备案，说明自己的身份，然后会拿到两个身份识别码：客户端 ID（client ID）和客户端密钥（client secret）。这是为了防止令牌被滥用，没有备案过的第三方应用，是不会拿到令牌的。


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



---

###### 第一种授权方式：授权码 `AUTHORIZATION_CODE` -> token

<font color=red>第三方应用先申请一个授权码，然后再用该码获取令牌</font>

这种方式是最常用的流程，安全性也最高，它适用于那些有后端的 Web 应用。授权码通过前端传送，令牌则是储存在后端，而且所有与资源服务器的通信都在后端完成。这样的前后端分离，可以避免令牌泄漏。

![pi](https://www.wangbase.com/blogimg/asset/201904/bg2019040905.jpg)

```bash
# A 网站提供一个链接，用户点击后就会跳转到 B 网站，授权用户数据给 A 网站使用。
https://b.com/oauth/authorize?
  response_type=code&
  client_id=CLIENT_ID&
  redirect_uri=CALLBACK_URL&
  scope=read

# 用户跳转后，B 网站会要求用户登录，
# 登录后询问是否同意给予 A 网站授权。
# 表示同意，这时 B 网站就会跳回指定的网址 https://a.com/callback
# 跳转时，会传回一个授权码 ?code=AUTHORIZATION_CODE
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
     - `response_type`参数表示要求返回授权码（`code`），
     - `client_id`参数让 B 知道是谁在请求，
     - `redirect_uri`参数是 B 接受或拒绝请求后的跳转网址，
     - `scope`参数表示要求的授权范围（这里是只读）。

2. 用户跳转后，B 网站会要求用户登录，然后询问是否同意给予 A 网站授权。
   - 用户表示同意，这时 B 网站就会跳回`redirect_uri`参数指定的网址。
   - 跳转时，会传回一个授权码，就像下面这样。
   - `https://a.com/callback?code=AUTHORIZATION_CODE`
   - 上面 URL 中，`code`参数就是授权码。


3. A 网站拿到授权码以后，就可以在后端，向 B 网站请求令牌。
   - `https://b.com/oauth/token?client_id=CLIENT_ID&client_secret=CLIENT_SECRET&grant_type=authorization_code&code=AUTHORIZATION_CODE&redirect_uri=CALLBACK_URL`
   - 上面 URL 中
     - `client_id`参数和`client_secret`参数用来让 B 确认 A 的身份（`client_secret`参数是保密的，因此只能在后端发请求）
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



###### 第二种方式：隐藏式 `implicit` -> token


> 有些 Web 应用是纯前端应用，没有后端。这时就不能用上面的方式了，必须将令牌储存在前端。
> **RFC 6749 就规定了第二种方式，允许直接向前端颁发令牌。这种方式没有授权码这个中间步骤，所以称为（授权码）"隐藏式"（implicit）**

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


- 注意，令牌的位置是 URL 锚点（fragment），而不是查询字符串（querystring），这是因为 OAuth 2.0 允许跳转网址是 HTTP 协议，因此存在"中间人攻击"的风险，而浏览器跳转时，锚点不会发到服务器，就减少了泄漏令牌的风险。

![pi](https://www.wangbase.com/blogimg/asset/201904/bg2019040906.jpg)

- 这种方式把令牌直接传给前端，是很不安全的。因此，只能用于一些安全要求不高的场景
- 并且令牌的有效期必须非常短，通常就是会话期间（session）有效，浏览器关掉，令牌就失效了。

---

###### 第三种方式：密码式 `password` -> token


**如果你高度信任某个应用，RFC 6749 也允许用户把用户名和密码，直接告诉该应用。该应用就使用你的密码，申请令牌，这种方式称为"密码式"（password）。**


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

###### 第四种方式：凭证式 `client credentials` -> token

**最后一种方式是凭证式（client credentials），适用于没有前端的命令行应用，即在命令行下请求令牌。**

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

###### 令牌的使用

**令牌的使用**
- A 网站拿到令牌以后，就可以向 B 网站的 API 请求数据了。
- 每个发到 API 的请求，都必须带有令牌。
- 具体做法是在请求的头信息，加上一个`Authorization`字段，令牌就放在这个字段里面。

```
curl -H "Authorization: Bearer ACCESS_TOKEN" "https://api.b.com"
```

上面命令中，`ACCESS_TOKEN`就是拿到的令牌。

---

###### **更新令牌** `refresh_token` -> token
- 令牌的有效期到了，如果让用户重新走一遍上面的流程，再申请一个新的令牌，很可能体验不好，而且也没有必要。
- OAuth 2.0 允许用户自动更新令牌。

```bash
# B 网站颁发令牌的时候，一次性颁发两个令牌
# 一个用于获取数据，另一个用于获取新的令牌（refresh token 字段）。
# 令牌到期前，用户使用 refresh token 发一个请求，去更新令牌。

https://b.com/oauth/token?
  grant_type=refresh_token&
  client_id=CLIENT_ID&
  client_secret=CLIENT_SECRET&
  refresh_token=REFRESH_TOKEN

# B 网站验证通过以后，就会颁发新的令牌。
```


---

##### OAuth2 Proxy

- 一个使用go编写的反向代理和静态文件服务器
- 使用提供程序（Google，GitHub和其他提供商）提供身份验证，以通过电子邮件，域或组验证帐户。

![Screen Shot 2022-03-23 at 11.13.30](https://i.imgur.com/G7dpktE.png)

---

##### oauth2 proxy with Github

1. 先去github -> developer 创建oauth应用， 输入自己的回调地址。 当用户被github认证后，会调用这个地址
2. 在服务端配置，利用一个开源 oauth2_proxy 工具, 项目地址：`https://github.com/oauth2-proxy/oauth2-proxy`
3. 配置 nginx




###### oauth2-proxy.cfg

```yaml
auth_logging = true
auth_logging_format = "{{.Client}} - {{.Username}} [{{.Timestamp}}] [{{.Status}}] {{.Message}}"
## pass HTTP Basic Auth, X-Forwarded-User and X-Forwarded-Email information to upstream
pass_basic_auth = true
# pass_user_headers = true
## pass the request Host Header to upstream
## when disabled the upstream Host is used as the Host Header
pass_host_header = true


## 可以通过验证的邮箱域名
## Email Domains to allow authentication for (this authorizes any email on this domain)
## for more granular authorization use `authenticated_emails_file`
## To authorize any email addresses use "*"
# email_domains = [
#     "yourcompany.com"
# ]
email_domains=["*"]

## callback的域名
whitelist_domains = [".example.com"]
cookie_domains = ["example.com"]
skip_auth_preflight = false


## Cookie Settings
## Name     - the cookie name
## Secret   - the seed string for secure cookies; should be 16, 24, or 32 bytes
##            for use with an AES cipher when cookie_refresh or pass_access_token
##            is set
## Domain   - (optional) cookie domain to force cookies to (ie: .yourcompany.com)
## Expire   - (duration) expire timeframe for cookie
## Refresh  - (duration) refresh the cookie when duration has elapsed after cookie was initially set.
##            Should be less than cookie_expire; set to 0 to disable.
##            On refresh, OAuth token is re-validated.
##            (ie: 1h means tokens are refreshed on request 1hr+ after it was set)
## Secure   - secure cookies are only sent by the browser of a HTTPS connection (recommended)
## HttpOnly - httponly cookies are not readable by javascript (recommended)
# cookie_name = "_oauth2_proxy"
## cookie加密密钥
cookie_secret = "beautyfly"
cookie_domains = "beautyflying.cn"
cookie_expire = "168h"
# cookie_refresh = ""
cookie_secure = false
# cookie_httponly = true



http_address="0.0.0.0:4180"
## 与GitHub callback URL一致
## The OAuth Client ID, Secret
redirect_url="https://example.com/oauth2/callback"
provider="github"
## 刚刚创建的GitHub OAuth Apps里有
client_id = "cef54714c84e3b0c2248"
client_secret = "a96d3d94771273b5295202d03c0c2d3ca7f625dc"
## Pass OAuth Access token to upstream via "X-Forwarded-Access-Token"
pass_access_token = false
## Authenticated Email Addresses File (one email per line)
# authenticated_emails_file = ""
## Htpasswd File (optional)
## Additionally authenticate against a htpasswd file. Entries must be created with "htpasswd -s" for SHA encryption
## enabling exposes a username/login signin form
# htpasswd_file = ""
## Templates
## optional directory with custom sign_in.html and error.html
# custom_templates_dir = ""
## skip SSL checking for HTTPS requests
# ssl_insecure_skip_verify = false


## 限制登录用户
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

```bash
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

---

###### upstreams.ymal

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



实验环境：
- k8s 1.15.0
- Ingress nginx 0.25.0
- gitlab 13.7.4

---

###### 在Gitlab配置**OpenID应用**

- 登录到Gitlab—>管理中心—>应用，创建一个应用
  - 参数：
    - **Authorization callback URL** 回调URL：
      - 指GitLab在用户通过身份验证后应将其发送到的端点
      - 填入oauth2-proxy的callback地址
      - 对于oauth2-proxy应该是`https://<应用域名>/oauth2/callback`
    - 范围：
      - 应用程序对GitLab用户配置文件的访问级别。
      - 对于大多数应用程序，选择openid，profile和email即可。
  - 创建完应用后，会生成`一对ID和密钥`，这个在后面会用到。

---

###### 生成**Cookie密钥**
- 生成**Cookie密钥**
  - 该Cookie密钥作为`种子字符串`以产生安全的cookie。
  - 使用base64编码，可利用以下的python脚`本生成字符串。

```py
import secrets
import base64
print(base64.b64encode(base64.b64encode(secrets.token_bytes(16))))
```

---


###### 部署**oauth2-proxy**

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
        # 设置安全（仅HTTPS）cookie标志
        - --cookie-secure=false
        # OAuth重定向URL
        - --redirect-url=https://nginx-test.ssgeek.com/oauth2/callback
        # 跳过登录页面直接进入下一步
        - --skip-provider-button=false
        # 设置X-Auth-Request-User，X-Auth-Request-Email和X-Auth-Request-Preferred-Username响应头（在Nginx auth_request模式下有用）。与结合使用时--pass-access-token，会将X-Auth-Request-Access-Token添加到响应标头中
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

###### 创建测试应用并配置Ingress

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



###### 测试外部认证

通过访问上面部署的nginx应用，在浏览器中进行测试，会被重定向到Gitlab登录页面；

输入账号，正确登录后，会被重定向回nginx应用。

![05zwoff77t](https://i.imgur.com/Xb90yDQ.gif)

---

###### 流程分析

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

# compare

usually seesion-based for web browser, token-based for app

Scalability
1. Session based authentication:
   - Because the sessions are stored in the server’s memory
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






ref:
- [Session vs Token Based Authentication](https://medium.com/@sherryhsu/session-vs-token-based-authentication-11a6c5ac45e4)
- [HTTP authentication](https://developer.mozilla.org/en-US/docs/Web/HTTP/Authentication)
- [Session vs Token-Based Authentication](https://medium.com/@allwinraju/session-vs-token-based-authentication-b1f862dd7ed8)
- [Difference between cookies, session and tokens](https://www.youtube.com/watch?v=44c1t_cKylo&ab_channel=ValentinDespa)
- [Authentication Types Ethical Hackers Academy || Cyber Security News](https://www.linkedin.com/posts/ethical-hackers-academy_authentication-types-ethical-hackers-academy-activity-6710268783136796672-g5Fl)



.
