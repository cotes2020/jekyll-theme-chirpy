---
title: Web Authentication - Concept Explained
# author: Grace JyL
date: 2020-09-23 11:11:11 -0400
description: 
excerpt_separator: 
categories: [Web, Authentication]
tags: [Authentication]
math: true
# pin: true
toc: true
# image: /assets/img/sample/devices-mockup.png
---

# Web Authentication - Concept Explained

[toc]

---

## auth

### Cookies
- When a server receives an `HTTP request` in the response, it can send a `Set-Cookie` header.
- The browser puts it into a cookie jar, and the cookie will be sent along with every request made to the same origin in the `Cookie HTTP header`.

To use cookies for authentication purposes, there are a few key principles that one must follow.
Always **use `HttpOnly` cookies**
- To mitigate the possibility of XSS attacks always
- use the `HttpOnly` flag when setting cookies.
- This way they won't show up in `document.cookies`.

Always use **signed cookies**
- With signed cookies, a server can tell if a cookie was modified by the client.

This can be observed in Chrome as well
- how a server set cookies:

![illustration of Chrome cookie set for web authentication purposes](https://i.imgur.com/VubG0Xs.png)

- Later on, all the requests use the cookies set for the given domain:

![web authentication method illustration Chrome cookie usage](https://i.imgur.com/cEM2xQw.png)


The cons:
1. Need to make extra effort to mitigate CSRF attacks
2. Incompatibility with REST - as it introduces a state into a stateless protocol

---

### Tokens
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

```
curl --header "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4gRG9lIiwiYWRtaW4iOnRydWV9.TJVA95OrM7E2cBab30RMHrHDcEfxjoYZgeFONFh7HgQ" my-website.com  
```

As the previous ones, the tokens can be observed in Chrome as well:


![google_chrome_json_web_token_as_a_web_authentication_method-1448359326265](https://i.imgur.com/f6P96PY.png)
Google Chrome JSON Web Token as a web authentication method

when writing APIs for native mobile applications or SPAs, JWT can be a good fit.
to use JWT in the browser have to stored in either `LocalStorage` or `SessionStorage`, can lead to XSS attacks.

The cons:
- Need to make extra effort to mitigate XSS attacks


### Signatures

> Either using cookies or tokens, if the transport layer for whatever reason gets exposed, credentials are easy to access - and with a token or cookie the attacker can act like the real user.

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

![aws_request_signing_flow_of_a_web_authentication_method-1448359478268](https://i.imgur.com/fLUPjQr.png)

go through all these steps, even if the transport layer gets compromised, an attacker can only read your traffic, won't be able to act as a user, as the attacker will not be able to sign requests
- as the private key is not in his/her possession.
- Most AWS services are using this kind of authentication.

`node-http-signature` deals with HTTP Request Signing and worth checking out.

The cons:
- annot use in the browser / client, only between APIs
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


## pick?

- to support a web application only
  - either cookies or tokens are fine
  - for cookies think about XSRF, for JWT take care of XSS.

- to support both a web application and a mobile client
  - go with an API that supports `token-based authentication`.

- If building APIs that communicate with each other
  - go with `request signing`.


ref:
[Web Authentication Methods Explained](https://blog.risingstack.com/web-authentication-methods-explained/)
