---
title: HTTP Header
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

- [HTTP Header](#http-header)
  - [Pragma](#pragma)
  - [`WWW-Authenticate` response header](#www-authenticate-response-header)


---

# HTTP Header


---

## Pragma

The `Pragma HTTP/1.0` general header is an implementation-specific header that may have various effects along the request-response chain.

It is used for backwards compatibility with HTTP/1.0 caches where the `Cache-Control HTTP/1.1` header is not yet present.

Pragma is not specified for HTTP responses and not a reliable replacement for the general `HTTP/1.1 Cache-Control` header
- although it does behave the same as `Cache-Control`: `no-cache`, if the` Cache-Control` header field is omitted in a request.
- Use Pragma only for backwards compatibility with HTTP/1.0 clients.


**Syntax**

`Pragma: no-cache`
- `no-cache`: Forces caches to submit the request to the origin server for validation before releasing a cached copy.



---

## `WWW-Authenticate` response header
- `WWW-Authenticate`
  - Header type	Response `header`
  - Forbidden header name	`no`
- defines the authentication method that should be used to gain access to a resource.
- sent along with a `401 Unauthorized response`.

**Syntax**

`WWW-Authenticate: <type> realm=<realm>[, charset="UTF-8"]`

**Directives**

Directives | Note
---|---
`<type>` | Authentication type. A common type is "Basic". IANA maintains a list of Authentication schemes.
`realm=<realm>` | A description of the protected area. <br> If no realm is specified, clients often display a formatted hostname instead.
`charset=<charset>` | Tells the client the server's prefered encoding scheme when submitting a username and password. The only allowed value is the case insensitive string "UTF-8". This does not relate to the encoding of the realm string.

**Examples**
- server response contains a WWW-Authenticate header

```
WWW-Authenticate: Basic
WWW-Authenticate: Basic realm="Access to the staging site", charset="UTF-8"
```
