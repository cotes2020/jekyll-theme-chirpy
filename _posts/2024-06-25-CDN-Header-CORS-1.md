---
title: CORS Header
author: Leesh
categories: [CDN, CORS Header]
tags: [CDN, 컨텐츠 전송 네트워크, 미디어/웹 캐시, WAAP, Cloud, HTTP Header, CORS]
date: '2024-06-25 15:48:00 +0900'
---

## CORS(Cross-Origin Resource Sharing)

---
CORS는 웹 브라우저가 현재 실행 중인 도메인의 웹 애플리케이션이<br>
다른 도메인에서 제공하는 리소스에 접근할 수 있는 권한을 부여하도록 하는 메커니즘입니다.<br>
이는 웹 애플리케이션이 외부 리소스를 안전하게 요청하고 사용할 수 있도록 도와줍니다.


## CORS가 중요한 이유

---
### 동일 오리진 정책 (Same-Origin Policy)

기본적으로 웹 브라우저는 보안상의 이유로 동일 오리진 정책(Same-Origin Policy)을 따르며,<br>
이 정책은 실행되는 웹 애플리케이션이 다른 도메인의 리소스에 접근하는 것을 제한합니다.<br>

> `https://domain.com`에서 로드된 스크립트는 기본적으로 `https://another-domain.com`의 리소스에 접근할 수 없습니다.\
> 악의적인 스크립트가 민감한 데이터를 훔치는 것을 방지하는 중요한 보안 기능입니다.

CORS는 이러한 제한을 컨트롤하여, 특정 조건 하에서 다른 도메인의 API나 리소스에 접근할 수 있도록 합니다. 



## CORS 동작 방식

---
CORS는 서버와 브라우저 간의 HTTP 헤더를 사용하여 동작합니다.<br>
브라우저가 다른 도메인의 리소스를 요청할 때, 요청 헤더에 CORS 관련 정보를 포함합니다.<br>
서버는 이러한 헤더를 확인하고 요청이 허용될지 여부를 결정합니다.<br>
서버가 이러한 요청을 허용하면, 응답 헤더에 해당 정보를 포함하여 브라우저에 전달합니다.<br>

> 1. 클라이언트가 `https://api.example.com`에 요청을 보냅니다.
> 2. 브라우저는 이 요청에 `CORS 요청 헤더`를 추가합니다.
> 3. 서버는 요청을 받고, CORS 설정을 확인하여 요청을 허용할지 결정합니다.
> 4. 허용될 경우, 서버는 `응답 헤더에 CORS 허용 정보`를 포함하여 응답을 반환합니다.
> 5. 브라우저는 응답을 받고 리소스를 사용할 수 있게 됩니다.


### CORS 사전 요청(Preflight)

사전 요청(Preflight Request)은 실제 요청이 수행되기 전에 브라우저가 서버에 보내는 요청입니다.<br>
사전 요청을 통해 브라우저는 서버가 실제 요청을 허용할지 여부를 확인합니다.

> 1. 클라이언트는 **OPTIONS 메서드**를 사용하여 사전 요청을 보냅니다.
> 2. 서버는 사전 요청을 받고, 허용 여부를 결정한 후 응답합니다.
> 3. 브라우저는 서버의 응답을 확인하고, 요청이 허용되면 **실제 요청**을 보냅니다.


## CORS 주요 응답 헤더

---
### Access-Control-Allow-Headers

서버가 허용하는 요청 헤더 목록을 지정합니다.<br>
브라우저가 요청에 포함할 수 있는 헤더를 정의하여 알려줍니다.

```
Access-Control-Allow-Headers: Content-Type, Authorization
```

### Access-Control-Allow-Methods

서버가 허용하는 HTTP 메서드 목록을 지정합니다.<br>
브라우저가 사용할 수 있는 메서드를 정의하여 알려줍니다.

```
Access-Control-Allow-Methods: GET, POST, PUT, DELETE
```

### Access-Control-Allow-Origin

허용하는 오리진(도메인)을 지정합니다. 특정 도메인이나 모든 도메인을 허용할 수 있습니다.

```
> 특정 도메인 허용:
Access-Control-Allow-Origin: https://example.com

> 여러 도메인 허용:
Access-Control-Allow-Origin: https://example.com, https://another-domain.com

> 모든 오리진을 허용:
Access-Control-Allow-Origin: *
```

### Access-Control-Max-Age

사전 요청의 응답을 캐시할 수 있는 시간을 초 단위로 지정합니다.<br>
이 시간 동안 동일한 요청에 대해 사전 요청을 보내지 않습니다.

```
Access-Control-Max-Age: 3600
```

### Access-Control-Expose-Headers

브라우저가 서버 응답에서 접근할 수 있는 헤더 목록을 지정합니다.<br>
기본적으로 몇 가지 안전한 헤더만`(CORS-safelisted response header)` 접근할 수 있기 때문에, 이 외의 헤더에 접근하기 위해 사용합니다.

```
Access-Control-Expose-Headers: Authorization, Content-Length, X-Forwarded-For
```

#### CORS-safelisted response header

브라우저가 기본적으로 접근할 수 있는 응답 헤더 목록입니다.<br>
이러한 기본 헤더 외의 헤더에 접근하려면 Access-Control-Expose-Headers를 사용해야 합니다.

기본적으로 접근 가능한 응답 헤더:

```
- Cache-Control
- Content-Language
- Content-Type
- Expires
- Last-Modified
- Pragma
```

## 결론

---
CORS는 현대 웹 애플리케이션 개발에서 매우 중요한 역할을 합니다.<br>
보안을 유지하면서도 다른 도메인의 리소스에 접근할 수 있게 해주기 때문입니다.<br>
이를 이해하고 적절하게 설정함으로써 더 안전하게 웹 애플리케이션을 만들 수 있습니다.<br>
CORS 문제를 해결하기 위해서는 서버 측에서 올바른 헤더를 설정하는 것이 주효합니다.

## 참고

---
* [AWS Docs : CORS란 무엇인가요?](https://aws.amazon.com/ko/what-is/cross-origin-resource-sharing/)
* [MDN Web Docs : CORS Header](https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/Access-Control-Expose-Headers)
