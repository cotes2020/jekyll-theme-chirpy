---
layout: post
title: "Go 响应http请求"
date: 2019-10-01 23:56:00.000000000 +09:00
categories: [Go Web]
tags: [Go Web]
---

在Web应用程序中，每个HTTP事务都由请求(Request)和响应(Response)构成，这次我们讲讲Go如何处理Web中的数据响应。

## Web数据响应

Web的响应与请求结构是类似的，响应分为三个部分：响应行、响应头部、响应体。

1. 响应行：协议、响应状态码和状态描述，如： HTTP/1.1 200 OK
2. 响应头部：包含各种头部字段信息，如cookie，Content-Type等头部信息。
3. 响应体：携带客户端想要的数据，格式与编码由头部的Content-Type决定。

> **响应状态码的有固定取值和意义：**
>
> - 100~199：表示服务端成功客户端接收请求，要求客户端继续提交下一次请求才能完成整个处理过程。
> - 200~299：表示服务端成功接收请求并已完成整个处理过程。最常用就是：200
> - 300~399：为完成请求，客户端需进一步细化请求。比较常用的如：客户端请求的资源已经移动一个新地址使用302表示将资源重定向,客户端请求的资源未发生改变，使用304，告诉客户端从本地缓存中获取。
> - 400~499：客户端的请求有错误，如：404表示你请求的资源在web服务器中找不到，403表示服务器拒绝客户端的访问，一般是权限不够。
> - 500~599：服务器端出现错误，最常用的是：500

## Go处理Web数据响应

Go将http响应封装在http.ResponseWriter结构体中，ResponseWriter的定义很简单，一共只有三个方法。

**ResponseWriter**

在net/http源码包中，http.ResponseWriter的结构信息定义如下所示：

```go
type ResponseWriter interface {
    Header() Header            //头部
    Write([]byte) (int, error) //写入方法
    WriteHeader(statusCode int)//状态码
}
```

**Header方法**

header方法返回http.Header结构体，用于设置响应头部信息，http.Header数据类型map，定义如下：

```go
type Header map[string][]string
```

http.Header的方法列表如下：

```go
type Header
    func (h Header) Add(key, value string)
    func (h Header) Del(key string)
    func (h Header) Get(key string) string
    func (h Header) Set(key, value string)
    func (h Header) Write(w io.Writer) error
    func (h Header) WriteSubset(w io.Writer, exclude map[string]bool) error
```

**Writer()方法**

Write方法定义如下，用于向客户端返回数据流，与io.Writer中的write方法的定义一致，是Go语言io流中标准方法。

```go
Write([]byte) (int, error)
```

**WriterHeader方法**

writerHeader方法的定义如下所示：

```go
WriteHeader(statusCode int)
```

参数statusCode表示响应码，其取值可以为http包中已经定义好的常量值：

```go
const (
    StatusContinue           = 100 // RFC 7231, 6.2.1
    StatusSwitchingProtocols = 101 // RFC 7231, 6.2.2
    StatusProcessing         = 102 // RFC 2518, 10.1

    StatusOK                   = 200 // RFC 7231, 6.3.1
    StatusCreated              = 201 // RFC 7231, 6.3.2
    StatusAccepted             = 202 // RFC 7231, 6.3.3
    StatusNonAuthoritativeInfo = 203 // RFC 7231, 6.3.4
    StatusNoContent            = 204 // RFC 7231, 6.3.5
    StatusResetContent         = 205 // RFC 7231, 6.3.6
    StatusPartialContent       = 206 // RFC 7233, 4.1
    StatusMultiStatus          = 207 // RFC 4918, 11.1
    StatusAlreadyReported      = 208 // RFC 5842, 7.1
    StatusIMUsed               = 226 // RFC 3229, 10.4.1

    StatusMultipleChoices  = 300 // RFC 7231, 6.4.1
    StatusMovedPermanently = 301 // RFC 7231, 6.4.2
    StatusFound            = 302 // RFC 7231, 6.4.3
    StatusSeeOther         = 303 // RFC 7231, 6.4.4
    StatusNotModified      = 304 // RFC 7232, 4.1
    StatusUseProxy         = 305 // RFC 7231, 6.4.5

    StatusTemporaryRedirect = 307 // RFC 7231, 6.4.7
    StatusPermanentRedirect = 308 // RFC 7538, 3

    StatusBadRequest                   = 400 // RFC 7231, 6.5.1
    StatusUnauthorized                 = 401 // RFC 7235, 3.1
    StatusPaymentRequired              = 402 // RFC 7231, 6.5.2
    StatusForbidden                    = 403 // RFC 7231, 6.5.3
    StatusNotFound                     = 404 // RFC 7231, 6.5.4
    StatusMethodNotAllowed             = 405 // RFC 7231, 6.5.5
    StatusNotAcceptable                = 406 // RFC 7231, 6.5.6
    StatusProxyAuthRequired            = 407 // RFC 7235, 3.2
    StatusRequestTimeout               = 408 // RFC 7231, 6.5.7
    StatusConflict                     = 409 // RFC 7231, 6.5.8
    StatusGone                         = 410 // RFC 7231, 6.5.9
    StatusLengthRequired               = 411 // RFC 7231, 6.5.10
    StatusPreconditionFailed           = 412 // RFC 7232, 4.2
    StatusRequestEntityTooLarge        = 413 // RFC 7231, 6.5.11
    StatusRequestURITooLong            = 414 // RFC 7231, 6.5.12
    StatusUnsupportedMediaType         = 415 // RFC 7231, 6.5.13
    StatusRequestedRangeNotSatisfiable = 416 // RFC 7233, 4.4
    StatusExpectationFailed            = 417 // RFC 7231, 6.5.14
    StatusTeapot                       = 418 // RFC 7168, 2.3.3
    StatusMisdirectedRequest           = 421 // RFC 7540, 9.1.2
    StatusUnprocessableEntity          = 422 // RFC 4918, 11.2
    StatusLocked                       = 423 // RFC 4918, 11.3
    StatusFailedDependency             = 424 // RFC 4918, 11.4
    StatusTooEarly                     = 425 // RFC 8470, 5.2.
    StatusUpgradeRequired              = 426 // RFC 7231, 6.5.15
    StatusPreconditionRequired         = 428 // RFC 6585, 3
    StatusTooManyRequests              = 429 // RFC 6585, 4
    StatusRequestHeaderFieldsTooLarge  = 431 // RFC 6585, 5
    StatusUnavailableForLegalReasons   = 451 // RFC 7725, 3

    StatusInternalServerError           = 500 // RFC 7231, 6.6.1
    StatusNotImplemented                = 501 // RFC 7231, 6.6.2
    StatusBadGateway                    = 502 // RFC 7231, 6.6.3
    StatusServiceUnavailable            = 503 // RFC 7231, 6.6.4
    StatusGatewayTimeout                = 504 // RFC 7231, 6.6.5
    StatusHTTPVersionNotSupported       = 505 // RFC 7231, 6.6.6
    StatusVariantAlsoNegotiates         = 506 // RFC 2295, 8.1
    StatusInsufficientStorage           = 507 // RFC 4918, 11.5
    StatusLoopDetected                  = 508 // RFC 5842, 7.2
    StatusNotExtended                   = 510 // RFC 2774, 7
    StatusNetworkAuthenticationRequired = 511 // RFC 6585, 6
)
```

**示例**

```go
package main
import (
	"fmt"
	"net/http"
)
func main() {
    http.HandleFunc("/test", func(writer http.ResponseWriter, request *http.Request) {
        header := writer.Header()
        header.Add("Content-Type","application/json")
        writer.WriteHeader(http.StatusBadGateway)
        fmt.Fprintln(writer,"Web响应")
    })
    http.ListenAndServe(":8080",nil)
}
```

在浏览器控制台的Network查看运行结果：

![](/assets/images/2019Go/go-response-01.png)

**Cookie**

> 注意区分Cookie与Session之间的区别，Cookie用服务端保存某些信息到客户端的，用于识别用户，一种用户追踪机制，而Session则是服务端实现用户多次请求之间保持会话的机制；两者可以配合使用，也可以独立使用。

net/http包提供了SetCookie方法用于向客户端写入Cookie.

```go
func SetCookie(w ResponseWriter, cookie *Cookie)
```

第一个参数为ResponseWriter结构体，而第二个参数则Cookie结构体，其定义如下：

```go
type Cookie struct {
    Name  string
    Value string

    Path       string    // optional
    Domain     string    // optional
    Expires    time.Time // optional
    RawExpires string    // for reading cookies only

    // MaxAge=0 means no 'Max-Age' attribute specified.
    // MaxAge<0 means delete cookie now, equivalently 'Max-Age: 0'
    // MaxAge>0 means Max-Age attribute present and given in seconds
    MaxAge   int
    Secure   bool
    HttpOnly bool
    SameSite SameSite // Go 1.11
    Raw      string
    Unparsed []string // Raw text of unparsed attribute-value pairs
}
```

##### 示例

```go
package main

import (
	"net/http"
	"time"
)

func main() {
    http.HandleFunc("/test", func(writer http.ResponseWriter, request *http.Request) {
        expire := time.Now()
        expire.AddDate(0,0,3)
        cookie := &http.Cookie{Name:"Auth",Value:"test",Expires:expire}
        http.SetCookie(writer,cookie)
    })
    http.ListenAndServe(":8080",nil)
}
```

运行结果

![](/assets/images/2019Go/go-response-02.png)

## JSON响应

Go标准库并没有封装直接向客户端响应JSON数据的方法，不过，自己封装一个也很简单的，可以使用encoding/json包的Encoder结构体，将JSON数据写入响应数据流。

##### 示例

```go
package main

import (
    "encoding/json"
    "log"
    "net/http"
)

func main() {
    http.HandleFunc("/profile", func(writer http.ResponseWriter, request *http.Request) {
        data := map[string]string{
            "username": "小明",
            "email":    "xiaoming@163.com",
        }
        err := JSON(writer, data)
        check(err)
    })
    http.ListenAndServe(":8080", nil)
}

func check(err error) {
    if err != nil {
        log.Fatal(err)
    }
}

func JSON(w http.ResponseWriter, data interface{}) error {
	w.Header().Set("Content-Type", "application/json")
	encoder := json.NewEncoder(w)
	return encoder.Encode(data)
}
```

## HTML模板

虽然在前后端分离的大趋势下，后端开发更多时候是以接口api响应JSON的方式返回数据给前端，但仍然有些简单的业务，是由后端直接将HTML模板返回由浏览器，由浏览器渲染。

Go语言可以使用html/template包渲染HTML模板。

##### 示例

```go
package main

import (
    "html/template"
    "log"
    "net/http"
)

const tpl = `
<!DOCTYPE html>
<html>
	<head>
		<meta charset="UTF-8">
		<title>{.Title}</title>
	</head>
	<body>
		{{range .Items}}<div>{ . }</div>{{else}}<div><strong>no rows</strong></div>{{end}}
	</body>
</html>`
// {.Title} 外面还有一个{}括号，{ . } 外面还有{}
func main() {

    t, err := template.New("webpage").Parse(tpl)
    check(err)

    data := struct {
        Title string
        Items []string
    }{
    Title: "我的第一个HTML页面",
    Items: []string{
            "技术文章",
            "我的Blog",
        },
    }

    http.HandleFunc("/profile", func(writer http.ResponseWriter, request *http.Request) {
        err = t.Execute(writer, data)
        check(err)
    })
    http.ListenAndServe(":8080", nil)
}

func check(err error) {
    if err != nil {
        log.Fatal(err)
    }
}
```

## 总结

Go语言在net/http包中对Web开发提供了很好的支持，让开发者在使用Go进行Web应用开发时，几乎不需要使用任何的Web框架，便可完成业务逻辑开发的工作。