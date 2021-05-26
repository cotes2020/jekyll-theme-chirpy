---
layout: post
title: "Go 处理Web请求"
date: 2019-10-01 23:10:00.000000000 +09:00
categories: [Go Web]
tags: [Go Web]
---

Web应用程序最重要的功能，就是接收来自客户端(一般是浏览器或APP)发起的请求，根据请求方法与类型返回静态HTML页面或动态生成客户端想要的数据，而Go语言中`net/http`已经在处理请求数据方面做了很好的封装，使得用Go代码处理Web请求的数据并生成响应数据变得非常简单。

## Web请求

我们知道，一个HTTP事务由`请求`和`响应`构成，这篇文章中，我们单讲有关Web请求的部分。

客户端一般是通过一个`URL`向服务器发起请求，最简单的比如在浏览地址栏输入：[Google](https://www.google.com/)

每一个Web请求都包括三个部分：`请求行`、`请求头`、`请求实体`。

**请求方法**

`请求方法`在`请求行`当中,HTTP协议支持多种`请求方法(method)`，主要有七种：

`GET`,`POST`,`PUT`,`HEADER`,`PATCH`,`DELETE`,`OPTIONS`。

其中Web开发最常见就是`GET`方法和`POST`方法，使用`GET`方法发起的请求，没有`请求实体(body)`，因些请求的数据一般只能通过`URL`中的`查询参数(query)`传给服务端，而使用POST方法的请求则会携带`请求实体`，在`请求实体`中带有传给服务端的数据。

**Content-Type**

`Content-Type`是`请求头部(Header)`中一个用于指`请求实体`类型的内容头部，在请求或响应用于指`请求实体`到底存放什么样的数据，所以只有会推带`请求实体`的方法起作用，如`POST`方法。

在一般Web开发中，Content-Type最常用取值有下面四种：

1. `application/json`：JSON数据
2. `application/x-www-form-urlencoded`：form表单请求使用的类型，在发送前会编码所有字符
3. `multipart/form-data`：不对字符编码，一般用于文件上传
4. `text/html`：一般用于响应中的响应头，告诉客户端返回的是HTML文档。

**查询参数query**

查询参数是URL中?后面跟着的部分，比如在掘金的搜索框中输入:Go,我们会看到浏览器的地址栏变成：

[juejin.im/search?quer…](https://juejin.im/search?query=Go&type=1)

查询参数就是指`query=Go&type=1`,查询参数由&分隔开，这是GET方法携带数据的方式，Go语言中支持获取查询参数部分的参数。

**请求实体body**

`请求实体`是一次Web请求中数据携带部分，一般只有POST请求才有这个部分，服务器由Content-Type首部来判断`请求实体`的内容编码格式，如果想向服务器发送大量数据，一般都用POST请求。

## Go处理Web请求数据

在Go语言中，使用`http.Request`结构来处理http请求的数据，在我们定义处理请求的方法，会传入http.Request的实例，如下代码中`request`就是代表一个请求的实例。

```go
http.HandleFunc("/hello", func(writer http.ResponseWriter, request *http.Request) {
    //使用request可以获取http请求的数据
})
```

在golang官方文档中，可以看到http.Request的包外可访问方法列表:

![](/assets/images/2019Go/go-webrequest-01.png)

下面是http.Request公开可访问的字段

```go
type Request struct {
        Method string //方法:POST,GET...
        URL *url.URL //URL结构体
        Proto      string // 协议："HTTP/1.0"
        ProtoMajor int    // 1
        ProtoMinor int    // 0
        Header Header    //头部信息
        Body io.ReadCloser //请求实体
        GetBody func() (io.ReadCloser, error) // Go 1.8
        ContentLength int64  //首部：Content-Length
        TransferEncoding []string
        Close bool           //是否已关闭
        Host string          //首部Host
        Form url.Values      //参数查询的数据
        PostForm url.Values // application/x-www-form-urlencoded类型的body解码后的数据
        MultipartForm *multipart.Form //文件上传时的数据
        Trailer Header
        RemoteAddr string          //请求地址
        RequestURI string          //请求的url地址
        TLS *tls.ConnectionState
        Cancel <-chan struct{} // 
        Response *Response //      响应数据
}
```

**获得请求头Header**

对于常用的请求头部信息，http.Request结构有对应的字段和方法，如下所示：

```go
http.HandleFunc("/hello", func(writer http.ResponseWriter, request *http.Request) {
    request.RemoteAddr
    request.RequestURI
    request.ContentLength 
    request.Proto
    request.Method 
    request.Referer()
    request.UserAgent()
})
```

可以通过request.Header字段来获取，request.Header的定义如下所示：

```go
type Header map[string][]string
```

request.Header是一个的类型是map，另外request.Header也提供相应的方法，如下所示：

![](/assets/images/2019Go/go-webrequest-02.png)

我们除了使用上面的方法获取头部信息外，也可以使用request.Header来获取，示例：

```go
http.HandleFunc("/hello", func(writer http.ResponseWriter, request *http.Request) {
    request.Header.Get("Content-Type")//返回的是string
    request.Header["Content-Type"] //返回的是[]string
})
```

**获取查询参数Query**

如何获取查询参数(url中?后面使用&分隔的部分)呢？可以使用request.FormValue(key)方法获取查询参数，其中key为参数的名称，代码如下：

```go
package main
import (
	"fmt"
	"net/http"
)
func main() {
    http.HandleFunc("/hello", func(writer http.ResponseWriter, request *http.Request) {
        username := request.FormValue("username")
        gender := request.FormValue("gender")
        fmt.Fprintln(writer,fmt.Sprintf("用户名：%s,性别:%s",username,gender))
    })
    fmt.Println(http.ListenAndServe(":8080",nil))
}
```

在Postman输入http://localhost:8080/hello?username=test&gender=男

![](/assets/images/2019Go/go-webrequest-03.png)

**获取表单信息Form**

我们说获取表单信息，一般是指获取Content-Type是`application/x-www-form-urlencoded`或`multipart/form-data`时，`请求实体`中的数据，如果你有做传统网页中的表单提交数据的经历，相信对这两种提交数据的方式应该是熟悉的，而`multipart/form-data`一般是用来上传文件的。

**application/x-www-form-urlencoded**

获取Content-Type为application/x-www-form-urlencoded时提交上来的数据，可以使用request.PostForm字段request.Form和request.PostFormValue(key)方法获取，但必须先调用request.ParseForm()将数据写入request.PostForm字段中。

步骤为：

1. 使用request.ParseForm()函数解析body参数，这时会将参数写入Form字段和PostForm字段当中。
2. 使用request.Form、request.PostForm或request.PostFormValue(key)都可以获取

注意，request.Form和request.PostForm的类型url.Values，结构定义如下：

```go
type Values map[string][]string
```

示例如下：

```go
package main
import (
	"fmt"
	"net/http"
)
func main(){
    http.HandleFunc("/hello", func(writer http.ResponseWriter, request *http.Request) {
        err := request.ParseForm()
        if err != nil{
        fmt.Fprintln(writer,"解析错误")
        }
        username1 := request.PostForm["username"][0]
        username2 := request.PostFormValue("username")
        username3 := request.Form["username"][0]
        fmt.Fprintln(writer,fmt.Sprintf("username1：%s,username2:%s,usernam3:%s",username1,username2,username3))
    })
    fmt.Println(http.ListenAndServe(":8080",nil))
}
```

使用Postman输入http://localhost:8080/hello，结果如下：

![](/assets/images/2019Go/go-webrequest-04.png)

#### multipart/form-data

获取`Content-Type`为`multipart/form-data`时提交上来的数据，步骤如下:

1. 使用request.ParseMultipartForm(maxMemory)，解析参数，将参数写入到MultipartForm字段当中，其中maxMemory为上传文件最大内存。
2. 使用request.FormFile(文件域)，可以获取上传的文件对象：multipart.File
3. 除了文件域，其中参数可以从request.PostForm字段获取，注意，此时不需要再调用request.ParseForm()了。

```go
package main
import (
	"fmt"
	"net/http"
)
func main() {
    http.HandleFunc("/upload", func(writer http.ResponseWriter, request *http.Request) {
        err := request.ParseMultipartForm(32 << 20)
        if err != nil {
            fmt.Fprintln(writer,"文件上传错误")
            return
        }
        fmt.Println(request.FormFile("file"))
    })
    fmt.Println(http.ListenAndServe(":8080",nil))
}
```

## 总结

上面简单地介绍了使用Go语言如何获取http请求提交上来的数据，重点为获了表单数据，但其实在现在前后端分离开发趋势和APP开发中，Content-Type指application/json是更常见的数据提交方式，以后有空可以学一下。