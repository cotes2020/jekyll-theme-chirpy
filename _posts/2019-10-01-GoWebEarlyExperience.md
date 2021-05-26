---
layout: post
title: "Go Web初体验"
date: 2019-10-01 22:23:00.000000000 +09:00
categories: [Go Web]
tags: [Go Web]
---

## 前言

在Go语言中开发Web应用，真的是一件非常简单的事情，因为Go语言标准库中就有非常成熟且简单的Web开发包：`net/http`。

net/http封装了开发Web应用所需要的大部分功能，因此，在Go语言中使用net/http开发Web应用程序时，我们甚至都不用像其他语言(比如PHP)一样需要自己再搭一个Apache或nginx等Web服务器，而是只需要简单几行代码就可以搭建一个Web服务应用。

## Web基础

当然，虽然使用Go的net/http包可以简单开发Web应用，但我们在开发中仍然需要牢固地掌握开发Web程序所需要的基础知识，而Web开发中最基础和最核心的知识就是：`HTTP协议`。

http协议是Web服务器与客户端(最常见就是浏览器)之间通讯的语言与规范，浏览器向Web发起请求到Web服务器响应并结束连接，整个过程如下图所示：

![](/assets/images/2019Go/go-experience-01.png)

**请求与响应**

一个完整http事务，由一个客户端的请求和Web服务器响应构成，客户端发起的请求，包括三个部分：`请求行`、`请求头`和`请求体`，而Web服务器的响应同样包含三部分：`响应行`、`响应头`和`响应体`，如下图所示：。

![](/assets/images/2019Go/go-experience-02.png)

http协议的相关知识远不只这些，我们有空再谈谈。

## Go创建Web服务器的几种方式

**http.HandleFunc函数**

使用HandleFunc函数是http封装好的一个函数，可以直接使用，第一个参数是web请求路径，第二个参数是的`func(writer http.ResponseWriter, request *http.Request)`函数。

再使用`http.ListenAndServe(":8080",nil)`语句，监听`8080`端口，运行程序后。

使用http://localhost:8080，便会输出`一起学习Go Web编程吧`。

其中`http.ResponseWriter`代表对客户端的响应体，而`http.Request`代表客户端发送服务器的请求数据。

```go
func hello(writer http.ResponseWriter, request *http.Request) {
    writer.Write([]byte("一起学习Go Web编程吧"));
}

func main(){
    http.HandleFunc("/hello",hello)
    log.Fatal(http.ListenAndServe(":8080",nil))
}
```

**http.Handle函数**

跟`HandleFunc`一样，`Handle`也是http封装好的函数，第一个参数跟`HandleFunc`一样，而第二个参数则是必须是实现了`http.Handler`接口的类型，`http.Handler`在http包的定义如下：

```go
type Handler interface {
	ServeHTTP(ResponseWriter, *Request)
}
```

下面我们定义一个Controller结构体，在该结构定义ServeHTTP方法，因此Controller结构也实现http.Handler接口，而通过`http.HandlerFunc`也能将hello方法转成一个实现`http.HandlerFunc`，`http.HandlerFunc`也实现`http.Handler`，`http.HandlerFunc`在http包的定义如下：

```go
type HandlerFunc func(ResponseWriter, *Request)

// ServeHTTP calls f(w, r).
func (f HandlerFunc) ServeHTTP(w ResponseWriter, r *Request) {
	f(w, r)
}
```

其实，在上面的例子中，我们将hello传给`http.HandleFunc`函数时，`HandleFunc`函数也是使用`http.HandlerFunc`将hello转换成`http.HandlerFunc`的。

下面有关http.Handle的示例：

```go
type Controller struct {}
func (c Controller)ServeHTTP(writer http.ResponseWriter, request *http.Request){
    writer.Write([]byte("hello,1"));
}

func hello(writer http.ResponseWriter, request *http.Request) {
    writer.Write([]byte("hello,2"));
}

func main(){
    http.Handle("/hello1",&Controller{})
    http.Handle("/hello2",http.HandlerFunc(hello))
    log.Fatal(http.ListenAndServe(":8080",nil))
}
```

运行程序后，在浏览器输入下面的地址：

[http://localhost:8080/hell1](https://link.juejin.im?target=http%3A%2F%2Flocalhost%3A8080%2Fhell1), 输出：hello,1

[http://localhost:8080/hell2](https://link.juejin.im?target=http%3A%2F%2Flocalhost%3A8080%2Fhell2), 输出：hello,2

### http.ServeMux

无论是使用`http.Handle`还是`http.HandleFunc`函数，其实底层代码都是使用`http.DefaultServeMux`，`DefaultServeMux`的定义如下代码所示：

```go
var DefaultServeMux = &defaultServeMux

var defaultServeMux ServeMux
复制代码
type Controller struct {}
func (c Controller)ServeHTTP(writer http.ResponseWriter, request *http.Request){
    writer.Write([]byte("hello,1"));
}

func hello(writer http.ResponseWriter, request *http.Request) {
    writer.Write([]byte("hello,2"));
}

func main(){
    mux := &http.ServeMux{}
    mux.HandleFunc("/hello1",hello)
    mux.Handle("/hello2",http.HandlerFunc(hello))
    mux.Handle("/hello3",&Controller{})

    log.Fatal(http.ListenAndServe(":8080",mux))
}
```

运行程序后，在浏览器输入下面的地址：

[http://localhost:8080/hell1](https://link.juejin.im?target=http%3A%2F%2Flocalhost%3A8080%2Fhell1), 输出：hello,1

[http://localhost:8080/hell2](https://link.juejin.im?target=http%3A%2F%2Flocalhost%3A8080%2Fhell2), 输出：hello,1

[http://localhost:8080/hell3](https://link.juejin.im?target=http%3A%2F%2Flocalhost%3A8080%2Fhell3), 输出：hello,2

### http.Server

`http.Server`是http包中对web更加底层的支持，我们前面使用的方法，都是对`http.Server`的封装而已，如果直接使用`http.Server`，则可以自定义更多的参数，如果连接超时等参数，因此我们下面直接使用`http.Server`开发Web服务。

```go
func main() {
    myHandler := &http.ServeMux{}
    myHandler.HandleFunc("/hello", func(w http.ResponseWriter, r *http.Request) {
        w.Write([]byte("hello"))
    })
    s := &http.Server{
        Addr:           ":8080",
        Handler:        myHandler,
        ReadTimeout:    10 * time.Second,
        WriteTimeout:   10 * time.Second,
        MaxHeaderBytes: 1 << 20,
    }
    log.Fatal(s.ListenAndServe())
}
```

运行程序后，在浏览器输入下面的地址：

[http://localhost:8080/hello](https://link.juejin.im?target=http%3A%2F%2Flocalhost%3A8080%2Fhello), 输出：hello

## 总结

通过上面的例子，可以看出Go Web开发可很简单，但是实际中，一个真正Web应用所要做的事，远不只这么简单，对于Go Web开发，还是有很多东西要学的。