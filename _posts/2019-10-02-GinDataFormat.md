---
layout: post
title: "Gin框架数据响应格式"
date: 2019-10-02 22:31:00.000000000 +09:00
categories: [Go Web]
tags: [Go Web, Gin]
---

## 前言

```go
r.GET("/test",func(c *gin.Context){
    //省略处理请求的代码
})
```

上面的例子中，我们定义了一个处理HTTP GET请求的方法，回调用函数的参数为`*gin.Context`,Gin框架在`*gin.Context`实例中封装了所以处理请求并响应客户端的方法，Gin支持多种响应方法，包括我们常见的`String`,`HTML`,`JSON`,`XML`,`YAML`,`JSONP`,也支持直接响应`Reader`和`[]byte`，而且还支持重定向。

下面列表出gin.Context中响应客户端的方法列表：

```go
func (c *Context) AsciiJSON(code int, obj interface{})
func (c *Context) Data(code int, contentType string, data []byte)
func (c *Context) DataFromReader(code int, contentLength int64, contentType string,
reader io.Reader, extraHeaders map[string]string)
func (c *Context) HTML(code int, name string, obj interface{})
func (c *Context) IndentedJSON(code int, obj interface{})
func (c *Context) JSON(code int, obj interface{})
func (c *Context) JSONP(code int, obj interface{})
func (c *Context) ProtoBuf(code int, obj interface{})
func (c *Context) PureJSON(code int, obj interface{})
func (c *Context) Redirect(code int, location string)
func (c *Context) Render(code int, r render.Render)
func (c *Context) SecureJSON(code int, obj interface{})
func (c *Context) String(code int, format string, values ...interface{})
func (c *Context) XML(code int, obj interface{})
func (c *Context) YAML(code int, obj interface{})
```

## JSON

json是轻量级数据交互格式，应用很广泛，尤其是Web API服务方面，因此Gin框架为JSON提供了支持，使用gin.Context中的JSON方法将格式化后的JSON数据返回给客户端。

```go
package main
import (
    "github.com/gin-gonic/gin"
    "net/http"
)
type User struct {
    Uid      int    `json:"uid" xml:"uid"`
    Username string `json:"username" xml:"username"`
}
func main() {
    r := gin.Default()
    r.GET("test", func(c *gin.Context) {
        data := &User{Uid:1,Username:"测试账号"}
        c.JSON(http.StatusOK,data)
    })
    r.Run()
}
```

## XML/YAML/String

响应XML、YAML或String格式的数据，处理方式如同JSON一样，Gin都提供相应的函数，示例如下：

```go
//xml
r.GET("xml",func(c *gin.Context){
    data := gin.H{"xml":"Hello World"}
    c.XML(200,data)
})
//yaml
r.GET("yaml",func(c *gin.Context){
    data := gin.H{"xml":"Hello World"}
    c.YAML(200,data)
})
//string
r.GET("yaml",func(c *gin.Context){
    c.String(200,"Hello World")
})
```

上面的例子中， 我们使用了一个叫gin.H类型，实际gin.H在gin包中的定义如下：

```go
type H map[string]interface{}
```

## Protobuf

Protobuf是一种与平台无关和语言无关，且可扩展且轻便高效的序列化数据结构的协议，可以用于网络通信和数据存储，其实序列化的速度要比JSON和XML快，但其易用性和可阅读性远不如JSON和XML，因此并没有广泛使用。

不过Gin也为Protobuf数据响应提供了支持，以下例子来自Gin官方文档：

```go
r.GET("/someProtoBuf", func(c *gin.Context) {
    reps := []int64{int64(1), int64(2)}
    label := "test"
    // protobuf 的具体定义写在 testdata/protoexample 文件中。
    data := &protoexample.Test{
        Label: &label,
        Reps:  reps,
    }
    // 请注意，数据在响应中变为二进制数据
    // 将输出被 protoexample.Test protobuf 序列化了的数据
    c.ProtoBuf(http.StatusOK, data)
})
```

## HTML渲染

Gin也支持传统Web编程中的HTML模板渲染，直接返回HTML代码给客户端，主要步骤如下：

**定义HTML模板**

> HTML模板文件：templates/index.html

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Title</title>
</head>
    <body>
    {.foo}
    </body>
</html>
// {.foo} 外面还有{}括号
```

**加载模板**

定义好模板之后，在渲染之前，要使用gin.Engine中的LoadHTMLFiles()或LoadHTMLGlob()方法加载模板。

> LoadHTMLFiles(files ...string)方法可以接收一个或多个参数，用于加载单个或多个模板文件。

```go
r := gin.Default()
r.LoadHTMLFiles("./templates/index.html")
```

> LoadHTMLGlob(pattern string)方法则用于加载整个目录下的模板文件，如果目录不存在或目录下没有模板文件会引发panic错误。

```go
r := gin.Default()
r.LoadHTMLGlob("./templates/*")
```

**完整示例**

```go
func main() {
    r := gin.Default()
    r.LoadHTMLFiles("./templates/index.html")
    //r.LoadHTMLGlob("./templates/*")
    r.GET("html", func(c *gin.Context) {
        data := map[string]interface{}{
            "foo": "bar",
        }
        c.HTML(http.StatusOK,"index.html",data)
    })
    r.Run()
}
```

## JSONP

JSONP是一种基于JSON，而用于解决浏览器跨域访问问题的机制，使用gin.Context的JSONP()返回数据时，会将URL中的callback参数按照JSONP的数据格式放在json数据前面,并返回给客户端。

```go
func main() {
    r := gin.Default()
    r.GET("/JSONP?callback=test", func(c *gin.Context) {
        data := map[string]interface{}{
            "foo": "bar",
        }
        c.JSONP(http.StatusOK,data)
    })
    r.Run()
}
```

## Reader

使用gin.Context中的DataFromReader()方法，可以直接从Reader读取数据，下面演示一个下载图片的HTTP请求：

```go
func main() {
    r := gin.Default()
    r.GET("file", func(c *gin.Context) {
        fileName := "./1.jpg"
        file, _ := os.Open(fileName)
        fileInfo, _ := os.Lstat(fileName)
        extraHeaders := map[string]string{
            "Content-Disposition": `attachment; filename="` + fileName + `"`,
        }
        c.DataFromReader(200, fileInfo.Size(), "image/png", file, extraHeaders)
    })
    r.Run()
}
```

## 字节数组

使用gin.Context中的Data()方法，可以返回一个字节数组([]byte)给客户端，下面的例子演示从图片中读取二进制数组并返回给客户端。

```go
func main() {
    r := gin.Default()
    r.GET("file", func(c *gin.Context) {
        file, _ := os.Open("./1.jpg")
        b, _ := ioutil.ReadAll(file)
        c.Data(200, "image/png", b)
    })
    r.Run()
}
```

## 重定向

除了返回不同格式的数据给客户端，Gin框架也支持重定向操作，重定向分为外部和内部重定向。

**外部重定向**

用于跳转其他外部的链接。

```go
func main(){
    r := gin.Default()
    r.GET("Redirect",func(c *gin.Context){
        c.Redirect(http.StatusMovedPermanently,"https://juejin.im")  
    })
    r.Run()
}
```

**内部重定向**

用于跳转内部路由。

```go
func main() {
    r := gin.Default()
    r.GET("test", func(c *gin.Context) {
        c.Request.URL.Path = "/profile"
        r.HandleContext(c)
    })
    r.GET("profile", func(c *gin.Context) {
        username := c.Query("username")
        fmt.Println(username)
        c.JSON(200, gin.H{"user": "profile"})
    })
    r.Run()
}
```

## 总结

从上面的多个例子中可以看到Gin响应Web请求的良好封装，因此使用Gin进入Web应用程序的开发，会大大提升开发速度与效率的。