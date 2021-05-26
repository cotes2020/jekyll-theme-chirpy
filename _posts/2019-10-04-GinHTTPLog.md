---
layout: post
title: "GiN框架HTTP请求日志"
date: 2019-10-04 20:11:00.000000000 +09:00
categories: [Go Web]
tags: [Go Web, Gin]
---

Gin框架在控制台中输出的日志如下：

```
[GIN] 2019/05/04 - 22:08:56 | 200 |      5.9997ms |      ::1 | GET   /test
```

那么我们如何输出上面的日志呢？

## 日志中件间

在Gin框架中，要输出用户的http请求日志，最直接简单的方式就是借助日志中间件，Gin框架的中间件定义如下：

```go
func Logger() HandlerFunc
```

所以我们使用下面的代码创建一个`gin.Engine`时，会在控制台中用户的请求日志：

```go
router := gin.Default()
```

而使用下面的代码创建`gin.Engine`时，则不会在控制台输出用户的请求日志：

```go
router := gin.New()
```

这是由于使用`Default()`函数创建的`gin.Engine`实例默认使用了日志中件间`gin.Logger()`；而当我们使用第二种方式创建`gin.Engine`时，可以调用`gin.Engine`中的`Use()`方法调用`gin.Logger()`，如下：

```go
router := gin.New()
router.Use(gin.Logger())
```

## 在控制台输出日志

Gin框架请求日志默认是在我们运行程序的控制台中输出，而且输出的日志中有些字体有标颜色，如下图所示：

![](/assets/images/2019Go/go-httplog-01.png)

我们也可以使用`DisableConsoleColor()`函数禁用控制台日志的颜色输出，代码如下所示

```go
gin.DisableConsoleColor() //禁用字体颜色
router := gin.Default()
router.GET("test",func(c *gin.Context){
    c.JSON(200,"test")
})
```

运行后发出Web请求，在控制台输出日志字体则没有颜色：

![](/assets/images/2019Go/go-httplog-02.png)

虽然Gin框架默认是开始日志字体颜色的，但可以使用`DisableConsoleColor()`函数来禁用，但当被禁用后，在程序中运行需要重新打开控制台日志的字体颜色输出时，可以使用`ForceConsoleColor()`函数重新开启，使用如下：

```go
gin.ForceConsoleColor()
```

## 在文件中输出日志

Gin框架的请求日志默认在控制台输出，但更多的时候，尤其上线运行时，我们希望将用户的请求日志保存到日志文件中，以便更好的分析与备份。

**DefaultWriter**

在Gin框架中，通过`gin.DefaultWriter`变量可能控制日志的保存方式，`gin.DefaultWriter`在Gin框架中的定义如下：

```go
var DefaultWriter io.Writer = os.Stdout
```

从上面的定义我们可以看出，`gin.DefaultWriter`的类型为`io.Writer`,默认值为`os.Stdout`,即控制台输出，因此我们可以通过修改`gin.DefaultWriter`值来将请求日志保存到日志文件或其他地方(比如数据库)。

```go
package main
import (
    "github.com/gin-gonic/gin"
    "io"
    "os"
)
func main() {
    gin.DisableConsoleColor()//保存到文件不需要颜色
    file, _ := os.Create("access.log")
    gin.DefaultWriter = file
    //gin.DefaultWriter = io.MultiWriter(file) 效果是一样的
    router := gin.Default()
    router.GET("/test", func(c *gin.Context) {
        c.String(200, "test")
    })
    _ = router.Run(":8080")
}
```

运行后上面的程序，会在程序所在目录创建`access.log`文件，当我们发起Web请求后，请求的日志会保存到`access.log`文件，而不会在控制台输出。

通过下面的代码，也可能让请求日志同行保存到文件和在控制台输出：

```go
file, _ := os.Create("access.log")
gin.DefaultWriter = io.MultiWriter(file,os.Stdout) //同时保存到文件和在控制台中输出
```

**LoggerWithWriter**

另外，我们可以使用`gin.LoggerWithWriter`中间件，其定义如下：

```go
func LoggerWithWriter(out io.Writer, notlogged ...string) HandlerFunc
```

示例代码：

```go
package main

import (
    "github.com/gin-gonic/gin"
    "os"
)

func main() {
    gin.DisableConsoleColor()
    router := gin.New()
    file, _ := os.Create("access.log")
    router.Use(gin.LoggerWithWriter(file,""))
    router.GET("test", func(c *gin.Context) {
        c.JSON(200,"test")
    })
    _ = router.Run()
}
```

`gin.LoggerWithWriter`中间件的第二个参数，可以指定哪个请求路径不输出请求日志，例如下面代码，`/test`请求不会输出请求日志，而`/ping`请求日志则会输出请求日志。

```go
router.Use(gin.LoggerWithWriter(file,"/test"))//指定/test请求不输出日志
router.GET("test", func(c *gin.Context) {
    c.JSON(200,"test")
})
router.GET("ping", func(c *gin.Context) {
    c.JSON(200,"pong")
})
```

## 定制日志格式

**LogFormatterParams**

上面的例子，我们都是采用Gin框架默认的日志格式，但默认格式可能并不能满足我们的需求，所以，我们可以使用Gin框架提供的`gin.LoggterWithFormatter()`中间件，定制日志格式，`gin.LoggterWithFormatter()`中间件的定义如下：

```go
func LoggerWithFormatter(f LogFormatter) HandlerFunc
```

从`gin.LoggterWithFormatter()`中间件的定义可以看到该中间件的接受一个数据类型为`LogFormatter`的参数，`LogFormatter`定义如下：

```go
type LogFormatter func(params LogFormatterParams) string
```

从`LogFormatter`的定义看到该类型为`func(params LogFormatterParams) string`的函数，其参数是为`LogFormatterParams`,其定义如下：

```go
type LogFormatterParams struct {
    Request *http.Request
    TimeStamp time.Time
    StatusCode int
    Latency time.Duration
    ClientIP string
    Method string
    Path string
    ErrorMessage string
    BodySize int
    Keys map[string]interface{}
}
```

定制日志格式示例代码：

```go
func main() {
    router := gin.New()
    router.Use(gin.LoggerWithFormatter(func(param gin.LogFormatterParams) string {
        //定制日志格式
        return fmt.Sprintf("%s - [%s] \"%s %s %s %d %s \"%s\" %s\"\n",
            param.ClientIP,
            param.TimeStamp.Format(time.RFC1123),
            param.Method,
            param.Path,
            param.Request.Proto,
            param.StatusCode,
            param.Latency,
            param.Request.UserAgent(),
            param.ErrorMessage,
        )
    }))
    router.Use(gin.Recovery())
	router.GET("/ping", func(c *gin.Context) {
        c.String(200, "pong")
    })
    _ = router.Run(":8080")
}
```

运行上面的程序后，发起Web请求，控制台会输出以下格式的请求日志：

```
::1 - [Wed, 08 May 2019 21:53:17 CST] "GET /ping HTTP/1.1 200 1.0169ms "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/74.0.3729.131 Safari/537.36" "
```

**LoggerWithConfig**

在前面的例子中，我们使用`gin.Logger()`开启请求日志、使用`gin.LoggerWithWriter`将日志写到文件中，使用`gin.LoggerWithFormatter`定制日志格式，而实际上，这三个中间件，其底层都是调用`gin.LoggerWithConfig`中间件，也就说，我们使用`gin.LoggerWithConfig`中间件，便可以完成上述中间件所有的功能，`gin.LoggerWithConfig`的定义如下：

```go
func LoggerWithConfig(conf LoggerConfig) HandlerFunc
```

`gin.LoggerWithConfig`中间件的参数为`LoggerConfig`结构，该结构体定义如下：

```go
type LoggerConfig struct {
    // 设置日志格式
    // 可选 默认值为：gin.defaultLogFormatter
    Formatter LogFormatter

    // Output用于设置日志将写到哪里去
    // 可选. 默认值为：gin.DefaultWriter.
    Output io.Writer

    // 可选，SkipPaths切片用于定制哪些请求url不在请求日志中输出.
    SkipPaths []string
}
```

以下例子演示如何使用`gin.LoggerConfig`达到日志格式、输出日志文件以及忽略某些路径的用法：

```go
func main() {
    router := gin.New()
    file, _ := os.Create("access.log")
    c := gin.LoggerConfig{
        Output:file,
        SkipPaths:[]string{"/test"},
        Formatter: func(params gin.LogFormatterParams) string {
            return fmt.Sprintf("%s - [%s] \"%s %s %s %d %s \"%s\" %s\"\n",
                params.ClientIP,
                params.TimeStamp.Format(time.RFC1123),
                params.Method,
                params.Path,
                params.Request.Proto,
                params.StatusCode,
                params.Latency,
                params.Request.UserAgent(),
                params.ErrorMessage,
            )
        },
    }
    router.Use(gin.LoggerWithConfig(c))
    router.Use(gin.Recovery())
    router.GET("/ping", func(c *gin.Context) {
        c.String(200, "pong")
    }) 
    router.GET("/test", func(c *gin.Context) {
        c.String(200, "test")
    })
    _ = router.Run(":8080")
}
```

运行上面的程序后，发起Web请求，控制台会输出以下格式的请求日志：

```
::1 - [Wed, 08 May 2019 22:39:43 CST] "GET /ping HTTP/1.1 200 0s "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/74.0.3729.131 Safari/537.36" "
::1 - [Wed, 08 May 2019 22:39:46 CST] "GET /ping HTTP/1.1 200 0s "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/74.0.3729.131 Safari/537.36" "
```

## 总结结

每条HTTP请求日志，都对应一次用户的请求行为，记录每一条用户请求日志，对于我们追踪用户行为，过滤用户非法请求，排查程序运行产生的各种问题至关重要，左移，开发Web应用时一定要记录用户请求行为，并且定时分析过滤。