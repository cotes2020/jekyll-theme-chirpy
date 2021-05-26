---
layout: post
title: "Gin框架安装与使用"
date: 2019-10-02 20:03:00.000000000 +09:00
categories: [Go Web]
tags: [Go Web, Gin]
---

Go标准库`net/http`对使用Go开发Web应用提供非常强大的支持，然而对于想要快速开发并及上线项目的团队来说，使用Web框架不失为一种更好的选择。

Go社区中，有许多非常优秀的Web框架，如`Gin`,`Iris`,`Echo`,`Martini`,`Revel`以及国人开发的`Beego`框架。

[Gin官网](<https://gin-gonic.com/>)

[Gin](<https://github.com/gin-gonic/gin>)

## Gin的优点

- 快速：基于`Radix`树的路由,性能非常强大。
- 支持中间件：内置许多中间件，如`Logger`,`Gzip`,`Authorization`等。
- 崩溃恢复：可以捕捉panic引发的程序崩溃，使Web服务可以一直运行。
- JSON验证：可以验证请求中`JSON`数据格式。
- 路由分组：支持路由分组(`RouteGroup`)，可以更方便组织路由。
- 错误管理机制：可以收集程序中的错误
- 多种数据渲染方式：支持`HTML`、`JSON`、`YAML`、`XML`等数据格式的响应。
- 扩展性：非常简单扩展中间件。
- 数据验证器：支持数据验证器且可以自定义。

## 安装与使用

Gin目前最新的版本是`V1.3.0`，其安装过程非常简单，不过在安装Gin之前，需要安装`Go1.6`或以上的版本(后续版本可能要`Go1.8`或以上)，下面介绍两种安装方式。

**直接安装**

```
$  go get -u github.com/gin-gonic/gin //使用-u安装最新版本
```

**使用Govendor安装**

> 提示：Govendor是使用Go语言开发Go项目依赖管理工具。

1. 安装Govendor

```
$ go get github.com/kardianos/govendor
```

1. 安装Gin

```
$ govendor init
$ govendor fetch github.com/gin-gonic/gin@v1.3
```

## 简单示例

通过上面的两种方式安装好Gin之后，下面通过一个简单示例看看怎么Gin使用开发Web应用。

```go
import "github.com/gin-gonic/gin"
func main(){
    r := gin.Default()
    r.GET("/test",func(c *gin.Context){
        c.JSON(200,gin.H{"hello":"world"})
    })
    r.Run()
}
```

可以看到，使用gin开发一个Web服务是很简单的一件事情，可以简单地分解为四步：

**导入gin包**

在我们安装Gin框架的时候，已经将gin包安装到本地，如果使用`go get`命令安装的，则这个包路径为`$GOPATH/src/github.com/gin-gonic/gin`,而我们只需要使用`import`命令便可以将包导入。

```
import "github.com/gin-gonic/gin"
```

**创建路由**

使用gin.Default()方法会返回gin.Engine实例，表示默认路由引擎。

```go
r := gin.Default()
```

通过这种方式创建的gin.Engine，会默认使用Logger和Recovery两个中间件，可以用gin.New()方法创建一个不包含任何中间件的默认路由。

```go
r := gin.New()
```

**定义处理HTTP的方法**

通过默认路由，我们可以创建处理HTTP请求的方法，示例中使用GET方法:

```go
r.GET("/test",func(c *gin.Context){
    c.JSON(200,gin.H{"hello":"world"})
})
```

Gin支持所有通用的HTTP请求方法：`GET`,`POST`,`PUT`,`PATCH`,`OPTIONS`,`HEAD`,`DELETE`,其使用方式与上面例子相同，如POST:

```go
r.POST("/test",func(c *gin.Context){
    c.JSON(200,gin.H{"hello":"world"})
})
```

每种方法都只处理对应的HTTP请求，使用Any方法则可以处理任何的HTTP请求。

```go
r.Any("/test",func(c *gin.Context){
    c.JSON(200,gin.H{"hello":"world"})
})
```

**监听端口**

定义好请求之后，使用Run()方法便可监听端口，开始接受HTTP请求,如果Run()方法没有传入参数的话，则默认监听的端口是8080。

```
r.Run() //r.Run(":3000")
```

## 总结

Gin是一个Go Web开发的轻量级框架，使用也非常地简单，容易上手，但是，使用Gin开发前，还是需要对Go原生支持的net/http有所了解。