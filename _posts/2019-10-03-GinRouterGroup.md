---
layout: post
title: "Gin框架路由分组"
date: 2019-10-03 22:21:00.000000000 +09:00
categories: [Go Web]
tags: [Go Web, Gin]
---

## HttpRouter

Gin框架的路由实际是使用第三方的`HttpRouter`库,`HttpRouter`是一个轻量级高性能的`HTTP`请求路由器(`路由多路复用器`),更简单地理解,`HttpRouter`是Go的net/http包中ServeMux更好的实现，由于`Gin`框架采用了`HttpRouter`,因此性能有很大的提升。

使用以下命令，可以在本地GOPATH中安装`HttpRouter`：

```
go get -u github.com/julienschmidt/httprouter
```

## 路由分组

**支持的HTTP方法**

定义路由是为了处理HTTP请求，而HTTP请求包含不同方法，包括`GET`,`POST`,`PUT`,`PATCH`,`OPTIONS`,`HEAD`,`DELETE`等七种方法，Gin框架中都有对应的方法来定义路由。

```go
router := gin.New()

router.GET("/testGet",func(c *gin.Context){
    //处理逻辑
})

router.POST("/testPost",func(c *gin.Context){
    //处理逻辑
})

router.PUT("/testPut",func(c *gin.Context){
    //处理逻辑
})

router.DELETE("/testDelete",func(c *gin.Context){
    //处理逻辑
})

router.PATCH("/testPatch",func(c *gin.Context){
    //处理逻辑
})

router.OPTIONS("/testOptions",func(c *gin.Context){
    //处理逻辑
})

router.OPTIONS("/testHead",func(c *gin.Context){
    //处理逻辑
})
```

上面通过对应方法创建的路由，只能处理对应请求方法，如果GET定义的路由无法处理POST请求，我们可以通过`Any()`方法定义可以处理任何请求的路由。

```go
//可以处理GET,POST等各种请求
router.Any("/testAny",func(c *gin.Context){
    //处理逻辑
})
```

除了上面几种简便的方法，也可以使用`Handle()`创建路由，通过指定`Handle()`函数的第一个参数来确定处理何种请求：

```go
//定义处理POST请求的方法
router.Handle("POST","/testHandlePost",func(c *gin.Context){
    
})
//定义处理GET请求的方法
router.Handle("GET","/testHandleGet",func(c *gin.Context){
    
})
```

**路由请求路径**

定义路由时，都需要定义该路由的请求路径，在Gin框架中，通过其支持的多种请求方法，可以很容易实现Restful风格Api请求路径。

在前面的示例中，都是使用直接匹配的路由路径，如下面的代码，只能匹配请求路径为`test`的请求。

```go
router.GET("test",func(c *gin.Context){
    
})
```

除了直接匹配路径，Gin框架还支持使用通配符冒号(:)和星号(*)来匹配请求路径，如：

使用冒号(:)定义路由路径：

```go
router.GET("user/:name",func(c *gin.Context){
})
```

上面的请求路径，请求结果对应如下：

```go
/user/gordon              匹配
/user/you                 匹配
/user/gordon/profile      不匹配
/user/                    不匹配
```

使用星号(*)定义路由请求路径：

```go
router.GET("user/*name",func(c *gin.Context){
})
```

上面的请求路径，请求结果对应如下：

```
/user/gordon              匹配
/user/you                 匹配
/user/gordon/profile      匹配
/user/                    匹配
```

**路由分组**

在前面的示例中，当我们通过`gin.New()`或`gin.Default()`方法创建`gin.Engine`结构体实例时，然后可以使用该实例中的`GET`,`POST`,`PUT`,`PATCH`,`OPTIONS`,`HEAD`,`DELETE`等方法来定义处理请求的路由，而其实`gin.Engine`的路由功能是通过组合`gin.RouterGroup`来实现的，从下面的`gin.Engine`代码可以看出。

> 通过组合的方式，获取其他数据类型的字段和方法，正是Go语言面向对象编码的体现。

##### `gin.Engine`的定义：

```go
type Engine struct {
    RouterGroup //组合gin.RouterGroup
   //省略其他字段
}
```

而直接通过`gin.New()`或`gin.Default()`方法创建的`gin.Engine`对象来创建路由时，实际这些路由看起来并没有在某个路由分组下，而实际上这些路由可以`根路由分组`下面。

可以理解为，在Gin中定义的所有路由，都在根路由分组下面.

##### `gin.RouterGroup`的定义：

```go
type RouterGroup struct {
    Handlers HandlersChain
    // contains filtered or unexported fields
}
```

**定义路由分组**

使用`gin.RouterGroup`的`Group()`方法可以定义路由分组，如下所示：

> 注意，下面使用花括号`{}`将分组下的路由包起来，只是为了更加直观，并非必要的。

```go
router := gin.New()
user := router.Group("user")
{
    user.GET("profile",func(c *gin.Context)(){
        //处理逻辑
    })
    
     user.POST("modify-password",func(c *gin.Context)(){
        //处理逻辑
    })
}
```

## 路由中使用中间件

**Use()方法：全局中间件**

Use()方法定义如下：

```go
func (group *RouterGroup) Use(middleware ...HandlerFunc) IRoutes
```

示例代码

```go
router := gin.New()
router.Use(gin.Loggter())//全局中间件
```

为什么通过`gin.Engine`返回的结构体中的Use()调用中间件是全局中间件呢？原因在于所有的路由分组都在`根路由分组`下面。

**在分组中使用中间件**

```go
router := gin.New()
user := router.Group("user",gin.Logger())//通过Group第二个参数，使用中间件
{
    user.GET("profile",func(c *gin.Context)(){
        //处理逻辑
    })
    
     user.POST("modify-password",func(c *gin.Context)(){
        //处理逻辑
    })
}
```

也可以使用返回的RouterGroup中的Use方法为路由分组应用中间件：

```go
user := router.Group("user",gin.Logger()).Use(gin.Recovery())
```

**在单个路由使用中间件**

下面代码演示了在单个路由定义使用中间件的用法：

```go
router := gin.New()
router.GET("profile",gin.Logger(),gin.Recovery(),func(c *gin.Context)(){
    //处理逻辑
})
```

或者在GET等方法之后，再使用Use()方法，为该路由应用中间件：

```go
router.GET("profile",func(c *gin.Context)(){
    //处理逻辑
}).Use(gin.Logger(),gin.Recovery())
```

## 总结

第三方库`HttpRouter`定义了自己内置路由多路复用器(`mux`)，弥补了Go语言`net/http`包中内置路由多路复用器的不足，而Gin框架中路由是在`HttpRouter`库之上的封装，更加易于使用，我们可以定义单个路由接收用户请求，也可以将路由分组，更加易于管理，也易于应用中间件，进行统一拦截处理。