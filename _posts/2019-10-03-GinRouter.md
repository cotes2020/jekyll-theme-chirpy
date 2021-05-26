---
layout: post
title: "Gin框架中间件"
date: 2019-10-03 19:40:00.000000000 +09:00
categories: [Go Web]
tags: [Go Web, Gin]
---

## Gin中间件

+ Gin中间件是什么？
+ Gin中间件的作用是什么？
+ 要怎么样使用中间件呢？

**Gin中间件的作用**

Gin中间件的作用有两个：

1. Web请求到到达我们定义的HTTP请求处理方法之前，拦截请求并进行相应处理(比如：权限验证，数据过滤等)，这个可以类比为`前置拦截器`或`前置过滤器`，
2. 在我们处理完成请求并响应客户端时，拦截响应并进行相应的处理(比如：添加统一响应部头或数据格式等)，这可以类型为`后置拦截器`或`后置过滤器`。

**Gin中间件的定义**

在Gin框架中，中间件的类型定义如下代码所示，可以看出，中间件实际上就是一个以gin.Context为形参的函数而已，与我们定义处理HTTP请求的Handler本质上是一样的，并没有什么神秘可言。

```go
type HandlerFunc func(*Context)
```

**Gin内置中间件**

在使用Gin框架开发Web应用时，常常需要自定义中间件，不过，Gin也内置一些中间件，我们可以直接使用，下面是内置中间件列表：

```go
func BasicAuth(accounts Accounts) HandlerFunc
func BasicAuthForRealm(accounts Accounts, realm string) HandlerFunc
func Bind(val interface{}) HandlerFunc //拦截请求参数并进行绑定
func ErrorLogger() HandlerFunc       //错误日志处理
func ErrorLoggerT(typ ErrorType) HandlerFunc //自定义类型的错误日志处理
func Logger() HandlerFunc //日志记录
func LoggerWithConfig(conf LoggerConfig) HandlerFunc
func LoggerWithFormatter(f LogFormatter) HandlerFunc
func LoggerWithWriter(out io.Writer, notlogged ...string) HandlerFunc
func Recovery() HandlerFunc
func RecoveryWithWriter(out io.Writer) HandlerFunc
func WrapF(f http.HandlerFunc) HandlerFunc //将http.HandlerFunc包装成中间件
func WrapH(h http.Handler) HandlerFunc //将http.Handler包装成中间件
```

## 中间件的使用

**不使用默认中间件**

使用`gin.Default()`返回的`gin.Engine`时，已经默认使用了`Recovery`和`Logger`中间件，从下面`gin.Default()`方法的源码可以看出：

```go
func Default() *Engine {
    debugPrintWARNINGDefault()
    engine := New()
    engine.Use(Logger(), Recovery())//使用Recovery和Logger中间
    return engine
}
```

当我们不想使用这两个中间件时，可以使用gin.New()方法返回一个不带中间件的gin.Engine对象：

```go
router := gin.New()//不带中间件
```

**全局使用中间件**

直拉使用`gin.Engine`结构体的`Use()`方法便可以在所有请求应用中间件，这样做，中间件便会在全局起作用。

```go
router := gin.New()
router.Use(gin.Recovery())//在全局使用内置中间件
```

**路由分组使用中间件**

更多的时候，我们会根据业务不同划分不同`路由分组(RouterGroup )`,不同的路由分组再应用不同的中间件，这样就达到了不同的请求由不同的中间件进行拦截处理。

```go
router := gin.New()
user := router.Group("user", gin.Logger(),gin.Recovery())
{
    user.GET("info", func(context *gin.Context) {

    })
    user.GET("article", func(context *gin.Context) {

    })
}
```

**单个路由使用中间件**

除了路由分组，单个请求路由，也可以应用中间件，如下：

```go
router := gin.New()
router.GET("/test",gin.Recovery(),func(c *gin.Context){
    c.JSON(200,"test")
})
```

也可以在单个路由中使用多个中间件，如下：

```go
router := gin.New()
router.GET("/test",gin.Recovery(),gin.Logger(),func(c *gin.Context){
    c.JSON(200,"test")
})
```

## 自定义中间件

上面的讲解中，我们看到，虽然Gin提供了一些中间件，我们直接使用即可，但内置中间件可能满足不我们业务开发的需求，在开发过程中我们需要开自己的中间件，这在Gin框架中是非常简单的一件事。

在前面，我们看到Gin框架自带的中间件方法，都是返回`HandlerFunc`类型，其定义如下：

```go
type HandlerFunc func(*Context)
```

HandlerFunc规范了Gin中间件的定义，所以自定义中间件，如下：

```go
//定义中间件
func MyMiddleware(c *gin.Context){
    //中间件逻辑    
}
```

定义好中间件，便可使用中间件，这里演示的是全局使用，也可以在单个路由或路由分组中使用：

```go
router = gin.Default()
router.Use(MyMiddleware)
```

或者，通过自定义方法，返回一个中间件函数，这是Gin框架中更常用的方式：

```go
//定义一个返回中间件的方法
func MyMiddleware(){
    //自定义逻辑
    
    //返回中间件
    return func(c *gin.Context){
        //中间件逻辑
    }
}
```

使用自定义的中间件，注意MyMiddleware方法后面有加括号：

```go
router = gin.Default()
router.Use(MyMiddleware())
```

## 数据传递

当我们在中间件拦截并预先处理好数据之后，要如何将数据传递我们定义的处理请求的HTTP方法呢？可以使用`gin.Context`中的`Set()`方法，其定义如下，`Set()`通过一个key来存储作何类型的数据，方便下一层处理方法获取。

```go
func (c *Context) Set(key string, value interface{})
```

当我们在中间件中通过Set方法设置一些数值，在下一层中间件或HTTP请求处理方法中，可以使用下面列出的方法通过key获取对应数据。

其中，gin.Context的Get方法返回`interface{}`，通过返回exists可以判断key是否存在。

```go
func (c *Context) Get(key string) (value interface{}, exists bool)
```

当我们确定通过Set方法设置对应数据类型的值时，可以使用下面方法获取应数据类型的值。

```go
func (c *Context) GetBool(key string) (b bool)
func (c *Context) GetDuration(key string) (d time.Duration)
func (c *Context) GetFloat64(key string) (f64 float64)
func (c *Context) GetInt(key string) (i int)
func (c *Context) GetInt64(key string) (i64 int64)
func (c *Context) GetString(key string) (s string)
func (c *Context) GetStringMap(key string) (sm map[string]interface{})
func (c *Context) GetStringMapString(key string) (sms map[string]string)
func (c *Context) GetStringMapStringSlice(key string) (smss map[string][]string)
func (c *Context) GetStringSlice(key string) (ss []string)
func (c *Context) GetTime(key string) (t time.Time)
```

示例代码：

```go
//自定义中间件
func MyMiddleware(c *gin.Context){
    c.Set("mykey",10)
}

router := gin.New()
router.GET("test",MyMiddleware,func(c *gin.Context){
    c.GetInt("mykey")//我们知道设置进行的是整型，所以使用GetInt方法来获取
})
```

## 拦截请求与后置拦截

**拦截请求**

我们说过，中间件的最大作用就是拦截过滤请求，比如我们有些请求需要用户登录或者需要特定权限才能访问，这时候便可以中间件中做过滤拦截，当用户请求不合法时，可以使用下面列出的`gin.Context`的几个方法中断用户请求：

下面三个方法中断请求后，直接返回200，但响应的body中不会有数据。

```go
func (c *Context) Abort()
func (c *Context) AbortWithError(code int, err error) *Error
func (c *Context) AbortWithStatus(code int)
```

使用AbortWithStatusJSON()方法，中断用户请求后，则可以返回json格式的数据.

```go
func (c *Context) AbortWithStatusJSON(code int, jsonObj interface{})
```

**后置拦截**

前面我们讲的都是到达我们定义的HTTP处理方法前进行拦截，其实，如果在中间件中调用`gin.Context`的`Next()`方法，则可以请求到达并完成业务处理后，再经过中间件后置拦截处理，`Next()`方法定义如下：。

```go
func (c *Context) Next()
```

在中间件调用`Next()`方法，`Next()`方法之前的代码会在到达请求方法前执行，`Next()`方法之后的代码则在请求方法处理后执行：

```go
func MyMiddleware(c *gin.Context){
    //请求前
    c.Next()
    //请求后
}
```

示例代码

```go
func MyMiddleware(c *gin.Context){
    c.Set("key",1000)//请求前
    c.Next()
    c.JSON(http.StatusOK,c.GetInt("key"))//请求后
}

router := gin.New()
router.GET("test", MyMiddleware, func(c *gin.Context) {
    k := c.GetInt("key")
    c.Set("key", k+2000)
})
router.Run()
```

上面示例程序运行结果为3000，通过上面这样一个简单的示例程序，我们可以看到中间件在请求拦截请求，处理数据并控制Web请求流程的作用。

## 总结

Gin框架中的中间件`middleware`是一块非常重要的知识，我们定义处理HTTP请求的方法前拦截不合法的HTTP请求，或者预先处理好数据，或响应时添加统一的响应头部，因此在使用Gin开发Web应用时，中间件是必用的知识。