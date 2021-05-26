---
layout: post
title: "Gin框架数据绑定"
date: 2019-10-03 01:22:00.000000000 +09:00
categories: [Go Web]
tags: [Go Web, Gin]
---

## 前言

`Gin`框架将处理`HTTP`请求参数以及如何响应等操作都封装到了`gin.Conetxt`结构体，并为`gin.Context`提供了非常多的方法，因此了解`gin.Context`的结构定义与方法，对使用`Gin`框架编写`Web`项目非常重要。

`gin.Context`结构定义代码：

```go
type Context struct {
    Request *http.Request
    Writer  ResponseWriter
    Params Params
    // Keys is a key/value pair exclusively for the context of each request.
    Keys map[string]interface{}
    // Errors is a list of errors attached to all the handlers/middlewares who used this context.
    Errors errorMsgs
    // Accepted defines a list of manually accepted formats for content negotiation.
    Accepted []string
    // contains filtered or unexported fields
}
```

从上面的`gin.Context`的结构定义来看，`gin.Context`封装了`http.Request`和`http.ResponseWriter`

## 获取请求参数

#### Path

path是指请求的url中域名之后从/开始的部分，如掘金首页地址：`https://juejin.im/timeline`，`/timeline`部分便是path，可以使用gin.Context中的Param()方法获取这部分参数。

```go
func (c *Context) Param(key string) string
```

使用Param()方法获取path中的参数：

```go
r.GET("/user/:id",func(c *gin.Context){
    id := c.Param("id")
})
```

除了使用gin.Context的中Param()方法外，还可以用gin.Context中的Params字段获取到path中的参数，Params的定义如下：

```go
type Params []Param
func (ps Params) ByName(name string) (va string)
func (ps Params) Get(name string) (string, bool)
```

使用gin.Context中的Params字段获取path中的参数示例如下:

```
r.GET("/user/:id",func(c *gin.Context){
    id,err := c.Params.Get("id")
    //id := c.Params.ByName("id")
})
```

####Query

query是指url请求地址中的问号后面的部，称为查询参数，如下面地址中，`query=%E6%96%87%E7%AB%A0&type=all`就是查询参数。

```
https://juejin.im/search?query=%E6%96%87%E7%AB%A0&type=all
```

`gin.Context`提供了以下几个方法，用于获取Query部分的参数。

**获取单个参数**

```go
func (c *Context) GetQuery(key string) (string, bool)
func (c *Context) Query(key string) string
func (c *Context) DefaultQuery(key, defaultValue string) string
```

上面三个方法用于获取单个数值，`GetQuery`比`Query`多返回一个error类型的参数，实际上`Query`方法只是封装了`GetQuery`方法，并忽略`GetQuery`方法返回的错误而已，而DefaultQuery方法则在没有获取相应参数值的返回一个默认值。

示例如下：

```go
r.GET("/user", func(c *gin.Context) {
    id,_ := c.GetQuery("id")
    //id := c.Query("id")
    //id := c.DefaultQuery("id","10")
    c.JSON(200,id)
})
```

请求：`http://localhost:8080/user?id=11`

响应：`11`

**获取数组**

> GetQueryArray方法和QueryArray的区别与GetQuery和Query的相似。

```go
func (c *Context) GetQueryArray(key string) ([]string, bool)
func (c *Context) QueryArray(key string) []string
```

示例如下：

```go
r.GET("/user", func(c *gin.Context) {
    ids := c.QueryArray("id")
    //id,_ := c.QueryArray("id")
    c.JSON(200,ids)
})
```

请求：`http://localhost:8080/user?id=10&id=11&id=12`

响应：`["10","11","12"]`

**获取map**

> GetQueryArray方法和QueryArray的区别与GetQuery和Query的相似。

```go
func (c *Context) QueryMap(key string) map[string]string
func (c *Context) GetQueryMap(key string) (map[string]string, bool)
```

示例如下：

```go
r.GET("/user", func(c *gin.Context) {
    ids := c.QueryMap("ids")
    //ids,_ := c.GetQueryMap("ids")
    c.JSON(200,ids)
})
```

请求：`http://localhost:8080/user?ids[10]=Huang`

响应：`{"10":"Huang"}`

#### Body

一般HTTP的Post请求参数都是通过body部分传给服务器端的，尤其是数据量大或安全性要求较高的数据，如登录功能中的账号密码等参数。

gin.Context提供了以下四个方法让我们获取body中的数据，不过要说明的是，下面的四个方法，只能获取`Content-type`是`application/x-www-form-urlencoded`或`multipart/form-data`时`body`中的数据。

下面方法的使用方式与上面获取Query的方法使用类型，区别只是数据来源不同而已，这里便不再写示例程序。

```go
func (c *Context) PostForm(key string) string
func (c *Context) PostFormArray(key string) []string
func (c *Context) PostFormMap(key string) map[string]string
func (c *Context) DefaultPostForm(key, defaultValue string) string
func (c *Context) GetPostForm(key string) (string, bool)
func (c *Context) GetPostFormArray(key string) ([]string, bool)
func (c *Context) GetPostFormMap(key string) (map[string]string, bool)
func (c *Context) GetRawData() ([]byte, error)
```

## 数据绑定

我们使用`gin.Context`提供的方法获取请求中通过`path`、`query`、`body`带上来的参数，但使用前面的那些方法，并不能处理请求中比较复杂的数据结构，比如Content-type为application/json或application/xml时，其所带上的数据会很复杂，因此我们需要使用另外一种方法获取这些数据，这种方式叫`数据绑定`。

`Gin`框架将数据绑定的操作都封装在gin/binding这个包中，下面是`gin/binding包`定义的常量，说明`gin/binding`包所支持的`Content-type`格式。

```go
const (
    MIMEJSON              = "application/json"
    MIMEHTML              = "text/html"
    MIMEXML               = "application/xml"
    MIMEXML2              = "text/xml"
    MIMEPlain             = "text/plain"
    MIMEPOSTForm          = "application/x-www-form-urlencoded"
    MIMEMultipartPOSTForm = "multipart/form-data"
    MIMEPROTOBUF          = "application/x-protobuf"
    MIMEMSGPACK           = "application/x-msgpack"
    MIMEMSGPACK2          = "application/msgpack"
    MIMEYAML              = "application/x-yaml"
)
```

`gin.binding`包也定义处理不同`Content-type`提交数据的处理结构体，并以变量的形式让其他包可以访问，如下：

```go
var (
    JSON          = jsonBinding{}
    XML           = xmlBinding{}
    Form          = formBinding{}
    Query         = queryBinding{}
    FormPost      = formPostBinding{}
    FormMultipart = formMultipartBinding{}
    ProtoBuf      = protobufBinding{}
    MsgPack       = msgpackBinding{}
    YAML          = yamlBinding{}
    Uri           = uriBinding{}
)
```

实际上并不需要调用`gin/binding`包的代码来完成数据绑定的功能，因为`gin.Context`中已经在`gin.Context`的基础上封装了许多更加快捷的方法供我们使用：

> `gin.Context`封装的相关绑定方法，分为以`Bind`为前缀的系列方法和以`ShouldBind`为前缀的系列方法，这两个系列方法之间的差别在于以Bind为前缀的方法，在用户输入数据不符合相应格式时，会直接返回http状态为400的响应给客户端。

#### 以Bind为前缀的系列方法

**Path**

```go
func (c *Context) BindUri(obj interface{}) error
```

代码示例：

```go
type User struct {
    Uid      int    //用户id
    Username string //用户名
}
func main() {
    r := gin.Default()
    r.GET("/bind/:uid/username", func(c *gin.Context) {
        var u User
        e := c.BindUri(&u)
        if e == nil{
            c.JSON(200,u)
        }
    })
    r.Run()
}
```

请求：`http://localhost:8080/bind/1/小文`

输入：`{1,"小文"}`

**Query**

```go
func (c *Context) BindQuery(obj interface{}) error
```

代码示例：

```go
r.GET("/bind/:uid/username", func(c *gin.Context) {
    var u User
    e := c.BindQuery(&u)
    if e == nil{
        c.JSON(200,u)
    }
})
```

请求：`http://localhost:8080/bind?uid=1&username=小文`

输出：`{1,"小文"}`

**Body**

当我们在`HTTP`请求中`Body`设置不同数据格式，需要设置相应头部`Content-Type`的值，比较常用为`json`、`xml`、`yaml`，`gin.Context`提供下面三个方法绑定对应Content-type时body中的数据。

```go
func (c *Context) BindJSON(obj interface{}) error
func (c *Context) BindXML(obj interface{}) error
func (c *Context) BindYAML(obj interface{}) error
```

除了上面三个方法外，更常用的Bind()方法，Bind()方法会自动根据Content-Type的值选择不同的绑定类型。

```go
func (c *Context) Bind(obj interface{}) error
```

示例

```go
r.POST("bind",func(c *gin.Context){
    u := User{}
    c.Bind(&u)
})
```

上面几个方法都是获取固定Content-type或自动根据Content-type选择绑定类型，我们也可以使用下面两个方法自行选择绑定类型。

> 下面两个方法的第二个参数值是gin.binding中定义好的常量，我们在上面讲过。

```go
func (c *Context) BindWith(obj interface{}, b binding.Binding) error
func (c *Context) MustBindWith(obj interface{}, b binding.Binding) error
```

示例

```go
r.POST("bind",func(c *gin.Context){
	u := User{}
	c.BindWith(&u,binding.JSON)
    c.MustBindWith(&u,binding.JSON)
})
```

#### 以ShouldBind为前缀的系列方法

以ShouldBind为前缀的相应的方法与以Bind为前缀的方法使用基本相同，因此下面没有相应演示的代码。

**Path**

```go
func (c *Context) ShouldBindUri(obj interface{}) error
```

**Query**

```go
func (c *Context) ShouldBindQuery(obj interface{}) error
```

**Body**

```go
func (c *Context) ShouldBind(obj interface{}) error
func (c *Context) ShouldBindJSON(obj interface{}) error
func (c *Context) ShouldBindXML(obj interface{}) error
func (c *Context) ShouldBindYAML(obj interface{}) error
func (c *Context) ShouldBindBodyWith(obj interface{}, bb binding.BindingBody) (err error)
func (c *Context) ShouldBindWith(obj interface{}, b binding.Binding) error
```

## 总结

`Gin`框架在`net/http`包的基础上封装了许多的方法，让我们可以接收客户端传递上来的各种不同格式的数据，但是从客户端得到的数据之后，还是要验证数据是否合法或是否我们想要的，这是`Gin`框架中有关`数据验证器`的知识了，有机会再写写这方面的文章。