---
layout: post
title: "Go Json编码和解码"
date: 2019-10-02 01:10:00.000000000 +09:00
categories: [Go Web]
tags: [Go Web]
---

在开发应用程序时，客户端(前端页面或APP)与服务端交互是在所难免的，在交互过程传递数据时，最通用和流行格式便是JSON，Go语言提供了encoding/json包，用于处理JSON数据的编码与解码。

> 除了JSON，XML也常用于前后端的数据交互，不过由于简洁性、可读性和流行程度，JSON用得更加广泛。

## JSON简介

**什么是JSON**

JSON全称为Javascript Object Notation，一种数据结构化交互的标准协议，易于阅读与编写，所以在数据交互时广泛使用。

**JSON中的数据类型**

- 数字：有十进制和科学记数学两种表示方式。
- 字符串：使用双引号表示的Unicode字符序列。
- 布尔：true或者false。
- 对象：花括号(`{}`)括起来的一个或多个键值对(key/value)，用`逗号(,)`隔开，最后一个键值对后面不能有逗号，键必是双引号(`""`)引起来的字符串，而值则可是作意类型(布尔、数字、对象、数组、字符串)。
- 数组：中括号(`[]`)括号值的集合，这些值可是任意类型(布尔、数字、对象、数组、字符串)。

> 下面的示例，是一个数组中包括两个对象。

```json
[{"id":1,"username":"xiaoming","gender":1,"email":"xiaoming@163.com"},{"id":2,"username":"xiaohong","gender":2,"email":"xiaohong@163.com"}]
```

**Json与Go结合**

使用encoding/json处理JSON编码与解码时，就必须处理好JSON数据类型与Go语言数据类型的对应关系。

- JSON的数字、字符串、布尔等在Go语言中相应内置数据类型一一对应。
- JSON的数组则对应Go的数组或Slice(切片)。
- JSON的对象则对应Go的struct(结构体)或map。

> 编码一个结构体时，结构体中只有首字母大写的成员才会被编码，首字母小写的成员会被忽略，另外，结构体中字段后面允许使用反引号声明成员的Tag，用于说明成员的元信息。

```go
type Member struct {
    Id       int    `json:"id"`
    Username string `json:"username"`
    Sex      uint   `json:"gender"`
    Email    string `json:"email"`
}
```

> 上面的结构体Member中，我们定义了四个成员，并声明了每个成员的Tag信息， 其中Sex的Tag信息声明为gender，所以编码后的结果为：

```json
[{"id":1,"username":"xiaoming","gender":1,"email":"xiaoming@163.com"},{"id":2,"username":"xiaohong","gender":2,"email":"xiaohong@163.com"}]
```

## 编码

将Go语言的数据序列化为JSON字符串的操作，称为编码;编码后的结果为一个JSON格式的字符串。

**json.Marshal函数**

> 使用json.Marshal函数可以直接编码任意数据类型。

```go
import (
    "encoding/json"
    "fmt"
)
func main() {
members := []Member{
    {
        Id:1,
        Username:"小明",
        Sex:1,
        Email:"xiaoming@163.com",
    },
    {
        Id:2,
        Username:"小红",
        Sex:1,
        Email:"xiaohong@163.com",
    },
    {
        Id:3,
        Username:"小华",
        Sex:2,
        Email:"xiaohua@163.com",
    },
}
    data,_ := json.Marshal(members)
    fmt.Printf("%s",data)
}
```

运行结果：

```json
[{"id":1,"username":"小明","gender":1,"email":"xiaoming@163.com"},{"id":2,"username":"小红","gender":1,"email":"xiaohong@163.com"},{"id":3,"username":"小华","gender":2,"email":"xiaohua@163.com"}]
```

**json.Encoder**

json.Marshal实际上只是对json.Encoder的封装,因此使用json.Encoder同样可以编码JSON。

```go
func main(){
    b := &bytes.Buffer{}
    encoder := json.NewEncoder(b)
    err := encoder.Encode(members)
    if err != nil{
	    panic(err)
    }
    fmt.Println(b.String())
}
```

## 解码

将JSON字符串反序列化为Go相对应类型的作品，称为解码。

**json.Unmarshal函数**

json.Unmarshal与json.Marshal函数相反，用于解码JSON字符串。

```go
func main() {
    str := `[
    {
        "id": 1,
        "username": "小明",
        "gender": 1,
        "email": "xiaoming@163.com"
    },
    {
        "id": 2,
        "username": "小红",
        "gender": 1,
        "email": "xiaohong@163.com"
    },
    {
        "id": 3,
        "username": "小华",
        "gender": 2,
        "email": "xiaohua@163.com"
    }
    ]`
    b := bytes.NewBufferString(str)
    var members []Member
    err := json.Unmarshal(b.Bytes(),&members)
    if err != nil{
        panic(err)
    }
    fmt.Println(members)
}
```

运行结果：

```
[{1 小明 1 xiaoming@163.com} {2 小红 1 xiaohong@163.com} {3 小华 2 xiaohua@163.com}]
```

**json.Decoder**

```go
func main(){
    b := bytes.NewBufferString(str)
    var members []Member
	decoder := json.NewDecoder(b)
	err = decoder.Decode(&members)
	if err != nil{
		panic(err)
	}
	fmt.Println(members)
}
```

## 小结

Go语言中的encoding/json提供了对JSON数据的编码与解码的各种便捷的方法，我们只要直接使用便完成完成有关JSON的各种处理操作，非常简单方便。