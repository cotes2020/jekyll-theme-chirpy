---
layout: post
title: "元组Tuple和其关键字inout"
date: 2019-06-04 22:12:00.000000000 +09:00
categories: [Swift]
tags: [Swift, Tuple, inout]
---

## 前言

最近在看Swift的开发tips，其中有一个自己比较感兴趣的就是多元组，打算和大家分享一下我理解中多元组的使用以及对关键字inout的理解。

## 元组的定义

元组是Objective-C中没有的数据结构，与数组类似，都是表示一组数据的集合。说到与数组类似，但是也有区别的地方：元组的长度任意，元组中数据可以是不同的数据类型。
元组的定义很简单的，用小括号括起来，里面的元素以逗号隔开，例如：

```swift
let user = ("hjw" ,true, 22)
```

## 元组的基本用法

1.可以通过索引去访问

```swift
let user = ("hjw" ,true, 22)
print(user.0)	// hjw
```

2.可以把元组分解开，然后再去使用

```swift
let (name, isFemale, age) = user
print(name)	// hjw
```

3、如果你只需要一部分的值，可以用“_”来代替忽略掉其他部分

```swift
let (name, _, _) = user
print(name)	//hjw
```

4、上面的使用看起来有点繁琐，有个简化的方式就是定义的时候给每个元素单独命名

```swift
let user = (name:"hjw", isFemale: true, age: 22)
print(user.name)	// hjw
print(user.0) // hjw
```

## inout的作用

声明函数时，在参数前面用inout修饰，在函数内部实现改变外部参数。
需要注意几点：
1、 只能传入变量，不能传入常量和字面量
2、传入的时候，在变量名字前面用&符号修饰表示
3、inout修饰的参数是不能有默认值的，有范围的参数集合也不能被修饰
4、一个参数一旦被inout修饰，就不能再被var和let修饰了
在Swift中inout参数前面使用的&符号可能会给你一种它是传递引用的印象，但是事实并非如此，引用官方的话就是:

> inout参数将一个值传递给函数，函数可以改变这个值，然后将原来的值替换掉，并从函数中传出

## 元组和inout的运用

前面是我对元组简单的一个理解，有了这样的认识之后再去看Swifter里面讲解的多元组的时候就会容易理解。里面举了一个例子是交换输入,普通的程序员普遍的写法：

```swift
func swapMe<T>(a: inout T, b: inout T) {
    let temp = a
    a=b
    b = temp
}
```

使用多元组之后的写法：

```swift
func swapMe<T>(a: inout T, b: inout T) {
    (a,b) = (b,a)
}
```

两个方法达到的目的是一样的，但是他们的实现方法是不一样。最大的区别就是第一个方法开辟了一个额外的空间来完成交换，而使用元组的方法则不去开辟一个额外的空间。他们的调用方法是一样的

```swift
var a: Int = 1
var b: Int = 2
swapMe(a: &a, b: &b)
print(a,b)//2,1
```

如果大家对inout的功能还存在疑惑的地方，我们换一种写法来比较一下：

```swift
func swapMe<T>( a: T, b: T){
    var b = b
    var a = a
    (a,b) = (b,a)
    print(a,b)//1,2
}
swapMe(a: a, b: b)
print(a,b)//2,1
```

在参数前面用inout修饰的方法，在函数内部实现改变外部参数。参数前面没有用inout修饰的方法，在函数执行体里面确实是交换了两个数，但是外部的参数是没有改变.