---
layout: post
title: "Swift Protocol协议"
date: 2019-07-22 22:12:00.000000000 +09:00
categories: [Swift]
tags: [Swift, Protocol]
---

## 面向协议编程

依赖倒置原则：告诉我们要面向接口编程

**定义**：高层模块不应该依赖低层模块，二者都应该依赖其抽象；抽象不应该依赖细节；细节应该依赖抽象。

**问题由来**：类A直接依赖类B，假如要将类A改为依赖类C，则必须通过修改类A的代码来达成。这种场景下，类A一般是高层模块，负责复杂的业务逻辑；类B和类C是低层模块，负责基本的原子操作；假如修改类A，会给程序带来不必要的风险。

**解决方案**：将类A修改为依赖接口I，类B和类C各自实现接口I，类A通过接口I间接与类B或者类C发生联系，则会大大降低修改类A的几率。

## 可选接口

Swift 中的protocol 所有方法都必须被实现，不存在@optional 这样的概念。为了实现可选接口有两个办法：（一）@objc  、（二）协议扩展

```swift
// 只能被Class实现，struct和enum不可以
@objc protocol StreetSnapTableViewCellDelegate : NSObjectProtocol{
    // 可选
    @objc optional func deleteSeeFewerPhoto(cell : DJStreetSnapTableViewCell, indexPath: IndexPath?)
    func updateCellSubLikeInfoView(cell : DJStreetSnapTableViewCell, indexPath: IndexPath?, likeCount :Int)
    func updateCellSubSimilarsView(cell:UITableViewCell ,indexPath: IndexPath?, selectedIndex : Int)
}
```

## 协议扩展 protocol extension

> 在Swift2以后，我们可以对一个已有的protocol进行扩展。而扩展中实现的方法作为“实现扩展的类型”的“**默认实现**”
>
> 通过提供protocol的extension,我们为protocol提供了默认实现，这相当于“变相”将protocol中的方法设定为了**optional**

```swift
protocol MyProtocol {
    func method()
}

//  默认实现 ，代替optional
extension MyProtocol {
    func method() {
        print("Called")
    }
}

struct MyStruct:MyProtocol {

}
MyStruct.method()   //输出： Called

struct MyStruct:MyProtocol {
    func method() {
        print("Called in struct")
    }
}
MyStruct.method()    //输出： Called in struct
```

## mutating 修饰方法

> “Structures and enumerations are value types. By default, the properties of a value type cannot be modified from within its instance methods.”
> 译：虽然结构体和枚举可以定义自己的方法，但是默认情况下，实例方法中是不可以修改值类型的属性。

```swift
protocol Vehicle {
    var wheel: Int {get set}
    // protocol的方法被mutating修饰，才能保证struct和enum实现时可以改变属性的值
    mutating func changeWheel()
}

struct Bike: Vehicle {
    var wheel: Int
    mutating func changeWheel() {
        wheel = 4
    }
}

class Car: Vehicle {
    var wheel: Int = 0
    func changeWheel() {
        wheel = 4
    }
}
var bike = Bike(wheel: 2)
bike.changeWheel()
print(bike.wheel)

let car = Car()
car.changeWheel()
print(car.wheel)
```

## static 修饰静态方法

```swift
protocol Vehicle {
    static func wheel() -> Int
}

struct Bike: Vehicle {
    static func wheel() -> Int {
        return 2
    }
}

// static: protocol、enum、struct、class：class 
class Car: Vehicle {
    class func wheel() -> Int {
        return 4
    }
}
```

## protocol 组合

```
1、接口隔离原则：利用 Protocol Composition 可以把协议分得非常细，通过灵活的组合来满足特定要求。
2、「&」这个操作符可不仅仅能组合协议而已，也能组合「Type + Protocol」（类型+协议）。
3、匿名使用：func checkSounds(animal: KittenLike & DogLike)
4、别名：typealias CuteLike = KittenLike & TigerLike & DogLike
```

```swift
// 分工详细：**接口隔离原则**告诉我们在设计接口的时候要精简单一
protocol KittenLike {
    func meow() -> String
}

protocol DogLike {
    func bark() -> String
}

protocol TigerLike {
    func aou() -> String
}

// 不推荐
class MysteryAnimal: KittenLike, DogLike, TigerLike {
    func meow() -> String {
        return "meow"
    }
    
    func bark() -> String {
        return "bark"
    }
    
    func aou() -> String {
        return "aou"
    }
}
// 推荐
typealias CuteLike = KittenLike & TigerLike & DogLike
class CuteAnimal: CuteLike {
    func meow() -> String {
        return "meow"
    }
    
    func bark() -> String {
        return "bark"
    }
    
    func aou() -> String {
        return "aou"
    }
}

struct CheckAnimal {
    // 
    static func checkSounds(animal: KittenLike & DogLike) -> Bool {
        return true
    }
}
```

## Protocols with Associated Type

参考：

- [关联值类型参考博客](https://www.natashatherobot.com/swift-what-are-protocols-with-associated-types/)
- [iOS～关联类型、关联值、关联对象](https://www.jianshu.com/p/1bbe8cbb8f29)

在协议中使用范型:

```swift
//  Protocols with Associated Types 举例
struct Model {
    let age: Int
}

//协议，使用关联类型
protocol TableViewCell {
    associatedtype T
    func updateCell(_ data: T)
}

//遵守TableViewCell
class MyTableViewCell: UITableViewCell, TableViewCell {
    typealias T = Model
    func updateCell(_ data: Model) {
        // do something ...
    }
}
```

