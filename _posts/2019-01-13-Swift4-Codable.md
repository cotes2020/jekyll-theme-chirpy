---
layout: post
title: "Swift 4 Codable 协议"
date: 2019-01-13 22:21:00.000000000 +09:00
categories: [Swift]
tags: [Swift, Codable]
---

WWDC 过去有一段时间了，最近终于有时间空闲，可以静下心来仔细研究一下相关内容。对于开发者来说，本届WWDC 最重要的消息还是得属 Swift 4 的推出。

Swift 经过三年的发展，终于在 API 层面趋于稳定。从 Swift 3 迁移代码到 Swift 4 终于不用像 2 到 3 那样痛苦了。这对开发者来说实在是个重大利好，应该会吸引一大批对 Swift 仍然处于观望状态的开发者加入。

另外 Swift 4 引入了许多新的特性，像是 fileprivate 关键字的限制范围更加精确了；声明属性终于可以同时限制类型和协议了；新的 KeyPath API 等等，从这些改进我们可以看到，Swift 的生态越来越完善，Swift 本身也越来越强大。

而 Swift 4 带来的新特性中，最让人眼前一亮的，我觉得非 **Codable** 协议莫属，下面就来介绍下我自己对 **Codable** 协议踩坑的经验总结。

## 介绍

Swift 由于类型安全的特性，对于像 JSON 这类弱类型的数据处理一直是一个比较头疼的问题，虽然市面上许多优秀的第三方库在这方面做了不少努力，但是依然存在着很多难以克服的缺陷，所以 **Codable** 协议的推出给我们解决类似问题提供了新的思路。

通过查看定义可以看到，**Codable** 其实是一个组合协议，由 `Decodable` 和 `Encodable` 两个协议组成：

```swift
/// A type that can convert itself into and out of an external representation.
public typealias Codable = Decodable & Encodable

/// A type that can encode itself to an external representation.
public protocol Encodable {
    public func encode(to encoder: Encoder) throws
}

/// A type that can decode itself from an external representation.
public protocol Decodable {
    public init(from decoder: Decoder) throws
}
```

`Encodable` 和 `Decodable` 分别定义了 `encode(to:)` 和 `init(from:)` 两个协议函数，分别用来实现数据模型的归档和外部数据的解析和实例化。最常用的场景就是接口 JSON 数据解析和模型创建。但是 **Codable** 的能力并不止于此，这个后面会说。

## 解析 JSON 对象

先来看 `Decodable` 对 JSON 数据对象的解析。Swift 为我们做了绝大部分的工作，Swift 中的基本数据类型比如 `String`、`Int`、`Float` 等都已经实现了 **Codable** 协议，因此如果你的数据类型只包含这些基本数据类型的属性，只需要在类型声明中加上 **Codable** 协议就可以了，不需要写任何实际实现的代码，这也是 **Codable** 最大的优势所在。

比如我们有下面这样一个学生信息的 JSON 字符串：

```swift
let jsonString =
"""
{
    "name": "小明",
    "age": 12,
    "weight": 43.2
}
"""
```

这时候，只需要定义一个 `Student` 类型，声明实现 `Decodable` 协议即可，Swift 4 已经为我们提供了默认的实现：

```swift
struct Student: Decodable {   
    var name: String
    var age: Int
    var weight: Float
}
```

然后，只需要一行代码就可以将 ***小明*** 解析出来了：

```swift
let xiaoming = try JSONDecoder().decode(Student.self, from: jsonString.data(using: .utf8)!)
```

这里需要注意的是, `decode` 函数需要外部数据类型为 `Data` 类型，如果是字符串需要先转换为 `Data` 之后操作，不过像 [**Alamofire**](https://link.juejin.im?target=https%3A%2F%2Fgithub.com%2FAlamofire%2FAlamofire) 之类的网络框架，返回数据原本就是 `Data` 类型的。 另外 `decode` 函数是标记为 `throws` 的，如果解析失败，会抛出一个异常，为了保证程序的健壮性，需要使用 `do-catch` 对异常情况进行处理:

```swift
do {
    let xiaoming = try JSONDecoder().decode(Student.self, from: data)
} catch {
    // 异常处理
}
```

## 特殊数据类型

很多时候光靠基本数据类型并不能完成工作，往往我们需要用到一些特殊的数据类型。Swift 对许多特殊数据类型也提供了默认的 **Codable** 实现，但是有一些限制。

##### 枚举

```swift
{
    ...
    "gender": "male"
    ...
}
```

性别是一个很常用的信息，我们经常会把它定义成枚举：

```swift
enum Gender {
    case male
    case female
    case other
}
```

枚举类型也默认实现了 **Codable** 协议，但是如果我们直接声明 `Gender` 枚举支持 **Codable** 协议，编译器会提示没有提供实现：

![img](../images/swift-codable-01.png)

其实这里有一个限制：枚举类型要默认支持 **Codable** 协议，需要声明为具有原始值的形式，并且原始值的类型需要支持 **Codable** 协议：

```swift
enum Gender: String, Decodable {
    case male
    case female
    case other
}
```

由于枚举类型原始值隐式赋值特性的存在，如果枚举值的名称和对应的 JSON 中的值一致，不需要显式指定原始值即可完成解析。

##### Bool

我们的数据模型现在新增了一个字段，用来表示某个学生是否是少先队员：

```swift
{
    ...
    "isYoungPioneer": true
    ...
}
```

这时候，直接声明对应的属性就可以了:

```
var isYoungPioneer: Bool
```

`Bool` 类型原本没什么好讲的，不过因为踩到了坑，所以还是得说一说： 目前发现的坑是：`Bool` 类型默认**只支持 true/false 形式的 Bool 值解析**。对于一些使用 `0`/`1` 形式来表示 `Bool` 值的后端框架，只能通过 `Int` 类型解析之后再做转换了，或者可以自定义实现 **Codable** 协议。

##### 日期解析策略

说了枚举和 `Bool`，另外一个常用的特殊类型就是 `Date` 了，`Date` 类型的特殊性在于它有着各种各样的格式标准和表示方式，从数字到字符串可以说是五花八门，解析 `Date` 类型是任何一个同类型的框架都必须面对的课题。

对此，**Codable** 给出的解决方案是：定义解析策略。`JSONDecoder` 类声明了一个 `DateDecodingStrategy` 类型的属性，用来制定 `Date` 类型的解析策略，同样先看定义：

```swift
/// The strategy to use for decoding `Date` values.
public enum DateDecodingStrategy {
    
    /// Defer to `Date` for decoding. This is the default strategy.
    case deferredToDate
    
    /// Decode the `Date` as a UNIX timestamp from a JSON number.
    case secondsSince1970
    
    /// Decode the `Date` as UNIX millisecond timestamp from a JSON number.
    case millisecondsSince1970
    
    /// Decode the `Date` as an ISO-8601-formatted string (in RFC 3339 format).
    case iso8601
    
    /// Decode the `Date` as a string parsed by the given formatter.
    case formatted(DateFormatter)
    
    /// Decode the `Date` as a custom value decoded by the given closure.
    case custom((Decoder) throws -> Date)
}
```

**Codable** 对几种常用格式标准进行了支持，默认启用的策略是 `deferredToDate`，即从 **UTC 时间2001年1月1日 **开始的秒数，对应 `Date` 类型中 `timeIntervalSinceReferenceDate` 这个属性。比如 `519751611.125429` 这个数字解析后的结果是 `2017-06-21 15:26:51 +0000`。

另外可选的格式标准有 `secondsSince1970`、`millisecondsSince1970`、[`iso8601`](https://link.juejin.im?target=https%3A%2F%2Fzh.wikipedia.org%2Fwiki%2FISO_8601) 等，这些都是有详细说明的通用标准，不清楚的自行谷歌吧 :)

同时 **Codable** 提供了两种方自定义 `Date` 格式的策略：

- `formatted(DateFormatter)` 这种策略通过设置 `DateFormatter` 来指定 `Date` 格式
- `custom((Decoder) throws -> Date)` `custom` 策略接受一个 `(Decoder) -> Date` 的闭包，基本上是把解析任务完全丢给我们自己去实现了，具有较高的自由度

##### 小数解析策略

小数类型（`Float`／`Double`） 默认也实现了 **Codable** 协议，但是小数类型在 Swift 中有许多特殊值，比如圆周率（`Float.pi`）等。这里要说的是另外两个属性，先看定义：

```swift
/// Positive infinity.
///
/// Infinity compares greater than all finite numbers and equal to other
/// infinite values.
public static var infinity: Double { get }

/// A quiet NaN ("not a number").
///
/// A NaN compares not equal, not greater than, and not less than every
/// value, including itself. Passing a NaN to an operation generally results
/// in NaN.
public static var nan: Double { get }
```

`infinity` 表示正无穷（负无穷写作：`-infinity`），`nan` 表示没有值，这些特殊值没有办法使用数字进行表示，但是在 Swift 中它们是确确实实的值，可以参与计算、比较等。 不同的语言、框架对此会有类似的实现，但是表达方式可能不完全相同，因此如果在某些场景下需要解析这样的值，就需要做特殊转换了。

**Codable** 的实现方式比较简单粗暴，`JSONDecoder` 类型有一个属性 `nonConformingFloatDecodingStrategy` ，用来指定不一致的小数转换策略，默认值为 `throw`， 即直接抛出异常，解析失败。另外一个选择就是自己指定 `infinity`、`-infinity`、`nan` 三个特殊值的表示方式：

```swift
let decoder = JSONDecoder()
decoder.nonConformingFloatDecodingStrategy = .convertFromString(positiveInfinity: "infinity", negativeInfinity: "-infinity", nan: "nan")
// 另外一种表示方式
// decoder.nonConformingFloatDecodingStrategy = .convertFromString(positiveInfinity: "∞", negativeInfinity: "-∞", nan: "n/a")
```

目前看来只支持这三个特殊值的转换，不过这种特殊值的使用场景应该非常有限，至少在我自己五六年的开发生涯中还没有遇到过。

## 特殊数据类型

纯粹的基本数据类型依然不能很好地工作，实际项目的数据结构往往是很复杂的，一个数据类型经常会包含另一个数据类型的属性。比如说我们这个例子中，每个学生信息中还包含了所在学校的信息：

```swift
{
    "name": "小明",
    "age": 12,
    "weight": 43.2
    "school": {
      "name": "市第一中学",
      "address": "XX市人民中路 66 号"
    }
}
```

这时候就需要 Student 和 School 两个类型来组合表示：

```swift
struct School: Decodable {
	var name: String
	var address: String
}
struct Student: Decodable {   
    var name: String
    var age: Int
    var weight: Float
    var school: School
}
```

由于所有基本类型都实现了 **Codable** 协议，因此 `School` 与 `Student` 一样，只要所有属性都实现了 **Codable** 协议，就不需要手动提供任何实现即可获得默认的 **Codable** 实现。由于 `School` 支持了 **Codable** 协议，保证了 `Student` 依然能够获得默认的 **Codable** 实现，因此，嵌套类型的解析同样不需要额外的代码了。

## 自定义字段

很多时候前后端不一定能完全步调一致，观念相同。所以往往后端给出的数据结构中会有一些比较个性的字段名，当然有时候是我们自己。另外有一些框架（比如我正在用的 Laravel）习惯使用蛇形命名法，而 iOS 的代码规范推荐使用驼峰命名法，为了保证代码风格和平台特色，这时候就必须要自行指定字段名了。

在研究自定义字段之前我们需要深入底层，了解下 **Codable** 默认是怎么实现属性的名称识别及赋值的。通过研究底层的 C++ 源代码可以发现，**Codable** 通过巧（kai）妙（guà）的方式，在编译代码时根据类型的属性，自动生成了一个 `CodingKeys` 的枚举类型定义，这是一个以 `String` 类型作为原始值的枚举类型，对应每一个属性的名称。然后再给每一个声明实现 **Codable** 协议的类型自动生成 `init(from:)` 和 `encode(to:)` 两个函数的具体实现，最终完成了整个协议的实现。

所以我们可以自己实现 `CodingKeys` 的类型定义，并且给属性指定不同的原始值来实现自定义字段的解析。这样编译器会直接采用我们已经实现好的方案而不再重新生成一个默认的。

比如 `Student` 需要增加一个出生日期的属性，后端接口使用蛇形命名，JSON 数据如下：

```swift
{
    "name": "小明",
    "age": 12,
    "weight": 43.2
    "birth_date": "1992-12-25"
}
```

这时候在 Student 类型声明中需要增加 `CodingKeys` 定义，并且将 `birthday` 的原始值设置为 `birth_date`：

```swift
struct Student: Codable {
	...
	var birthday: Date
	
	enum CodingKeys: String, CodingKey {
        case name
        case age
        case weight
        case birthday = "birth_date"
    }
}
```

需要注意的是，即使属性名称与 JSON 中的字段名称一致，如果自定义了 `CodingKeys`，这些属性也是无法省略的，否则会得到一个 `Type 'Student' does not conform to protocol 'Codable'` 的编译错误，这一点还是有点坑的。不过在编译时给 `CodingKeys` 补全其他默认的属性的声明在理论上是可行的，期待苹果后续的优化了。

## 可选值

有些字段有可能会是空值。还是用学生的出生日期来举例，假设有些学生的出生日期没有统计到，这时候后台返回数据格式有两种选择，一种是对于没有出生日期的数据，直接不包含 `birth_date` 字段，另一种是指定为空值：`"birth_date": null`

对于这两种形式，都只需要将 birthday 属性声明为可选值即可正常解析：

```swift
...
var birthday: Date?
...
```

## 解析 JSON 数组

**Codable** 协议同样支持数组类型，只需要满足一个前提：只要数组中的元素实现了 **Codable** 协议，数组将自动获得 **Codable** 协议的实现。

使用 `JSONDecoder` 解析时只需要指定类型为对应的数组即可：

```swift
do {
    let students = try JSONDecoder().decode([Student].self, from: data)
} catch {
    // 异常处理
}
```

## 归档数据

归档数据使用 `Encodable` 协议，使用方式与 `Decodable` 一致。

## 导出为 JSON

将数据模型转换为 JSON 与解析过程类似，将 JSONDecoder 更换为 JSONEncoder 即可：

```swift
let data = try JSONEncoder().encode(xiaomin)
let json = String(data: data, encoding: .utf8)
```

JSONEncoder 有一个 outputFormatting 的属性，可以指定输出 JSON 的排版风格，看定义：

```swift
public enum OutputFormatting {
    
    /// Produce JSON compacted by removing whitespace. This is the default formatting.
    case compact
    
    /// Produce human-readable JSON with indented output.
    case prettyPrinted
}
```

- compact

  默认的 compact 风格会移除 JSON 数据中的所有格式信息，比如换行、空格和缩紧等，以减小 JSON 数据所占的空间。如果导出的 JSON 数据用户程序间的通讯，对阅读要求不高时，推荐使用这个设置。

- prettyPrinted

  如果输出的 JSON 数据是用来阅读查看的，那么可以选择 prettyPrinted，这时候输出的 JSON 会自动进行格式化，添加换行、空格和缩进，以便于阅读。类似于上面文中使用的 JSON 排版风格。

## 属性列表(PropertyList)

**Codable** 协议并非只支持 JSON 格式的数据，它同样支持属性列表，即 mac 上常用的 `plist` 文件格式。这在我们做一些系统配置之类的工作时会很有用。

属性列表的解析和归档秉承了苹果API一贯的简洁易用的特点，使用方式 JSON 格式一致，并不需要对已经实现的 **Codable** 协议作任何修改，只需要将 `JSONEncoder` 和 `JSONDecoder` 替换成对应的 `PropertyListEncoder` 和 `PropertyListDecoder` 即可。

属性列表本质上是特殊格式标准的 `XML` 文档，所以理论上来说，我们可以参照系统提供的 Decoder/Encoder 自己实现任意格式的数据序列化与反序列化方案。同时苹果也随时可能通过实现新的 Decoder/Encoder 类来扩展其他数据格式的处理能力。这也正是文章开头所说的，**Codable** 的能力并不止于此，它具有很大的可扩展空间。

## 结语

 **Codable** 比较常用的几个框架，个人比较喜欢

[**ObjectMapper**](https://github.com/tristanhimmelman/ObjectMapper) 使用范型机制进行模型解析，但是需要手动对每一个属性写映射关系，比较繁琐。我自己项目中也是用的这个框架，后来自己对其做了些优化，利用反射机制对基本数据类型实现了自动解析，但是自定义类型仍然需要手动写映射，并且必须继承实现了自动解析的 Model 基类，限制较多。

[**SwiftyJSON**](https://github.com/SwiftyJSON/SwiftyJSON) 简单了解过，其本质其实只是将 JSON 解析成了字典类型的数据，而实际使用时依然需要使用下标方式去取值，非常繁琐且容易出错，不易阅读和维护，个人认为这是很糟糕的设计。

[**HandyJSON**](https://github.com/alibaba/HandyJSON) 是阿里推出的框架，思路与 **Codable** 殊途同归，之前也用过一阵，当时因为对枚举和 `Date` 等类型的支持还不够完善，最终还是用回了**ObjectMapper**。不过目前看来完善程度已经很高了，或许可以再次尝试踩下坑。

总体来说，**Codable** 作为语言层面对模型解析的支持方案，有其自身的优势。不过在灵活性上稍有欠缺，对自定义字段的支持也还不够人性化，期待后续的完善。
