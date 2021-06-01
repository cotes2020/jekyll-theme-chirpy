---
layout: post
title: "Swift中String和Substring"
date: 2019-07-11 21:52:00.000000000 +09:00
categories: [Swift]
tags: [Swift,String, Substring]
---

以下所有示例都基于下面这行代码：

```swift
var str = "Hello World!"
```

## 字符串与子字符串

字符串在不同的 Swift 版本中变化比较大。在 Swift 4 中，当需要从一个字符串中获取子串时，我们获取到的是 `Substring` 类型而不是一个 `String` 类型的值。为什么会这样？在 Swift 中，字符串是值类型。这意味着如果想从一个字符串中生成一个新的字符串，则需要做拷贝处理。这对于稳定性是有益的，但效率却不高。

从另一方面讲，一个 `Substring` 则是对其原始 `String` 的一个引用。它不需要做拷贝操作，所以更加高效。但是有另一个问题，假设我们要从一个很长的字符串中获取一个长度为 10 的 Substring，由于这个 Substring 会引用 String，那么只要 Substring 一直存在，String 就必须一直存在。所以，任何时候当处理完 Substring 后，需要将其转换为 String。

```swift
let myString = String(mySubstring)
```

这样只会拷贝子串，而原来的字符串可以被正确地回收。Substring 作为一种类型，本身即表示临时存在的。

在 Swift 4 另一个大的改进是字符串又成为集合了。这意味着在集合上的所有操作，也可以应用在字符串中(如下标、迭代字符、过滤等)。

## String.Index

在我们更多地了解子字符串之前，了解 String 索引如何作用于字符串中的字符会有很大的帮助。

**startIndex 和 endIndex**

- `startIndex` 是开始字符的索引
- `endIndex` 是结束字符的索引

示例

```swift
var str = "Hello, playground"
// character
str[str.startIndex] // H
str[str.endIndex]   // error: after last character
// range
let range = str.startIndex..<str.endIndex
str[range]  // "Hello, playground"
```

基于 Swift 4 的单侧范围(`one-side ranges`)功能，可将范围简化为以下形式之一。

```swift
let range = str.startIndex...
let range = ..<str.endIndex
```

为了清晰起见，在下面的示例中，我将使用完整形式，不过为了更可读，你可能在你的代码中倾向于使用单侧范围。

**after**

如在 `index(after: String.Index)`

- `after` 指的是给定索引后面的字符索引。

示例：

```swift
// character
let index = str.index(after: str.startIndex)
str[index]  // "e"
// range
let range = str.index(after: str.startIndex)..<str.endIndex
str[range]  // "ello, playground"
```

**before**

如在 `index(before: String.Index)`

- `before` 指的是给定索引前面的字符索引。

示例

```swift
// character
let index = str.index(before: str.endIndex)
str[index]  // d
// range
let range = str.startIndex..<str.index(before: str.endIndex)
str[range]  // Hello, playgroun
```

**offsetBy**

如在 `index(String.Index, offsetBy: String.IndexDistance)`

- `offsetBy` 的值可为正也可为负，以指定索引为起始位置。虽然它的类型是 String.IndexDistance，但可以给它一个 Int 值

示例

```swift
// character
let index = str.index(str.startIndex, offsetBy: 7)
str[index]  // p
// range
let start = str.index(str.startIndex, offsetBy: 7)
let end = str.index(str.endIndex, offsetBy: -6)
let range = start..<end
str[range]  // play
```

**limitedBy**

如在 `index(String.Index, offsetBy: String.IndexDistance, limitedBy: String.Index)`

- `limitedBy` 可用于确保偏移量不会导致索引超出范围。这是一个有界的索引。由于偏移量可能超出限制，因此此方法返回 `Optional`。如果索引超出范围，则返回 nil。

示例：

```swift
// character
if let index = str.index(str.startIndex, offsetBy: 7, limitedBy: str.endIndex) {
    str[index]  // p
}
```

如果偏移量为 `77` 而不是 `7`，那么将跳过 `if` 语句。

## 获取子字符串

我们可以使用下标或许多其他方法（例如，前缀，后缀，拆分）从字符串中获取子字符串。但是，我们仍然需要使用 `String.Index` 而不是 `Int` 值来指定索引范围。

**字符串的起始值**

我们可以使用下标(注意 Swift 4 的单侧范围)：

```swift
let index = str.index(str.startIndex, offsetBy: 5)
let mySubstring = str[..<index] // Hello
```

或者 `prefix`：

```swift
let index = str.index(str.startIndex, offsetBy: 5)
let mySubstring = str.prefix(upTo: index) // Hello
```

或者更简单的方式：

```swift
let mySubstring = str.prefix(5) // Hello
```

### 字符串的结束值

使用下标：

```swift
let index = str.index(str.endIndex, offsetBy: -10)
let mySubstring = str[index...] // playground
```

或者 `suffix`：

```swift
let index = str.index(str.endIndex, offsetBy: -10)
let mySubstring = str.suffix(from: index) // playground
```

或者更简单的方式：

```swift
let mySubstring = str.suffix(10) // playground
```

请注意，当使用 `suffix(from: index)` 时，我必须使用 -10 从最后算起。 当只使用 `suffix(x)` 时，则不需要，后缀只占用字符串的最后 x 个字符。

**字符串的范围**

我们再次使用下标：

```swift
let start = str.index(str.startIndex, offsetBy: 7)
let end = str.index(str.endIndex, offsetBy: -6)
let range = start..<end
let mySubstring = str[range]  // play
```

**将 Substring 转换为 String**

不要忘记，当您准备保存子字符串时，应将其转换为 `String`，以便清除旧字符串的内存。

```
let myString = String(mySubstring)
```

## 使用 Int 索引扩展

对字符串使用 `Int` 索引要容易得多。实际上，通过使用基于 Int 的扩展，可以隐藏 String 索引的复杂性。然而，在读完 Airspeed Velocity 和 Ole Begemann 的文章 `Strings in Swift 3` 后，我犹豫不决。此外，Swift 团队故意没有使用 `Int` 索引，而仍然使用 `String.Index`。原因是 Swift 中的 Character 在底层的长度并不完全相同。单个 Swift 字符可能由一个、两个甚至更多 `Unicode` 组成。 因此，每个唯一的 String 必须计算其 Character 的索引。