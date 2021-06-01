---
layout: post
title: "Swift 5 中使用原始字符串"
date: 2019-07-15 23:42:00.000000000 +09:00
categories: [Swift]
tags: [Swift, Original String]
---

## 前言

Swift 5 中的`原始字符串(raw string)`让我们能够编写出更自然的字符串，尤其是在使用`反斜杠`和`引号`时。正如将在下面看到的那样，在某些情况下效果是很明显的，如正则表达式。

我之前在 `What’s new in Swift 5.0` 一文中介绍了 Swift 5 中的所有新功能，甚至还有一个专门用于[跟踪 Swift 最新功能的网站](https://www.whatsnewinswift.com)。在本文中我想谈谈如何在 Swift 5 中使用原始字符串，同时还提供了一些有用的详细示例，以便更清楚地了解它们的用处。

如果你曾经问过自己“Swift 中的那些带标签 `#` 的字符串是什么？” 的话，你应该能在这篇文章中找到答案！

> 提示：原始字符串是完全可选的 - 重要的是你至少知道它们是什么，这样你就可以在某段代码中看到它们时知道这是什么，而不一定需要在你自己的代码中使用它们。

## 什么是原始字符串？

在 Swift 5 中，我们能够使用 `#` 符号来指定自定义字符串的分割符。 当我们使用带 `#` 的字符串时，它会影响到 Swift 解析字符串中特殊字符的方式：`\` 不再作为转义字符，所以 `\n` 字面意思是反斜杠跟着 `“n”` 而不是换行符，而 `\(variable)` 不再表示字符串插值，而是实实在在的字符串。

这样，以下两个字符串是相同的：

```swift
let regularString = "\\Hello \\World"
let rawString = #"\Hello \World"#
```

请注意，在第二个示例中，字符串以 `#` 开头和结尾，这标志着它是一个原始字符串。

现在可以在字符串内使用相同的 `#` 符号，用于标记特殊字符。例如，如果要使用字符串插值，现在应该使用 `\#(variableName)` 而不是 `\(variableName)`，如下所示：

```swift
let name = "Taylor"
let greeting = #"Hello, \#(name)!"#
print(greeting)
```

我们也可以将 `#` 与多行字符串一起使用，如下所示：

```swift
let message = #"""
This is rendered as text: \(example).
This uses string interpolation: \#(example).
"""#
```

## 使用分隔符

虽然这是理论上应该永远不需要的功能，但可以在字符串周围添加更多 `#`，以创建更多的唯一的字符串分隔符。

例如，以下这些都创建相同的字符串：

```swift
let zero = "This is a string"
let one = #"This is a string"#
let two = ##"This is a string"##
let three = ###"This is a string"###
let four = ####"This is a string"####
```

这种情况存在的原因是我们想根据自己的需要来结束字符串，这样当你需要在字符串中使用 `"#` 这种比较少的情形时，也不会遇到问题。

应该强调的是，这种情况非常少见。例如，你想写一个字符串，如 `My dog said "woof"#gooddog` -- 注意在 `"woof"` 后面没有空格，后面直接跟了一个 Twitter 风格的标签 `#gooddog`。如果只使用单个分割符的原始字符串，Swift 会将 `#gooddog` 中的 `#` 视为结束符，所以我们需要如下处理：

```swift
let str = ##"My dog said "woof"#gooddog"##
```

## 为什么原始字符串有用？

Swift Evolution 在原始字符串的 proposal 中列出了三个使用原始字符串的例子。具体来说，是以下情形的代码：

- 被转义掩盖了。转义会损害代码审查和验证。
- 已经转义了。转义的内容不应由编译器预先解释。
- 无论是为了测试还是仅更新源，都需要在源和代码之间轻松传输。

前两个是最有可能影响你的：向已经转义的字符串添加转义通常会使代码更难以阅读。

作为一个例子，让我们来看看正则表达式。假设我们有一个像这样的字符串：

```swift
let message = #"String interpolation looks like this: \(age)."#
```

这里使用原始字符串来展示字符串插值的语义而不是实际使用它 - 字符串 `(age)` 将出现在文本中，而不是被 `age` 的实际值替换。

如果我们想要创建一个正则表达式来查找所有字符串插值，我们将以 `\([^)])` 开头。这表示着“反斜杠，左括号，一个或多个不是右括号的字符，然后是右括号。（如果你还没有使用达正则表达式，建议看下 [Beyond Code 这本书](https://www.hackingwithswift.com/store/beyond-code).

但是，我们不能在 Swift 中使用它 - 因为这是无效的：

```swift
let regex = try NSRegularExpression(pattern: "\([^)])")
```

Swift将 `\` 视为转义字符，并假定我们正在尝试在正则表达式中使用字符串插值。所以，我们需要两个反斜杠来做转义，如下所示：

```swift
let regex = try NSRegularExpression(pattern: "\\([^)]+)")
```

But now there’s a second problem: when that string reaches the regex system it will be read as ([^)]), so the regex system will assume we’re escaping the opening parenthesis as opposed to typing a literal backslash, so we need to add another escape for the regex system:

但现在又有第二个问题：当正则表达式系统处理该字符串时，会将 `\([^]])` 作为输入，因此正则表达式系统将假设我们正在转义左括号而不是将 `\` 当作文本处理，所以我们需要为正则表达式系统添加另一个转义：

```swift
let regex = try NSRegularExpression(pattern: "\\\([^)]+)")
```

而这时 Swift 又会抱怨，因为它认为我们要同时转义反斜杠并括号，所以我们需要第四个反斜杠：

```swift
let regex = try NSRegularExpression(pattern: "\\\\([^)]+)")
```

是的，现在有四个反斜杠：一个是我们想要匹配的，一个是在 Swift 中用于转义的，一个是在正则表达式引擎中用于转义的，另一个是转义正在使用 Swift 中的一个正则表达式引擎（太绕）。

然而这个正则表达式仍然无法正常使用。

你看，我们还需要转义我们想要匹配的左括号和右括号，这意味着完整的正则表达式是这样的：

```swift
let regex = try NSRegularExpression(pattern: "\\\\\\([^)]+\\)")
```

请记住，我们在正则表达式引擎中添加 `\` 以转义 `(` ，同时在 Swift 中也要添加了一个 `\` 以转义正则表达式的引用。

如果我们使用原始字符串，我们仍然需要转义正则表达式引擎的字符：为了匹配 `\` 我们必须写 `\`，为了匹配 `(` 我们必须写 `(`。但是，至少我们不再需要为Swift添加额外的转义字符。

所以，我们最终只需要一半的 `\`：

```swift
let regex = try NSRegularExpression(pattern: #"\\\([^)]+\)"#)
```

该正则表达式模式没有 Swift 独有的转义，因此您可以在 [regex101.com](https://regex101.com) 等网站上试用它而无需修改。