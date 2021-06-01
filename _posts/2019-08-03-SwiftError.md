---
layout: post
title: "Swift Error 的分类"
date: 2019-08-03 23:12:00.000000000 +09:00
categories: [Swift]
tags: [Swift, Error]
---

## Swift 错误类型的种类

**1.Simple domain error**

简单的，显而易见的错误。这类错误的最大特点是我们不需要知道原因，只需要知道错误发生，并且想要进行处理。用来表示这种错误发生的方法一般就是返回一个 `nil` 值。在 Swift 中，这类错误最常见的情况就是将某个字符串转换为整数，或者在字典尝试用某个不存在的 key 获取元素：

```swift
// Simple Domain Error 的例子

let num = Int("hello world") // nil

let element = dic["key_not_exist"] // nil
```

在使用层面 (或者说应用逻辑) 上，这类错误一般用 `if let` 的可选值绑定或者是 `guard let` 提前进行返回处理即可，不需要再在语言层面上进行额外处理。

**2.Recoverable error**

正如其名，这类错误应该是被容许，并且是可以恢复的。可恢复错误的发生是正常的程序路径之一，而作为开发者，我们应当去检出这类错误发生的情况，并进一步对它们进行处理，让它们恢复到我们期望的程序路径上。

这类错误在 Objective-C 的时代通常用 `NSError` 类型来表示，而在 Swift 里则是 `throw` 和 `Error` 的组合。一般我们需要检查错误的类型，并作出合理的响应。而选择忽视这类错误往往是不明智的，因为它们是用户正常使用过程中可能会出现的情况，我们应该尝试对其恢复，或者至少向用户给出合理的提示，让他们知道发生了什么。像是网络请求超时，或者写入文件时磁盘空间不足：

```swift
// 网络请求
let url = URL(string: "https://www.example.com/")!
let task = URLSession.shared.dataTask(with: url) { data, response, error in
    if let error = error {
        // 提示用户
        self.showErrorAlert("Error: \(error.localizedDescription)")
    }
    let data = data!
    // ...
}

// 写入文件
func write(data: Data, to url: URL) {
    do {
        try data.write(to: url)
    } catch let error as NSError {
        if error.code == NSFileWriteOutOfSpaceError {
            // 尝试通过释放空间自动恢复
            removeUnusedFiles()
            write(data: data, to: url)
        } else {
            // 其他错误，提示用户
            showErrorAlert("Error: \(error.localizedDescription)")
        }
    } catch {
        showErrorAlert("Error: \(error.localizedDescription)")
    }
}
```

**3.Universal error**

这类错误理论上可以恢复，但是由于语言本身的特性所决定，我们难以得知这类错误的来源，所以一般来说也不会去处理这种错误。这类错误包括类似下面这些情形：

```swift
// 内存不足
[Int](repeating: 100, count: .max)

// 调用栈溢出
func foo() { foo() }
foo()
```

我们可以通过设计一些手段来对这些错误进行处理，比如：检测当前的内存占用并在超过一定值后警告，或者监视栈 frame 数进行限制等。但是一般来说这是不必要的，也不可能涵盖全部的错误情况。更多情况下，这是由于代码触碰到了设备的物理限制和边界情况所造成的，一般我们也不去进行处理（除非是人为操成的 bug）。

在 Swift 中，各种被使用 `fatalError` 进行强制终止的错误一般都可以归类到 Universal error。

**4.Logic failure**

逻辑错误是程序员的失误所造成的错误，它们应该在开发时通过代码进行修正并完全避免，而不是等到运行时再进行恢复和处理。

常见的 Logic failure 包括有：

```swift
// 强制解包一个 `nil` 可选值
var name: String? = nil
name!

// 数组越界访问
let arr = [1,2,3]
let num = arr[3]

// 计算溢出
var a = Int.max
a += 1

// 强制 try 但是出现错误
try! JSONDecoder().decode(Foo.self, from: Data())
```

这类错误在实现中触发的一般是 [`assert` 或者 `precondition`](https://github.com/apple/swift/blob/a05cd35a7f8e3cc70e0666bc34b5056a543eafd4/stdlib/public/core/Collection.swift#L1009-L1046)。

`断言的作用范围和错误转换`

和 `fatalError` 不同，`assert` 只在进行编译优化的 `-O` 配置下是不触发的，而如果更进一步，将编译优化选项配置为 `-Ounchecked` 的话，`precondition` 也将不触发。此时，各方法中的 `precondition` 将被跳过，因此我们可以得到最快的运行速度。但是相对地代码的安全性也将降低，因为对于越界访问或者计算溢出等错误，我们得到的将是不确定的行为。

| 函数        | faltaError | precondition | assert |
| ----------- | ---------- | ------------ | ------ |
| -Onone      | 触发       | 触发         | 触发   |
| -O          | 触发       | 触发         |        |
| -Ounchecked | 触发       |              |        |

对于 Universal error 一般使用 `fatalError`，而对于 Logic failure 一般使用 `assert` 或者 `precondition`。遵守这个规则会有助于我们在编码时对错误进行界定。而有时候我们也希望能尽可能多地在开发的时候捕获 Logic failure，而在产品发布后尽量减少 crash 比例。这种情况下，相比于直接将 Logic failure 转换为可恢复的错误，我们最好是使用 `assert` 在内部进行检查，来让程序在开发时崩溃。

## 练习1: app 内资源加载

假设我们在处理一个机器学习的模型，需要从磁盘读取一份预先训练好的模型。该模型以文件的方式存储在 app bundle 中，如果读取时没有找到该模型，我们应该如何处理这个错误？

**方案 1 Simple domain error**

```swift
func loadModel() -> Model? {
    guard let path = Bundle.main.path(forResource: "my_pre_trained_model", ofType: "mdl") else {
        return nil
    }
    let url = URL(fileURLWithPath: path)
    guard let data = try? Data(contentOf: url) else {
        return nil
    }
    
    return try? ModelLoader.load(from: data)
}
```

**方案 2 Recoverable error**

```swift
func loadModel() throws -> Model {
    guard let path = Bundle.main.path(forResource: "my_pre_trained_model", ofType: "mdl") else {
        throw AppError.FileNotExisting
    }
    let url = URL(fileURLWithPath: path)
    let data = try Data(contentOf: url)
    return try ModelLoader.load(from: data)
}
```

**方案 3 Universal error**

```swift
func loadModel() -> Model {
    guard let path = Bundle.main.path(forResource: "my_pre_trained_model", ofType: "mdl") else {
        fatalError("Model file not existing")
    }
    let url = URL(fileURLWithPath: path)
    do {
        let data = try Data(contentOf: url)
        return try ModelLoader.load(from: data)
    } catch {
        fatalError("Model corrupted.")
    }
}
```

**方案 4 Logic failure**

```swift
func loadModel() -> Model {
    let path = Bundle.main.path(forResource: "my_pre_trained_model", ofType: "mdl")!
    let url = URL(fileURLWithPath: path)
    let data = try! Data(contentOf: url)
    return try! ModelLoader.load(from: data)
}
```

**答案**

:::tip

正确答案应该是**方案 4，使用 Logic failure 让代码直接崩溃**。

作为内建的存在于 app bundle 中模型或者配置文件，如果不存在或者无法初始化，在不考虑极端因素的前提下，一定是开发方面出现了问题，这不应该是一个可恢复的错误，无论重试多少次结果肯定是一样的。也许是开发者忘了将文件放到合适的位置，也许是文件本身出现了问题。不论是哪种情况，我们都会希望尽早发现并强制我们修正错误，而让代码崩溃可以很好地做到这一点。

使用 Universal error 同样可以让代码崩溃，但是 Universal error 更多是用在语言的边界情况下。而这里并非这种情况。

:::

## 练习2: 加载当前用户信息时发生错误

我们在用户登录后会将用户信息存储在本地，每次重新打开 app 时我们检测并使用用户信息。当用户信息不存在时，应该进行的处理：

**方案 1 Simple domain error**

```swift
func loadUser() -> User? {
    let username = UserDefaults.standard.string(forKey: "com.onevcat.app.defaults.username")
    if let username {
        return User(name: username)
    } else {
        return nil
    }
}
```

**方案 2 Recoverable error**

```swift
func loadUser() throws -> User {
    let username = UserDefaults.standard.string(forKey: "com.onevcat.app.defaults.username")
    if let username {
        return User(name: username)
    } else {
        throws AppError.UsernameNotExisting
    }
}
```

**方案 3 Universal error**

```swift
func loadUser() -> User {
    let username = UserDefaults.standard.string(forKey: "com.onevcat.app.defaults.username")
    if let username {
        return User(name: username)
    } else {
        fatalError("User name not existing")
    }
}
```

**方案 4 Logic failure**

```swift
func loadUser() -> User {
    let username = UserDefaults.standard.string(forKey: "com.onevcat.app.defaults.username")
    return User(name: username!)
}
```

> 首先肯定排除方案 3 和 4。“用户名不存在”是一个正常的现象，肯定不能直接 crash。所以我们应该在方案 1 和方案 2 中选择。
>
> 对于这种情况，选择**方案 1 Simple domain error 会更好**。因为用户信息不存在是很简单的一个状况，如果用户不存在，那么我们直接让用户登录即可，这并不需要知道额外的错误信息，返回 `nil` 就能够很好地表达意图了。
>
> 当然，我们不排除今后随着情况越来越复杂，会需要区分用户信息缺失的原因 (比如是否是新用户还没有注册，还是由于原用户注销等)。但是在当前的情况下来看，这属于过度设计，暂时并不需要考虑。如果之后业务复杂到这个程度，在编译器的帮助下将 Simple domain error 修改为 Recoverable error 也不是什么难事儿。

## 练习3: 还没有实现的代码

假设你在为你的服务开发一个 iOS 框架，但是由于工期有限，有一些功能只定义了接口，没有进行具体实现。这些接口会在正式版中完成，但是我们需要预先发布给友商内测。所以除了在文档中明确标明这些内容，这些方法内部应该如何处理呢？

**方案 1 Simple domain error**

```swift
func foo() -> Bar? {
    return nil
}
```

**方案 2 Recoverable error**

```swift
func foo() throws -> Bar? {
    throw FrameworkError.NotImplemented
}
```

**方案 3 Universal error**

```swift
func foo() -> Bar? {
    fatalError("Not implemented yet.")
}
```

**方案 4 Logic failure**

```swift
func foo() -> Bar? {
    assertionFailure("Not implemented yet.")
    return nil
}
```

> 正确答案是**方案 3 Universal error**。对于没有实现的方法，返回 `nil` 或者抛出错误期待用户恢复都是没有道理的，这会进一步增加框架用户的迷惑。这里的问题是语言层面的边界情况，由于没有实现，我们需要给出强力的提醒。在任意 build 设定下，都不应该期待用户可以成功调用这个函数，所以 `fatalError` 是最佳选择。

## 练习4: 调用设备上的传感器收集数据

调用传感器的 app 最有意思了！不管是相机还是陀螺仪，传感器相关的 app 总是能带给我们很多乐趣。那么，如果想要调用传感器获取数据时，发生了错误，应该怎么办呢？

**方案 1 Simple domain error**

```swift
func getDataFromSensor() -> Data? {
    let sensorState = sensor.getState()
    guard sensorState == .normal else {
        return nil
    }
    return try? sensor.getData()
}
```

**方案 2 Recoverable error**

```swift
func getDataFromSensor() throws -> Data {
    let sensorState = sensor.getState()
    guard sensorState == .normal else {
        throws SensorError.stateError
    }
    return try sensor.getData()
}
```

**方案 3 Universal error**

```swift
func loadUser() -> Data {
    let sensorState = sensor.getState()
    guard sensorState == .normal, let data = try? sensor.getData() else {
        fatalError("Sensor get data failed!")
    }
    return data
}
```

**方案 4 Logic failure**

```swift
func loadUser() -> Data {
    let sensorState = sensor.getState()
    assert(sensorState == .normal, "The sensor state is not normal")
    return try! sensor.getData()
}
```

> 传感器由于种种原因暂时不能使用 (比如正在被其他进程占用，或者甚至设备上不存在对应的传感器)，是很有可能发生的情况。即使这个传感器的数据对应用是至关重要，不可或缺的，我们可能也会希望至少能给用户一些提示。基于这种考虑，使用**方案 2 Recoverable error** 是比较合理的选择。
>
> 方案 1 在传感器数据无关紧要的时候可能也会是一个更简单的选项。但是方案 3 和 4 会直接让程序崩溃，而且这实际上也并不是代码边界或者开发者的错误，所以不应该被考虑。

## 总结

可以看到，其实在错误处理的时候，选用哪种错误是根据情景和处理需求而定的，我在参考答案也使用了很多诸如“可能”，“相较而言”等语句。虽然对于特定的场景，我们可以进行直观的考虑和决策，但这并不是教条主义般的一成不变。错误类型之间可以很容易地通过代码互相转换，这让我们在处理错误的时候可以自由选择使用的策略：比如 API 即使提供给我们的是 Recoverable 的 throws 形式，我们也还是可以按照需要，通过 `try ?` 将其转为 Simple domain error，或者用 `try !` 将其转为 Logic failure。
