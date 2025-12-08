---
title: "TypeScript 实现 @required 参数装饰器"
date: 2020-11-23
permalink: /2020-11-23-ts-required/
categories: ["B源码精读", "Others"]
---
## 参数装饰器作用


可以借助参数装饰器，提前记录参数的某些信息。


## 实现思路


**首先，参数装饰器单独声明是没用的。**需要在其定义中，设定某个参数的要求（元信息metadata），例如此参数不能为空、必须为某个类型。


**然后借助函数装饰器改写函数运行逻辑**，这里可以根据参数装饰器保存的对参数的要求，在函数运行前一个个校验。


**总结来说，参数装饰器记录参数元信息，函数装饰器读取元信息并且进行校验。**


## 代码实现


被`required`装饰过的函数参数，它们的位置会作为当前方法的metadata保存下来。


被`validateEmptyStr` 装饰过的函数，它的默认行为会被改写。会在运行前遍历参数，并且检查其是否在当前方法的metadata中保存下标数组中，如果在，则不能为空，否则会报错。


```typescript
// 定义一个私有 key
const requiredMetadataKey = Symbol.for('router:required')

// 定义参数装饰器，大概思路就是把要校验的参数索引保存到成员中
const required = function (target, propertyKey: string, parameterIndex: number) {
  // 属性附加
  const rules = Reflect.getMetadata(requiredMetadataKey, target, propertyKey) || []
  rules.push(parameterIndex)
  Reflect.defineMetadata(requiredMetadataKey, rules, target, propertyKey)
}

// 定义一个方法装饰器，从成员中获取要校验的参数进行校验
const validateEmptyStr = function (target, propertyKey: string, descriptor: PropertyDescriptor) {
  // 保存原来的方法
  let method = descriptor.value
  // 重写原来的方法
  descriptor.value = function () {
    let args = arguments
    // 看看成员里面有没有存的私有的对象
    const rules = Reflect.getMetadata(requiredMetadataKey, target, propertyKey) as Array<number>
    if (rules && rules.length) {
      // 检查私有对象的 key
      rules.forEach(parameterIndex => {
        // 对应索引的参数进行校验
        if (!args[parameterIndex]) throw Error(`arguments${parameterIndex} is invalid`)
      })
    }
    return method.apply(this, arguments)
  }
}

class User {
  name: string
  id: number
  constructor(name:string, id: number) {
    this.name = name
    this.id = id
  }

  // 方法装饰器做校验
  @validateEmptyStr
  changeName (@required newName: string) { // 参数装饰器做描述
    this.name = newName
  }
}
```


## 为什么不使用`design:paramtypes` 元数据来检查函数参数？


在 [TypeScript元编程实现对象函数参数类型检查](https://www.notion.so/7a93d2b0588f4a4886679f25a133db6c)  一文中，实现参数检验的逻辑只使用了方法装饰器，外加reflect-metadata默认支持的`design:paramtypes` 方法。相较于本文的实现，看起来更简单。


**但是，****`design:paramtypes`** **无法识别TS语法的可选参数**，例如：`changeName(newName?: string)` 。元数据上只有newName参数的类型是String，但是没有「是否可选」的信息。


