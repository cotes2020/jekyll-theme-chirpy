---
title: "TypeScript 装饰器"
date: 2020-09-19
permalink: /2020-09-19-ts-decorator/
---
## 认识装饰器


首先装饰器只能用在class上，不能用在func上。包括类、类方法、类属性、类方法的参数、类装饰器。


然后装饰器执行的时机，在定义类的时候执行的。如果对于一个class的func，多个装饰器，那么执行顺序是靠近方法的优先执行。


## 类装饰器


类装饰器的入参是类的构造函数。有2种常见用法。


**用法1：当返回新的构造函数时，可以重载类的构造函数**


```typescript
type ClassConstructorType = { new (...args: any[]): {} }; // 类的构造函数类型声明
function userClassDecorator<T extends ClassConstructorType>(constructor: T) {
    // 注意：此种模板语法，为 constructor 参数扩展构造函数的类型声明，否则会类型报错
    return class extends constructor {
        readonly author: string = "dongyuanxin.github.io";
        readonly version: string = "1.1.0";
    };
}
```


**用法2：可以用于处理原型链，例如不支持新增属性**


```typescript
function sealed(constructor: Function) {
    Object.seal(constructor);
    Object.seal(constructor.prototype);
}
```


使用效果：


```typescript
@sealed
@userClassDecorator
class User {
    private readonly name: string;
    constructor(name: string) {
        this.name = name;
    }

    printInfo() {
        console.log(JSON.stringify(this));
    }
}

const user = new User("dongyuanxin");
user.printInfo(); // 输出：{"name":"dongyuanxin","author":"dongyuanxin.github.io","version":"1.1.0"}

```


## 类属性装饰器


类属性装饰器入参有2个：

- 第一个参数：对于实例成员是类的原型对象,对于静态成员来说是类的构造函数
- 第二个参数：成员名字（属性名字）

一个类属性装饰器demo：


```typescript
export function god(target: any, propertyKey: string) {
    console.log(">>> target, propertyKey", target, propertyKey);
}
```


使用效果：


```typescript
export class Duty {
    @god
    private readonly version: string;

    constructor() {
        this.version = "1.0.0";
    }
}
// 代码输出：>>> target, propertyKey Duty {} version
```


## 类方法装饰器


类方法装饰器入参有3个：

- 第一个参数：对于实例成员是类的原型对象,对于静态成员来说是类的构造函数
- 第二个参数：成员的名字(方法装饰器中，就是方法名称)
- 第三个参数：成员的属性描述符**(可以用于改造函数的原生行为)**

一个记录函数运行时间的类方法装饰器：


```typescript
export function timeLog(
    target: any,
    propertyKey: string,
    descriptor: PropertyDescriptor
) {
    const originalFunc = descriptor.value;
    descriptor.value = function (...args: any[]) {
        const startTime = Date.now();
        const result = originalFunc.call(this, ...args);
        console.log(
            `>>> ${propertyKey}() cost time is`,
            Date.now() - startTime,
            "ms"
        );
        return result;
    };
}
```


使用效果：


```typescript
export class Duty {
    private readonly version: string;

    constructor() {
        this.version = "1.0.0";
    }

    @timeLog
    run() {
        const fs = require("fs");
        for (let i = 0; i < 100000; ++i) {
            fs.existsSync("package.json");
        }
    }
}

const duty = new Duty();
duty.run();
// 代码输出：>>> run() cost time is 144 ms
```


## 参考链接


[bookmark](https://tasaid.com/blog/20171011233014.html)


[bookmark](https://saul-mirone.github.io/zh-hans/a-complete-guide-to-typescript-decorator/)


