---
title: "TypeScript元编程实现对象函数参数类型检查"
date: 2020-11-20
permalink: /2020-11-20-ts-metadata/
---
## 环境配置


TypeScript配置：


```typescript
// tsconfig.json
{
    "compilerOptions": {
        "target": "ESNext",
				// 支持装饰器语法
        "experimentalDecorators": true,
				// 支持元数据
        "emitDecoratorMetadata": true
    }
}
```


## Reflect-metadata 获取元数据


使用装饰器+Reflect-metadata，能获取元数据。这属于元编程。


代码实现：


```typescript
import "reflect-metadata";

/**
 * 3种常见的从装饰器上元信息
 */
export function metadataDecorator(target: any, propertyKey: string) {
    console.log(">>>");
    console.log(Reflect.getMetadata("design:type", target, propertyKey));
    console.log(Reflect.getMetadata("design:paramtypes", target, propertyKey));
    console.log(Reflect.getMetadata("design:returntype", target, propertyKey));
}

export class Duty {
    private readonly name: string;
    private readonly createTime: number;
    constructor() {
        this.name = "duty info";
        this.createTime = Date.now();
    }
}

export class Runner {
    @metadataDecorator
    private readonly version: string;

    constructor() {
        this.version = "1.0.0";
    }

    @metadataDecorator
    run(duty: Duty): string {
        return "";
    }
}
```


上述代码的执行结果是：


```shell
>>>
[Function: String]
undefined
undefined
>>>
[Function: Function]
[ [Function: Duty] ]
[Function: String]
```


可以看出来：

- `design:type` 代表成员类型
- `design:paramtypes` 代表函数入参类型
- `design:returntype` 代表函数返回类型

## 实现函数参数类型检查装饰器


实现一个用于检查函数的参数类型的装饰器：`@paramTypeDecorator` 。被装饰的函数，无须在函数内部显式的检查数据类型。


此装饰器的实现：


```typescript
const map = new Map();
export function paramTypeDecorator(
    target: any,
    propertyKey: string,
    descriptor: PropertyDescriptor
) {
    const paramTypes = Reflect.getMetadata(
        "design:paramtypes",
        target,
        propertyKey
    ); // 获得函数参数类型
    map.set(propertyKey, paramTypes); // 保存函数的参数类型

    const originalFunc = descriptor.value;
    // 修改函数的模型行为，在运行前，读取函数的参数类型，并且进行检查
    descriptor.value = function (...args: any[]) {
        const paramTypes = map.get(propertyKey);
        for (let i = 0; i < args.length; ++i) {
            if (!(args[i] instanceof paramTypes[i])) {
                throw new TypeError(
                    `Params[${i}] type should be ${paramTypes[i]?.name}`
                );
            }
        }
        const result = originalFunc.call(this, ...args);
        return result;
    };
}
```


看下在Runner中的使用效果：


```typescript
export class Runner {
    // ... 其它部分和上面的Runner类似

    @paramTypeDecorator
    addDuty(duty: Duty) {}
}

const runner = new Runner();
const duty = new Duty();
// 能通过参数类型检查
runner.addDuty(duty);
```


这段代码会正常运行，类型检查成功。如果最后一句换成：`runner.addDuty({} as any)` ，那么会报错，输出：`TypeError: Params[0] type should be Duty`


