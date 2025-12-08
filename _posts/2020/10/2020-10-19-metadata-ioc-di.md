---
title: "认识元编程、控制反转(IoC)以及依赖注入(DI)"
date: 2020-10-19
permalink: /2020-10-19-metadata-ioc-di/
categories: ["C工作实践分享"]
---
## 元编程


### 定义


**狭义来说，应该是指「编写能改变语言语法特性或者运行时特性的程序」**。换言之，一种语言本来做不到的事情，通过你编程来修改它，使得它可以做到了，这就是元编程。


### 在 ES6 中的体现


从这个角度来看，Proxy、Reflect是js的元编程。并且装饰器中，为类方法的参数设置要求（类型、是否必须）等都是元编程，因为js本身或者ts本身（只能声明类型、不能校验或者转化）都做不到，通过编码做到了。


**Proxy**：可以改变对象的默认行为。例如可以拦截对象的读、写等操作。


**Reflect**：反射，可以获取元信息。配合 [reflect-metadata](https://github.com/rbuckton/reflect-metadata)，更加强壮。例如可以在对象未实例化时，在对象外部，获取构造函数参数类型、方法参数类型、属性类型。


### 更多资料


[bookmark](https://www.zhihu.com/question/23856985/answer/25965835)


[bookmark](https://zhuanlan.zhihu.com/p/56114395)


[bookmark](https://www.kancloud.cn/kancloud/you-dont-know-js-es6-beyond/520583)


## 控制反转(IoC)


控制反转是面向对象编程中的一种原则，用于降低代码之间的耦合度。


**传统方法**：在类的内部主动创建依赖对象，这样将导致类与类之间耦合度非常高，并且不容易测试。


```typescript
import { ModuleA } from './module-A';
import { ModuleB } from './module-B';

class ModuleC {
  constructor() {
    this.a = new ModuleA();

    this.b = new ModuleB(this.a);
  }
}

@Injectable()
class ModuleC {
  constructor(
    private a: ModuleA,
    private b: ModuleB
  ) {}
}
```


**控制反转：将创建和查找依赖对象的控制权交给了IoC容器**，这样对象与对象之间就是松散耦合了，方便测试与功能复用


```typescript
// 文件1：container.js
import { ModuleA } from './module-A';
import { ModuleB } from './module-B';
// 将模块统一注入到IoC容器中
export const iocContainer = new Container();
container.bindModule(ModuleA);
container.bindModule(ModuleB);

// 文件2: ModuleC文件
import { container } from './container';
class ModuleC {
  constructor() {
    this.a = container.getModule('ModuleA');
    this.b = container.getModule('ModuleB');
  }
}
```


此时，对于ModuleC来说，可以对接不同的容器，而不同容器中的ModuleA和ModuleB各不相同，相比于传统方法，可以做到在不改动ModuleC的情况下，实现不同类的注入。


同理，对于测试来说，可以mock ModuleA和mock ModuleB，将测试的点聚焦于ModuleC本身的逻辑。


## 依赖注入(DI)


依赖注入是控制反转最常见的一种应用方式（或者实现方法），即通过控制反转，在对象创建的时候，自动注入一些依赖对象。


在TS中，使用装饰器和元编程，可以实现依赖注入。看看NestJS中DI的写法：


```typescript
import { Injectable } from '@nestjs/common';
import { TcbService } from './../../services/tcb.service';

// 通过装饰器@Injectable()让依赖（TcbService）注入到类实例中
@Injectable({ scope: Scope.DEFAULT })
export class SearchService {
    constructor(private readonly tcbService: TcbService) { }

    public async searchPassages(): Promise<> {
        const db = this.tcbService.getDB(); // 可以访问被注入的依赖
        // ......
    }
}
```


实现原理请看： [NestJS源码：实现依赖注入(DI)](https://www.notion.so/3a6909fd9fb74b608dba44e4aa0e4b71) 


## 参考链接


[bookmark](https://zhuanlan.zhihu.com/p/72323250)


