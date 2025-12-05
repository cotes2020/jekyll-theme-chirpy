---
title: "TypeScript基础实践"
url: "2019-08-27-typescript-notes"
date: 2019-08-27
---

## 联合类型与交叉类型


联合类型：使用`|`作为标记，常用于对象类型是多种类型之一。


交叉类型：使用`&` 作为标记，创建新类型，拥有多个类型的所有定义。


```typescript
interface Student {
    name: string;
    school: string;
}
interface People {
    name: string;
    age: number;
}
// 作为交叉类型，两个接口的定义必须都符合
const a: Student & People = {
    name: "",
    school: "",
    age: 1
};
// 作为联合类型，只能访问接口公有定义
let b: Student | People = a;
console.log(b.name);
// console.log(b.age) // error：编译错误

```


## 数组、元祖与枚举


数组的定义，可以使用范型，也可以使用`[]`：


```typescript
const nums: Array<number> = [1, 2, 3];
const nums2: number[] = [1, 2, 3]; // 更推荐
```


元祖使用`[]`声明，枚举使用`enum`声明。它们的应用场景比较少。


```typescript
// 枚举👇
enum Days {
    Sun, // 0
    Mon, // 1
    Tue, // 2
    Wed, // 3
    Thu, // 4
    Fri, // 5
    Sat // 6
}
// 元祖👇
let yuanzu: [string, number];
yuanzu[0] = "Xcat Liu";
yuanzu[1] = 1;
// yuanzu[2] = 'fwef' // error：编译错误
```


## 类型断言


当「类型推断」无法判定类型的时候，需要利用「类型断言」来告诉编译器对象的类型。


对于类型断言，推荐使用：`as`。而不是在对象前使用`<>`，这和 jsx 语法有冲突。


```typescript
interface LogInfo {
    time: Date;
    info: string;
    level?: "log" | "error" | "warning";
}
const obj1: LogInfo = {
    time: new Date(),
    info: "obj1"
};
const obj2 = {};
const logInfo = process.env.NODE_ENV === "development" ? obj2 : obj1;
// console.log(logInfo.info) // error: 编译报错
console.log((logInfo as LogInfo).info); // 类型断言，指明接口类型为：LogInfo
```


在 node 开发中，通常使用`<>`，而不是`as`：


```typescript
interface Person {
    prop1: string;
    prop2: string;
    prop3: string;
}

type someProps = Record<12 | keyof Person, string>;
let dic = <someProps>{};
```


## 类型别名


这功能非常好用了，比如声明一个联合声明：`'log' | 'info' | 'error'`。许多地方要用到，总不能每次都写一遍，因此：


```typescript
type LogLevel = "log" | "info" | "error";
```


## 函数


### 默认参数与剩余参数


默认参数的语法是`=`，不能给「可选参数」。


```typescript
function buildName(firstName: string = "默认值", lastName?: string) {}
```


「剩余参数」是数组类型，并且元素类型无法确定，因此指定为 `any`:


```typescript
function push(array: any[], ...items: any[]) {
    items.forEach(function(item) {
        array.push(item);
    });
}
```


### Async Function


对于`async/await`函数来说，返回值是 Promise 对象。Promise 对象的类型定义如下：


```typescript
interface Promise<T> {
    /**
     * Attaches a callback that is invoked when the Promise is settled (fulfilled or rejected). The
     * resolved value cannot be modified from the callback.
     * @param onfinally The callback to execute when the Promise is settled (fulfilled or rejected).
     * @returns A Promise for the completion of the callback.
     */
    finally(onfinally?: (() => void) | undefined | null): Promise<T>;
}
```


因此，asnyc 返回结果的类型需要用到范型语法：


```typescript
async function demo(): Promise<number> {
    return 1;
}
```


### 函数重载


ts 的函数重载和 c++、java 等语言中的函数重载不一样。ts 函数重载最终，还是编译成一个函数（c 语言等是编译成不同函数）。


它的目的是为了提供编译器进行更多种类的类型判断，而不需要使用“类型断言”技术。在定义函数重载的时候，要按照「精确 => 宽泛」的级层来定义函数。


```typescript
function reverse(x: number): number;
function reverse(x: string): string;
function reverse(x: number | string): number | string {
    if (typeof x === "number") {
        return Number(
            x
                .toString()
                .split("")
                .reverse()
                .join("")
        );
    } else if (typeof x === "string") {
        return x
            .split("")
            .reverse()
            .join("");
    }
}
```


## 接口


### 任意属性


一个接口允许有任意的属性：


```typescript
interface Person {
    name: string;
    [propName: string]: any;
}
let tom: Person = {
    name: "Tom",
    gender: "male"
};
```


### 索引签名


借助「索引签名」，可以进一步规范存储对象的结构。索引签名的参数可以是 number 或 string。


例如下面，Info 接口就是所有的字符串字读的值必须为 number 类型：


```typescript
interface Info {
    [propName: string]: number;
}
interface Info {
    [propName: string]: number;
    x: string; // 编译error：x不符合索引签名的定义
}
```


当然，此处也可以同时拥有两种类型的索引签名：


```typescript
interface People {
    [name: string]: number | string;
    [age: number]: number;
}
```


如果想规定「有限」的字符串字面量，借助 `in` 即可：


```typescript
type LogLevel = "log" | "info" | "warning" | "error" | "success";
const localLogFile: {
    [level in LogLevel]?: string | void;
} = {
    info: "info.log",
    warning: "warning.log",
    error: "error.log",
    success: "success.log"
};
```


### 接口组合


```typescript
interface MongoConf {
    host: string;
    port: number;
    db: string;
}
interface ProcessConf {
    pid: number;
    mongodb: MongoConf;
}
```


## 类型声明复用


在 typescript 的编译器中，类型定义是可以导出的。比如上面定义的 `MongoConf` 就可以 export 出来，给别的文件使用：


```typescript
// a.ts
export interface MongoConf {
    host: string;
    port: number;
    db: string;
}
// b.ts
import { MongoConf } from "./a.ts";
```


