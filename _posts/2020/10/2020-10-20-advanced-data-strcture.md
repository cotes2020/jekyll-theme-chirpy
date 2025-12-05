---
title: "TypeScript进阶实战"
date: 2020-10-20
permalink: /2020-10-20-advanced-data-strcture/
---
## TypeScript 高级数据类型


### Dictionary 和 NumericDictionary


Dictionary：一个对象，键是 string 类型。


NumericDictionary：一个对象，键是 number 类型。


注意：要手动实现下，通过「索引签名」+「范型」。


定义：


```typescript
interface Dictionary<T = any> {
    [index: string]: T;
}

interface NumericDictionary<T = any> {
    [index: number]: T;
}
```


使用起来：


```typescript
const data2: Dictionary<number> = {
    a: 3,
    b: 4,
};
```


### Record


ts 原生支持，看下它的定义：


```typescript
type Record<K extends string | number | symbol, T> = { [P in K]: T };
```


简单来说，对于类型 `Record<KEY_TYPE, VALUE_TYPE>` 声明的对象，键的类型就是 KEY_TYPE，值的类型是 VALUE_TYPE。


**它和直接用 interface 有啥区别？** 一般 Record 可以用来做字段扩展、将值的类型转化为指定的 VALUE_TYPE、将键的类型转化为指定的 KEY_TYPE。


例子：


```typescript
interface Person {
    name: string
    age: number
}

type PersonType = Record<'location' | keyof Person, number>
// type PersonType = {
//     location: number;
//     name: number;
//     age: number;
}
```


### Pick


ts 原生支持：


```typescript
type Pick<T, K extends keyof T> = { [P in K]: T[P] };
```


作用：将某个类型指定键挑出来，组成新的类型。


例如：


```typescript
interface Student {
    name: string;
    age: number;
    score: string;
}

type Person = Pick<Student, "age" | "name">;
// type Person = {
//     age: number;
//     name: string;
// }
```


### Exclude


ts 原生支持：


```typescript
// 提取T包含在U中的类型
type Extract<T, U> = T extends U ? T : never;
```


作用：将联合类型中的指定类型去掉。


例如：


```typescript
type NUMBERS = 0 | 1 | 2 | "a" | "b";

type CORRECT_NUMBERS = Exclude<NUMBERS, "a" | "b">; // CORRECT_NUMBERS 类型是 0 | 1 | 2
```


### Extract


作用：将公有属性提取出来。


例如：


```typescript
type NUMBERS = 0 | 1 | 2 | "a" | "b";

type PARTIAL_NUMBERS = Extract<NUMBERS, 0 | 1>; // PARTIAL_NUMBERS 类型是 0 | 1
```


### 函数相关-Parameters


作用：获取函数参数类型。


例如：


```typescript
function getPerson(name: string, value?: number): any {}

type F = Parameters<typeof getPerson>; // F 类型是 [name: string, value?: number]
```


### 函数相关-ReturnType


作用：获取函数返回类型。


例如：


```typescript
function getPerson(name: string, value?: number): any {}

type F = ReturnType<typeof getPerson>; // F 类型是 any
```


## TypeScript 高级操作符


### const 断言


特点：

- 字面量（数组、接口）类型变为 readonly
- 字面量类型不能被扩展

举例：


不使用 const 断言：


```typescript
const CONFIG_KEYS = ["name", "school", "country"];

type KEY_TYPES = typeof CONFIG_KEYS; // KEY_TYPES 是 string[]

type KEY_TYPE = typeof CONFIG_KEYS[number]; // KEY_TYPE 是 string
```


使用 const 断言，ts 推断更精确：


```typescript
const CONST_CONFIG_KEYS = ["name", "school", "country"] as const;

type COPNST_KEY_TYPES = typeof CONST_CONFIG_KEYS; // COPNST_KEY_TYPES 是 readonly ["name", "school", "country"]

type CONST_KEY_TYPE = typeof CONST_CONFIG_KEYS[number]; // CONST_KEY_TYPE 是 "name" | "school" | "country"
```


### typoef 和 type


typeof 在 js 中，可以获得对象类型.


在 ts 中，能够配合 type 关键词，将对象类型保存在类型字面量中。


```typescript
const CONFIG_KEYS = ["name", "school", "country"];

console.log(typeof CONFIG_KEYS); // 输出；object
type KEY_TYPES = typeof CONFIG_KEYS; // KEY_TYPES 是 string[]
```


### keyof


keyof 用来获取**某种类型**的所有键，注意不是某个变量。


例子：假设声明了一个 json 对象，然后想获取它的所有键的类型应该怎么办？


```typescript
// 利用 const 断言，告诉解释器不能被扩展
const VARS = {
    NODE_ENV: "production",
    AUTHOR: "xintan",
} as const;
// 再获取它的类型
type VARS_TYPE = typeof VARS;
// 最后获取类型上的key的类型
type VARS_KEYS_TYPE = keyof VARS_TYPE; // "NODE_ENV" | "AUTHOR"
```


再来一个用例，比如断言`Reflect.ownKeys()`的返回结果的类型：


```typescript
const obj = {
    name: "12yuanxin3",
    age: 22,
} as const;

const objKeys = <(keyof typeof obj)[]>Reflect.ownKeys(obj); // objKeys 类型是： ("name" | "age")[]
```


### extends


基础的用法：可以用于 class 继承、interface 继承。除此之外，还用于“类型约束”。


### 用法 1: `T extends U ? X : Y`


用法：`T extends U ? X : Y`。这个比较难理解，而且很常见。其实就是：**如果 T 包含的类型是 U 包含的类型的 '子集'，那么取结果 X，否则取结果 Y**。


举例：


```typescript
type Diff<T, U> = T extends U ? never : T; // 找出T和U的差集
type Filter<T, U> = T extends U ? T : never; // 找出T和U的交集
```


### 用法 2: 范型约束


用法：`<T extends Lengthwise>`。


举例：


```typescript
interface Lengthwise {
    length: number;
}

function loggingIdentity<T extends Lengthwise>(arg: T): T {
    console.log(arg.length);
    return arg;
}
```


如果写成 `function loggingIdentity<T>(arg: T): T`，就会报错：T 上不存在 length 属性。正如所见，约束范型，传入的 arg 必须有 length 属性。


### infer


作用：针对 ts 推断的类型，声明一个字面量，在`extends`语句中使用。


举例：


```typescript
// 获得函数的返回类型
// 首先，范型约束了 T 是传入的函数的类型
// 然后，利用 extends 扩展表达式 + infer，将函数返回类型设置为R
// 最后，因为对T进行了约束，所以表达式一定是 true，也就是一定返回 R
// 我的理解：第2和3步骤就是为了拿到函数返回类型
type ReturnType<T extends (...args: any) => any> = T extends (
    ...args: any
) => infer R
    ? R
    : any;
```


例子中的`infer R`就是传入参数（某个函数）的返回类型，在这个表达式中，函数的返回类型被`R`变量代替。


:::warning infer 使用的位置
infer 声明的这个变量只能在 true 分支中使用
:::


infer的更多内容可以参考 [https://jkchao.github.io/typescript-book-chinese/tips/infer.html#一些用例](https://jkchao.github.io/typescript-book-chinese/tips/infer.html#%E4%B8%80%E4%BA%9B%E7%94%A8%E4%BE%8B) ：

- `ReturnType`：获取函数返回类型（内置）
- `ConstructorParameters`：获取构造函数类型（内置）
- `InstanceType` ：获取 class 类型（内置）
- `ElementOf` ：获取数组元素类型

### 反解 Promise


理解了 ReturnType，我写出了一个**反解 Promise**的 typescript 扩展类型：


```typescript
type UnPromiseType<T extends (...args: any) => any> = T extends (
    ...args: any
) => Promise<infer U>
    ? U
    : never;
```


使用效果：


```typescript
async function main(): Promise<string> {
    return "abc";
}

type returnTypeWithoutPromise = UnPromiseType<typeof main>; // returnTypeWithoutPromise 是 string
```


## 参考链接

- [Typescript docs Types](https://www.typescriptlang.org/docs/handbook/utility-types.html)
- [TypeScript 强大的类型别名](https://juejin.im/post/6844903753431138311)
- [TypeScript typeof 操作符](http://www.semlinker.com/ts-typeof/)
- [白话 typescript 中的【extends】和【infer】](https://juejin.im/post/6844904146877808653#heading-1)
- 《深入理解 TypeScript》

