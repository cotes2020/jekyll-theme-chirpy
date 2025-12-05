---
title: "函数式编程中的Function Composition"
url: "2021-09-02-function-comp"
date: 2021-09-02
---

## Function Composition


函数组合是FP中的重要组成部分。**它是更强大的抽象化的工具，能够把命令式的****`how`****（指令式编程）抽象为可读性更好的声明式，让程式专注在****`what`****（描述式编程）。**


通常，input会经过很多fn，从而转换为程序所需要的数据。可以将「很多fn」作为一个新函数，供使用者调用，从而像搭建积木一样，组装起完整程序。


> 因为JS只能return单值的特性，所以将参数调整成 `arity = 1` 单参数函数会非常方便做组合。


### Compose


Compose流程如下：


```shell
output <-- fn1 <-- fn2 <-- ... <-- fnN <--input
```


**定义****`compose(...)`** **函数：**


```typescript
function compose(...fns) {
    return function composed(result) {
        // 拷贝fns
        var list = fns.slice()
        // 从右到左依次拿出函数执行，每个函数的参数是上个函数的返回结果
        while (list.length > 0) {
            result = list.pop()(result)
        }

        return result
    }
}

// reduce版的compose函数
function compose(...fns) {
    return function composed(result) {
        return fns
        .reverse()
        .reduce(function reducer( result, fn ) => {
            return fn(result)
        }, result )
    }
}
```


下面是对比使用`compose(...)`函数前后的写法：


```typescript
// 不使用compose
var result = skipShortWords( deDuplicate( splitString(str) ) )

// 使用compose
var longWords = compose(skipShortWords, deDuplicate, splitString) // step1: 先生成新函数
var result = longWords(str) // step2: 计算结果
```


### Pipe


在FP还有另一种合成的方式是从左到右处理，通常在FP函数库称作`pipe(..)`。


**定义**`pipe(...)`**函数：**


```typescript
function pipe(...fns) {
    return function piped(result) {
        var list = fns.slice()

        while (list.length > 0) {
            result = list.shift()(result)
        }

        return result
    }
}
```


使用`pipe(..)`最大的优点是在于以函数的执行顺序排列参数，参数顺序就是执行顺序。


## 参考文章


[bookmark](https://ithelp.ithome.com.tw/users/20075633/ironman/1375)


