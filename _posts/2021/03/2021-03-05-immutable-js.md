---
title: "ImmutableJS 实战"
date: 2021-03-05
permalink: /2021-03-05-immutable-js/
---
## 特性 1: 内置对象


ImmutableJs 提供了大量的内置对象，它们和 js 的原生对象不同，可以互相转换。


这意味着在使用 ImmutableJs 时，对原来的代码会有比较大的侵入型，需要改造。


```typescript
const { Map, List } = require("immutable");
const assert = require("assert");
// Immutable Map 对象
const map1 = Map({ a: 1, b: 2, c: 3 });
const map2 = map1.set("b", 3); // 更新属性，根据Immutable定义，返回一个崭新对象
console.log(map1.get("b"), map2.get("b"));

// Immutable List 对象
const list1 = List([1, 2]);
const list2 = list1.push(3, 4, 5);
const list3 = list2.unshift(0);
const list4 = list1.concat(list2, list3);
// 以下验证都能通过，长度不同说明list1 ～ list4，每个都是一个崭新对象
assert.equal(list1.size, 2);
assert.equal(list2.size, 5);
assert.equal(list3.size, 6);
assert.equal(list4.size, 13);
assert.equal(list4.get(0), 1);
```


除了常见的数据类型，immutable.js 额外提供了 SortedMap 、SortedSet、Stack、Recodr、Range 等对象。


## 特性 2: 基于 values 的比较


在 js 中，对于复杂对象，`===`运算符是基于指针进行比较的。


而在 immutable.js 中，由于每次操作都返回一个新对象，所以对象的指针肯定不同。对于 Immutable，比较本身也是基于对象的值（value）来进行的。


可以将 Immutable 的对象看作 collection，决定 collection 性质的是它的 values，而不是 references。


```typescript
const map3 = Map({ a: 1, b: 2, c: 3 });
const map4 = Map({ a: 1, b: 2, c: 3 });
console.log(map3.equals(map4)); // true
console.log(map3 === map4); // false
```


如果 values 不变，那么会避免创建新对象。所以利用`===`比较后，是相等的：


```typescript
const originalMap = Map({ a: 1, b: 2, c: 3 });
const updatedMap = originalMap.set("b", 2);
// output: true
console.log("originalMap === updatedMap: ", originalMap === updatedMap);
```


## 特性 3: 原生 js 类型转换


immutable 对象，可以和 js 的 Array、Object 进行转换。


转换 Map 和原生 JSON：


```typescript
const { Map, List, fromJS } = require("immutable");

const map1 = Map({ a: 1, b: 2, c: 3, d: 4 });
const map2 = Map({ c: 10, a: 20, t: 30 });
const obj = { d: 100, o: 200, g: 300 };
const map3 = map1.merge(map2, obj);
console.log(map3.toJSON()); // 转换为 json
// output: { a: 20, b: 2, c: 10, d: 100, t: 30, o: 200, g: 300 }
```


转换 List 和原生 List：


```typescript
const list1 = List([1, 2, 3]);
const list2 = List([4, 5, 6]);
const array = [7, 8, 9];
const list3 = list1.concat(list2, array);
console.log(list3.toArray()); // toArray、toObject都是将其转换为js原生对象（非递归）
```


`fromtJS()` 和 `toJS()` 操作嵌套对象：


```typescript
// 对于嵌套对象，它们提供mergeDeep, getIn, setIn, and updateIn，
// 来操作嵌套对象
const nested = fromJS({ a: { b: { c: [3, 4, 5] } } });
console.log(nested.getIn(["a", "b", "c"]).toArray()); // output：[ 3, 4, 5 ]

const nested2 = nested.mergeDeep({ a: { b: { d: 6 } } }); // merge 会覆盖合并，mergeDeep不会直接覆盖（推荐）
console.log(nested2.toJS());

const nested3 = nested.updateIn(["a", "b", "c"], (list) => list.push(4));
console.log(nested3.toJS()["a"]); // toJS将其嵌套对象递归转换
```


## 特性 4: 性能优化-支持可变副本


批量操作（Batching Mutations）中可以使用可变副本。它有什么好处呢？


在 immutable 默认行为中，每次值的改变都会返回新的不可变副本。虽然对深拷贝进行了优化，但是每次都返回不可变还是会有性能损耗。


此时，可以使用 withMutations 来进行批量操作，它的回调函数中的参数，是可变副本。基于可变副本操作，避免了每次计算产生新副本。


```typescript
const list1 = List([1, 2, 3]);
const list2 = list1.withMutations(function (list) {
    // list是可变副本
    list.push(4).push(5).push(6);
});
assert.equal(list1.size, 3);
assert.equal(list2.size, 6);
```


## 特性 5: 性能优化-支持惰性运算


支持“惰性运算”。~~类似函数柯里化~~，只有在使用值的时候，才进行运算。


Seq 提供了惰性操作运算，避免创建中间集合：


```typescript
const oddSquares = Seq([1, 2, 3, 4, 5, 6, 7, 8])
    .filter((x) => x % 2 !== 0)
    .map((x) => x * x);

// 耗时：7.477ms
console.time("a");
console.log(oddSquares.get(1)); // 这时才会进行计算，之前不会进行计算
console.timeEnd("a");
// 耗时：0.061ms
console.time("b");
console.log(oddSquares.get(2));
console.timeEnd("b");
```


## 实现惰性计算


惰性计算就是先保存计算过程，但不执行计算。


等到被访问到值的时候，再进行计算。


核心有2个：

- 保存函数过程
- 流式组装函数，FP中的pipe函数

实现如下：


```typescript
class LazyDemo {
    constructor(data) {
        this.data = data;
        this.fns = [];
        this.calced = false; // 是否计算过
    }

    filter(fn) {
        if (typeof fn !== 'function') {
            throw new Error('Filter')
        }

        this.fns.push({
            type: 'FILTER',
            fn
        })
        return this
    }

    map(fn) {
        if (typeof fn !== 'function') {
            throw new Error('Filter')
        }

        this.fns.push({
            type: 'MAP',
            fn
        })
        return this
    }

    getVal(index) {
        if (this.calced) {
            return this.data[index]
        }
        let tmpData = [...this.data]
        for (const item of this.fns) {
            const { fn, type } = item
            if (type === 'MAP') {
                tmpData = tmpData.map(fn)
            } else if (type === 'FILTER') {
                tmpData = tmpData.filter(fn)
            }
        }

        this.calced = true;
        this.data = tmpData;
        return this.data[index]

    }
}
```


使用效果：


```typescript
const lazy = (new LazyDemo([1, 2, 3, 4, 5, 6, 7, 8]))
    .filter((x) => x % 2 !== 0)
    .map((x) => x * x)

// 第一次
console.time("a");
console.log(lazy.getVal(1)); // 这时才会进行计算，之前不会进行计算
console.timeEnd("a");
// 第二次
console.time("b");
console.log(lazy.getVal(2));
console.timeEnd("b");
```


输出：和immutable完全一样


```shell
9
a: 6.375ms
25
b: 0.055ms
```


## 参考链接

- immutable.js 文档：[https://immutable-js.github.io/immutable-js/](https://immutable-js.github.io/immutable-js/)

