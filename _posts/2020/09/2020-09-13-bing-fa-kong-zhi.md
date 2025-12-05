---
title: "Promise Pool并发控制"
url: "2020-09-13-bing-fa-kong-zhi"
date: 2020-09-13
---

## 场景


当异步处理过多的时候，比如`Promise.all()` 并发发起多个 Promise，假设 Promise 是 tcp 连接，并且数量达到几万个，会带来性能问题或触发系统限制。


## 解决思路


对 Promise 做并发限制。也就是准备一个 Pool（池），用来限制并发上限数。


例如 Pool 中上限是 4，而需要并发的 Promise 数量是 8。那么会先取前 4 个 Promise 执行，剩余的 Promise“排队”等候。


## 设计思路


简述一个简易实现思路：

1. 封装一个 ConcurrencyPromisePool 类
2. 方法有`all()`，和`Promise.prototype.all`类似。
3. 属性有 `limit`、`queue`。前者是并发上限，后者存放排队的 promise。

**注意**：第 2 点中，all 函数传入的是生成 Promise 的方法，而不是 Promise 实例。因为 Promise 一旦生成实例，会直接执行。所以要把这个执行交给 ConcurrencyPromisePool 来控制。


## 实现思路


实现思路简述：

- 将promise函数传入给`all`方法
- 依次执行promise函数

在执行promise函数时，需要考虑并发上限控制，处理思路如下：

- 检查当前在执行的promise是否达到上限
	- 达到：进入等待队列
	- 未达到：将运行次数+1，并且执行promise
- 当promise执行完，检查是否全部执行完
	- 全部执行完：则返回
	- 否则：取出等待队列中的promise，继续执行

代码实现：


```typescript
class ConcurrencyPromisePool {
    constructor(limit) {
        this.limit = limit;
        this.runningNum = 0;
        this.queue = [];
        this.results = [];
    }

    all(promises = []) {
        return new Promise((resolve, reject) => {
            for (const promise of promises) {
                this._run(promise, resolve, reject);
            }
        });
    }

    _run(promise, resolve, reject) {
        if (this.runningNum >= this.limit) {
            console.log(">>> 达到上限，入队：", promise);
            this.queue.push(promise);
            return;
        }

        ++this.runningNum;
        promise()
            .then((res) => {
                this.results.push(res);
                --this.runningNum;

                if (this.queue.length === 0 && this.runningNum === 0) {
                    return resolve(this.results);
                }
                if (this.queue.length) {
                    this._run(this.queue.shift(), resolve, reject);
                }
            })
            .catch(reject);
    }
}
```


代码使用：


```javascript
const promises = [];
for (let i = 0; i < 5; ++i) {
    promises.push(
        () =>
            new Promise((resolve) => {
                console.log(`${i} start`);
                setTimeout(() => {
                    console.log(`${i} end`);
                    resolve(i);
                }, 1000);
            })
    );
}

const pool = new ConcurrencyPromisePool(4);
pool.all(promises);
```


输出结果：


```shell
0 start
1 start
2 start
3 start
>>> 达到上限，入队： [Function]
0 end
4 start
1 end
2 end
3 end
4 end
```


## 社区方案


推荐[p-limit.js](https://www.npmjs.com/package/p-limit)


源码设计很有意思，不侵入 all 方法，改动成本小：


```typescript
const pLimit = require("p-limit");

const limit = pLimit(1);

const input = [
    limit(() => fetchSomething("foo")),
    limit(() => fetchSomething("bar")),
    limit(() => doSomething()),
];

(async () => {
    // Only one promise is run at once
    const result = await Promise.all(input);
    console.log(result);
})();
```


## 参考文章

- [What is the best way to limit concurrency when using ES6's Promise.all()?](https://stackoverflow.com/questions/40639432/what-is-the-best-way-to-limit-concurrency-when-using-es6s-promise-all)
- [Improve Your Node.js Performance With Promise Pools](https://medium.com/better-programming/improve-your-node-js-performance-with-promise-pools-65615bee2adb)
- [Promise.all 并发限制](https://segmentfault.com/a/1190000016389127)

