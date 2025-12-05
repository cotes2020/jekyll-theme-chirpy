---
title: "限流算法-时间窗口算法"
date: 2020-10-10
permalink: /2020-10-10-time-window/
---
## 用途


假设QPS是2。那么时间窗口大小设置为1s，并且其中的大小设置为2。


从而保证 [0, 1], [1, 2], ... , [n - 1, n] 每个时间窗口内，请求量都是1。


## 实现

1. 限制时间窗口内的请求总数(size)
2. 时间窗口内的已有请求量(num)定时刷新

```typescript
/**
 * 1、限制时间窗口内的请求总数(size)
 * 2、时间窗口内的已有请求量(num)定时刷新
 */
let lastTime = Date.now(); // 时间窗口上次刷新的时间
let internal = 10000; // 时间窗口时间大小，单位ms
let size = 2; // 时间窗口内能通过的请求总数
let num = 0; // 时间窗口内目前已通过的请求数

function check() {
    let nowTime = Date.now();
    if (nowTime - lastTime > internal) {
        // 定时（internal）清空时间窗口
        num = 0;
        lastTime = nowTime;
    }

    if (num >= size) {
        return false;
    }
    ++num;
    return true;

```


测试代码效果：每10s，打印2遍，相当于放通2个请求


```typescript
// 测试代码
// c
while (1) {
    if (check()) {
        console.log(">>> 请求可以通过");
    }
}
```


## 缺陷


时间窗口不是平滑限流。


例如对于 [0, 1], [1, 2], QPS限制是2。假设在 0.8s 和 1.2s的时候各有2个请求打入，能通过`check()`函数。但是对于 [0.5, 1.5] 这段时间内，请求数是4，而没有限制成2。


平滑流量需要使用漏桶算法或者令牌桶算法。参考 [限流算法-漏桶和令牌桶](https://www.notion.so/a8a2ee5d8175448a8ac3f9b84a1e6430) 


