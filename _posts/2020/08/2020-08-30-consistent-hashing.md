---
title: "一致性Hash算法原理和应用"
date: 2020-08-30
permalink: /2020-08-30-consistent-hashing/
categories: ["C工作实践分享"]
tags: [一致性Hash, 分布式系统, 路由系统]
---

> 一致哈希是一种特殊的哈希算法。在使用一致哈希算法后，哈希表槽位数（大小）的改变平均只需要对 K/n 个关键字重新映射，其中 K 是关键字的数量，n 是槽位数量。然而在传统的哈希表中，添加或删除一个槽位的几乎需要对所有关键字进行重新映射。  
> — 维基百科


## 应用场景


多用于分布式 Web 服务中。每个服务代表哈希环的一个节点，当此节点发生改变时候，例如挂掉，只有部分缓存会重新计算分布，而不是全部失效。


特性如下：

- 冗余少
- 负载均衡
- 过渡平滑
- 存储均衡
- 关键词单调

## 算法过程


假设我们目前有一个缓存集群，图中 node 代表集群节点。


下图就是哈希环，区间范围是 0 ~ 2^32，所有的值都会被 hash 过来（比如采用 hashcode 算法）。


![e6c9d24egy1h57g555ixyj20hz0dd3z5.jpg](https://raw.githubusercontent.com/dongyuanxin/static/main/blog/imgs/2020-08-30-consistent-hashing/e6c9d24egy1h57g555ixyj20hz0dd3z5.jpg)


读取缓存的步骤：

- 计算 node 哈希值，将其映射到 0 ~ 2^32 范围内
- 读取缓存时，计算缓存键对应的哈希值，映射到同一个圆上
- 顺时针查找圆，遇到的第一个 node
- 其上读取缓存，如果没有，那么更新缓存

更新缓存步骤：类似「读取缓存」


![Untitled.png](https://raw.githubusercontent.com/dongyuanxin/static/main/blog/imgs/2020-08-30-consistent-hashing/615a80b5c0f6360560e261795533eb5c.png)


当 node 失效时，或者新加 node 时，只会有一部分缓存会变动，如上所示。


## 数据倾斜问题


如果服务节点太少，假设上图中，只有 node1 和 node3，那么大多数的缓存会落到 node1。


解决方法：每个 node，生成对应随机的 node 节点即可。


例如 node1 生成 node1#1、node1#2；node2 生成 node2#1、node2#2。


## 谷歌一致性哈希


jump consistent hash 是一种一致性哈希算法, 此算法零内存消耗，均匀分配，快速，并且只有 5 行代码。


```c++
int32_t JumpConsistentHash(uint64_t key, int32_t num_buckets) {
    int64_t b = -1, j = 0;
    while (j < num_buckets) {
        b = j;
        key = key * 2862933555777941757ULL + 1;
        j = (b + 1) * (double(1LL << 31) / double((key >> 33) + 1));
    }
    return b;
}
```


## 代码实现


在 nodejs 中，使用`hashring.js`。


```javascript
const HashRing = require('hashring');
// 传入节点权重，影响虚拟节点数量
const ring = new HashRing({
  '127.0.0.1:11211': 200,
  '127.0.0.2:11211': { weight: 200 }, // same as above
  '127.0.0.3:11211': 3200
});
console.log('>>> 命中节点', ring.get('your own ip'))
```


## 参考链接

- wiki: [https://zh.wikipedia.org/wiki/一致哈希](https://zh.wikipedia.org/wiki/%E4%B8%80%E8%87%B4%E5%93%88%E5%B8%8C)
- 五分钟看懂一致性哈希算法：[https://juejin.im/post/6844903598694858766](https://juejin.im/post/6844903598694858766)
- [https://zhuanlan.zhihu.com/p/34985026](https://zhuanlan.zhihu.com/p/34985026)
- 谷歌算法：[http://arxiv.org/ftp/arxiv/papers/1406/1406.2294.pdf](http://arxiv.org/ftp/arxiv/papers/1406/1406.2294.pdf)

