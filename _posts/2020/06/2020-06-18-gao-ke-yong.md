---
title: "数据存储高可用：数据冗余的多种做法"
date: 2020-06-18
permalink: /2020-06-18-gao-ke-yong/
---
存储高可用通过数据复制，来实现数据冗余，进而实现高可用。


难点在于明确节点的职责、数据复制策略、如何处理意外（复制延迟、中断）。


## 双机架构


### 主备复制


![007S8ZIlgy1gh6pb3yrr1j30v60m1ab7.jpg](https://raw.githubusercontent.com/dongyuanxin/static/main/blog/imgs/2020-06-18-gao-ke-yong/007S8ZIlgy1gh6pb3yrr1j30v60m1ab7.jpg)


备机不提供读写服务。当主机挂掉后，人工升级备机为主机。


**一般用于内部系统，例如学生管理、员工管理等，数据变更频率低，可以人工方法补全。**


### 主从复制


![007S8ZIlgy1gh6pbcgl4xj30v60m1gmo.jpg](https://raw.githubusercontent.com/dongyuanxin/static/main/blog/imgs/2020-06-18-gao-ke-yong/007S8ZIlgy1gh6pbcgl4xj30v60m1gmo.jpg)


从机对外提供读服务。客户端需要感知主从关系，一些业务数据需在代码中到从机上读取，以分散主机读写压力。


**一般用于读多写少的场景，例如 BBS、新网网站等。**


### 双机切换


主备和主从的共性问题：主机故障后，无法自动切换，无法保证读写业务正常进行。这时需要用到双机切换（以主备为例，主从相似）。


思考点是：切换时机、切换策略、自动程度。


**1、互连式**


![007S8ZIlgy1gh6pbk2mskj30bd0bv3z0.jpg](https://raw.githubusercontent.com/dongyuanxin/static/main/blog/imgs/2020-06-18-gao-ke-yong/007S8ZIlgy1gh6pbk2mskj30bd0bv3z0.jpg)


在主备的基础上，主和备之间多了一条“状态传递”通道。备机获取主机的状态，当主机有问题时，备机升级为主机。


**2、中介式**


![007S8ZIlgy1gh6pbq0q4lj30ch0cw0ta.jpg](https://raw.githubusercontent.com/dongyuanxin/static/main/blog/imgs/2020-06-18-gao-ke-yong/007S8ZIlgy1gh6pbq0q4lj30ch0cw0ta.jpg)


和“互连式”相比，主备不直接相连，通过中介来传递状态信息。


**3、模拟式**


![007S8ZIlgy1gh6pc45wesj30ee0ctgm9.jpg](https://raw.githubusercontent.com/dongyuanxin/static/main/blog/imgs/2020-06-18-gao-ke-yong/007S8ZIlgy1gh6pc45wesj30ee0ctgm9.jpg)


备机模拟客户端，像主机起读写请求，根据请求判断主机状态。


### 主主复制


![007S8ZIlgy1gh6pccef6rj309e0aa3yv.jpg](https://raw.githubusercontent.com/dongyuanxin/static/main/blog/imgs/2020-06-18-gao-ke-yong/007S8ZIlgy1gh6pccef6rj309e0aa3yv.jpg)


两台均为主机，都对外提供读写服务，互相复制数据给对方。


主主复制限制比较多，要求“数据可以双向复制”。以下场景不行：

- 自增的数据 ID。例如在 A 注册用户，ID 为 99；在 B 注册了用户，ID 也为 99。同步数据时会发生冲突。
- 库存。例如在 A 上减少到 99，在 B 上减少到 98。

主要用于：临时性、可丢失、可复制的数据场景。例如 session、用户行为上报、日志等。


## 数据集群


**双机架构和数据集群的关系**


双机架构只考虑特定数量的主机，假设前提是主机本身的能力够用。
但是单机本身是有瓶颈的。突破单机瓶颈，就要用到数据集群了。


### 数据集中集群


类似主备、主从，是 1 主多备，或者 1 主多从。


需要考虑的问题：

- 复制策略：多个备机，复制压力大；备机可能数据不相同，需要校验一致性和修正。
- 主机状态：多台备机对主机进行状态检测，需要考量结果不同时如何判断。
- 主机故障：如何选取新的主机

常见场景：数据量不大、数据机器不多。例如 `ZooKeeper` 集群（5 台左右）


### 数据分散集群


类似于主主，由多个服务器组成，每台都会存储部分数据。


需要考虑的问题：

- 数据分布均衡
- 服务容错：故障时，分配给故障服务器的数据要分配给其他服务器
- 可伸缩

常见场景：数据庞大、机器庞大。例如 `Hadoop` 集群、 `HBase` 集群（百台起步）。


**集中集群和分散集群中的“主机”**


集中集群：承担读写业务
分散集群：负责执行数据分配算法，类似于管理员。


## 数据分区


**数据分区的作用**


所站角度更高，主要应对大型灾难和事故。
一个系统有多个分区，每个分区下可以有自己的集群。


### 集中式


![007S8ZIlgy1gh6pct7uddj30aj065glw.jpg](https://raw.githubusercontent.com/dongyuanxin/static/main/blog/imgs/2020-06-18-gao-ke-yong/007S8ZIlgy1gh6pct7uddj30aj065glw.jpg)


集中式备份指存在一个总的备份中心，所有的分区都将数据备份到备份中心。


设计简单，扩展容易，但是独立备份成本性能要求高，成本高。


### 互备式


![007S8ZIlgy1gh6pd2de9lj30ak04o74h.jpg](https://raw.githubusercontent.com/dongyuanxin/static/main/blog/imgs/2020-06-18-gao-ke-yong/007S8ZIlgy1gh6pd2de9lj30ak04o74h.jpg)


互备式备份指每个分区备份另外一个分区的数据。


设计复杂，扩展麻烦（需要修改指向），但是成本低，利用已有设备。


### 独立式


![007S8ZIlgy1gh6pdbb9umj30c4052jrp.jpg](https://raw.githubusercontent.com/dongyuanxin/static/main/blog/imgs/2020-06-18-gao-ke-yong/007S8ZIlgy1gh6pdbb9umj30c4052jrp.jpg)


独立式备份指每个分区自己有独立的备份中心。


设计简单，扩展容易，比集中式成本更高。


## 参考链接

- [极客时间：从 0 学架构](https://time.geekbang.org/column/article/6354)

