---
title: "计算高可用：分散计算压力"
date: 2020-06-20
permalink: /2020-06-20-gao-ke-yong/
---
## 概述


设计复杂度体现在「任务管理」方面。当任务在服务器 A 上执行失败，如何分配到新服务器执行。


## 主备


![007S8ZIlgy1gh6pdjp931j30au0a7mxh.jpg](https://raw.githubusercontent.com/dongyuanxin/static/main/blog/imgs/2020-06-20-gao-ke-yong/007S8ZIlgy1gh6pdjp931j30au0a7mxh.jpg)


计算高可用的主备架构，不需要数据复制。


对于不同的备机状态，可以细分成冷备和热备：

- 冷备：备机需要人工启动
- 热备（推荐）：备机已经启动，不对外提供服务

## 主从


![007S8ZIlgy1gh6pdsiv7lj309q0a4aad.jpg](https://raw.githubusercontent.com/dongyuanxin/static/main/blog/imgs/2020-06-20-gao-ke-yong/007S8ZIlgy1gh6pdsiv7lj309q0a4aad.jpg)


需要进行任务分配，部分给主机，部分给从机。


## 集群


分为 2 类：对称集群、非对称集群。


对称集群中，节点角色一样，执行相同任务。


非对称集群中，节点角色不相同，执行任务不同，例如 Master-Slave 架构。


## 云服务


在特定场景下，可以借助「云服务」来分散计算压力：

- 构建服务：CI服务
- 弹性计算：Faas云函数
- 异步任务：编排系统

## 计算高可用 vs 存储高可用


计算高可用中，节点故障后，任务发给非故障节点即可，不存在数据一致性问题。


相比较，存储高可用更加复杂。


