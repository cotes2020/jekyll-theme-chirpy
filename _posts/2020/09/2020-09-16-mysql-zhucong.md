---
title: "MySQL 主从一致性问题"
url: "2020-09-16-mysql-zhucong"
date: 2020-09-16
---

## 概述


**MySQL 的主从一致是通过 binlog 复制实现的。**


关于 binlog 复制可以从这三个方面来把握：

1. binlog 复制机制
2. binlog 复制延迟
3. 异步、半同步、MGR

## 什么是 binlog？


binlog 叫做归档日志，是 mysql 提供的，所有的存储引擎都可以使用这个日志，追加写，日志文件会不断增大，在数据备份时我们就会用到这个文件，binlog 只提供归档能力，binlog 日志包含了引起或可能引起数据库改变(如 delete 语句但没有匹配行)的事件信息，但绝不会包括 select 和 show 这样的查询语句。语句以"事件"的形式保存，所以包含了时间、事件开始和结束位置等信息。


## binlog 复制机制

- master节点处理请求时，写入 binlog。
- binlog 的内容发给 slave节点。
- slave节点按照内容写入 relay log。再从 relay log 读出，重放 SQL 语句。达成在 slave 上重做一遍 master 操作的效果。

## binlog 复制延迟


slave 是晚于 master 的。在 slave 上执行`show slave status`就能看到有多少延迟。


很多情况都会造成延迟，列举些常见的场景：

- slave 用的机器本身比 master 差
- 为了不影响线上业务，运营系统的统计类 SQL 会放在 slave 上，造成 slave 压力更大
- 大表 DDL、一次性大量 DELETE
- slave 的并行度低

延迟意味着 master 和 slave 上数据不一致。对读写分离和主从切换都会有影响。


## 异步复制、半同步复制、MGR


MySQL 5.5 之前是异步复制。master 将 binlog 中的事务异步地发给 slave，不会等待 slave 的应答。


5.5 引入了半同步复制。master 将事务发给 slave 后，slave 写 relay log，master 要等 slave 返回一个 ack 后，才能确认成功。由于 slave 上只是写到 relay log 就返回 ack 了，所以这个应答我们认为是很快的。当然，这里 master 要等待 slave 的 ack，如果 ack 迟迟没有，超过阈值后就会退回到异步复制。所以称作“**半**同步复制”。


MySQL 5.7 引入了 MGR，即组复制。多个节点共同组成一个复制组，读写事务要经过大多数节点一致后才会提交，而不是发起方说了算。


## 参考文章


[mysql binlog 复制](https://blog.csdn.net/arkblue/article/details/39484071)


