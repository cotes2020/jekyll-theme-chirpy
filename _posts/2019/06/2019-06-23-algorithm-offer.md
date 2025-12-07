---
title: "JavaScript版 · 剑指offer"
date: 2019-06-23
permalink: /2019-06-23-algorithm-offer/
categories: ["开源技术课程", "JavaScript版·剑指offer刷题笔记"]
---

## 介绍

这是笔者在上半年去阿里(蚂蚁)和腾讯面试时候，开始刷的一本书。**对于面试过程中的算法和数据结构帮助非常大，所以墙裂推荐**。

大概三月份，面试都通过之后，就开始断断续续的阅读、刷题。最近终于刷完了这本书，收货颇丰，把过程中每道题目的想法以及 JavaScript 的解题版本都记录和整理了下来。

**由于内容太多，所以划分成了 10 个专题，分别是：位运算、哈希表、堆、字符串、数组、查找、栈和队列、树、递归与循环、链表。**

由于篇幅过多，不再针对每篇在 Issue 上开设单独评论，可以在[此页面](https://xin-tan.com/passages/2019-06-23-algorithm-offer/)下讨论。如果发现问题，欢迎发起 PR 讨论。

网上已经有很多 js 版本，这版也没什么特别的地方，但是题目收录完整、专题划分明确、代码和文章风格统一，是花了蛮多心血和业余时间在里面，所以自荐一下。

希望这对于你之后的学习和面试稍有帮助，文章源码开放，食用请来[心谭博客](https://xin-tan.com/passages/2019-06-23-algorithm-offer/)。

**如果您觉得还有点意思，欢迎鼓励一个 Star**：[https://github.com/dongyuanxin/blog](https://github.com/dongyuanxin/blog)

## 特别鸣谢

[《剑指 Offer》](https://book.douban.com/subject/6966465/)，除了“手动实现 atoi”采用的是 Leetcode 版，其余题目均来自此书。

## 字符串

- [01-替换空格](https://xin-tan.com/passages/2019-06-23-str-replace-empty/)
- [02-字符串的全排列](https://xin-tan.com/passages/2019-06-23-str-perm/)
- [03-翻转单词顺序](https://xin-tan.com/passages/2019-06-23-str-reverse-sentence/)
- [04-实现 atoi](https://xin-tan.com/passages/2019-06-23-str-atoi/)

## 查找

- [01-旋转数组最小的数字](https://xin-tan.com/passages/2019-06-23-find-min-num/)
- [02-数字在排序数组中出现的次数](https://xin-tan.com/passages/2019-06-23-find-times-in-sorted/)

## 链表

- [01-从尾到头打印链表](https://xin-tan.com/passages/2019-06-23-list-print/)
- [02-快速删除链表节点](https://xin-tan.com/passages/2019-06-23-list-delete-node/)
- [03-链表倒数第 k 节点](https://xin-tan.com/passages/2019-06-23-list-last-kth-node/)
- [04-反转链表](https://xin-tan.com/passages/2019-06-23-list-reverse/)
- [05-合并两个有序链表](https://xin-tan.com/passages/2019-06-23-list-merge/)
- [06-复杂链表的复制](https://xin-tan.com/passages/2019-06-23-list-clone/)
- [07-两个链表中的第一个公共节点](https://xin-tan.com/passages/2019-06-23-list-first-same-node/)

## 数组

- [01-二维数组中的查找](https://xin-tan.com/passages/2019-06-23-array-find/)
- [02-数组顺序调整](https://xin-tan.com/passages/2019-06-23-array-change-location/)
- [03-把数组排成最小的数](https://xin-tan.com/passages/2019-06-23-array-min-numbers/)
- [04-数组中的逆序对](https://xin-tan.com/passages/2019-06-23-array-inverse-pair/)

## 栈和队列

- [01-用两个栈实现队列](https://xin-tan.com/passages/2019-06-23-stack-queue-exchange/)
- [02-包含 min 函数的栈](https://xin-tan.com/passages/2019-06-23-stack-queue-min-stack/)
- [03-栈的压入弹出序列](https://xin-tan.com/passages/2019-06-23-stack-queue-push-pop-order/)

## 递归和循环

- [01-青蛙跳台阶](https://xin-tan.com/passages/2019-06-23-recursive-loop-fibonacci/)
- [02-数值的整次方](https://xin-tan.com/passages/2019-06-23-recursive-loop-pow/)
- [03-打印从 1 到最大的 n 位数](https://xin-tan.com/passages/2019-06-23-recursive-loop-from-one-to-one/)
- [04-顺时针打印矩阵](https://xin-tan.com/passages/2019-06-23-recursive-loop-print-matrix/)
- [05-数组中出现次数超过一半的数字](https://xin-tan.com/passages/2019-06-23-recursive-loop-times-more-than-half/)
- [06-最小的 k 个数](https://xin-tan.com/passages/2019-06-23-recursive-loop-min-kth/)
- [07-和为 s 的两个数字](https://xin-tan.com/passages/2019-06-23-recursive-loop-and-number-is-s/)
- [08-和为 s 的连续正数序列](https://xin-tan.com/passages/2019-06-23-recursive-loop-s-sequence/)
- [09-n 个骰子的点数](https://xin-tan.com/passages/2019-06-23-recursive-loop-n-probability/)
- [10-扑克牌的顺子](https://xin-tan.com/passages/2019-06-23-recursive-loop-playing-cards/)
- [11-圆圈中最后剩下的数字](https://xin-tan.com/passages/2019-06-23-recursive-loop-joseph-ring/)

## 树

- [01-重建二叉树](https://xin-tan.com/passages/2019-06-23-tree-rebuild/)
- [02-判断是否子树](https://xin-tan.com/passages/2019-06-23-tree-subtree/)
- [03-二叉树的镜像](https://xin-tan.com/passages/2019-06-23-tree-mirror/)
- [04-二叉搜索树的后序遍历序列](https://xin-tan.com/passages/2019-06-23-tree-tail-order/)
- [05-二叉树中和为某一值的路径](https://xin-tan.com/passages/2019-06-23-tree-path-with-number/)
- [06-二叉树层序遍历](https://xin-tan.com/passages/2019-06-23-tree-level-travel/)
- [07-二叉树转双向链表](https://xin-tan.com/passages/2019-06-23-tree-convert-to-list/)
- [08-判断是否是平衡二叉树](https://xin-tan.com/passages/2019-06-23-tree-is-balance/)

## 位运算

- [01-二进制中 1 的个数](https://xin-tan.com/passages/2019-06-23-bit-number-of-one/)
- [02-二进制中 1 的个数进阶版](https://xin-tan.com/passages/2019-06-23-bit-number-of-one-more/)
- [03-数组中只出现一次的数字](https://xin-tan.com/passages/2019-06-23-bit-first-one/)

## 哈希表

- [01-丑数](https://xin-tan.com/passages/2019-06-23-hash-ugly/)
- [02-第一次只出现一次的字符](https://xin-tan.com/passages/2019-06-23-hash-first-no-repeat-char/)

## 堆

- [01-最小的 k 个数](https://xin-tan.com/passages/2019-06-23-heap-kth-numbers/)
