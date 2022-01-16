---
title: Labuladong
# author: Grace JyL
date: 2021-10-11 11:11:11 -0400
description:
excerpt_separator:
categories: [04CodeNote, DS]
tags:
math: true
# pin: true
toc: true
# image: /assets/img/sample/devices-mockup.png
---

- [Labuladong](#labuladong)
  - [re-check](#re-check)
  - [question to ask](#question-to-ask)
  - [basic](#basic)
  - [timeline](#timeline)
  - [学习算法和刷题的框架思维](#学习算法和刷题的框架思维)
    - [一、数据结构的存储方式](#一数据结构的存储方式)
    - [二、数据结构的基本操作](#二数据结构的基本操作)
      - [**数组遍历框架**，典型的`线性` `迭代`结构：](#数组遍历框架典型的线性-迭代结构)
      - [**链表遍历框架**，兼具`迭代`和`递归`结构：](#链表遍历框架兼具迭代和递归结构)
      - [**二叉树遍历框架**，典型的`非线性` `递归` `遍历` 结构：](#二叉树遍历框架典型的非线性-递归-遍历-结构)
      - [二叉树框架 扩展为 **N 叉树的遍历框架**](#二叉树框架-扩展为-n-叉树的遍历框架)
      - [**图的遍历**](#图的遍历)
    - [三、算法刷题指南](#三算法刷题指南)
    - [四、总结几句](#四总结几句)
- [two pointer](#two-pointer)
  - [two pointer - Array 数组](#two-pointer---array-数组)
    - [26. Remove Duplicates from Sorted Array 有序数组去重（简单）`快慢指针前后走`](#26-remove-duplicates-from-sorted-array-有序数组去重简单快慢指针前后走)
    - [80. Remove Duplicates from Sorted Array II `nums[i]!=nums[i-2]`](#80-remove-duplicates-from-sorted-array-ii-numsinumsi-2)
    - [FU. Each unique element should appear at most K times](#fu-each-unique-element-should-appear-at-most-k-times)
    - [27. Remove Element 移除元素 （简单）`快慢指针前后走`](#27-remove-element-移除元素-简单快慢指针前后走)
    - [83. Remove Duplicates from Sorted List 有序链表去重 `快慢指针前后走`](#83-remove-duplicates-from-sorted-list-有序链表去重-快慢指针前后走)
    - [283. Move Zeroes 移除0 `快慢指针前后走`](#283-move-zeroes-移除0-快慢指针前后走)
    - [349. Intersection of Two Arrays (Easy)](#349-intersection-of-two-arrays-easy)
      - [++++++++++ `Hash(num1 had), Hash.remove(num2 has)` BEST](#-hashnum1-had-hashremovenum2-has-best)
      - [`sorting, compare, get the same`](#sorting-compare-get-the-same)
    - [1385. Find the Distance Value Between Two Arrays (Easy)](#1385-find-the-distance-value-between-two-arrays-easy)
      - [brute force](#brute-force)
      - [Binary Search](#binary-search)
      - [???](#)
      - [`sort + sliding window` BEST](#sort--sliding-window-best)
    - [696. Count Binary Substrings (Easy)](#696-count-binary-substrings-easy)
      - [Brute Force](#brute-force-1)
  - [two pointer - 链表](#two-pointer---链表)
    - [203. Remove Linked List Elements (Easy)](#203-remove-linked-list-elements-easy)
      - [++++++++++ recursive solution](#-recursive-solution)
    - [237. Delete Node in a Linked List (Easy)](#237-delete-node-in-a-linked-list-easy)
    - [876. Middle of the Linked List 寻找单链表的中点](#876-middle-of-the-linked-list-寻找单链表的中点)
    - [2095. Delete the Middle Node of a Linked List (Medium)](#2095-delete-the-middle-node-of-a-linked-list-medium)
    - [寻找单链表的倒数n节点](#寻找单链表的倒数n节点)
    - [19. Remove Nth Node From End of List remove倒数n节点 `删除倒数n,找倒数n+1`](#19-remove-nth-node-from-end-of-list-remove倒数n节点-删除倒数n找倒数n1)
    - [Delete N Nodes After M Nodes of a Linked List ??????????](#delete-n-nodes-after-m-nodes-of-a-linked-list-)
    - [160. 判断两个单链表是否相交并找出交点](#160-判断两个单链表是否相交并找出交点)
  - [two pointer - palindrome 回文](#two-pointer---palindrome-回文)
    - [2108. Find First Palindromic String in the Array (Easy)](#2108-find-first-palindromic-string-in-the-array-easy)
      - [++++++++++ 2 pointer Check each word](#-2-pointer-check-each-word)
      - [++++++++++ StringBuilder.reverse.equals](#-stringbuilderreverseequals)
    - [832. Flipping an Image (Easy) `only same values flip both.`](#832-flipping-an-image-easy-only-same-values-flip-both)
    - [1332. Remove Palindromic Subsequences (Easy)](#1332-remove-palindromic-subsequences-easy)
      - [++++++++++ `只有0，1，2 三种答案，aaabbb最多两下消完` Best](#-只有012-三种答案aaabbb最多两下消完-best)
      - [reverse logic also](#reverse-logic-also)
  - [two pointer - String](#two-pointer---string)
    - [2000. Reverse Prefix of Word (Easy)](#2000-reverse-prefix-of-word-easy)
      - [++++++++++ `char[]`](#-char)
      - [++++++++++ `StringBuilder`](#-stringbuilder)
    - [557. Reverse Words in a String III (Easy)](#557-reverse-words-in-a-string-iii-easy)
    - [541. Reverse String II (Easy) `2134 6578`](#541-reverse-string-ii-easy-2134-6578)
    - [942. DI String Match (Easy) `Increase l++; Decrease r--`](#942-di-string-match-easy-increase-l-decrease-r--)
    - [905. Sort Array By Parity (Easy)](#905-sort-array-by-parity-easy)
      - [++++++++++ `new int[i] = nums[l/r]`](#-new-inti--numslr)
      - [++++++++++ In Place Solution Best](#-in-place-solution-best)
    - [1768. Merge Strings Alternately (Easy)](#1768-merge-strings-alternately-easy)
      - [++++++++++ `for (int i=0; i<Math.max(s1,s2); i++); `](#-for-int-i0-imathmaxs1s2-i-)
      - [++++++++++ substring](#-substring)
    - [977. Squares of a Sorted Array (Easy)](#977-squares-of-a-sorted-array-easy)
      - [++++++++++ Brute Force Approach](#-brute-force-approach)
      - [++++++++++ `Math.abs(nums[l]) > Math.abs(nums[r])` Best](#-mathabsnumsl--mathabsnumsr-best)
    - [821. Shortest Distance to a Character (Easy)](#821-shortest-distance-to-a-character-easy)
      - [++++++++++ ``Math.min(fromLeft, fromRight)`](#-mathminfromleft-fromright)
      - [++++++++++ `when s.char==c, j=i-1; j=i+1`](#-when-scharc-ji-1-ji1)
      - [++++++++++ `combine 2` BEST](#-combine-2-best)
    - [922. Sort Array By Parity II (Easy)](#922-sort-array-by-parity-ii-easy)
      - [++++++++++ `new res, nums[i]%2==0?; res[oddindex] oddindex++, res[evenindex] evenindex++`](#-new-res-numsi20-resoddindex-oddindex-resevenindex-evenindex)
      - [++++++++++ `for(int i=0;i<n; i+=2) should be even, if (odd), check prev num[odd]` BEST](#-forint-i0in-i2-should-be-even-if-odd-check-prev-numodd-best)
- [数组](#数组)
  - [TWOSUM问题](#twosum问题)
    - [1. Two Sum](#1-two-sum)
    - [167. Two Sum II - Input Array Is Sorted](#167-two-sum-ii---input-array-is-sorted)
  - [前缀和技巧](#前缀和技巧)
    - [303. Range Sum Query - Immutable 计算索引区间/list中指定位置的和 `preSum[i] = preSum[i - 1] + nums[i - 1];`](#303-range-sum-query---immutable-计算索引区间list中指定位置的和-presumi--presumi---1--numsi---1)
    - [560. Subarray Sum Equals K 和为k的子数组 `if (preSum[j] == preSum[i] - k) res++;`](#560-subarray-sum-equals-k-和为k的子数组-if-presumj--presumi---k-res)
    - [304. Range Sum Query 2D - Immutable 二维区域和检索 `图像块之间相互减`](#304-range-sum-query-2d---immutable-二维区域和检索-图像块之间相互减)
  - [差分](#差分)
    - [差分数组 `increment(i,j,val)->{diff[i]+=val; diff[j+1]-=val;`](#差分数组-incrementijval-diffival-diffj1-val)
    - [370. 区间加法（中等）`Difference df = new Difference(nums); df.increment(i, j, val);`](#370-区间加法中等difference-df--new-differencenums-dfincrementi-j-val)
    - [1109. Corporate Flight Bookings 航班预订统计](#1109-corporate-flight-bookings-航班预订统计)
    - [1094 题「拼车」](#1094-题拼车)
- [LinkedList](#linkedlist)
  - [单链表的六大解题套路](#单链表的六大解题套路)
    - [合并两个有序链表 Merge 2 Sorted Lists](#合并两个有序链表-merge-2-sorted-lists)
    - [23. Merge k Sorted Lists 合并 k 个有序链表 Merge k Sorted Lists](#23-merge-k-sorted-lists-合并-k-个有序链表-merge-k-sorted-lists)
  - [递归反转链表](#递归反转链表)
    - [206. Reverse Linked List 递归反转整个链表 `递归+pointer`](#206-reverse-linked-list-递归反转整个链表-递归pointer)
      - [++++++++++ 递归](#-递归)
      - [++++++++++ 2 pointer](#-2-pointer)
    - [反转链表前 N 个节点](#反转链表前-n-个节点)
    - [92. Reverse Linked List II 反转链表的一部分](#92-reverse-linked-list-ii-反转链表的一部分)
      - [++++++++++ iterative](#-iterative)
      - [++++++++++ recursive](#-recursive)
    - [25. Reverse Nodes in k-Group K个一组反转链表](#25-reverse-nodes-in-k-group-k个一组反转链表)
      - [++++++++++ `a,b reverse(), a.next=reverseK(b,k)`](#-ab-reverse-anextreversekbk)
    - [143. Reorder List (Medium)](#143-reorder-list-medium)
      - [++++++++++ `Two pointer, find middle, reverse(), combine(n1,n2)`](#-two-pointer-find-middle-reverse-combinen1n2)
      - [++++++++++ `2 pointer. list.add(ListNode), reorder list`](#-2-pointer-listaddlistnode-reorder-list)
    - [1721. Swapping Nodes in a Linked List (Medium)](#1721-swapping-nodes-in-a-linked-list-medium)
    - [24. Swap Nodes in Pairs (Medium)](#24-swap-nodes-in-pairs-medium)
      - [++++++++++ `2 pointer and swap`](#-2-pointer-and-swap)
      - [++++++++++ `recursive`](#-recursive-1)
    - [example](#example)
      - [870 题「优势洗牌」](#870-题优势洗牌)
  - [左右指针](#左右指针)
    - [二分查找](#二分查找)
    - [在有序数组中搜索指定元素](#在有序数组中搜索指定元素)
      - [704. Binary Search 寻找一个数（基本的二分搜索）](#704-binary-search-寻找一个数基本的二分搜索)
      - [寻找左侧边界的二分搜索](#寻找左侧边界的二分搜索)
        - [278. First Bad Version](#278-first-bad-version)
      - [寻找右侧边界的二分查找](#寻找右侧边界的二分查找)
      - [34. Find First and Last Position of Element in Sorted Array 寻找左右边界的二分搜索](#34-find-first-and-last-position-of-element-in-sorted-array-寻找左右边界的二分搜索)
      - [二分搜索算法运用](#二分搜索算法运用)
      - [example](#example-1)
        - [875. Koko Eating Bananas](#875-koko-eating-bananas)
        - [运送货物？？？？？？？？？？？？？？](#运送货物)
        - [https://labuladong.github.io/algo/2/21/59/ ？？？？](#httpslabuladonggithubioalgo22159-)
    - [两数之和](#两数之和)
    - [344. Reverse String 反转数组](#344-reverse-string-反转数组)
    - [滑动窗口技巧 `right++, missing==0, left++`](#滑动窗口技巧-right-missing0-left)
      - [最小覆盖子串](#最小覆盖子串)
      - [567. Permutation in String 字符串排列](#567-permutation-in-string-字符串排列)
      - [438. Find All Anagrams in a String 找所有字母异位词](#438-find-all-anagrams-in-a-string-找所有字母异位词)
      - [3. Longest Substring Without Repeating Characters 最长无重复子串](#3-longest-substring-without-repeating-characters-最长无重复子串)
  - [链表的环](#链表的环)
    - [判断单链表是否包含环](#判断单链表是否包含环)
    - [142. Linked List Cycle II 计算链表中环起点](#142-linked-list-cycle-ii-计算链表中环起点)
- [回文链表 Palindromic](#回文链表-palindromic)
  - [other](#other)
    - [9. Palindrome Number 判断回文Number](#9-palindrome-number-判断回文number)
      - [reverse half of it **Best**](#reverse-half-of-it-best)
    - [Elimination Game !!! Perform String Shifts !!! Subtree Removal Game with Fibonacci Tree](#elimination-game--perform-string-shifts--subtree-removal-game-with-fibonacci-tree)
    - [125. Valid Palindrome 判断回文链表String](#125-valid-palindrome-判断回文链表string)
      - [判断回文单链表 - 把原始链表反转存入一条新的链表，然后比较](#判断回文单链表---把原始链表反转存入一条新的链表然后比较)
      - [判断回文单链表 - 二叉树后序遍历](#判断回文单链表---二叉树后序遍历)
      - [判断回文单链表 - 用栈结构倒序处理单链表](#判断回文单链表---用栈结构倒序处理单链表)
      - [判断回文单链表 - 不完全反转链表，仅仅反转部分链表，空间复杂度O(1)。](#判断回文单链表---不完全反转链表仅仅反转部分链表空间复杂度o1)
  - [排序](#排序)
    - [快速排序](#快速排序)
    - [归并排序](#归并排序)
- [stack](#stack)
  - [队列 栈](#队列-栈)
    - [用栈实现队列](#用栈实现队列)
    - [用队列实现栈](#用队列实现栈)
  - [单调栈](#单调栈)
    - [返回等长数组for更大的元素](#返回等长数组for更大的元素)
    - [返回等长数组for更大的元素的index](#返回等长数组for更大的元素的index)
    - [环形数组](#环形数组)
  - [单调队列结构](#单调队列结构)
    - [滑动窗口问题](#滑动窗口问题)
- [Tree](#tree)
  - [二叉树](#二叉树)
    - [计算一棵二叉树共有几个节点](#计算一棵二叉树共有几个节点)
    - [翻转二叉树](#翻转二叉树)
    - [填充二叉树节点的右侧指针](#填充二叉树节点的右侧指针)
    - [将二叉树展开为链表](#将二叉树展开为链表)
    - [构造最大二叉树](#构造最大二叉树)
    - [通过前序和中序/后序和中序遍历结果构造二叉树(kong)](#通过前序和中序后序和中序遍历结果构造二叉树kong)
    - [寻找重复子树(kong)](#寻找重复子树kong)
  - [层序遍历框架](#层序遍历框架)
    - [二叉树max层级遍历 用Queue和q.size去遍历左右](#二叉树max层级遍历-用queue和qsize去遍历左右)
    - [多叉树的层序遍历框架  用Queue和q.size去遍历child](#多叉树的层序遍历框架--用queue和qsize去遍历child)
  - [BFS（广度优先搜索）用Queue和q.size去遍历child + not visited](#bfs广度优先搜索用queue和qsize去遍历child--not-visited)
    - [111. Minimum Depth of Binary Tree 二叉树min层级遍历 `用Queue和q.size去遍历左右`](#111-minimum-depth-of-binary-tree-二叉树min层级遍历-用queue和qsize去遍历左右)
    - [穷举所有可能的密码组合 用Queue和q.size去遍历all](#穷举所有可能的密码组合-用queue和qsize去遍历all)
  - [二叉搜索树](#二叉搜索树)
    - [判断 BST 的合法性](#判断-bst-的合法性)
    - [在 BST 中搜索元素](#在-bst-中搜索元素)
    - [在 BST 中插入一个数](#在-bst-中插入一个数)
    - [在 BST 中删除一个数](#在-bst-中删除一个数)
    - [不同的二叉搜索树 - 穷举问题](#不同的二叉搜索树---穷举问题)
    - [不同的二叉搜索树II](#不同的二叉搜索树ii)
    - [二叉树后序遍历](#二叉树后序遍历)
    - [二叉树的序列化与反序列化](#二叉树的序列化与反序列化)
    - [二叉树打平到一个字符串](#二叉树打平到一个字符串)
- [Binary Heap 二叉堆](#binary-heap-二叉堆)
  - [最大堆和最小堆](#最大堆和最小堆)
- [Graphy](#graphy)
  - [图的遍历](#图的遍历-1)
    - [转换成图](#转换成图)
    - [所有可能路径](#所有可能路径)
    - [判断有向图是否存在环](#判断有向图是否存在环)
    - [拓扑排序](#拓扑排序)
  - [搜索名人](#搜索名人)
    - [暴力解法](#暴力解法)
    - [优化解法](#优化解法)
    - [最终解法](#最终解法)
  - [UNION-FIND 并查集算法 计算 连通分量](#union-find-并查集算法-计算-连通分量)
    - [UNION-FIND算法](#union-find算法)
      - [基本思路](#基本思路)
      - [平衡性优化](#平衡性优化)
      - [路径压缩](#路径压缩)
  - [UNION-FIND算法应用](#union-find算法应用)
    - [DFS 的替代方案](#dfs-的替代方案)
    - [判定合法等式](#判定合法等式)
  - [DIJKSTRA 算法](#dijkstra-算法)
  - [DIJKSTRA 算法 起点 start 到某一个终点 end 的最短路径](#dijkstra-算法-起点-start-到某一个终点-end-的最短路径)
    - [网络延迟时间](#网络延迟时间)
    - [路径经过的权重最大值](#路径经过的权重最大值)
    - [概率最大的路径](#概率最大的路径)
- [设计数据结构](#设计数据结构)
  - [缓存淘汰](#缓存淘汰)
    - [LRU 缓存淘汰算法 Least Recently Used](#lru-缓存淘汰算法-least-recently-used)
      - [造轮子 LRU 算法](#造轮子-lru-算法)
      - [使用 Java 内置的 LinkedHashMap 来实现一遍。](#使用-java-内置的-linkedhashmap-来实现一遍)
    - [LFU 淘汰算法 Least Frequently Used](#lfu-淘汰算法-least-frequently-used)
  - [最大栈 Maximum Frequency Stack](#最大栈-maximum-frequency-stack)
- [数据流](#数据流)
  - [Reservoir Sampling 随机 水塘抽样算法](#reservoir-sampling-随机-水塘抽样算法)
    - [382. Linked List Random Node 无限序列随机抽取1元素](#382-linked-list-random-node-无限序列随机抽取1元素)
      - [be list, size, random n](#be-list-size-random-n)
      - [Reservoir Sampling](#reservoir-sampling)
    - [无限序列随机抽取 k 个数](#无限序列随机抽取-k-个数)
    - [398. Random Pick Index (Medium)](#398-random-pick-index-medium)
      - [Reservoir Sampling](#reservoir-sampling-1)
      - [HashMap](#hashmap)
    - [380. Insert Delete GetRandom O(1) 实现随机集合](#380-insert-delete-getrandom-o1-实现随机集合)
    - [710. Random Pick with Blacklist 避开黑名单的随机数 `blacklist index to good index`](#710-random-pick-with-blacklist-避开黑名单的随机数-blacklist-index-to-good-index)
    - [528. Random Pick with Weight (Medium)](#528-random-pick-with-weight-medium)
      - [`2 for: [1,2,3] -> [1,2,2,3,3,3]`](#2-for-123---122333)
      - [Reservoir Sampling](#reservoir-sampling-2)
      - [reservoir sampling **BEST**](#reservoir-sampling-best)
  - [other](#other-1)
    - [295. Find Median from Data Stream 中位数](#295-find-median-from-data-stream-中位数)
- [DFS and BFS](#dfs-and-bfs)
  - [BFS](#bfs)
    - [752. Open the Lock 解开密码锁最少次数 `用Queue和q.size去遍历all + visited + deads`](#752-open-the-lock-解开密码锁最少次数-用queue和qsize去遍历all--visited--deads)
      - [BFS](#bfs-1)
      - [双向 BFS 优化 `用Queue和q.size去遍历 q1=q2;q2=temp`](#双向-bfs-优化-用queue和qsize去遍历-q1q2q2temp)
  - [DFS backtrack 回溯算法](#dfs-backtrack-回溯算法)
    - [46. Permutations 全排列问题 ??????????/](#46-permutations-全排列问题-)
    - [51. N-Queens N 皇后问题 ??????????](#51-n-queens-n-皇后问题-)
    - [78. Subsets 子集（中等）](#78-subsets-子集中等)
    - [90. Subsets II](#90-subsets-ii)
    - [77. Combinations](#77-combinations)
- [功能](#功能)
  - [设计朋友圈时间线](#设计朋友圈时间线)
- [动态规划](#动态规划)
  - [斐波那契数列](#斐波那契数列)
  - [动态规划解法](#动态规划解法)
    - [322. Coin Change 凑零钱 ` for i, for coin, dp[i] = Math.min(dp[i], dp[i-coin]+1);`](#322-coin-change-凑零钱--for-i-for-coin-dpi--mathmindpi-dpi-coin1)
      - [暴力解法](#暴力解法-1)
      - [best 带备忘录的递归](#best-带备忘录的递归)
      - [dp 数组的迭代解法](#dp-数组的迭代解法)
      - [983. Minimum Cost For Tickets (Medium)](#983-minimum-cost-for-tickets-medium)
      - [bottom-up dp](#bottom-up-dp)
      - [Memoization](#memoization)
    - [64. Minimum Path Sum 最小路径和（中等）](#64-minimum-path-sum-最小路径和中等)
    - [931. Minimum Falling Path Sum 下降路径最小和](#931-minimum-falling-path-sum-下降路径最小和)
    - [174. Dungeon Game 地下城游戏 ????????????](#174-dungeon-game-地下城游戏-)
    - [514. Freedom Trail 自由之路（困难）??????](#514-freedom-trail-自由之路困难)
  - [加权有向图 最短路径](#加权有向图-最短路径)
    - [787. K 站中转内最便宜的航班（中等）](#787-k-站中转内最便宜的航班中等)
  - [子序列](#子序列)
    - [300. Longest Increasing Subsequence 最长递增子序列](#300-longest-increasing-subsequence-最长递增子序列)
    - [1143. Longest Common Subsequence 最长公共子序列](#1143-longest-common-subsequence-最长公共子序列)
    - [583. Delete Operation for Two Strings 两个字符串的删除操作](#583-delete-operation-for-two-strings-两个字符串的删除操作)
    - [712. Minimum ASCII Delete Sum for Two Strings 最小 ASCII 删除和](#712-minimum-ascii-delete-sum-for-two-strings-最小-ascii-删除和)
    - [5. Longest Palindromic Substring 最长回文子序列](#5-longest-palindromic-substring-最长回文子序列)
    - [516. Longest Palindromic Subsequence 最长回文子序列长度](#516-longest-palindromic-subsequence-最长回文子序列长度)
    - [494. Target Sum 目标和](#494-target-sum-目标和)
      - [回溯思路](#回溯思路)
      - [消除重叠子问题](#消除重叠子问题)
    - [72. Edit Distance 编辑距离（困难）](#72-edit-distance-编辑距离困难)
    - [354. Russian Doll Envelopes 俄罗斯套娃信封问题（困难）](#354-russian-doll-envelopes-俄罗斯套娃信封问题困难)
    - [53 最大子序和（简单)](#53-最大子序和简单)
  - [背包类型问题](#背包类型问题)
    - [子集背包问题](#子集背包问题)
      - [416. Partition Equal Subset Sum 分割等和子集（中等）](#416-partition-equal-subset-sum-分割等和子集中等)
      - [698. Partition to K Equal Sum Subsets](#698-partition-to-k-equal-sum-subsets)
      - [215. Kth Largest Element in an Array](#215-kth-largest-element-in-an-array)
- [system design](#system-design)


---

# Labuladong

- https://github.com/labuladong/fucking-algorithm
- https://labuladong.github.io

---


## re-check

1. Palindrome



```java
// fast be the last one, slow in the middle.
while(fast.next!=null) {
    slow=slow.next;
    fast=fast.next;
}

```


## question to ask


1. can the values in the array be negative.
2. can square of values can exceed Integer.MAX_VALUE.
3. values are in long or Integer.
4. is given array sorted.(even if the example are sorted) this helped me in google interview interviewer told me that this is nice question. (I was not asked this question but a question where sample cases where sorted )

---


## basic


1. Two pointers
2. HashMap

```java

Math.abs(a-b);
Math.min(a,b);
Math.max(a,b);

StringBuilder sb = new StringBuilder("");
StringBuffer sb = new StringBuffer(s);
sb.setCharAt(i, Char);
sb.append('.');
sb.insert(pos[i],'Q');
sb.toString()
sb.reverse();



String Str1 = new String("Welcome to Tutorialspoint.com");
String Str1 = new String(char[] chars);
Str1.length()
Str1.toCharArray()
Str1.charAt()
Str1.substring(lo, lo+maxLen)
Str1.indexOf(ch);
str1.contains("h")

String CipherText=""
CipherText += (char)(cipherMatrix[i] + 65);

String[] words = Str1.split(" ");

char ch = (char)(i + 97);
Character.getNumericValue(c);  
Character.isLowerCase(s.charAt(i));
Character.toUpperCase(s.charAt(i));
Character.isWhitespace()

String.valueOf(char[]);
String.join(" ", array);


int[] distTo = new int[V];
Arrays.fill(distTo, Integer.MAX_VALUE);
int[].length;

Arrays.asList(int k);
Arrays.toString(subCoin)
Arrays.sort(nums1);
Arrays.sort(
    envelopes,
    new Comparator<int[]>() {
        public int compare(int[] a, int[] b) {
            return a[0] == b[0] ? b[1] - a[1] : a[0] - b[0];
        }
    }
);

Stack<String> Stack= new Stack<>();
Stack.push();
Stack.pop();
// Access element from top of Stack
Stack.peek();
Stack.empty();

ArrayList ans = new ArrayList();
ArrayList<Integer> ans = new ArrayList<>();
ans.add(num);
ans.size()
ans.get(i);


Vector myVect = new Vector();
myVect.add('one');
myVect.get(i);


ListNode<Integer> head = new ListNode<>();
ListNode.

Queue q = new LinkedList<>();
q.push();
q.poll();
q.peek();
q.isEmpty();


List<int[]>[] graph = new LinkedList[n+1];
List<List<String>> res = new ArrayList<List<String>>();
List<int[]> res = new ArrayList<>();
List<Integer> res = new ArrayList<>();
List.length;
List.add(a);
List.remove(i);

LinkedList<Integer> linkedlist = new LinkedList<>();
linkedlist.addFirst('k');
linkedlist.addLast('k');
linkedlist.getFirst();
linkedlist.getLast();
linkedlist.pollLast();
linkedlist.isEmpty();
linkedlist.removeLast();
Iterator iter = linkedlist.iterator();
iter.hashNext();
iter.next();


LinkedHashSet<Character> set = new LinkedHashSet<Character>();
set.add("kkk");
Iterator<Character> iter = set.iterator();
iter.hasNext();
iter.next();


Set<Character> set = new HashSet<Character>();
set.add("kkk");

Set<String> deads = new HashSet<>();


TreeSet<Integer> tree = new TreeSet<>();
tree.add(number);
Set<Integer> set = tree.subSet(leftValue, rightValue+1);


HashMap<Integer, Integer> hm = new HashMap<>();
hm.put(key, val);
hm.get(key);
hm.containsKey(key);
hm.size();
hm.putIfAbsent(1, new LinkedHashSet<>());
hm.getOrDefault(val, 0);


HashSet<Integer> set = new HashSet<Integer>();
set.contains(num);
set.add(num);
set.remove(num);
set.size();


LinkedHashSet<Integer> keyList = ;
LinkedHashSet.iterator().next();
LinkedHashSet.remove(Key);


BinaryHeap bh = new BinaryHeap();
bh.insert(k);
bh.findMin();
bh.delMin();
bh.isEmpty();
bh.size();
bh.buildHeap(list);

```

## timeline

10/24:
11/8:61
11/9:63
11/10:64
11/11:65
11/12:66
11/14:67
11/15:70
11/16:?
11/17:79
11/18:
11/19:
11/20:
11/21:87
11/22:
11/23:
11/24:94
11/25:







---

## 学习算法和刷题的框架思维




---

### 一、数据结构的存储方式

数据结构的存储方式只有两种：`数组`（顺序存储）和`链表`（链式存储）。
- 散列表、栈、队列、堆、树、图等等各种数据结构都属于「上层建筑」，而数组和链表才是「结构基础」。
- 因为那些多样化的数据结构，究其源头，都是在链表或者数组上的特殊操作，API 不同而已。

「队列」、「栈」这两种数据结构既可以使用链表也可以使用数组实现。
- 用数组实现，就要处理扩容缩容的问题；
- 用链表实现，没有这个问题，但需要更多的内存空间存储节点指针。

「图」的两种表示方法，
- 邻接表就是链表，邻接矩阵就是二维数组。
- 邻接矩阵判断连通性迅速，并可以进行矩阵运算解决一些问题，但是如果图比较稀疏的话很耗费空间。
- 邻接表比较节省空间，但是很多操作的效率上肯定比不过邻接矩阵。

「散列表」就是通过`散列函数`把`键`映射到一个大`数组`里。
- 而且对于解决`散列冲突`的方法，
- `拉链法`需要链表特性，操作简单，但需要额外的空间存储指针；
- `线性探查法`就需要数组特性，以便连续寻址，不需要指针的存储空间，但操作稍微复杂些。

「树」
- 用数组实现就是「堆」，因为「堆」是一个完全二叉树，用数组存储不需要节点指针，操作也比较简单；
- 用链表实现就是很常见的那种「树」，因为不一定是完全二叉树，所以不适合用数组存储。
  - 为此，在这种链表「树」结构之上，又衍生出各种巧妙的设计，
  - 比如二叉搜索树、AVL 树、红黑树、区间树、B 树等等，以应对不同的问题。

> example:
> Redis 数据库
> Redis 提供列表、字符串、集合等等几种常用数据结构，
> 但是对于每种数据结构，底层的存储方式都至少有两种，以便于根据存储数据的实际情况使用合适的存储方式。

综上，**数据结构**种类很多，但是底层存储无非`数组`或者`链表`，二者的优缺点如下：

**数组**
- 由于是`紧凑连续存储`,可以随机访问，通过`索引`快速找到对应元素，而且相对节约存储空间。
- 但正因为连续存储，内存空间必须一次性分配够，
- 数组如果要扩容，需要重新分配一块更大的空间，再把数据全部复制过去，时间复杂度 O(N)；
- 数组如果想在中间进行插入和删除，每次必须搬移后面的所有数据以保持连续，时间复杂度 O(N)。

**链表**
- 因为`元素不连续`，而是靠`指针`指向下一个元素的位置，所以不存在数组的扩容问题；
- 如果知道某一元素的`前驱`和`后驱`，`操作指针`即可删除该元素或者插入新元素，时间复杂度 O(1)。
- 但是正因为存储空间不连续，无法根据一个`索引`算出对应元素的地址，所以`不能随机访问`；
- 而且由于每个元素必须存储指向`前后元素位置的指针`，会消耗相对更多的储存空间。

---

### 二、数据结构的基本操作

对于任何数据结构，其基本操作无非 `遍历 + 访问`，再具体一点就是：`增删查改`。
- 数据结构种类很多，但它们存在的目的都是在不同的应用场景，尽可能高效地增删查改。 -> 数据结构的使命

遍历 + 访问
- 各种数据结构的遍历 + 访问无非两种形式：`线性`的和`非线性`的。
- **线性**就是 `for/while` 迭代为代表，
- **非线性**就是`递归`为代表。


再具体一步，无非以下几种框架：


#### **数组遍历框架**，典型的`线性` `迭代`结构：

```java
void traverse(int[] arr) {
    for (int i = 0; i < arr.length; i++) {
        // 迭代访问 arr[i]
    }
}
```


#### **链表遍历框架**，兼具`迭代`和`递归`结构：

```java
/* 基本的单链表节点 */
class ListNode {
    int val;
    ListNode next;
}
​
void traverse(ListNode head) {
    for (ListNode p = head; p != null; p = p.next) {
        // 迭代访问 p.val
    }
}
​
void traverse(ListNode head) {
    // 递归访问 head.val
    traverse(head.next);
}
```


#### **二叉树遍历框架**，典型的`非线性` `递归` `遍历` 结构：

```java
/* 基本的二叉树节点 */
class TreeNode {
    int val;
    TreeNode left, right;
}
​
void traverse(TreeNode root) {
    traverse(root.left);
    traverse(root.right);
}
```

你看二叉树的`递归遍历`方式和链表的`递归遍历`方式，相似不？
- 再看看二叉树结构和单链表结构，相似不？
- 如果再多几条叉，N 叉树你会不会遍历？


#### 二叉树框架 扩展为 **N 叉树的遍历框架**

```java
/* 基本的 N 叉树节点 */
class TreeNode {
    int val;
    TreeNode[] children;
}
​
void traverse(TreeNode root) {
    for (TreeNode child : root.children) {
        traverse(child);
    }
}
```

#### **图的遍历**

- N 叉树的遍历又可以扩展为图的遍历，因为图就是好几 N 叉棵树的结合体。
- 你说图是可能出现环的？这个很好办，用个布尔数组 visited 做标记就行了，这里就不写代码了。



所谓框架，就是套路。
- 不管增删查改，这些代码都是永远无法脱离的结构，
- 你可以把这个结构作为大纲，根据具体问题在框架上添加代码就行了

---

### 三、算法刷题指南

首先要明确的是，数据结构是工具，算法是通过合适的工具解决特定问题的方法。
- 也就是说，学习算法之前，最起码得了解那些常用的数据结构，了解它们的特性和缺陷。

先刷二叉树，先刷二叉树，先刷二叉树！

刷二叉树看到题目没思路, 没有理解我们说的「框架」是什么。

不要小看这几行破代码，几乎所有二叉树的题目都是一套这个框架就出来了：

```java
void traverse(TreeNode root) {
    // 前序遍历代码位置
    traverse(root.left)
    // 中序遍历代码位置
    traverse(root.right)
    // 后序遍历代码位置
}
```

比如说我随便拿几道题的解法出来，不用管具体的代码逻辑，只要看看框架在其中是如何发挥作用的就行。

```java
// LeetCode 124 题，难度 Hard，
// 求二叉树中最大路径和，主要代码如下：

int ans = INT_MIN;
int oneSideMax(TreeNode* root) {
    if (root == nullptr) return 0;
    int left = max(0, oneSideMax(root->left));
    int right = max(0, oneSideMax(root->right));

    // 后序遍历代码位置
    ans = max(ans, left + right + root->val);
    return max(left, right) + root->val;
}
```

注意递归函数的位置，这就是个后序遍历嘛，无非就是把 traverse 函数名字改成 oneSideMax 了。

```java
// LeetCode 105 题，难度 Medium，
// 根据前序遍历和中序遍历的结果还原一棵二叉树，很经典的问题吧，主要代码如下：

TreeNode buildTree(int[] preorder, int preStart, int preEnd,
                    int[] inorder, int inStart, int inEnd,
                    Map<Integer, Integer> inMap) {
​
    if(preStart > preEnd || inStart > inEnd) return null;
​
    TreeNode root = new TreeNode(preorder[preStart]);
    int inRoot = inMap.get(root.val);
    int numsLeft = inRoot - inStart;
​
    root.left = buildTree(preorder, preStart + 1, preStart + numsLeft,
                          inorder, inStart, inRoot - 1,
                          inMap);
    root.right = buildTree(preorder, preStart + numsLeft + 1, preEnd,
                           inorder, inRoot + 1, inEnd,
                           inMap);
    return root;
}
```

不要看这个函数的参数很多，只是为了控制数组索引而已。
- 注意找递归函数的位置，本质上该算法也就是一个`前序遍历`，因为它在前序遍历的位置加了一坨代码。

```java
// LeetCode 99 题，难度 Hard
// 恢复一棵 BST，主要代码如下：

void traverse(TreeNode* node) {
    if (!node) return;
    traverse(node->left);
    if (node->val < prev->val) {
        s = (s == NULL) ? prev : s;
        t = node;
    }
    prev = node;
    traverse(node->right);
}
```

这不就是个中序遍历嘛，对于一棵 BST 中序遍历意味着什么，应该不需要解释了吧。

你看，Hard 难度的题目不过如此，而且还这么有规律可循，只要把框架写出来，然后往相应的位置加东西就行了，这不就是思路吗。

对于一个理解二叉树的人来说，刷一道二叉树的题目花不了多长时间。
- 那么如果你对刷题无从下手或者有畏惧心理，不妨从二叉树下手，
- 前 10 道也许有点难受；结合框架再做 20 道，也许你就有点自己的理解了；
- 刷完整个专题，再去做什么回溯动规分治专题，你就会发现只要涉及递归的问题，都是树的问题。

再举例吧，说几道我们之前文章写过的问题。

​动态规划详解说过凑零钱问题，暴力解法就是遍历一棵 N 叉树：

```py
def coinChange(coins: List[int], amount: int):
    def dp(n):
        if n == 0: return 0
        if n < 0: return -1
        res = float('INF')
        for coin in coins:
            subproblem = dp(n - coin)
            # 子问题无解，跳过
            if subproblem == -1: continue
            res = min(res, 1 + subproblem)
        return res if res != float('INF') else -1
​
    return dp(amount)
# 这么多代码看不懂咋办？直接提取出框架，就能看出核心思路了：

# 不过是一个 N 叉树的遍历问题而已
def dp(n):
    for coin in coins:
        dp(n - coin)
```

其实很多动态规划问题就是在遍历一棵树，
- 你如果对树的遍历操作烂熟于心，起码知道怎么把思路转化成代码，也知道如何提取别人解法的核心思路。

再看看回溯算法
- `回溯算法`就是个 N 叉树的`前后序遍历`问题，没有例外。

比如全排列问题吧，本质上全排列就是在遍历下面这棵树，到叶子节点的路径就是一个全排列：

```java
// 全排列算法的主要代码如下：

// void backtrack(int[] nums, LinkedList<Integer> track) {
//     if (track.size() == nums.length) {
//         res.add(new LinkedList(track));
//         return;
//     }
// ​
//     for (int i = 0; i < nums.length; i++) {
//         if (track.contains(nums[i]))
//             continue;
//         track.add(nums[i]);
//         // 进入下一层决策树
//         backtrack(nums, track);
//         track.removeLast();
//     }
​
// /提取出 N 叉树遍历框架/
// void backtrack(int[] nums, LinkedList<Integer> track) {
//     for (int i = 0; i < nums.length; i++) {
//         backtrack(nums, track);
// }
```

N 叉树的遍历框架
- 先刷树的相关题目，试着从框架上看问题，而不要纠结于细节问题。
- 纠结细节问题，就比如纠结 i 到底应该加到 n 还是加到 `n - 1`，这个数组的大小到底应该开 n 还是 n + 1？

从框架上看问题
- 基于框架进行抽取和扩展，既可以在看别人解法时快速理解核心逻辑，也有助于找到我们自己写解法时的思路方向。
- 如果细节出错，你得不到正确的答案，但是只要有框架，你再错也错不到哪去，因为你的方向是对的。
- 没有框架，那根本无法解题，给了你答案，你也不会发现这就是个树的遍历问题。
- 这种思维是很重要的，动态规划详解中总结的找状态转移方程的几步流程，有时候按照流程写出解法，说实话我自己都不知道为啥是对的，反正它就是对了。。。
- 这就是框架的力量，能够保证你在快睡着的时候，依然能写出正确的程序；就算你啥都不会，都能比别人高一个级别。

### 四、总结几句

数据结构的
- **基本存储方式** 就是`链式`和`顺序`两种，
  - `数组`（顺序存储）
  - `链表`（链式存储）。
- **基本操作** 就是`增删查改`，
- **遍历方式** 无非`迭代`和`递归`。


---


# two pointer

## two pointer - Array 数组

原地修改数组

数组
- 在尾部插入、删除元素是比较高效的，时间复杂度是`1`，
- 在中间或者开头插入、删除元素，就会涉及数据的搬移，时间复杂度为`O(N)`，效率较低。

Do not allocate extra space for another array. You must do this by modifying the input array in-place with O(1) extra memory.

如何在原地修改数组，避免数据的搬移。
- 如果不是原地修改的话，直接 new 一个 int[] 数组，把去重之后的元素放进这个新数组中，然后返回这个新数组即可。
- 原地删除不允许 new 新数组，只能在原数组上操作，然后返回一个长度，这样就可以通过返回的长度和原始数组得到我们去重后的元素有哪些了。


---


### 26. Remove Duplicates from Sorted Array 有序数组去重（简单）`快慢指针前后走`

[26. Remove Duplicates from Sorted Array](https://leetcode.com/problems/remove-duplicates-from-sorted-array/)

![Screen Shot 2021-10-10 at 10.21.49 PM](https://i.imgur.com/71PNcPT.png)

在数组相关的算法题中时非常常见的，通用解法就是使用快慢指针技巧。
- 让慢指针 slow 走在后面，快指针 fast 走在前面探路
- 找到一个不重复的元素就告诉 slow 并让 slow 前进一步。
- 这样当 fast 指针遍历完整个数组 nums 后，`nums[0..slow]` 就是不重复元素。

```java
int removeDuplicates(int[] nums) {
    if (nums.length == 0) return 0;
    int slow = 0, fast = 0;
    while (fast < nums.length) {
        if (nums[fast] != nums[slow]) {
            slow++;
            // 维护 nums[0..slow] 无重复
            nums[slow] = nums[fast];
        }
        fast++;
    }
    // 数组长度为索引 + 1
    return slow + 1;
}

// Runtime: 1 ms, faster than 82.01% of Java online submissions for Remove Duplicates from Sorted Array.
// Memory Usage: 45.1 MB, less than 6.26% of Java online submissions for Remove Duplicates from Sorted Array.
/**
 * Using 2 pointers.
 *
 * Time Complexity: O(N)
 *
 * Space Complexity: O(1)
 *
 * N = Length of input array.
 */
int removeDuplicates(int[] nums) {
    if (nums == null) throw new IllegalArgumentException("Input is invalid");
    if (nums.length <= 1) return nums.length;
    int slow = 0, fast = 1;
    while (fast < nums.length) {
        if (nums[fast] != nums[slow]) {
            slow++;
            // 维护 nums[0..slow] 无重复
            nums[slow] = nums[fast];
        }
        fast++;
    }
    // 数组长度为索引 + 1
    return slow + 1;
}

// Runtime: 1 ms, faster than 82.01% of Java online submissions for Remove Duplicates from Sorted Array.
// Memory Usage: 44.3 MB, less than 23.95% of Java online submissions for Remove Duplicates from Sorted Array.

int removeDuplicates(int[] nums) {
    if (nums == null) throw new IllegalArgumentException("Input is invalid");  
    if (nums.length <= 1) return nums.length;
    int slow = 0;
    for(int i=1; i<nums.length; i++){
        if (nums[i] != nums[slow]) nums[++slow] = nums[i];
    }
    // 数组长度为索引 + 1
    return slow + 1;
}
```


```java
// Runtime: 1 ms, faster than 82.01% of Java online submissions for Remove Duplicates from Sorted Array.
// Memory Usage: 40.2 MB, less than 80.01% of Java online submissions for Remove Duplicates from Sorted Array.

public int removeDuplicates(int[] nums) {
        int i = 0;
        for (int n : nums){
            if (i == 0 || n > nums[i-1]){
                nums[i] = n;
                i++;
            }
        }
        return i;
    }

public int removeDuplicates(int[] nums) {
    int i = nums.length > 0 ? 1 : 0;
    for (int n : nums)
        if (n > nums[i-1])
            nums[i++] = n;
    return i;
}
```




```py
from collections import OrderedDict
from typing import List

# Method 1 ----- new list
def removeDuplicates(test_list):
    res = []
    for i in test_list:
        if i not in res:
            res.append(i)

# Method 2 ----- new list
def removeDuplicates(test_list):
    res = []
    [res.append(x) for x in test_list if x not in res]

# Method 3 ------ set(x)
def removeDuplicates(test_list):
    # the ordering of the element is lost
    test_list = list(set(test_list))

# Method 4 ------ Using list comprehension + enumerate()
def removeDuplicates(test_list):
    res = [i for n, i in enumerate(test_list)]

# Method 5 : Using collections.OrderedDict.fromkeys()
def removeDuplicates(test_list):
    res = list(OrderedDict.fromkeys(test_list))
    # maintain the insertion order as well
    res = list(dict.fromkeys(test_list))

# Method 6 ------ 快慢指针
def removeDuplicates(test_list):
    # Runtime: 72 ms, faster than 99.60% of Python3 online submissions for Remove Duplicates from Sorted Array.
    # Memory Usage: 15.7 MB, less than 45.93% of Python3 online submissions for Remove Duplicates from Sorted Array.
    fast, slow = 0,0
    if len(test_list) == 0: return 0
    while fast < len(test_list):
        print(test_list)
        print(test_list[fast])

        if test_list[slow] != test_list[fast]:
            slow +=1
            test_list[slow] = test_list[fast]
        fast += 1
    print(test_list[0:slow+1])
    return slow+1

# removeDuplicates([0,0,1,2,2,3,3])
```


---

### 80. Remove Duplicates from Sorted Array II `nums[i]!=nums[i-2]`

[80. Remove Duplicates from Sorted Array II](https://leetcode.com/problems/remove-duplicates-from-sorted-array-ii/)

Given an integer array nums sorted in non-decreasing order, remove some duplicates in-place such that each unique element appears at most twice. The relative order of the elements should be kept the same.

Since it is impossible to change the length of the array in some languages, you must instead have the result be placed in the first part of the array nums. More formally, if there are k elements after removing the duplicates, then the first k elements of nums should hold the final result. It does not matter what you leave beyond the first k elements.

Return k after placing the final result in the first k slots of nums.

Do not allocate extra space for another array. You must do this by modifying the input array in-place with O(1) extra memory.

```java
// Runtime: 0 ms, faster than 100.00% of Java online submissions for Remove Duplicates from Sorted Array II.
// Memory Usage: 39.3 MB, less than 39.45% of Java online submissions for Remove Duplicates from Sorted Array II.
/**
 * In place, one pass solution using 2 pointers
 *
 * Time Complexity: O(N)
 *
 * Space Complexity: O(1)
 *
 * N = Length of input array.
 */
public int removeDuplicates(int[] nums) {
    if (nums == null) throw new IllegalArgumentException("Input array is null");  
    if (nums.length <= 2) return nums.length;
    int insertPos = 1;
    for (int i = 2; i < nums.length; i++) {
        if (nums[i] != nums[insertPos - 1]) {
            nums[++insertPos] = nums[i];
        }
    }
    return insertPos + 1;
}
```

---

### FU. Each unique element should appear at most K times

```java
/**
 * Follow-Up: Each unique element should appear at most K times.
 *
 * In place, one pass solution using 2 pointers
 *
 * Time Complexity: O(N-K)
 *
 * Space Complexity: O(1)
 *
 * N = Length of input array.
 */
class Solution {
    public int removeDuplicates(int[] nums) {
        return removeDuplicatesMoreThanK(nums, 2);
    }
    public int removeDuplicatesMoreThanK(int[] nums, int k) {
        if (nums == null || k < 0) throw new IllegalArgumentException("Invalid Input");
        if (k == 0) return 0;  
        if (nums.length <= k) return nums.length;
        int insertPos = k - 1;
        for (int i = k; i < nums.length; i++) {
            if (nums[i] != nums[insertPos - (k - 1)]) {
                nums[++insertPos] = nums[i];
            }
        }
        return insertPos + 1;
    }
}
```

---

### 27. Remove Element 移除元素 （简单）`快慢指针前后走`

把 nums 中所有值为 val 的元素原地删除，依然需要使用 `双指针技巧` 中的 `快慢指针`：
- 如果 fast 遇到需要去除的元素，则直接跳过，
- 否则就告诉 slow 指针，并让 slow 前进一步。

[27. Remove Element](https://leetcode.com/problems/remove-element/)

Given an integer array nums and an integer val, remove all occurrences of val in nums in-place. The relative order of the elements may be changed.

Since it is impossible to change the length of the array in some languages, you must instead have the result be placed in the first part of the array nums. More formally, if there are k elements after removing the duplicates, then the first k elements of nums should hold the final result. It does not matter what you leave beyond the first k elements.

Return k after placing the final result in the first k slots of nums.

`Do not allocate extra space` for another array. You must do this by modifying the input array in-place with O(1) extra memory.

```java
// Runtime: 0 ms, faster than 100.00% of Java online submissions for Remove Element.
// Memory Usage: 38.9 MB, less than 24.52% of Java online submissions for Remove Element.
/**
 * Using Two Pointers. Output array maintains the order of the input array.
 *
 * Time Complexity: O(N)
 *
 * Space Complexity: O(1)
 *
 * N = Length of input array.
 */
int removeElement(int[] nums, int val) {
    int fast = 0, slow = 0;
    while (fast < nums.length) {
        if (nums[fast] != val) {
            nums[slow] = nums[fast];
            slow++;
        }
        fast++;
    }
    return slow;
}

// Runtime: 0 ms, faster than 100.00% of Java online submissions for Remove Element.
// Memory Usage: 38.4 MB, less than 25.52% of Java online submissions for Remove Element.
public int removeElement(int[] nums, int val) {
    if (nums == null) throw new IllegalArgumentException("Input array is null");
    if(nums.length==0) return 0;
    int slow=0;
    for(int i=0;i<nums.length;i++){
        if(nums[i]!=val) {
            nums[slow++]=nums[i];
        }
    }
    return slow;
}
```





```py
# Runtime: 32 ms, faster than 81.50% of Python3 online submissions for Remove Element.
# Memory Usage: 14.2 MB, less than 47.25% of Python3 online submissions for Remove Element.
def removeElement(nums: List[int], val: int) -> int:
    slow, fast = 0,0
    while fast < len(nums):
        if nums[fast] != val:
            nums[slow] = nums[fast]
            slow += 1
        fast += 1

# removeElement([0,0,1,2,2,3,3], 2)
```



---



### 83. Remove Duplicates from Sorted List 有序链表去重 `快慢指针前后走`

[83. Remove Duplicates from Sorted List](https://leetcode.com/problems/remove-duplicates-from-sorted-list/submissions/)

Given the head of a sorted linked list, delete all duplicates such that each element appears only once. Return the linked list sorted as well.


```java
ListNode deleteDuplicates(ListNode head) {
    if (head == null) return null;
    ListNode slow = head, fast = head;
    while (fast != null) {
        if (fast.val != slow.val) {
            // nums[slow] = nums[fast];
            slow.next = fast;
            // slow++;
            slow = slow.next;
        }
        // fast++
        fast = fast.next;
    }
    // 断开与后面重复元素的连接
    slow.next = null;
    return head;
}
```

```py
from basic import LinkedList, Node

# 两个指针
# Runtime: 40 ms, faster than 84.87% of Python3 online submissions for Remove Duplicates from Sorted List.
# Memory Usage: 14.2 MB, less than 56.16% of Python3 online submissions for Remove Duplicates from Sorted List.
def deleteDuplicates(LL):
    if not LL: return 0
    slow, fast = LL.head, LL.head
    if LL.head == None: return LL.head
    while fast != None:
        if slow.val != fast.val:
            slow.next = fast
            slow = slow.next
        fast = fast.next
    slow.next = None
    # print(LL.val)
    return LL

# 一个指针
def deleteDuplicates(LL):
    cur = LL.head
    while cur:
        while cur.next and cur.val == cur.next.val:
            cur.next = cur.next.next     # skip duplicated node
        cur = cur.next     # not duplicate of current node, move to next node
    return LL

# nice for if the values weren't sorted in the linked list
def deleteDuplicates(LL):
    dic = {}
    node = LL.head
    while node:
        dic[node.val] = dic.get(node.val, 0) + 1
        node = node.next
    node = LL.head
    while node:
        tmp = node
        for _ in range(dic[node.val]):
            tmp = tmp.next
        node.next = tmp
        node = node.next
    return LL

# recursive
def deleteDuplicates(LL):
    if not LL.head: return LL
    if LL.head.next is not None:
        if LL.head.val == LL.head.next.val:
            LL.head.next = LL.head.next.next
            deleteDuplicates(LL.head)
        else:
            deleteDuplicates(LL.head.next)
    return LL

LL = LinkedList()
list_num = [0,0,1,2,2,3,3]
for i in list_num:
    LL.insert(i)
LL.printLL()

LL = deleteDuplicates(LL)
LL.printLL()
```


---


### 283. Move Zeroes 移除0 `快慢指针前后走`

[283. Move Zeroes](https://leetcode.com/problems/move-zeroes/)

Given an integer array nums, move all 0's to the end of it while maintaining the relative order of the non-zero elements.

Note that you must do this in-place without making a copy of the array.


```java
void moveZeroes(int[] nums) {
    // 去除 nums 中的所有 0
    // 返回去除 0 之后的数组长度
    int p = removeElement(nums, 0);
    // 将 p 之后的所有元素赋值为 0
    for (; p < nums.length; p++) {
        nums[p] = 0;
    }
}

// 见上文代码实现
int removeElement(int[] nums, int val) {
    int fast = 0, slow = 0;
    while (fast < nums.length) {
        if (nums[fast] != val) {
            nums[slow] = nums[fast];
            slow++;
        }
        fast++;
    }
    return slow;
}
```


```py

# =============== 移除0
# 两个指针
def moveZeroes(nums: List[int]) -> None:
    # Runtime: 188 ms, faster than 17.89% of Python3 online submissions for Move Zeroes.
    # Memory Usage: 15.6 MB, less than 7.33% of Python3 online submissions for Move Zeroes.
    slow, fast = 0,0
    if nums == []:
        return []
    while fast < len(nums):
        print(nums[fast])
        if nums[fast] != 0:
            nums[slow] = nums[fast]
            slow+=1
        fast+=1
    for i in range(slow, len(nums)):
        nums[i] = 0
    print(nums)

# 一个指针
def moveZeroes(nums: List[int]) -> None:
    # Runtime: 172 ms, faster than 25.48% of Python3 online submissions for Move Zeroes.
    # Memory Usage: 15.4 MB, less than 24.21% of Python3 online submissions for Move Zeroes.
    slow = 0
    if nums == []:
        return []
    for i in range(len(nums)):
        if nums[i] != 0:
            nums[slow] = nums[i]
            slow+=1
        i+=1
    for i in range(slow, len(nums)):
        nums[i] = 0
    print(nums)


def moveZeroes(self, nums: List[int]) -> None:
    # Runtime: 248 ms, faster than 13.91% of Python3 online submissions for Move Zeroes.
    # Memory Usage: 15.2 MB, less than 88.67% of Python3 online submissions for Move Zeroes.
    slow = 0
    leng = len(nums)
    if nums == []:
        return []
    for i in range(leng):
        if nums[i] != 0:
            nums[slow] = nums[i]
            slow+=1
    for i in range(slow, leng):
        nums[i] = 0
    return nums

# Runtime: 260 ms, faster than 13.33% of Python3 online submissions for Move Zeroes.
# Memory Usage: 15.5 MB, less than 24.34% of Python3 online submissions for Move Zeroes.
def moveZeroes(nums: List[int]) -> None:
    slow = 0
    for i in range(len(nums)):
        if nums[i] != 0:
            nums[slow],nums[i] = nums[i],nums[slow]
            slow +=1

# moveZeroes([0,1,0,3,12])
```

---


### 349. Intersection of Two Arrays (Easy)


[349. Intersection of Two Arrays](https://leetcode.com/problems/intersection-of-two-arrays/)
Given two integer arrays nums1 and nums2, return an array of their intersection. Each element in the result must be unique and you may return the result in any order.

Example 1:

Input: nums1 = [1,2,2,1], nums2 = [2,2]
Output: [2]



#### ++++++++++ `Hash(num1 had), Hash.remove(num2 has)` BEST

```java
// Runtime: 2 ms, faster than 95.44% of Java online submissions for Intersection of Two Arrays.
// Memory Usage: 38.9 MB, less than 87.06% of Java online submissions for Intersection of Two Arrays.
class Solution {
    public int[] intersection(int[] nums1, int[] nums2) {
        HashSet<Integer> set = new HashSet<Integer>();
        ArrayList<Integer> ans = new ArrayList<>();
        for(int num:nums1) set.add(num);
        for(int num:nums2) {
            if(set.contains(num)){
                ans.add(num);
                set.remove(num);
            }
        }
        int[] res = new int[ans.size()];
        for(int i=0; i<ans.size(); i++){
            res[i] = ans.get(i);
        }
        return res;
    }
}
```

#### `sorting, compare, get the same`

```java
// Runtime: 2 ms, faster than 95.33% of Java online submissions for Intersection of Two Arrays.
// Memory Usage: 38.9 MB, less than 86.77% of Java online submissions for Intersection of Two Arrays.
class Solution {
    public int[] intersection(int[] nums1, int[] nums2) {
        Arrays.sort(nums1);
        Arrays.sort(nums2);
        int pt1 = 0, pt2=0;
        ArrayList<Integer> ans = new ArrayList<>();
        while (pt1 < nums1.length && pt2 < nums2.length) {
            if(nums1[pt1]<nums2[pt2]) pt1 = nextPT(nums1, pt1);
            else if(nums1[pt1]>nums2[pt2]) pt2 = nextPT(nums2, pt2);
            else{
                ans.add(nums1[pt1]);
                pt1 = nextPT(nums1, pt1);
                pt2 = nextPT(nums2, pt2);
            }
        }
        int[] res = new int[ans.size()];
        for(int i=0; i<res.length; i++) {
            res[i] = ans.get(i);
        }
        return res;
    }
    public int nextPT(int[] nums, int pt) {
        int value = nums[pt];
        while(pt<nums.length && nums[pt] == value) pt++;
        return pt;
    }
}
```

---


### 1385. Find the Distance Value Between Two Arrays (Easy)

[1385. Find the Distance Value Between Two Arrays](https://leetcode.com/problems/find-the-distance-value-between-two-arrays/)
Given two integer arrays arr1 and arr2, and the integer d, return the distance value between the two arrays.

The distance value is defined as the number of elements arr1[i] such that there is not any element arr2[j] where |arr1[i]-arr2[j]| <= d.

Example 1:

Input: arr1 = [4,5,8], arr2 = [10,9,1,8], d = 2
Output: 2
Explanation:
For arr1[0]=4 we have:
|4-10|=6 > d=2
|4-9|=5 > d=2
|4-1|=3 > d=2
|4-8|=4 > d=2
For arr1[1]=5 we have:
|5-10|=5 > d=2
|5-9|=4 > d=2
|5-1|=4 > d=2
|5-8|=3 > d=2
For arr1[2]=8 we have:
|8-10|=2 <= d=2
|8-9|=1 <= d=2
|8-1|=7 > d=2
|8-8|=0 <= d=2


#### brute force

```java
// Runtime: 3 ms, faster than 75.47% of Java online submissions for Find the Distance Value Between Two Arrays.
// Memory Usage: 38.5 MB, less than 70.69% of Java online submissions for Find the Distance Value Between Two Arrays.
// O(n^2)
class Solution {
    public int findTheDistanceValue(int[] arr1, int[] arr2, int d) {
        int count = arr1.length;
        for(int nums1:arr1){
            for(int nums2:arr2){
                if(Math.abs(nums1-nums2)<=d){
                    count--;
                    break;
                }
            }
        }
        return count;
    }
}

```


#### Binary Search

```java
// Runtime: 3 ms, faster than 76.94% of Java online submissions for Find the Distance Value Between Two Arrays.
// Memory Usage: 38.6 MB, less than 56.39% of Java online submissions for Find the Distance Value Between Two Arrays.
class Solution {
    public int findTheDistanceValue(int[] arr1, int[] arr2, int d) {
        Arrays.sort(arr1);
        Arrays.sort(arr2);
        int count = 0, closeDis;
        for(int nums1:arr1) {
            closeDis = bs(arr2, 0, arr2.length-1 , nums1);
            if(closeDis>d) count++;
        }
        return count;
    }
    public int bs(int[] arr2, int lo, int hi , int value) {  
        while(lo>hi) return Integer.MAX_VALUE;
        int mid = (lo + hi)/2;
        int dis=Math.abs(arr2[mid] - value);
        if(arr2[mid] > value) dis = Math.min(dis, bs(arr2, lo, mid-1 , value));
        else dis = Math.min(dis, bs(arr2, mid+1, hi , value));
        return dis;
    }
}
```

#### ???

```java
// O(nlogm)
class Solution {
    public int findTheDistanceValue(int[] arr1, int[] arr2, int d) {
        int count = 0;
        TreeSet<Integer> tree = new TreeSet<>();
        for (int number: arr2) {
            tree.add(number);
        }
        for (int i=0; i<arr1.length; i++) {
            int leftValue = arr1[i] - d;
            int rightValue = arr1[i] + d;
            Set<Integer> set = tree.subSet(leftValue, rightValue+1);
            if (set.isEmpty())
                count += 1;
        }
        return count;
    }
}
```

#### `sort + sliding window` BEST

```java
// O(NLogN)

// Runtime: 2 ms, faster than 96.65% of Java online submissions for Find the Distance Value Between Two Arrays.
// Memory Usage: 38.6 MB, less than 68.97% of Java online submissions for Find the Distance Value Between Two Arrays.

class Solution {
    public int findTheDistanceValue(int[] arr1, int[] arr2, int d) {
        Arrays.sort(arr1);
        Arrays.sort(arr2);
        int count=0, j=0;
        for(int i=0;i<arr1.length;i++){
            int min = arr1[i]-d;
            int max = arr1[i]+d;
            while(j<arr2.length && arr2[j]<min) j++;
            if(outband(arr2, j, min, max)) count++;
        }
        return count;
    }
    public boolean outband(int[] arr2, int j, int min, int max) {  
        return j==arr2.length || !(min<=arr2[j] && arr2[j]<=max);
    }
}
```

---

### 696. Count Binary Substrings (Easy)

Give a binary string s, return the number of non-empty substrings that have the same number of 0's and 1's, and all the 0's and all the 1's in these substrings are grouped consecutively.

Substrings that occur multiple times are counted the number of times they occur.

Example 1:

Input: s = "00110011"
Output: 6
Explanation: There are 6 substrings that have equal number of consecutive 1's and 0's: "0011", "01", "1100", "10", "0011", and "01".
Notice that some of these substrings repeat and are counted the number of times they occur.
Also, "00110011" is not a valid substring because all the 0's (and 1's) are not grouped together.


the number that we should add to ans is equal to min(zeros, ones), or pre count

#### Brute Force                            
Check for every substring either they are valid substring or not. if valid increase the count but time complexity :O(n^3)


```java
// Runtime: 21 ms, faster than 7.53% of Java online submissions for Count Binary Substrings.
// Memory Usage: 46 MB, less than 15.02% of Java online submissions for Count Binary Substrings.
class Solution {
    public int countBinarySubstrings(String s) {
        int res=0, pre=0, cur=1, i=0;
        while(i<s.length()-1){
            if(s.charAt(i+1)!=s.charAt(i)){
                res+=Math.min(pre, cur);
                pre=cur;
                cur=1;
            }
            else cur++;
            i++;
        }
        return res+=Math.min(pre, cur);
    }
}
```



---



## two pointer - 链表

---

### 203. Remove Linked List Elements (Easy)

[203. Remove Linked List Elements](https://leetcode.com/problems/remove-linked-list-elements/)

Given the head of a linked list and an integer val, remove all the nodes of the linked list that has Node.val == val, and return the new head.

Input: head = [1,2,6,3,4,5,6], val = 6
Output: [1,2,3,4,5]


```java
/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode() {}
 *     ListNode(int val) { this.val = val; }
 *     ListNode(int val, ListNode next) { this.val = val; this.next = next; }
 * }
 */

// Runtime: 1 ms, faster than 74.37% of Java online submissions for Remove Linked List Elements.
// Memory Usage: 39.4 MB, less than 98.31% of Java online submissions for Remove Linked List Elements.

class Solution {
    public ListNode removeElements(ListNode head, int val) {
        if (head == null) return null;
        ListNode dummy = new ListNode(-1);
        dummy.next = head;
        ListNode cur = head, pre = dummy;
        while(cur !=null){
            if(cur.val == val) pre.next = cur.next;
            else pre = cur;
            cur = cur.next;
        }
        return dummy.next;
    }
}

// Runtime: 0 ms, faster than 100.00% of Java online submissions for Remove Linked List Elements.
// Memory Usage: 40.6 MB, less than 18.70% of Java online submissions for Remove Linked List Elements.
class Solution {
    public ListNode removeElements(ListNode head, int val) {
        if (head == null) return null;
        if (head.val==val) return removeElements(head.next,  val);
        ListNode dummy = new ListNode(-1);
        dummy.next = head;
        ListNode cur = head;
        while(cur.next !=null){
            if(cur.next.val == val) cur.next = cur.next.next;
            else cur = cur.next;
        }
        return dummy.next;
    }
}

```



#### ++++++++++ recursive solution

```java
public ListNode removeElements(ListNode head, int val) {
        if (head == null) return null;
        head.next = removeElements(head.next, val);
        return head.val == val ? head.next : head;
}
```


---

### 237. Delete Node in a Linked List (Easy)

[237. Delete Node in a Linked List](https://leetcode.com/problems/delete-node-in-a-linked-list/)
Write a function to delete a node in a singly-linked list. You will not be given access to the head of the list, instead you will be given access to the node to be deleted directly.

It is guaranteed that the node to be deleted is not a tail node in the list.

```java
/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode(int x) { val = x; }
 * }
 */
// Runtime: 0 ms, faster than 100.00% of Java online submissions for Delete Node in a Linked List.
// Memory Usage: 40.9 MB, less than 12.23% of Java online submissions for Delete Node in a Linked List.
class Solution {
    public void deleteNode(ListNode node) {
        node.val=node.next.val;
        node.next = node.next.next;
    }
}
```

---


### 876. Middle of the Linked List 寻找单链表的中点

point: 无法直接得到单链表的长度 n，
- 常规方法也是先遍历链表计算 n，再遍历一次得到第 n / 2 个节点，也就是中间节点。

solution:
- 两个指针 slow 和 fast 分别指向链表头结点 head。
- 每当慢指针 slow 前进一步，快指针 fast 就前进两步，
- 这样当 fast 走到链表末尾时，slow 就指向了链表中点。

> 如果链表长度为偶数，中点有两个的时候，返回的节点是靠后的那个节点。
> 这段代码稍加修改就可以直接用到判断链表成环的算法题上。

让快指针一次前进两步，慢指针一次前进一步，当快指针到达链表尽头时，慢指针就处于链表的中间位置。

[876. Middle of the Linked List](https://leetcode.com/problems/middle-of-the-linked-list/)
- Given the head of a singly linked list, return the middle node of the linked list.
- If there are two middle nodes, return the second middle node.


```java
// Runtime: 0 ms, faster than 100.00% of Java online submissions for Middle of the Linked List.
// Memory Usage: 36.4 MB, less than 67.08% of Java online submissions for Middle of the Linked List.

ListNode middleNode(ListNode head) {
    ListNode fast, slow;
    fast = slow = head;
    while (fast != null && fast.next != null) {
        fast = fast.next.next;
        slow = slow.next;
    }
    // slow 就在中间位置
    return slow;
}
```


---

### 2095. Delete the Middle Node of a Linked List (Medium)


[2095. Delete the Middle Node of a Linked List](https://leetcode.com/problems/delete-the-middle-node-of-a-linked-list/)
You are given the head of a linked list. Delete the middle node, and return the head of the modified linked list.

The middle node of a linked list of size n is the ⌊n / 2⌋th node from the start using 0-based indexing, where ⌊x⌋ denotes the largest integer less than or equal to x.

For n = 1, 2, 3, 4, and 5, the middle nodes are 0, 1, 1, 2, and 2, respectively.

Input: head = [1,3,4,7,1,2,6]
Output: [1,3,4,1,2,6]


```java
/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode() {}
 *     ListNode(int val) { this.val = val; }
 *     ListNode(int val, ListNode next) { this.val = val; this.next = next; }
 * }
 */
// O(n), O(1)
class Solution {
    public ListNode deleteMiddle(ListNode head) {
        if(head ==null || head.next == null) return null; // 0 or 1 nodes
        ListNode dummy = new ListNode(-1), fast = dummy, slow=dummy;
        dummy.next=head;
        while(fast.next !=null&&fast.next.next !=null){
            slow = slow.next;
            fast = fast.next.next;
        }
        slow.next=slow.next.next;
        return dummy.next;
    }
}
```

---


### 寻找单链表的倒数n节点

point: 算法题一般只给你一个 ListNode 头结点代表一条单链表，
- 不能直接得出这条链表的长度 n，
- 而需要先遍历一遍链表算出 n 的值，
- 然后再遍历链表计算第 n - k 个节点。

**只遍历一次链表**

```java
// 返回链表的倒数第 k 个节点
ListNode findFromEnd(ListNode head, int k) {
    ListNode fast = head, slow = head;
    // fast 先走 k 步
    while (n-- > 0) fast = fast.next;
    // 让慢指针和快指针同步向前
    while (fast != null && fast.next != null) {
        slow = slow.next;
        fast = fast.next;
    }
    // slow 现在指向第 n - k 个节点
    return slow;
}
```

时间复杂度
- 无论遍历一次链表和遍历两次链表的时间复杂度都是 O(N)，但上述这个算法更有技巧性。

---


### 19. Remove Nth Node From End of List remove倒数n节点 `删除倒数n,找倒数n+1`


[19. Remove Nth Node From End of List](https://leetcode.com/problems/remove-nth-node-from-end-of-list/)

Given the head of a linked list, remove the nth node from the end of the list and return its head.

```java
// Runtime: 0 ms, faster than 100.00% of Java online submissions for Remove Nth Node From End of List.
// Memory Usage: 37 MB, less than 75.59% of Java online submissions for Remove Nth Node From End of List.
public ListNode removeNthFromEnd(ListNode head, int n){
    // 虚拟头结点
    ListNode dummy = new ListNode(-1);
    dummy.next = head;
    // 删除倒数第 n 个，要先找倒数第 n + 1 个节点
    ListNode x = findFromEnd(dummy, n + 1);
    // 删掉倒数第 n 个节点
    x.next = x.next.next;
    return dummy.next;
}

// 返回链表的倒数第 k 个节点
private ListNode findFromEnd(ListNode head, int k){
    ListNode fast = head, slow = head;
    // fast 先走 k 步
    for(int i=0;i<k;i++) fast = fast.next;
    // 让慢指针和快指针同步向前
    while (fast != null && fast.next != null) {
        slow = slow.next;
        fast = fast.next;
    }
    // slow 现在指向第 n - k 个节点
    return slow;
}
```


```java
// Runtime: 1 ms, faster than 24.37% of Java online submissions for Remove Nth Node From End of List.
// Memory Usage: 38.6 MB, less than 26.69% of Java online submissions for Remove Nth Node From End of List.
// O(1) space
class Solution {
    public ListNode removeNthFromEnd(ListNode head, int n) {
        if(head==null) return head;
        // 删除倒数第 n 个，要先找倒数第 n + 1 个节点
        ListNode dummy = new ListNode(0,head);
        ListNode fast=dummy, slow=dummy;
        for(int i=0;i<n+1;i++){
            fast=fast.next;
        }
        while(fast!=null){
            slow=slow.next;
            fast=fast.next;
        }
        slow.next = slow.next.next;
        return dummy.next;
    }
}
```


---

### Delete N Nodes After M Nodes of a Linked List ??????????

Given a linked list and two integers M and N. Traverse the linked list such that you retain M nodes then delete next N nodes, continue the same till end of the linked list.

Input:
M = 2, N = 2
Linked List: 1->2->3->4->5->6->7->8
Output:
Linked List: 1->2->5->6

```java
// Function to skip M nodes and then
// delete N nodes of the linked list.
static void skipMdeleteN( Node head, int M, int N) {
    Node curr = head, t;
    int count;
    // The main loop that traverses through the whole list
    while (curr!=null)
    {
        // Skip M nodes
        for (count = 1; count < M && curr != null; count++) curr = curr.next;

        // If we reached end of list, then return
        if (curr == null) return;

        // Start from next node and delete N nodes
        t = curr.next;
        for (count = 1; count <= N && t != null; count++) {
            Node temp = t;
            t = t.next;
        }

        // Link the previous list with remaining nodes
        curr.next = t;

        // Set current pointer for next iteration
        curr = t;
    }
}
```




---

### 160. 判断两个单链表是否相交并找出交点

160 题「相交链表」
- 给你输入两个链表的头结点 headA 和 headB，这两个链表可能存在相交。
- 如果相交，你的算法应该返回相交的那个节点；如果没相交，则返回 null。


```java
// Runtime: 1 ms, faster than 98.52% of Java online submissions for Intersection of Two Linked Lists.
// Memory Usage: 42.2 MB, less than 57.90% of Java online submissions for Intersection of Two Linked Lists.

ListNode getIntersectionNode(ListNode headA, ListNode headB) {
    // p1 指向 A 链表头结点，p2 指向 B 链表头结点
    ListNode p1 = headA, p2 = headB;
    while (p1 != p2) {
        // p1 走一步，如果走到 A 链表末尾，转到 B 链表
        if (p1 == null) p1 = headB;
        else p1 = p1.next;
        // p2 走一步，如果走到 B 链表末尾，转到 A 链表
        if (p2 == null) p2 = headA;
        else p2 = p2.next;
    }
    return p1;
}
```


---


## two pointer - palindrome 回文

寻找回文串的核心思想是从中心向两端扩展：
- 回文串是对称的，所以正着读和倒着读应该是一样的，这一特点是解决回文串问题的关键。
- 因为回文串长度可能为奇数也可能是偶数，长度为奇数时只存在一个中心点，而长度为偶数时存在两个中心点，所以上面这个函数需要传入l和r。
- 「双指针技巧」，从两端向中间逼近即可：


```java
string palindrome(string& s, int l, int r) {
    // 防止索引越界
    while (l >= 0 && r < s.size() && s[l] == s[r]) {
        // 向两边展开
        l--; r++;
    }
    // 返回以 s[l] 和 s[r] 为中心的最长回文串
    return s.substr(l + 1, r - l - 1);
}
```

---


### 2108. Find First Palindromic String in the Array (Easy)

[2108. Find First Palindromic String in the Array](https://leetcode.com/problems/find-first-palindromic-string-in-the-array/)

Given an array of strings words, return the first palindromic string in the array. If there is no such string, return an empty string "".

A string is palindromic if it reads the same forward and backward.


#### ++++++++++ 2 pointer Check each word

```java
// Runtime: 2 ms, faster than 83.75% of Java online submissions for Find First Palindromic String in the Array.
// Memory Usage: 39.1 MB, less than 86.28% of Java online submissions for Find First Palindromic String in the Array.
class Solution {
    public String firstPalindrome(String[] words) {
        outers:
        for (String w : words) {
            for (int i = 0, j = w.length() - 1; i < j; i++, j--) {
                if (w.charAt(i) != w.charAt(j)) continue outers;
            }
            return w;
        }
        return "";
    }
}

// Runtime: 2 ms, faster than 83.75% of Java online submissions for Find First Palindromic String in the Array.
// Memory Usage: 39 MB, less than 86.28% of Java online submissions for Find First Palindromic String in the Array.
class Solution {
    public String firstPalindrome(String[] words) {
        for (String wd : words) {
            if (checkPali(wd)) return wd;
        }
        return "";
    }
    public boolean checkPali(String w) {
        for (int i = 0, j = w.length() - 1; i < j; i++, j--) {
            if (w.charAt(i) != w.charAt(j)) return false;
        }
        return true;
    }
}
```


#### ++++++++++ StringBuilder.reverse.equals

```java
class Solution {
    public String firstPalindrome(String[] words) {
        for(int i=0; i<words.length;i++){
            StringBuilder sb = new StringBuilder();
            sb.append(words[i]);
            sb.reverse();
            if(words[i].equals(sb. toString())) return words[i];
        }
        return "";
    }
}
```


---


### 832. Flipping an Image (Easy) `only same values flip both.`

[832. Flipping an Image](https://leetcode.com/problems/flipping-an-image/)
Given an n x n binary matrix image, flip the image horizontally, then invert it, and return the resulting image.

To flip an image horizontally means that each row of the image is reversed.

For example, flipping [1,1,0] horizontally results in [0,1,1].
To invert an image means that each 0 is replaced by 1, and each 1 is replaced by 0.

For example, inverting [0,1,1] results in [1,0,0].

Example 1:
Input: image = [[1,1,0],[1,0,1],[0,0,0]]
Output: [[1,0,0],[0,1,0],[1,1,1]]
Explanation: First reverse each row: [[0,1,1],[1,0,1],[0,0,0]].
Then, invert the image: [[1,0,0],[0,1,0],[1,1,1]]


```java
/**
 * Optimal one-pass in-place solution
 * If the values are not same, swap and flip will not change anything.
 * If the values are same, we will flip both.
 *
 * Time Complexity: O(N^2)
 * Space Complexity: O(1)
 * N = Matrix Size
 */
// Runtime: 0 ms, faster than 100.00% of Java online submissions for Flipping an Image.
// Memory Usage: 39.1 MB, less than 70.53% of Java online submissions for Flipping an Image.
class Solution {
    public int[][] flipAndInvertImage(int[][] image) {
        if (image == null || image.length == 0 || image[0].length == 0) return image;
        for(int[] row : image){
            int start=0, end=row.length-1;
            while(start<=end){
                if(row[start] == row[end]){
                    row[start] ^= 1; // XOR operate
                    row[end] = row[start];    
                }                
                start++;
                end--;
            }
        }
        return image;
    }
}
```

---

### 1332. Remove Palindromic Subsequences (Easy)

[1332. Remove Palindromic Subsequences](https://leetcode.com/problems/remove-palindromic-subsequences/)
You are given a string s consisting only of letters 'a' and 'b'. In a single step you can remove one palindromic subsequence from s.

Return the minimum number of steps to make the given string empty.

A string is a subsequence of a given string if it is generated by deleting some characters of a given string without changing its order. Note that a subsequence does not necessarily need to be contiguous.

A string is called palindrome if is one that reads the same backward as well as forward.

#### ++++++++++ `只有0，1，2 三种答案，aaabbb最多两下消完` Best

```java
// Runtime: 0 ms, faster than 100.00% of Java online submissions for Remove Palindromic Subsequences.
// Memory Usage: 37.1 MB, less than 38.55% of Java online submissions for Remove Palindromic Subsequences.
class Solution {
    public int removePalindromeSub(String s) {
        if(s.length()==0) return 0;
        int i=0, j=s.length()-1;
        while(i<j){
            if(s.charAt(i)!=s.charAt(j)) return 2;
            i++;
            j--;
        }
        return 1;
    }
}

class Solution {
    public int removePalindromeSub(String s) {
        if (s.length() == 0) return 0;
        return isPalindrome(s) ? 1 : 2;
    }
	//palindrome check
    private boolean isPalindrome(String s){
        int left = 0, right = s.length()-1;
        while (left < right)
            if (s.charAt(left++) != s.charAt(right--)) return false;
        return true;
    }
}
```


#### reverse logic also

check if the string is same as the reverse string then return 1 otherwise return 2



---

## two pointer - String

---

### 2000. Reverse Prefix of Word (Easy)

[2000. Reverse Prefix of Word](https://leetcode.com/problems/reverse-prefix-of-word/)
Given a 0-indexed string word and a character ch, reverse the segment of word that starts at index 0 and ends at the index of the first occurrence of ch (inclusive). If the character ch does not exist in word, do nothing.

For example, if word = "abcdefd" and ch = "d", then you should reverse the segment that starts at 0 and ends at 3 (inclusive). The resulting string will be "dcbaefd".
Return the resulting string.



Example 1:

Input: word = "abcdefd", ch = "d"
Output: "dcbaefd"
Explanation: The first occurrence of "d" is at index 3.
Reverse the part of word from 0 to 3 (inclusive), the resulting string is "dcbaefd".

#### ++++++++++ `char[]`

```java
// Runtime: 0 ms, faster than 100.00% of Java online submissions for Reverse Prefix of Word.
// Memory Usage: 37.2 MB, less than 88.30% of Java online submissions for Reverse Prefix of Word.
class Solution {
    public String reversePrefix(String word, char ch) {
        int loc = word.indexOf(ch);
        if (loc == -1) return word; // not in
        char[] chr=word.toCharArray();
        for(int i=0, j=loc; i<j; i++, j--){
            char temp = chr[i];
            chr[i] = chr[j];
            chr[j] = temp;
        }
        return String.valueOf(chr);
    }
}
```


#### ++++++++++ `StringBuilder`

```java
// Runtime: 0 ms, faster than 100.00% of Java online submissions for Reverse Prefix of Word.
// Memory Usage: 37.2 MB, less than 77.86% of Java online submissions for Reverse Prefix of Word.
class Solution {
    public String reversePrefix(String word, char ch) {
        int loc = word.indexOf(ch);
        if (loc == -1) return word; // not in
        StringBuilder sb = new StringBuilder();
        sb.append(word.substring(0, loc+1));
        sb.reverse();
        sb.append(word.substring(loc+1));
        return sb.toString();
    }
}
```


---

### 557. Reverse Words in a String III (Easy)

[557. Reverse Words in a String III](https://leetcode.com/problems/reverse-words-in-a-string-iii/)

Given a string s, reverse the order of characters in each word within a sentence while still preserving whitespace and initial word order.

Example 1:

Input: s = "Let's take LeetCode contest"
Output: "s'teL ekat edoCteeL tsetnoc"
Example 2:

Input: s = "God Ding"
Output: "doG gniD"

```java
// Runtime: 3 ms, faster than 87.03% of Java online submissions for Reverse Words in a String III.
// Memory Usage: 39.5 MB, less than 74.19% of Java online submissions for Reverse Words in a String III.
class Solution {
    public String reverseWords(String s) {
        String[] str = s.split(" ");
        StringBuilder sb = new StringBuilder("");
        for(String wd : str) sb.append(" ").append(reverse(wd));
        return sb.toString().substring(1);        
    }
    public String reverse(String s) {
        StringBuilder sb = new StringBuilder(s);
        return sb.reverse().toString();
    }
}


class Solution {
    public String reverseWords(String s) {
        String[] array = s.split(" ");
        for (int i=0;i<array.length;i++) {
            String a = array[i];
            int left = 0, right = a.length()-1;
            while (left<right) {
                a = swapCharUsingCharArray(a, left, right);
                left ++;
                right --;
            }
            array[i] = a;    
        }
        return String.join(" ", array);
    }
    private String swapCharUsingCharArray(String str, int left, int right) {
        char[] chars = str.toCharArray();
        char temp = chars[left];
        chars[left] = chars[right];
        chars[right] = temp;
        return String.valueOf(chars);
    }
}
```

---


### 541. Reverse String II (Easy) `2134 6578`

[541. Reverse String II](https://leetcode.com/problems/reverse-string-ii/)

Given a string s and an integer k, reverse the first k characters for every 2k characters counting from the start of the string.

If there are fewer than k characters left, reverse all of them. If there are less than 2k but greater than or equal to k characters, then reverse the first k characters and left the other as original.

Example 1:
Input: s = "abcdefg", k = 2
Output: "bacdfeg"

```java
// Runtime: 0 ms, faster than 100.00% of Java online submissions for Reverse String II.
// Memory Usage: 38.9 MB, less than 76.53% of Java online submissions for Reverse String II.
class Solution {
    public String reverseStr(String s, int k) {
        char[] chars = s.toCharArray();
        int i=0;
        while(i<s.length()-1){
            int end = i + k - 1;
            if (end > chars.length - 1) end = chars.length - 1;
            reverse(chars, i, end);
            i = i + 2 * k;
        }
        return new String(chars);
    }

    public String reverseStr(String s, int k) {
        char[] chars = s.toCharArray();
        for (int i=0 ; i<s.length(); i += 2*k) {
            int end = i + k - 1;
            if (end > chars.length - 1) end = chars.length - 1;
            reverse(chars, i, end);
        }
        return new String(chars);
    }

    public void reverse(char[] chars, int i, int k) {
        while(i<k){
            char temp = chars[i];
            chars[i] = chars[k];
            chars[k] = temp;
            i++;
            k--;
        }
    }
}
```

---

### 942. DI String Match (Easy) `Increase l++; Decrease r--`

[942. DI String Match](https://leetcode.com/problems/di-string-match/)
A permutation perm of n + 1 integers of all the integers in the range [0, n] can be represented as a string s of length n where:

s[i] == 'I' if perm[i] < perm[i + 1], and
s[i] == 'D' if perm[i] > perm[i + 1].
Given a string s, reconstruct the permutation perm and return it. If there are multiple valid permutations perm, return any of them.

Example 1:
Input: s = "IDID"
Output: [0,4,1,3,2]

```java
// Runtime: 2 ms, faster than 95.15% of Java online submissions for DI String Match.
// Memory Usage: 40.2 MB, less than 69.63% of Java online submissions for DI String Match.
// O(n) time, O(n) space, n is length of S
class Solution {
    public int[] diStringMatch(String s) {
        int[] res = new int[s.length()+1];
        int l=0, r=s.length();
        for(int i=0; i<s.length(); i++){
            if(s.charAt(i)=='I') res[i] = l++;
            else res[i] = r--;
        }
        res[s.length()]=(s.charAt(s.length()-1)=='I')?l:r;
        return res;
    }
}

class Solution {
    public int[] diStringMatch(String s) {
        int[] res = new int[s.length()+1];
        int l=0, r=s.length();
        for(int i=0; i<s.length(); i++ ) res[i]= s.charAt(i)=='I' ? l++:r--;
        res[s.length()]=(s.charAt(s.length()-1)=='I')?l:r;
        return res;
    }
}
```

---

### 905. Sort Array By Parity (Easy)

Given an integer array nums, move all the even integers at the beginning of the array followed by all the odd integers.

Return any array that satisfies this condition.

Example 1:
Input: nums = [3,1,2,4]
Output: [2,4,3,1]
Explanation: The outputs [4,2,3,1], [2,4,1,3], and [4,2,1,3] would also be accepted.

#### ++++++++++ `new int[i] = nums[l/r]`

```java
class Solution {
    public int[] sortArrayByParity(int[] A) {
        int arr[]=new int[A.length];
        int j=0, k=A.length-1;
        for(int i=0;i<A.length;i++) {
            if(A[i]%2==0) {   
                arr[j]=A[i];
                j++;
            }
            else {   
                arr[k]=A[i];
                k--;
            }
        }
        return arr;
    }
}

// O(n)
class Solution {
    public int[] sortArrayByParity(int[] A) {
        int[] res = new int[A.length];
        int l=0,r=A.length-1;
        for(int a: A){
            if(a%2 == 0) res[l++]=a;
            else res[r--]=a;
        }
        return res;
    }
}
```

#### ++++++++++ In Place Solution Best

```java
// Runtime: 1 ms, faster than 98.86% of Java online submissions for Sort Array By Parity.
// Memory Usage: 39.7 MB, less than 81.43% of Java online submissions for Sort Array By Parity.

class Solution {
    public int[] sortArrayByParity(int[] nums) {
        int fast=0;
        for(int slow=0; slow<nums.length; slow++){
            if(nums[slow]%2==0){
                int temp = nums[slow];
                nums[slow]=nums[fast];
                nums[fast]=temp;
                fast++;
            }
        }
        return nums;
    }
}
```



---

### 1768. Merge Strings Alternately (Easy)

You are given two strings word1 and word2. Merge the strings by adding letters in alternating order, starting with word1. If a string is longer than the other, append the additional letters onto the end of the merged string.

Return the merged string.

Example 1:

Input: word1 = "abc", word2 = "pqr"
Output: "apbqcr"
Explanation: The merged string will be merged as so:
word1:  a   b   c
word2:    p   q   r
merged: a p b q c r

#### ++++++++++ `for (int i=0; i<Math.max(s1,s2); i++); `

```java
// Runtime: 0 ms, faster than 100.00% of Java online submissions for Merge Strings Alternately.
// Memory Usage: 36.8 MB, less than 99.89% of Java online submissions for Merge Strings Alternately.
class Solution {
    public String mergeAlternately(String word1, String word2) {
        StringBuilder sb = new StringBuilder();
        int s1 = word1.length(), s2 = word2.length();
        int stop = Math.max(s1,s2);
        for(int i=0; i<stop; i++){
            if(i<s1) sb.append(word1.charAt(i));
            if(i<s2) sb.append(word2.charAt(i));
        }
        return sb.toString();
    }
}
```


#### ++++++++++ substring

```java
// Runtime: 0 ms, faster than 100.00% of Java online submissions for Merge Strings Alternately.
// Memory Usage: 37.3 MB, less than 81.03% of Java online submissions for Merge Strings Alternately.
// Java O(n)class
Solution {
    public String mergeAlternately(String word1, String word2) {
        StringBuilder sb = new StringBuilder();
        int s1 = word1.length(), s2 = word2.length();
        int stop = Math.min(s1,s2);

        String bigger = stop == s1? word2:word1;

        for(int i=0; i<stop; i++){
            sb.append(word1.charAt(i));
            sb.append(word2.charAt(i));
        }

        return sb.toString()+bigger.substring(stop);
    }
}
```


---

### 977. Squares of a Sorted Array (Easy)

[977. Squares of a Sorted Array](https://leetcode.com/problems/squares-of-a-sorted-array/discuss/410331/Java-O(N)-two-pointer.-w-comments.-beats-100)
Given an integer array nums sorted in non-decreasing order, return an array of the squares of each number sorted in non-decreasing order.

Example 1:

Input: nums = [-4,-1,0,3,10]
Output: [0,1,9,16,100]
Explanation: After squaring, the array becomes [16,1,0,9,100].
After sorting, it becomes [0,1,9,16,100].


#### ++++++++++ Brute Force Approach

Squares of sorted array seems like the easiest problem

```java
// O(nlogn)
class Solution {
    public int[] sortedSquares(int[] nums) {
        for(int i = 0;i<nums.length;i++)
        {
            nums[i] *= nums[i];
        }
        Arrays.sort(nums);
        return nums;
    }
}
```

#### ++++++++++ `Math.abs(nums[l]) > Math.abs(nums[r])` Best

1. can the values in the array be negative.
2. can square of values can exceed Integer.MAX_VALUE.
3. values are in long or Integer.
4. is given array sorted.(even if the example are sorted) this helped me in google interview interviewer told me that this is nice question. (I was not asked this question but a question where sample cases where sorted )

```java
// Runtime: 1 ms, faster than 100.00% of Java online submissions for Squares of a Sorted Array.
// Memory Usage: 40.6 MB, less than 90.34% of Java online submissions for Squares of a Sorted Array.
// O(N)

class Solution {
    public int[] sortedSquares(int[] nums) {
        int[] res = new int[nums.length];
        int l=0, r=nums.length-1;
        for(int i=nums.length-1; i>=0 ; i--){
            if(Math.abs(nums[l]) > Math.abs(nums[r])) {
                res[i] = nums[l]*nums[l++];  
            }
            else {
                res[i] = nums[r]*nums[r--];  
            }
        }
        return res;
    }
}
```

---

### 821. Shortest Distance to a Character (Easy)

[821. Shortest Distance to a Character](https://leetcode.com/problems/shortest-distance-to-a-character/)
Given a string s and a character c that occurs in s, return an array of integers answer where answer.length == s.length and answer[i] is the distance from index i to the closest occurrence of character c in s.

The distance between two indices i and j is abs(i - j), where abs is the absolute value function.

Example 1:

Input: s = "loveleetcode", c = "e"
Output: [3,2,1,0,1,0,0,1,2,2,1,0]
Explanation: The character 'e' appears at indices 3, 5, 6, and 11 (0-indexed).
The closest occurrence of 'e' for index 0 is at index 3, so the distance is abs(0 - 3) = 3.
The closest occurrence of 'e' for index 1 is at index 3, so the distance is abs(1 - 3) = 2.
For index 4, there is a tie between the 'e' at index 3 and the 'e' at index 5, but the distance is still the same: abs(4 - 3) == abs(4 - 5) = 1.
The closest occurrence of 'e' for index 8 is at index 6, so the distance is abs(8 - 6) = 2.


#### ++++++++++ ``Math.min(fromLeft, fromRight)`

```java
// Time Complexity: Forward loop & Backward Loop : O(N) + O(N) ~ O(N)
// Space Complexity: Without considering answer array : O(1)
// Runtime: 1 ms, faster than 96.28% of Java online submissions for Shortest Distance to a Character.
// Memory Usage: 38.9 MB, less than 92.57% of Java online submissions for Shortest Distance to a Character.

class Solution {
    public int[] shortestToChar(String s, char c) {
        int n = s.length(), prev = n;
        int[] res= new int[n];
        // forward
        for(int i=0; i< n; i++){
            if(s.charAt(i)==c) {
                prev=0;
                res[i]=0;
            }
            else res[i] = ++prev;
        }
        // backward
        prev = n;
        for(int i=n-1; i>=0; i--){
            if(s.charAt(i)==c) prev=0;
            else res[i]=Math.min(res[i], ++prev);
        }
        return res;
    }
}
```

#### ++++++++++ `when s.char==c, j=i-1; j=i+1`

```java
// Runtime: 1 ms, faster than 96.28% of Java online submissions for Shortest Distance to a Character.
// Memory Usage: 38.8 MB, less than 97.27% of Java online submissions for Shortest Distance to a Character.
class Solution {
    public int[] shortestToChar(String s, char c) {
        int n = s.length();
        int j;
        int[] res= new int[n];
        Arrays.fill(res, n + 1);
        // forward
        for(int i=0; i< n; i++){
            if(s.charAt(i)==c) {
                res[i]=0;
                // backforward
                j = i-1;
                while(j>=0 && res[j] > i-j){
                    res[j] =i-j;
                    j--;
                }
                // forward
                j = i+1;
                while(j<n && s.charAt(j) != c){
                    res[j] =j-i;
                    j++;
                }
            }
        }
        return res;
    }
}
```

#### ++++++++++ `combine 2` BEST

```java
// Runtime: 1 ms, faster than 96.28% of Java online submissions for Shortest Distance to a Character.
// Memory Usage: 38.9 MB, less than 84.76% of Java online submissions for Shortest Distance to a Character.

class Solution {
    public int[] shortestToChar(String s, char c) {
        int n = s.length();
        int prev = n, j;
        int[] res= new int[n];
        Arrays.fill(res, n + 1);
        // forward
        for(int i=0; i< n; i++){
            if(s.charAt(i)==c) {
                res[i]=0;
                prev=0;
                // backforward
                j = i-1;
                while(j>=0 && res[j] > i-j){
                    res[j] =i-j;
                    j--;
                }
            }
            else res[i]=++prev;
        }
        return res;
    }
}
```

---


### 922. Sort Array By Parity II (Easy)

[922. Sort Array By Parity II](https://leetcode.com/problems/sort-array-by-parity-ii/)
Given an array of integers nums, half of the integers in nums are odd, and the other half are even.

Sort the array so that whenever nums[i] is odd, i is odd, and whenever nums[i] is even, i is even.

Return any answer array that satisfies this condition.

Example 1:

Input: nums = [4,2,5,7]
Output: [4,5,2,7]
Explanation: [4,7,2,5], [2,5,4,7], [2,7,4,5] would also have been accepted.



#### ++++++++++ `new res, nums[i]%2==0?; res[oddindex] oddindex++, res[evenindex] evenindex++`

```java
// Runtime: 2 ms, faster than 98.92% of Java online submissions for Sort Array By Parity II.
// Memory Usage: 41.6 MB, less than 47.17% of Java online submissions for Sort Array By Parity II.
class Solution {
    public int[] sortArrayByParityII(int[] nums) {
        int oddindex = 1, evenindex = 0;
        int[] res=new int[nums.length];
        for(int i=0;i<nums.length; i++){
            if(nums[i]%2==0){
                res[evenindex] = nums[i];
                evenindex+=2;
            }
            else {
                res[oddindex] = nums[i];
                oddindex+=2;
            }
        }
        return res;
    }

    public void swap(int[] nums, int a, int b) {
        int temp=nums[a];
        nums[a]=nums[b];
        nums[b]=temp;
    }
}
```


#### ++++++++++ `for(int i=0;i<n; i+=2) should be even, if (odd), check prev num[odd]` BEST


```java
// Runtime: 2 ms, faster than 98.92% of Java online submissions for Sort Array By Parity II.
// Memory Usage: 39.9 MB, less than 89.85% of Java online submissions for Sort Array By Parity II.

class Solution {
    public int[] sortArrayByParityII(int[] nums) {
        int oddindex = 1, n=nums.length;
        for(int i=0;i<n; i+=2){
            if(nums[i]%2!=0){
                while(nums[oddindex]%2!=0) oddindex+=2;
                swap(nums, oddindex, i);
            }
        }
        return nums;
    }
    public void swap(int[] nums, int a, int b) {
        int temp=nums[a];
        nums[a]=nums[b];
        nums[b]=temp;
    }
}
```

---
# 数组

---


## TWOSUM问题

对于 TwoSum 问题，一个难点就是给的数组无序。对于一个无序的数组，我们似乎什么技巧也没有，只能暴力穷举所有可能。

一般情况下，我们会首先把数组排序再考虑双指针技巧。TwoSum 启发我们，HashMap 或者 HashSet 也可以帮助我们处理无序数组相关的简单问题。
- 设计的核心在于权衡，利用不同的数据结构，可以得到一些针对性的加强。

```java
int[] twoSum(int[] nums, int target) {
    int left = 0, right = nums.length - 1;
    while (left < right) {
        int sum = nums[left] + nums[right];
        if (sum == target) {
            return new int[]{left, right};
        } else if (sum < target) {
            left++; // 让 sum 大一点
        } else if (sum > target) {
            right--; // 让 sum 小一点
        }
    }
    // 不存在这样两个数
    return new int[]{-1, -1};
}
```


---

### 1. Two Sum

[1. Two Sum](https://leetcode.com/problems/two-sum/)
Given an array of integers nums and an integer target, return indices of the two numbers such that they add up to target.

You may assume that each input would have exactly one solution, and you may not use the same element twice.

You can return the answer in any order.



最简单粗暴的办法当然是穷举了：
- 时间复杂度 O(N^2)，空间复杂度 O(1)。

```java
int[] twoSum(int[] nums, int target) {
    for (int i = 0; i < nums.length; i++)
        for (int j = i + 1; j < nums.length; j++)
            if (nums[j] == target - nums[i]) return new int[] { i, j };
    // 不存在这么两个数
    return new int[] {-1, -1};
}
```

通过一个哈希表减少时间复杂度：
- 时间复杂度降低到 O(N)，但是需要 O(N) 的空间复杂度


```java
// Runtime: 8 ms, faster than 45.82% of Java online submissions for Two Sum.
// Memory Usage: 43.6 MB, less than 6.09% of Java online submissions for Two Sum.

int[] twoSum(int[] nums, int target) {
    int n = nums.length;
    HashMap<Integer, Integer> index = new HashMap<>();
    // 构造一个哈希表：元素映射到相应的索引
    for (int i = 0; i < n; i++) index.put(nums[i], i);

    for (int i = 0; i < n; i++) {
        int other = target - nums[i];
        // 如果 other 存在且不是 nums[i] 本身
        if (index.containsKey(other) && index.get(other) != i) return new int[] {i, index.get(other)};
    }
    return new int[] {-1, -1};
}
```

---

### 167. Two Sum II - Input Array Is Sorted


[167. Two Sum II - Input Array Is Sorted](https://leetcode.com/problems/two-sum-ii-input-array-is-sorted/)

Given a 1-indexed array of integers numbers that is already sorted in non-decreasing order, find two numbers such that they add up to a specific target number. Let these two numbers be numbers[index1] and numbers[index2] where 1 <= index1 < index2 <= numbers.length.

Return the indices of the two numbers, index1 and index2, added by one as an integer array [index1, index2] of length 2.

The tests are generated such that there is exactly one solution. You may not use the same element twice.



```java
// Solution 1 : BinarySearch
// Time : O(nlogn)
// space : O(1)
class Solution {
    public int[] twoSum(int[] numbers, int target) {
        int n = numbers.length;
        for(int i=0;i<n-1;i++){
           int pos = Arrays.binarySearch(numbers, i+1 , n, target-numbers[i]);
           if(pos>0) return new int[]{i+1,pos+1};
        }
        return null;
    }
}

// Solution 2: HashMap
// Time : O(n)
// space : O(n)
// Runtime: 4 ms, faster than 16.01% of Java online submissions for Two Sum II - Input Array Is Sorted.
// Memory Usage: 42.3 MB, less than 7.27% of Java online submissions for Two Sum II - Input Array Is Sorted.
public int[] twoSum(int[] numbers, int target) {
    int n = numbers.length;
    HashMap<Integer, Integer> index = new HashMap<>();
    // 构造一个哈希表：元素映射到相应的索引
    for (int i = 0; i < n; i++) index.put(numbers[i], i);
    for (int i = 0; i < n; i++) {
        int other = target - numbers[i];
        // 如果 other 存在且不是 numbers[i] 本身
        if (index.containsKey(other) && index.get(other) != i) return new int[] {i+1, index.get(other)+1};
    }
    return new int[] {-1, -1};
}

// Solution 3 : Two pointers
// Time : O(n)
// space : O(1)
// Runtime: 1 ms, faster than 53.58% of Java online submissions for Two Sum II - Input Array Is Sorted.
// Memory Usage: 41.5 MB, less than 14.83% of Java online submissions for Two Sum II - Input Array Is Sorted.
public int[] twoSum(int[] numbers, int target) {
    int l = 0, r = numbers.length - 1;
    while (numbers[l] + numbers[r] != target) {
        if (numbers[l] + numbers[r] > target) r--;
        else l++;
        if (r == l) return new int[]{};
    }
    return new int[]{l + 1, r + 1};
}
```









---


## 前缀和技巧

### 303. Range Sum Query - Immutable 计算索引区间/list中指定位置的和 `preSum[i] = preSum[i - 1] + nums[i - 1];`

[303. Range Sum Query - Immutable](https://leetcode.com/problems/range-sum-query-immutable/)
- Given an integer array nums, handle multiple queries of the following type:

- Calculate the sum of the elements of nums between indices left and right inclusive where left <= right.
- Implement the NumArray class:

- `NumArray(int[] nums)` Initializes the object with the integer array nums.
- `int sumRange(int left, int right)` Returns the sum of the elements of nums between indices left and right inclusive (i.e. `nums[left] + nums[left + 1] + ... + nums[right]`).



```java
// Runtime: 63 ms, faster than 18.70% of Java online submissions for Range Sum Query - Immutable.
// Memory Usage: 41.8 MB, less than 91.48% of Java online submissions for Range Sum Query - Immutable.
class NumArray {
    private int[] nums;
    public NumArray(int[] nums) {
        this.nums = nums;
    }
    public int sumRange(int left, int right) {
        int res = 0;
        for (int i = left; i <= right; i++) res += nums[i];
        return res;
    }
}
// 可以达到效果，但是效率很差，
// 因为 sumRange 的时间复杂度是 O(N)，其中 N 代表 nums 数组的长度。
// 这道题的最优解法是使用前缀和技巧，将 sumRange 函数的时间复杂度降为 O(1)。

// 时间复杂度就是代码在最坏情况下的执行次数。
// 如果调用方输入 left = 0, right = 0，那相当于没有循环，时间复杂度是 O(1)；
// 如果调用方输入 left = 0, right = nums.length-1，for 循环相当于遍历了整个 nums 数组，时间复杂度是 O(N)，其中 N 代表 nums 数组的长度。



// solution2
// 说白了就是不要在 sumRange 里面用 for 循环
// Runtime: 7 ms, faster than 57.01% of Java online submissions for Range Sum Query - Immutable.
// Memory Usage: 43.6 MB, less than 59.62% of Java online submissions for Range Sum Query - Immutable.

class NumArray {
    private int[] preSum;   // 前缀和数组

    /* 输入一个数组，构造前缀和 */
    public NumArray(int[] nums) {
        preSum = new int[nums.length + 1];
        // 计算 nums 的累加和
        for (int i = 1; i < preSum.length; i++) {
            preSum[i] = preSum[i - 1] + nums[i - 1];
        }
    }

    /* 查询闭区间 [left, right] 的累加和 */
    public int sumRange(int left, int right) {
        return preSum[right + 1] - preSum[left];
    }
}
```

![Screen Shot 2021-10-11 at 10.18.11 PM](https://i.imgur.com/9FGiMm1.png)

- 求索引区间 `[1, 4]` 内的所有元素之和，就可以通过 `preSum[5] - preSum[1]` 得出。
- sumRange 函数仅仅需要做一次减法运算，避免for循环，最坏时间复杂度为常数 O(1)。



```java
// 存储着所有同学的分数
int[] scores;
// 试卷满分 100 分
int[] count = new int[100 + 1]
// 记录每个分数有几个同学
for (int score : scores)
    count[score]++
// 构造前缀和
for (int i = 1; i < count.length; i++)
    count[i] = count[i] + count[i-1];

// 利用 count 这个前缀和数组进行分数段查询
```

---

### 560. Subarray Sum Equals K 和为k的子数组 `if (preSum[j] == preSum[i] - k) res++;`

[560. Subarray Sum Equals K](https://leetcode.com/problems/subarray-sum-equals-k/)
- Given an array of integers nums and an integer k,
- return the total number of continuous subarrays whose sum equals to k.


```java
// 时间复杂度 O(N^2) 空间复杂度 O(N)
int subarraySum(int[] nums, int k) {
    // 构造前缀和
    int[] preSum = new int[nums.length + 1];
    preSum[0] = 0;
    for (int i = 0; i < nums.length; i++) preSum[i + 1] = preSum[i] + nums[i];

    int res = 0;
    // 穷举所有子数组
    for (int i = 1; i <= n; i++)
        for (int j = 0; j < i; j++)
            // 子数组 nums[j..i-1] 的元素和
            // if (preSum[i] - preSum[j] == k) res++;
            if (preSum[j] == preSum[i] - k) res++;
    return res;
}


// 用哈希表，在记录前缀和的同时记录该前缀和出现的次数。
// Runtime: 19 ms, faster than 53.05% of Java online submissions for Subarray Sum Equals K.
Memory Usage: 41.6 MB, less than 58.42% of Java online submissions for Subarray Sum Equals K.
int subarraySum(int[] nums, int k) {
    int n = nums.length;
    // map：前缀和 -> 该前缀和出现的次数
    HashMap<Integer, Integer> preSum = new HashMap<>();
    // base case
    preSum.put(0, 1);

    int res = 0, sum0_i = 0;

    for (int i = 0; i < n; i++) {
        sum0_i += nums[i];
        // 这是我们想找的前缀和 nums[0..j]
        int sum0_j = sum0_i - k;
        // 如果前面有这个前缀和，则直接更新答案
        if (preSum.containsKey(sum0_j)) res += preSum.get(sum0_j);
        // 把前缀和 nums[0..i] 加入并记录出现次数
        preSum.put(sum0_i, preSum.getOrDefault(sum0_i, 0) + 1);
    }
    return res;
}

```


---

### 304. Range Sum Query 2D - Immutable 二维区域和检索 `图像块之间相互减`

[youtube](https://www.youtube.com/watch?v=PwDqpOMwg6U)

![Screen Shot 2021-10-13 at 11.35.52 PM](https://i.imgur.com/f55K6B4.png)

[304. Range Sum Query 2D - Immutable](https://leetcode.com/problems/range-sum-query-2d-immutable/)
- Given a 2D matrix matrix, handle multiple queries of the following type:

- Calculate the sum of the elements of matrix inside the rectangle defined by its upper left corner (row1, col1) and lower right corner (row2, col2).
Implement the NumMatrix class:

- NumMatrix(int[][] matrix) Initializes the object with the integer matrix matrix.
- int sumRegion(int row1, int col1, int row2, int col2) Returns the sum of the elements of matrix inside the rectangle defined by its upper left corner (row1, col1) and lower right corner (row2, col2).

```java
// Runtime: 107 ms, faster than 74.69% of Java online submissions for Range Sum Query 2D - Immutable.
// Memory Usage: 66.8 MB, less than 65.71% of Java online submissions for Range Sum Query 2D - Immutable.
// O(1)
class NumMatrix {

    int[][] preSum;

    public NumMatrix(int[][] matrix) {
        int m = matrix.length, n = matrix[0].length;
        if(m==0|n==0) return;

        preSum = new int[m+1][n+1];
        for(int x=1; x<=m; x++){
            for(int y=1; y<=n; y++){
                // 计算每个矩阵 [0, 0, i, j] 的元素和
                preSum[x][y] = matrix[x-1][y-1] + preSum[x][y-1] + preSum[x-1][y] - preSum[x-1][y-1];
            }
        }
    }

    public int sumRegion(int row1, int col1, int row2, int col2) {
        return preSum[row2+1][col2+1] - preSum[row1][col2+1]- preSum[row2+1][col1] + preSum[row1][col1];
    }
}

```



---

## 差分


### 差分数组 `increment(i,j,val)->{diff[i]+=val; diff[j+1]-=val;`

差分数组的主要适用场景是频繁对原始数组的某个区间的元素进行增减。
- 输入一个数组 nums，然后又要求给区间 nums[2..6] 全部加 1，再给 nums[3..9] 全部减 3，再给 nums[0..4] 全部加 2，再给…
- 最后 nums 数组的值是什么？
- 常规的思路, for 循环给它们都加上, 时间复杂度是 O(N)，

用preSum，修改的区域改变preSum，从preSum推原list

```java
// 差分数组工具类
class Difference {

    // 差分数组
    private int[] diff;

    /* 输入一个初始数组，区间操作将在这个数组上进行 */
    public Difference(int[] nums) {
        assert nums.length > 0;
        // 根据初始数组构造差分数组
        diff = new int[nums.length];
        diff[0] = nums[0];
        for (int i = 1; i < nums.length; i++) diff[i] = nums[i] - nums[i - 1];
    }

    /* 给闭区间 [i,j] 增加 val（可以是负数）*/
    public void increment(int i, int j, int val) {
        diff[i] += val;
        if (j + 1 < diff.length) diff[j + 1] -= val;
    }

    /* 返回结果数组 */
    public int[] result() {
        int[] res = new int[diff.length];
        // 根据差分数组构造结果数组
        res[0] = diff[0];
        for (int i = 1; i < diff.length; i++) res[i] = res[i - 1] + diff[i];
        return res;
    }
}
```

---



### 370. 区间加法（中等）`Difference df = new Difference(nums); df.increment(i, j, val);`



```java
int[] getModifiedArray(int length, int[][] updates) {
    // nums 初始化为全 0
    int[] nums = new int[length];
    // 构造差分解法
    Difference df = new Difference(nums);

    for (int[] update : updates) {
        int i = update[0];
        int j = update[1];
        int val = update[2];
        df.increment(i, j, val);
    }

    return df.result();
}
```


---

### 1109. Corporate Flight Bookings 航班预订统计

[1109. Corporate Flight Bookings](https://leetcode.com/problems/corporate-flight-bookings/)
- There are n flights that are labeled from 1 to n.

- You are given an array of flight bookings bookings, where bookings[i] = [firsti, lasti, seatsi] represents a booking for flights firsti through lasti (inclusive) with seatsi seats reserved for each flight in the range.

- Return an array answer of length n, where answer[i] is the total number of seats reserved for flight i.

```
Example 1:
Input: bookings = [[1,2,10],[2,3,20],[2,5,25]], n = 5
Output: [10,55,45,25,25]
Explanation:
Flight labels:        1   2   3   4   5
Booking 1 reserved:  10  10
Booking 2 reserved:      20  20
Booking 3 reserved:      25  25  25  25
Total seats:         10  55  45  25  25
Hence, answer = [10,55,45,25,25]
```


```java
// Runtime: 5 ms, faster than 43.51% of Java online submissions for Corporate Flight Bookings.
// Memory Usage: 54.6 MB, less than 54.64% of Java online submissions for Corporate Flight Bookings.
class Solution {
    int[] corpFlightBookings(int[][] bookings, int n) {
        // nums 初始化为全 0
        int[] nums = new int[n];
        // 构造差分解法
        Difference df = new Difference(nums);
        for (int[] booking : bookings) {
            // 注意转成数组索引要减一哦
            int i = booking[0] - 1;
            int j = booking[1] - 1;
            int val = booking[2];
            // 对区间 nums[i..j] 增加 val
            df.increment(i, j, val);
        }
        // 返回最终的结果数组
        return df.result();
    }
}

class Difference {

    private int[] diff;

    public Difference(int[] nums){
        assert nums.length > 0;
        diff = new int[nums.length];
        diff[0] = nums[0];
        for(int i=1;i<nums.length;i++) diff[i] = nums[i] - nums[i-1];
    }

    public void increment(int x, int y, int k){
        diff[x] += k;
        if(y+1<diff.length) diff[y+1] -=k;
    }

    public int[] result(){
        int[] res = new int[diff.length];
        res[0] = diff[0];
        for(int i=1; i<diff.length; i++) res[i] = res[i-1] + diff[i];
        return res;
    }
}
```

---


###  1094 题「拼车」


```java

// Runtime: 3 ms, faster than 70.75% of Java online submissions for Car Pooling.
// Memory Usage: 38.9 MB, less than 52.80% of Java online submissions for Car Pooling.

boolean carPooling(int[][] trips, int capacity) {
    // 最多有 1000 个车站
    int[] nums = new int[1001];
    // 构造差分解法
    Difference df = new Difference(nums);

    for (int[] trip : trips) {
        // 乘客数量
        int val = trip[0];
        // 第 trip[1] 站乘客上车
        int i = trip[1];
        // 第 trip[2] 站乘客已经下车，
        // 即乘客在车上的区间是 [trip[1], trip[2] - 1]
        int j = trip[2] - 1;
        // 进行区间操作
        df.increment(i, j, val);
    }
    int[] res = df.result();
    // 客车自始至终都不应该超载
    for (int i = 0; i < res.length; i++) {
        if (capacity < res[i]) return false;
    }
    return true;
}


class Difference{
    private int[] diff;

    public Difference(int[] nums){
        assert nums.length > 0;
        if(nums.length==0) return;
        diff = new int[nums.length];
        diff[0] = nums[0];
        for(int i=1;i<nums.length;i++) diff[i] = nums[i] - nums[i-1];
    }

    public void increment(int x, int y, int k){
        diff[x] +=k;
        if(y+1<diff.length) diff[y+1] -=k;
    }

    public int[] result(){
        int[] res = new int[diff.length];
        res[0] = diff[0];
        for(int i=1;i<diff.length;i++) res[i] = diff[i] + res[i-1];
        return res;
    }
}
```


---




# LinkedList

---

## 单链表的六大解题套路


---

### 合并两个有序链表 Merge 2 Sorted Lists

>  21 题合并两个有序链表

两个有序链表，合并成一个新的有序链表

Solution:「拉拉链」，l1, l2 类似于拉链两侧的锯齿，指针 p 就好像拉链的拉索，将两个有序链表合并。
- 链表的算法题中是很常见的「虚拟头结点」技巧，`dummy` 节点。
  - 如果不使用 dummy 虚拟节点，代码会复杂很多，
  - 而有了 dummy 节点这个占位符，可以避免处理空指针的情况，降低代码的复杂性。
  - 比如说链表总共有 5 个节点，题目就让你删除倒数第 5 个节点，也就是第一个节点，那按照算法逻辑，应该首先找到倒数第 6 个节点。但第一个节点前面已经没有节点了，这就会出错。
  - 但有了我们虚拟节点 dummy 的存在，就避免了这个问题，能够对这种情况进行正确的删除。



```java
// Definition for singly-linked list.
public class ListNode {
    int val;
    ListNode next;
    ListNode(){}
    ListNode(int val){this.val = val;}
    ListNode(int val, ListNode next) {this.val = val; this.next = next;}
}

// O(n)
// Runtime: 0 ms, faster than 100.00% of Java online submissions for Merge Two Sorted Lists.
// Memory Usage: 38.4 MB, less than 75.55% of Java online submissions for Merge Two Sorted Lists.
ListNode mergeTwoLists(ListNode l1, ListNode l2) {
    if(l1==null) return l2;
    if(l2==null) return l1;
    ListNode dummy = new ListNode(-1);
    ListNode p = dummy;
    while(l1!=null && l2!=null){
        if(l1.val<l2.val){
            p.next = l1;
            l1=l1.next;
        }else{
            p.next = l2;
            l2=l2.next;
        }
        p = p.next;
    }
    if(l1==null) p.next = l2;
    if(l2==null) p.next = l1;
    return dummy.next;
}

// recursion
// won't use recursion for a O(n) solution.
// This solution will result into Stack overflow error with some-thousand elements input.
// It's nice but impractical.
// Runtime: 0 ms, faster than 100.00% of Java online submissions for Merge Two Sorted Lists.
// Memory Usage: 38.3 MB, less than 75.55% of Java online submissions for Merge Two Sorted Lists.
ListNode mergeTwoLists(ListNode l1, ListNode l2){
    if(l1==null) return l2;
    if(l2==null) return l1;
    if(l1.val < l2.val){
        l1.next = mergeTwoLists(l1.next, l2)
        return l1
    }else{
        l2.next = mergeTwoLists(l2.next, l1)
        return l2
    }
}
```


```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class Solution:
    # iteratively
    def mergeTwoLists(self, l1, l2):
        # while
        dummy = ListNode(0)
        p = dummy
        while l1 and l2:
            if l1.val < l2.val:
                p.next = l1
                l1 = l1.next
            else:
                p.next = l2
                l2 = l2.next
            p = p.next
        p.next = l1 or l2
        return dummy.next

    # recursively
    def mergeTwoLists(self, l1, l2):
        if not l1 or not l2:
             return l1 or l2
        if l1.val < l2.val:
            l1.next = self.mergeTwoLists(l1.next, l2)
            return l1
        else:
            l2.next = self.mergeTwoLists(l1, l2.next)
            return l2

    # recursively
    def mergeTwoLists(self, a, b):
        if a and b:
            if a.val > b.val:
                a, b = b, a
            a.next = self.mergeTwoLists(a.next, b)
        return a or b

    # in-place, iteratively
    def mergeTwoLists(self, l1, l2):
        if None in (l1, l2):
            return l1 or l2
        dummy = cur = ListNode(0)
        dummy.next = l1
        while l1 and l2:
            if l1.val < l2.val:
                l1 = l1.next
            else:
                nxt = cur.next
                cur.next = l2
                tmp = l2.next
                l2.next = nxt
                l2 = tmp
            cur = cur.next
        cur.next = l1 or l2
        return dummy.next
```



---

### 23. Merge k Sorted Lists 合并 k 个有序链表 Merge k Sorted Lists

[23. Merge k Sorted Lists]

合并 k 个有序链表的逻辑类似合并两个有序链表

point: 如何快速得到 k 个节点中的最小节点，接到结果链表上？
- 用到 优先级队列（二叉堆） 这种数据结构，把链表节点放入一个最小堆，就可以每次获得 k 个节点中的最小节点：

时间复杂度:
- 优先队列 pq 中的元素个数最多是 k，
- 所以一次 poll 或者 add 方法的时间复杂度是 O(logk)；
- 所有的链表节点都会被加入和弹出 pq，所以算法整体的时间复杂度是 O(Nlogk)，
- 其中 k 是链表的条数，N 是这些链表的节点总数。



1. Brute-Force
   1. It is okay if N is not too large.
   2. Traverse all the linked lists and collect the values of the nodes into an array. - O(N)
   3. Sort the array. - O(Nlog{N})
   4. Traverse the array and make the linked list. - O(N)
   5. Time: O(Nlog{N}) where N is the total number of nodes.
   6. Space: O(N) since we need an array and a new linked list.


2. Compare One-By-One

```java
public ListNode mergeKLists(ListNode[] lists) {
  if (lists.length == 0) return null;
  ListNode dummy = new ListNode(-1);
  ListNode prev = dummy;

  while (true) {
    ListNode minNode = null;
    int minIdx = -1;

    // Iterate over lists
    for (int i = 0; i < lists.length; ++i) {
      ListNode currList = lists[i];
      if (currList == null) continue;
      if (minNode == null || currList.val < minNode.val) {
        minNode = currList;
        minIdx = i;
      }
    }
    // check if finished
    if (minNode == null) break;

    // link
    prev.next = minNode;
    prev = prev.next;

    // delete
    lists[minIdx] = minNode.next; // may be null
  }
  return dummy.next;
}
```


3. Compare One-By-One (minPQ)


```java
ListNode mergeKLists(ListNode[] lists) {
    if (lists.length == 0) return null;
    // 虚拟头结点
    ListNode dummy = new ListNode(-1);
    ListNode p = dummy;
    // 优先级队列，最小堆
    PriorityQueue<ListNode> pq = new PriorityQueue<>(
        lists.length, (a, b)->(a.val - b.val)
    );
    // 将 k 个链表的头结点加入最小堆
    for (ListNode head : lists) {
        if (head != null)
            pq.add(head);
    }
    while (!pq.isEmpty()) {
        // 获取最小节点，接到结果链表中
        ListNode node = pq.poll();
        p.next = node;
        if (node.next != null) {
            pq.add(node.next);
        }
        // p 指针不断前进
        p = p.next;
    }
    return dummy.next;
}
```

时间复杂度
- 优先队列 pq 中的元素个数最多是 k，
- 所以一次 poll 或者 add 方法的时间复杂度是 `O(logk)`；
- 所有的链表节点都会被加入和弹出 pq，所以算法整体的时间复杂度是 `O(Nlogk)`，其中 k 是链表的条数，N 是这些链表的节点总数。


---



## 递归反转链表

---

### 206. Reverse Linked List 递归反转整个链表 `递归+pointer`

[206. Reverse Linked List](https://leetcode.com/problems/reverse-linked-list/)
- Given the head of a singly linked list, reverse the list, and return the reversed list.
- Input: head = [1,2,3,4,5]
- Output: [5,4,3,2,1]


#### ++++++++++ 递归

```java
// recursion
// Runtime: 0 ms, faster than 100.00% of Java online submissions for Reverse Linked List.
// Memory Usage: 39.3 MB, less than 38.00% of Java online submissions for Reverse Linked List.
ListNode reverseList(ListNode head) {
    if (head==null || head.next == null) return head;
    ListNode last = reverseList(head.next);
    head.next.next = head;
    head.next = null;
    return last;
}
```

#### ++++++++++ 2 pointer

```java
// Runtime: 0 ms, faster than 100.00% of Java online submissions for Reverse Linked List.
// Memory Usage: 39 MB, less than 51.90% of Java online submissions for Reverse Linked List.
ListNode reverseList(ListNode a) {
    ListNode pre, cur, nxt;
    pre = null; cur = a; nxt = a;
    while (cur != null) {
        nxt = cur.next;
        // 逐个结点反转
        cur.next = pre;
        // 更新指针位置
        pre = cur;
        cur = nxt;
    }
    // 返回反转后的头结点
    return pre;
}
```



---


### 反转链表前 N 个节点

具体的区别：
1. base case 变为 n == 1，反转一个元素，就是它本身，同时要记录后驱节点。
2. 刚才我们直接把 head.next 设置为 null，因为整个链表反转后原来的 head 变成了整个链表的最后一个节点。
   1. 但现在 head 节点在递归反转之后不一定是最后一个节点了，所以要记录后驱 successor（第 n + 1 个节点），反转之后将 head 连接上。



```java
ListNode successor = null; // 后驱节点

// 反转以 head 为起点的 n 个节点，返回新的头结点
ListNode reverseN(ListNode head, int n) {
    if (n == 1) {
        // 记录第 n + 1 个节点
        successor = head.next;
        return head;
    }
    // 以 head.next 为起点，需要反转前 n - 1 个节点
    ListNode last = reverseN(head.next, n - 1);
    head.next.next = head;
    // 让反转之后的 head 节点和后面的节点连起来
    head.next = successor;
    return last;
}
```

---

### 92. Reverse Linked List II 反转链表的一部分

[92. Reverse Linked List II](https://leetcode.com/problems/reverse-linked-list-ii/)
- Given the head of a singly linked list and two integers left and right where left <= right, reverse the nodes of the list from position left to position right, and return the reversed list.

- Input: head = [1,2,3,4,5], left = 2, right = 4
- Output: [1,4,3,2,5]


#### ++++++++++ iterative

```java
// Runtime: 0 ms, faster than 100.00% of Java online submissions for Reverse Linked List II.
// Memory Usage: 36.6 MB, less than 58.03% of Java online submissions for Reverse Linked List II.
// O(1) space
// O(n) Solution
class Solution {
    public ListNode reverseBetween(ListNode head, int left, int right) {
        if(head==null || left==right) return head;
        ListNode dummy = new ListNode(0,head);
        ListNode prev = dummy, curr = dummy.next;  
        int i=1;
        while(i<left){
            prev = curr;
            curr = curr.next;
            i++;
        }
        // flow of execution in each iteration (for the 2nd input):
        // 1->2->3->4->5->6->7 |  
        // 1->2->4->3->5->6->7 |
        // 1->2->5->4->3->6->7 |
        // 1->2->6->5->4->3->7 |
        // 1->2->7->6->5->4->3
        ListNode node = prev;
        while(i<=right){
            ListNode next = curr.next;
            curr.next = prev;
            prev = curr;
            curr = next;
            i++;
        }
        node.next.next = curr;
        node.next = prev;
        return dummy.next;
    }
}
```


#### ++++++++++ recursive

```java
// Runtime: 0 ms, faster than 100.00% of Java online submissions for Reverse Linked List II.
// Memory Usage: 36.6 MB, less than 75.28% of Java online submissions for Reverse Linked List II.
ListNode reverseBetween(ListNode head, int m, int n) {
    // base case
    if (m == 1) return reverseN(head, n);
    // 前进到反转的起点触发 base case
    head.next = reverseBetween(head.next, m - 1, n - 1);
    return head;
}

// 反转以 head 为起点的 n 个节点，返回新的头结点
ListNode reverseN(ListNode head, int n){
    ListNode successor = null; // 后驱节点
    if (n == 1) {
        // 记录第 n + 1 个节点
        successor = head.next;
        return head;
    }
    // 以 head.next 为起点，需要反转前 n - 1 个节点
    ListNode last = reverseN(head.next, n - 1);
    head.next.next = head;
    // 让反转之后的 head 节点和后面的节点连起来
    head.next = successor;
    return last;
}
```


---


### 25. Reverse Nodes in k-Group K个一组反转链表

[25. Reverse Nodes in k-Group]()
- Given a linked list, reverse the nodes of a linked list k at a time and return its modified list.
- k is a positive integer and is less than or equal to the length of the linked list. If the number of nodes is not a multiple of k then left-out nodes, in the end, should remain as it is.
- You may not alter the values in the list's nodes, only nodes themselves may be changed.
- Input:
- head = [1,2,3,4,5], k = 2
- Output: [2,1,4,3,5]


#### ++++++++++ `a,b reverse(), a.next=reverseK(b,k)`


```java
// Runtime: 0 ms, faster than 100.00% of Java online submissions for Reverse Nodes in k-Group.
// Memory Usage: 39.4 MB, less than 60.83% of Java online submissions for Reverse Nodes in k-Group.

ListNode reverseKGroup(ListNode head, int k) {
    if (head == null) return null;
    // 区间 [a, b) 包含 k 个待反转元素
    // 1,2,3,4,5,6
    ListNode a= head, b= head;
    for (int i = 0; i < k; i++) {
        // 不足 k 个，不需要反转，base case
        if (b == null) return head;
        b = b.next;
    }
    // 3,2,1,   4,5,6
    // 反转前 k 个元素
    ListNode newHead = reverse(a, b);
    // 递归反转后续链表并连接起来
    a.next = reverseKGroup(b, k);
    return newHead;
}

/** 反转区间 [a, b) 的元素，注意是左闭右开 */
ListNode reverse(ListNode a, ListNode b) {
    ListNode pre, cur, nxt;
    pre = null; cur = a; nxt = a;
    // while 终止的条件改一下就行了
    while (cur != b) {
        nxt = cur.next;
        cur.next = pre;
        pre = cur;
        cur = nxt;
    }
    // 返回反转后的头结点
    return pre;
}
```

--

### 143. Reorder List (Medium)

You are given the head of a singly linked-list. The list can be represented as:

L0 → L1 → … → Ln - 1 → Ln
Reorder the list to be on the following form:

L0 → Ln → L1 → Ln - 1 → L2 → Ln - 2 → …
You may not modify the values in the list's nodes. Only nodes themselves may be changed.

#### ++++++++++ `Two pointer, find middle, reverse(), combine(n1,n2)`

```java
// Runtime: 1 ms, faster than 99.86% of Java online submissions for Reorder List.
// Memory Usage: 42 MB, less than 45.07% of Java online submissions for Reorder List.
class Solution {
    public void reorderList(ListNode head) {
        //Find the middle of the list
        ListNode fast=head, slow=head;
        while(fast.next!=null && fast.next.next!=null){
            slow=slow.next;
            fast=fast.next.next;
        }  
        // reverse
        ListNode second = reverse(slow.next);
        slow.next=null;  
        // merge
        combine(head, second);
    }

    public ListNode reverse(ListNode head) {
        ListNode pre=null, cur=head, nxt=head;
        while(cur!=null){
            nxt = cur.next;
            cur.next=pre;
            pre=cur;
            cur=nxt;
        }
        return pre;
    }

    public void combine(ListNode n1, ListNode n2) {   
        while(n2!=null){
            ListNode f_temp = n1.next;
            ListNode s_temp = n2.next;
            n1.next = n2;
            n2.next = f_temp;
            n1 = n2.next;
            n2 = s_temp;
        }
    }
}
```

#### ++++++++++ `2 pointer. list.add(ListNode), reorder list`

```java
// Runtime: 2 ms, faster than 51.01% of Java online submissions for Reorder List.
// Memory Usage: 41.5 MB, less than 85.84% of Java online submissions for Reorder List.

class Solution {
    public void reorderList(ListNode head) {
        if(head == null) return;
        ArrayList<ListNode> list = new ArrayList<>();
        ListNode dummy = head;
        while(dummy != null){
            list.add(dummy);
            dummy = dummy.next;
        }
        int i = 1, left = 1, right = list.size() - 1;
        dummy = head;
        while(i < list.size()){
            if(i % 2 == 0) dummy.next = list.get(left++);
            else dummy.next = list.get(right--);
            dummy = dummy.next;
            i++;        
        }
        dummy.next = null;
    }
}
```

---


### 1721. Swapping Nodes in a Linked List (Medium)

[1721. Swapping Nodes in a Linked List](https://leetcode.com/problems/swapping-nodes-in-a-linked-list/)

You are given the head of a linked list, and an integer k.

Return the head of the linked list after swapping the values of the kth node from the beginning and the kth node from the end (the list is 1-indexed).

Example 1:
Input: head = [1,2,3,4,5], k = 2
Output: [1,4,3,2,5]


```java
// Runtime: 2 ms, faster than 100.00% of Java online submissions for Swapping Nodes in a Linked List.
// Memory Usage: 53.9 MB, less than 96.45% of Java online submissions for Swapping Nodes in a Linked List.
// traverse the list only once, time complexity is O(n)
// store only 4 pointers for every list; the space complexity is constant: O(1)
public ListNode swapNodes(ListNode head, int k) {
    ListNode fast=head, slow=head;
    for(int i=0; i<k-1; i++) fast=fast.next;
    ListNode first=fast;
    while(fast.next!=null) {
        slow=slow.next;
        fast=fast.next;
    }
    int val = first.val;
    first.val = slow.val;
    slow.val = val;
    return head;
}
```


---

### 24. Swap Nodes in Pairs (Medium)

Given a linked list, swap every two adjacent nodes and return its head. You must solve the problem without modifying the values in the list's nodes (i.e., only nodes themselves may be changed.)

Example 1:
Input: head = [1,2,3,4]
Output: [2,1,4,3]


#### ++++++++++ `2 pointer and swap`

```java
class Solution {
    public ListNode swapPairs(ListNode head) {
        if (head == null || head.next == null) return head;
        ListNode cur = head;
        ListNode newHead = head.next;
        while (cur != null && cur.next != null) {
            ListNode tmp = cur;
            cur = cur.next;

            tmp.next = cur.next;
            cur.next = tmp;

            cur = tmp.next;
            if (cur != null && cur.next != null) tmp.next = cur.next;
        }
        return newHead;
    }
}
```


#### ++++++++++ `recursive`

```java
// Runtime: 0 ms, faster than 100.00% of Java online submissions for Swap Nodes in Pairs.
// Memory Usage: 36.2 MB, less than 97.27% of Java online submissions for Swap Nodes in Pairs.
class Solution {
    public ListNode swapPairs(ListNode head) {
        if (head == null || head.next == null) return head;
        ListNode p1=head, p2=head.next, p3=p2.next;
        p2.next=p1;
        p1.next=p3;
        if(p3!=null) p1.next=swapPairs(p3);
        return p2;
    }
}
```



---

### example

---


#### 870 题「优势洗牌」

[870. Advantage Shuffle](https://leetcode.com/problems/advantage-shuffle/)

- You are given two integer arrays nums1 and nums2 both of the same length.
- The advantage of nums1 with respect to nums2 is the number of indices i for which `nums1[i] > nums2[i]`.
- Return any permutation of nums1 that maximizes its advantage with respect to nums2.

Example 1:
Input: nums1 = [2,7,11,15], nums2 = [1,10,4,11]
Output: [2,11,7,15]

```java
// Runtime: 67 ms, faster than 74.91% of Java online submissions for Advantage Shuffle.
// Memory Usage: 59.7 MB, less than 43.82% of Java online submissions for Advantage Shuffle.

int[] advantageCount(int[] nums1, int[] nums2) {
    int n = nums1.length;

    // 给 nums2 降序排序
    PriorityQueue<int[]> maxpq = new PriorityQueue<>(
        (int[] pair1, int[] pair2) -> {return pair2[1] - pair1[1];}
    );
    for (int i = 0; i < n; i++) maxpq.offer(new int[]{i, nums2[i]});

    // 给 nums1 升序排序
    Arrays.sort(nums1);

    // nums1[left] 是最小值，nums1[right] 是最大值
    int left = 0, right = n - 1;
    int[] res = new int[n];

    while (!maxpq.isEmpty()) {
        int[] pair = maxpq.poll();
        // maxval 是 nums2 中的最大值，i 是对应索引
        int i = pair[0], maxval = pair[1];
        // 如果 nums1[right] 能胜过 maxval，那就自己上
        if (maxval < nums1[right]) {
            res[i] = nums1[right];
            right--;
        // 否则用最小值混一下，养精蓄锐
        } else {
            res[i] = nums1[left];
            left++;
        }
    }
    return res;
}
```





---

## 左右指针

只要数组有序，就应该想到双指针技巧

---

### 二分查找

最简单的二分算法，旨在突出它的双指针特性：
- 分析二分查找的一个技巧是：不要出现 else，而是把所有情况用 else if 写清楚，这样可以清楚地展现所有细节。
- left + (right - left) / 2 就和 (left + right) / 2 的结果相同，但是有效防止了 left 和 right 太大直接相加导致溢出。



```java
int binarySearch(int[] nums, int target) {
    int left = 0;
    int right = nums.length - 1;
    while(left <= right) {
        // int mid = (right + left) / 2;
        int mid = left + (right - left) / 2;
        if(nums[mid] == target) return mid;
        else if (nums[mid] < target) left = mid + 1;
        else if (nums[mid] > target) right = mid - 1;
    }
    return -1;
}
```

---

### 在有序数组中搜索指定元素

---

#### 704. Binary Search 寻找一个数（基本的二分搜索）

- 初始化 right 的赋值是 nums.length - 1，最后一个元素的索引，而不是 nums.length。
- `nums.length - 1` 两端都闭区间 [left, right]
  - while(left <= right) 的终止条件是 left == right + 1，
  - [right + 1, right]，
  - 或者带个具体的数字进去 [3, 2]，可见这时候区间为空，因为没有数字既大于等于 3 又小于等于 2 的吧。所以这时候 while 循环终止是正确的，直接返回 -1 即可。


- `nums.length` 左闭右开区间 [left, right)
  - while(left < right) 的终止条件是 left == right [right, right]，
  - 或者带个具体的数字进去 [2, 2]，这时候区间非空，还有一个数 2，但此时 while 循环终止了。也就是说这区间 [2, 2] 被漏掉了，索引 2 没有被搜索，如果这时候直接返回 -1 就是错误的。


[704. Binary Search](https://leetcode.com/problems/binary-search/)

Given an array of integers nums which is sorted in ascending order, and an integer target, write a function to search target in nums. If target exists, then return its index. Otherwise, return -1.

You must write an algorithm with O(log n) runtime complexity.

Example 1:

Input: nums = [-1,0,3,5,9,12], target = 9
Output: 4


```java
// Runtime: 0 ms, faster than 100.00% of Java online submissions for Binary Search.
// Memory Usage: 51.7 MB, less than 16.03% of Java online submissions for Binary Search.

public int search(int[] nums, int target) {
    if (nums.length == 0) return -1;
    int left=0, right=nums.length-1;
    while(left<=right){
        int mid = left + (right-left)/2;
        if(nums[mid]==target) return mid;
        else if(nums[mid]<target) left=mid+1;
        else if(nums[mid]>target) right=mid-1;
    }
    return -1;
}
```

- 这个算法存在局限性。
- 比如说给你有序数组 nums = [1,2,2,2,3]，target 为 2，此算法返回的索引是 2，没错。但是如果我想得到 target 的左侧边界，即索引 1，或者我想得到 target 的右侧边界，即索引 3，这样的话此算法是无法处理的。

---

#### 寻找左侧边界的二分搜索



```java
int left_bound(int[] nums, int target) {
    if (nums.length == 0) return -1;
    int left = 0;
    int right = nums.length; // 注意

    while (left < right) { // 注意
        int mid = left + (right - left) / 2;
        if (nums[mid] == target) right = mid;
        else if (nums[mid] < target) left = mid + 1;
        else if (nums[mid] > target) right = mid; // 注意
        }
    }
    return left;
}

int left_bound(int[] nums, int target) {
    int left = 0, right = nums.length - 1;
    // 搜索区间为 [left, right]
    while (left <= right) {
        int mid = left + (right - left) / 2;
        // 搜索区间变为 [mid+1, right]
        if (nums[mid] < target) left = mid + 1;
        // 搜索区间变为 [left, mid-1]
        else if (nums[mid] > target) right = mid - 1;
        // 收缩右侧边界
        else if (nums[mid] == target) right = mid - 1;
    }
    // 检查出界情况
    if (left >= nums.length || nums[left] != target) return -1;
    return left;
}
```

---


##### 278. First Bad Version


[278. First Bad Version](https://leetcode.com/problems/first-bad-version/)

You are a product manager and currently leading a team to develop a new product. Unfortunately, the latest version of your product fails the quality check. Since each version is developed based on the previous version, all the versions after a bad version are also bad.

Suppose you have n versions [1, 2, ..., n] and you want to find out the first bad one, which causes all the following ones to be bad.

You are given an API bool isBadVersion(version) which returns whether version is bad. Implement a function to find the first bad version. You should minimize the number of calls to the API.

Example 1:

Input: n = 5, bad = 4
Output: 4
Explanation:
call isBadVersion(3) -> false
call isBadVersion(5) -> true
call isBadVersion(4) -> true
Then 4 is the first bad version.


```java
// Runtime: 13 ms, faster than 49.29% of Java online submissions for First Bad Version.
// Memory Usage: 36.1 MB, less than 29.13% of Java online submissions for First Bad Version.
public int firstBadVersion(int n) {
    if (isBadVersion(1)) return 1;
    int left=0, right=n;
    while(left<right){
        int mid = left+(right-left)/2;
        if(isBadVersion(mid)) right=mid;
        else left=mid+1;
    }
    return left;
}
```



---



#### 寻找右侧边界的二分查找

```java
int left_bound(int[] nums, int target) {
    int left = 0, right = nums.length - 1;
    // 搜索区间为 [left, right]
    while (left <= right) {
        int mid = left + (right - left) / 2;
        // 搜索区间变为 [mid+1, right]
        if (nums[mid] < target) right = mid + 1;
        // 搜索区间变为 [left, mid-1]
        else if (nums[mid] > target) right = mid - 1;
        // 收缩右侧边界
        else if (nums[mid] == target) right = mid + 1;
    }
    // 检查出界情况
    if (right<0 || nums[right] != target) return -1;
    return right;
}
```

---


#### 34. Find First and Last Position of Element in Sorted Array 寻找左右边界的二分搜索

[34. Find First and Last Position of Element in Sorted Array]

Given an array of integers nums sorted in non-decreasing order, find the starting and ending position of a given target value.

If target is not found in the array, return [-1, -1].

You must write an algorithm with O(log n) runtime complexity.



Example 1:

Input: nums = [5,7,7,8,8,10], target = 8
Output: [3,4]



```java
// Runtime: 0 ms, faster than 100.00% of Java online submissions for Find First and Last Position of Element in Sorted Array.
// Memory Usage: 44.4 MB, less than 6.19% of Java online submissions for Find First and Last Position of Element in Sorted Array.

int binary_search(int[] nums, int target) {
    int left = 0, right = nums.length - 1;
    while(left <= right) {
        int mid = left + (right - left) / 2;
        if (nums[mid] < target) left = mid + 1;
        else if (nums[mid] > target) right = mid - 1;
        // 直接返回
        else if(nums[mid] == target) return mid;
    }
    // 直接返回
    return -1;
}

int left_bound(int[] nums, int target) {
    int left = 0, right = nums.length - 1;
    while (left <= right) {
        int mid = left + (right - left) / 2;
        if (nums[mid] < target) left = mid + 1;
        else if (nums[mid] > target) right = mid - 1;
        // 别返回，锁定左侧边界
        else if (nums[mid] == target) right = mid - 1;
    }
    // 最后要检查 left 越界的情况
    if (left >= nums.length || nums[left] != target) return -1;
    return left;
}

int right_bound(int[] nums, int target) {
    int left = 0, right = nums.length - 1;
    while (left <= right) {
        int mid = left + (right - left) / 2;
        if (nums[mid] < target) left = mid + 1;
        else if (nums[mid] > target) right = mid - 1;
        // 别返回，锁定右侧边界
        else if (nums[mid] == target) left = mid + 1;
    }
    // 最后要检查 right 越界的情况
    if (right < 0 || nums[right] != target)
        return -1;
    return right;
}
```

---

#### 二分搜索算法运用

二分搜索的原型就是在「有序数组」中搜索一个元素target，返回该元素对应的索引。

- 如果该元素不存在，那可以返回一个什么特殊值，这种细节问题只要微调算法实现就可实现。

- 还有一个重要的问题，如果「有序数组」中存在多个target元素，那么这些元素肯定挨在一起，这里就涉及到算法应该返回最左侧的那个target元素的索引还是最右侧的那个target元素的索引，「搜索左侧边界」和「搜索右侧边界」


从题目中抽象出一个自变量 x，一个关于 x 的函数 f(x)，以及一个目标值 target。

- 同时，x, f(x), target 还要满足以下条件：

- 1、f(x) 必须是在 x 上的单调函数（单调增单调减都可以）。

- 2、题目是让你计算满足约束条件 f(x) == target 时的 x 的值。

```java
int f(int x, int[] nums) {
    return nums[x];
}

int left_bound(int[] nums, int target) {
    if (nums.length == 0) return -1;
    int left = 0, right = nums.length;

    while (left < right) {
        int mid = left + (right - left) / 2;
        // 当找到 target 时，收缩右侧边界
        if (f(mid, nums) == target) right = mid;
        else if (f(mid, nums) < target) left = mid + 1;
        else if (f(mid, nums) > target) right = mid;
    }
    return left;
}


// 主函数，在 f(x) == target 的约束下求 x 的最值
int solution(int[] nums, int target) {
    if (nums.length == 0) return -1;
    // 问自己：自变量 x 的最小值是多少？
    int left = ...;
    // 问自己：自变量 x 的最大值是多少？
    int right = ... + 1;

    while (left < right) {
        int mid = left + (right - left) / 2;
        if (f(mid) == target) {
            // 问自己：题目是求左边界还是右边界？
            // ...
        } else if (f(mid) < target) {
            // 问自己：怎么让 f(x) 大一点？
            // ...
        } else if (f(mid) > target) {
            // 问自己：怎么让 f(x) 小一点？
            // ...
        }
    }
    return left;
}
```

---

#### example

##### 875. Koko Eating Bananas

[875. Koko Eating Bananas](https://leetcode.com/problems/koko-eating-bananas/)

Koko loves to eat bananas. There are n piles of bananas, the ith pile has piles[i] bananas. The guards have gone and will come back in h hours.

Koko can decide her bananas-per-hour eating speed of k. Each hour, she chooses some pile of bananas and eats k bananas from that pile. If the pile has less than k bananas, she eats all of them instead and will not eat any more bananas during this hour.

Koko likes to eat slowly but still wants to finish eating all the bananas before the guards return.

Return the minimum integer k such that she can eat all the bananas within h hours.


Example 1:

Input: piles = [3,6,7,11], h = 8
Output: 4

```java
// Runtime: 21 ms, faster than 37.05% of Java online submissions for Koko Eating Bananas.
// Memory Usage: 52.4 MB, less than 7.41% of Java online submissions for Koko Eating Bananas.

// 定义：速度为 x 时，需要 f(x) 小时吃完所有香蕉
// f(x) 随着 x 的增加单调递减
int f(int[] piles, int x) {
    int hours = 0;
    for (int i = 0; i < piles.length; i++) {
        hours += piles[i] / x;
        if (piles[i] % x > 0) hours++;
    }
    return hours;
}

public int minEatingSpeed(int[] piles, int H) {
    int left = 1, right = 1000000000 + 1;

    while (left < right) {
        int mid = left + (right - left) / 2;

        // // 搜索左侧边界，则需要收缩右侧边界
        // if (f(piles, mid) == H) right = mid;
        // // 需要让 f(x) 的返回值大一些
        // else if (f(piles, mid) < H) right = mid;
        // // 需要让 f(x) 的返回值小一些
        // else if (f(piles, mid) > H) left = mid + 1;

        // if (f(piles, mid) <= H) right = mid;
        if (f(piles, mid) <= H) right = mid-1;
        // 需要让 f(x) 的返回值小一些
        else left = mid + 1;
    }
    return left;
}
```


---

##### 运送货物？？？？？？？？？？？？？？

[1011. Capacity To Ship Packages Within D Days](https://leetcode.com/problems/capacity-to-ship-packages-within-d-days/)

A conveyor belt has packages that must be shipped from one port to another within days days.

The ith package on the conveyor belt has a weight of weights[i]. Each day, we load the ship with packages on the conveyor belt (in the order given by weights). We may not load more weight than the maximum weight capacity of the ship.

Return the least weight capacity of the ship that will result in all the packages on the conveyor belt being shipped within days days.



Example 1:

Input: weights = [1,2,3,4,5,6,7,8,9,10], days = 5
Output: 15


x = capacity
target = D
weights[i]



```java
public int f(int[] weights, int x){
    int days=0;
    int left = 0;
    for(int wei : weights){
        if(wei>x)
        days += (left+wei)/x;
        left = (left+wei)%x;
    }
    return days;
}



public int shipWithinDays(int[] weights, int days){
    int left=0; right=
}


```

---


##### https://labuladong.github.io/algo/2/21/59/ ？？？？




---


### 两数之和


[167. Two Sum II - Input Array Is Sorted](https://leetcode.com/problems/two-sum-ii-input-array-is-sorted/)

Given a (索引是从 1 开始的) `1-indexed array` of integers numbers that is already sorted in non-decreasing order, find two numbers such that they add up to a specific target number. Let these two numbers be numbers[index1] and numbers[index2] where 1 <= index1 < index2 <= numbers.length.

Return the indices of the two numbers, index1 and index2, added by one as an integer array [index1, index2] of length 2.

The tests are generated such that there is exactly one solution. You may not use the same element twice.

```java
int[] twoSum(int[] nums, int target) {
    int left = 0, right = nums.length - 1;
    while (left < right) {
        int sum = nums[left] + nums[right];
        // 题目要求的索引是从 1 开始的
        if (sum == target) return new int[]{left + 1, right + 1};
        // 让 sum 大一点
        else if (sum < target) left++;
        // 让 sum 小一点
        else if (sum > target) right--;
    }
    return new int[]{-1, -1};
}
```


---


### 344. Reverse String 反转数组

一般编程语言都会提供 reverse 函数

[344. Reverse String](https://leetcode.com/problems/reverse-string/)

Write a function that reverses a string. The input string is given as an array of characters s.

You must do this by modifying the input array in-place with O(1) extra memory.

反转一个 char[] 类型的字符数组


```java
// Runtime: 1 ms, faster than 95.40% of Java online submissions for Reverse String.
// Memory Usage: 45.6 MB, less than 89.34% of Java online submissions for Reverse String.

void reverseString(char[] arr) {
    int left = 0, right = arr.length - 1;
    while (left < right) {
        // 交换 arr[left] 和 arr[right]
        char temp = arr[left];
        arr[left] = arr[right];
        arr[right] = temp;
        left++; right--;
    }
}
```

---

### 滑动窗口技巧 `right++, missing==0, left++`

维护一个窗口，不断滑动，然后更新答案么。

该算法的大致逻辑, 时间复杂度是 O(N)，比字符串暴力算法要高效得多。

```java
int left = 0, right = 0;

while (right < s.size()) {
    // 增大窗口
    window.add(s[right]);
    right++;

    while (window needs shrink) {
        // 缩小窗口
        window.remove(s[left]);
        left++;
    }
}
```

```java
/* 滑动窗口算法框架 */
void slidingWindow(string s, string t) {
    unordered_map<char, int> need, window;

    for (char c : t) need[c]++;

    int left = 0, right = 0;
    int valid = 0;

    while (right < s.size()) {
        // c 是将移入窗口的字符
        char c = s[right];
        // 右移窗口
        right++;
        // 进行窗口内数据的一系列更新
        ...

        /*** debug 输出的位置 ***/
        printf("window: [%d, %d)\n", left, right);
        /********************/

        // 判断左侧窗口是否要收缩
        while (window needs shrink) {
            // d 是将移出窗口的字符
            char d = s[left];
            // 左移窗口
            left++;
            // 进行窗口内数据的一系列更新
            ...
        }
    }
}
```


---

#### 最小覆盖子串

[76. Minimum Window Substring](https://leetcode.com/problems/minimum-window-substring/)

- Given two strings s and t of lengths m and n respectively, return the minimum window substring of s such that every character in t (including duplicates) is included in the window.
- If there is no such substring, return the empty string "".
- The testcases will be generated such that the answer is unique.
- A substring is a contiguous sequence of characters within the string.

暴力解法，代码大概是这样的：
```java
for (int i = 0; i < s.size(); i++)
    for (int j = i + 1; j < s.size(); j++)
        if s[i:j] 包含 t 的所有字母:
            更新答案

```


滑动窗口算法的思路:

1. 我们在字符串 S 中使用双指针中的左右指针技巧，初始化 left = right = 0，把索引左闭右开区间 [left, right) 称为一个「窗口」。

2. 我们先不断地增加 right 指针扩大窗口 [left, right)，直到窗口中的字符串符合要求（包含了 T 中的所有字符）。

3. 此时，我们停止增加 right，转而不断增加 left 指针缩小窗口 [left, right)，直到窗口中的字符串不再符合要求（不包含 T 中的所有字符了）。同时，每次增加 left，我们都要更新一轮结果。

4. 重复第 2 和第 3 步，直到 right 到达字符串 S 的尽头。

```java
public static String minWindow(String s, String t) {
    Map<Character, Integer> need = new HashMap<>();
    for(char i:t.toCharArray()) need.put(i, need.getOrDefault(i,0)+1); // {A:2, B:1}

    int start=0, len = s.length()+1;
    int left=0, right=0, valid=0;
    int missing = t.length(); //The number of chars are missing.

    Map<Character, Integer> win = new HashMap<>();
    while(right<s.length()){
        right++;

        char x = s.charAt(right);
        if(need.containsKey(x)){
            win.put(x, win.getOrDefault(x,0)+1);
            if(win.get(x) == need.get(x)) valid++;
        }

        while(valid == need.size()){
            if(right-left < len){
                start=left;
                len = right-left;
            }

            char y = s.charAt(left);
            left++;
            if(need.containsKey(y)){
                if(win.get(y) == need.get(y)) valid--;
                win.put(y, win.get(y)-1);
            }
        }
    }
    return len==s.length()+1 ? "" : s.substring(start,start+len);
}

// Runtime: 10 ms, faster than 70.54% of Java online submissions for Minimum Window Substring.
// Memory Usage: 39.1 MB, less than 82.86% of Java online submissions for Minimum Window Substring.
class Solution {
    public String minWindow(String s, String t) {
        if(s==null || t==null) throw new IllegalArgumentException("Input string is null");
        if(s.length() < t.length()) return "";
        HashMap<Character, Integer> map = new HashMap<>();
        for(int i=0;i<t.length();i++) map.put(t.charAt(i), map.getOrDefault(t.charAt(i), 0)+1);
        int left=0, right=0;
        int start=0, len = Integer.MAX_VALUE;
        int missing = t.length();
        while(right<s.length()){
            char x = s.charAt(right);
            if(map.containsKey(x)){
                int countX = map.get(x);
                if(countX > 0) missing--;
                map.put(x, countX-1);
            }
            right++;
            while(missing==0){
                if(right-left < len){
                    start=left;
                    len = right-left;
                }
                char y = s.charAt(left);
                if(map.containsKey(y)){
                    int countY = map.get(y);
                    if(countY == 0) missing++;
                    map.put(y, countY+1);
                }
                left++;
            }
        }
        return len==Integer.MAX_VALUE ? "" : s.substring(start, start+len);
    }
}
```

---


#### 567. Permutation in String 字符串排列

[567. Permutation in String](https://leetcode.com/problems/permutation-in-string/)

Given two strings s1 and s2, return true if s2 contains a permutation of s1, or false otherwise.

In other words, return true if one of s1's permutations is the substring of s2.



Example 1:

Input: s1 = "ab", s2 = "eidbaooo"
Output: true
Explanation: s2 contains one permutation of s1 ("ba").


```java
// 判断 s 中是否存在 t 的排列
bool checkInclusion(string t, string s) {
    unordered_map<char, int> need, window;
    for (char c : t) need[c]++;

    int left = 0, right = 0;
    int valid = 0;
    while (right < s.size()) {
        char c = s[right];
        right++;
        // 进行窗口内数据的一系列更新
        if (need.count(c)) {
            window[c]++;
            if (window[c] == need[c])
                valid++;
        }

        // 判断左侧窗口是否要收缩
        while (right - left >= t.size()) {
            // 在这里判断是否找到了合法的子串
            if (valid == need.size())
                return true;
            char d = s[left];
            left++;
            // 进行窗口内数据的一系列更新
            if (need.count(d)) {
                if (window[d] == need[d])
                    valid--;
                window[d]--;
            }
        }
    }
    // 未找到符合条件的子串
    return false;
}

// Runtime: 12 ms, faster than 48.78% of Java online submissions for Permutation in String.
// Memory Usage: 38.9 MB, less than 91.88% of Java online submissions for Permutation in String.
class Solution {
    public boolean checkInclusion(String s1, String s2) {
        if(s1==null || s2==null) throw new IllegalArgumentException("Input string is null");
        if(s1.length()>s2.length()) return false;

        int left=0, right=0;
        int start=0, len=Integer.MAX_VALUE;
        int missing=s1.length();

        HashMap<Character, Integer> map = new HashMap<>();
        for(int i=0;i<s1.length();i++) map.put(s1.charAt(i), map.getOrDefault(s1.charAt(i), 0)+1);

        while(right<s2.length()){
            char x = s2.charAt(right);
            if(map.containsKey(x)){
                if(map.get(x)>0) missing--;
                map.put(x, map.get(x)-1);
            }
            right++;

            while(missing==0){
                if(right-left==s1.length()) return true;
                char y = s2.charAt(left);
                if(map.containsKey(y)){
                    if(map.get(y)==0) missing++;
                    map.put(y, map.get(y)+1);
                }
                left++;
            }
        }
        return false;
    }
}
```

---


#### 438. Find All Anagrams in a String 找所有字母异位词

[438. Find All Anagrams in a String](https://leetcode.com/problems/find-all-anagrams-in-a-string/)

- Given two strings s and p,
- return an array of all the start indices of `p's anagrams in s`.
- You may return the answer in any order.
- An `Anagram` is a word or phrase formed by rearranging the letters of a different word or phrase, typically using all the original letters exactly once.

Example 1:

Input: s = "cbaebabacd", p = "abc"
Output: [0,6]

1. size same
2. missing==0


```java
// Runtime: 29 ms, faster than 34.79% of Java online submissions for Find All Anagrams in a String.
// Memory Usage: 45.6 MB, less than 10.47% of Java online submissions for Find All Anagrams in a String.
class Solution {
    public List<Integer> findAnagrams(String s, String p) {
        HashMap<Character, Integer> map = new HashMap<>();
        for(int i=0; i<p.length(); i++) map.put(p.charAt(i), map.getOrDefault(p.charAt(i),0)+1);

        List<Integer> res = new ArrayList<>();

        int left=0, right=0;
        int missing=p.length();

        while(right<s.length()){
            char x = s.charAt(right);
            if(map.containsKey(x)){
                if(map.get(x)>0) missing--;
                map.put(x, map.get(x)-1);
            }
            right++;

            while(missing==0 && left<s.length()){
                if(right-left==p.length()) res.add(left);
                char y = s.charAt(left);
                if(map.containsKey(y)){
                    if(map.get(y)==0) missing++;
                    map.put(y, map.get(y)+1);
                }
                left++;
            }
        }
        return res;
    }
}
```


---


#### 3. Longest Substring Without Repeating Characters 最长无重复子串

[3. Longest Substring Without Repeating Characters](https://leetcode.com/problems/longest-substring-without-repeating-characters/)

Given a string s, find the length of the longest substring without repeating characters.

Example 1:

Input: s = "abcabcbb"
Output: 3
Explanation: The answer is "abc", with the length of 3.

```java
public int lengthOfLongestSubstring(String s) {
    HashMap<Character, Integer> map = new HashMap<>();
    int left=0, right=0;
    int res=0;

    while(right<s.length()){
        char x = s.charAt(right);
        map.put(x, map.getOrDefault(x, 0)+1);
        right++;

        while(map.get(x)>1){
            char y = s.charAt(left);
            left++;
            map.put(y, map.get(y)-1);
        }
        res=Math.max(res,right-left);
    }
    return res;
}
```




---


## 链表的环


--

### 判断单链表是否包含环

[142. Linked List Cycle II](https://leetcode.com/problems/linked-list-cycle-ii/)
- Given the head of a linked list, return the node where the cycle begins. If there is no cycle, return null.

- There is a cycle in a linked list if there is some node in the list that can be reached again by continuously following the next pointer. Internally, pos is used to denote the index of the node that tail's next pointer is connected to (0-indexed). It is -1 if there is no cycle. Note that pos is not passed as a parameter.

- Do not modify the linked list.


solution:
- 每当慢指针 slow 前进一步，快指针 fast 就前进两步。
- 如果 fast 最终遇到空指针，说明链表中没有环；
- 如果 fast 最终和 slow 相遇，那肯定是 fast 超过了 slow 一圈，说明链表中含有环。


```java
boolean hasCycle(ListNode head) {
    // 快慢指针初始化指向 head
    ListNode slow = head, fast = head;
    // 快指针走到末尾时停止
    while (fast != null && fast.next != null) {
        // 慢指针走一步，快指针走两步
        slow = slow.next;
        fast = fast.next.next;
        // 快慢指针相遇，说明含有环
        if (slow == fast) return true;
    }
    // 不包含环
    return false;
}
```


---

### 142. Linked List Cycle II 计算链表中环起点

快慢指针相遇时，慢指针 slow 走了 k 步，那么快指针 fast 一定走了 2k 步：
- fast 一定比 slow 多走了 k 步，这多走的 k 步其实就是 fast 指针在环里转圈圈，所以 k 的值就是环长度的「整数倍」。
- 假设相遇点距环的起点的距离为 m，那么环的起点距头结点 head 的距离为 k - m，也就是说如果从 head 前进 k - m 步就能到达环起点。
- 如果从相遇点继续前进 k - m 步，也恰好到达环起点。
  - 因为结合上图的 fast 指针，从相遇点开始走k步可以转回到相遇点，那走 k - m 步肯定就走到环起点了
- 所以，只要我们把快慢指针中的任一个重新指向 head，然后两个指针同速前进，k - m 步后一定会相遇，相遇之处就是环的起点了。


[142. Linked List Cycle II](https://leetcode.com/problems/linked-list-cycle-ii/)

Given the head of a linked list, return the node where the cycle begins. If there is no cycle, return null.

There is a cycle in a linked list if there is some node in the list that can be reached again by continuously following the next pointer. Internally, pos is used to denote the index of the node that tail's next pointer is connected to (0-indexed). It is -1 if there is no cycle. Note that pos is not passed as a parameter.

Do not modify the linked list.


```java
// Runtime: 0 ms, faster than 100.00% of Java online submissions for Linked List Cycle II.
// Memory Usage: 39.1 MB, less than 62.77% of Java online submissions for Linked List Cycle II.
ListNode detectCycle(ListNode head) {
    ListNode fast, slow;
    fast = slow = head;
    while (fast != null && fast.next != null) {
        fast = fast.next.next;
        slow = slow.next;
        if (fast == slow) break;
    }
    // 上面的代码类似 hasCycle 函数
    if (fast == null || fast.next == null) {
        // fast 遇到空指针说明没有环
        return null;
    }
    // 重新指向头结点
    slow = head;
    // 快慢指针同步前进，相交点就是环起点
    while (slow != fast) {
        fast = fast.next;
        slow = slow.next;
    }
    return slow;
}
```

---



---

# 回文链表 Palindromic

- 寻找回文串是从中间向两端扩展，
- 判断回文串是从两端向中间收缩。

对于单链表
- 无法直接倒序遍历，可以造一条新的反转链表，
- 可以利用链表的后序遍历，也可以用栈结构倒序处理单链表。

---



---

## other


### 9. Palindrome Number 判断回文Number

[9. Palindrome Number](https://leetcode.com/problems/palindrome-number/)
- Given an integer x, return true if x is palindrome integer.
- An integer is a palindrome when it reads the same backward as forward.
- For example, 121 is palindrome while 123 is not.


#### reverse half of it **Best**

O(1) space solution

```java
/**
 * Reverse Half & Compare
 *
 * Time Complexity: O((log10 N) / 2)
 *
 * Space Complexity: O(1)
 *
 * N = Number of digits in input number.
 */
//  Runtime: 6 ms, faster than 99.94% of Java online submissions for Palindrome Number.
// Memory Usage: 38.2 MB, less than 89.24% of Java online submissions for Palindrome Number.
public boolean isPalindrome(int x) {
    if(x<0 || (x!=0 && x%10==0)) return false;
    if(x<0) return false;
    if(x==0) return true;
    if(x%10==0) return false;
    if(x<10) return true;
    if(x<100 && x%11==0) return true;
    if(x<1000 && ((x/100)*10+x%10)%11 == 0) return true;

    // 12321
    // 1232 1
    // 123 12
    // 12 123
    int res = 0;
    while(x>res){
        res = res*10 + x%10;
        x = x/10;
       }
    return (x==res || x==res/10);
}

class Solution {
    public boolean isPalindrome(int x) {
        int reversed = 0;
        for (int i = x; i > 0; i /= 10) reversed = reversed * 10 + i % 10;
        return reversed == x;
    }
}
```

---


### Elimination Game !!! Perform String Shifts !!! Subtree Removal Game with Fibonacci Tree


---

### 125. Valid Palindrome 判断回文链表String


[125. Valid Palindrome]
- A phrase is a palindrome if, after converting all uppercase letters into lowercase letters and removing all non-alphanumeric characters, it reads the same forward and backward. Alphanumeric characters include letters and numbers.
- Given a string s, return true if it is a palindrome, or false otherwise.
- Input: s = "A man, a plan, a canal: Panama"
- Output: true


```java
// Runtime: 23 ms, faster than 31.39% of Java online submissions for Valid Palindrome.
// Memory Usage: 39.9 MB, less than 60.42% of Java online submissions for Valid Palindrome.
// 双指针
class Solution {
    public boolean isPalindrome(String s) {
        String scheck = s.replaceAll("[^a-zA-Z0-9]", "").toLowerCase();
        int a = 0, b = scheck.length() - 1;
        while(a<b){
            if(scheck.charAt(a)!=scheck.charAt(b)) return false;
            a++; b--;
        }
        return true;
    }
}

public boolean isPalindrome(String s){
    char[] charMap = new char[256];
    for (int i = 0; i < 10; i++)
        charMap['0'+i] = (char) (1+i);
        // numeric - don't use 0 as it's reserved for illegal chars
    for (int i = 0; i < 26; i++)
        charMap['a'+i] = charMap['A'+i] = (char) (11+i);
        //alphabetic, ignore cases, continue from 11
    for (int start = 0, end = s.length()-1; start < end;) {
        // illegal chars
        if (charMap[s.charAt(start)] == 0) start++;
        else if (charMap[s.charAt(end)] == 0) end--;
        else if (charMap[s.charAt(start++)] != charMap[s.charAt(end--)]) return false;
    }
    return true;
}
```

---


#### 判断回文单链表 - 把原始链表反转存入一条新的链表，然后比较

point: 单链表无法倒着遍历，无法使用双指针技巧。

把原始链表反转存入一条新的链表，然后比较这两条链表是否相同。

```java
```

---

#### 判断回文单链表 - 二叉树后序遍历

借助二叉树后序遍历的思路，不需要显式反转原始链表也可以倒序遍历链表



```java
void traverse(TreeNode root) {
    // 前序遍历代码
    traverse(root.left);
    // 中序遍历代码
    traverse(root.right);
    // 后序遍历代码
}
```


链表其实也有前序遍历和后序遍历：

```java
void traverse(ListNode head) {
    // 前序遍历代码
    traverse(head.next);
    // 后序遍历代码
}
```


正序打印链表中的 val 值，可以在前序遍历位置写代码；
反之，如果想倒序遍历链表，就可以在后序遍历位置操作：

```java
/* 倒序打印单链表中的元素值 */
void traverse(ListNode head) {
    if (head == null) return;
    traverse(head.next);
    // 后序遍历代码
    print(head.val);
}
```

---

#### 判断回文单链表 - 用栈结构倒序处理单链表

模仿双指针实现回文判断的功能：
- 把链表节点放入一个栈，然后再拿出来，
- 这时候元素顺序就是反的，只不过我们利用的是递归函数的堆栈而已。

```java
// 左侧指针
ListNode left;

boolean isPalindrome(ListNode head) {
    left = head;
    return traverse(head);
}

boolean traverse(ListNode right) {
    if (right == null) return true;
    boolean res = traverse(right.next);
    // 后序遍历代码
    res = res && (right.val == left.val);
    left = left.next;
    return res;
}
```

---

#### 判断回文单链表 - 不完全反转链表，仅仅反转部分链表，空间复杂度O(1)。

更好的思路是这样的：

```java
// 1234 5 6789
// 1 23 45 67 89
// 1 2  3  4
// 先通过 双指针技巧 中的快慢指针来找到链表的中点：
boolean isPalindrome(ListNode head){
    ListNode slow=head, fast=head;
    while(fast!=null&&fast.next!=null){
        slow=slow.next;
        fast=fast.next.next;
    }
    if(fast!=null){
        slow=slow.next;
    }
    ListNode right=head;
    ListNode left=reverse(slow);
    while(right!=null){
        if(left.val!=right.val) return false;
        right=right.next, left=left.next;
    }
    return true;
}

ListNode reverse(ListNode head) {
    ListNode pre = null, cur = head;
    while (cur != null) {
        ListNode next = cur.next;
        cur.next = pre;
        pre = cur;
        cur = next;
    }
    return pre;
}
```


- 时间复杂度 O(N)，
- 空间复杂度 O(1)，已经是最优的了。


---


## 排序

- 快速排序就是个二叉树的前序遍历，
- 归并排序就是个二叉树的后序遍历


### 快速排序

快速排序的逻辑是，
- 对 nums[lo..hi] 进行排序，我们先找一个分界点 p，通过交换元素使得 nums[lo..p-1] 都小于等于 nums[p]，且 nums[p+1..hi] 都大于 nums[p]，
- 然后递归地去 nums[lo..p-1] 和 nums[p+1..hi] 中寻找新的分界点，
- 最后整个数组就被排序了。

先构造分界点，然后去左右子数组构造分界点，
- 就是一个二叉树的前序遍历


```java
void sort(int[] nums, int lo, int hi) {
    /****** 前序遍历位置 ******/
    // 通过交换元素构建分界点 p
    int p = partition(nums, lo, hi);
    /************************/

    sort(nums, lo, p - 1);
    sort(nums, p + 1, hi);
}
```

### 归并排序

归并排序的逻辑，
- 要对 nums[lo..hi] 进行排序，我们先对 nums[lo..mid] 排序，再对 nums[mid+1..hi] 排序，最后把这两个有序的子数组合并，整个数组就排好序了。

二叉树的后序遍历框架
- 先对左右子数组排序，然后合并（类似合并有序链表的逻辑）

```java
void sort(int[] nums, int lo, int hi) {
    int mid = (lo + hi) / 2;
    sort(nums, lo, mid);
    sort(nums, mid + 1, hi);
    /****** 后序遍历位置 ******/
    // 合并两个排好序的子数组
    merge(nums, lo, mid, hi);
    /************************/
}
```


---


# stack

栈（stack）是很简单的一种数据结构，先进后出的逻辑顺序，符合某些问题的特点，比如说函数调用栈。

---


## 队列 栈

---


### 用栈实现队列


[232. Implement Queue using Stacks](https://leetcode.com/problems/implement-queue-using-stacks/)
- Implement a first in first out (FIFO) queue using only two stacks. The implemented queue should support all the functions of a normal queue (push, peek, pop, and empty).
- Implement the MyQueue class:

- void push(int x) Pushes element x to the back of the queue.
- int pop() Removes the element from the front of the queue and returns it.
- int peek() Returns the element at the front of the queue.
- boolean empty() Returns true if the queue is empty, false otherwise.




```java
// Runtime: 0 ms, faster than 100.00% of Java online submissions for Implement Queue using Stacks.
// Memory Usage: 36.7 MB, less than 89.99% of Java online submissions for Implement Queue using Stacks

class MyQueue {

    private Stack<Integer> s1, s2;

    public MyQueue() {
        s1 = new Stack<>();
        s2 = new Stack<>();
    }

    /** 添加元素到队尾 */
    public void push(int x){
        s1.push(x);
    };

    /** 删除队头的元素并返回 */
    public int pop(){
        // 先调用 peek 保证 s2 非空
        peek();
        return s2.pop();
    };

    /** 返回队头元素 */
    // 触发 while 循环，这样的话时间复杂度是 O(N)
    public int peek() {
        if (s2.isEmpty())
            // 把 s1 元素压入 s2
            while (!s1.isEmpty())
                s2.push(s1.pop());
        return s2.peek();
    }

    /** 判断队列是否为空 */
    public boolean empty(){
        return s1.isEmpty() && s2.isEmpty();
    };
}
```

---

### 用队列实现栈

[225. Implement Stack using Queues](https://leetcode.com/problems/implement-stack-using-queues/)
- Implement a last-in-first-out (LIFO) stack using only two queues. The implemented stack should support all the functions of a normal stack (push, top, pop, and empty).
- Implement the MyStack class:
- void push(int x) Pushes element x to the top of the stack.
- int pop() Removes the element on the top of the stack and returns it.
- int top() Returns the element on the top of the stack.
- boolean empty() Returns true if the stack is empty, false otherwise.

pop 操作时间复杂度是 O(N)，其他操作都是 O(1)​。​

```java
// Runtime: 0 ms, faster than 100.00% of Java online submissions for Implement Stack using Queues.
// Memory Usage: 37.2 MB, less than 35.03% of Java online submissions for Implement Stack using Queues.

class MyStack {

    Queue<Integer> q = new LinkedList<>();
    int top = 0;

    /** 添加元素到栈顶 */
    public void push(int x){
        q.offer(x);
        top = x;
    };

    /** 删除栈顶的元素并返回 */
    public int pop(){
        int size = q.size();
        while (size > 2) {
            q.offer(q.pop());
            size--;
        }
        top = q.peek()
        q.offer(q.pop());
        return q.poll()
    };


    /** 返回栈顶元素 */
    public int top(){
        return top;
    };

    /** 判断栈是否为空 */
    public boolean empty(){
        return q.isEmpty();
    };
}
```




---

## 单调栈

- 单调栈实际上就是栈，只是利用了一些巧妙的逻辑，使得每次新元素入栈后，栈内的元素都保持有序（单调递增或单调递减）。
- 有点像堆（heap）？不是的，单调栈用途不太广泛，只处理一种典型的问题，叫做 Next Greater Element。


### 返回等长数组for更大的元素

给你一个数组 nums，请你返回一个等长的结果数组，结果数组中对应索引存储着下一个更大元素，如果没有更大的元素，就存 -1。

```java
vector<int> nextGreaterElement(vector<int>& nums) {
    vector<int> res(nums.size()); // 存放答案的数组
    stack<int> s;
    // 倒着往栈里放
    for (int i = nums.size() - 1; i >= 0; i--) {
        // 判定个子高矮
        while (!s.empty() && s.peek() <= nums[i]) {
            // 矮个起开，反正也被挡着了。。。
            s.pop();
        }
        // nums[i] 身后的 next great number
        res[i] = s.empty() ? -1 : s.peek();
        s.push(nums[i]);
    }
    return res;
}
```




---

### 返回等长数组for更大的元素的index


[739. Daily Temperatures](https://leetcode.com/problems/daily-temperatures/)
- Given an array of integers temperatures represents the daily temperatures, return an array answer such that answer[i] is the number of days you have to wait after the ith day to get a warmer temperature. If there is no future day for which this is possible, keep answer[i] == 0 instead.
- 给你一个数组 T，这个数组存放的是近几天的天气气温，你返回一个等长的数组，计算：对于每一天，你还要至少等多少天才能等到一个更暖和的气温；如果等不到那一天，填 0。

```java
// Runtime: 39 ms, faster than 43.88% of Java online submissions for Daily Temperatures.
// Memory Usage: 48.4 MB, less than 85.99% of Java online submissions for Daily Temperatures.
public int[] dailyTemperatures(int[] temperatures) {
    while(temperatures.length==0) return temperatures;
    Stack<Integer> s = new Stack<>();
    int[] res  = new int[temperatures.length];
    for(int i=temperatures.length-1; i>=0; i--){
        while(!s.empty() && temperatures[s.peek()] <= temperatures[i]) s.pop();
        res[i] = s.empty() ? 0 : s.peek()-i;
        s.push(i);
    }
    return res;
}
```

---

### 环形数组

对于这种需求，常用套路就是将数组长度翻倍：

```java
vector<int> nextGreaterElements(vector<int>& nums) {
    int n = nums.size();
    vector<int> res(n);
    stack<int> s;
    // 假装这个数组长度翻倍了
    for (int i = 2*n - 1; i >= 0; i--) {
        // 索引要求模，其他的和模板一样
        while (!s.empty() && s.peak() <= nums[i % n]) s.pop();
        res[i % n] = s.empty() ? -1 : s.peak();
        s.push(nums[i % n]);
    }
    return res;
}
```

---

## 单调队列结构

一个「队列」，队列中的元素全都是单调递增（或递减）的。

---

### 滑动窗口问题

在 O(1) 时间算出每个「窗口」中的最大值

239 题「滑动窗口最大值」，难度 Hard：
- 给你输入一个数组 nums 和一个正整数 k，有一个大小为 k 的窗口在 nums 上从左至右滑动，请你输出每次窗口中 k 个元素的最大值。
- 在一堆数字中，已知最值为 A，如果给这堆数添加一个数 B，那么比较一下 A 和 B 就可以立即算出新的最值；
- 但如果减少一个数，就不能直接得到最值了，因为如果减少的这个数恰好是 A，就需要遍历所有数重新找新的最值。

```java
// Runtime: 36 ms, faster than 47.84% of Java online submissions for Sliding Window Maximum.
// Memory Usage: 55.2 MB, less than 45.54% of Java online submissions for Sliding Window Maximum.

/* 单调队列的实现 */
class MonotonicQueue {
    LinkedList<Integer> q = new LinkedList<>();

    public void push(int n) {
        // 将小于 n 的元素全部删除
        while (!q.isEmpty() && q.getLast() < n) q.pollLast();
        // 然后将 n 加入尾部
        q.addLast(n);
    }

    public int max() {
        return q.getFirst(); // 队头的元素肯定是最大的
    }

    public void pop(int n) {
        if (n == q.getFirst()) q.pollFirst(); // 在队头删除元素 n
    }
}

/* 解题函数的实现 */
int[] maxSlidingWindow(int[] nums, int k) {
    MonotonicQueue window = new MonotonicQueue();
    List<Integer> res = new ArrayList<>();

    for (int i = 0; i < nums.length; i++) {
        //先填满窗口的前 k - 1
        if (i < k - 1) window.push(nums[i]);
        else {
            // 窗口向前滑动，加入新数字
            window.push(nums[i]);
            // 记录当前窗口的最大值
            res.add(window.max());
            // 移出旧数字
            window.pop(nums[i - k + 1]);
        }
    }
    // 需要转成 int[] 数组再返回
    int[] arr = new int[res.size()];
    for (int i = 0; i < res.size(); i++) arr[i] = res.get(i);
    return arr;
}
```



---

# Tree



---

## 二叉树

树的问题就永远逃不开树的递归遍历框架这几行代码：
- 二叉树题目的一个难点就是，如何把`题目的要求`细化成`每个节点需要做的事情`。

```java
/* 二叉树遍历框架 */
void traverse(TreeNode root) {
    // 前序遍历
    traverse(root.left)
    // 中序遍历
    traverse(root.right)
    // 后序遍历
}
```

---


### 计算一棵二叉树共有几个节点

[222. Count Complete Tree Nodes](https://leetcode.com/problems/count-complete-tree-nodes/)
- Given the root of a complete binary tree, return the number of the nodes in the tree.
- According to Wikipedia, every level, except possibly the last, is completely filled in a complete binary tree, and all nodes in the last level are as far left as possible. It can have between 1 and 2h nodes inclusive at the last level h.
- Design an algorithm that runs in less than O(n) time complexity.


```java
// Runtime: 0 ms, faster than 100.00% of Java online submissions for Count Complete Tree Nodes.
// Memory Usage: 41.7 MB, less than 66.40% of Java online submissions for Count Complete Tree Nodes.

// 定义：count(root) 返回以 root 为根的树有多少节点
// 时间复杂度 O(N)：
int count(TreeNode root) {
    // base case
    if (root == null) return 0;
    // 自己加上子树的节点数就是整棵树的节点数
    return 1 + count(root.left) + count(root.right);
}

// 一棵满二叉树，节点总数就和树的高度呈指数关系：
public int countNodes(TreeNode root) {
    int h = 0;
    // 计算树的高度
    while (root != null) {
        root = root.left;
        h++;
    }
    // 节点总数就是 2^h - 1
    return (int)Math.pow(2, h) - 1;
}

// 完全二叉树比普通二叉树特殊，但又没有满二叉树那么特殊，
// 计算它的节点总数，可以说是普通二叉树和完全二叉树的结合版，先看代码：
public int countNodes(TreeNode root) {
    TreeNode l = root, r = root;
    // 记录左、右子树的高度
    int hl = 0, hr = 0;
    while (l != null) {
        l = l.left;
        hl++;
    }
    while (r != null) {
        r = r.right;
        hr++;
    }
    // 如果左右子树的高度相同，则是一棵满二叉树
    if (hl == hr) return (int)Math.pow(2, hl) - 1;
    // 如果左右高度不同，则按照普通二叉树的逻辑计算
    return 1 + countNodes(root.left) + countNodes(root.right);
}
```



---


### 翻转二叉树

[226. Invert Binary Tree](https://leetcode.com/problems/invert-binary-tree/)
- Given the root of a binary tree, invert the tree, and return its root.

```java
// Runtime: 0 ms, faster than 100.00% of Java online submissions for Invert Binary Tree.
// Memory Usage: 36.7 MB, less than 57.60% of Java online submissions for Invert Binary Tree.
// 将整棵树的节点翻转
TreeNode invertTree(TreeNode root) {
    // base case
    if (root == null) return null;
    /**** 前序遍历位置 ****/
    // root 节点需要交换它的左右子节点
    TreeNode tmp = root.left;
    root.left = root.right;
    root.right = tmp;
    // 让左右子节点继续翻转它们的子节点
    invertTree(root.left);
    invertTree(root.right);
    return root;
}
```

---

### 填充二叉树节点的右侧指针

[116. Populating Next Right Pointers in Each Node](https://leetcode.com/problems/populating-next-right-pointers-in-each-node/)
- You are given a perfect binary tree where all leaves are on the same level, and every parent has two children. The binary tree has the following definition:

  ```java
  struct Node {
    int val;
    Node *left;
    Node *right;
    Node *next;
  }
  ```
- Populate each next pointer to point to its next right node. If there is no next right node, the next pointer should be set to NULL.
- Initially, all next pointers are set to NULL.

![116_sample](https://i.imgur.com/35aMwHI.png)


```java
// Runtime: 2 ms, faster than 52.34% of Java online submissions for Populating Next Right Pointers in Each Node.
// Memory Usage: 39.3 MB, less than 69.08% of Java online submissions for Populating Next Right Pointers in Each Node.
// 主函数
Node connect(Node root) {
    if (root == null) return null;
    connectTwoNode(root.left, root.right);
    return root;
}

// 辅助函数
void connectTwoNode(Node node1, Node node2) {
    if (node1 == null || node2 == null) return;
    /**** 前序遍历位置 ****/
    // 将传入的两个节点连接
    node1.next = node2;
    // 连接相同父节点的两个子节点
    connectTwoNode(node1.left, node1.right);
    connectTwoNode(node2.left, node2.right);
    // 连接跨越父节点的两个子节点
    connectTwoNode(node1.right, node2.left);
}
```

---

### 将二叉树展开为链表

[114. Flatten Binary Tree to Linked List](https://leetcode.com/problems/flatten-binary-tree-to-linked-list/)
- Given the root of a binary tree, flatten the tree into a "linked list":
- The "linked list" should use the same TreeNode class where the right child pointer points to the next node in the list and the left child pointer is always null.
- The "linked list" should be in the same order as a pre-order traversal of the binary tree.
- Input: root = [1,2,5,3,4,null,6]
- Output: [1,null,2,null,3,null,4,null,5,null,6]

尝试给出这个函数的定义：
- 给 flatten 函数输入一个节点 root，那么以 root 为根的二叉树就会被拉平为一条链表。
- 1、将 root 的左子树和右子树拉平。
- 2、将 root 的右子树接到左子树下方，然后将整个左子树作为右子树。


```java
// Runtime: 0 ms, faster than 100.00% of Java online submissions for Flatten Binary Tree to Linked List.
// Memory Usage: 38.5 MB, less than 70.26% of Java online submissions for Flatten Binary Tree to Linked List.
// 定义：将以 root 为根的树拉平为链表
void flatten(TreeNode root) {
    // base case
    if (root == null) return;
    flatten(root.left);
    flatten(root.right);

    /**** 后序遍历位置 ****/
    // 1、左右子树已经被拉平成一条链表
    // 2、将左子树作为右子树
    TreeNode temp = root.right;
    root.right = root.left;
    root.left = null;
    // 3、将原先的右子树接到当前右子树的末端
    TreeNode p = root;
    while (p.right != null) {
        p = p.right;
    }
    p.right = temp;
}
```

---


### 构造最大二叉树

[654. Maximum Binary Tree](https://leetcode.com/problems/maximum-binary-tree/)
- You are given an integer array nums with no duplicates. A maximum binary tree can be built recursively from nums using the following algorithm:
- Create a root node whose value is the maximum value in nums.
- Recursively build the left subtree on the subarray prefix to the left of the maximum value.
- Recursively build the right subtree on the subarray suffix to the right of the maximum value.
- Return the maximum binary tree built from nums.

- 先明确根节点做什么？对于构造二叉树的问题，根节点要做的就是把想办法把自己构造出来。
- 肯定要遍历数组把找到最大值 maxVal，把根节点 root 做出来，
- 然后对 maxVal 左边的数组和右边的数组进行递归调用，作为 root 的左右子树。


```java
// Runtime: 2 ms, faster than 90.01% of Java online submissions for Maximum Binary Tree.
// Memory Usage: 39.1 MB, less than 82.91% of Java online submissions for Maximum Binary Tree.

/* 主函数 */
TreeNode constructMaximumBinaryTree(int[] nums) {
    return build(nums, 0, nums.length-1);
}
/* 将 nums[lo..hi] 构造成符合条件的树，返回根节点 */
TreeNode build(int[] nums, int lo, int hi) {
    // base case
    if(lo > hi) return null;
    // 找到数组中的最大值和对应的索引
    int index = lo;
    for(int i = lo; i <= hi; i++) {
        if (nums[index] < nums[i]) index = i;
    }
    TreeNode root = new TreeNode(nums[index]);
    // 递归调用构造左右子树
    root.left = build(nums, lo, index - 1);
    root.right = build(nums, index + 1, hi);
    return root;
}
```

---

### 通过前序和中序/后序和中序遍历结果构造二叉树(kong)

105.从前序与中序遍历序列构造二叉树（中等）

106.从中序与后序遍历序列构造二叉树（中等）

---

### 寻找重复子树(kong)


 652 题「寻找重复子树」



---

## 层序遍历框架


---

### 二叉树max层级遍历 用Queue和q.size去遍历左右

[104. Maximum Depth of Binary Tree](https://leetcode.com/problems/maximum-depth-of-binary-tree/)
- Given the root of a binary tree, return its maximum depth.
- A binary tree's maximum depth is the number of nodes along the longest path from the root node down to the farthest leaf node.

```java
// 输入一棵二叉树的根节点，层序遍历这棵二叉树
void levelTraverse(TreeNode root) {
    if (root == null) return 0;
    Queue<TreeNode> q = new LinkedList<>();
    q.offer(root);
    int depth = 0;
    // 从上到下遍历二叉树的每一层
    while (!q.isEmpty()) {
        int sz = q.size();
        // 从左到右遍历每一层的每个节点
        for (int i = 0; i < sz; i++) {
            TreeNode cur = q.poll();
            // 将下一层节点放入队列
            if (cur.left != null) q.offer(cur.left);
            if (cur.right != null) q.offer(cur.right);
        }
        depth++;
    }
}
```



```java
class State {
    // 记录 node 节点的深度
    int depth;
    TreeNode node;
    State(TreeNode node, int depth) {
        this.depth = depth;
        this.node = node;
    }
}

// 输入一棵二叉树的根节点，遍历这棵二叉树所有节点
void levelTraverse(TreeNode root) {
    if (root == null) return 0;
    Queue<State> q = new LinkedList<>();
    q.offer(new State(root, 1));

    // 遍历二叉树的每一个节点
    while (!q.isEmpty()) {
        State cur = q.poll();
        TreeNode cur_node = cur.node;
        int cur_depth = cur.depth;
        // 将子节点放入队列
        if (cur_node.left != null) q.offer(new State(cur_node.left, cur_depth + 1));
        if (cur_node.right != null) q.offer(new State(cur_node.right, cur_depth + 1));
    }
}
```



---

### 多叉树的层序遍历框架  用Queue和q.size去遍历child

[559. Maximum Depth of N-ary Tree](https://leetcode.com/problems/maximum-depth-of-n-ary-tree/)
- Given a n-ary tree, find its maximum depth.
- The maximum depth is the number of nodes along the longest path from the root node down to the farthest leaf node.
- Nary-Tree input serialization is represented in their level order traversal, each group of children is separated by the null value (See examples).


```java
// Runtime: 1 ms, faster than 55.15% of Java online submissions for Maximum Depth of N-ary Tree.
// Memory Usage: 39.3 MB, less than 55.15% of Java online submissions for Maximum Depth of N-ary Tree.
// 输入一棵多叉树的根节点，层序遍历这棵多叉树
void levelTraverse(TreeNode root) {
    if (root == null) return 0;
    Queue<TreeNode> q = new LinkedList<>();
    q.offer(root);
    int depth = 0;
    // 从上到下遍历多叉树的每一层
    while (!q.isEmpty()) {
        int sz = q.size();
        // 从左到右遍历每一层的每个节点
        for (int i = 0; i < sz; i++) {
            TreeNode cur = q.poll();
            // 将下一层节点放入队列
            for (TreeNode child : cur.children) q.offer(child);
        }
        depth++;
    }
}
```

---

## BFS（广度优先搜索）用Queue和q.size去遍历child + not visited

BFS 找到的路径一定是最短的，但代价就是空间复杂度可能比 DFS 大很多

BFS 的核心数据结构；
- cur.adj() 泛指 cur 相邻的节点，比如说二维数组中，cur 上下左右四面的位置就是相邻节点；
- visited 的主要作用是防止走回头路，大部分时候都是必须的，但是像一般的二叉树结构，没有子节点到父节点的指针，不会走回头路就不需要 visited。



```java
// 输入起点，进行 BFS 搜索
int BFS(Node start) {
    Queue<Node> q;     // 核心数据结构
    Set<Node> visited; // 避免走回头路

    q.offer(start);    // 将起点加入队列
    visited.add(start);
    int step = 0; // 记录搜索的步数

    while (!q.isEmpty()) {
        int sz = q.size();
        /* 将当前队列中的所有节点向四周扩散一步 */
        for (int i = 0; i < sz; i++) {
            Node cur = q.poll();
            /* 将 cur 的相邻节点加入队列 */
            for (Node x : cur.adj()) {
                if (x not in visited) {
                    q.offer(x);
                    visited.add(x);
                }
            }
        }
        step++;
    }
}
```

---

### 111. Minimum Depth of Binary Tree 二叉树min层级遍历 `用Queue和q.size去遍历左右`


[111. Minimum Depth of Binary Tree](https://leetcode.com/problems/minimum-depth-of-binary-tree/)
- Given a binary tree, find its minimum depth.
- The minimum depth is the number of nodes along the shortest path from the root node down to the nearest leaf node.
- Note: A leaf is a node with no children.

```java
// Runtime: 0 ms, faster than 100.00% of Java online submissions for Minimum Depth of Binary Tree.
// Memory Usage: 59.3 MB, less than 87.89% of Java online submissions for Minimum Depth of Binary Tree.
int minDepth(TreeNode root) {
    if (root == null) return 0;
    Queue<TreeNode> q = new LinkedList<>();
    q.offer(root);
    int depth = 0;
    while (!q.isEmpty()) {
        int sz = q.size();
        /* 将当前队列中的所有节点向四周扩散 */
        for (int i = 0; i < sz; i++) {
            TreeNode cur = q.poll();
            /* 判断是否到达终点 */
            if (cur.left == null && cur.right == null) return depth+1;
            /* 将 cur 的相邻节点加入队列 */
            if (cur.left != null) q.offer(cur.left);
            if (cur.right != null) q.offer(cur.right);
        }
        /* 这里增加步数 */
        depth++;
    }
    return depth;
}
```


---

### 穷举所有可能的密码组合 用Queue和q.size去遍历all

如果你只转一下锁，有几种可能？总共有 4 个位置，每个位置可以向上转，也可以向下转，也就是有 8 种可能对吧。

比如说从 "0000" 开始，转一次，可以穷举出 "1000", "9000", "0100", "0900"... 共 8 种密码。然后，再以这 8 种密码作为基础，对每个密码再转一下，穷举出所有可能…

仔细想想，这就可以抽象成一幅图，每个节点有 8 个相邻的节点，又让你求最短距离，这不就是典型的 BFS 嘛，框架就可以派上用场了，先写出一个「简陋」的 BFS 框架代码再说别的：

1、会走回头路。比如说我们从 "0000" 拨到 "1000"，但是等从队列拿出 "1000" 时，还会拨出一个 "0000"，这样的话会产生死循环。

2、没有终止条件，按照题目要求，我们找到 target 就应该结束并返回拨动的次数。

3、没有对 deadends 的处理，按道理这些「死亡密码」是不能出现的，也就是说你遇到这些密码的时候需要跳过。


```java
// 将 s[j] 向上拨动一次
String plusOne(String s, int j) {
    char[] ch = s.toCharArray();
    if (ch[j] == '9') ch[j] = '0';
    else ch[j] += 1;
    return new String(ch);
}
// 将 s[i] 向下拨动一次
String minusOne(String s, int j) {
    char[] ch = s.toCharArray();
    if (ch[j] == '0') ch[j] = '9';
    else ch[j] -= 1;
    return new String(ch);
}

// BFS 框架，打印出所有可能的密码
void BFS(String target) {
    Queue<String> q = new LinkedList<>();
    q.offer("0000");
    while (!q.isEmpty()) {
        int sz = q.size();
        /* 将当前队列中的所有节点向周围扩散 */
        for (int i = 0; i < sz; i++) {
            String cur = q.poll();
            /* 判断是否到达终点 */
            System.out.println(cur);
            /* 将一个节点的相邻节点加入队列 */
            for (int j = 0; j < 4; j++) {
                String up = plusOne(cur, j);
                String down = minusOne(cur, j);
                q.offer(up);
                q.offer(down);
            }
        }
        /* 在这里增加步数 */
    }
    return;
}
```


---


---



## 二叉搜索树


```java
void BST(TreeNode root, int target) {
    if (root.val == target)
        // 找到目标，做点什么
    if (root.val < target) BST(root.right, target);
    if (root.val > target) BST(root.left, target);
}
```


---

### 判断 BST 的合法性

[98. Validate Binary Search Tree](https://leetcode.com/problems/validate-binary-search-tree/)
- Given the root of a binary tree, determine if it is a valid binary search tree (BST).
- A valid BST is defined as follows:
- The left subtree of a node contains only nodes with keys less than the node's key.
- The right subtree of a node contains only nodes with keys greater than the node's key.
- Both the left and right subtrees must also be binary search trees.

```java
// Runtime: 0 ms, faster than 100.00% of Java online submissions for Validate Binary Search Tree.
// Memory Usage: 38.4 MB, less than 92.75% of Java online submissions for Validate Binary Search Tree.

boolean isValidBST(TreeNode root) {
    return checkBST(root, null, null);
}

/* 限定以 root 为根的子树节点必须满足 max.val > root.val > min.val */
boolean checkBST(TreeNode root, TreeNode min, TreeNode max) {
    // base case
    if (root == null) return true;
    // 若 root.val 不符合 max 和 min 的限制，说明不是合法 BST
    if (min!=null && root.val<=min.val) return false;
    if (max!=null && root.val>=max.val) return false;
    // 限定左子树的最大值是 root.val，右子树的最小值是 root.val
    return checkBST(root.left, min, root) && checkBST(root.right, root, max);
}
```

---


### 在 BST 中搜索元素

```java
// 穷举了所有节点，适用于所有普通二叉树
TreeNode searchBST(TreeNode root, int target);
    if (root == null) return null;
    if (root.val == target) return root;
    // 当前节点没找到就递归地去左右子树寻找
    TreeNode left = searchBST(root.left, target);
    TreeNode right = searchBST(root.right, target);
    return left != null ? left : right;
}

TreeNode searchBST(TreeNode root, int target) {
    if (root == null) return null;
    // 去左子树搜索
    if (root.val > target) return searchBST(root.left, target);
    // 去右子树搜索
    if (root.val < target) return searchBST(root.right, target);
    return root;
}
```

---

### 在 BST 中插入一个数

[701. Insert into a Binary Search Tree](https://leetcode.com/problems/insert-into-a-binary-search-tree/)
- You are given the root node of a binary search tree (BST) and a value to insert into the tree. Return the root node of the BST after the insertion. It is guaranteed that the new value does not exist in the original BST.
- Notice that there may exist multiple valid ways for the insertion, as long as the tree remains a BST after insertion. You can return any of them.

```java
// Runtime: 0 ms, faster than 100.00% of Java online submissions for Insert into a Binary Search Tree.
// Memory Usage: 39.7 MB, less than 66.92% of Java online submissions for Insert into a Binary Search Tree.
TreeNode insertIntoBST(TreeNode root, int val) {
    // 找到空位置插入新节点
    if (root == null) return new TreeNode(val);
    // if (root.val == val)
    //     BST 中一般不会插入已存在元素
    if (root.val < val) root.right = insertIntoBST(root.right, val);
    if (root.val > val) root.left = insertIntoBST(root.left, val);
    return root;
}
```

---

### 在 BST 中删除一个数

[450. Delete Node in a BST](https://leetcode.com/problems/delete-node-in-a-bst/)
- Given a root node reference of a BST and a key, delete the node with the given key in the BST. Return the root node reference (possibly updated) of the BST.
- Basically, the deletion can be divided into two stages:
- Search for a node to remove.
- If the node is found, delete the node.

```java
// Runtime: 0 ms, faster than 100.00% of Java online submissions for Delete Node in a BST.
// Memory Usage: 39 MB, less than 97.99% of Java online submissions for Delete Node in a BST.
TreeNode deleteNode(TreeNode root, int key) {
    if (root == null) return null;
    if (root.val == key) {
        // 这两个 if 把情况 1 和 2 都正确处理了
        if (root.left == null) return root.right;
        if (root.right == null) return root.left;
        // 处理情况 3
        // 找到右子树的最小节点
        TreeNode minNode = getMin(root.right);
        // 把 root 改成 minNode
        root.val = minNode.val;
        // 转而去删除 minNode
        root.right = deleteNode(root.right, minNode.val);
    }
    else if (root.val > key) root.left = deleteNode(root.left, key);
    else if (root.val < key) root.right = deleteNode(root.right, key);
    return root;
}

TreeNode getMin(TreeNode node) {
    // BST 最左边的就是最小的
    while (node.left != null) node = node.left;
    return node;
}
```


---


### 不同的二叉搜索树 - 穷举问题

[96. Unique Binary Search Trees](https://leetcode.com/problems/unique-binary-search-trees/)
- Given an integer n, return the number of structurally unique BST's (binary search trees) which has exactly n nodes of unique values from 1 to n.
- 给你输入一个正整数 n，存储 `{1,2,3...,n}` 这些值共有有多少种不同的 BST 结构。

```java
// Runtime Error
// /* 主函数 */
// int numTrees(int n) {
//     // 计算闭区间 [1, n] 组成的 BST 个数
//     return count(1, n);
// }

// /* 计算闭区间 [lo, hi] 组成的 BST 个数 */
// int count(int lo, int hi) {
//     // base case
//     if (lo > hi) return 1;
//     int res = 0;
//     for (int i = lo; i <= hi; i++) {
//         // i 的值作为根节点 root
//         int left = count(lo, i - 1);
//         int right = count(i + 1, hi);
//         // 左右子树的组合数乘积是 BST 的总数
//         res += left * right;
//     }
//     return res;
// }
```


```java
// Runtime: 0 ms, faster than 100.00% of Java online submissions for Unique Binary Search Trees.
// Memory Usage: 35.3 MB, less than 99.03% of Java online submissions for Unique Binary Search Trees.

// 备忘录
int[][] memo;

int numTrees(int n) {
    // 备忘录的值初始化为 0
    memo = new int[n + 1][n + 1];
    return count(1, n);
}

int count(int lo, int hi) {
    if (lo > hi) return 1;
    // 查备忘录
    if (memo[lo][hi] != 0) return memo[lo][hi];
    int res = 0;
    for (int mid = lo; mid <= hi; mid++) {
        int left = count(lo, mid - 1);
        int right = count(mid + 1, hi);
        res += left * right;
    }
    // 将结果存入备忘录
    memo[lo][hi] = res;
    return res;
}
```



---

### 不同的二叉搜索树II

95.不同的二叉搜索树II（Medium）
- 不止计算有几个不同的 BST，而是要你构建出所有合法的 BST

```java
/* 主函数 */
List<TreeNode> generateTrees(int n) {
    if (n == 0) return new LinkedList<>();
    // 构造闭区间 [1, n] 组成的 BST
    return build(1, n);
}

/* 构造闭区间 [lo, hi] 组成的 BST */
List<TreeNode> build(int lo, int hi) {
    List<TreeNode> res = new LinkedList<>();
    // base case
    if (lo > hi) {
        res.add(null);
        return res;
    }
    // 1、穷举 root 节点的所有可能。
    for (int i = lo; i <= hi; i++) {
        // 2、递归构造出左右子树的所有合法 BST。
        List<TreeNode> leftTree = build(lo, i - 1);
        List<TreeNode> rightTree = build(i + 1, hi);
        // 3、给 root 节点穷举所有左右子树的组合。
        for (TreeNode left : leftTree) {
            for (TreeNode right : rightTree) {
                // i 作为根节点 root 的值
                TreeNode root = new TreeNode(i);
                root.left = left;
                root.right = right;
                res.add(root);
            }
        }
    }
    return res;
}
```

---

### 二叉树后序遍历

后序遍历的代码框架：
- 如果当前节点要做的事情需要通过左右子树的计算结果推导出来，就要用到后序遍历。


```java
void traverse(TreeNode root) {
    traverse(root.left);
    traverse(root.right);
    /* 后序遍历代码的位置 */
    /* 在这里处理当前节点 */
}
```

[1373. Maximum Sum BST in Binary Tree](https://leetcode.com/problems/maximum-sum-bst-in-binary-tree/)
- Given a binary tree root, return the maximum sum of all keys of any sub-tree which is also a Binary Search Tree (BST).
- Assume a BST is defined as follows:
- The left subtree of a node contains only nodes with keys less than the node's key.
- The right subtree of a node contains only nodes with keys greater than the node's key.
- Both the left and right subtrees must also be binary search trees.
- 1、我肯定得知道左右子树是不是合法的 BST，如果这俩儿子有一个不是 BST，以我为根的这棵树肯定不会是 BST，对吧。
- 2、如果左右子树都是合法的 BST，我得瞅瞅左右子树加上自己还是不是合法的 BST 了。因为按照 BST 的定义，当前节点的值应该大于左子树的最大值，小于右子树的最小值，否则就破坏了 BST 的性质。
- 3、因为题目要计算最大的节点之和，如果左右子树加上我自己还是一棵合法的 BST，也就是说以我为根的整棵树是一棵 BST，那我需要知道我们这棵 BST 的所有节点值之和是多少，方便和别的 BST 争个高下，对吧。

根据以上三点，站在当前节点的视角，需要知道以下具体信息：
- 1、左右子树是否是 BST。
- 2、左子树的最大值和右子树的最小值。
- 3、左右子树的节点值之和。


```java
// 全局变量，记录 BST 最大节点之和
int maxSum = 0;

/* 主函数 */
public int maxSumBST(TreeNode root) {
    traverse(root);
    return maxSum;
}

// 函数返回 int[]{ isBST, min, max, sum}
int[] traverse(TreeNode root) {
    // base case
    if (root == null) return new int[] {1, Integer.MAX_VALUE, Integer.MIN_VALUE, 0};
    // 递归计算左右子树
    int[] left = traverse(root.left);
    int[] right = traverse(root.right);

    /******* 后序遍历位置 *******/
    int[] res = new int[4];
    // 这个 if 在判断以 root 为根的二叉树是不是 BST
    if (left[0] == 1 && right[0] == 1 &&
        root.val > left[2] && root.val < right[1]) {
        // 以 root 为根的二叉树是 BST
        res[0] = 1;
        // 计算以 root 为根的这棵 BST 的最小值
        res[1] = Math.min(left[1], root.val);
        // 计算以 root 为根的这棵 BST 的最大值
        res[2] = Math.max(right[2], root.val);
        // 计算以 root 为根的这棵 BST 所有节点之和
        res[3] = left[3] + right[3] + root.val;
        // 更新全局变量
        maxSum = Math.max(maxSum, res[3]);
    } else {
        // 以 root 为根的二叉树不是 BST
        res[0] = 0;
        // 其他的值都没必要计算了，因为用不到
    }
    return res;
}
```


---

### 二叉树的序列化与反序列化

二叉树的遍历方式有哪些？递归遍历方式有
- 前序遍历，中序遍历，后序遍历；
- 迭代方式一般是层级遍历。



[297. Serialize and Deserialize Binary Tree](https://leetcode.com/problems/serialize-and-deserialize-binary-tree/)
- Serialization is the process of converting a data structure or object into a sequence of bits so that it can be stored in a file or memory buffer, or transmitted across a network connection link to be reconstructed later in the same or another computer environment.
- Design an algorithm to serialize and deserialize a binary tree. There is no restriction on how your serialization/deserialization algorithm should work. You just need to ensure that a binary tree can be serialized to a string and this string can be deserialized to the original tree structure.
- Clarification: The input/output format is the same as how LeetCode serializes a binary tree. You do not necessarily need to follow this format, so please be creative and come up with different approaches yourself.


```java
public class Codec {
    // 把一棵二叉树序列化成字符串
    public String serialize(TreeNode root) {}
    // 把字符串反序列化成二叉树
    public TreeNode deserialize(String data) {}
}
```

```java
LinkedList<Integer> res;
void traverse(TreeNode root) {
    if (root == null) {
        // 暂且用数字 -1 代表空指针 null
        res.addLast(-1);
        return;
    }
    /****** 前序遍历位置 ******/
    res.addLast(root.val);
    /***********************/
    traverse(root.left);
    traverse(root.right);
}
```


---

### 二叉树打平到一个字符串


```java
String SEP = ',';
String NULL = '#';

/* 主函数，将二叉树序列化为字符串 */
String serialize(TreeNode root) {
    StringBuilder sb = new StringBuilder();
    serialize(root, sb);
    return sb.toString();
}
void serialize(TreeNode root, StringBuilder sb){
    if(root==null){
        sb.append(Null).append(SEP);
        return;
    }
    sb.append(root.val).append(SEP);
    traverse(root.left, sb);
    traverse(root.right, sb);
}
```

---

# Binary Heap 二叉堆

- 其主要操作就两个，sink（下沉）和 swim（上浮），用以维护二叉堆的性质。
- 其主要应用有两个，
  - 首先是一种排序方法「堆排序」，
  - 第二是一种很有用的数据结构「优先级队列」。

![1](https://i.imgur.com/vStOOwC.png)

因为这棵二叉树是「完全二叉树」，所以把 arr[1] 作为整棵树的根的话，每个节点的父节点和左右孩子的索引都可以通过简单的运算得到，这就是二叉堆设计的一个巧妙之处。

```java
// 父节点的索引
int parent(int root) {return root / 2;}

// 左孩子的索引
int left(int root) {return root * 2;}

// 右孩子的索引
int right(int root) {return root * 2 + 1;}
```


## 最大堆和最小堆

二叉堆还分为最大堆和最小堆。
- 最大堆的性质是：每个节点都大于等于它的两个子节点。
- 最小堆的性质是：每个节点都小于等于它的子节点。

优先级队列 数据结构
- 插入或者删除元素的时候，元素会自动排序
- 这底层的原理就是二叉堆的操作。

```java
public class MaxPQ
    <Key extends Comparable<Key>> {

    private Key[] pq;    // 存储元素的数组
    private int N = 0;   // 当前 Priority Queue 中的元素个数

    public MaxPQ(int cap) {
        // 索引 0 不用，所以多分配一个空间
        pq = (Key[]) new Comparable[cap + 1];
    }

    /* 返回当前队列中最大元素 */
    public Key max() {
        return pq[1];
    }

    /* 插入元素 e */
    public void insert(Key e) {
        N++;
        pq[N] = e;
        swim(e);
    }

    /* 删除并返回当前队列中最大元素 */
    public Key delMax() {
        exch(pq[1],pq[N]);
        pq[N] = null;
        N--;
        sink(pq[1]);
        Key max = pq[1];
        return max;
    }

    /* 上浮第 k 个元素，以维护最大堆性质 */
    private void swim(int k) {
        while(k>1 && less(parent(k),k)) exch(parent(k),k);
        k=parent(k);
    }

    /* 下沉第 k 个元素，以维护最大堆性质 */
    private void sink(int k) {
        while(left(k)<=N){
            int bigger = left(k);
            if(right(k)<=N && less(bigger, right(k))) bigger = right(k);
            if(less(bigger, k)) break;
            exch(bigger,k);
            k=bigger;
        }
    }

    /* 交换数组的两个元素 */
    private void exch(int i, int j) {
        Key temp = pq[i];
        pq[i] = pq[j];
        pq[j] = temp;
    }

    /* pq[i] 是否比 pq[j] 小？ */
    private boolean less(int i, int j) {
        return pq[i].compareTo(pq[j]) < 0;
    }

    /* 还有 left, right, parent 三个方法 */
}
```



---

# Graphy



邻接表
- 把每个节点 x 的邻居都存到一个列表里，
- 然后把 x 和这个列表关联起来，
- 这样就可以通过一个节点 x 找到它的所有相邻节点。

邻接矩阵
- 二维布尔数组，我们权且成为 matrix
- 如果节点 x 和 y 是相连的，那么就把 matrix[x][y] 设为 true（上图中绿色的方格代表 true）。
- 如果想找节点 x 的邻居，去扫一圈 matrix[x][..] 就行了。

有向加权图
- 如果是邻接表，我们不仅仅存储某个节点 x 的所有邻居节点，还存储 x 到每个邻居的权重
- 如果是邻接矩阵，matrix[x][y] 不再是布尔值，而是一个 int 值，0 表示没有连接，其他值表示权重

无向图
- 所谓的「无向」，是不是等同于「双向」？
- 如果连接无向图中的节点 x 和 y，把 matrix[x][y] 和 matrix[y][x] 都变成 true 不就行了；邻接表也是类似的操作。


图和多叉树最大的区别是，图是可能包含环的，
- 你从图的某一个节点开始遍历，有可能走了一圈又回到这个节点。
- 所以，如果图包含环，遍历框架就要一个 visited 数组进行辅助：


---


---

## 图的遍历


```java
boolean[] visited;

/* 图遍历框架 */
void traverse(Graph graph, int s) {
    if (visited[s]) return;
    // 经过节点 s
    visited[s] = true;
    for (int neighbor : graph.neighbors(s))
        traverse(graph, neighbor);
    // 离开节点 s
    visited[s] = false;
}
```

---







---

### 转换成图

图的两种存储形式
- 邻接矩阵
- 和邻接表。


```java
// 邻接表
// [ [1,0], [0,1] ]
// 节点编号分别是 0, 1, ..., numCourses-1
List<Integer>[] buildGraph(int numCourses, int[][] prerequisites) {
    // 图中共有 numCourses 个节点
    // create graph
    List<Integer>[] graph = new LinkedList[numCourses];
    for (int i = 0; i < numCourses; i++) {
        graph[i] = new LinkedList<>();
    }
    for (int[] edge : prerequisites) {
        int from = edge[1];
        int to = edge[0];
        // 修完课程 from 才能修课程 to
        // 在图中添加一条从 from 指向 to 的有向边
        graph[from].add(to);
    }
    return graph;
}
```



---

### 所有可能路径

797.所有可能的路径（中等）


```java
// 记录所有路径
List<List<Integer>> res = new LinkedList<>();

public List<List<Integer>> allPathsSourceTarget(int[][] graph) {
    LinkedList<Integer> path = new LinkedList<>();
    traverse(graph, 0, path);
    return res;
}

/* 图的遍历框架 */
void traverse(int[][] graph, int s, LinkedList<Integer> path) {
    // 添加节点 s 到路径
    path.addLast(s);
    int n = graph.length;
    if (s == n - 1) {
        // 到达终点
        res.add(new LinkedList<>(path));
        path.removeLast();
        return;
    }
    // 递归每个相邻节点
    for (int v : graph[s]) {
        traverse(graph, v, path);
    }
    // 从路径移出节点 s
    path.removeLast();
}
```



```java
// Runtime: 7 ms, faster than 26.00% of Java online submissions for All Paths From Source to Target.
// Memory Usage: 41.5 MB, less than 47.72% of Java online submissions for All Paths From Source to Target.
class Solution {
    public List<List<Integer>> allPathsSourceTarget(int[][] graph) {
        // if null, return null
        if(graph.length == 0 || graph[0].length==0) return new ArrayList<>();
        // start
        List<List<Integer>> res = new ArrayList<>();
        bfscheckPath(graph, 0, res);
        return res;
    }
    public void bfscheckPath(int[][] graph, int start, List<List<Integer>> res) {
        // start the que
        Queue<List<Integer>> que = new ArrayDeque<>();
        que.add(Arrays.asList(start));
        while(!que.isEmpty()){
            // start the path
            List<Integer> path = que.poll();
            // path last node
            int end = path.get(path.size()-1);
            if(end == graph.length-1) {
                res.add(path);
                continue;
            }
            for(int node : graph[end]) {
                List<Integer> list = new ArrayList<>(path);
                list.add(node);
                que.add(list);
            }
        }
    }
}
```


---

### 判断有向图是否存在环

有向图的环检测、拓扑排序算法。

看到依赖问题，首先想到的就是把问题转化成「有向图」这种数据结构
- 只要图中存在环，那就说明存在循环依赖。

[207 题「课程表」207. Course Schedule](https://leetcode.com/problems/course-schedule/)
- 只要会遍历，就可以判断图中是否存在环了。
- There are a total of numCourses courses you have to take, labeled from 0 to numCourses - 1. You are given an array prerequisites where `prerequisites[i] = [ai, bi]` indicates that you must take course bi first if you want to take course ai.
- For example, the pair [0, 1], indicates that to take course 0 you have to first take course 1.
Return true if you can finish all courses. Otherwise, return false.



DFS 算法遍历图的框架
- 无非就是从多叉树遍历框架扩展出来的，加了个 visited 数组罢了：

```java
// Runtime: 2 ms, faster than 99.48% of Java online submissions for Course Schedule.
// Memory Usage: 40.3 MB, less than 46.35% of Java online submissions for Course Schedule.

// 防止重复遍历同一个节点
boolean hasCycle = false;
boolean[] onPath, visited;
List<Integer>[] buildGraph(int numCourses, int[][] prerequisites) {
    // 图中共有 numCourses 个节点
    List<Integer>[] graph = new LinkedList[numCourses];
    // create graph edge first
    for (int i = 0; i < numCourses; i++) graph[i] = new LinkedList<>();
    // check edge
    for (int[] edge : prerequisites) {
        int from = edge[1];
        int to = edge[0];
        // 修完课程 from 才能修课程 to
        // 在图中添加一条从 from 指向 to 的有向边
        graph[from].add(to);
    }
    return graph;
}
boolean canFinish(int numCourses, int[][] prerequisites) {
    List<Integer>[] graph = buildGraph(numCourses, prerequisites);
    visited = new boolean[numCourses];
    onPath = new boolean[numCourses];
    for (int i = 0; i < numCourses; i++) traverse(graph, i);
    return !hasCycle;
}
void traverse(List<Integer>[] graph, int s) {
    // 发现环！！
    if (onPath[s]) hasCycle = true;
    if (visited[s]) return;
    /* 前序遍历代码位置 */
    // 将当前节点标记为已遍历
    visited[s] = true;
    onPath[s] = true;
    for (int t : graph[s]) traverse(graph, t);
    /* 后序遍历代码位置 */
    onPath[s] = false;
}
```

---

### 拓扑排序

拓扑排序的结果就是反转之后的后序遍历结果

- 如果把课程抽象成节点，课程之间的依赖关系抽象成有向边，
- 那么这幅图的拓扑排序结果就是上课顺序。
- 先判断一下题目输入的课程依赖是否成环，成环的话是无法进行拓扑排序的，复用上一道题的主函数


[210. Course Schedule II](https://leetcode.com/problems/course-schedule-ii/)
- There are a total of numCourses courses you have to take, labeled from 0 to numCourses - 1. You are given an array prerequisites where `prerequisites[i] = [ai, bi]` indicates that you must take course bi first if you want to take course ai.
- For example, the pair [0, 1], indicates that to take course 0 you have to first take course 1.
- Return the ordering of courses you should take to finish all courses.
- If there are many valid answers, return any of them.
- If it is impossible to finish all courses, return an empty array.


```java
// Runtime: 3 ms, faster than 96.43% of Java online submissions for Course Schedule II.
// Memory Usage: 40.7 MB, less than 49.39% of Java online submissions for Course Schedule II.

// 记录后序遍历结果
List<Integer> postorder = new ArrayList<>();
// 记录是否存在环
boolean hasCycle = false;
// 防止重复遍历同一个节点
boolean[] onPath, visited;

List<Integer>[] buildGraph(int numCourses, int[][] prerequisites) {
    // 图中共有 numCourses 个节点
    List<Integer>[] graph = new LinkedList[numCourses];
    // create graph edge first
    for (int i = 0; i < numCourses; i++) graph[i] = new LinkedList<>();
    // check edge
    for (int[] edge : prerequisites) {
        int from = edge[1];
        int to = edge[0];
        // 修完课程 from 才能修课程 to
        // 在图中添加一条从 from 指向 to 的有向边
        graph[from].add(to);
    }
    return graph;
}
// 主函数
public int[] findOrder(int numCourses, int[][] prerequisites) {
    List<Integer>[] graph = buildGraph(numCourses, prerequisites);
    visited = new boolean[numCourses];
    onPath = new boolean[numCourses];
    for (int i = 0; i < numCourses; i++) traverse(graph, i);
    // 有环图无法进行拓扑排序
    if (hasCycle) return new int[]{};

    // 逆后序遍历结果即为拓扑排序结果
    int[] res = new int[numCourses];
    Collections.reverse(postorder);
    for (int i = 0; i < numCourses; i++) {
        res[i] = postorder.get(i);
    }
    return res;
}

void traverse(List<Integer>[] graph, int s) {
    // 发现环！！
    if (onPath[s]) hasCycle = true;
    if (visited[s]|| hasCycle) return;
    /* 前序遍历代码位置 */
    // 将当前节点标记为已遍历
    visited[s] = true;
    onPath[s] = true;
    // 前序遍历位置
    for (int t : graph[s]) traverse(graph, t);
    // 后序遍历位置
    postorder.add(s);

    onPath[s] = false;
}
```

---


## 搜索名人

277.搜索名人（中等）
- 给你 n 个人的社交关系（你知道任意两个人之间是否认识），然后请你找出这些人中的「名人」。
- 所谓「名人」有两个条件：
  - 、所有其他人都认识「名人」。
  - 、「名人」不认识任何其他人。


把名流问题描述成算法的形式就是这样的：
- 给你输入一个大小为 n x n 的二维数组（邻接矩阵） graph 表示一幅有 n 个节点的图，每个人都是图中的一个节点，编号为 0 到 n - 1。
- 如果 graph[i][j] == 1 代表第 i 个人认识第 j 个人，如果 graph[i][j] == 0 代表第 i 个人不认识第 j 个人。
- 有了这幅图表示人与人之间的关系，请你计算，这 n 个人中，是否存在「名人」？
- 如果存在，算法返回这个名人的编号，如果不存在，算法返回 -1。

---

### 暴力解法

```java
int findCelebrity(int n) {
    for (int cand = 0; cand < n; cand++) {
        int other;
        for (other = 0; other < n; other++) {
            if (cand == other) continue;
            // 保证其他人都认识 cand，且 cand 不认识任何其他人
            // 否则 cand 就不可能是名人
            if (knows(cand, other) || !knows(other, cand)) break;
        }
        if (other == n) {
            // 找到名人
            return cand;
        }
    }
    // 没有一个人符合名人特性
    return -1;
}
```

---

### 优化解法

我再重复一遍所谓「名人」的定义：
- 1、所有其他人都认识名人
- 2、名人不认识任何其他人。
- 这个定义就很有意思，它保证了人群中最多有一个名人。
- 这很好理解，如果有两个人同时是名人，那么这两条定义就自相矛盾了。
- 只要观察任意两个候选人的关系，我一定能确定其中的一个人不是名人，把他排除。


逐一分析每种情况，看看怎么排除掉一个人。
- 对于情况一，cand 认识 other，所以 cand 肯定不是名人，排除。因为名人不可能认识别人。
- 对于情况二，other 认识 cand，所以 other 肯定不是名人，排除。
- 对于情况三，他俩互相认识，肯定都不是名人，可以随便排除一个。
- 对于情况四，他俩互不认识，肯定都不是名人，可以随便排除一个。因为名人应该被所有其他人认识。
- 我们可以不断从候选人中选两个出来，然后排除掉一个，直到最后只剩下一个候选人，这时候再使用一个 for 循环判断这个候选人是否是货真价实的「名人」。
- 避免了嵌套 for 循环，时间复杂度降为 O(N) 了，
- 不过引入了一个队列来存储候选人集合，使用了 O(N) 的空间复杂度。

```java
int findCelebrity(int n) {
    if (n == 1) return 0;
    // 将所有候选人装进队列
    LinkedList<Integer> q = new LinkedList<>();
    for (int i = 0; i < n; i++) q.addLast(i);
    // 一直排除，直到只剩下一个候选人停止循环
    while (q.size() >= 2) {
        // 每次取出两个候选人，排除一个
        int cand = q.removeFirst();
        int other = q.removeFirst();
        // cand 不可能是名人，排除，让 other 归队
        if (knows(cand, other) || !knows(other, cand)) q.addFirst(other);
        // other 不可能是名人，排除，让 cand 归队
        else q.addFirst(cand);
    }

    // 现在排除得只剩一个候选人，判断他是否真的是名人
    int cand = q.removeFirst();
    for (int other = 0; other < n; other++) {
        if (other == cand) continue;
        // 保证其他人都认识 cand，且 cand 不认识任何其他人
        if (!knows(other, cand) || knows(cand, other)) return -1;
    }
    // cand 是名人
    return cand;
}
```


---

### 最终解法

时间复杂度为 O(N)，空间复杂度为 O(1)

```java
int findCelebrity(int n) {
    // 先假设 cand 是名人
    int cand = 0;
    for (int other = 1; other < n; other++) {
        // if other x-> cand or cand->other
        // cand 不可能是名人，排除
        // 假设 other 是名人
        if (!knows(other, cand) || knows(cand, other)) cand = other;
        // other 不可能是名人，排除
        // 什么都不用做，继续假设 cand 是名人 下一个other
        else {}
    }
    // 现在的 cand 是排除的最后结果，但不能保证一定是名人
    for (int other = 0; other < n; other++) {
        if (cand == other) continue;
        // 需要保证其他人都认识 cand，且 cand 不认识任何其他人
        if (!knows(other, cand) || knows(cand, other)) return -1;
    }
    return cand;
}
```


---

## UNION-FIND 并查集算法 计算 连通分量

---

### UNION-FIND算法

动态连通性
- 抽象成给一幅图连线。
- 比如总共有 10 个节点，他们互不相连，分别用 0~9 标记：


Union-Find 算法主要需要实现这两个 API：

```java
class UF {
    /* 将 p 和 q 连接 */
    public void union(int p, int q);
    /* 判断 p 和 q 是否连通 */
    public boolean connected(int p, int q);
    /* 返回图中有多少个连通分量 */
    public int count();
}
```

「连通」是一种等价关系，也就是说具有如下三个性质：
- 1、自反性：节点 p 和 p 是连通的。
- 2、对称性：如果节点 p 和 q 连通，那么 q 和 p 也连通。
- 3、传递性：如果节点 p 和 q 连通，q 和 r 连通，那么 p 和 r 也连通。
- 比如说之前那幅图，0～9 任意两个不同的点都不连通，调用 connected 都会返回 false，连通分量为 10 个。
- 如果现在调用 union(0, 1)，那么 0 和 1 被连通，连通分量降为 9 个。
- 再调用 union(1, 2)，这时 0,1,2 都被连通，调用 connected(0, 2) 也会返回 true，连通分量变为 8 个。


判断这种「等价关系」非常实用
- 比如说编译器判断同一个变量的不同引用，比如社交网络中的朋友圈计算等等。
- Union-Find 算法的关键就在于 union 和 connected 函数的效率。

算法的关键点有 3 个：
- 1、用 parent 数组记录每个节点的父节点，相当于指向父节点的指针，所以 parent 数组内实际存储着一个森林（若干棵多叉树）。
- 2、用 size 数组记录着每棵树的重量，目的是让 union 后树依然拥有平衡性，而不会退化成链表，影响操作效率。
- 3、在 find 函数中进行路径压缩，保证任意树的高度保持在常数，使得 union 和 connected API 时间复杂度为 O(1)。



---

#### 基本思路

- 设定树的每个节点有一个指针指向其父节点，如果是根节点的话，这个指针指向自己
- 如果某两个节点被连通，则让其中的（任意）一个节点的根节点接到另一个节点的根节点上：


```java
class UF {
    private int count;      // 记录连通分量
    private int[] parent;   // 节点 x 的节点是 parent[x]

    /* 构造函数，n 为图的节点总数 */
    public UF(int n) {
        // 一开始互不连通
        this.count = n;
        // 父节点指针初始指向自己
        parent = new int[n];
        for (int i = 0; i < n; i++) parent[i] = i;
    }

    /* 返回某个节点 x 的根节点 */
    private int find(int x) {
        // 根节点的 parent[x] == x
        while (parent[x] != x) x = parent[x];
        return x;
    }

    /* 返回当前的连通分量个数 */
    public int count() {
        return count;
    }

    /* 其他函数 */
    public void union(int p, int q) {
        int rootP = find(p);
        int rootQ = find(q);
        if (rootP == rootQ) return;
        // 将两棵树合并为一棵
        // 简单粗暴的把 p 所在的树接到 q 所在的树的根节点下面
        parent[rootP] = rootQ;
        // parent[rootQ] = rootP 也一样
        count--; // 两个分量合二为一
    }

    // 如果节点 p 和 q 连通的话，它们一定拥有相同的根节点
    public boolean connected(int p, int q) {
        int rootP = find(p);
        int rootQ = find(q);
        return rootP == rootQ;
    }
}

```


主要 API connected 和 union 中的复杂度都是 find 函数造成的，它们的复杂度和 find 一样。
- find 主要功能就是从某个节点向上遍历到树根，其时间复杂度就是树的高度。
- 我们可能习惯性地认为树的高度就是 logN，但这并不一定。树的高度最坏情况下可能变成 N。

所以说上面这种解法，find , union , connected 的时间复杂度都是 O(N)。


---

#### 平衡性优化

for 小一些的树接到大一些的树下面，这样就能避免头重脚轻，更平衡一些
- 树的高度大致在 logN 这个数量级，极大提升执行效率。
- 此时，find , union , connected 的时间复杂度都下降为 O(logN)，即便数据规模上亿，所需时间也非常少。



```java
class UF {
    private int count;      // 记录连通分量
    private int[] parent;   // 节点 x 的节点是 parent[x]
    private int[] size;     // 新增一个数组记录树的“重量”

    /* 构造函数，n 为图的节点总数 */
    public UF(int n) {
        // 一开始互不连通
        this.count = n;
        // 父节点指针初始指向自己
        parent = new int[n];
        for (int i = 0; i < n; i++) {
            size[i] = 1;      // 记录每棵树包含的节点数
            parent[i] = i;
        }
    }

    /* 返回某个节点 x 的根节点 */
    private int find(int x) {
        // 根节点的 parent[x] == x
        while (parent[x] != x) {
            x = parent[x];
        }
        return x;
    }

    /* 返回当前的连通分量个数 */
    public int count() {
        return count;
    }

    /* 其他函数 */
    public void union(int p, int q) {
        int rootP = find(p);
        int rootQ = find(q);
        if (rootP == rootQ) return;
        // 将两棵树合并为一棵
        // 简单粗暴的把 p 所在的树接到 q 所在的树的根节点下面
        if(size[rootP]>size[rootQ]) {
            parent[rootQ] = rootP;
            size[rootP] += size[rootQ];
        } else {
            parent[rootP] = rootQ;
            size[rootQ] += size[rootP];
        }
        count--; // 两个分量合二为一
    }

    // 如果节点 p 和 q 连通的话，它们一定拥有相同的根节点
    public boolean connected(int p, int q) {
        int rootP = find(p);
        int rootQ = find(q);
        return rootP == rootQ;
    }
}
```

---

#### 路径压缩

进一步压缩每棵树的高度，使树高始终保持为常数
- 这样 find 就能以 O(1) 的时间找到某一节点的根节点，
- 相应的，connected 和 union 复杂度都下降为 O(1)。
- 调用 find 函数每次向树根遍历的同时，顺手将树高缩短了，最终所有树高都不会超过 3（union 的时候树高可能达到 3）。


Union-Find 算法的复杂度可以这样分析：
- 构造函数初始化数据结构需要 O(N) 的时间和空间复杂度；
- 连通两个节点 union、判断两个节点的连通性 connected、计算连通分量 count 所需的时间复杂度均为 O(1)。


![1](https://i.imgur.com/gUVWOv6.jpg)

如果带有重量平衡优化，一定会得到情况一，而不带重量优化，可能出现情况二。
- 高度为 3 时才会触发路径压缩那个 while 循环，
- 所以情况一根本不会触发路径压缩，而情况二会多执行很多次路径压缩，将第三层节点压缩到第二层。


```java
class UF {
    private int count;      // 记录连通分量
    private int[] parent;   // 节点 x 的节点是 parent[x]
    private int[] size;     // 新增一个数组记录树的“重量”

    /* 构造函数，n 为图的节点总数 */
    public UF(int n) {
        // 一开始互不连通
        this.count = n;
        // 父节点指针初始指向自己
        parent = new int[n];
        size = new int[n];
        for (int i = 0; i < n; i++) {
            size[i] = 1;      // 记录每棵树包含的节点数
            parent[i] = i;
        }
    }

    /* 返回某个节点 x 的根节点 */
    private int find(int x) {
        // 根节点的 parent[x] == x
        while (parent[x] != x) {
            parent[x] = parent[parent[x]];
            x = parent[x];
        }
        return x;
    }

    /* 返回当前的连通分量个数 */
    public int count() {
        return count;
    }

    /* 其他函数 */
    public void union(int p, int q) {
        int rootP = find(p);
        int rootQ = find(q);
        if (rootP == rootQ) return;
        // 将两棵树合并为一棵
        // 简单粗暴的把 p 所在的树接到 q 所在的树的根节点下面
        if(size[rootP]>size[rootQ]) {
            parent[rootQ] = rootP;
            size[rootP] += size[rootQ];
        } else {
            parent[rootP] = rootQ;
            size[rootQ] += size[rootP];
        }
        count--; // 两个分量合二为一
    }

    // 如果节点 p 和 q 连通的话，它们一定拥有相同的根节点
    public boolean connected(int p, int q) {
        int rootP = find(p);
        int rootQ = find(q);
        return rootP == rootQ;
    }
}
```

---

## UNION-FIND算法应用

使用 Union-Find 算法，主要是如何把原问题转化成`图的动态连通性问题`。
- 对于算式合法性问题，可以直接利用等价关系，
- 对于棋盘包围问题，则是利用一个虚拟节点，营造出动态连通特性。
- 将二维数组映射到一维数组，利用方向数组 d 来简化代码量

---


### DFS 的替代方案


[130. Surrounded Regions](https://leetcode.com/problems/surrounded-regions/)
- Given an m x n matrix board containing 'X' and 'O', capture all regions that are 4-directionally surrounded by 'X'.
- A region is captured by flipping all 'O's into 'X's in that surrounded region.
- 被围绕的区域：给你一个 M×N 的二维矩阵，其中包含字符 X 和 O，
- 让你找到矩阵中四面被 X 围住的 O，并且把它们替换成 X。

传统方法
- 先用 for 循环遍历棋盘的四边，
- 用 DFS 算法把那些与边界相连的 O 换成一个特殊字符，比如 #；
- 然后再遍历整个棋盘，把剩下的 O 换成 X，
- 把 # 恢复成 O。
- 只有和边界 O 相连的 O 才具有和 dummy 的连通性，他们不会被替换。
- 这样就能完成题目的要求，时间复杂度 O(MN)。


将二维坐标映射到一维的常用技巧。
- 二维坐标 (x,y) 可以转换成 x * n + y 这个数
- （m 是棋盘的行数，n 是棋盘的列数）。


```java
// Runtime: 6 ms, faster than 12.80% of Java online submissions for Surrounded Regions.
// Memory Usage: 41 MB, less than 73.76% of Java online submissions for Surrounded Regions.
void solve(char[][] board) {
    if (board.length == 0) return;

    int m = board.length;
    int n = board[0].length;

    // 给 dummy 留一个额外位置
    UF uf = new UF(m * n + 1);
    int dummy = m * n;

    // 将首列和末列的 O 与 dummy 连通
    for (int i = 0; i < m; i++) {
        if (board[i][0] == 'O') uf.union(i * n, dummy);
        if (board[i][n - 1] == 'O') uf.union(i * n + n - 1, dummy);
    }
    // 将首行和末行的 O 与 dummy 连通
    for (int j = 0; j < n; j++) {
        if (board[0][j] == 'O') uf.union(j, dummy);
        if (board[m - 1][j] == 'O') uf.union(n * (m - 1) + j, dummy);
    }
    // 方向数组 d 是上下左右搜索的常用手法
    int[][] d = new int[][] ({1,0}, {0,1}, {0,-1}, {-1,0});
    for (int i = 1; i < m - 1; i++)
        for (int j = 1; j < n - 1; j++)
            if (board[i][j] == 'O')
                // 将此 O 与上下左右的 O 连通
                for (int k = 0; k < 4; k++) {
                    int x = i + d[k][0];
                    int y = j + d[k][1];
                    if (board[x][y] == 'O') uf.union(x * n + y, i * n + j);
                }
    // 所有不和 dummy 连通的 O，都要被替换
    for (int i = 1; i < m - 1; i++)
        for (int j = 1; j < n - 1; j++)
            if (!uf.connected(dummy, i * n + j)) board[i][j] = 'X';
}
```


```java
class UF {
    private int count;      // 记录连通分量
    private int[] parent;   // 节点 x 的节点是 parent[x]
    private int[] size;     // 新增一个数组记录树的“重量”

    /* 构造函数，n 为图的节点总数 */
    public UF(int n) {
        // 一开始互不连通
        this.count = n;
        // 父节点指针初始指向自己
        parent = new int[n];
        size = new int[n];
        for (int i = 0; i < n; i++) {
            size[i] = 1;
            parent[i] = i;
        }
    }

    /* 返回某个节点 x 的根节点 */
    private int find(int x) {
        // 根节点的 parent[x] == x
        while (parent[x] != x) {
            parent[x] = parent[parent[x]];
            x = parent[x];
        }
        return x;
    }

    /* 返回当前的连通分量个数 */
    public int count() {
        return count;
    }

    /* 其他函数 */
    public void union(int p, int q) {
        int rootP = find(p);
        int rootQ = find(q);
        if (rootP == rootQ) return;
        // 将两棵树合并为一棵
        // 简单粗暴的把 p 所在的树接到 q 所在的树的根节点下面
        if(size[rootP]>size[rootQ]) {
            parent[rootQ] = rootP;
            size[rootP] += size[rootQ];
        } else {
            parent[rootP] = rootQ;
            size[rootQ] += size[rootP];
        }
        count--; // 两个分量合二为一
    }

    // 如果节点 p 和 q 连通的话，它们一定拥有相同的根节点
    public boolean connected(int p, int q) {
        int rootP = find(p);
        int rootQ = find(q);
        return rootP == rootQ;
    }
}
```


---


### 判定合法等式

[990. Satisfiability of Equality Equations](https://leetcode.com/problems/satisfiability-of-equality-equations/)
- You are given an array of strings equations that represent relationships between variables where each string equations[i] is of length 4 and takes one of two different forms: "xi==yi" or "xi!=yi".Here, xi and yi are lowercase letters (not necessarily different) that represent one-letter variable names.
- Return true if it is possible to assign integers to variable names so as to satisfy all the given equations, or false otherwise.

```java
// Runtime: 1 ms, faster than 77.43% of Java online submissions for Satisfiability of Equality Equations.
// Memory Usage: 38.6 MB, less than 75.05% of Java online submissions for Satisfiability of Equality Equations.

class UF{
    private int count;
    private int[] parent;
    private int[] size;

    public UF(int n){
        this.count=n;
        parent=new int[n];
        size=new int[n];
        for(int i=0; i<n;i++){
          parent[i]=i;
          size[i]=1;
        }
    }

    private int find(int x){
        while(parent[x]!=x){
            parent[x]=parent[parent[x]];
            x=parent[x];
        }
        return x;
    }

    public int count(){
        return count;
    }

    public void union(int p, int q){
        int rootP=find(p);
        int rootQ=find(q);
        if(rootP == rootQ) return;
        if(size[rootP]>size[rootQ]){
            parent[rootQ]=rootP;
            size[rootP] += size[rootQ];
        } else{
            parent[rootP]=rootQ;
            size[rootQ] += size[rootP];
        }
        count--;
    }

    public boolean connected(int p, int q){
        int rootP=find(p);
        int rootQ=find(q);
        return rootP == rootQ;
    }
}

class Solution{
    public boolean equationsPossible(String[] equations){
        // 26 个英文字母
        UF uf=new UF(26);
        // 先让相等的字母形成连通分量
        for(String eq : equations){
            if(eq.charAt(1) == '='){
                char x=eq.charAt(0);
                char y=eq.charAt(3);
                uf.union(x - 'a', y - 'a');
            }
        }
        // 检查不等关系是否打破相等关系的连通性
        for(String eq : equations){
            if(eq.charAt(1) == '!'){
                char x=eq.charAt(0);
                char y=eq.charAt(3);
                // 如果相等关系成立，就是逻辑冲突
                if(uf.connected(x - 'a', y - 'a')) return false;
            }
        }
        return true;
    }
}
```

---

## DIJKSTRA 算法

「无权图」
- 与其说每条「边」没有权重，不如说每条「边」的权重都是 1，
- 从起点 start 到任意一个节点之间的路径权重就是它们之间「边」的条数

「加权图」
- 不能默认每条边的「权重」都是 1 了，
- 这个权重可以是任意正数（Dijkstra 算法要求不能存在负权重边）


DIJKSTRA
- 输入是一幅图 graph 和一个起点 start
- 返回是一个记录最短路径权重的数组。
- 比方说，
  - 输入起点 start = 3，函数返回一个 int[] 数组，
  - 假设赋值给 distTo 变量，那么从起点 3 到节点 6 的最短路径权重的值就是 distTo[6]。
- 是的，标准的 Dijkstra 算法会把从起点 start 到所有其他节点的最短路径都算出来。
- 当然，如果你的需求只是计算从起点 start 到某一个终点 end 的最短路径，那么在标准 Dijkstra 算法上稍作修改就可以更高效地完成这个需求，这个我们后面再说。
- 其次，我们也需要一个 State 类来辅助算法的运行：

Dijkstra 可以理解成一个带 dp table（备忘录）的 BFS 算法

```java
class State {
    int id;               // 图节点的 id
    int distFromStart;    // 从 start 节点到当前节点的距离
    State(int id, int distFromStart) {
        this.id = id;
        this.distFromStart = distFromStart;
    }
}

// 返回节点 from 到节点 to 之间的边的权重
int weight(int from, int to);

// 输入节点 s 返回 s 的相邻节点
List<Integer> adj(int s);

// 输入一幅图和一个起点 start，计算 start 到其他节点的最短距离
int[] dijkstra(int start, List<Integer>[] graph) {

    int V = graph.length;   // 图中节点的个数

    // 记录最短路径的权重, dp table
    int[] distTo = new int[V];              // distTo[i] 的值就是节点 start 到达节点 i 的最短路径权重
    Arrays.fill(distTo, Integer.MAX_VALUE); // 求最小值，所以 dp table 初始化为正无穷
    distTo[start] = 0;                      // base case，start 到 start 的最短距离就是 0

    // 优先级队列，distFromStart 较小的排在前面
    Queue<State> pq = new PriorityQueue<>((a, b) -> {
        return a.distFromStart - b.distFromStart;
    });

    // 从起点 start 开始进行 BFS
    pq.offer(new State(start, 0));
    while (!pq.isEmpty()) {
        State curState = pq.poll();
        int curNodeID = curState.id;
        int curDistFromStart = curState.distFromStart;

        // 已经有一条更短的路径到达 curNode 节点了
        if (curDistFromStart > distTo[curNodeID]) continue;
        // 将 curNode 的相邻节点装入队列
        for (int nextNodeID : adj(curNodeID)) {
            // 看看从 curNode 达到 nextNode 的距离是否会更短
            int distToNextNode = distTo[curNodeID] + weight(curNodeID, nextNodeID);
            if (distTo[nextNodeID] > distToNextNode) {
                // 更新 dp table
                distTo[nextNodeID] = distToNextNode;
                // 将这个节点以及距离放入队列
                pq.offer(new State(nextNodeID, distToNextNode));
            }
        }
    }
    return distTo;
}
```

---

## DIJKSTRA 算法 起点 start 到某一个终点 end 的最短路径


因为优先级队列自动排序的性质，每次从队列里面拿出来的都是 distFromStart 值最小的，所以当你从队头拿出一个节点，如果发现这个节点就是终点 end，那么 distFromStart 对应的值就是从 start 到 end 的最短距离。



```java
class State {
    int id;               // 图节点的 id
    int distFromStart;    // 从 start 节点到当前节点的距离
    State(int id, int distFromStart) {
        this.id = id;
        this.distFromStart = distFromStart;
    }
}

// 返回节点 from 到节点 to 之间的边的权重
int weight(int from, int to);

// 输入节点 s 返回 s 的相邻节点
List<Integer> adj(int s);

// 输入一幅图和一个起点 start，计算 start 到其他节点的最短距离
int[] dijkstra(int start, int end, List<Integer>[] graph) {

    int V = graph.length;   // 图中节点的个数

    // 记录最短路径的权重, dp table
    int[] distTo = new int[V];              // distTo[i] 的值就是节点 start 到达节点 i 的最短路径权重
    Arrays.fill(distTo, Integer.MAX_VALUE); // 求最小值，所以 dp table 初始化为正无穷
    distTo[start] = 0;                      // base case，start 到 start 的最短距离就是 0

    // 优先级队列，distFromStart 较小的排在前面
    Queue<State> pq = new PriorityQueue<>((a, b) -> {
        return a.distFromStart - b.distFromStart;
    });

    // 从起点 start 开始进行 BFS
    pq.offer(new State(start, 0));
    while (!pq.isEmpty()) {
        State curState = pq.poll();
        int curNodeID = curState.id;
        int curDistFromStart = curState.distFromStart;
        if (curNodeID == end) return curDistFromStart;
        // 已经有一条更短的路径到达 curNode 节点了
        if (curDistFromStart > distTo[curNodeID]) continue;
        // 将 curNode 的相邻节点装入队列
        for (int nextNodeID : adj(curNodeID)) {
            // 看看从 curNode 达到 nextNode 的距离是否会更短
            int distToNextNode = distTo[curNodeID] + weight(curNodeID, nextNodeID);
            if (distTo[nextNodeID] > distToNextNode) {
                // 更新 dp table
                distTo[nextNodeID] = distToNextNode;
                // 将这个节点以及距离放入队列
                pq.offer(new State(nextNodeID, distToNextNode));
            }
        }
    }
    return Integer.MAX_VALUE;
}
```


---


### 网络延迟时间

[743. Network Delay Time](https://leetcode.com/problems/network-delay-time/)
- You are given a network of n nodes, labeled from 1 to n.
- You are also given times, a list of travel times as directed edges `times[i] = (ui, vi, wi)`, where ui is the source node, vi is the target node, and wi is the time it takes for a signal to travel from source to target.
- We will send a signal from a given node k. Return the time it takes for all the n nodes to receive the signal.
- If it is impossible for all the n nodes to receive the signal, return -1.

求所有节点都收到信号的时间
- 把所谓的传递时间看做距离，实际上就是「从节点 k 到其他所有节点的最短路径中，最长的那条最短路径距离是多少」
- 从节点 k 出发到其他所有节点的最短路径，就是标准的 Dijkstra 算法。


```
int networkDelayTime(int[][] times, int n, int k) {
    if(n==0) return -1;
    // 节点编号是从 1 开始的，所以要一个大小为 n + 1 的邻接表
    List<int[]>[] graph = new LinkedList[n + 1];
    for (int i = 1; i <= n; i++) graph[i] = new LinkedList<>();

    // 构造图
    for (int[] edge : times) {
        int from = edge[0];
        int to = edge[1];
        int weight = edge[2];
        // from -> List<(to, weight)>
        // 邻接表存储图结构，同时存储权重信息
        graph[from].add(new int[]{to, weight});
    }
    // 启动 dijkstra 算法计算以节点 k 为起点到其他节点的最短路径
    int[] distTo = dijkstra(k, graph);

    // 找到最长的那一条最短路径
    int res = 0;
    for (int i = 1; i < distTo.length; i++) {
        if (distTo[i] == Integer.MAX_VALUE) return -1; // 有节点不可达，返回 -1
        res = Math.max(res, distTo[i]);
    }
    return res;
}


class State{
    int id;
    int distFromStart;
    State(int id, int distFromStart){
        this.id=id;
        this.distFromStart=distFromStart;
    }
}

// 输入一个起点 start，计算从 start 到其他节点的最短距离
int[] dijkstra(int start, List<int[]>[] graph) {
    // 图中节点的个数
    // 记录最短路径的权重，你可以理解为 dp table
    // 定义：distTo[i] 的值就是节点 start 到达节点 i 的最短路径权重
    // 求最小值，所以 dp table 初始化为正无穷
    // base case，start 到 start 的最短距离就是 0
    int[] disTo = new int[graph.length];
    Arrays.fill(disTo, Integer.MAX_VALUE);
    disTo[start] = 0;

    // 优先级队列，distFromStart 较小的排在前面
    Queue<State> pq = new PriorityQueue<>(
        (a,b) -> {return a.distFromStart - b.distFromStart;}
    );

    // 从起点 start 开始进行 BFS
    pq.offer(new State(start, 0));

    while(!pq.isEmpty()){
        State cur = pq.poll();

        int curId = cur.id;
        int curDistFromStart = cur.curDistFromStart;

        if(curDistFromStart > disTo[curId]) continue;

        for(int[] node : graph[curId]){
            int nextNodeID = node[0];
            int nextnodeDis = disTo[curId] + node[1];
            if( nextnodeDis < disTo[nextNodeID]){
                disTo[nextNodeID] = nextnodeDis;
                pq.offer(new State(nextNodeID, nextnodeDis);
            }
        }
    }
    return disTo;
}


```




```java

class State{
    int id;
    int distFromStart;
    State(int id, int distFromStart){
        this.id = id;
        this.distFromStart = distFromStart;
    }
}


class Solution {
    public int networkDelayTime(int[][] times, int n, int k) {
        if(n==0) return -1;
        // graph
        List<int[]>[] graph = new LinkedList[n+1];
        for(int i=0; i<n;i++) graph[i] = new LinkedList<>();
        for(int[] edge : times){
            int from = edge[0];
            int to = edge[1];
            int weight = edge[2];
            graph[from].add(new int[] {to, weight});
        }

        int[] disTo = dijkstra(k,graph);

        int res=0;
        for(int i=1; i<n;i++){
            if(disTo[i]==Integer.MAX_VALUE) return -1;
            res=Math.max(res, disTo[i]);
        }
        return res;
    }


    public int[] dijkstra(int start, List<int[]>[] graph) {
        int[] distTo = new int[graph.length];
        Arrays.fill(distTo, Integer.MAX_VALUE);
        distTo[start] = 0;

        Queue<State> pq = new PriorityQueue<>(
            (a,b) -> {return a.distFromStart = b.distFromStart;}
        );
        pq.offer(new State(start,0));

        while (!pq.isEmpty()) {
            State curState = pq.poll();
            int curNodeID = curState.id;
            int curDistFromStart = curState.distFromStart;

            if (curDistFromStart > distTo[curNodeID]) {
                continue;
            }

            // 将 curNode 的相邻节点装入队列
            for (int[] neighbor : graph[curNodeID]) {
                int nextNodeID = neighbor[0];
                int distToNextNode = distTo[curNodeID] + neighbor[1];
                // 更新 dp table
                if (distTo[nextNodeID] > distToNextNode) {
                    distTo[nextNodeID] = distToNextNode;
                    pq.offer(new State(nextNodeID, distToNextNode));
                }
            }
        }
        return distTo;
    }

}
```

---

### 路径经过的权重最大值

[1631. Path With Minimum Effort]()
- You are a hiker preparing for an upcoming hike. You are given heights, a 2D array of size rows x columns, where heights[row][col] represents the height of cell (row, col). You are situated in the top-left cell, (0, 0), and you hope to travel to the bottom-right cell, (rows-1, columns-1) (i.e., 0-indexed). You can move up, down, left, or right, and you wish to find a route that requires the minimum effort.
- A routes effort is the maximum absolute difference in heights between two consecutive cells of the route.
- Return the minimum effort required to travel from the top-left cell to the bottom-right cell.

这道题中评判一条路径是长还是短的标准不再是路径经过的权重总和，而是路径经过的权重最大值。


```java
// Runtime: 47 ms, faster than 75.18% of Java online submissions for Path With Minimum Effort.
// Memory Usage: 39.5 MB, less than 76.36% of Java online submissions for Path With Minimum Effort.

class State {
    int x, y;             // 矩阵中的一个位置
    int effortFromStart;  // 从起点 (0, 0) 到当前位置的最小体力消耗（距离）
    State(int x, int y, int effortFromStart) {
        this.x = x;
        this.y = y;
        this.effortFromStart = effortFromStart;
    }
}

// 方向数组，上下左右的坐标偏移量
int[][] dirs = new int[][]括号{0,1}, {1,0}, {0,-1}, {-1,0}括号;

// 返回坐标 (x, y) 的上下左右相邻坐标
List<int[]> adj(int[][] matrix, int x, int y) {
    int m = matrix.length;
    int n = matrix[0].length;
    // 存储相邻节点
    List<int[]> neighbors = new ArrayList<>();
    for (int[] dir : dirs) {
        int nx = x + dir[0];
        int ny = y + dir[1];
        if (nx >= m || nx < 0 || ny >= n || ny < 0) continue; // 索引越界
        neighbors.add(new int[]{nx, ny});
    }
    return neighbors;
}

// Dijkstra 算法
// 计算 (0, 0) 到 (m - 1, n - 1) 的最小体力消耗
int minimumEffortPath(int[][] heights){
    int m = heights.length;
    int n = heights[0].length;

    // 定义：从 (0, 0) 到 (i, j) 的最小体力消耗是 effortTo[i][j]
    // dp table 初始化为正无穷
    int[][] effortTo = new int[m][n];
    for (int i = 0; i < m; i++) Arrays.fill(effortTo[i], Integer.MAX_VALUE);
    // base case，起点到起点的最小消耗就是 0
    effortTo[0][0] = 0;

    // 优先级队列，effortFromStart 较小的排在前面
    Queue<State> pq = new PriorityQueue<>(
        (a, b) -> {return a.effortFromStart - b.effortFromStart;}
    );
    // 从起点 (0, 0) 开始进行 BFS
    pq.offer(new State(0, 0, 0));

    while (!pq.isEmpty()) {
        State curState = pq.poll();
        int curX = curState.x;
        int curY = curState.y;
        int curEffortFromStart = curState.effortFromStart;
        // 到达终点提前结束
        if (curX == m - 1 && curY == n - 1) return curEffortFromStart;
        if (curEffortFromStart > effortTo[curX][curY]) continue;
        // 将 (curX, curY) 的相邻坐标装入队列
        for (int[] neighbor : adj(heights, curX, curY)) {
            int nextX = neighbor[0];
            int nextY = neighbor[1];
            // 计算从 (curX, curY) 达到 (nextX, nextY) 的消耗
            int effortToNextNode = Math.max(
                effortTo[curX][curY],
                Math.abs(heights[curX][curY] - heights[nextX][nextY]));
            // 更新 dp table
            if (effortTo[nextX][nextY] > effortToNextNode) {
                effortTo[nextX][nextY] = effortToNextNode;
                pq.offer(new State(nextX, nextY, effortToNextNode));
            }
        }
    }
    // 正常情况不会达到这个 return
    return -1;
}
```

---

### 概率最大的路径

[1514. Path with Maximum Probability](https://leetcode.com/problems/path-with-maximum-probability/)
- You are given an undirected weighted graph of n nodes (0-indexed), represented by an edge list where edges[i] = [a, b] is an undirected edge connecting the nodes a and b with a probability of success of traversing that edge succProb[i].
- Given two nodes start and end, find the path with the maximum probability of success to go from start to end and return its success probability.
- If there is no path from start to end, return 0. Your answer will be accepted if it differs from the correct answer by at most 1e-5.

```java
// Runtime: 28 ms, faster than 96.09% of Java online submissions for Path with Maximum Probability.
// Memory Usage: 52.4 MB, less than 58.96% of Java online submissions for Path with Maximum Probability.

class State {
    // 图节点的 id
    int id;
    // 从 start 节点到达当前节点的概率
    double probFromStart;

    State(int id, double probFromStart) {
        this.id = id;
        this.probFromStart = probFromStart;
    }
}

double maxProbability(int n, int[][] edges, double[] succProb, int start, int end) {
    List<double[]>[] graph = new LinkedList[n];
    for (int i = 0; i < n; i++) {
        graph[i] = new LinkedList<>();
    }
    // 构造邻接表结构表示图
    for (int i = 0; i < edges.length; i++) {
        int from = edges[i][0];
        int to = edges[i][1];
        double weight = succProb[i];
        // 无向图就是双向图；先把 int 统一转成 double，待会再转回来
        graph[from].add(new double[]{(double)to, weight});
        graph[to].add(new double[]{(double)from, weight});
    }

    // 定义：probTo[i] 的值就是节点 start 到达节点 i 的最大概率
    double[] probTo = new double[n];
    // dp table 初始化为一个取不到的最小值
    Arrays.fill(probTo, -1);
    // base case，start 到 start 的概率就是 1
    probTo[start] = 1;

    // 优先级队列，probFromStart 较大的排在前面
    Queue<State> pq = new PriorityQueue<>((a, b) -> {
        return Double.compare(b.probFromStart, a.probFromStart);
    });
    // 从起点 start 开始进行 BFS
    pq.offer(new State(start, 1));

    while (!pq.isEmpty()) {
        State curState = pq.poll();
        int curNodeID = curState.id;
        double curProbFromStart = curState.probFromStart;

        // 遇到终点提前返回
        if (curNodeID == end) {
            return curProbFromStart;
        }

        if (curProbFromStart < probTo[curNodeID]) {
            // 已经有一条概率更大的路径到达 curNode 节点了
            continue;
        }
        // 将 curNode 的相邻节点装入队列
        for (double[] neighbor : graph[curNodeID]) {
            int nextNodeID = (int)neighbor[0];
            // 看看从 curNode 达到 nextNode 的概率是否会更大
            double probToNextNode = probTo[curNodeID] * neighbor[1];
            if (probTo[nextNodeID] < probToNextNode) {
                probTo[nextNodeID] = probToNextNode;
                pq.offer(new State(nextNodeID, probToNextNode));
            }
        }
    }
    // 如果到达这里，说明从 start 开始无法到达 end，返回 0
    return 0.0;
}
```


---


# 设计数据结构


- LRU 算法的淘汰策略是 Least Recently Used， 淘汰那些最久没被使用的数据；
  - LRU 算法的核心数据结构是使用哈希链表 LinkedHashMap，
  - 借助链表的`有序性`使得链表元素维持插入顺序，
  - 借助哈希映射的`快速访问能力`使得我们可以在 O(1) 时间访问链表的任意元素。
  - LRU 算法相当于把数据按照时间排序
    - 这个需求借助链表很自然就能实现，
    - 一直从链表头部加入元素的话，越靠近头部的元素就是新的数据，越靠近尾部的元素就是旧的数据，
    - 进行缓存淘汰的时候只要简单地将尾部的元素淘汰掉就行了。

- 而 LFU 算法的淘汰策略是 Least Frequently Used， 淘汰那些使用次数最少的数据。
  - LFU 算法的难度大于 LRU 算法
  - 把数据按照访问频次进行排序，
  - 还有一种情况，如果多个数据拥有相同的访问频次，我们就得删除最早插入的那个数据。
    - 也就是说 LFU 算法是淘汰访问频次最低的数据，
    - 如果访问频次最低的数据有多条，需要淘汰最旧的数据。


---

## 缓存淘汰

### LRU 缓存淘汰算法 Least Recently Used

让 put 和 get 方法的时间复杂度为 O(1)，cache 这个数据结构必要的条件：
- cache 中的元素必须有时序，
  - 以区分最近使用的和久未使用的数据，
  - 当容量满了之后要删除最久未使用的那个元素腾位置。
- 要在 cache 中快速找某个 key 是否已存在并得到对应的 val；
- 每次访问 cache 中的某个 key，需要将这个元素变为最近使用的，
  - 也就是说 cache 要支持在任意位置快速插入和删除元素。

数据结构
- 哈希表查找快，但是数据无固定顺序；
- 链表有顺序之分，插入删除快，但是查找慢。
- 结合一下，形成一种新的数据结构：哈希链表 LinkedHashMap。

LRU 缓存算法的核心数据结构就是哈希链表，双向链表和哈希表的结合体。这个数据结构长这样：

---

#### 造轮子 LRU 算法

- 我们实现的双链表 API 只能从尾部插入
- 也就是说靠尾部的数据是最近使用的，靠头部的数据是最久为使用的。

```java
// 双链表的节点类
class Node {
    public int key, val;
    public Node next, prev;
    public Node(int k, int v) {
        this.key = k;
        this.val = v;
    }
}

// 依靠我们的 Node 类型构建一个双链表
class DoubleList {
    private Node head, tail;   // 头尾虚节点
    private int size;          // 链表元素数

    public DoubleList() {
        // 初始化双向链表的数据
        head = new Node(0, 0);
        tail = new Node(0, 0);
        head.next = tail;
        tail.prev = head;
        size = 0;
    }

    // 在链表尾部添加节点 x，时间 O(1)
    public void addLast(Node x) {
        x.prev = tail.prev;
        x.next = tail;
        tail.prev.next = x;
        tail.prev = x;
        size++;
    }

    // 删除链表中的 x 节点（x 一定存在）
    // 由于是双链表且给的是目标 Node 节点，时间 O(1)
    public void remove(Node x) {
        x.prev.next = x.next;
        x.next.prev = x.prev;
        size--;
    }

    // 删除链表中第一个节点，并返回该节点，时间 O(1)
    public Node removeFirst() {
        if (head.next == tail)
            return null;
        Node first = head.next;
        remove(first);
        return first;
    }

    // 返回链表长度，时间 O(1)
    public int size() { return size; }

}

class LRUCache {
    // key -> Node(key, val)
    private HashMap<Integer, Node> map;
    // Node(k1, v1) <-> Node(k2, v2)...
    private DoubleList cache;
    // 最大容量
    private int cap;

    public LRUCache(int capacity) {
        this.cap = capacity;
        map = new HashMap<>();
        cache = new DoubleList();
    }

    public int get(int key) {
        if (!map.containsKey(key)) return -1;
        // 将该数据提升为最近使用的
        makeRecently(key);
        return map.get(key).val;
    }

    public void put(int key, int val) {
        if (map.containsKey(key)) {
            // 删除旧的数据
            deleteKey(key);
            // 新插入的数据为最近使用的数据
            addRecently(key, val);
            return;
        }

        if (cap == cache.size()) {
            // 删除最久未使用的元素
            removeLeastRecently();
        }
        // 添加为最近使用的元素
        addRecently(key, val);
    }

}

/* 将某个 key 提升为最近使用的 */
private void makeRecently(int key) {
    Node x = map.get(key);
    // 先从链表中删除这个节点
    cache.remove(x);
    // 重新插到队尾
    cache.addLast(x);
}

/* 添加最近使用的元素 */
private void addRecently(int key, int val) {
    Node x = new Node(key, val);
    // 链表尾部就是最近使用的元素
    cache.addLast(x);
    // 别忘了在 map 中添加 key 的映射
    map.put(key, x);
}

/* 删除某一个 key */
private void deleteKey(int key) {
    Node x = map.get(key);
    // 从链表中删除
    cache.remove(x);
    // 从 map 中删除
    map.remove(key);
}

/* 删除最久未使用的元素 */
private void removeLeastRecently() {
    // 链表头部的第一个元素就是最久未使用的
    Node deletedNode = cache.removeFirst();
    // 同时别忘了从 map 中删除它的 key
    int deletedKey = deletedNode.key;
    map.remove(deletedKey);
}
```

---


#### 使用 Java 内置的 LinkedHashMap 来实现一遍。


```java
class LRUCache {
    int cap;
    LinkedHashMap<Integer, Integer> cache = new LinkedHashMap<>();

    public LRUCache(int capacity) {
        this.cap = capacity;
    }

    // get + 将 key 变为最近使用
    public int get(int key) {
        if (!cache.containsKey(key)) return -1;
        // 将 key 变为最近使用
        makeRecently(key);
        return cache.get(key);
    }

    // add + 将 key 变为最近使用
    public void put(int key, int val) {
        if (cache.containsKey(key)) {
            // 修改 key 的值
            cache.put(key, val);
            // 将 key 变为最近使用
            makeRecently(key);
            return;
        }
        if (cache.size() >= this.cap) {
            // 链表头部就是最久未使用的 key
            int oldestKey = cache.keySet().iterator().next();
            cache.remove(oldestKey);
        }
        // 将新的 key 添加链表尾部
        cache.put(key, val);
    }

    private void makeRecently(int key) {
        int val = cache.get(key);
        // 删除 key，重新插入到队尾
        cache.remove(key);
        cache.put(key, val);
    }
}
```

---


### LFU 淘汰算法 Least Frequently Used



```java
// Runtime: 71 ms, faster than 56.22% of Java online submissions for LFU Cache.
// Memory Usage: 122.6 MB, less than 78.55% of Java online submissions for LFU Cache.

class LFUCache {
    HashMap<Integer, Integer> keyToVal;
    HashMap<Integer, Integer> keyToFreq;
    HashMap<Integer, LinkedHashSet<Integer>> freqToKeys;
    int minFreq;
    int cap;

    public LFUCache(int capacity){
        keyToVal = new HashMap<>();
        keyToFreq = new HashMap<>();
        freqToKeys = new HashMap<>();
        this.cap = capacity;
        this.minFreq = 0;
    }

    public int get(int key){
        if(!keyToVal.containsKey(key))return -1;
        increaseFreq(key);
        return keyToVal.get(key);
    }

    public void put(int key, int val){
        if(this.cap <= 0)return;
        if(keyToVal.containsKey(key)){
            keyToVal.put(key, val);
            increaseFreq(key);
            return;
        }
        if(keyToVal.size()>= this.cap)removeMinFreqKey();
        keyToVal.put(key, val);
        keyToFreq.put(key, 1);

        freqToKeys.putIfAbsent(1, new LinkedHashSet<>());
        freqToKeys.get(1).add(key);
        // 插入新 key 后最小的 freq 肯定是 1
        this.minFreq = 1;
    }

    public void removeMinFreqKey(){
        LinkedHashSet<Integer> keyList = freqToKeys.get(this.minFreq);
        // 其中最先被插入的那个 key 就是该被淘汰的 key
        int deletedKey = keyList.iterator().next();
        keyList.remove(deletedKey);
        if(keyList.isEmpty())freqToKeys.remove(this.minFreq);
        keyToVal.remove(deletedKey);
        keyToFreq.remove(deletedKey);
    }
    private void increaseFreq(int key){
        int freq = keyToFreq.get(key);
        keyToFreq.put(key, freq+1);
        freqToKeys.putIfAbsent(freq+1, new LinkedHashSet<>());
        freqToKeys.get(freq+1).add(key);

        freqToKeys.get(freq).remove(key);
        if(freqToKeys.get(freq).isEmpty()){
            freqToKeys.remove(freq);
            if(this.minFreq == freq)this.minFreq++;
        }
    }
}

/**
 * Your LFUCache object will be instantiated and called as such:
 * LFUCache obj = new LFUCache(capacity);
 * int param_1 = obj.get(key);
 * obj.put(key,value);
 */
```


---

## 最大栈 Maximum Frequency Stack


[895. Maximum Frequency Stack](https://leetcode.com/problems/maximum-frequency-stack/)
- Design a stack-like data structure to push elements to the stack and pop the most frequent element from the stack.
- Implement the FreqStack class:
  - FreqStack() constructs an empty frequency stack.
  - void push(int val) pushes an integer val onto the top of the stack.
  - int pop() removes and returns the most frequent element in the stack.
  - If there is a tie for the most frequent element, the element closest to the stacks top is removed and returned.


```java
FreqStack stk = new FreqStack();

// 向最大频率栈中添加元素
stk.push(2); stk.push(7); stk.push(2);
stk.push(7); stk.push(2); stk.push(4);

// 栈中元素：[2,7,2,7,2,4]
stk.pop() // 返回 2
// 因为 2 出现了三次

// 栈中元素：[2,7,2,7,4]
stk.pop() // 返回 7
// 2 和 7 都出现了两次，但 7 是最近添加的

// 栈中元素：[2,7,2,4]
stk.pop() // 返回 2

// 栈中元素：[2,7,4]89-p-0p098-0p
stk.pop() // 返回 4

// 栈中元素：[2,7]


// Runtime: 27 ms, faster than 67.35% of Java online submissions for Maximum Frequency Stack.
// Memory Usage: 49.8 MB, less than 38.95% of Java online submissions for Maximum Frequency Stack.

class FreqStack {
    int maxFre;
    HashMap<Integer, Integer> ValFre;
    HashMap<Integer, Stack<Integer>> FreVal;


    public FreqStack() {
        // 记录 FreqStack 中元素的最大频率
        maxFreq = 0;
        // 记录 FreqStack 中每个 val 对应的出现频率，后文就称为 VF 表
        valToFreq = new HashMap<>();
        // 记录频率 freq 对应的 val 列表，后文就称为 FV 表
        freqToVals = new HashMap<>();
    }

    public void push(int val) {
        // 修改 VF 表：val 对应的 freq 加一
        int freq = valToFreq.getOrDefault(val, 0) + 1;
        valToFreq.put(val, freq);
        // 修改 FV 表：在 freq 对应的列表加上 val
        freqToVals.putIfAbsent(freq, new Stack<>());
        freqToVals.get(freq).push(val);
        // 更新 maxFreq
        maxFreq = Math.max(maxFreq, freq);
    }


    public int pop() {
        // 修改 FV 表：pop 出一个 maxFreq 对应的元素 v
        Stack<Integer> vals = freqToVals.get(maxFreq);
        int v = vals.pop();
        // 修改 VF 表：v 对应的 freq 减一
        int freq = valToFreq.get(v) - 1;
        valToFreq.put(v, freq);
        // 更新 maxFreq
        if (vals.isEmpty()) {
            // 如果 maxFreq 对应的元素空了
            maxFreq--;
        }
        return v;
    }
}

```


---

# 数据流

---

## Reservoir Sampling 随机 水塘抽样算法


随机是均匀随机（uniform random）
- 如果有 n 个元素，每个元素被选中的概率都是 1/n，不可以有统计意义上的偏差。

一般的想法就是，先遍历一遍链表，得到链表的总长度 n，再生成一个 [1,n] 之间的随机数为索引，然后找到索引对应的节点，就是一个随机的节点了.
- 但只能遍历一次，意味着这种思路不可行。
- 题目还可以再泛化，给一个未知长度的序列，如何在其中随机地选择 k 个元素？想要解决这个问题，就需要著名的水塘抽样算法了。

但是这种问题的关键在于证明，你的算法为什么是对的？为什么每次以 1/i 的概率更新结果就可以保证结果是平均随机（uniform random）？
- 证明：
- 假设总共有 n 个元素，每个元素被选择的概率都是 1/n，
- 那么对于第 i 个元素，它被选择的概率就是：

![formula1](https://i.imgur.com/dYosNcJ.png)


---

### 382. Linked List Random Node 无限序列随机抽取1元素

[382. Linked List Random Node](https://leetcode.com/problems/linked-list-random-node/)
- Given a singly linked list,
- return a random nodes value from the linked list.
- Each node must have the same probability of being chosen.
- Implement the Solution class:
  - Solution(ListNode head) Initializes the object with the integer array nums.
  - int getRandom() Chooses a node randomly from the list and returns its value. All the nodes of the list should be equally likely to be choosen.

当你遇到第 i 个元素时，应该有 1/i 的概率选择该元素，1 - 1/i 的概率保持原有的选择。

证明：
- 假设总共有 n 个元素，
- 随机性 每个元素被选择的概率都是 1/n
- 那么对于第 i 个元素，它被选择的概率就是：
- 第 i 个元素被选择的概率是 1/i，
- 第 i+1 次不被替换的概率是 1 - 1/(i+1)，以此类推，相乘就是第 i 个元素最终被选中的概率，就是 1/n。

因此，该算法的逻辑是正确的。

#### be list, size, random n

```java
// time: O(N) + O(1)
// space: O(N) a list store all n
public Solution(ListNode head) {
    ArrayList<Integer> arr = new ArrayList<>();
    while(head!=null){
        arr.add(head.val);
        head=head.next;
    }
}
public int getRandom() {
     return arr.get( (int)(Math.random() * arr.size()) );
}
```

#### Reservoir Sampling

```java
// Runtime: 17 ms, faster than 28.48% of Java online submissions for Linked List Random Node.
// Memory Usage: 40.7 MB, less than 82.71% of Java online submissions for Linked List Random Node.
class Solution {
    ListNode n;
    Random r;
    public Solution(ListNode head) {
        this.r = new Random();
        this.n=head;
    }
    public int getRandom() {
        int res = 0, i = 1;
        ListNode cur = n;
        while(cur!=null){
            // 生成一个 [0, i) 之间的整数
            // 这个整数等于 0 的概率就是 1/i
            if(r.nextInt(i++) == 0) res = cur.val;
            cur = cur.next;
        }
        return res;
    }
}
```

---

### 无限序列随机抽取 k 个数

![formula2](https://i.imgur.com/Lk6Pim9.png)

```java
/* 返回链表中 k 个随机节点的值 */
int[] getRandom(ListNode head, int k) {
    Random r = new Random();
    int[] res = new int[k];
    ListNode p = head;

    // 前 k 个元素先默认选上
    for (int j = 0; j < k && p != null; j++) {
        res[j] = p.val;
        p = p.next;
    }

    int i = k;
    // while 循环遍历链表
    while (p != null) {
        // 生成一个 [0, i) 之间的整数
        int j = r.nextInt(++i);
        // 这个整数小于 k 的概率就是 k/i
        if (j < k) res[j] = p.val;
        p = p.next;
    }
    return res;
}
```

---

### 398. Random Pick Index (Medium)

[398. Random Pick Index](https://leetcode.com/problems/random-pick-index/)
Given an integer array nums with possible duplicates, randomly output the index of a given target number. You can assume that the given target number must exist in the array.

Implement the Solution class:

Solution(int[] nums) Initializes the object with the array nums.
int pick(int target) Picks a random index i from nums where nums[i] == target. If there are multiple valid i's, then each index should have an equal probability of returning.

Example 1:
Input
["Solution", "pick", "pick", "pick"]
[[[1, 2, 3, 3, 3]], [3], [1], [3]]
Output
[null, 4, 0, 2]



#### Reservoir Sampling

```java
// Runtime: 93 ms, faster than 21.70% of Java online submissions for Random Pick Index.
// Memory Usage: 72.5 MB, less than 21.60% of Java online submissions for Random Pick Index.
/**
 * Using Reservoir Sampling
 *
 * Suppose the indexes of the target element in array are from 1 to N. You have
 * already picked i-1 elements. Now you are trying to pick ith element. The
 * probability to pick it is 1/i. Now you do not want to pick any future
 * numbers.. Thus, the final probability for ith element = 1/i * (1 - 1/(i+1)) *
 * (1 - 1/(i+2)) * .. * (1 - 1/N) = 1 / N.
 *
 * Time Complexity:
 * 1) Solution() Constructor -> O(1)
 * 2) pick() -> O(N)
 *
 * Space Complexity: O(1)
 *
 * N = Length of the input array.
 */
class Solution {
    int[] nums;
    Random r;
    public Solution(int[] nums) {
        this.nums=nums;
        this.r = new Random();
    }

    public int pick(int target) {
        int res=-1, count=0;
        for(int i=0;i<nums.length;i++){
            if(target == nums[i]){
                if(r.nextInt(++count)==0) res=i;
            }
        }
        return res;
    }
}
```


#### HashMap

```java
/**
 * Preprocessing input using HashMap
 *
 * Time Complexity:
 * 1) Solution() Constructor -> O(N)
 * 2) pick() -> O(1)
 *
 * Space Complexity: O(N)
 *
 * N = Length of the input array.
 */

class Solution {
    int[] nums;
    Random r;
    Map<Integer, List<Integer>> map;

    public Solution(int[] nums) {
        this.nums=nums;
        this.r = new Random();
        this.map = new HashMap<>();
        for(int i=0;i<nums.length;i++){
            if(!map.containsKey(nums[i])) map.put(nums[i], new ArrayList<>());
            map.get(nums[i]).add(i);
        }
    }

    public int pick(int target) {
        int res=0;
        if (!map.containsKey(target)) return -1;
        List<Integer> curList = map.get(target);
        return curList.get(r.nextInt(curList.size()));
    }
}
```





---

### 380. Insert Delete GetRandom O(1) 实现随机集合

[380. Insert Delete GetRandom O(1)](https://leetcode.com/problems/insert-delete-getrandom-o1/)

Implement the RandomizedSet class:

RandomizedSet() Initializes the RandomizedSet object.
bool insert(int val) Inserts an item val into the set if not present. Returns true if the item was not present, false otherwise.
bool remove(int val) Removes an item val from the set if present. Returns true if the item was present, false otherwise.
int getRandom() Returns a random element from the current set of elements (it's guaranteed that at least one element exists when this method is called). Each element must have the same probability of being returned.
You must implement the functions of the class such that each function works in average O(1) time complexity.


```java
Example 1:

Input
["RandomizedSet", "insert", "remove", "insert", "getRandom", "remove", "insert", "getRandom"]
[[], [1], [2], [2], [], [1], [2], []]
Output
[null, true, false, true, 2, true, false, 2]

Explanation
RandomizedSet randomizedSet = new RandomizedSet();
randomizedSet.insert(1); // Inserts 1 to the set. Returns true as 1 was inserted successfully.
randomizedSet.remove(2); // Returns false as 2 does not exist in the set.
randomizedSet.insert(2); // Inserts 2 to the set, returns true. Set now contains [1,2].
randomizedSet.getRandom(); // getRandom() should return either 1 or 2 randomly.
randomizedSet.remove(1); // Removes 1 from the set, returns true. Set now contains [2].
randomizedSet.insert(2); // 2 was already in the set, so return false.
randomizedSet.getRandom(); // Since 2 is the only number in the set, getRandom() will always return 2.
```
难点：

1. 插入，删除，获取随机元素这三个操作的时间复杂度必须都是 O(1)。
   1. 想「等概率」且「在 O(1) 的时间」取出元素，一定要满足：底层用数组实现，且数组必须是紧凑的。
   2. 这样就可以直接生成随机数作为索引，从数组中取出该随机索引对应的元素，作为随机元素。
   1. 但如果用数组存储元素的话，插入，删除的时间复杂度怎么可能是 O(1) 呢？
      1. 对数组尾部进行插入和删除操作不会涉及数据搬移，时间复杂度是 O(1)。
      2. 所以在 O(1) 的时间删除数组中的某一个元素 val，可以先把这个元素交换到数组的尾部，然后再 pop 掉。
      3. 交换两个元素必须通过索引进行交换对吧，那么我们需要一个哈希表 valToIndex 来记录每个元素值对应的索引。



2. getRandom 方法返回的元素必须等概率返回随机元素，如果集合里面有 n 个元素，每个元素被返回的概率必须是 1/n。

```java
class RandomizedSet {

    HashMap<Integer,Integer> list=null;
    int[] array=null;
    int index=0;
    Random random=null;

    public RandomizedSet() {
        //val, index
        list=new HashMap<Integer,Integer>();
        //{[index]val, }
        array=new int[100001];
        int index=0;
        random=new Random();
    }

    public boolean insert(int val) {
        // 若 val 已存在，不用再插入
        if(list.containsKey(val)) return false;
        // 若 val 不存在，插入到 nums 尾部，
        // 并记录 val 对应的索引值
        else {
            array[index] = val;
            list.put(val, index);
            index++;
            return true;
        }
    }

    public boolean remove(int val) {
        if(!list.containsKey(val)) return false;
        else {
            // 先拿到 val 的索引
            int pos = list.remove(val);
            array[pos] = array[index-1];
            if(list.containsKey(array[index-1])){
                list.put(array[index-1], pos);
            }
            index--;
            return true;
        }
    }

    public int getRandom() {
        // 随机获取 nums 中的一个元素
        return array[random.nextInt(index)];
    }
}
```

---

### 710. Random Pick with Blacklist 避开黑名单的随机数 `blacklist index to good index`

[710. Random Pick with Blacklist](https://leetcode.com/problems/random-pick-with-blacklist/)

You are given an integer n and an array of unique integers blacklist. Design an algorithm to pick a random integer in the range [0, n - 1] that is not in blacklist. Any integer that is in the mentioned range and not in blacklist should be equally likely to be returned.

Optimize your algorithm such that it minimizes the number of calls to the built-in random function of your language.

Implement the Solution class:

Solution(int n, int[] blacklist) Initializes the object with the integer n and the blacklisted integers blacklist.
int pick() Returns a random integer in the range [0, n - 1] and not in blacklist.


- 给你输入一个正整数 N，代表左闭右开区间 [0,N)，
- 再给你输入一个数组 blacklist，其中包含一些「黑名单数字」，且 blacklist 中的数字都是区间 [0,N) 中的数字。


```java
// Runtime: 42 ms, faster than 40.45% of Java online submissions for Random Pick with Blacklist.
// Memory Usage: 54 MB, less than 17.08% of Java online submissions for Random Pick with Blacklist.

class Solution {

    Random ran = new Random();
    HashMap<Integer, Integer> map = new HashMap<>();
    int range;

    public Solution(int n, int[] blacklist) {
        Arrays.sort(blacklist);
        for(int b:blacklist) map.put(b, b);

        range = n-blacklist.length;
        int last=n-1;

        for(int b:blacklist) {
            if(b<range){
                while(map.containsKey(last)) last--;
                map.put(b,last);
                last--;
            }
        }
    }

    public int pick() {
        int res = ran.nextInt(range);
        return map.getOrDefault(res, res);
    }
}
```


---


### 528. Random Pick with Weight (Medium)

You are given a 0-indexed array of positive integers w where w[i] describes the weight of the ith index.

You need to implement the function pickIndex(), which randomly picks an index in the range [0, w.length - 1] (inclusive) and returns it. The probability of picking an index i is w[i] / sum(w).

For example, if w = [1, 3], the probability of picking index 0 is 1 / (1 + 3) = 0.25 (i.e., 25%), and the probability of picking index 1 is 3 / (1 + 3) = 0.75 (i.e., 75%).

#### `2 for: [1,2,3] -> [1,2,2,3,3,3]`

```java
// memory limit exceeds. Then pick a random value from the arraylist.
class Solution {
    private ArrayList<Integer> nums;
    private Random rand;
    public Solution(int[] w) {
        this.nums = new ArrayList<>();
        this.rand = new Random();
        for (int i = 0; i < w.length; i++){
            for (int j = 0; j < w[i]; j++) this.nums.add(i);
        }
    }
    public int pickIndex() {
        int n = this.rand.nextInt(nums.size());
        return nums.get(n);
    }
}
```


#### Reservoir Sampling

[1,2,3,4,5,]
[1,3,6,10,15]


```java
class Solution {
    int[] nums;
    int total;
    Random r;
    public Solution(int[] w) {
        this.nums = new int[w.length];
        this.r = new Random();
        int runningTotal = 0;
        for (int i = 0; i < w.length; i++) {
            runningTotal += w[i];
            this.nums[i] = runningTotal;
        }
        this.total = runningTotal;
    }
    public int pickIndex() {
        if (this.total == 0) return -1;
        int n = this.r.nextInt(this.total);
        for (int i = 0; i < this.nums.length; i++) {
            if (n < this.nums[i]) return i;
        }
        return -1;
    }
}
```

#### reservoir sampling **BEST**

```java
// Runtime: 20 ms, faster than 96.28% of Java online submissions for Random Pick with Weight.
// Memory Usage: 44 MB, less than 69.48% of Java online submissions for Random Pick with Weight.
class Solution {
    int[] nums;
    int total;
    Random r;
    public Solution(int[] w) {
        this.nums = new int[w.length];
        this.r = new Random();
        int runningTotal = 0;
        for (int i = 0; i < w.length; i++) {
            runningTotal += w[i];
            this.nums[i] = runningTotal;
        }
        this.total = runningTotal;
    }
    public int pickIndex() {
        if (this.total == 0) return -1;
        int n = this.r.nextInt(nums[nums.length - 1]) + 1;
        return binarySearch(n);
    }
    private int binarySearch(int pos) {
        int left = 0, right = nums.length - 1;
        while(left < right) {
            int mid = left + (right - left) / 2;
            if(nums[mid] < pos) left = mid + 1;
            else right = mid;
        }
        return left;
    }
}
```




---


## other


### 295. Find Median from Data Stream 中位数

[295. Find Median from Data Stream](https://leetcode.com/problems/find-median-from-data-stream/)
- The median is the middle value in an ordered integer list. If the size of the list is even, there is no middle value and the median is the mean of the two middle values.
  - For example, for arr = [2,3,4], the median is 3.
  - For example, for arr = [2,3], the median is (2 + 3) / 2 = 2.5.

- Implement the MedianFinder class:
  - MedianFinder() initializes the MedianFinder object.
  - void addNum(int num) adds the integer num from the data stream to the data structure.
  - double findMedian() returns the median of all elements so far.
    - Answers within 10^-5 of the actual answer will be accepted.

1. 如果输入一个数组，排个序，长度是奇数，最中间的一个元素就是中位数，长度是偶数，最中间两个元素的平均数作为中位数。
2. 如果数据规模非常大，排序不现实，使用概率算法，随机抽取一部分数据，排序，求中位数，作为所有数据的中位数。

必然需要有序数据结构，本题的核心思路是使用两个优先级队列。


```java
// Runtime: 102 ms, faster than 70.85% of Java online submissions for Find Median from Data Stream.
// Memory Usage: 69.2 MB, less than 50.74% of Java online submissions for Find Median from Data Stream.

class MedianFinder {

    private PriorityQueue<Integer> large;
    private PriorityQueue<Integer> small;

    public MedianFinder() {
        // 小顶堆
        large = new PriorityQueue<>();
        // 大顶堆
        small = new PriorityQueue<>(
            (a, b) -> {return b - a;}
        );
    }

    public double findMedian() {
        // 如果元素不一样多，多的那个堆的堆顶元素就是中位数
        if (large.size() < small.size()) return small.peek();
        else if (large.size() > small.size()) return large.peek();
        // 如果元素一样多，两个堆堆顶元素的平均数是中位数
        return (large.peek() + small.peek()) / 2.0;
    }

    public void addNum(int num) {
        if (small.size() >= large.size()) {
            small.offer(num);
            large.offer(small.poll());
        } else {
            large.offer(num);
            small.offer(large.poll());
        }
    }
}

```

---

# DFS and BFS

1. 为什么 BFS 可以找到最短距离，DFS 不行吗？
   1. BFS 的逻辑，depth 每增加一次，队列中的所有节点都向前迈一步，这保证了第一次到达终点的时候，走的步数是最少的。
   2. DFS 也是可以的，但是时间复杂度相对高很多。DFS 实际上是靠递归的堆栈记录走过的路径，找最短路径得把二叉树中所有树杈都探索完, 才能对比出最短的路径有多长
   3. BFS 借助队列做到一次一步「齐头并进」，是可以在不遍历完整棵树的条件下找到最短距离的。
   4. 形象点说，DFS 是线，BFS 是面；DFS 是单打独斗，BFS 是集体行动

2. 既然 BFS 那么好，为啥 DFS 还要存在？
   1. BFS 可以找到最短距离，但是空间复杂度高，而 DFS 的空间复杂度较低。
   2. 假设给你的这个二叉树是满二叉树，节点数为 N，对于 DFS 算法来说，空间复杂度无非就是递归堆栈，最坏情况下顶多就是树的高度，也就是 O(logN)。
   3. BFS 算法，队列中每次都会储存着二叉树一层的节点，这样的话最坏情况下空间复杂度应该是树的最底层节点的数量，也就是 N/2，用 Big O 表示的话也就是 O(N)。
   4. 由此观之，BFS 还是有代价的，一般来说在找最短路径的时候使用 BFS，其他时候还是 DFS 使用得多一些（主要是递归代码好写）。

---

## BFS


BFS 相对 DFS 的最主要的区别是：BFS 找到的路径一定是最短的，但代价就是空间复杂度可能比 DFS 大很多

BFS 出现的常见场景好吧，
- 问题的本质就是让你在一幅「图」中找到从起点 start 到终点 target 的最近距离
- BFS 算法问题其实都是在干这个事儿，
- 比如走迷宫，有的格子是围墙不能走，从起点到终点的最短距离是多少？如果这个迷宫带「传送门」可以瞬间传送呢？
- 再比如说两个单词，要求你通过某些替换，把其中一个变成另一个，每次只能替换一个字符，最少要替换几次？
- 比如说连连看游戏，两个方块消除的条件不仅仅是图案相同，还得保证两个方块之间的最短连线不能多于两个拐点。你玩连连看，点击两个坐标，游戏是如何判断它俩的最短连线有几个拐点的？
- 本质上就是一幅「图」，让你从一个起点，走到终点，问最短路径。

```java
// 计算从起点 start 到终点 target 的最近距离
int BFS(Node start, Node target) {
    Queue<Node> q; // 核心数据结构
    Set<Node> visited; // 避免走回头路

    // 将起点加入队列
    q.offer(start);
    visited.add(start);
    int step = 0; // 记录扩散的步数

    while (q not empty) {
        int sz = q.size();
        /* 将当前队列中的所有节点向四周扩散 */
        for (int i = 0; i < sz; i++) {
            Node cur = q.poll();
            /* 划重点：这里判断是否到达终点 */
            if (cur is target) return step;
            /* 将 cur 的相邻节点加入队列 */
            for (Node x : cur.adj())
                if (x not in visited) {
                    q.offer(x);
                    visited.add(x);
                }
        }
        /* 划重点：更新步数在这里 */
        step++;
    }
}
```

---

### 752. Open the Lock 解开密码锁最少次数 `用Queue和q.size去遍历all + visited + deads`

[752. Open the Lock](https://labuladong.github.io/algo/4/29/108/)
- You have a lock in front of you with 4 circular wheels.
- Each wheel has 10 slots: '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'.
- The wheels can rotate freely and wrap around: for example we can turn '9' to be '0', or '0' to be '9'.
- Each move consists of turning one wheel one slot.
- The lock initially starts at '0000', a string representing the state of the 4 wheels.
- You are given a list of deadends dead ends, meaning if the lock displays any of these codes, the wheels of the lock will stop turning and you will be unable to open it.
- Given a target representing the value of the wheels that will unlock the lock, return the minimum total number of turns required to open the lock, or -1 if it is impossible.


#### BFS

```java
// Runtime: 76 ms, faster than 79.81% of Java online submissions for Open the Lock.
// Memory Usage: 44.9 MB, less than 79.14% of Java online submissions for Open the Lock.
// 将 s[j] 向上拨动一次

String plusOne(String s, int j) {
    char[] ch = s.toCharArray();
    if (ch[j] == '9') ch[j] = '0';
    else ch[j] += 1;
    return new String(ch);
}
// 将 s[i] 向下拨动一次
String minusOne(String s, int j) {
    char[] ch = s.toCharArray();
    if (ch[j] == '0') ch[j] = '9';
    else ch[j] -= 1;
    return new String(ch);
}

int openLock(String[] deadends, String target) {
    // 记录需要跳过的死亡密码
    Set<String> deads = new HashSet<>();
    for (String s : deadends) deads.add(s);

    // 记录已经穷举过的密码，防止走回头路
    Set<String> visited = new HashSet<>();
    Queue<String> q = new LinkedList<>();
    // 从起点开始启动广度优先搜索
    int step = 0;
    q.offer("0000");
    visited.add("0000");

    while (!q.isEmpty()) {
        int sz = q.size();
        /* 将当前队列中的所有节点向周围扩散 */
        for (int i = 0; i < sz; i++) {
            String cur = q.poll();
            /* 判断是否到达终点 */
            if (deads.contains(cur)) continue;
            if (cur.equals(target)) return step;
            /* 将一个节点的未遍历相邻节点加入队列 */
            for (int j = 0; j < 4; j++) {
                String up = plusOne(cur, j);
                String down = minusOne(cur, j);
                if (!visited.contains(up)) {
                    q.offer(up);
                    visited.add(up);
                }
                if (!visited.contains(down)) {
                    q.offer(down);
                    visited.add(down);
                }
            }
        }
        /* 在这里增加步数 */
        step++;
    }
    // 如果穷举完都没找到目标密码，那就是找不到了
    return -1;
}
```


#### 双向 BFS 优化 `用Queue和q.size去遍历 q1=q2;q2=temp`


无论传统 BFS 还是双向 BFS，无论做不做优化，
- 从 Big O 衡量标准来看，时间复杂度都是一样的，
- 只能说双向 BFS 是一种 trick，算法运行的速度会相对快一点


- 双向 BFS 也有局限，因为你必须知道终点在哪里。
  - 比如我们刚才讨论的二叉树最小高度的问题，你一开始根本就不知道终点在哪里，也就无法使用双向 BFS；
  - 但是第二个密码锁的问题，是可以使用双向 BFS 算法来提高效率的，代码稍加修改即可：

- 还是遵循 BFS 算法框架的，
  - 只是不再使用队列，而是使用 HashSet 方便快速判断两个集合是否有交集。
- 另外的一个技巧点就是 while 循环的最后交换 q1 和 q2 的内容，
  - 所以只要默认扩散 q1 就相当于轮流扩散 q1 和 q2。

```java
// Runtime: 20 ms, faster than 96.72% of Java online submissions for Open the Lock.
// Memory Usage: 39.4 MB, less than 98.61% of Java online submissions for Open the Lock.

String plusOne(String s, int j) {
    char[] ch = s.toCharArray();
    if (ch[j] == '9') ch[j] = '0';
    else ch[j] += 1;
    return new String(ch);
}
// 将 s[i] 向下拨动一次
String minusOne(String s, int j) {
    char[] ch = s.toCharArray();
    if (ch[j] == '0') ch[j] = '9';
    else ch[j] -= 1;
    return new String(ch);
}

int openLock(String[] deadends, String target) {
    Set<String> deads = new HashSet<>();
    for (String s : deadends) deads.add(s);
    // 用集合不用队列，可以快速判断元素是否存在
    Set<String> q1 = new HashSet<>();
    Set<String> q2 = new HashSet<>();
    Set<String> visited = new HashSet<>();

    int step = 0;
    q1.add("0000");
    q2.add(target);
    while (!q1.isEmpty() && !q2.isEmpty()) {
        // 哈希集合在遍历的过程中不能修改，用 temp 存储扩散结果
        Set<String> temp = new HashSet<>();
        /* 将 q1 中的所有节点向周围扩散 */
        for (String cur : q1) {
            /* 判断是否到达终点 */
            if (deads.contains(cur)) continue;
            if (q2.contains(cur)) return step;
            visited.add(cur);
            /* 将一个节点的未遍历相邻节点加入集合 */
            for (int j = 0; j < 4; j++) {
                String up = plusOne(cur, j);
                String down = minusOne(cur, j);
                if (!visited.contains(up)) temp.add(up);
                if (!visited.contains(down)) temp.add(down);
            }
        }
        /* 在这里增加步数 */
        step++;
        // temp 相当于 q1
        // 这里交换 q1 q2，下一轮 while 就是扩散 q2
        q1 = q2;
        q2 = temp;
    }
    return -1;
}
```

双向 BFS 还有一个优化，就是在 while 循环开始时做一个判断：
- 因为按照 BFS 的逻辑，队列（集合）中的元素越多，扩散之后新的队列（集合）中的元素就越多；
- 在双向 BFS 算法中，如果我们每次都选择一个较小的集合进行扩散，那么占用的空间增长速度就会慢一些，效率就会高一些。


```java
// ...
while (!q1.isEmpty() && !q2.isEmpty()) {
    if (q1.size() > q2.size()) {
        // 交换 q1 和 q2
        temp = q1;
        q1 = q2;
        q2 = temp;
    }
    // ...
```

---

## DFS backtrack 回溯算法


回溯算法其实就是我们常说的 DFS 算法，本质上就是一种暴力穷举算法。
- 1、路径：也就是已经做出的选择。
- 2、选择列表：也就是你当前可以做的选择。
- 3、结束条件：也就是到达决策树底层，无法再做选择的条件。

这也是回溯算法的一个特点，不像动态规划存在重叠子问题可以优化，回溯算法就是纯暴力穷举，复杂度一般都很高。

```java
// 防止重复遍历同一个节点
boolean[] visited;
// 从节点 s 开始 BFS 遍历，将遍历过的节点标记为 true
void traverse(List<Integer>[] graph, int s) {
    if (visited[s]) return;
    /* 前序遍历代码位置 */
    // 将当前节点标记为已遍历
    visited[s] = true;
    for (int t : graph[s]) traverse(graph, t);
    /* 后序遍历代码位置 */
}

result = []
def backtrack(路径, 选择列表):
    if 满足结束条件:
        result.add(路径)
        return

    for 选择 in 选择列表:
        做选择
        backtrack(路径, 选择列表)
        撤销选择
```

---

### 46. Permutations 全排列问题 ??????????/

[46. Permutations](https://leetcode.com/problems/permutations/)

Given an array nums of distinct integers, return all the possible permutations.
You can return the answer in any order.

Example 1:

Input: nums = [1,2,3]
Output: [[1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,1,2],[3,2,1]]

1. Iterative Solution

```java
/**
 * Iterative Solution
 *
 * The idea is to add the nth number in every possible position of each
 * permutation of the first n-1 numbers.
 *
 * Time Complexity: O(N * N!). Number of permutations = P(N,N) = N!. Each permutation takes O(N) to construct
 * T(n) = (x=2->n) ∑ (x-1)!*x(x+1)/2
 *      = (x=1->n-1) ∑ (x)!*x(x-1)/2
 *      = O(N * N!)
 * Space Complexity: O((N-1) * (N-1)!) = O(N * N!). All permutations of the first n-1 numbers.
 */
class Solution {
    public List<List<Integer>> permute(int[] nums) {
        List<List<Integer>> result = new ArrayList<>();
        if (nums == null || nums.length == 0) return result;
        result.add(Arrays.asList(nums[0]));

        for (int i = 1; i < nums.length; i++) {
            List<List<Integer>> subres = new ArrayList<>();
            for (List<Integer> cur : result) {
                for (int j = 0; j <= i; j++) {
                    List<Integer> newCur = new ArrayList<>(cur);
                    newCur.add(j, nums[i]);
                    subres.add(newCur);
                }
            }
            result = subres;
        }
        return result;
    }
}
```

2. Recursive Backtracking using visited array

```java
// Runtime: 1 ms, faster than 94.15% of Java online submissions for Permutations.
// Memory Usage: 39.2 MB, less than 76.83% of Java online submissions for Permutations.
// Time Complexity: O(N * N!). Number of permutations = P(N,N) = N!. Each permutation takes O(N) to construct
//  * T(n) = n*T(n-1) + O(n)
//  * T(n-1) = (n-1)*T(n-2) + O(n)
//  * ...
//  * T(2) = (2)*T(1) + O(n)
//  * T(1) = O(n)
// Space Complexity: O(N). Recursion stack + visited array

class Solution {
    public List<List<Integer>> permute(int[] nums) {
        if (nums == null || nums.length == 0) return result;
        // 记录「路径」
        List<List<Integer>> res = new LinkedList<>();
        LinkedList<Integer> track = new LinkedList<>();
        boolean[] used = new boolean[nums.length];
        backtrack(track, used, res, nums);
        return res;
    }

    // 从节点 s 开始 BFS 遍历，将遍历过的节点标记为 true
    void backtrack(LinkedList<Integer> track, boolean[] used, List<List<Integer>> res, int[] nums) {
        // 触发结束条件
        if (track.size() == nums.length) {
            res.add(new LinkedList(track));
            return;
        }
        for(int i=0; i<nums.length; i++){
            // skip used letters
            if (used[i]) continue;
            // add letter to permutation, mark letter as used
            track.add(nums[i]);
            used[i] = true;
            backtrack(track, used, res, nums);
            // remove letter from permutation, mark letter as unused
            track.removeLast();
            used[i] = false;
        }
    }
}
```


---


### 51. N-Queens N 皇后问题 ??????????

[51. N-Queens](https://leetcode.com/problems/n-queens/)

The n-queens puzzle is the problem of placing n queens on an n x n chessboard such that no two queens attack each other.

Given an integer n, return all distinct solutions to the n-queens puzzle. You may return the answer in any order.

Each solution contains a distinct board configuration of the n-queens' placement, where 'Q' and '.' both indicate a queen and an empty space, respectively.


```java
class Solution {
    public List<List<String>> solveNQueens(int n) {
        List<List<String>> res = new ArrayList<List<String>>();
        int[] pos = new int[n];
        dfs(pos, 0, n, res);
        return res;
    }

    public void dfs(int[] pos, int step, int n, List<List<String>> list) {
        if(step==n) {
            ArrayList<String> ls = printboard(pos,n);
            res.add(new ArrayList<String>(ls));
            return;
        }
        for(int i=0;i<n;i++) {
            pos[step]=i;
            if(isvalid(pos,step)) dfs(pos,step+1,n,list);
        }
    }

    public boolean isvalid(int[] pos, int step) {
        for(int i=0;i<step;i++) {
            if( pos[i]==pos[step] || (Math.abs(pos[i]-pos[step]))==(step-i) ) return false;
        }
        return true;
    }

    public ArrayList<String> printboard(int[] pos,int n) {
        ArrayList<String> ls=new ArrayList<String>();
        for(int i=0;i<n;i++) {
            StringBuilder sb = new StringBuilder();
            for(int j=0;j<n-1;j++) sb.append('.');
            sb.insert(pos[i],'Q');
            ls.add(sb.toString());
        }
        return ls;
    }
}
```



```java
/**
 * Space Optimized Backtracking
 * Total number of permutations can be found by this equation
 * T(N) = N * T(N-1) + O(N)
 * T(N-1) = (N-1) * T(N-2) + O(N)
 * T(N-2) = (N-2) * T(N-3) + O(N)
 * T(N-3) = (N-3) * T(N-4) + O(N)
 * ...
 * T(2) = 2 * T(1) + O(N)
 * T(1) = O(1)
 * Thus total number of permutations
 *      = N * (P(N,0) + P(N,1) + ... + P(N, N-2)) + P(N,N-1)
 *      = N * (e * N! - P(N,N-1) - P(N,N)) + N!
 *      = ((e-2)*N + 1) * N!
        = (0.718 * N + 1) * N!
 * Also, if there are S(N) solutions, then time taken to generate these solution will be N^2 * S(N).
 * Here number of solutions will be much less than the total number of permutations.
 * Thus we can ignore the time taken for generating and adding the board in the result list.
 * Total Time Complexity = O(N * N!)
 * Space Complexity:
 * -> O(N) for queensPos arr
 * -> O(N) for recursion depth
 * -> O(1) for occupied BitSet
 * Total Space Complexity = O(N)
 * N = Input board size.
 */

class Solution {
    public List<List<String>> solveNQueens(int n) {
        if (n <= 0) throw new IllegalArgumentException("Invalid board");
        List<List<String>> result = new ArrayList<>();
        int[] queensPos = new int[n];
        solveNQueensHelper(result, queensPos, new BitSet(5 * n), 0);
        return result;
    }

    private void solveNQueensHelper(List<List<String>> result, int[] queensPos, BitSet occupied, int row) {
        int n = queensPos.length;
        if (row == n) {
            result.add(generateResultBoard(queensPos));
            return;
        }

        for (int col = 0; col < n; col++) {
            // First N bits are for columns
            // Then 2*N bits are for diagonal at 45 degrees
            // Then 2*N bits are for diagonal at 135 degrees
            int diag45 = n + (row + col);
            int diag135 = 3 * n + (n + row - col);
            if (occupied.get(col) || occupied.get(diag45) || occupied.get(diag135)) continue;

            occupied.set(col);
            occupied.set(diag45);
            occupied.set(diag135);
            queensPos[row] = col;

            solveNQueensHelper(result, queensPos, occupied, row + 1);

            occupied.clear(col);
            occupied.clear(diag45);
            occupied.clear(diag135);
        }
    }

    private List<String> generateResultBoard(int[] queensPos) {
        List<String> temp = new ArrayList<>();
        char[] b = new char[queensPos.length];
        Arrays.fill(b, '.');
        for (int q : queensPos) {
            b[q] = 'Q';
            temp.add(new String(b));
            b[q] = '.';
        }
        return temp;
    }
}
```




```java
vector<vector<string>> res;
/* 输入棋盘边长 n，返回所有合法的放置 */
vector<vector<string>> solveNQueens(int n) {
    // '.' 表示空，'Q' 表示皇后，初始化空棋盘。
    vector<string> board(n, string(n, '.'));
    backtrack(board, 0);
    return res;
}

// 路径：board 中小于 row 的那些行都已经成功放置了皇后
// 选择列表：第 row 行的所有列都是放置皇后的选择
// 结束条件：row 超过 board 的最后一行
void backtrack(vector<string>& board, int row) {
    // 触发结束条件
    if (row == board.size()) {
        res.push_back(board);
        return;
    }
    int n = board[row].size();
    for (int col = 0; col < n; col++) {
        // 排除不合法选择
        if (!isValid(board, row, col)) continue;
        // 做选择
        board[row][col] = 'Q';
        // 进入下一行决策
        backtrack(board, row + 1);
        // 撤销选择
        board[row][col] = '.';
    }
}

/* 是否可以在 board[row][col] 放置皇后？ */
bool isValid(vector<string>& board, int row, int col) {
    int n = board.size();
    // 检查列是否有皇后互相冲突
    for (int i = 0; i < n; i++) {
        if (board[i][col] == 'Q') return false;
    }
    // 检查右上方是否有皇后互相冲突
    for (int i = row - 1, j = col + 1; i >= 0 && j < n; i--, j++) {
        if (board[i][j] == 'Q') return false;
    }
    // 检查左上方是否有皇后互相冲突
    for (int i = row - 1, j = col - 1; i >= 0 && j >= 0; i--, j--) {
        if (board[i][j] == 'Q') return false;
    }
    return true;
}
```

---

### 78. Subsets 子集（中等）

[78. Subsets](https://leetcode.com/problems/subsets/)

Given an integer array nums of unique elements, return all possible subsets (the power set).

The solution set must not contain duplicate subsets. Return the solution in any order.

Example 1:
Input: nums = [1,2,3]
Output: [[],[1],[2],[1,2],[3],[1,3],[2,3],[1,2,3]]

Example 2:
Input: nums = [0]
Output: [[],[0]]


1. 数学归纳 递归结构 Iterative
   1. [1,2,3] 的子集可以由 [1,2] 追加得出，[1,2] 的子集可以由 [1] 追加得出，base case 显然就是当输入集合为空集时，输出子集也就是一个空集。
   2. `subset([1,2,3]) = A + [A[i].add(3) for i = 1..len(A)]`
   3. The idea is simple. We go through the elements in the nums list. For each element, we loop over the current result list we have constructed so far. For each list in the result, we make a copy of this list and append the current element to it (it means picking the element). It is based on the same idea in backtracking (in each step you have choices: pick or not pick).
   4. 计算递归算法时间复杂度的方法
      1. 递归深度 乘以 每次递归中迭代的次数
      2. 递归深度显然是 N，每次递归 for 循环的迭代次数取决于 res 的长度，并不是固定的。
      3. res 的长度应该是每次递归都翻倍，所以说总的迭代次数应该是 2^N。
      4. 大小为 N 的集合的子集总共有几个？2^N 个
      5. 2^N 个子集是 push_back 添加进 res 的，所以要考虑 push_back 这个操作的效率：
      6. 总的时间复杂度就是 O(N*2^N)，还是比较耗时的。
   5.  如果不计算储存返回结果所用的空间的，只需要 O(N) 的递归堆栈空间。如果计算 res 所需的空间，应该是 O(N*2^N)。

```java
// Runtime: 0 ms, faster than 100.00% of Java online submissions for Subsets.
// Memory Usage: 38.8 MB, less than 97.56% of Java online submissions for Subsets.
/**
 * Constant Space Iterative Solution

 * S(n) = (0 × (n C 0) + 1 × (n C 1) + 2 × (n C 2) + … + n × (n C n))
 * Note that (n C k) = (n C n-k). Therefore:
 * S(n) = 0 × (n C n) + 1 × (n C n-1) + 2 × (n C n-2) + … + n × (n C 0)
 * If we add these two together, we get
 * 2S(n) = n × (n C 0) + n × (n C 1) + … + n × (n C n)
 *       = n × (n C 0 + n C 1 + … + n C n)
 * As per binomial theorem, (n C 0 + n C 1 + … + n C n) = 2^n, so
 * 2*S(n) = n * 2^n => S(n) = n * 2^(n-1)
 *
 * Time Complexity: O(S(N) + n C 0) = O(N * 2^(N-1) + 1) = O(N * 2^N)
 *
 * Space Complexity: O(1) (Excluding the result space)
 *
 * N = Length of input nums array
 */
public List<List<Integer>> subsets(int[] nums) {
    // Arrays.sort(nums); // make sure subsets are ordered, not needed
    List<List<Integer>> res = new ArrayList<>();
    res.add(new ArrayList<>()); // start with empty set
    for (int i = 0; i < nums.length; ++i) {
        for (int j = 0, size = res.size(); j < size; ++j) { // remember
            List<Integer> subset = new ArrayList<>(res.get(j)); // copy a new one
            subset.add(nums[i]); // expand
            res.add(subset); // collect
        }
    }
    return res;
}

// Runtime: 0 ms, faster than 100.00% of Java online submissions for Subsets.
// Memory Usage: 39.2 MB, less than 77.10% of Java online submissions for Subsets.
// Time: O(N * 2^N)
// The outer loop takes O(N) time.
// The inner loop takes 2, 4, 8, ..., 2^N time respectively.
// In inner loop, making a new copy of L takes at most O(N) time.
// Total runtime T(N) = N * (2 + 4 + 8 + ... + 2^N) ~= N * 2^N
// Space: O(N * 2^N)
public List<List<Integer>> subsets(int[] nums) {
  List<List<Integer>> result = new ArrayList<>();
  result.add(new ArrayList<>());  // empty set

  for (int i = 0; i < nums.length; ++i) {
    List<List<Integer>> subres = new ArrayList<>(); // used for new lists
    for (List<Integer> L : result) {
      L = new ArrayList<>(L); // copy
      L.add(nums[i]);
      subres.add(L);
    }
    result.addAll(subres);  // concatenate
  }
  return result;
}
```



2. backtracking

```java
// Runtime: 2 ms, faster than 21.39% of Java online submissions for Subsets.
// Memory Usage: 40 MB, less than 20.20% of Java online submissions for Subsets.
/**
 * Backtracking (Recursion)
 *
 * Time Complexity: O(N * 2 ^ N) Refer to above explanation
 *
 * Space Complexity: O(N) (Recursion Depth + TempList)
 *
 * N = Length of input nums array
 */
public List<List<Integer>> subsets(int[] nums) {
    List<List<Integer>> list = new ArrayList<>();
    Arrays.sort(nums);
    backtrack(list, new ArrayList<>(), nums, 0);
    return list;
}
private void backtrack(List<List<Integer>> list , List<Integer> tempList, int [] nums, int start){
    list.add(new ArrayList<>(tempList));
    for(int i = start; i < nums.length; i++){
        tempList.add(nums[i]);
        backtrack(list, tempList, nums, i + 1);
        tempList.remove(tempList.size() - 1);
    }
}


// Runtime: 1 ms, faster than 60.07% of Java online submissions for Subsets.
// Memory Usage: 39.9 MB, less than 20.20% of Java online submissions for Subsets.
// Time: O(N * 2^N) since the recurrence is T(N) = 2T(N - 1) and we also spend at most O(N) time within a call.
// Space: O(N * 2^N) since there are 2^N subsets. If we only print the result, we just need O(N) space.
private void backtrack(List<List<Integer>> result, List<Integer> numList, int[] nums, int offset) {
  if (offset >= nums.length) return;
  int val = nums[offset];
  // pick
  // add to result
  numList.add(val);
  subsets(offset + 1, nums, numList, result);
  result.add(new ArrayList<>(numList));
  // not pick
  numList.remove(numList.size() - 1);
  subsets(offset + 1, nums, numList, result);
}

```


---

### 90. Subsets II

[90. Subsets II](https://leetcode.com/problems/subsets-ii/)
Given an integer array nums that may contain duplicates, return all possible subsets (the power set).

The solution set must not contain duplicate subsets. Return the solution in any order.

Example 1:

Input: nums = [1,2,2]
Output: [[],[1],[1,2],[1,2,2],[2],[2,2]]

https://leetcode.com/problems/subsets-ii/discuss/388566/Subsets-I-and-II-Java-Solution-with-Detailed-Explanation-and-Comments-(Recursion-and-Iteration)

1. iteration


```java
// Runtime: 2 ms, faster than 43.93% of Java online submissions for Subsets II.
// Memory Usage: 40.9 MB, less than 11.52% of Java online submissions for Subsets II.
// Time: O(N * 2^N)
// Space: O(N * 2^N)
public List<List<Integer>> subsetsWithDup(int[] nums) {
  // sort
  Arrays.sort(nums);
  List<List<Integer>> res = new ArrayList<>();
  res.add(new ArrayList<>());  // empty set

  int cachedSize = 0, startIdx = 0;

  for (int i = 0; i < nums.length; ++i) {
    List<List<Integer>> subres = new ArrayList<>();  // used for new lists
    // set startIdx first before we update cachedSize
    startIdx = (i > 0 && nums[i - 1] == nums[i]) ? cachedSize : 0; // if duplicate occurs
    cachedSize = res.size(); // cache the size for startIdx in the next round
    for (int j = startIdx; j < res.size(); ++j) {
      List<Integer> L = res.get(j);
      L = new ArrayList<>(L);  // copy
      L.add(nums[i]);
      subres.add(L);
    }
    res.addAll(subres);  // concatenate
  }
  return res;
}
```


2. Backtracking


The information of whether it picks or not could be passed down by a boolean parameter isPicked.

If the above condition is satisfied:

Do not add the list to the result list.
Do not do the subproblem after picking the current element.
Only do the subproblem after not picking the current element.

```java
public List<List<Integer>> subsetsWithDup(int[] nums) {
  // sorting
  Arrays.sort(nums);
  List<List<Integer>> result = new ArrayList<>();
  List<Integer> numList = new ArrayList<>();
  result.add(new ArrayList<>());
  subsets(0, nums, numList, result, true);
  return result;
}

private void subsets(int offset, int[] nums, List<Integer> numList, List<List<Integer>> result, boolean isPicked) {
  // base case
  if (offset >= nums.length) return;
  int val = nums[offset];
  // duplicate checking (convert && to ||)
  if (offset == 0 || nums[offset - 1] != nums[offset] || isPicked == true) {
    // pick
    numList.add(val);
    subsets(offset + 1, nums, numList, result, true);
    result.add(new ArrayList<>(numList));  // add to the result list
    numList.remove(numList.size() - 1);
  }
  // not pick
  subsets(offset + 1, nums, numList, result, false);
}


// Time: O(N * 2^N)
// Space: O(N * 2^N)
private void subsets(int offset, int[] nums, List<Integer> numList, List<List<Integer>> result, boolean isPicked) {
  // base case
  if (offset >= nums.length) return;
  int val = nums[offset];
  // not pick
  subsets(offset + 1, nums, numList, result, false);
  // duplicate check
  if (offset >= 1 && nums[offset - 1] == nums[offset] && isPicked == false) return;
  // pick
  numList.add(val);
  subsets(offset + 1, nums, numList, result, true);
  result.add(new ArrayList<>(numList));  // add to the result list
  numList.remove(numList.size() - 1);
}
```



---


### 77. Combinations

[77. Combinations](https://leetcode.com/problems/combinations/)

Given two integers n and k, return all possible combinations of k numbers out of the range [1, n].

You may return the answer in any order.


Example 1:

Input: n = 4, k = 2
Output:
[
  [2,4],
  [3,4],
  [2,3],
  [1,2],
  [1,3],
  [1,4],
]

1. backtracking 典型的回溯算法，
   1. k 限制了树的高度，n 限制了树的宽度，
   2. 直接套我们以前讲过的回溯算法模板框架就行了：


```java
// Runtime: 17 ms, faster than 68.79% of Java online submissions for Combinations.
// Memory Usage: 41.6 MB, less than 35.55% of Java online submissions for Combinations.
/**
 * Backtracking (Recursive Solution)
 *
 * Time complexity = InternalNodes in the RecursionTree   +   K * LeafNodes in RecursionTree
 *                 = (C(N,0) + C(N,1) + ... + C(N,K-1))   +   K * C(N,K)
 *
 * Space Complexity = O(K) -> Depth of Recursion tree + Size of TempList
 *
 * N, K -> Input numbers.
 */
class Solution {
    public List<List<Integer>> combine(int n, int k) {
        List<List<Integer>> res = new LinkedList<>();
        if (k > n || k < 0) return result;
        if(k==0) {
            res.add(new LinkedList<Integer>());
            return res;
        }
        backtracking(res, new LinkedList<Integer>(), 1, n, k);
        return res;
    }
    public void backtracking(List<List<Integer>> res, LinkedList<Integer> curr, int start, int n, int k) {
        if (curr.size()==k) res.add(new LinkedList(curr));
        for(int i=start; i<=n && curr.size()<k; i++){
            curr.add(i);
            backtracking(res, curr, i+1, n, k);
            curr.removeLast();
        }
    }
}



class Solution {
    public List<List<Integer>> combine(int n, int k) {
        List<List<Integer>> res = new ArrayList<>();
        backtracking(res, new ArrayList<>(), 1, n, k);
        return res;
    }

    public void backtracking(List<List<Integer>>res, List<Integer> , 1, int n, int k) {
        if(k==0) {
			combs.add(new ArrayList<Integer>(comb));
			return;
		}
		for(int i=start;i<=n;i++) {
			comb.add(i);
			combine(combs, comb, i+1, n, k-1);
			comb.remove(comb.size()-1);
		}
    }
}


```


```java
// Runtime: 7 ms, faster than 84.10% of Java online submissions for Combinations.
// Memory Usage: 52.6 MB, less than 14.45% of Java online submissions for Combinations.
class Solution {
    public List<List<Integer>> combine(int n, int k) {
        List<List<Integer>> result = new ArrayList<List<Integer>>();
        if (k > n || k < 0) throw new IllegalArgumentException("invalid input");
        if (k == 0) {
            result.add(new ArrayList<Integer>());
            return result;
        }
        // Case I: Add number n to answer
        // Add current element to final solution combine(n-1, k-1)
        result = combine(n - 1, k - 1);
        for (List<Integer> list : result) list.add(n);
        // Case II: Do not add number n to answer
        result.addAll(combine(n - 1, k));
        return result;
    }
}
```






---

# 功能


---


## 设计朋友圈时间线

![design](https://i.imgur.com/FA6dYX3.png)



```java
// Runtime: 9 ms, faster than 75.98% of Java online submissions for Design Twitter.
// Memory Usage: 37.5 MB, less than 44.43% of Java online submissions for Design Twitter.

class Twitter {
    private static int timestamp = 0;

    private static class Tweet {
        private int id;
        private int time;
        private Tweet next;

        // 需要传入推文内容（id）和发文时间
        public Tweet(int id, int time) {
            this.id = id;
            this.time = time;
            this.next = null;
        }
    }

    private static class User {
        private int id;
        public Set<Integer> followed;
        // 用户发表的推文链表头结点
        public Tweet head;

        public User(int userId) {
            followed = new HashSet<>();
            this.id = userId;
            this.head = null;
            // 关注一下自己
            follow(id);
        }

        public void follow(int userId) {
            followed.add(userId);
        }

        public void unfollow(int userId) {
            // 不可以取关自己
            if (userId != this.id)
                followed.remove(userId);
        }

        public void post(int tweetId) {
            Tweet twt = new Tweet(tweetId, timestamp);
            timestamp++;
            // 将新建的推文插入链表头
            // 越靠前的推文 time 值越大
            twt.next = head;
            head = twt;
        }
    }

    // 我们需要一个映射将 userId 和 User 对象对应起来
    private HashMap<Integer, User> userMap = new HashMap<>();

    /** user 发表一条 tweet 动态 */
    public void postTweet(int userId, int tweetId) {
        // 若 userId 不存在，则新建
        if (!userMap.containsKey(userId))
            userMap.put(userId, new User(userId));
        User u = userMap.get(userId);
        u.post(tweetId);
    }

    /** follower 关注 followee */
    public void follow(int followerId, int followeeId) {
        // 若 follower 不存在，则新建
		if(!userMap.containsKey(followerId)){
			User u = new User(followerId);
			userMap.put(followerId, u);
		}
        // 若 followee 不存在，则新建
		if(!userMap.containsKey(followeeId)){
			User u = new User(followeeId);
			userMap.put(followeeId, u);
		}
		userMap.get(followerId).follow(followeeId);
    }

    /** follower 取关 followee，如果 Id 不存在则什么都不做 */
    public void unfollow(int followerId, int followeeId) {
        if (userMap.containsKey(followerId)) {
            User flwer = userMap.get(followerId);
            flwer.unfollow(followeeId);
        }
    }

    /** 返回该 user 关注的人（包括他自己）最近的动态 id，
    最多 10 条，而且这些动态必须按从新到旧的时间线顺序排列。*/
    public List<Integer> getNewsFeed(int userId) {
        List<Integer> res = new ArrayList<>();
        if (!userMap.containsKey(userId)) return res;
        // 关注列表的用户 Id
        Set<Integer> users = userMap.get(userId).followed;
        // 自动通过 time 属性从大到小排序，容量为 users 的大小
        PriorityQueue<Tweet> pq =
            new PriorityQueue<>(users.size(), (a, b)->(b.time - a.time));

        // 先将所有链表头节点插入优先级队列
        for (int id : users) {
            Tweet twt = userMap.get(id).head;
            if (twt == null) continue;
            pq.add(twt);
        }

        while (!pq.isEmpty()) {
            // 最多返回 10 条就够了
            if (res.size() == 10) break;
            // 弹出 time 值最大的（最近发表的）
            Tweet twt = pq.poll();
            res.add(twt.id);
            // 将下一篇 Tweet 插入进行排序
            if (twt.next != null)
                pq.add(twt.next);
        }
        return res;
    }

}
```

---

# 动态规划


求解动态规划的核心问题是穷举。
- 动态规划的穷举 存在「重叠子问题」如果暴力穷举的话效率会极其低下，所以需要「备忘录」或者「DP table」来优化穷举过程，避免不必要的计算。
  - 穷举所有可行解其实并不是一件容易的事，只有列出正确的「状态转移方程」，才能正确地穷举。
- 而且，动态规划问题一定会具备「最优子结构」，才能通过子问题的最值得到原问题的最值。

明确 base case -> 明确「状态」-> 明确「选择」 -> 定义 dp 数组/函数的含义。

---



## 斐波那契数列

[509. Fibonacci Number](https://leetcode.com/problems/fibonacci-number/)

The Fibonacci numbers, commonly denoted F(n) form a sequence, called the Fibonacci sequence, such that each number is the sum of the two preceding ones, starting from 0 and 1. That is,

F(0) = 0, F(1) = 1
F(n) = F(n - 1) + F(n - 2), for n > 1.
Given n, calculate F(n).



Example 1:

Input: n = 2
Output: 1
Explanation: F(2) = F(1) + F(0) = 1 + 0 = 1.


1. 暴力递归

观察递归树，很明显发现了算法低效的原因：存在大量重复计算，比如 f(18) 被计算了两次，而且你可以看到，以 f(18) 为根的这个递归树体量巨大，多算一遍，会耗费巨大的时间。更何况，还不止 f(18) 这一个节点被重复计算，所以这个算法及其低效。

这就是动态规划问题的第一个性质：重叠子问题。下面，我们想办法解决这个问题。

```java
int fib(int N) {
    if (N == 1 || N == 2) return 1;
    return fib(N - 1) + fib(N - 2);
}
```


2. 带备忘录的递归解法
时间复杂度是 O(n)

```java
// Runtime: 0 ms, faster than 100.00% of Java online submissions for Fibonacci Number.
// Memory Usage: 37.3 MB, less than 26.68% of Java online submissions for Fibonacci Number.


class Solution {

    public int fib(int n) {
        // 备忘录全初始化为 0
        int[] memo = new int[ n+ 1];
        // 进行带备忘录的递归
        return helper(memo, n);
    }

    public int helper(int[] memo, int n) {
        // base case
        if (n == 0 || n == 1) return n;

        // 已经计算过，不用再计算了
        if (memo[n] != 0) return memo[n];

        memo[n] = helper(memo, n - 1) + helper(memo, n - 2);
        return memo[n];
    }
}
```


3. dp 数组的迭代解法


```java
// Runtime: 0 ms, faster than 100.00% of Java online submissions for Fibonacci Number.
// Memory Usage: 37.3 MB, less than 26.68% of Java online submissions for Fibonacci Number.

int fib(int N) {
    if (N == 0) return 0;
    int[] dp = new int[N + 1];

    // base case
    dp[0] = 0; dp[1] = 1;

    // 状态转移
    for (int i = 2; i <= N; i++) dp[i] = dp[i - 1] + dp[i - 2];
    return dp[N];
}

```


4. 状态压缩 pre+cur

能够使用状态压缩技巧的动态规划都是二维 dp 问题，你看它的状态转移方程，如果计算状态 dp[i][j] 需要的都是 dp[i][j] 相邻的状态，那么就可以使用状态压缩技巧，将二维的 dp 数组转化成一维，将空间复杂度从 O(N^2) 降低到 O(N)。


```java
int fib(int n) {
    if (n < 1) return 0;
    if (n == 2 || n == 1) return 1;
    int prev = 1, curr = 1;
    for (int i = 3; i <= n; i++) {
        int sum = prev + curr;
        prev = curr;
        curr = sum;
    }
    return curr;
}
```


---

## 动态规划解法


---



### 322. Coin Change 凑零钱 ` for i, for coin, dp[i] = Math.min(dp[i], dp[i-coin]+1);`


[322. Coin Change](https://leetcode.com/problems/coin-change/)

You are given an integer array coins representing coins of different denominations and an integer amount representing a total amount of money.

Return the fewest number of coins that you need to make up that amount. If that amount of money cannot be made up by any combination of the coins, return -1.

You may assume that you have an infinite number of each kind of coin.


Example 1:

Input: coins = [1,2,5], amount = 11
Output: 3
Explanation: 11 = 5 + 5 + 1

1. 从小到大，
   1. 5=coin(4)+1

```java
// Runtime: 28 ms, faster than 33.57% of Java online submissions for Coin Change.
// Memory Usage: 41.4 MB, less than 30.06% of Java online submissions for Coin Change.
// Performance:
// time: O(n * m), n is the amount, m is n of coins
// memory: O(n), n is the amount
public int coinChange(int[] coins, int amount) {
    int[] subCoin = new int[amount+1];
    Arrays.fill(subCoin, -1);
    subCoin[0]=0;
    for(int i=1; i<=amount;i++){
        int minC = Integer.MAX_VALUE;
        for(int coin:coins){
            int rest = i - coin;
            if(rest < 0) continue;
            if(subCoin[rest] ==-1) continue;
            minC = Math.min(minC, subCoin[rest]);
        }
        if(minC==Integer.MAX_VALUE) continue;
        subCoin[i]=minC+1;
    }
    return subCoin[amount];
}

```

#### 暴力解法

```java
int coinChange(int[] coins, int amount) {
    return dp(coins, amount)
}
int dp(int[] coins, int amount) {
    // base case
    if (amount == 0) return 0;
    if (amount < 0) return -1;
    int res = Integer.MAX_VALUE;
    for (int coin : coins) {
        // 计算子问题的结果
        int subProblem = dp(coins, amount - coin);
        // 子问题无解则跳过
        if (subProblem == -1) continue;
        // 在子问题中选择最优解，然后加一
        res = Math.min(res, subProblem + 1);
    }
    return res == Integer.MAX_VALUE ? -1 : res;
}
```

#### best 带备忘录的递归

```java
int[] memo;
int coinChange(int[] coins, int amount) {
    memo = new int[amount + 1];
    // dp 数组全都初始化为特殊值
    Arrays.fill(memo, -666);
    return dp(coins, amount);
}
int dp(int[] coins, int amount) {
    if (amount == 0) return 0;
    if (amount < 0) return -1;
    // 查备忘录，防止重复计算
    if (memo[amount] != -666) return memo[amount];
    int res = Integer.MAX_VALUE;
    for (int coin : coins) {
        // 计算子问题的结果
        int subProblem = dp(coins, amount - coin);
        // 子问题无解则跳过
        if (subProblem == -1) continue;
        // 在子问题中选择最优解，然后加一
        res = Math.min(res, subProblem + 1);
    }
    // 把计算结果存入备忘录
    memo[amount] = (res == Integer.MAX_VALUE) ? -1 : res;
    return memo[amount];
}
```

#### dp 数组的迭代解法

自底向上使用 dp table 来消除重叠子问题


```java
// Runtime: 12 ms, faster than 79.68% of Java online submissions for Coin Change.
// Memory Usage: 38.5 MB, less than 70.58% of Java online submissions for Coin Change.
public int coinChange(int[] coins, int amount) {
    int[] dp = new int[amount+1];
    Arrays.fill(dp,amount+1);
    dp[0]=0;
    for(int i=1; i<=amount;i++){
        for(int coin:coins){
            if(i>=coin) dp[i] = Math.min(dp[i], dp[i-coin]+1);
        }
    }
    return dp[amount]>amount? -1: dp[amount];
}
```



---

#### 983. Minimum Cost For Tickets (Medium)

[983. Minimum Cost For Tickets](https://leetcode.com/problems/minimum-cost-for-tickets/)

You have planned some train traveling one year in advance. The days of the year in which you will travel are given as an integer array days. Each day is an integer from 1 to 365.

Train tickets are sold in three different ways:

a 1-day pass is sold for costs[0] dollars,
a 7-day pass is sold for costs[1] dollars, and
a 30-day pass is sold for costs[2] dollars.
The passes allow that many days of consecutive travel.

For example, if we get a 7-day pass on day 2, then we can travel for 7 days: 2, 3, 4, 5, 6, 7, and 8.
Return the minimum number of dollars you need to travel every day in the given list of days.



Example 1:

Input: days = [1,4,6,7,8,20], costs = [2,7,15]
Output: 11
Explanation: For example, here is one way to buy passes that lets you travel your travel plan:
On day 1, you bought a 1-day pass for costs[0] = $2, which covered day 1.
On day 3, you bought a 7-day pass for costs[1] = $7, which covered days 3, 4, ..., 9.
On day 20, you bought a 1-day pass for costs[0] = $2, which covered day 20.
In total, you spent $11 and covered all the days of your travel.



#### bottom-up dp

O(N) Time | O(N) Space


```java
// Runtime: 1 ms, faster than 81.07% of Java online submissions for Minimum Cost For Tickets.
// Memory Usage: 36.9 MB, less than 65.32% of Java online submissions for Minimum Cost For Tickets.
class Solution {
    int[] dp;
    public int mincostTickets(int[] days, int[] costs) {
        dp = new int[366];
        dp[0]=0; //no days to travel, no need to spend
        return dp(days, costs);
    }

    public int dp(int[] days, int[] costs) {
        int j=0;
        for(int i =1; i<=365; i++){
            if(j >=days.length) break; //when we are done with our travel days we break.
            if(days[j] != i) dp[i] = dp[i-1]; // if the day is not a travel day, put the previous day cost.
            else{
                dp[i] = Math.min(dp[i-1]+costs[0], dp[Math.max(0,i-7)]+ costs[1]);
                dp[i] = Math.min(dp[i], dp[Math.max(0,i-30)]+ costs[2]);
                j++;
            }
        }
        return dp[days[days.length-1]];
    }
}
```


#### Memoization

```java
// Runtime: 1 ms, faster than 81.07% of Java online submissions for Minimum Cost For Tickets.
// Memory Usage: 36.9 MB, less than 65.32% of Java online submissions for Minimum Cost For Tickets.
class Solution {
    int[] dp;
    public int mincostTickets(int[] days, int[] costs) {
        dp = new int[366];
        Arrays.fill(dp, -1);
        dp[0]=0; //no days to travel, no need to spend
        return dp(days, costs);
    }

    public int dp(int[] days, int[] costs) {
        int j=0;
        for(int i =1; i<=365; i++){
            if(j >=days.length) break; //when we are done with our travel days we break.
            if(dp[i]!= -1) continue;
            if(days[j] != i) dp[i] = dp[i-1]; // if the day is not a travel day, put the previous day cost.
            else{
                dp[i] = Math.min(dp[i-1]+costs[0], dp[Math.max(0,i-7)]+ costs[1]);
                dp[i] = Math.min(dp[i], dp[Math.max(0,i-30)]+ costs[2]);
                j++;
            }
        }
        return dp[days[days.length-1]];
    }
}
```


---


### 64. Minimum Path Sum 最小路径和（中等）

[64. Minimum Path Sum](https://leetcode.com/problems/minimum-path-sum/)

Given a m x n grid filled with non-negative numbers, find a path from top left to bottom right, which minimizes the sum of all numbers along its path.

Note: You can only move either down or right at any point in time.

1. 自顶向下动态规划解法 `int[][] memo, memo[i][j] = Math.min( dp(grid, i - 1, j), dp(grid, i, j - 1) ) + grid[i][j];)`
   1. 时间复杂度和空间复杂度都是 `O(MN)`，标准的自顶向下动态规划解法。
   2. 一般来说，让你在二维矩阵中求最优化问题（最大值或者最小值），肯定需要递归 + 备忘录，也就是动态规划技巧。
   3. 从 D 走到 A 的最小路径和是 6，而从 D 走到 C 的最小路径和是 8，6 小于 8，所以一定要从 A 走到 B 才能使路径和最小。
   4. 我们把「从 D 走到 B 的最小路径和」这个问题转化成了
   5. 「从 D 走到 A 的最小路径和」和 「从 D 走到 C 的最小路径和」这两个问题。
   6. 从左上角位置 (0, 0) 走到位置 (i, j) 的最小路径和为 dp(grid, i, j)。
   7. dp(grid, i, j) 的值取决于 dp(grid, i - 1, j) 和 dp(grid, i, j - 1) 返回的值。

```java
// Runtime: 2 ms, faster than 84.10% of Java online submissions for Minimum Path Sum.
// Memory Usage: 43 MB, less than 19.44% of Java online submissions for Minimum Path Sum.

int[][] memo;

int minPathSum(int[][] grid) {
    int m = grid.length;
    int n = grid[0].length;
    memo = new int[m][n];
    for(int row:memo) Arrays.fill(row, -1)
    // 计算从左上角走到右下角的最小路径和
    return dp(grid, m - 1, n - 1);
}

int dp(int[][] grid, int i, int j) {
    // base case
    if(i==0 && j==0) return grid[0][0];

    // 如果索引出界，返回一个很大的值，
    // 保证在取 min 的时候不会被取到
    if(i<0 || j<0) return Integer.MAX_VALUE;

    if(memo[i][j]!=-1) return memo[i][j];

    // 左边和上面的最小路径和加上 grid[i][j]
    // 就是到达 (i, j) 的最小路径和
    memo[i][j] = Math.min(
        dp(grid, i - 1, j),
        dp(grid, i, j - 1)
    ) + grid[i][j];

    return  memo[i][j];
}
```


1. 重叠子问题, 自底向上的迭代解法 `int[][] memo, memo[i][j] = Math.min(memo[i-1][j], memo[i][j-1] ) + grid[i][j];)`

```java
// Runtime: 4 ms, faster than 21.15% of Java online submissions for Minimum Path Sum.
// Memory Usage: 43 MB, less than 19.44% of Java online submissions for Minimum Path Sum.
public int minPathSum(int[][] grid) {
    int m=grid.length;
    int n=grid[0].length;
    int[][] memo = new int[m][n];

    memo[0][0] = grid[0][0];

    for(int i=1;i<m;i++) memo[i][0] = memo[i-1][0] + grid[i][0];
    for(int j=1;j<n;j++) memo[0][j] = memo[0][j-1] + grid[0][j];

    // 状态转移
    for (int i = 1; i < m; i++) {
        for (int j = 1; j < n; j++) {
            memo[i][j] = Math.min(
                memo[i-1][j],
                memo[i][j-1]
            ) + grid[i][j];
        }
    }
    return memo[m-1][n-1];
}
```


---


### 931. Minimum Falling Path Sum 下降路径最小和

[931. Minimum Falling Path Sum](https://leetcode.com/problems/minimum-falling-path-sum/)

Given an n x n array of integers matrix, return the minimum sum of any falling path through matrix.

A falling path starts at any element in the first row and chooses the element in the next row that is either directly below or diagonally left/right. Specifically, the next element from position (row, col) will be (row + 1, col - 1), (row + 1, col), or (row + 1, col + 1).


1. 暴力穷举解法

```java
int minFallingPathSum(int[][] matrix) {
    int n = matrix.length;
    int res = Integer.MAX_VALUE;
    // 终点可能在最后一行的任意一列
    for (int j = 0; j < n; j++) res = Math.min(res, dp(matrix, n - 1, j));
    return res;
}

int dp(int[][] matrix, int i, int j) {
    // 非法索引检查
    if (i < 0 || j < 0 ||
        i >= matrix.length ||
        j >= matrix[0].length) {
        // 返回一个特殊值
        return 99999;
    }
    // base case
    if (i == 0) return matrix[i][j];

    // 状态转移
    return matrix[i][j] + min(
            dp(matrix, i - 1, j),
            dp(matrix, i - 1, j - 1),
            dp(matrix, i - 1, j + 1)
        );
}

int min(int a, int b, int c) {
    return Math.min(a, Math.min(b, c));
}

```


2. 用备忘录的方法消除重叠子问题

```java
// Runtime: 3 ms, faster than 84.22% of Java online submissions for Minimum Falling Path Sum.
// Memory Usage: 45 MB, less than 5.34% of Java online submissions for Minimum Falling Path Sum.

// 备忘录
int[][] memo;

int minFallingPathSum(int[][] matrix) {
    int n = matrix.length;
    int res = Integer.MAX_VALUE;
    // 备忘录里的值初始化为 66666
    memo = new int[n][n];
    for (int i = 0; i < n; i++) Arrays.fill(memo[i], 66666);
    // 终点可能在 matrix[n-1] 的任意一列
    for (int j = 0; j < n; j++) res = Math.min(res, dp(matrix, n - 1, j));
    return res;
}

int dp(int[][] matrix, int i, int j) {
    // 1、索引合法性检查
    if (i < 0 || j < 0 ||
        i >= matrix.length ||
        j >= matrix[0].length) {
        return 99999;
    }

    // 2、base case
    if (i == 0) return matrix[0][j];

    // 3、查找备忘录，防止重复计算
    if (memo[i][j] != 66666) return memo[i][j];

    // 进行状态转移
    memo[i][j] = matrix[i][j] + min(
        dp(matrix, i - 1, j),
        dp(matrix, i - 1, j - 1),
        dp(matrix, i - 1, j + 1)
    );
    return memo[i][j];
}

int min(int a, int b, int c) {
    return Math.min(a, Math.min(b, c));
}
```



---

### 174. Dungeon Game 地下城游戏 ????????????

[174. Dungeon Game](https://leetcode.com/problems/dungeon-game/)

The demons had captured the princess and imprisoned her in the bottom-right corner of a dungeon. The dungeon consists of m x n rooms laid out in a 2D grid. Our valiant knight was initially positioned in the top-left room and must fight his way through dungeon to rescue the princess.

The knight has an initial health point represented by a positive integer. If at any point his health point drops to 0 or below, he dies immediately.

Some of the rooms are guarded by demons (represented by negative integers), so the knight loses health upon entering these rooms; other rooms are either empty (represented as 0) or contain magic orbs that increase the knight's health (represented by positive integers).

To reach the princess as quickly as possible, the knight decides to move only rightward or downward in each step.

Return the knight's minimum initial health so that he can rescue the princess.

Note that any room can contain threats or power-ups, even the first room the knight enters and the bottom-right room where the princess is imprisoned.

```java
/* 主函数 */
int calculateMinimumHP(int[][] grid) {
    int m = grid.length;
    int n = grid[0].length;
    // 备忘录中都初始化为 -1
    memo = new int[m][n];
    for (int[] row : memo) {
        Arrays.fill(row, -1);
    }

    return dp(grid, 0, 0);
}

// 备忘录，消除重叠子问题
int[][] memo;

/* 定义：从 (i, j) 到达右下角，需要的初始血量至少是多少 */
int dp(int[][] grid, int i, int j) {
    int m = grid.length;
    int n = grid[0].length;
    // base case
    if (i == m - 1 && j == n - 1) return grid[i][j] >= 0 ? 1 : -grid[i][j] + 1;
    if (i == m || j == n) return Integer.MAX_VALUE;
    // 避免重复计算
    if (memo[i][j] != -1) return memo[i][j];
    // 状态转移逻辑
    int res = Math.min(
            dp(grid, i, j + 1),
            dp(grid, i + 1, j)
        ) - grid[i][j];
    // 骑士的生命值至少为 1
    memo[i][j] = res <= 0 ? 1 : res;

    return memo[i][j];
}
```

---

### 514. Freedom Trail 自由之路（困难）??????

[514. Freedom Trail](https://leetcode.com/problems/freedom-trail/)

In the video game Fallout 4, the quest "Road to Freedom" requires players to reach a metal dial called the "Freedom Trail Ring" and use the dial to spell a specific keyword to open the door.

Given a string ring that represents the code engraved on the outer ring and another string key that represents the keyword that needs to be spelled, return the minimum number of steps to spell all the characters in the keyword.

Initially, the first character of the ring is aligned at the "12:00" direction. You should spell all the characters in key one by one by rotating ring clockwise or anticlockwise to make each character of the string key aligned at the "12:00" direction and then by pressing the center button.

At the stage of rotating the ring to spell the key character key[i]:

You can rotate the ring clockwise or anticlockwise by one place, which counts as one step. The final purpose of the rotation is to align one of ring's characters at the "12:00" direction, where this character must equal key[i].
If the character key[i] has been aligned at the "12:00" direction, press the center button to spell, which also counts as one step. After the pressing, you could begin to spell the next character in the key (next stage). Otherwise, you have finished all the spelling.


Example 1:
Input: ring = "godding", key = "gd"
Output: 4
Explanation:
For the first key character 'g', since it is already in place, we just need 1 step to spell this character.
For the second key character 'd', we need to rotate the ring "godding" anticlockwise by two steps to make it become "ddinggo".
Also, we need 1 more step for spelling.
So the final output is 4.
Example 2:
Input: ring = "godding", key = "godding"
Output: 13

- 遇到求最值的问题，基本都是由动态规划算法来解决，
- 动态规划本身就是运筹优化算法的一种
- 状态是什么？状态就是「下一个需要弹奏的音符」和「当前的手的状态」。
- 选择是什么？选择就是「下一个音符应该由哪个手指头来弹」，无非就是穷举五个手指头。结合当前手的状态，做出每个选择需要对应代价的，刚才说过这个代价是因人而异的，所以我需要给自己定制一个损失函数，计算不同指法切换的「别扭程度」。
- 现在的问题就变成了一个标准的动态规划问题，根据损失函数做出「别扭程度」最小的选择，使得整段演奏最流畅……


- 题目给你输入一个字符串 ring 代表圆盘上的字符（指针位置在 12 点钟方向，初始指向 ring[0]），再输入一个字符串 key 代表你需要拨动圆盘输入的字符串，你的算法需要返回输入这个 key 至少进行多少次操作（拨动一格圆盘和按下圆盘中间的按钮都算是一次操作）。
- 原题可以转化为：圆盘固定，我们可以拨动指针；现在需要我们拨动指针并按下按钮，以最少的操作次数输入 key 对应的字符串。
- 「状态」就是「当前需要输入的字符」和「当前圆盘指针的位置」。
- 「状态」就是 i 和 j 两个变量。
- 用 i 表示当前圆盘上指针指向的字符（也就是 ring[i]）；
- 用 j 表示需要输入的字符（也就是 key[j]）。
- 当圆盘指针指向 ring[i] 时，输入字符串 key[j..] 至少需要 dp(ring, i, key, j) 次操作。


```java
public int findRotateSteps(String ring, String key) {
    int[][] memo = new int[ring.length()][key.length()];
    // the first key is the position on the ring we are on, the second is how many letters we have completed!
    return dfs(ring, key, 0, 0);
}

// pos represents the pos of the ring we are sitting on, and the steps represents how many letters we've gone through
public int dfs(String ring, String key, int pos, int steps) {
    // have we already analyzed one path? -- no point moving further left or right from this position        
    if (memo[pos][steps] != 0) return memo[pos][steps];

    boolean clockWise = false, counterClockWise = false;
    int clockSteps = Integer.MAX_VALUE, counterClockSteps = Integer.MAX_VALUE;

    int n = ring.length();

    for (int i = 0; i < n; i++) {
        int curr = (i + pos) % n; // handles loop
        if (!clockWise && ring.charAt(curr) == key.charAt(steps)) {
            clockSteps = i + dfs(ring, key, curr, steps + 1);
            clockWise = true;
        }

        int curr2 = pos - i;
        if (curr2 < 0) curr2 = n + pos - i; // every position we are negative is one off of the length!

        if (!counterClockWise && ring.charAt(curr2) == key.charAt(steps)) {
            counterClockSteps = i + dfs(ring, key, curr2, steps + 1);
            counterClockWise = true;
        }

        if (clockWise && counterClockWise) break;
        return this.memo[pos][steps] = Math.min(clockSteps, counterClockSteps) + 1; // +1 to click the button
    }
}
}
```


```java
// 字符 -> 索引列表
unordered_map<char, vector<int>> charToIndex;
// 备忘录
vector<vector<int>> memo;

/* 主函数 */
int findRotateSteps(string ring, string key) {
    int m = ring.size();
    int n = key.size();
    // 备忘录全部初始化为 0
    memo.resize(m, vector<int>(n, 0));
    // 记录圆环上字符到索引的映射
    for (int i = 0; i < ring.size(); i++) {
        charToIndex[ring[i]].push_back(i);
    }
    // 圆盘指针最初指向 12 点钟方向，
    // 从第一个字符开始输入 key
    return dp(ring, 0, key, 0);
}

// 计算圆盘指针在 ring[i]，输入 key[j..] 的最少操作数
int dp(string& ring, int i, string& key, int j) {
    // base case 完成输入
    if (j == key.size()) return 0;
    // 查找备忘录，避免重叠子问题
    if (memo[i][j] != 0) return memo[i][j];

    int n = ring.size();
    // 做选择
    int res = INT_MAX;
    // ring 上可能有多个字符 key[j]
    for (int k : charToIndex[key[j]]) {
        // 拨动指针的次数
        int delta = abs(k - i);
        // 选择顺时针还是逆时针
        delta = min(delta, n - delta);
        // 将指针拨到 ring[k]，继续输入 key[j+1..]
        int subProblem = dp(ring, k, key, j + 1);
        // 选择「整体」操作次数最少的
        // 加一是因为按动按钮也是一次操作
        res = min(res, 1 + delta + subProblem);
    }
    // 将结果存入备忘录
    memo[i][j] = res;
    return res;
}
```

---

## 加权有向图 最短路径

### 787. K 站中转内最便宜的航班（中等）

[787. Cheapest Flights Within K Stops](https://leetcode.com/problems/cheapest-flights-within-k-stops/)

There are n cities connected by some number of flights. You are given an array flights where flights[i] = [fromi, toi, pricei] indicates that there is a flight from city fromi to city toi with cost pricei.

You are also given three integers src, dst, and k, return the cheapest price from src to dst with at most k stops. If there is no such route, return -1.

Example 1:
Input: n = 3, flights = [[0,1,100],[1,2,100],[0,2,500]], src = 0, dst = 2, k = 1
Output: 200
Explanation: The graph is shown.
The cheapest price from city 0 to city 2 with at most 1 stop costs 200, as marked red in the picture.


- 一幅加权有向图，让你求 src 到 dst 权重最小的一条路径，同时要满足，这条路径最多不能超过 K + 1 条边（经过 K 个节点相当于经过 K + 1 条边。


1. BFS 算法

- 对于加权图的场景，我们需要优先级队列「自动排序」的特性，将路径权重较小的节点排在队列前面，以此为基础施展 BFS 算法。



2. 动态规划思路


```java
minPath(src, dst) = min(
    minPath(src, s1) + w1,
    minPath(src, s2) + w2
)

dp(dst, k) = min(
    dp(s1, k - 1) + w1,
    dp(s2, k - 1) + w2
)

// Time Limit Exceeded
HashMap<Integer, List<int[]>> indegree;
int src, dst;

public int findCheapestPrice(int n, int[][] flights, int src, int dst, int K) {
    k++;
    this.src=src;
    this.dst=dst;
    indegree=new HashMap<>();
    for(int[] f:flights){
        int from = f[0];
        int to = f[1];
        int price = f[2];
        // 记录谁指向该节点，以及之间的权重
        indegree.putIfAbsent(to, new LinkedList<>());
        indegree.get(to).add(new int[]{from, price});
    }
    return dp(dst,k)
}

// 定义：从 src 出发，k 步之内到达 s 的最短路径权重
int dp(int s, int k) {
    if(s==this.src) return 0;
    if(k==0) return -1;

    // 初始化为最大值，方便等会取最小值
    int res = Integer.MAX_VALUE;
    if(indegree.containsKey(s)){
        // 当 s 有入度节点时，分解为子问题
        for (int[] v : indegree.get(s)) {
            int from = v[0];
            int price = v[1];

            // 从 src 到达相邻的入度节点所需的最短路径权重
            int subProblem = dp(from, k - 1);
            // 跳过无解的情况
            if (subProblem != -1) res = Math.min(res, subProblem + price);
        }
    }
    // 如果还是初始值，说明此节点不可达
    return res == Integer.MAX_VALUE ? -1 : res;
}
```

2. memo

```java
// Runtime: 7 ms, faster than 47.07% of Java online submissions for Cheapest Flights Within K Stops.
// Memory Usage: 39.3 MB, less than 98.25% of Java online submissions for Cheapest Flights Within K Stops.


HashMap<Integer, List<int[]>> indegree;
int src, dst;
// 备忘录
int[][] memo;

public int findCheapestPrice(int n, int[][] flights, int src, int dst, int K) {
    k++;
    this.src=src;
    this.dst=dst;
    // 初始化备忘录，全部填一个特殊值
    memo = new int[n][K + 1];
    for (int[] row : memo) Arrays.fill(row, -888);
    indegree=new HashMap<>();
    for(int[] f:flights){
        int from = f[0];
        int to = f[1];
        int price = f[2];
        // 记录谁指向该节点，以及之间的权重
        indegree.putIfAbsent(to, new LinkedList<>());
        indegree.get(to).add(new int[]{from, price});
    }
    return dp(dst,k)
}

// 定义：从 src 出发，k 步之内到达 s 的最短路径权重
int dp(int s, int k) {
    if(s==this.src) return 0;
    if(k==0) return -1;
    // 查备忘录，防止冗余计算
    if (memo[s][k] != -888) return memo[s][k];

    // 初始化为最大值，方便等会取最小值
    int res = Integer.MAX_VALUE;
    if(indegree.containsKey(s)){
        // 当 s 有入度节点时，分解为子问题
        for (int[] v : indegree.get(s)) {
            int from = v[0];
            int price = v[1];

            // 从 src 到达相邻的入度节点所需的最短路径权重
            int subProblem = dp(from, k - 1);
            // 跳过无解的情况
            if (subProblem != -1) res = Math.min(res, subProblem + price);
        }
    }
    // 存入备忘录
    memo[s][k] = res == Integer.MAX_VALUE ? -1 : res;
    return memo[s][k];
}
```





---


## 子序列

一个字符串，它的子序列有多少种可能？起码是指数级的吧，这种情况下，不用动态规划技巧，还想怎么着呢？

既然要用动态规划，那就要定义 dp 数组，找状态转移关系。


1. 一个一维的 dp 数组：

```java
int n = array.length;
int[] dp = new int[n];

for (int i = 1; i < n; i++) {
    for (int j = 0; j < i; j++) {
        dp[i] = 最值(dp[i], dp[j] + ...)
    }
}
```


2. 二维的 dp 数组：

```java
int n = arr.length;
int[][] dp = new dp[n][n];

for (int i = 0; i < n; i++) {
    for (int j = 1; j < n; j++) {
        if (arr[i] == arr[j])
            dp[i][j] = dp[i][j] + ...
        else
            dp[i][j] = 最值(...)
    }
}
```

这种思路运用相对更多一些，尤其是涉及两个字符串/数组的子序列。

dp 数组的含义
1. 涉及两个字符串/数组时（比如最长公共子序列）
   1. 在子数组arr1[0..i]和子数组arr2[0..j]中
   2. 我们要求的子序列（最长公共子序列）长度为dp[i][j]。

2. 只涉及一个字符串/数组时（比如本文要讲的最长回文子序列）
   1. 在子数组array[i..j]中
   2. 我们要求的子序列（最长回文子序列）的长度为dp[i][j]。



---

### 300. Longest Increasing Subsequence 最长递增子序列

[300. Longest Increasing Subsequence](https://leetcode.com/problems/longest-increasing-subsequence/)

Given an integer array nums, return the length of the longest strictly increasing subsequence.

A subsequence is a sequence that can be derived from an array by deleting some or no elements without changing the order of the remaining elements. For example, [3,6,2,7] is a subsequence of the array [0,3,1,6,2,2,7].

Example 1:

Input: nums = [10,9,2,5,3,7,101,18]
Output: 4
Explanation: The longest increasing subsequence is [2,3,7,101], therefore the length is 4.


1. 动态规划解法


```java
public int lengthOfLIS(int[] nums) {
    int[] dp = new int[nums.length];
    Arrays.fill(dp, 1);
    for(int i=1;i<nums.length;i++){
        dp[i]=1;
        for(int j=0;j<i;j++){
            if(nums[i]>nums[j]) dp[i]= Math.max(dp[j]+1,dp[i]);
        }
    }
    int res = 0;
    for(int num:dp) res=Math.max(res, num);
    return res;
}
```

---

### 1143. Longest Common Subsequence 最长公共子序列

[1143. Longest Common Subsequence](https://leetcode.com/problems/longest-common-subsequence/)

Given two strings text1 and text2, return the length of their longest common subsequence. If there is no common subsequence, return 0.

A subsequence of a string is a new string generated from the original string with some characters (can be none) deleted without changing the relative order of the remaining characters.

For example, "ace" is a subsequence of "abcde".
A common subsequence of two strings is a subsequence that is common to both strings.


Example 1:

Input: text1 = "abcde", text2 = "ace"
Output: 3
Explanation: The longest common subsequence is "ace" and its length is 3.


1. 暴力算法
   1. 把 s1 和 s2 的所有子序列都穷举出来，
   2. 看有没有公共的，
   3. 然后在所有公共子序列里面再寻找一个长度最大的。
   4. 复杂度就是指数级的，不实际。

2. 不考虑整个字符串，细化到s1和s2的每个字符

1. 用memo备忘录消除子问题

```java
// 备忘录，消除重叠子问题
int[][] memo;

/* 主函数 */
int longestCommonSubsequence(String s1, String s2) {
    int m = s1.length(), n = s2.length();
    // 备忘录值为 -1 代表未曾计算
    memo = new int[m][n];
    for (int[] row : memo) Arrays.fill(row, -1);
    // 计算 s1[0..] 和 s2[0..] 的 lcs 长度
    return dp(s1, 0, s2, 0);
}

/* 主函数 */
int dp(String s1, int i, String s2, int j) {
    // base case
    // s1[i..]或s2[j..]就相当于空串了，最长公共子序列的长度显然是 0
    if (i == s1.length() || j == s2.length()) return 0;

    // 如果之前计算过，则直接返回备忘录中的答案
    if (memo[i][j] != -1) return memo[i][j];

    // 根据 s1[i] 和 s2[j] 的情况做选择
    // s1[i] 和 s2[j] 必然在 lcs 中
    if (s1.charAt(i) == s2.charAt(j)) memo[i][j] = 1 + dp(s1, i + 1, s2, j + 1);
    // s1[i] 和 s2[j] 至少有一个不在 lcs 中
    else memo[i][j] = Math.max( dp(s1, i + 1, s2, j), dp(s1, i, s2, j + 1) );

    return memo[i][j];
```

2. 自底向上的迭代的动态规划思路


```java
// Runtime: 9 ms, faster than 89.16% of Java online submissions for Longest Common Subsequence.
// Memory Usage: 42.9 MB, less than 64.72% of Java online submissions for Longest Common Subsequence.

int longestCommonSubsequence(String s1, String s2) {
    int m = s1.length(), n = s2.length();
    int[][] dp = new int[m + 1][n + 1];
    // 定义：s1[0..i-1] 和 s2[0..j-1] 的 lcs 长度为 dp[i][j]
    // 目标：s1[0..m-1] 和 s2[0..n-1] 的 lcs 长度，即 dp[m][n]
    // base case: dp[0][..] = dp[..][0] = 0

    for (int i = 1; i <= m; i++) {
        for (int j = 1; j <= n; j++) {

            // 现在 i 和 j 从 1 开始，所以要减一

            // s1[i-1] 和 s2[j-1] 必然在 lcs 中
            if (s1.charAt(i - 1) == s2.charAt(j - 1)) dp[i][j] = 1 + dp[i - 1][j - 1];
            // s1[i-1] 和 s2[j-1] 至少有一个不在 lcs 中
            else dp[i][j] = Math.max(dp[i][j - 1], dp[i - 1][j]);
        }
    }
    return dp[m][n];
}
```


---

### 583. Delete Operation for Two Strings 两个字符串的删除操作

[583. Delete Operation for Two Strings](https://leetcode.com/problems/delete-operation-for-two-strings/)
- Given two strings word1 and word2, return the minimum number of steps required to make word1 and word2 the same.
- In one step, you can delete exactly one character in either string.
- 要计算删除的次数，就可以通过最长公共子序列的长度推导出来
- 删除的结果就是它俩的最长公共子序列

Example 1:
Input: word1 = "sea", word2 = "eat"

Output: 2
Explanation: You need one step to make "sea" to "ea" and another step to make "eat" to "ea".


```java
// Runtime: 7 ms, faster than 90.79% of Java online submissions for Delete Operation for Two Strings.
// Memory Usage: 39.2 MB, less than 97.09% of Java online submissions for Delete Operation for Two Strings.

class Solution {
    public int minDistance(String word1, String word2) {
        int m = word1.length(), n = word2.length();
        // 复用前文计算 lcs 长度的函数
        int lcs = longestCommonSubsequence(word1, word2);
        return m - lcs + n - lcs;
    }

    // 最长公共子序列的长度
    public int longestCommonSubsequence(String s1, String s2) {
        int m = s1.length(), n = s2.length();
        int[][] dp = new int[m + 1][n + 1];
        for (int i = 1; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
                // 现在 i 和 j 从 1 开始，所以要减一
                // s1[i-1] 和 s2[j-1] 必然在 lcs 中
                if (s1.charAt(i - 1) == s2.charAt(j - 1)) dp[i][j] = 1 + dp[i - 1][j - 1];
                // s1[i-1] 和 s2[j-1] 至少有一个不在 lcs 中
                else dp[i][j] = Math.max(dp[i][j - 1], dp[i - 1][j]);
            }
        }
        return dp[m][n];
    }
}
```

---

### 712. Minimum ASCII Delete Sum for Two Strings 最小 ASCII 删除和

[712. Minimum ASCII Delete Sum for Two Strings](https://leetcode.com/problems/minimum-ascii-delete-sum-for-two-strings/submissions/)

Given two strings s1 and s2, return the lowest ASCII sum of deleted characters to make two strings equal.

Example 1:

Input: s1 = "sea", s2 = "eat"
Output: 231
Explanation: Deleting "s" from "sea" adds the ASCII value of "s" (115) to the sum.
Deleting "t" from "eat" adds 116 to the sum.
At the end, both strings are equal, and 115 + 116 = 231 is the minimum sum possible to achieve this.

```java
// Runtime: 32 ms, faster than 36.57% of Java online submissions for Minimum ASCII Delete Sum for Two Strings.
// Memory Usage: 39.5 MB, less than 80.73% of Java online submissions for Minimum ASCII Delete Sum for Two Strings.

// 备忘录
int memo[][];

/* 主函数 */
int minimumDeleteSum(String s1, String s2) {
    int m = s1.length(), n = s2.length();
    // 备忘录值为 -1 代表未曾计算
    memo = new int[m][n];
    for (int[] row : memo) Arrays.fill(row, -1);
    return dp(s1, 0, s2, 0);
}

// 定义：将 s1[i..] 和 s2[j..] 删除成相同字符串，
// 最小的 ASCII 码之和为 dp(s1, i, s2, j)。
int dp(String s1, int i, String s2, int j) {
    int res = 0;
    // base case
    if (i == s1.length()) {
        // 如果 s1 到头了，那么 s2 剩下的都得删除
        for (; j < s2.length(); j++) res += s2.charAt(j);
        return res;
    }
    if (j == s2.length()) {
        // 如果 s2 到头了，那么 s1 剩下的都得删除
        for (; i < s1.length(); i++) res += s1.charAt(i);
        return res;
    }
    if (memo[i][j] != -1) return memo[i][j];

    // s1[i] 和 s2[j] 都是在 lcs 中的，不用删除
    if (s1.charAt(i) == s2.charAt(j)) memo[i][j] = dp(s1, i + 1, s2, j + 1);
    // s1[i] 和 s2[j] 至少有一个不在 lcs 中，删一个
    else {
        memo[i][j] = Math.min(
            s1.charAt(i) + dp(s1, i + 1, s2, j),
            s2.charAt(j) + dp(s1, i, s2, j + 1)
        );
    }
    return memo[i][j];
}

```

---

### 5. Longest Palindromic Substring 最长回文子序列


[5. Longest Palindromic Substring]

Given a string s, return the longest palindromic substring in s.

Example 1:

Input: s = "babad"
Output: "bab"
Note: "aba" is also a valid answer.

这个问题对 dp 数组的定义是：在子串s[i..j]中，最长回文子序列的长度为dp[i][j]。一定要记住这个定义才能理解算法。


```java
class Solution {

    private int maxLen = 0;
    private int lo = 0;

    public String longestPalindrome(String s) {
        int len=s.length();
        if(len<2) return s;
        for(int i=0;i<len-1;i++){
            checkpalin(s,i,i);
            checkpalin(s,i,i+1);
        }
        return s.substring(lo, lo+maxLen);
    }

    public void checkpalin(String s, int i, int j) {
        while(i>=0 && j<s.length() && s.charAt(i)==s.charAt(j)){
            i--;
            j++;
        }
        if(maxLen < j-i-1){
            lo = i+1;
            maxLen = j-i-1;

        }
    }
}
```



---

### 516. Longest Palindromic Subsequence 最长回文子序列长度


[516. Longest Palindromic Subsequence](https://leetcode.com/problems/longest-palindromic-subsequence/)
- Given a string s, find the longest palindromic subsequence's length in s.
- A subsequence is a sequence that can be derived from another sequence by deleting some or no elements without changing the order of the remaining elements.

Example 1:

Input: s = "bbbab"
Output: 4
Explanation: One possible longest palindromic subsequence is "bbbb".


```java
// Runtime: 41 ms, faster than 67.06% of Java online submissions for Longest Palindromic Subsequence.
// Memory Usage: 49.1 MB, less than 71.59% of Java online submissions for Longest Palindromic Subsequence.

class Solution {
    public int longestPalindromeSubseq(String s) {
        int m = s.length();
        int[][] memo = new int[m][m];

        for(int i=0;i<m;i++) memo[i][i] = 1;

        for(int i=m-1;i>=0;i--){
            for(int j=i+1;j<m;j++){
                if(s.charAt(i)==s.charAt(j)) memo[i][j] = memo[i+1][j-1] +2;
                else memo[i][j] = Math.max(
                    memo[i][j-1],
                    memo[i+1][j]
                );
            }
        }
        return memo[0][m-1];
    }
}
```

---

### 494. Target Sum 目标和

[494. Target Sum](https://leetcode.com/problems/target-sum/)

You are given an integer array nums and an integer target.

You want to build an expression out of nums by adding one of the symbols '+' and '-' before each integer in nums and then concatenate all the integers.

For example, if nums = [2, 1], you can add a '+' before 2 and a '-' before 1 and concatenate them to build the expression "+2-1".
Return the number of different expressions that you can build, which evaluates to target.

Example 1:

Input: nums = [1,1,1,1,1], target = 3
Output: 5
Explanation: There are 5 ways to assign symbols to make the sum of nums be target 3.
-1 + 1 + 1 + 1 + 1 = 3
+1 - 1 + 1 + 1 + 1 = 3
+1 + 1 - 1 + 1 + 1 = 3
+1 + 1 + 1 - 1 + 1 = 3
+1 + 1 + 1 + 1 - 1 = 3



#### 回溯思路
任何算法的核心都是穷举，回溯算法就是一个暴力穷举算法

```java
// Runtime: 556 ms, faster than 13.66% of Java online submissions for Target Sum.
// Memory Usage: 36.5 MB, less than 84.78% of Java online submissions for Target Sum.

int result = 0;

/* 主函数 */
int findTargetSumWays(int[] nums, int target) {
    if (nums.length == 0) return 0;
    backtrack(nums, 0, target);
    return result;
}

/* 回溯算法模板 */
void backtrack(int[] nums, int i, int rest) {
    // base case
    if (i == nums.length) {
        // 说明恰好凑出 target
        if (rest == 0) result++;
        return;
    }

    // 给 nums[i] 选择 - 号
    rest += nums[i];
    // 穷举 nums[i + 1]
    backtrack(nums, i + 1, rest);
    // 撤销选择
    rest -= nums[i];

    // 给 nums[i] 选择 + 号
    rest -= nums[i];
    // 穷举 nums[i + 1]
    backtrack(nums, i + 1, rest);
    // 撤销选择
    rest += nums[i];
}
```


#### 消除重叠子问题

```java
int findTargetSumWays(int[] nums, int target) {
    if (nums.length == 0) return 0;
    return dp(nums, 0, target);
}

// 备忘录
HashMap<String, Integer> memo = new HashMap<>();

int dp(int[] nums, int i, int rest) {
    // base case
    if (i == nums.length) {
        if (rest == 0) return 1;
        return 0;
    }
    // 把它俩转成字符串才能作为哈希表的键
    String key = i + "," + rest;
    // 避免重复计算
    if (memo.containsKey(key)) {
        return memo.get(key);
    }
    // 还是穷举
    int result = dp(nums, i + 1, rest - nums[i]) + dp(nums, i + 1, rest + nums[i]);
    // 记入备忘录
    memo.put(key, result);
    return result;
}
```

---


### 72. Edit Distance 编辑距离（困难）

![dp](https://i.imgur.com/JWV0ewv.jpg)

[72. Edit Distance](https://leetcode.com/problems/edit-distance/)

Given two strings word1 and word2, return the minimum number of operations required to convert word1 to word2.

You have the following three operations permitted on a word:

Insert a character
Delete a character
Replace a character


Example 1:

Input: word1 = "horse", word2 = "ros"
Output: 3
Explanation:
horse -> rorse (replace 'h' with 'r')
rorse -> rose (remove 'r')
rose -> ros (remove 'e')


1. 暴力解法，存在重叠子问题，需要用动态规划技巧来优化。


```java
class Solution {
    public int minDistance(String word1, String word2) {
        return dp(word1, word1.length()-1, word2, word2.length()-1);
    }

    public int dp(String word1, int i, String word2, int j) {
        if(i==-1) return j+1;
        if(j==-1) return i+1;
        if(word1.charAt(i)==word2.charAt(j)) return dp(word1, i-1, word2, j-1);
        else return 1+min(
            dp(word1, i, word2, j-1),
            dp(word1, i-1, word2, j),
            dp(word1, i-1, word2, j-1)
        );
    }

    public int min(int x, int y, int z) {
        return Math.min(x, Math.min(y,z));
    }
}
```

2. 动态规划优化
对于重叠子问题呢，前文 动态规划详解 详细介绍过，优化方法无非是备忘录或者 DP table。

备忘录很好加，原来的代码稍加修改即可：

```java
// Runtime: 4 ms, faster than 91.95% of Java online submissions for Edit Distance.
// Memory Usage: 38.8 MB, less than 92.87% of Java online submissions for Edit Distance.

class Solution {
    public int minDistance(String word1, String word2) {
        int m=word1.length(), n = word2.length();
        int[][] memo = new int[m+1][n+1];
        for (int i = 1; i <= m; i++) memo[i][0] = i;
        for (int j = 1; j <= n; j++) memo[0][j] = j;
        for(int i=1;i<=m;i++){
            for(int j=1;j<=n;j++){
                if(word1.charAt(i-1)==word2.charAt(j-1)) memo[i][j]=memo[i-1][j-1];
                else memo[i][j] = 1+min(
                    memo[i][j-1],
                    memo[i-1][j-1],
                    memo[i-1][j]
                );
            }
        }
        return memo[m][n];
    }

    public int min(int x, int y, int z) {
        return Math.min(x, Math.min(y,z));
    }
}
```


3. 具体的操作


```java
// int[][] dp;
Node[][] dp;

class Node {
    int val;
    int choice;
    // 0 代表啥都不做
    // 1 代表插入
    // 2 代表删除
    // 3 代表替换
}
```


---

### 354. Russian Doll Envelopes 俄罗斯套娃信封问题（困难）

[354. Russian Doll Envelopes](https://leetcode.com/problems/russian-doll-envelopes/)

You are given a 2D array of integers envelopes where envelopes[i] = [wi, hi] represents the width and the height of an envelope.

One envelope can fit into another if and only if both the width and height of one envelope are greater than the other envelope's width and height.

Return the maximum number of envelopes you can Russian doll (i.e., put one inside the other).

Note: You cannot rotate an envelope.

Example 1:

Input: envelopes = [[5,4],[6,4],[6,7],[2,3]]
Output: 3
Explanation: The maximum number of envelopes you can Russian doll is 3 ([2,3] => [5,4] => [6,7]).


```java
// Runtime: 148 ms, faster than 53.28% of Java online submissions for Russian Doll Envelopes.
// Memory Usage: 39.9 MB, less than 82.57% of Java online submissions for Russian Doll Envelopes.

// envelopes = [[w, h], [w, h]...]
public int maxEnvelopes(int[][] envelopes) {
    int n = envelopes.length;
    // 按宽度升序排列，如果宽度一样，则按高度降序排列
    Arrays.sort(
        envelopes,
        new Comparator<int[]>() {
            public int compare(int[] a, int[] b) {
                return a[0] == b[0] ? b[1] - a[1] : a[0] - b[0];
            }
        }
    );
    // 对高度数组寻找 LIS
    int[] height = new int[n];
    for (int i = 0; i < n; i++) height[i] = envelopes[i][1];
    return lengthOfLIS(height);
}

public int lengthOfLIS(int[] nums) {
        int n = nums.length;
        int[] dp = new int[n];
        Arrays.fill(dp, 1);
        for(int i=1; i<n; i++) {
            for(int j=0; j<i; j++) {
                if(nums[i] > nums[j]) dp[i] = Math.max(dp[i], dp[j]+1);
            }
        }
        int res=0;
        for(int num:dp) res=Math.max(res, num);
        return res;
    }
}
```

---

### 53 最大子序和（简单)

1. simple
   1. 复杂度是 O(N)，
   2. 空间复杂度也是 O(N)

```java
// Runtime: 2 ms, faster than 41.34% of Java online submissions for Maximum Subarray.
// Memory Usage: 47.7 MB, less than 98.44% of Java online submissions for Maximum Subarray.

int maxSubArray(int[] nums) {
    int n = nums.length;
    if(n == 0) return 0;

    int[] dp = new int[n];
    // base case
    // 第一个元素前面没有子数组
    dp[0] = nums[0];
    int res = dp[0];

    for(int i = 1; i < n; i++) {
        // 状态转移方程
        dp[i] = Math.max(nums[i], nums[i] + dp[i - 1]);
        // 得到 nums 的最大子数组
        res = Math.max(res, dp[i]);
    }
    return res;
}
```

2. 状态压缩
   1. dp[i] 仅仅和 dp[i-1] 的状态有关

```java
// Runtime: 1 ms, faster than 100.00% of Java online submissions for Maximum Subarray.
// Memory Usage: 49.2 MB, less than 78.66% of Java online submissions for Maximum Subarray.
int maxSubArray(int[] nums) {
    int n = nums.length;
    if(n == 0) return 0;
    int res = dp[0];
    int num_pre = nums[0];
    int num_cur;

    for(int i = 1; i < n; i++) {
        // 状态转移方程
        num_cur = Math.max(nums[i], nums[i] + num_pre);
        num_pre = num_cur;
        // 得到 nums 的最大子数组
        res = Math.max(res, num_cur);
    }
    return res;
}
```


---


## 背包类型问题

1. 动规标准
   1. 第一步要明确两点，「状态」和「选择」。
   2. 状态:「背包的容量」和「可选择的物品」。
      1. 「状态」，有两个，也就是说我们需要一个二维 dp 数组。
      2. dp[i][w] 的定义如下：对于前 i 个物品，当前背包的容量为 w，这种情况下可以装的最大价值是 dp[i][w]。
   3. 选择:「装进背包」或者「不装进背包」
      1. 没有把这第 i 个物品装入背包: 最大价值 dp[i][w] 应该等于 dp[i-1][w]，继承之前的结果。
      2. 把这第 i 个物品装入了背包，那么 dp[i][w] = dp[i-1][w - wt[i-1]] + val[i-1]。

```cpp
int knapsack(int W, int N, vector<int>& wt, vector<int>& val) {
    // base case 已初始化
    vector<vector<int>> dp(N + 1, vector<int>(W + 1, 0));
    for (int i = 1; i <= N; i++) {
        for (int w = 1; w <= W; w++) {
            // 这种情况下只能选择不装入背包
            if (w - wt[i-1] < 0) dp[i][w] = dp[i - 1][w];
            // 装入或者不装入背包，择优
            else dp[i][w] = max(dp[i - 1][w - wt[i-1]] + val[i-1], dp[i - 1][w]);
        }
    }
    return dp[N][W];
}
```



---

### 子集背包问题

#### 416. Partition Equal Subset Sum 分割等和子集（中等）

[416. Partition Equal Subset Sum](https://leetcode.com/problems/partition-equal-subset-sum/)

Given a non-empty array nums containing only positive integers, find if the array can be partitioned into two subsets such that the sum of elements in both subsets is equal.


先对集合求和，得出 sum，把问题转化为背包问题：

给一个可装载重量为 sum / 2 的背包和 N 个物品，每个物品的重量为 nums[i]。
现在让你装物品，是否存在一种装法，能够恰好将背包装满？

1. 第一步要明确两点，「状态」和「选择」。
   1. 状态就是「背包的容量」和「可选择的物品」，
   2. 选择就是「装进背包」或者「不装进背包」。
2. 第二步要明确 dp 数组的定义。

- dp[i][j] = x 表示，
- 对于前 i 个物品，当前背包的容量为 j 时，
- 若 x 为 true，则说明可以恰好将背包装满，
- 若 x 为 false，则说明不能恰好将背包装满。

- 我们想求的最终答案就是 dp[N][sum/2]，
- base case 就是
  - dp[..][0] = true 因为背包没有空间的时候，就相当于装满了，
  - dp[0][..] = false，没有物品可选择的时候，肯定没办法装满背包。

```java
// Runtime: 77 ms, faster than 19.78% of Java online submissions for Partition Equal Subset Sum.
// Memory Usage: 51.4 MB, less than 20.88% of Java online submissions for Partition Equal Subset Sum.

class Solution {
    public boolean canPartition(int[] nums) {

        int sum=0;
        for(int num:nums) sum+=num;
        // 和为奇数时，不可能划分成两个和相等的集合
        if(sum%2!=0) return false;
        sum = sum/2;

        int n = nums.length;
        boolean [][] dp = new boolean[n+1][sum+1];

        // base case
        for(int i=0;i<=n;i++) dp[i][0]=true;
        for(int j=0;j<=sum;j++) dp[0][j]=false;

        for(int i=1;i<=n;i++){
            for(int j=1;j<=sum;j++){
                // 背包容量不足，不能装入第 i 个物品
                if(j < nums[i-1]) dp[i][j] = dp[i-1][j];
                // 装入或不装入背包
                else dp[i][j] = dp[i-1][j] || dp[i-1][j-nums[i-1]];
            }
        }
        return dp[n][sum];
    }
}
```

2. 状态压缩
   1. dp[i][j] 都是通过上一行 dp[i-1][..] 转移过来的

```java
boolean canPartition(int[] nums) {
    int sum = 0;
    for (int num : nums) sum += num;
    // 和为奇数时，不可能划分成两个和相等的集合
    if (sum % 2 != 0) return false;
    int n = nums.length;
    sum = sum / 2;
    boolean[] dp = new boolean[sum + 1];

    // base case
    dp[0] = true;

    for (int i = 0; i < n; i++) {
        for (int j = sum; j >= 0; j--) {
            if (j >= nums[i]) dp[j] = dp[j] || dp[j - nums[i]];
        }
    }
    return dp[sum];
}
```


---


#### 698. Partition to K Equal Sum Subsets

Given an integer array nums and an integer k, return true if it is possible to divide this array into k non-empty subsets whose sums are all equal.

Example 1:

Input: nums = [4,3,2,3,5,2,1], k = 4
Output: true
Explanation: It's possible to divide it into 4 subsets (5), (1, 4), (2,3), (2,3) with equal sums.


1. backtracking

```java
class Solution {
    public boolean canPartitionKSubsets(int[] nums, int k) {
        if(k==1) return true;
        int n = nums.length;
        //if sum is not a multiple of K we can't divide
        int sum = 0;
        for(int num:nums) sum+=num;
        if(sum%k!=0 || k>n) return false;
        sum=sum/k;
        boolean[] used = new boolean[n];
        Arrays.fill(used, false);
        return findPart(nums, used, sum, 0,0, k);
    }

    public boolean findPart(int[] nums, boolean[] used, int target, int cur_sum, int i, int k) {
        if(k<=1) return true;
        if(target==cur_sum) return findPart(nums, used, target, 0, 0, k-1);

        for(int j=i; j<nums.length; j++){
            if(visited[j] || curr_sum+arr[j]>target) continue;
            used[j] = true;
            if( findPart(nums, used, target, cur_sum+nums[j], j+1, k) ) return true;
            used[j] = false;
        }
        return false;
    }
}
```

2. remove visited[], `each time used a number, nums[i]=0, undo: nums[i]=temp`.

```java
// Runtime: 1 ms, faster than 90.92% of Java online submissions for Partition to K Equal Sum Subsets.
// Memory Usage: 36.3 MB, less than 80.51% of Java online submissions for Partition to K Equal Sum Subsets.
class Solution {
    public boolean canPartitionKSubsets(int[] nums, int k) {
        // Not possible to divide into equal subsets, if sum of all
        // nums[] is not a multiple of k.
        int sum = 0;
        for (int i = nums.length - 1; i >= 0; i--)  sum += nums[i];
        if (sum % k != 0 || k>nums.length)  return false;
        // Determine the target number that each subset must total.
        // Then start recursion to find if possible to have k equal subsets.
        sum = sum / k;
        return dfs(k, 0, 0, nums, sum);
    }

    private boolean dfs(int k, int curSum, int numsIdx, int[] nums, int target) {
        if (k <= 1) return true;                      // If no more subsets to fill, then done
        for (int i = numsIdx; i < nums.length; i++) { // Loop in nums[] values to find next unused
            if (nums[i] != 0 && curSum + nums[i] <= target) {
                int temp = nums[i];
                nums[i] = 0;
                // Mark this nums value as "used". If subset exactly filled, start new subset
                if (curSum + temp == target) {
                    if (dfs(k - 1, 0, 0, nums, target))  return true;
                }
                // Else subset not filled, find more to fill it.
                else {
                    if (dfs(k, curSum + temp, i + 1, nums, target))  return true;
                }
                nums[i] = temp;
                // This nums[i] value didn't result in a good solution
                // so "unuse" this nums[] value and loop back to try another nums[] value
            }
        }
        return false;
    }
}
```




---



















#### 215. Kth Largest Element in an Array






---

# system design

https://github.com/donnemartin/system-design-primer
















.
