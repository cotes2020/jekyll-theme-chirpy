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
- [https://labuladong.gitbook.io](#httpslabuladonggitbookio)
  - [day1. 原地修改数组](#day1-原地修改数组)
    - [有序数组去重](#有序数组去重)
    - [有序链表去重](#有序链表去重)
    - [移除元素](#移除元素)
    - [移除0](#移除0)
  - [day2. 前缀和技巧](#day2-前缀和技巧)
    - [计算list中间指定位置的和](#计算list中间指定位置的和)
    - [和为k的子数组](#和为k的子数组)
    - [304. 二维区域和检索 - 矩阵不可变](#304-二维区域和检索---矩阵不可变)
- [链表 解题](#链表-解题)
  - [单链表的六大解题套路](#单链表的六大解题套路)
    - [合并两个有序链表 Merge 2 Sorted Lists](#合并两个有序链表-merge-2-sorted-lists)
      - [java](#java)
      - [python](#python)
    - [合并 k 个有序链表 Merge k Sorted Lists](#合并-k-个有序链表-merge-k-sorted-lists)
    - [寻找单链表的倒数第 k 个节点](#寻找单链表的倒数第-k-个节点)
    - [寻找单链表的中点](#寻找单链表的中点)
    - [判断单链表是否包含环并找出环起点](#判断单链表是否包含环并找出环起点)
    - [链表中含有环，计算这个环的起点](#链表中含有环计算这个环的起点)
    - [判断两个单链表是否相交并找出交点](#判断两个单链表是否相交并找出交点)
  - [递归反转链表](#递归反转链表)
    - [递归反转整个链表](#递归反转整个链表)
    - [反转链表前 N 个节点](#反转链表前-n-个节点)
    - [反转链表的一部分](#反转链表的一部分)
    - [K个一组反转链表](#k个一组反转链表)
  - [回文链表](#回文链表)
    - [寻找回文](#寻找回文)
    - [判断回文链表 - 双指针技巧](#判断回文链表---双指针技巧)
      - [判断回文链表number](#判断回文链表number)
      - [判断回文链表String](#判断回文链表string)
      - [判断回文单链表 - 把原始链表反转存入一条新的链表，然后比较](#判断回文单链表---把原始链表反转存入一条新的链表然后比较)
      - [判断回文单链表 - 二叉树后序遍历](#判断回文单链表---二叉树后序遍历)
      - [判断回文单链表 - 用栈结构倒序处理单链表](#判断回文单链表---用栈结构倒序处理单链表)
      - [判断回文单链表 - 不完全反转链表，仅仅反转部分链表，空间复杂度O(1)。](#判断回文单链表---不完全反转链表仅仅反转部分链表空间复杂度o1)
  - [排序](#排序)
    - [快速排序](#快速排序)
    - [归并排序](#归并排序)
- [Tree](#tree)
  - [二叉树](#二叉树)
    - [计算一棵二叉树共有几个节点](#计算一棵二叉树共有几个节点)
    - [翻转二叉树](#翻转二叉树)
    - [填充二叉树节点的右侧指针](#填充二叉树节点的右侧指针)
    - [将二叉树展开为链表](#将二叉树展开为链表)
    - [构造最大二叉树](#构造最大二叉树)
    - [通过前序和中序/后序和中序遍历结果构造二叉树(kong)](#通过前序和中序后序和中序遍历结果构造二叉树kong)
    - [寻找重复子树(kong)](#寻找重复子树kong)
  - [二叉搜索树](#二叉搜索树)
    - [判断 BST 的合法性](#判断-bst-的合法性)
    - [在 BST 中搜索元素](#在-bst-中搜索元素)
    - [在 BST 中插入一个数](#在-bst-中插入一个数)
    - [在 BST 中删除一个数](#在-bst-中删除一个数)
    - [不同的二叉搜索树 - 穷举问题](#不同的二叉搜索树---穷举问题)

---

# Labuladong

- https://github.com/labuladong/fucking-algorithm
- https://labuladong.github.io

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


# https://labuladong.gitbook.io


## day1. 原地修改数组

数组
- 在尾部插入、删除元素是比较高效的，时间复杂度是`1`，
- 在中间或者开头插入、删除元素，就会涉及数据的搬移，时间复杂度为 `O(N)`，效率较低。

Do not allocate extra space for another array. You must do this by modifying the input array in-place with O(1) extra memory.


如何在原地修改数组，避免数据的搬移。
- 如果不是原地修改的话，直接 new 一个 int[] 数组，把去重之后的元素放进这个新数组中，然后返回这个新数组即可。
- 原地删除不允许 new 新数组，只能在原数组上操作，然后返回一个长度，这样就可以通过返回的长度和原始数组得到我们去重后的元素有哪些了。

### 有序数组去重

![Screen Shot 2021-10-10 at 10.21.49 PM](https://i.imgur.com/71PNcPT.png)

在数组相关的算法题中时非常常见的，通用解法就是使用快慢指针技巧。
- 让慢指针 slow 走在后面，快指针 fast 走在前面探路
- 找到一个不重复的元素就告诉 slow 并让 slow 前进一步。
- 这样当 fast 指针遍历完整个数组 nums 后，`nums[0..slow]` 就是不重复元素。

```java
int removeDuplicates(int[] nums) {
    if (nums.length == 0) { return 0; }
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

### 有序链表去重

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

### 移除元素

把 nums 中所有值为 val 的元素原地删除，依然需要使用 `双指针技巧` 中的 `快慢指针`：
- 如果 fast 遇到需要去除的元素，则直接跳过，
- 否则就告诉 slow 指针，并让 slow 前进一步。

```java
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
# Runtime: 32 ms, faster than 81.50% of Python3 online submissions for Remove Element.
# Memory Usage: 14.2 MB, less than 47.25% of Python3 online submissions for Remove Element.
def removeElement(nums: List[int], val: int) -> int:
    slow, fast = 0,0
    while fast < len(nums):
        if nums[fast] != val:
            nums[slow] = nums[fast]
            slow += 1
        fast += 1
    print(nums)
    print(nums[0:slow])

# removeElement([0,0,1,2,2,3,3], 2)
```

---

### 移除0

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


## day2. 前缀和技巧

快速计算一个索引区间内的元素之和。



### 计算list中间指定位置的和

```
303. Range Sum Query - Immutable
Given an integer array `nums`, handle multiple queries of the following type:

Calculate the sum of the elements of nums between `indices left and right` inclusive where `left <= right`.

Implement the NumArray class:
- `NumArray(int[] nums)` Initializes the object with the integer array nums.
- `int sumRange(int left, int right)` Returns the sum of the elements of nums between indices left and right inclusive (i.e. `nums[left] + nums[left + 1] + ... + nums[right]`).

Example 1:

Input
["NumArray", "sumRange", "sumRange", "sumRange"]
[[[-2, 0, 3, -5, 2, -1]], [0, 2], [2, 5], [0, 5]]
Output:
[null, 1, -1, -3]

Explanation
NumArray numArray = new NumArray([-2, 0, 3, -5, 2, -1]);
numArray.sumRange(0, 2); // return (-2) + 0 + 3 = 1
numArray.sumRange(2, 5); // return 3 + (-5) + 2 + (-1) = -1
numArray.sumRange(0, 5); // return (-2) + 0 + 3 + (-5) + 2 + (-1) = -3
```


```java
// solution 1
class NumArray {
    private int[] nums;
    public NumArray(int[] nums) {
        this.nums = nums;
    }
    public int sumRange(int left, int right) {
        int res = 0;
        for (int i = left; i <= right; i++) {
            res += nums[i];
        }
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
class NumArray {

    // 前缀和数组
    private int[] preSum;

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

### 和为k的子数组

---

### 304. 二维区域和检索 - 矩阵不可变

[youtube](https://www.youtube.com/watch?v=PwDqpOMwg6U)

图像块之间相互减

![Screen Shot 2021-10-13 at 11.35.52 PM](https://i.imgur.com/f55K6B4.png)



---




# 链表 解题

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



---

#### java

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

---

#### python

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

### 合并 k 个有序链表 Merge k Sorted Lists

> 23. Merge k Sorted Lists

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

### 寻找单链表的倒数第 k 个节点


point: 算法题一般只给你一个 ListNode 头结点代表一条单链表，
- 不能直接得出这条链表的长度 n，
- 而需要先遍历一遍链表算出 n 的值，
- 然后再遍历链表计算第 n - k 个节点。

**只遍历一次链表**

```java
// 返回链表的倒数第 k 个节点
ListNode findFromEnd(ListNode head, int k) {
    ListNode p1 = head;
    ListNode p2 = head;
    // p1 先走 k 步
    for (int i = 0; i < k; i++) {
        p1 = p1.next;
    }
    // p1 和 p2 同时走 n - k 步
    while (p1 != null) {
        p2 = p2.next;
        p1 = p1.next;
    }
    // p2 现在指向第 n - k 个节点
    return p2;
}
```

时间复杂度
- 无论遍历一次链表和遍历两次链表的时间复杂度都是 O(N)，但上述这个算法更有技巧性。

> 第 19 题「删除链表的倒数第 N 个结点」：

```java
// Runtime: 0 ms, faster than 100.00% of Java online submissions for Remove Nth Node From End of List.
// Memory Usage: 37 MB, less than 75.59% of Java online submissions for Remove Nth Node From End of List.


// 主函数
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

private ListNode findFromEnd(ListNode head, int k){
    ListNode p1 = head, p2 = head;
    // p1 先走 k 步
    for (int i = 0; i < k; i++) {
        p1 = p1.next;
    }
    // p1 和 p2 同时走 n - k 步
    while (p1 != null) {
        p2 = p2.next;
        p1 = p1.next;
    }
    // p2 现在指向第 n - k 个节点
    return p2;
}
```

---

### 寻找单链表的中点

point: 无法直接得到单链表的长度 n，
- 常规方法也是先遍历链表计算 n，再遍历一次得到第 n / 2 个节点，也就是中间节点。

solution:
- 两个指针 slow 和 fast 分别指向链表头结点 head。
- 每当慢指针 slow 前进一步，快指针 fast 就前进两步，
- 这样当 fast 走到链表末尾时，slow 就指向了链表中点。

> 如果链表长度为偶数，中点有两个的时候，返回的节点是靠后的那个节点。
> 这段代码稍加修改就可以直接用到判断链表成环的算法题上。

```java
// Runtime: 0 ms, faster than 100.00% of Java online submissions for Middle of the Linked List.
// Memory Usage: 36.8 MB, less than 45.65% of Java online submissions for Middle of the Linked List.

ListNode middleNode(ListNode head) {
    // 快慢指针初始化指向 head
    ListNode slow = head, fast = head;
    // 快指针走到末尾时停止
    while (fast != null && fast.next != null) {
        // 慢指针走一步，快指针走两步
        slow = slow.next;
        fast = fast.next.next;
    }
    // 慢指针指向中点
    return slow;
}
```

---

### 判断单链表是否包含环并找出环起点

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
        if (slow == fast) {
            return true;
        }
    }
    // 不包含环
    return false;
}
```

---

### 链表中含有环，计算这个环的起点

快慢指针相遇时，慢指针 slow 走了 k 步，那么快指针 fast 一定走了 2k 步：
- fast 一定比 slow 多走了 k 步，这多走的 k 步其实就是 fast 指针在环里转圈圈，所以 k 的值就是环长度的「整数倍」。
- 假设相遇点距环的起点的距离为 m，那么环的起点距头结点 head 的距离为 k - m，也就是说如果从 head 前进 k - m 步就能到达环起点。
- 如果从相遇点继续前进 k - m 步，也恰好到达环起点。因为结合上图的 fast 指针，从相遇点开始走k步可以转回到相遇点，那走 k - m 步肯定就走到环起点了
- 所以，只要我们把快慢指针中的任一个重新指向 head，然后两个指针同速前进，k - m 步后一定会相遇，相遇之处就是环的起点了。


```java
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

### 判断两个单链表是否相交并找出交点

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

## 递归反转链表

---

### 递归反转整个链表

206 Reverse Linked List
- Given the head of a singly linked list, reverse the list, and return the reversed list.
- Input: head = [1,2,3,4,5]
- Output: [5,4,3,2,1]


```java
// recursion
// Runtime: 0 ms, faster than 100.00% of Java online submissions for Reverse Linked List.
// Memory Usage: 39.3 MB, less than 38.00% of Java online submissions for Reverse Linked List.
ListNode reverseList(ListNode head) {
    if (head==null || head.next == null) return head;
    ListNode last = reverse(head.next);
    head.next.next = head;
    head.next = null;
    return last;
}


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


### 反转链表的一部分

92. Reverse Linked List II
- Given the head of a singly linked list and two integers left and right where left <= right, reverse the nodes of the list from position left to position right, and return the reversed list.

- Input: head = [1,2,3,4,5], left = 2, right = 4
- Output: [1,4,3,2,5]


```java
// Runtime: 0 ms, faster than 100.00% of Java online submissions for Reverse Linked List II.
// Memory Usage: 36.6 MB, less than 75.28% of Java online submissions for Reverse Linked List II.
ListNode reverseBetween(ListNode head, int m, int n) {
    // base case
    if (m == 1) {
        return reverseN(head, n);
    }
    // 前进到反转的起点触发 base case
    head.next = reverseBetween(head.next, m - 1, n - 1);
    return head;
}

// 反转以 head 为起点的 n 个节点，返回新的头结点
ListNode reverseN(ListNode head, int n) {
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


### K个一组反转链表

25. Reverse Nodes in k-Group
- Given a linked list, reverse the nodes of a linked list k at a time and return its modified list.
- k is a positive integer and is less than or equal to the length of the linked list. If the number of nodes is not a multiple of k then left-out nodes, in the end, should remain as it is.
- You may not alter the values in the list's nodes, only nodes themselves may be changed.
- Input: head = [1,2,3,4,5], k = 2
- Output: [2,1,4,3,5]


```java
// Runtime: 0 ms, faster than 100.00% of Java online submissions for Reverse Nodes in k-Group.
// Memory Usage: 39.4 MB, less than 60.83% of Java online submissions for Reverse Nodes in k-Group.

ListNode reverseKGroup(ListNode head, int k) {
    if (head == null) return null;
    // 区间 [a, b) 包含 k 个待反转元素
    ListNode a, b;
    a = b = head;
    for (int i = 0; i < k; i++) {
        // 不足 k 个，不需要反转，base case
        if (b == null) return head;
        b = b.next;
    }
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


---

## 回文链表

- 寻找回文串是从中间向两端扩展，
- 判断回文串是从两端向中间收缩。

对于单链表
- 无法直接倒序遍历，可以造一条新的反转链表，
- 可以利用链表的后序遍历，也可以用栈结构倒序处理单链表。

---

### 寻找回文

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


### 判断回文链表 - 双指针技巧

寻找回文串的核心思想是从中心向两端扩展：
- 回文串是对称的，所以正着读和倒着读应该是一样的，这一特点是解决回文串问题的关键。
- 因为回文串长度可能为奇数也可能是偶数，长度为奇数时只存在一个中心点，而长度为偶数时存在两个中心点，所以上面这个函数需要传入l和r。
- 「双指针技巧」，从两端向中间逼近即可：

---


#### 判断回文链表number

9. Palindrome Number
- Given an integer x, return true if x is palindrome integer.
- An integer is a palindrome when it reads the same backward as forward.
- For example, 121 is palindrome while 123 is not.


---

#### 判断回文链表String


125. Valid Palindrome
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
int count(TreeNode root) {
    // base case
    if (root == null) return 0;
    // 自己加上子树的节点数就是整棵树的节点数
    return 1 + count(root.left) + count(root.right);
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


































.
