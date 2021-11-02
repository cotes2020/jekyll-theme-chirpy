


- [Intro 0](#intro-0)
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

---

# Intro 0

## 学习算法和刷题的框架思维

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
- 在尾部插入、删除元素是比较高效的，时间复杂度是 ·，
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


303. Range Sum Query - Immutable
Given an integer array `nums`, handle multiple queries of the following type:

Calculate the sum of the elements of nums between `indices left and right` inclusive where `left <= right`.

Implement the NumArray class:
- `NumArray(int[] nums)` Initializes the object with the integer array nums.
- `int sumRange(int left, int right)` Returns the sum of the elements of nums between indices left and right inclusive (i.e. `nums[left] + nums[left + 1] + ... + nums[right]`).

```
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



### 和为k的子数组




### 304. 二维区域和检索 - 矩阵不可变

[youtube](https://www.youtube.com/watch?v=PwDqpOMwg6U)

图像块之间相互减

![Screen Shot 2021-10-13 at 11.35.52 PM](https://i.imgur.com/f55K6B4.png)
























.
