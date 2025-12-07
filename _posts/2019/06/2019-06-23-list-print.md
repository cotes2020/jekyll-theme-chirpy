---
title: "剑指Offer JavaScript-链表专题"
date: 2019-06-23
permalink: /2019-06-23-list-print/
categories: ["开源技术课程", "JavaScript版·剑指offer刷题笔记"]
---
## 从尾到头打印链表


### 1. 题目描述


输入一个链表，从尾到头打印链表每个节点的值。


### 2. 解题思路


可以从头到尾遍历一遍链表，将节点放入栈中，然后依次取出打印（后入先出）。


优化就是借助“递归”，先向下查找再打印输出，也可实现这种“后入先出”。可以类比二叉树的后序遍历。


### 3. 代码


用 JS 实现了简单实现了链表这种数据结构，这不是重点。


重点在`printFromTailToHead`函数。


```typescript
class Node {
    /**
     * 节点构造函数
     * @param {*} value
     * @param {Node} next
     */
    constructor(value, next) {
        this.value = value;
        this.next = next;
    }
}

class List {
    constructor() {
        this.head = new Node(null, null);
    }

    /**
     * 从0开始计算，找到包括head在内的位于index的节点
     * @param {Number} index
     */
    find(index) {
        let current = this.head;
        for (let i = 0; i < index; ++i) {
            current = current.next;
        }
        return current;
    }

    /**
     * 向index位置插入元素
     * @param {*} value
     * @param {Number} index
     */
    insert(value, index) {
        const prev = this.find(index);
        const next = new Node(value, prev.next);
        prev.next = next;
    }
}

/**
 * 逆序打印链表
 * @param {Node} node
 */
function printFromTailToHead(node) {
    if (node.next) {
        printFromTailToHead(node.next);
    }
    node.value && console.log(node.value);
}

/**
 * 以下是测试代码
 */
let list = new List();
list.insert("a", 0);
list.insert("b", 1);
list.insert("c", 2);

printFromTailToHead(list.head);

```


## 快速删除链表节点


### 1. 题目描述


给定单向链表的头指针和一个结点指针，定义一个函数在 $O(1)$ 时间删除该结点。


### 2. 思路描述


正常的做法肯定是在 $O(N)$ 时间内删除节点。而这么过分的要求，显然是通过“重新赋值”才能做到。


比如要删除节点 a，那么就将 a.next 的 value 和 next 赋值给节点 a，然后删除 a.next。


表面“看起来”像是删除了节点 a，其实是将其后节点的信息转移到了它自己身上。


除此之外，对于最后一个节点，还是要退化成 $O(N)$ 的复杂度。而整体分析一下复杂度：


$$
O(T) = (O(N) + O(1) * (n - 1)) / n = O(1)
$$


### 3. 代码实现


```typescript
class Node {
    /**
     * 节点构造函数
     * @param {*} value
     * @param {Node} next
     */
    constructor(value, next) {
        this.value = value;
        this.next = next;
    }
}

/**
 *
 * @param {Node} head
 * @param {Node} toDelete
 */
function deleteNode(head, toDelete) {
    if (head === toDelete || !toDelete || !head) {
        return;
    }

    let nextNode = toDelete.next;

    if (!nextNode) {
        // 尾节点
        let node = head;
        while (node.next !== toDelete) {
            node = node.next;
        }
        node.next = null;
        toDelete = null;
    } else {
        toDelete.value = nextNode.value;
        toDelete.next = nextNode.next;
        nextNode = null;
    }
}

/**
 * 测试代码
 */

let node3 = new Node(3, null),
    node2 = new Node(2, node3),
    node1 = new Node(1, node2),
    head = new Node(null, node1);

deleteNode(head, node2);
let node = head.next;
while (node) {
    console.log(node.value);
    node = node.next;
}
```


## 链表倒数第k节点


### 1. 题目描述


输入一个单链表，输出该链表中倒数第 k 个结点。


### 2. 思路描述


**思路一**：从头到尾遍历一遍，统计长度`length`。再从头遍历，直到`length - k`个节点停止，这就是倒数第 k 个节点。


**思路二**：只需要遍历一遍。准备 2 个指针`a`和`b`均指向第一个节点，`a`先移动`k`个位置；然后`a`和`b`一起向后移动，此时两个只指针的位置差为`k`；直到`a`移动到尾结点停止，此时`b`指向的节点就是倒数第 k 个节点。


### 3. 代码实现


下面是“思路二”的实现。


```typescript
/**
 * 节点定义
 */
class Node {
    constructor(value, next) {
        this.value = value;
        this.next = next;
    }
}

/**
 * 寻找倒数第k个节点
 * @param {Node} head 初始节点
 * @param {Number} k 顺序(倒数)
 */
function findKthFromTail(head, k) {
    if (!head || k <= 0) {
        return null;
    }

    let a = head,
        b = head;

    for (let i = 0; i < k; ++i) {
        a = a.next;
        if (!a) {
            return null;
        }
    }

    while (a) {
        b = b.next;
        a = a.next;
    }

    return b;
}

/**
 * 以下是测试代码, 分别输出倒数第2、3和5个节点
 */

let node3 = new Node(3, null),
    node2 = new Node(2, node3),
    node1 = new Node(1, node2),
    head = new Node(0, node1);

console.log(findKthFromTail(head, 2));
console.log(findKthFromTail(head, 3));
console.log(findKthFromTail(head, 5));
```


## 反转链表


### 1. 题目描述


定义一个函数，输入一个链表的头结点，反转该链表并输出反转后链表的头结点。


### 2. 思路描述


**思路一**：经典的“链表头插法”，时间复杂度是 $O(N)$，但是空间复杂度也是 $O(N)$


**思路二**：链表原地操作，时间复杂度是 $O(N)$，但是空间复杂度只有 $O(1)$。

1. 保存当前节点`node`的上一个节点`pre`
2. 节点`node`的`next`指向`pre`
3. 分别将`pre`和`node`向后移动 1 个位置
- 如果`node`为 null，链表翻转完毕，此时`pre`指向新的头节点，返回即可
- 否则，回到第 1 步继续执行

### 3. 代码实现


```typescript
/**
 * 节点定义
 */
class Node {
    constructor(value, next) {
        this.value = value;
        this.next = next;
    }
}

/**
 * 翻转链表
 * @param {Node} head 未翻转链表的头节点
 * @return {Node} 翻转链表后的头节点
 */
function reverseList(head) {
    let node = head,
        pre = null;

    while (node) {
        let next = node.next;

        node.next = pre;

        pre = node;
        node = next;
    }

    return pre;
}

/**
 * 以下是测试代码, 分别输出倒数第2、3和5个节点
 */

let node3 = new Node(3, null),
    node2 = new Node(2, node3),
    node1 = new Node(1, node2),
    head = new Node(0, node1);

let newHead = reverseList(head);
while (newHead) {
    console.log(newHead);
    newHead = newHead.next;
}
```


## 合并两个有序链表


### 1. 题目描述


输入两个递增排序的链表，合并这两个链表并使新链表中的结点仍然是按照递增排序的。


### 2. 思路分析


准备一个指针`node`，假设指向两个链表的中节点的指针分别是：`p1`和`p2`。

1. 比较`p1`和`p2`的`value`大小
- 如果，p1.value 小于 p2.value, node.next 指向 p1, p1 向后移动
- 否则，node.next 指向 p2, p2 向后移动
1. 重复第 1 步，直到其中一个链表遍历完
2. 跳出循环，将 node.next 指向未遍历完的链表的剩余部分

整个过程的时间复杂度是 O(N), 空间复杂度是 O(1)


### 3. 代码实现


```typescript
/**
 * 节点定义
 */
class Node {
    constructor(value = null, next = null) {
        this.value = value;
        this.next = next;
    }
}

/**
 * 合并2个有序单链表成为1个新的有序单链表
 * @param {Node} p1
 * @param {Node} p2
 */
function merge(p1, p2) {
    if (!p1) {
        return p2;
    } else if (!p2) {
        return p1;
    }

    let head = new Node(),
        node = head;

    while (p1 && p2) {
        if (p1.value < p2.value) {
            node.next = p1;
            p1 = p1.next;
        } else {
            node.next = p2;
            p2 = p2.next;
        }

        node = node.next;
    }

    if (!p1) {
        node.next = p2;
    }

    if (!p2) {
        node.next = p1;
    }

    return head.next;
}

/**
 * 以下是测试代码
 */

let list1 = new Node(1, new Node(3, new Node(5, new Node(7, null))));
let list2 = new Node(2, new Node(4, new Node(6, new Node(8, null))));

let head = merge(list1, list2);
while (head) {
    console.log(head.value);
    head = head.next;
}
```


## 复杂链表的复制


### 1. 题目描述


请实现函数`ComplexListNode *Clone（ComplexListNode* pHead）`，复制一个复杂链表。在复杂链表中，每个结点除了有一个 next 指针指向下一个结点外，还有一个 sibling 指向链表中的任意结点或者 NULL。


### 2. 思路分析


按照正常的思路，首先从头到尾遍历链表，拷贝每个节点的 value 和 next 指针。然后从头再次遍历，第二次遍历的目的在于拷贝每个节点的 sibling 指针。


然而即使找到原节点的 sibling 指针，还是得为了找到复制节点对应的 sibling 指针而再遍历一遍。那么对于 n 个要寻找 sibling 指针的节点，复杂度就是 O(N\*N)。


显然，为了降低复杂度，必须从第二次遍历着手。这里采用的方法是，在第一次遍历的时候，把 `(原节点, 复制节点)` 作为映射保存在表中。那么第二次遍历的时候，就能在 O(1) 的复杂度下立即找到原链上 sibling 指针在复制链上对应的映射。


### 3. 代码分析


```typescript
class Node {
    constructor(value, next = null, sibling = null) {
        this.value = value;
        this.next = next;
        this.sibling = sibling;
    }
}

/**
 * 复制复杂链表
 * @param {Node} first
 */
function cloneNodes(first) {
    if (!first) {
        return null;
    }

    const map = new Map();

    let copyFirst = new Node(first.value),
        node = first.next, // 被copy链的当前节点
        last = copyFirst; // copy链的当前节点, 此节点相对于被copy链短位移少1位

    map.set(first, copyFirst);

    while (node) {
        last.next = new Node(node.value);
        last = last.next;
        map.set(node, last);
        node = node.next;
    }

    // 第二次遍历, 迁移sibling
    node = first;
    while (node) {
        map.get(node) && (map.get(node).sibling = map.get(node.sibling));
        node = node.next;
    }

    return copyFirst;
}

/**
 * 测试代码
 */
const node1 = new Node("a"),
    node2 = new Node("b"),
    node3 = new Node("c"),
    node4 = new Node("d");

node1.next = node2;
node2.next = node3;
node3.next = node4;

node1.sibling = node3;
node4.sibling = node2;

let copyNode = cloneNodes(node1);
while (copyNode) {
    console.log(copyNode);
    copyNode = copyNode.next;
}
```


## 两个链表中的第一个公共节点


### 1. 题目描述


输入两个链表，找出它们的第一个公共结点。


### 2.1 思路一：栈实现


在第一个公共节点前的节点都是不相同的，因此只要倒序遍历两个链表，找出最后一个出现的相同节点即可。


因为链表不能倒序遍历，所以借助栈实现。


### 2.2 思路二：快慢指针


假设链表 A 长度大于链表 B 长度，它们的长度差为 diff。


让 A 的指针先移动 diff 的位移，然后 A 和 B 的指针再同时向后移动，每次比较节点，选出第一个出现的相同节点。


### 3. 代码实现


为了方便，先简单实现节点数据结构：


```typescript
class Node {
    constructor(value, next) {
        this.value = value;
        this.next = next;
    }
}
```


### 3.1 思路一：栈实现


```typescript
/**
 * 思路一：利用栈实现
 *
 * @param {Node} list1
 * @param {Node} list2
 */
function method1(list1, list2) {
    const stack1 = [],
        stack2 = [];

    let node = list1;
    while (node) {
        stack1.push(node);
        node = node.next;
    }

    node = list2;
    while (node) {
        stack2.push(node);
        node = node.next;
    }

    node = null;
    while (stack1.length && stack2.length) {
        let top1 = stack1.pop(),
            top2 = stack2.pop();
        if (top1 === top2) {
            node = top1;
        } else {
            break;
        }
    }

    return node;
}
```


### 3.2 思路二：快慢指针


```typescript
/**
 * 思路二：快慢指针
 *
 * @param {Node} list1
 * @param {Node} list2
 */
function method2(list1, list2) {
    let length1 = 0,
        length2 = 0;

    let node = list1;
    while (node) {
        ++length1;
        node = node.next;
    }

    node = list2;
    while (node) {
        ++length2;
        node = node.next;
    }

    let diff = Math.abs(length1 - length2),
        longList = null,
        shortList = null;
    if (length1 > length2) {
        longList = list1;
        shortList = list2;
    } else {
        longList = list2;
        shortList = list1;
    }

    while (diff > 0) {
        longList = longList.next;
        --diff;
    }

    while (longList && shortList) {
        if (longList === shortList) {
            return longList;
        }
        longList = longList.next;
        shortList = shortList.next;
    }

    return null;
}
```


### 3.3 测试代码


```typescript
const node4th = new Node(4);
const node3th = new Node(3, node4th);
const list1 = new Node(1, new Node(2, new Node(3, node3th)));
const list2 = new Node(5, new Node(6, node3th));

console.log(method2(list1, list2));
```


