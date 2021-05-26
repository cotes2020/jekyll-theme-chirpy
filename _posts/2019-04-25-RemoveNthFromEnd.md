---
layout: post
title: "删除链表的倒数第N个节点"
date: 2019-04-25 22:08:00.000000000 +09:00
categories: [算法]
tags: [算法, 删除链表]
---

> 给定一个链表，删除链表的倒数第 *n* 个节点，并且返回链表的头结点。

**示例：**

```
给定一个链表: 1->2->3->4->5, 和 n = 2.

当删除了倒数第二个节点后，链表变为 1->2->3->5.
```

**说明：**

给定的 *n* 保证是有效的。

**进阶：**

你能尝试使用一趟扫描实现吗？

## 思路

我们可以使用两个指针而不是一个指针。第一个指针从列表的开头向前移动 n+1*n*+1 步，而第二个指针将从列表的开头出发。现在，这两个指针被 n*n* 个结点分开。我们通过同时移动两个指针向前来保持这个恒定的间隔，直到第一个指针到达最后一个结点。此时第二个指针将指向从最后一个结点数起的第 n个结点。我们重新链接第二个指针所引用的结点的 `next` 指针指向该结点的下下个结点。

![](/assets/images/al-RemoveNthFromEnd-01.png)

**解题步骤**

设想假设设定了双指针pos和pre的话，当pos指向末尾的NULL，pos与pre之间相隔的元素个数为n时，那么删除掉pre的下一个指针就完成了要求。

- 设置虚拟节点dummy指向head
- 设定双指针pos和pre，初始都指向虚拟节点dummy
- 移动pos，直到pos与pre之间相隔的元素个数为n
- 同时移动pos与pre，直到pos指向的为NULL
- 将pre的下一个节点指向下下个节点

**动画描述**

![](/assets/images/al-RemoveNthFromEnd-02.gif)

**复杂度分析**

- 时间复杂度：O(L)，该算法对含有 L个结点的列表进行了一次遍历。因此时间复杂度为 O(L)。
- 空间复杂度：O(1) ，我们只用了常量级的额外空间。

**代码实现**

```swift
public class ListNode {
    
    public var val: Int
    public var next: ListNode?
    public init(_ val: Int) {
        
        self.val = val
        self.next = nil
    }
}
```

```swift
func removeNthFromEnd(_ head: ListNode?, _ n: Int) -> ListNode? {
        
    let dummy = ListNode(0)
    dummy.next = head
    var pre: ListNode? = dummy
    var pos: ListNode? = dummy
    // 移动pos, 直到pos与pre之间相隔的元素个数为n
    for _ in 0..<n {
        pos = pos!.next
    }
    // 同时移动pos和pre
    while pos!.next != nil {

        pre = pre!.next
        pos = pos!.next
    }
    pre!.next = pre!.next!.next
    return dummy.next
}
```

[源码地址](https://github.com/Jovins/Algorithm)