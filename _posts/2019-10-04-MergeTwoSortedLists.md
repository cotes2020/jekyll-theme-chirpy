---
layout: post
title: "合并两个有序的链表"
date: 2019-10-04 14:23:00.000000000 +09:00
categories: [算法]
tags: [算法, 合并两个有序的链表]
---

> 将两个有序链表合并为一个新的有序链表并返回。新链表是通过拼接给定的两个链表的所有节点组成的。 

**示例：**

```
输入：1->2->4, 1->3->4
输出：1->1->2->3->4->4
```

**思路**

迭代方法

每次选两个链表头结点最小的，比如：我们生活中，有两个已经按照高矮排好的队伍，我们如何把变成一个队伍！当然，每次选两个队伍排头的，比较他们的高矮!组成新的的队伍。

时间复杂度：O(m+n)*O*(*m*+*n*)

空间复杂度：O(m+n)*O*(*m*+*n*)

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
func mergeTwoLists(_ l1: ListNode?, _ l2: ListNode?) -> ListNode? {
        
    let dummy = ListNode(0)
    var node = dummy // node移动节点

    var l1 = l1
    var l2 = l2

    while l1 != nil && l2 != nil {

        if l1!.val < l2!.val {

            node.next = l1
            l1 = l1!.next
        } else {

            node.next = l2
            l2 = l2!.next
        }
        node = node.next!
    }
    node.next = l1 ?? l2 // 假如一个链表遍历空了，直接拼接另一个链表
    return dummy.next
}
```

[源码地址](<https://github.com/Jovins/Algorithm>)