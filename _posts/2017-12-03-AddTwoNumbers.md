---
layout: post
title: "两数相加"
date: 2017-12-03 22:05:00.000000000 +09:00
categories: [算法]
tags: [算法, 两数相加]
---

给出两个 **非空** 的链表用来表示两个非负的整数。其中，它们各自的位数是按照 **逆序** 的方式存储的，并且它们的每个节点只能存储 **一位** 数字。如果，我们将这两个数相加起来，则会返回一个新的链表来表示它们的和。您可以假设除了数字 0 之外，这两个数都不会以 0 开头。

**示例：**

```
输入：(2 -> 4 -> 3) + (5 -> 6 -> 4)
输出：7 -> 0 -> 8
原因：342 + 465 = 807
```

**思路**

我们使用变量来跟踪进位，并从包含最低有效位的表头开始模拟逐位相加的过程。

![](/assets/images/al-AddTwoNumbers-01.png)

**复杂度分析**

- 时间复杂度：O(\max(m, n))*O*(max(*m*,*n*))，假设 m*m* 和 n*n* 分别表示 l1*l*1 和 l2*l*2 的长度，上面的算法最多重复 \max(m, n)max(*m*,*n*)次。
- 空间复杂度：O(\max(m, n))*O*(max(*m*,*n*))， 新列表的长度最多为 \max(m,n) + 1max(*m*,*n*)+1。

**算法**

就像你在纸上计算两个数字的和那样，我们首先从最低有效位也就是列表L1和L2 的表头开始相加。由于每位数字都应当处于 0…..9 的范围内，我们计算两个数字的和时可能会出现 “溢出”。例如，5 + 7 = 12。在这种情况下，我们会将当前位的数值设置为 22，并将进位 carry = 1带入下一次迭代。进位 carry 必定是 0 或 1，这是因为两个数字相加（考虑到进位）可能出现的最大和为 9 + 9 + 1 = 19。

分析如下：

- 将当前结点初始化为返回列表的哑结点。
- 将进位 carry初始化为 0。
- 将 p和 q 分别初始化为列表L1和L2 的头部。
- 遍历列表L1和L2 直至到达它们的尾端。
  - 将  x 设为结点 p  的值。如果 p 已经到达 L1 的末尾，则将其值设置为 0。
  - 将 y 设为结点 q  的值。如果 q 已经到达 L2 的末尾，则将其值设置为 0。
  - 设定 sum = x + y + carry。
  - 更新进位的值，carry = sum / 10。
  - 创建一个数值为 (sum mod 10)的新结点，并将其设置为当前结点的下一个结点，然后将当前结点前进到下一个结点。
  - 同时，将 P 和 q 前进到下一个结点。
- 检查 carry = 1 是否成立，如果成立，则向返回列表追加一个含有数字 1 的新结点。
- 返回哑结点的下一个结点。

请注意，我们使用哑结点来简化代码。如果没有哑结点，则必须编写额外的条件语句来初始化表头的值。

请特别注意以下情况：

| 测试用例                    | 说明                                               |
| --------------------------- | -------------------------------------------------- |
| L1 = [0, 1], L2 = [0, 1, 2] | 当一个列表比另一个列表长时                         |
| L1 = [], L2 = [0, 1]        | 当一个列表为空时，即出现空列表                     |
| L1 = [9, 9], L2 = [1]       | 求和运算最后可能出现额外的进位，这一点很容易被遗忘 |

**Swift代码实现**

```swift
public class ListNode {
    
    public var val: Int
    public var next: ListNode?
    public init(_ val: Int) {
        
        self.val = val
        self.next = nil
    }
}

class AddTwoNumbers {
    
    func addTwoNumbers(_ l1: ListNode?, _ l2: ListNode?) -> ListNode? {
        
        guard let l1 = l1 else {
            return l2
        }
        guard let l2 = l2  else {
            return l1
        }
        let outputNode = ListNode((l1.val + l2.val) % 10)
        // 加入l1.val + l2.val > 10, 把溢出的1带到下一个节点
        if l1.val + l2.val > 9 {
            
            outputNode.next = addTwoNumbers(addTwoNumbers(l1.next, l2.next), ListNode(1))
        } else {
            outputNode.next = addTwoNumbers(l1.next, l2.next)
        }
        return outputNode
    }
}
```

**测试**

```swift
func listNodeTest() {
        
    var head1: ListNode?
    var tail1: ListNode?
    var head2: ListNode?
    var tail2: ListNode?

    let values1 = [2, 4, 3]
    let values2 = [5, 6, 4]

    for i in values1 {
        appendToTail(&head1, &tail1, i)
    }
    for i in values2 {
        appendToTail(&head2, &tail2, i)
    }

    let sum = AddTwoNumbers()
    let outNode = sum.addTwoNumbers(head1, head2)
    traverse(outNode)
}

// 尾插法
func appendToTail(_ head: inout ListNode?, _ tail: inout ListNode?, _ val: Int) {

    if tail == nil {
        tail = ListNode(val)
        head = tail
    } else {
        tail!.next = ListNode(val)
        tail = tail!.next
    }
}

// 头插法
func appendToHead(_ head: inout ListNode?, _ val: Int) {
    if head == nil {
        head = ListNode(val)
    } else {
        let temp = ListNode(val)
        temp.next = head
        head = temp
    }
}

// 遍历链表
func traverse(_ head: ListNode?) {
    if head == nil {
        fatalError("没有创建过链表")
    }
    var node = head
    while node != nil {
        print(node!.val)
        node = node!.next
    }
}
```

**输出**

```
输入：(2 -> 4 -> 3) + (5 -> 6 -> 4)
输出：7 -> 0 -> 8
```

[源码地址](<https://github.com/Jovins/Algorithm>)