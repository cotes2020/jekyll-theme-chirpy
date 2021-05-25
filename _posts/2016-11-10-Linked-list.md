---
layout: post
title: "链表"
date: 2016-11-10 22:10:00.000000000 +09:00
categories: [算法]
tags: [算法, 链表]
---

## 链表的基本概念

链表是一组节点组成的集合，每个节点都使用一个对象的引用来指向它的后一个节点。指向另一节点的引用讲做链.

![](/assets/images/struct-linklist-01.png)

链表实现了，内存零碎数据的有效组织。比如，当我们用 malloc 来进行内存申请的时候，当内存足够，但是由于碎片太多，没有连续内存时，只能以申请失败而告终，而用链表这种数据结构来组织数据，就可以解决上类问题。

![](/assets/images/struct-linklist-02.png)

## 静态链表

![](/assets/images/struct-linklist-03.png)

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// 1.定义链表节点
typedef struct node{
    int data;
    struct node *next;
}Node;
int main()
{

    // 2.创建链表节点
    Node a;
    Node b;
    Node c;

    // 3.初始化节点数据
    a.data = 1;
    b.data = 3;
    c.data = 5;

    // 4.链接节点
    a.next = &b;
    b.next = &c;
    c.next = NULL;

    // 5.创建链表头
    Node *head = &a;

    // 6.使用链表
    while(head != NULL){
        int currentData = head->data;
        printf("currentData = %i\n", currentData);
        head = head->next;
    }
    return 0;
}
```

## 动态链表

- 静态链表的意义不是很大，主要原因，数据存储在栈上，栈的存储空间有限，不能动态分配。所以链表要实现存储的自由，要动态的申请堆里的空间。
- 有一个点要说清楚，我们的实现的链表是带头节点。至于，为什么带头节点，需等大家对链表有个整体的的认知以后，再来体会，会更有意义。
- 空链表
  - 头指针带了一个空链表节点, 空链表节点中的next指向NULL

![](/assets/images/struct-linklist-04.png)

```c
#include <stdio.h>
#include <stdlib.h>

// 1.定义链表节点
typedef struct node{
    int data;
    struct node *next;
}Node;
int main()
{
    Node *head = createList();
    return 0;
}
// 创建空链表
Node *createList(){
    // 1.创建一个节点
    Node *node = (Node *)malloc(sizeof(Node));
    if(node == NULL){
        exet(-1);
    }
    // 2.设置下一个节点为NULL
    node->next = NULL;
    // 3.返回创建好的节点
    return node;
}
```

- 非空链表
  - 头指针带了一个非空节点, 最后一个节点中的next指向NULL

![](/assets/images/struct-linklist-05.png)

## 动态链表尾插法

- 1.让新节点的下一个节点等于头结点的下一个节点
- 2.让头节点的下一个节点等于新节点

```c
#include <stdio.h>
#include <stdlib.h>

// 1.定义链表节点
typedef struct node{
    int data;
    struct node *next;
}Node;
Node *createList();
void printNodeList(Node *node);
int main()
{
    Node *head = createList();
    printNodeList(head);
    return 0;
}
/**
 * @brief createList 创建链表
 * @return  创建好的链表
 */
Node *createList(){
    // 1.创建头节点
    Node *head = (Node *)malloc(sizeof(Node));
    if(head == NULL){
        return NULL;
    }
    head->next = NULL;

    // 2.接收用户输入数据
    int num = -1;
    printf("请输入节点数据\n");
    scanf("%i", &num);

    // 3.通过循环创建其它节点
    while(num != -1){
        // 3.1创建一个新的节点
        Node *cur = (Node *)malloc(sizeof(Node));
        cur->data = num;

        // 3.2让新节点的下一个节点指向头节点的下一个节点
        cur->next = head->next;
        // 3.3让头节点的下一个节点指向新节点
        head->next = cur;

        // 3.4再次接收用户输入数据
        scanf("%i", &num);
    }

    // 3.返回创建好的节点
    return head;
}
/**
 * @brief printNodeList 遍历链表
 * @param node 链表指针头
 */
void printNodeList(Node *node){
    Node *head = node->next;
    while(head != NULL){
        int currentData = head->data;
        printf("currentData = %i\n", currentData);
        head = head->next;
    }
}

```

## 动态链表头插法

- 1.定义变量记录新节点的上一个节点
- 2.将新节点添加到上一个节点后面
- 3.让新节点成为下一个节点的上一个节点

```c
#include <stdio.h>
#include <stdlib.h>

// 1.定义链表节点
typedef struct node{
    int data;
    struct node *next;
}Node;
Node *createList();
void printNodeList(Node *node);
int main()
{
    Node *head = createList();
    printNodeList(head);
    return 0;
}
/**
 * @brief createList 创建链表
 * @return  创建好的链表
 */
Node *createList(){
    // 1.创建头节点
    Node *head = (Node *)malloc(sizeof(Node));
    if(head == NULL){
        return NULL;
    }
    head->next = NULL;

    // 2.接收用户输入数据
    int num = -1;
    printf("请输入节点数据\n");
    scanf("%i", &num);

    // 3.通过循环创建其它节点
    // 定义变量记录上一个节点
    Node *pre = head;
    while(num != -1){
        // 3.1创建一个新的节点
        Node *cur = (Node *)malloc(sizeof(Node));
        cur->data = num;

        // 3.2让新节点链接到上一个节点后面
        pre->next = cur;
        // 3.3当前节点下一个节点等于NULL
        cur->next = NULL;
        // 3.4让当前节点编程下一个节点的上一个节点
        pre = cur;

        // 3.5再次接收用户输入数据
        scanf("%i", &num);
    }

    // 3.返回创建好的节点
    return head;
}
/**
 * @brief printNodeList 遍历链表
 * @param node 链表指针头
 */
void printNodeList(Node *node){
    Node *head = node->next;
    while(head != NULL){
        int currentData = head->data;
        printf("currentData = %i\n", currentData);
        head = head->next;
    }
}

```

## 动态链优化

```c
#include <stdio.h>
#include <stdlib.h>

// 1.定义链表节点
typedef struct node{
    int data;
    struct node *next;
}Node;
Node *createList();
void printNodeList(Node *node);
void insertNode1(Node *head, int data);
void insertNode2(Node *head, int data);
int main()
{
    // 1.创建一个空链表
    Node *head = createList();
    // 2.往空链表中插入数据
    insertNode1(head, 1);
    insertNode1(head, 3);
    insertNode1(head, 5);
    printNodeList(head);
    return 0;
}
/**
 * @brief createList 创建空链表
 * @return  创建好的空链表
 */
Node *createList(){
    // 1.创建头节点
    Node *head = (Node *)malloc(sizeof(Node));
    if(head == NULL){
        return NULL;
    }
    head->next = NULL;
    // 3.返回创建好的节点
    return head;
}
/**
 * @brief insertNode1 尾插法插入节点
 * @param head 需要插入的头指针
 * @param data 需要插入的数据
 * @return  插入之后的链表
 */
void insertNode1(Node *head, int data){
    // 1.定义变量记录最后一个节点
    Node *pre = head;
    while(pre != NULL && pre->next != NULL){
        pre = pre->next;
    }
    // 2.创建一个新的节点
    Node *cur = (Node *)malloc(sizeof(Node));
    cur->data = data;

    // 3.让新节点链接到上一个节点后面
    pre->next = cur;
    // 4.当前节点下一个节点等于NULL
    cur->next = NULL;
    // 5.让当前节点编程下一个节点的上一个节点
    pre = cur;
}
/**
 * @brief insertNode1 头插法插入节点
 * @param head 需要插入的头指针
 * @param data 需要插入的数据
 * @return  插入之后的链表
 */
void insertNode2(Node *head, int data){
    // 1.创建一个新的节点
    Node *cur = (Node *)malloc(sizeof(Node));
    cur->data = data;

    // 2.让新节点的下一个节点指向头节点的下一个节点
    cur->next = head->next;
    // 3.让头节点的下一个节点指向新节点
    head->next = cur;
}
/**
 * @brief printNodeList 遍历链表
 * @param node 链表指针头
 */
void printNodeList(Node *node){
    Node *head = node->next;
    while(head != NULL){
        int currentData = head->data;
        printf("currentData = %i\n", currentData);
        head = head->next;
    }
}

```

## 链表销毁

```c
/**
 * @brief destroyList 销毁链表
 * @param head 链表头指针
 */
void destroyList(Node *head){
    Node *cur = NULL;
    while(head != NULL){
        cur = head->next;
        free(head);
        head = cur;
    }
}
```

## 链表长度计算

```c
/**
 * @brief listLength 计算链表长度
 * @param head 链表头指针
 * @return 链表长度
 */
int listLength(Node *head){
    int count = 0;
    head = head->next;
    while(head){
       count++;
       head = head->next;
    }
    return count;
}

```

## 链表查找

```c
/**
 * @brief searchList 查找指定节点
 * @param head 链表头指针
 * @param key 需要查找的值
 * @return
 */
Node *searchList(Node *head, int key){
    head = head->next;
    while(head){
        if(head->data == key){
            break;
        }else{
            head = head->next;
        }
    }
    return head;
}
```

## 链表删除

```c
void deleteNodeList(Node *head, Node *find){
    while(head->next != find){
        head = head->next;
    }
    head->next = find->next;
    free(find);
}
```

## 作业

- 给链表排序

```c
/**
 * @brief bubbleSort 对链表进行排序
 * @param head 链表头指针
 */
void bubbleSort(Node *head){
    // 1.计算链表长度
    int len = listLength(head);
    // 2.定义变量记录前后节点
    Node *cur = NULL;
   // 3.相邻元素进行比较, 进行冒泡排序
    for(int i = 0; i < len - 1; i++){
        cur = head->next;
        for(int j = 0; j < len - 1 - i; j++){
            printf("%i, %i\n", cur->data, cur->next->data);
            if((cur->data) > (cur->next->data)){
                int temp = cur->data;
                cur->data = cur->next->data;
                cur->next->data = temp;
            }
            cur = cur->next;
        }
    }
}
```

```c
/**
 * @brief sortList 对链表进行排序
 * @param head 链表头指针
 */
void sortList(Node *head){
    // 0.计算链表长度
    int len = listLength(head);
    // 1.定义变量保存前后两个节点
    Node *sh, *pre, *cur;
    for(int i = 0; i < len - 1; i ++){
        sh = head; // 头节点
        pre = sh->next; // 第一个节点
        cur = pre->next; // 第二个节点
        for(int j = 0; j < len - 1 - i; j++){
            if(pre->data > cur->data){
                // 交换节点位置
                sh->next = cur;
                pre->next = cur->next;
                cur->next = pre;
                // 恢复节点名称
                Node *temp = pre;
                pre = cur;
                cur = temp;
            }
            // 让所有节点往后移动
            sh = sh->next;
            pre = pre->next;
            cur = cur->next;
        }
    }
}
```

- 链表反转

```c
/**
 * @brief reverseList 反转链表
 * @param head 链表头指针
 */
void reverseList(Node *head){
    // 1.将链表一分为二
    Node *pre, *cur;
    pre = head->next;
    head->next = NULL;
    // 2.重新插入节点
    while(pre){
        cur = pre->next;
        pre->next = head->next;
        head->next = pre;

        pre = cur;
    }
}
```

## Swift实现链表

```swift
import UIKit

// 节点
class Node {
    
    var val: Int
    var next: Node?
    
    init(_ val: Int) {
        self.val = val
        self.next = nil
    }
}

// 链表
// 头插法: 1.让新节点的下一个节点等于头节点的下一个节点。2.让头节点的下一个节点等于新节点。规律新的节点永远都是插入到头结点后面
// 尾插法: 1.定义变量记录新节点的上一个节点。2.将新节点添加到上一个节点的后面。3.让新节点成为下一个节点的上一个节点。规律: 新节点永远添加到最后面
// 注意点: 1. 一定要注意头节点可能就是nil,所以给定链表后，要看清楚head是不是optional,然后判定是不是要处理这种边界条件
//        2. 注意每个节点的next可能是nil。如果不为nil，则用“!”修饰变量。在赋值的时候也要注意"!"，将optionald节点传给废optional节点的情况
class List {
    
    var head: Node?
    var tail: Node?
    
    // 尾插法
    func appendToTail(_ val: Int) {
        if tail == nil {
            tail = Node(val)
            head = tail
        } else {
            tail!.next = Node(val)
            tail = tail!.next
        }
    }
    
    // 头插法
    func appendToHead(_ val: Int) {
        if head == nil {
            head = Node(val)
            tail = head
        } else {
            let temp = Node(val)
            temp.next = head
            head = temp
        }
    }
    
    // 遍历链表
    func traverse() {
        if head == nil {
            fatalError("没有创建过链表")
        }
        var node = head
        while node != nil {
            print(node!.val)
            node = node!.next
        }
    }
    
    // 给出一个链表和一个值x,要求将链表中所有小于x的值放到左边，所有大于或者等于x的值放到右边
    // 并且原链表的节点顺序不变。如: 1->5->3->2->4->2，给定x=3;则返回1->2->2->5->3->4
    func partition(_ x: Int) {
        // 引入dummy节点
        var pre = Node(0), post = Node(0)
        let preV = pre, postV = post
        var node = head
        
        // 用尾插法处理左边和右边
        while node != nil { // 遍历
            if node!.val < x {
                pre.next = node
                pre = node!
            } else {
                post.next = node
                post = node!
            }
            node = node!.next
        }
        // 防止构成环
        post.next = nil
        // 左右拼接
        pre.next = postV.next
        head = preV.next
    }
    
    // 快行指针
    // 如何检测一个链表中是否有环？
    // 用两个指针同时访问链表，其中一个的速度是另一个的两倍，如果他们变成相等的了，那么这个链表就有环了
    func hasCycle() -> Bool {
        var slow = head
        var fast = head
        while fast != nil && fast!.next != nil {
            slow = slow!.next
            fast = fast!.next!.next
            if slow === fast {
                return true
            }
        }
        return false
    }
    
    // 删除链表中倒数第n个节点。如：1->5->3->2->4->2, n = 2 返回 1->5->3->2->2
    // 思路依然是快行指针，这次两个指针的移动速度相同。但是一开始，第一个指针(在指向头结点之前)就落后第二个指针n个节点。
    // 接着两者同时移动，当第二个节点指针移动到尾节点是，第一个节点的下一个节点就是我们要删除的节点
    func deleteNthFromEnd(_ n: Int) {
        
        let dummy = Node(0)
        dummy.next = head
        
        var pre: Node? = dummy
        var post: Node? = dummy
        // 一开始post就落后n位
        for _ in 0..<n {
            if post == nil {
                break
            }
            post = post!.next
        }
        // 同时移动两个指针
        while post != nil && post!.next != nil {
            pre = pre!.next
            post = post!.next
        }
        // 删除节点
        pre!.next = pre!.next!.next
        head = dummy.next
    }
}
```

```swift
// 链表
func listTest() {
    
    let list = List()
    for _ in 0..<10 {
        let num = Int(arc4random_uniform(UInt32(10)))
        list.appendToHead(num)
    }
//    print("遍历链表")
//    list.traverse()
//    list.partition(5)
//    print("重新遍历链表")
//    list.traverse()
//    print(list.hasCycle())
    
    print("遍历链表")
    list.traverse()
    list.deleteNthFromEnd(3)
    print("删除第n个后链表")
    list.traverse()
}
```

## 小结

前面用Swift实现了链表的基本结构，并且介绍了链表的几个技巧。在本节最后，我们还想敲掉一下用Swift处理链表的问题应注意的两个细节:

+ 一定要注意头节点可能是nil。所以给定链表后，要清楚head是不是optional，然后判断是不是要处理这种边界条件。
+ 注意每个节点的next可能是nil，如果不为nil，则用"!"修饰变量。在赋值的时候也要注意"!"将optional节点传给非optional节点的情况。

[源码地址](<https://github.com/Jovins/Algorithm>)