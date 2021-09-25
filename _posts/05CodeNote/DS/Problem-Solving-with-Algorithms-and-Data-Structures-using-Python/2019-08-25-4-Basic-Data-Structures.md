---
title: pythonds3 - 4. Basic Data Structures
# author: Grace JyL
date: 2019-08-25 11:11:11 -0400
description:
excerpt_separator:
categories: [1CodeNote, PythonNote]
tags:
math: true
# pin: true
toc: true
# image: /assets/img/sample/devices-mockup.png
---

[toc]

---

# 4. Basic Data Structures

Problem Solving with Algorithms and Data Structures using Python 4

---

## 4.2. Linear Structures
Stacks, queues, deques, and lists are examples of data collections whose items are **ordered** depending on `how they are added or removed`.
- Once an item is added, it stays in that position relative to the other elements that came before and came after it.
- these Collections are often referred as **linear data structures**.


Linear structures can be thought of as having two ends.
- “left” and the “right”
- or “front” and the “rear”
- “top” and the “bottom.”


What distinguishes one linear structure from another is `the way in which items are added and removed`, in particular the location where these additions and removals occur.

唯一前后，左右 前后，
- 添加，去除，插入
- stack, queue, deque, list


栈stack：先进后出 FILO，操作较快；缺点：查询慢，读非栈顶数值需要遍历

队列queue：先进先出 FIFO，同样操作较快；缺点：读取内部数值需要遍历

表list：可以根据索引取值；缺点：插入和删除是O(n)的

![Screen Shot 2020-05-27 at 17.33.48](https://i.imgur.com/A4GXdUf.png)

![Screen Shot 2020-05-27 at 17.34.09](https://i.imgur.com/w1SEFHH.png)

![Screen Shot 2020-05-27 at 17.35.05](https://i.imgur.com/qDjs9OT.png)

---

## 4.3. Stack?

- “push-down stack”
- an `ordered collection of items`
- the addition and the removal always takes place at the same end.
- This end is commonly referred to as the “top” and “base”

<kbd>LIFO, last-in first-out</kbd>
- items stored closer to the base, been in the stack the longest.
- The most recently added item, be removed first.
- It provides an ordering based on length of time in the collection.
- Newer items are near the top, while older items are near the base.

![bookstack2](https://i.imgur.com/q4Ncl3u.png)

![primitive](https://i.imgur.com/HvrFbiF.png)

Stacks are fundamentally important, as they can be used to reverse the order of items. The order of insertion is the reverse of the order of removal.
- every web browser's Back button. url in stack

![simplereversal](https://i.imgur.com/in4R6v7.png)

---

### Stack Data Structure

Stack is a `linear data structure`
- follows a particular order in which the operations are performed.
- LIFO(Last In First Out), FILO(First In Last Out).


### 4.4 The Stack Abstract Data Type

![Screen Shot 2020-05-26 at 14.28.21](https://i.imgur.com/viZ9E8J.png)

![Screen Shot 2020-05-26 at 14.29.19](https://i.imgur.com/rTL2FM6.png)

stack operations

- `Stack()`
  - creates a new stack that is empty.
  - It needs no parameters and **returns an empty stack**.
- `push(item)`:
  - adds a new item to the top of the stack
  - It needs the item and **returns nothing**.
  - If the stack is full, Overflow condition.
- `pop()`:
  - removes the top item from the stack
  - It needs no parameters and **returns the item**
  - The stack is modified.
  - If the stack is empty, Underflow condition.
- `peek()` or `Top`:
  - Returns top element of stack, but does not remove it
  - It needs no parameters.
  - The stack is not modified.
- `is_empty()`:
  - Returns true if stack is empty, else false.
  - It needs no parameters and returns a boolean value.
- `size()`
  - returns the number of items on the stack.
  - It needs no parameters and **returns an integer**

---


### code

![Screen Shot 2020-05-26 at 14.45.22](https://i.imgur.com/uxkuJEi.png)


#### example:

uses a stack to reverse the characters in a string. [code](https://github.com/ocholuo/language/tree/master/0.code/leecode/Algorithms/2020-09-06-In-reverse-string.py)


simple Balanced Parentheses. [code](https://github.com/ocholuo/language/tree/master/0.code/leecode/leecode_Java_Py/020.Valid-Parentheses)

![simpleparcheck](https://i.imgur.com/2xB79Lt.png)

Balanced Symbols (A General Case)


convert-integer-into-different-base: [code](https://github.com/ocholuo/language/tree/master/0.code/leecode/Algorithms/convert-integer-into-otherBase.py)


![dectobin](https://i.imgur.com/sdKnZuW.png)



#### 4.5. Implementing a Stack in Python

```py
class Stack:
    """Stack implementation as a list"""

    def __init__(self):
        """Create new stack"""
        self._items = []

    def is_empty(self):
        """Check if the stack is empty"""
        return not bool(self._items)

    def push(self, item):
        """Add an item to the stack"""
        self._items.append(item)

    def pop(self):
        """Remove an item from the stack"""
        return self._items.pop()

    def peek(self):
        """Get the value of the top item in the stack"""
        return self._items[-1]

    def size(self):
        """Get the number of items in the stack"""
        return len(self._items)

from pythonds3.basic import Stack

s = Stack()
```

> Recall that the append and pop() operations were both O(1).

> The performance of the second implementation suffers in that the insert(0) and pop(0) operations will both require O(n) for a stack of size n.


#### stack in java

```java
class Stack {

	static final int MAX = 1000;
	int top;
	int a[] = new int[MAX]; // Maximum size of Stack

	boolean isEmpty() {
		return (top < 0);
	}

	Stack() {
		top = -1;   // empty: top is -1
	}

    // add item
	boolean push(int x) {
		if (top >= (MAX - 1)) {
			System.out.println("Stack Overflow");
			return false;
		}
		else {
			a[++top] = x;
			System.out.println(x + " pushed into stack");
			return true;
		}
	}

	int pop() {
		if (top < 0) {
			System.out.println("Stack Underflow");
			return 0;
		}
		else {
			int x = a[top--];
			return x;
		}
	}

	int peek() {
		if (top < 0) {
			System.out.println("Stack Underflow");
			return 0;
		}
		else {
			int x = a[top];
			return x;
		}
	}
}

// Driver code
class Main {

	public static void main(String args[]) {
		Stack s = new Stack();
		s.push(10);
		s.push(20);
		s.push(30);
		System.out.println(s.pop() + " Popped from stack");
	}
}
```

---

#### Stack using Linked List

```java
public class StackAsLinkedList {

	StackNode root;

	static class StackNode {
		int data;
		StackNode next;

		StackNode(int data) {
			this.data = data;
		}
	}

	public boolean isEmpty() {
		if (root == null) {
			return true;
		}
		else
			return false;
	}

	public void push(int data) {
		StackNode newNode = new StackNode(data);

		if (root == null) {
			root = newNode;
		}
		else {
			StackNode temp = root;
			root = newNode;
			newNode.next = temp;
		}
		System.out.println(data + " pushed to stack");
	}

	public int pop() {
		int popped = Integer.MIN_VALUE;
		if (root == null) {
			System.out.println("Stack is Empty");
		}
		else {
			popped = root.data;
			root = root.next;
		}
		return popped;
	}

	public int peek() {
		if (root == null) {
			System.out.println("Stack is empty");
			return Integer.MIN_VALUE;
		}
		else {
			return root.data;
		}
	}

	public static void main(String[] args) {
		StackAsLinkedList sll = new StackAsLinkedList();
		sll.push(10);
		sll.push(20);
		sll.push(30);
		System.out.println(sll.pop() + " popped from stack");
		System.out.println("Top element is " + sll.peek());
	}
}
```


---

#### Stack Class in Java

```java
// Java code for stack implementation

import java.io.*;
import java.util.*;

class Test {
	// Pushing element on the top of the stack
	static void stack_push(Stack<Integer> stack) {
		for(int i = 0; i < 5; i++) {
			stack.push(i);
		}
	}

	// Popping element from the top of the stack
	static void stack_pop(Stack<Integer> stack) {
		System.out.println("Pop :");
		for(int i = 0; i < 5; i++) {
			Integer y = (Integer) stack.pop();
			System.out.println(y);
		}
	}

	// Displaying element on the top of the stack
	static void stack_peek(Stack<Integer> stack) {
		Integer element = (Integer) stack.peek();
		System.out.println("Element on stack top : " + element);
	}

	// Searching element in the stack
	static void stack_search(Stack<Integer> stack, int element) {
		Integer pos = (Integer) stack.search(element);
		if(pos == -1)
			System.out.println("Element not found");
		else
			System.out.println("Element is found at position " + pos);
	}
}
```

---

## 4.9. Infix, Prefix, and Postfix Expressions


> computers need to know exactly what operators to perform and in what order.

**infix**: the operator is in between the two operands that it is working on.
**Prefix**: all operators precede the two operands that they work on.
**Postfix**: operators come after the corresponding operands.


- 前缀`(*+ABC)`，后缀`(AB+C*)`，中缀`(A+B)*C`

![Screen Shot 2020-05-26 at 16.02.12](https://i.imgur.com/YlduKeO.png)


`fully parenthesized expression`: uses one pair of parentheses for each operator.
- `A + B * C + D` -> `((A + (B * C)) + D)`
- `A + B + C + D` -> `(((A + B) + C) + D)`


### 4.9.1. Conversion of Infix Expressions to Prefix and Postfix

![Screen Shot 2020-05-26 at 16.05.36](https://i.imgur.com/XHTifZt.png)

### 4.9.2. General Infix-to-Postfix Conversion

[code](https://github.com/ocholuo/language/tree/master/0.code/leecode/Algorithms/convert-infix-prefix-postfix.py)

![intopost](https://i.imgur.com/avxcc7z.png)


### 4.9.3. Postfix Evaluation

[code](https://github.com/ocholuo/language/tree/master/0.code/leecode/Algorithms/convert-infix-prefix-postfix.py)

![evalpostfix1](https://i.imgur.com/11IM0tC.png)

![evalpostfix2](https://i.imgur.com/BuVv3tc.png)

---


## 4.10. Queue

![basicqueue](https://i.imgur.com/ODLgXMw.png)

Queue is used when things don’t have to be processed immediatly, but have to be processed in First InFirst Out order like Breadth First Search. This property of Queue makes it also useful in following kind of scenarios.
- printing queues,
- operating systems use different queues to control processes
- keystrokes are being placed in a queue-like buffer so that they can eventually be displayed on the screen in the proper order.
- When a resource is shared among multiple consumers.
  - Examples include CPU scheduling, Disk Scheduling.
- When data is transferred asynchronously (data not necessarily received at same rate as sent) between two processes.
  - Examples include IO Buffers, pipes, file IO, etc.

queue
- FIFO: first in first out.
- only change from 2 side. **no insert**
  - front : rear

![Screen Shot 2020-05-26 at 22.35.29](https://i.imgur.com/Yqex15c.png)

- `Queue()`
  - creates a new queue that is empty.
  - It needs no parameters and returns an empty queue.
- `enqueue(item)`  <kbd>𝑂(𝑛)</kbd>
  - adds a new item to the rear of the queue.
  - It needs the item and **returns nothing**.
- `dequeue()` <kbd>𝑂(1)</kbd>
  - removes the front item from the queue.
  - It needs no parameters and **returns the item**.
  - The queue is modified.
- `is_empty()`
  - tests to see whether the queue is empty.
  - It needs no parameters and returns a boolean value.
- `size()`
  - returns the number of items in the queue.
  - It needs no parameters and returns an integer.

---

### code

#### example:

Simulation: Hot Potato [code](https://github.com/ocholuo/language/tree/master/0.code/leecode/Algorithms/queue-hot-potato.py)

![hotpotato](https://i.imgur.com/VB33sdN.png)

![namequeue](https://i.imgur.com/k57sicw.png)

Simulation: Printing Tasks

![simulationsetup](https://i.imgur.com/2RoMcuf.png)



Simulation: Printing Tasks

#### impliment in py

```py
class Queue:
    """Queue implementation as a list"""

    def __init__(self):
        """Create new queue"""
        self._items = []

    def is_empty(self):
        """Check if the queue is empty"""
        return not bool(self._items)

    def enqueue(self, item):
        """Add an item to the queue"""
        self._items.insert(0, item)

    def dequeue(self):
        """Remove an item from the queue"""
        return self._items.pop()

    def size(self):
        """Get the number of items in the queue"""
        return len(self._items)
```

#### implement by java


```java
// A class to represent a queue
class Queue {

	int front, rear, size;
	int capacity;
	int array[];

	public Queue(int capacity) {
		this.capacity = capacity;
		front = this.size = 0;
		rear = capacity - 1;
		array = new int[this.capacity];
	}

	// Queue is full when size becomes equal to the capacity
	boolean isFull(Queue queue) {
		return (queue.size == queue.capacity);
	}

	// Queue is empty when size is 0
	boolean isEmpty(Queue queue) {
		return (queue.size == 0);
	}

	// Method to add an item to the queue.
	// It changes rear and size
	void enqueue(int item) {
		if (isFull(this))
			return;
		this.rear = (this.rear + 1) % this.capacity;
		this.array[this.rear] = item;
		this.size = this.size + 1;
		System.out.println(item + " enqueued to queue");
	}

	// Method to remove an item from queue.
	// It changes front and size
	int dequeue() {
		if (isEmpty(this))
			return Integer.MIN_VALUE;

		int item = this.array[this.front];
		this.front = (this.front + 1) % this.capacity;
		this.size = this.size - 1;
		return item;
	}

	// Method to get front of queue
	int front() {
		if (isEmpty(this))
			return Integer.MIN_VALUE;
		return this.array[this.front];
	}

	// Method to get rear of queue
	int rear() {
		if (isEmpty(this))
			return Integer.MIN_VALUE;
		return this.array[this.rear];
	}
}

// Driver class
public class Test {
	public static void main(String[] args) {
		Queue queue = new Queue(1000);

		queue.enqueue(10);
		queue.enqueue(20);
		queue.enqueue(30);
		queue.enqueue(40);

		System.out.println(queue.dequeue() + " dequeued from queue\n");
		System.out.println("Front item is " + queue.front());
		System.out.println("Rear item is " + queue.rear());
	}
}

// Output:
// 10 enqueued to queue
// 20 enqueued to queue
// 30 enqueued to queue
// 40 enqueued to queue
// 10 dequeued from queue
// Front item is 20
// Rear item is 40
```

---

## 4.15. deque


- double-ended queue
- an ordered collection of items similar to the queue.
- It has two ends, a front and a rear, and the items remain positioned in the collection.

What makes a deque different is the unrestrictive nature of adding and removing items.
- New items can be added at either the front or the rear.
- Likewise, existing items can be removed from either end.
- provides the capabilities of stacks and queues in a single data structure.
- 可以模拟stack或者queue

It is important to note that even though the deque can assume many of the characteristics of stacks and queues, it does not require the LIFO and FIFO orderings that are enforced by those data structures.
- It is up to you to make consistent use of the addition and removal operations.

![basicdeque](https://i.imgur.com/5dSSfi2.png)


### 4.16. The Deque Abstract Data Type

![Screen Shot 2020-05-27 at 16.08.22](https://i.imgur.com/1N3DXwM.png)

![Screen Shot 2020-05-27 at 16.08.36](https://i.imgur.com/bplESLY.png)

- `Deque()`
  - creates a new deque that is empty.
  - It needs no parameters and returns an empty deque.
- `add_front(item)`
  - adds a new item to the front of the deque.
  - It needs the item and returns nothing.
- `add_rear(item)`
  - adds a new item to the rear of the deque.
  - It needs the item and returns nothing.
- `remove_front()`
  - removes the front item from the deque.
  - It needs no parameters and returns the item.
  - The deque is modified.
- `remove_rear()`
  - removes the rear item from the deque.
  - It needs no parameters and returns the item.
  - The deque is modified.
- `is_empty()`
  - tests to see whether the deque is empty.
  - It needs no parameters and returns a boolean value.
- `size()`
  - returns the number of items in the deque.
  - It needs no parameters and returns an integer.

### code


#### implement in py

```py
class Deque:
    """Queue implementation as a list"""

    def __init__(self):
        """Create new deque"""
        self._items = []

    def is_empty(self):
        """Check if the deque is empty"""
        return not bool(self._items)

    def add_front(self, item):
        """Add an item to the front of the deque"""
        self._items.append(item)

    def add_rear(self, item):
        """Add an item to the rear of the deque"""
        self._items.insert(0, item)

    def remove_front(self):
        """Remove an item from the front of the deque"""
        return self._items.pop()

    def remove_rear(self):
        """Remove an item from the rear of the deque"""
        return self._items.pop(0)

    def size(self):
        """Get the number of items in the deque"""
        return len(self._items)
```


#### 4.18. Palindrome-Checker

![palindromesetup](https://i.imgur.com/BiBLdwq.png)

```py
def pal_checker(a_string):
    char_deque = Deque()

    for ch in a_string:
        char_deque.add_rear(ch)

    while char_deque.size() > 1:
        first = char_deque.remove_front()
        last = char_deque.remove_rear()
        if first != last:
            return False
    return True

palchecker("lsdkjfhd")
palchecker("radar")    
```

---

## 4.19. Lists

- a collection of items
- each item holds a relative position with respect to the others.
- More specifically, we will refer to this type of list as an unordered list.
- [54, 26, 93, 17, 77, 31].


### 4.20. The Unordered List Abstract Data Type

- `List()`
  - creates a new list that is empty.
  - It needs no parameters and returns an empty list.
- `add(item)`
  - adds a new item to the list.
  - It needs the item and returns nothing.
  - Assume the item is not already in the list.
- `remove(item)`
  - removes the item from the list.
  - It needs the item and modifies the list.
  - Raise an error if the item is not present in the list.
- `search(item)`
  - searches for the item in the list.
  - It needs the item and returns a boolean value.
- `is_empty()`
  - tests to see whether the list is empty.
  - It needs no parameters and returns a boolean value.
- `size()`
  - returns the number of items in the list.
  - It needs no parameters and returns an integer.
- `append(item)`
  - adds a new item to the end of the list making it the last item in the collection.
  - It needs the item and returns nothing.
- `index(item)`
  - returns the position of item in the list.
  - It needs the item and returns the index.
- `insert(pos, item)`
  - adds a new item to the list at position pos.
  - It needs the item and returns nothing.
- `pop()`
  - removes and returns the last item in the list.
  - It needs nothing and returns an item.
- `pop(pos)`
  - removes and returns the item at position pos.
  - It needs the position and returns the item.

---

## 4.21. Unordered List: Linked Lists


![idea2](https://i.imgur.com/SqXvGO8.png)


无序表： `unordered list`
- 一种数据按照相对位置存放的数据集
- (for easy, assum that no repeat)
- 无序存放，但是在数据相之间建立`链接指向`, 就可以保持其前后相对位置。
  - 显示标记 `head` `end`
- 每个节点 `node` 包含2信息：
  - 数据本身，指向下一个节点的引用信息`next`
  - `next=None` 没有下一个节点了

### 4.21.1. The Node Class

the constructor that a node is initially created with next set to `None`.
- sometimes referred to as “grounding the node,”
- use the standard ground symbol to denote a reference that is referring to `None`

![node](https://i.imgur.com/CK40mon.png)

![node2](https://i.imgur.com/b0X4X3K.png)

```py
class Node:
    """A node of a linked list"""

    def __init__(self, node_data):
        self._data = node_data
        self._next = None

    def get_data(self):
        """Get node data"""
        return self._data

    def set_data(self, node_data):
        """Set node data"""
        self._data = node_data

    data = property(get_data, set_data)

    def get_next(self):
        """Get next node"""
        return self._next

    def set_next(self, node_next):
        """Set next node"""
        self._next = node_next

    next = property(get_next, set_next)

    def __str__(self):
        """String"""
        return str(self._data)

# create Node objects in the usual way.
>>> temp = Node(93)
>>> temp.data
93
```

### 4.21.2. The Unordered List Class

A linked list, then, is nothing more than a single chain of nodes with a few well defined properties and methods such as:
- Head Pointer: This is a pointer to the origin, or first node in a linked list. Only when the list has a length of 1 will it’s value be None.
- Tail Pointer: This is a pointer to the last node in a list. When a list has a length of 1, the Head and the Tail refer to the same node. By definition, the Tail will have a next value of None.
- Count*: We’ll also be keeping track of the number of nodes we have in our linked list. Though this is not strictly necessary, I find it to be more efficient and convenient than iterating through the entire linked list when polling for size.

![1_73b9zu3H5pjLd8W0RZPeng](https://i.imgur.com/fFWCFCl.jpg)

```py
class Node:
    def __init__(self, node_data):
        self._data = node_data
        self._next = None

    def get_data(self):
        return self._data

    def set_data(self, node_data):
        self._data = node_data

    data = property(get_data, set_data)

    def get_next(self):
        return self._next

    def set_next(self, node_next):
        self._next = node_next

    next = property(get_next, set_next)

    def __str__(self):
        return str(self._data)


class UnorderedList:
    def __init__(self):
        self.head = None
        self.tail = None  # add the tail point
        self.count = 0

    def __str__(self):
        list_str = "head"
        current = self.head
        while current != None:
            list_str = list_str +  "->" + str(current.get_data())
            current = current.get_next()
        list_str = list_str +  "->" + str(None)
        return list_str

    def is_empty(self):
        return self.head == None

    # """
    # Add node to start of list
           # (Head) -> [2] -> [3](Tail)
    # (Head) -> [1] -> [2] -> [3](Tail)
    # """
    def add_to_start(self, item):
        temp = Node(item)
        temp.set_next(self.head)
        self.head = temp
        self.count += 1
        if self.count == 1:
            self.tail = self.head
            # 只有从头来 会设定tail

    # """
    # Add node to end of list
    # (Head)1 -> 2(Tail)
    # (Head)1 -> 2 -> 3(Tail)
    # """
    def add_to_end(self, item):
        temp = Node(item)
        if self.count == 0:
            self.head = new_node
        else:
            self.tail.next = new_node
        self.tail = new_node
        self.count += 1


    def size(self):
        return self.count


    def search(self, item):
        current = self.head
        while current is not None:
            if current.data == item:
                return True
            current = current.next
        return False


    # """    
    # Remove node from start of list
    # (Head)[1] -> [2] -> [3](Tail)
    #        (Head)[2] -> [3](Tail)
    # """
    def remove_first(self):
        if self.count > 0:
            self.head = self.head.next
            self.count -= 1
            if self.count == 0:
                self.tail = None  

    # """
    # Remove node from end of list
    # (Head)1 -> 2 -> 3(Tail)
    # (Head)1 -> 2(Tail)
    # """
	# def remove_last(self):
	# 	if self.count > 0:
	# 		if self.count == 1:
	# 			self.head = None
	# 			self.tail = None
	# 		else:
	# 			current = self.head
	# 			while current.next != self.tail:
	# 				current = current.next
	# 			current.next = None
	# 			self.tail = current
	# 		self.count -= 1


    # """
    #     Remove node by value
    #     (Head)[1] -> [2] -> [3](Tail)
    #     (Head)[1] --------> [3](Tail)
    # """
    def remove_by_value(self, item):
        current = self.head
        previous = None

        while current is not None:
            if current.data == item:
                break
            previous = current
            current = current.next

        if current is None:
            raise ValueError("{} is not in the list".format(item))
        if previous is None:
            self.head = current.next
        else:
            previous.next = current.next

    def append(self, item):
        temp = Node(item)
        # print(temp)
        # print(self.tail)
        # print(self.tail.next)
        temp.next = self.tail.next
        self.tail.next = temp
        self.tail=temp

my_list = UnorderedList()
my_list.add_to_start(31)
my_list.add_to_start(77)
my_list.add_to_start(17)
my_list.add_to_start(93)
my_list.add_to_start(26)
my_list.add_to_start(54)
my_list.append(123)
print(my_list)
```

#### `unorderedList`:
   - 无序表必须要有对第一个节点的引用信息
   - 设立属性head，保存对第一个节点的引用空表的head为None
   - the unordered list will be built from a collection of nodes, each linked to the next by explicit references.
   - As long as we know where to find the first node (containing the first item), each item after that can be found by successively following the next links.
   - the UnorderedList class must maintain a reference to the first node.
   - each `list` object will maintain a single reference to the head of the list.


```py
class UnorderedList:
    def __init__(self):
        self.head = None

# Initially when construct a list, there are no items.
mylist = UnorderedList()
print(mylist.head)
# None
```

---

#### `is_empty()`
   - the special reference `None` will again be used to state that the head of the list does not refer to anything.
   - Eventually, the example list given earlier will be represented by a linked list as below

![initlinkedlist](https://i.imgur.com/HugjffZ.png)

```py
# checks to see if the head of the list is a reference to None.
# The result of the boolean expression self.head == None will only be true if there are no nodes in the linked list.
def is_empty(self):
    return self.head == None
```

![linkedlist](https://i.imgur.com/t0sWHTx.png)

- The `head` of the list refers to the `first node` which contains the `first item of the list`.
- In turn, that node holds a reference to the next node (the next item) and so on.
- **the list class itself does not contain any node objects**.
- Instead it contains `a single reference to the first node in the linked structure`.

---

#### `add()`
   - The new item can go anywhere
   - item added to the list will be the last node on the linked list

```py
def add(self, item):
    temp = Node(item)
    temp.set_next(self.head)
    self.head = temp
```

![addtohead](https://i.imgur.com/otXEbFu.png)

![wrongorder](https://i.imgur.com/lqFETgv.png)

---

`size`, `search`, and `remove`
- all based on a technique known as linked list traversal
- Traversal refers to the process of systematically visiting each node.

#### `size()`
- use an external reference that starts at the first node in the list.
- visit each node, move the reference to the next node by “traversing” the next reference.
- traverse the linked list and keep a count of the number of nodes that occurred.

![traversal](https://i.imgur.com/0v26VgY.png)

```py
def size(self):
    current = self.head
    count = 0
    while current is not None:
        count = count + 1
        current = current.next
    return count
```

#### `search(item):`
- Searching for a value in a linked list implementation of an unordered list also uses the traversal technique.
- visit each node in the linked list, ask whether the data matches the item
- may not have to traverse all the way to the end of the list.
  - if get to the end of the list, that means that the item we are looking for must not be present.
  - if we do find the item, there is no need to continue.

![search](https://i.imgur.com/613fLHu.png)

```py
def search(self, item):
    current = self.head
    while current is not None:
        if current.data == item:
            return True
        current = current.next
    return False
```

#### `remove()`
- requires two logical steps.
- traverse the list for the item to remove.
  - Once find the item , must remove it.
  - If item is not in the list, raise a ValueError.
  - The first step is very similar to search.
    - Starting with an external reference set to the head of the list,
    - traverse the links until discover the item
    - When the item is found, break out of the loop
- use two external references as we traverse down the linked list.
  - `current`, marking the current location of the traverse.
  - `previous`, always travel one node behind current.

![remove](https://i.imgur.com/j75iXk5.png)

![remove2](https://i.imgur.com/CoWnc9o.png)

```py
def remove(self, item):
    current = self.head
    previous = None

    while current is not None:
        if current.data == item:
            break
        previous = current
        current = current.next

    if current is None:
        raise ValueError("{} is not in the list".format(item))
    if previous is None:   # remove the frist item
        self.head = current.next
    else:
        previous.next = current.next
```


#### `pop()`

```py
def pop(self, index):
    self.remove(self.getItem(index))
```


#### `append()`

```py
# 1. 𝑂(𝑛)
def append(self, item):
    current = self.head
    while current.set_next() is not None:
        current = current.set_next()

    temp = Node(item)
    temp.set_next(current.set_next())
	current.set_next(temp)

# 2. 𝑂(1)
# use tail point & head point
```

#### `insert()`

```py
def insert(self, index, item):
    current = self.head
	# count = 0
    # while current is not None:
	# 	if count == index:
	# 		temp = Node(item)
	# 		temp.set_next(current.set_next())
	# 		current.set_next(temp)
	# 		break
	# 	current = current.set_next()
	# 	count += 1
    for i in range(index):
        current = current.set_next()
	if current != None:
        temp = Node(item)
        temp.set_next(current.set_next())
        current.set_next(temp)
    else:
        raise("index out of range")
```


#### `index()`

```py
def index(self, index):
    current = self.head
    for i in range(index):
        current = current.set_next()
	if current != None:
		return current.get_data()
    else:
        raise("index out of range")
```

---

## 4.22. The Ordered List Abstract Data Type

### 4.23. Implementing an Ordered List in py

```py
class OrderedList:
    def __init__(self):
        self.head = None
        self.count = 0

    def is_empty(self):
        return self.head == None

    def size(self):
        return self.count

    def remove(self, item):
        current = self.head
        previous = None
        while current is not None:
            if current.data == item:
                break
            previous = current
            current = current.next
        if current is None:
            raise ValueError("{} is not in the list".format(item))
        if previous is None:   # remove the frist item
            self.head = current.next
        else:
            previous.next = current.next

    def search(self, item):
        current = self.head
        while (current is not None):
            if current.data > item:
                return False
            if current.data == item:
                return True
            current = current.next
        return False

    def add(self, item):
        temp = Node(item)
        current = self.head
        previous = None
        self.count += 1
        while (current is not None) and current.data < item:
            previous = current
            current = current.next
        if previous is None:
            temp.next = self.head
            self.head = temp
        else:
            temp.next = current
            previous.next = temp

my_list = OrderedList()
my_list.add(31)
my_list.add(77)
my_list.add(17)
my_list.add(93)
my_list.add(26)
my_list.add(54)

print(my_list.size())
print(my_list.search(93))
print(my_list.search(100))
```

#### Node:

```py
class Node:
    def __init__(self, node_data):
        self._data = node_data
        self._next = None

    def get_data(self):
        return self._data

    def set_data(self, node_data):
        self._data = node_data

    data = property(get_data, set_data)

    def get_next(self):
        return self._next

    def set_next(self, node_next):
        self._next = node_next

    next = property(get_next, set_next)

    def __str__(self):
        return str(self._data)
```


#### an empty list

```py
class OrderedList:
    def __init__(self):
        self.head = None
		self.count = 0
```

#### `is_empty` and `size`

some method will work just fine since

`is_empty` <kbd>𝑂(1)</kbd>

`size` <kbd>𝑂(1)</kbd>

`remove` <kbd>𝑂(n)</kbd>


```py
def is_empty(self):
	return self.head == None

def size(self):
	return self.count

def remove(self, item):
    current = self.head
    previous = None
    while current is not None:
        if current.data == item:
            break
        previous = current
        current = current.next
    if current is None:
        raise ValueError("{} is not in the list".format(item))
    if previous is None:   # remove the frist item
        self.head = current.next
    else:
        previous.next = current.next
```

#### `search()`

![orderedsearch](https://i.imgur.com/cXdshUF.png)

<kbd>𝑂(n)</kbd>

as in order, if current data is bigger, no need to continue

```py
def search(self, item):
    current = self.head
    while (current is not None):
		if current.data > item:
			return False
        if current.data == item:
            return True
        current = current.next
    return False
```

#### `add()`

![linkedlistinsert](https://i.imgur.com/dZE3tzH.png)

```py
def add(self, item):
    temp = Node(item)
    current = self.head
    previous = None
	self.count += 1
	while (current is not None) and current.data < item:
		previous = current
		current = current.next
    if previous is None:
        temp.next = self.head
        self.head = temp
    else:
        temp.next = current
        previous.next = temp
```



















---

ref
- [Problem Solving with Algorithms and Data Structures using Python](https://runestone.academy/runestone/books/published/pythonds3/BasicDS/ImplementinganOrderedList.html)
.
