---
title: DS - pythonds3 - 4. Basic Data Structures
# author: Grace JyL
date: 2019-08-25 11:11:11 -0400
description:
excerpt_separator:
categories: [04CodeNote, PythonNote]
tags:
math: true
# pin: true
toc: true
# image: /assets/img/sample/devices-mockup.png
---

- [DS - pythonds3 - 4. Basic Data Structures](#ds---pythonds3---4-basic-data-structures)
  - [Linear Structures](#linear-structures)
  - [Stack](#stack)
    - [stack operations](#stack-operations)
    - [code](#code)
      - [Stack <- list  (!!!!!!!!!!!!!)](#stack---list--)
      - [stack in java](#stack-in-java)
      - [Stack <- Linked List](#stack---linked-list)
      - [Stack Class in Java](#stack-class-in-java)
      - [reverse char in string](#reverse-char-in-string)
      - [simple Balanced Parentheses](#simple-balanced-parentheses)
      - [Balanced Symbols (A General Case)](#balanced-symbols-a-general-case)
      - [convert-integer-into-different-base](#convert-integer-into-different-base)
      - [Infix, Prefix, and Postfix Expressions](#infix-prefix-and-postfix-expressions)
  - [Queue](#queue)
    - [code](#code-1)
      - [queue as a list  (!!!!!!!!!!!!!)](#queue-as-a-list--)
      - [queue in java](#queue-in-java)
      - [Simulation: Hot Potato](#simulation-hot-potato)
      - [Simulation: Printing Tasks](#simulation-printing-tasks)
  - [Deque](#deque)
    - [Deque - Abstract Data Type](#deque---abstract-data-type)
    - [code](#code-2)
      - [dequeue as a list in py (!!!!!!!!!!!!!)](#dequeue-as-a-list-in-py-)
      - [Palindrome-Checker 回文 对称的单词](#palindrome-checker-回文-对称的单词)
  - [Lists](#lists)
  - [Unordered List - Abstract Data Type](#unordered-list---abstract-data-type)
    - [Unordered List: Linked Lists](#unordered-list-linked-lists)
    - [Node Class](#node-class)
      - [Node Class <- unordered linked list  (!!!!!!!!!!!!!)](#node-class---unordered-linked-list--)
    - [Unordered List Class <- unordered linked list (old)](#unordered-list-class---unordered-linked-list-old)
    - [Unordered List Class <- unordered linked list (new)  (!!!!!!!!!!!!!)](#unordered-list-class---unordered-linked-list-new--)
      - [`is_empty()`](#is_empty)
      - [`add()`](#add)
      - [`size`, `search`, and `remove`](#size-search-and-remove)
      - [`size()`](#size)
      - [`search(item):`](#searchitem)
      - [`remove()`](#remove)
      - [`pop()`](#pop)
      - [`append()`](#append)
      - [`insert()`](#insert)
      - [`index()`](#index)
  - [Ordered List - Abstract Data Type](#ordered-list---abstract-data-type)
    - [Ordered List in py (!!!!!!!!!!!!!)](#ordered-list-in-py-)

- ref:
  - https://runestone.academy/runestone/books/published/pythonds/BasicDS/toctree.html
  - [Problem Solving with Algorithms and Data Structures using Python](https://runestone.academy/runestone/books/published/pythonds3/BasicDS/ImplementinganOrderedList.html)


---

# DS - pythonds3 - 4. Basic Data Structures

Problem Solving with Algorithms and Data Structures using Python 4

---

## Linear Structures
`Stacks, queues, deques, lists`
- examples of data collections whose items are **ordered** depending on `how they are added or removed`.
- Once an item is added, it stays in that position relative to the other elements that came before and came after it.
- these Collections are often referred as **linear data structures**.

Linear structures can be thought of as having `two ends`.
- “left” and the “right”
- or “front” and the “rear”
- “top” and the “bottom.”


What distinguishes one linear structure from another is `the way in which items are added and removed`
- in particular the location where these additions and removals occur.
- 唯一前后，左右 前后，
- 添加，去除，插入
- stack, queue, deque, list


栈 stack：
- 先进后出 FILO，操作较快；
- 缺点：查询慢，读非栈顶数值需要`遍历`

队列 queue：
- 先进先出 FIFO，同样操作较快；
- 缺点：读取内部数值需要`遍历`

表 list：
- 可以根据索引取值；
- 缺点：插入和删除是O(n)的

![Screen Shot 2020-05-27 at 17.33.48](https://i.imgur.com/A4GXdUf.png)

![Screen Shot 2020-05-27 at 17.34.09](https://i.imgur.com/w1SEFHH.png)

![Screen Shot 2020-05-27 at 17.35.05](https://i.imgur.com/qDjs9OT.png)

---

## Stack

- “push-down stack”
- an `ordered collection of items`
- the addition and the removal always takes place at the same end.
- This end is commonly referred to as the “top” and “base”

<kbd>LIFO, last-in first-out</kbd>

- items stored closer to the base, been in the stack the longest.
- The most recently added item, be removed first.
- It provides an ordering based on length of time in the collection.
- Newer items are near the top, while older items are near the base.


Stacks are fundamentally important, as they can be used to reverse the order of items.
- The order of insertion is the reverse of the order of removal.
- every web browser's Back button. url in stack

![primitive](https://i.imgur.com/HvrFbiF.png)

![simplereversal](https://i.imgur.com/in4R6v7.png)


Stack is a `linear data structure`
- Abstract Data Type
- follows a particular order in which the operations are performed.
- LIFO(Last In First Out), FILO(First In Last Out).


![Screen Shot 2020-05-26 at 14.28.21](https://i.imgur.com/viZ9E8J.png)

![Screen Shot 2020-05-26 at 14.29.19](https://i.imgur.com/rTL2FM6.png)


### stack operations

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


---

#### Stack <- list  (!!!!!!!!!!!!!)

consider the performance of the two implementations, there is definitely a difference.

![Screen Shot 2020-05-26 at 14.45.22](https://i.imgur.com/uxkuJEi.png)

```py
# Stack implementation as a list

# the end is at the beginning
class Stack:
    """Stack implementation as a list"""
    def __init__(self): self.items = []
    def is_empty(self): return not bool(self.items)
    def push(self, item): self.items.append(item)
    def pop(self): return self.items.pop()
    # append() and pop() operations were both O(1)
    def peek(self): return self.items[-1]
    def size(self): return len(self.items)

# the top is at the beginning
class Stack:
    def __init__(self): self.items = []
    def isEmpty(self): return self.items == []
    def push(self, item): self.items.insert(0,item)
    def pop(self): return self.items.pop(0)
    # will both require O(n) for a stack of size n.
    def peek(self): return self.items[0]
    def size(self): return len(self.items)

from pythonds3.basic import Stack
s = Stack()
```


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

#### Stack <- Linked List

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


#### reverse char in string

```py
from test import testEqual
from pythonds.basic.stack import Stack

def rev_string(my_str):
    s = Stack()
    revstr=''
    for char in my_str:
        s.push(char)
    while not s.is_empty():
        revstr += s.pop()
    return revstr

print(rev_string('apple'))
testEqual(rev_string('x'), 'x')
testEqual(rev_string('abc'), 'cba')
```

```java
public class Reversestring {
    public String reverseString(String s){
        String result = "";
        for(int i = 0; i < s.length(); i++){
            result = s.charAt(i) + result;
        }
        return result;
    }
}
```
---

#### simple Balanced Parentheses

[code](https://github.com/ocholuo/ocholuo.github.io/blob/master/_posts/05CodeNote/0.Leetcode/leepy/020.valid-paretheses.py)

![simpleparcheck](https://i.imgur.com/2xB79Lt.png)

---

#### Balanced Symbols (A General Case)

```py
from pythonds.basic import Stack

def parChecker(symbolString):
    s = Stack()
    balanced = True
    index = 0
    while index < len(symbolString) and balanced:
        symbol = symbolString[index]
        if symbol in "([{":
            s.push(symbol)
        else:
            if s.isEmpty():
                balanced = False
            else:
                top = s.pop()
                if not matches(top,symbol):
                       balanced = False
        index = index + 1
    if balanced and s.isEmpty():
        return True
    else:
        return False

def matches(open,close):
    opens = "([{"
    closers = ")]}"
    return opens.index(open) == closers.index(close)
```


---

#### convert-integer-into-different-base


![dectobin](https://i.imgur.com/sdKnZuW.png)

```py
# The decimal number 233(10) and its corresponding binary equivalent 11101001(2) are interpreted respectively as
# 2×10^2+3×10^1+3×10^0
# and
# 1×2^7+1×2^6+1×25+0×24+1×23+0×22+0×21+1×20

# convert-integer-into-base(16)
# Class solution1: stack, put num%16 in stack
def divide_by_base(num, base):
    digits = "0123456789ABCDEF"
    s = Stack()
    while num > 0:
        rem = num % base
        s.push(rem)
        num = num // base
    ans_string = ""
    while not s.is_empty():
        ans_string += str(digits[s.pop()])
    return ans_string

print(divide_by_base(25,2))
print(divide_by_base(25,16))
print(divide_by_base(25,8))
print(divide_by_base(256,16))
print(divide_by_base(26,26))


# convert-integer-into-binary
# Class solution1: stack, put num%2 in stack
def divide_by_2(decimal_num):
    s = Stack()
    while decimal_num > 0:
        rem = decimal_num % 2
        s.push(rem)
        decimal_num = decimal_num // 2
    bin_string = ""
    while not s.is_empty():
        bin_string += str(s.pop())
    return bin_string
# print(divide_by_2(42))
# print(divide_by_2(31))
```

---

#### Infix, Prefix, and Postfix Expressions


> computers need to know exactly what operators to perform and in what order.

- **infix**:
  - 中缀`(A+B)*C`
  - the operator is in between the two operands that it is working on.
- **Prefix**:
  - 前缀`(*+ABC)`
  - all operators precede the two operands that they work on.
- **Postfix**:
  - 后缀`(AB+C*)`
  - operators come after the corresponding operands.

![Screen Shot 2020-05-26 at 16.02.12](https://i.imgur.com/YlduKeO.png)


`fully parenthesized expression`: uses one pair of parentheses for each operator.
- `A + B * C + D` -> `((A + (B * C)) + D)`
- `A + B + C + D` -> `(((A + B) + C) + D)`


1. Conversion of Infix to Prefix and Postfix

![Screen Shot 2020-05-26 at 16.05.36](https://i.imgur.com/XHTifZt.png)

2. Infix-to-Postfix Conversion

[code](https://github.com/ocholuo/language/tree/master/0.code/leecode/Algorithms/convert-infix-prefix-postfix.py)

![intopost](https://i.imgur.com/avxcc7z.png)

```py
# use stack
# (A+B+D)*C -> (AB+D+)C*
# A*B+C*D -> AB*+CD*
def infixToPostfix(infixexpr):
    # Assume the infix expression is a string of tokens delimited by spaces.
    # The operator tokens are *, /, +, and -, along with the left and right parentheses, ( and ).
    # The operand tokens are the single-tokenacter identifiers A, B, C, and so on.
    # The following steps will produce a string of tokens in postfix order.
    prec = {'*':3, '/':3, '+':2, '-':2, '(':1}
    operand_tokens = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

    # Create an empty stack called opStack for keeping operators. Create an empty list for output.
    opStack = Stack()
    postfixList = []
    # Convert the input infix string to a list by using the string method split.
    token_list =infixexpr.split()

    # Scan the token list from left to right.
    for token in token_list:
    # If the token is an operand, append it to the end of the output list.
        if token in operand_tokens:
            postfixList.append(token)
            print("postfixList.append:", token,  "add operand")
    # If the token is a left parenthesis, push it on the opStack.
        elif token == '(':
            opStack.push(token)
            print("opStack.push:", token)
    # If the token is a right parenthesis,
    #   pop the opStack until the corresponding left parenthesis is removed.
    #   Append each operator to the end of the output list.
        elif token == ')':
            topToken = opStack.pop()
            print("opStack.pop:", token)
            while topToken != '(':
                postfixList.append(topToken)
                print("postfixList.append:", token, postfixList)
                topToken = opStack.pop()

    # If the token is an operator, *, /, +, or -, push it on the opStack.
    #   However, first remove any operators already on the opStack that have higher or equal precedence and append them to the output list.
        else:
            while (not opStack.isEmpty()) and (prec[opStack.peek()] >= prec[token]):
                postfixList.append(opStack.pop())
                print("postfixList.append:", token, postfixList)
            opStack.push(token)
            print("opStack.push:", token)
    # When the input expression has been completely processed, check the opStack. Any operators still on the stack can be removed and appended to the end of the output list.
    while not opStack.isEmpty():
        out = opStack.pop()
        postfixList.append(out)
        print("postfixList.append:", out, postfixList)
    return " ".join(postfixList)
print(infixToPostfix("( A + B ) * C"))
print(infixToPostfix("A * B + C * D"))
print(infixToPostfix("( A + B ) * C - ( D - E ) * ( F + G )"))
```

3. Postfix Evaluation

[code](https://github.com/ocholuo/language/tree/master/0.code/leecode/Algorithms/convert-infix-prefix-postfix.py)

![evalpostfix1](https://i.imgur.com/11IM0tC.png)

![evalpostfix2](https://i.imgur.com/BuVv3tc.png)

```py
# use stack -> calculate Postfix
def postfixEval(postfixExpr):
    # Create an empty stack called operandStack.
    operandStack = Stack()
    # Convert the string to a list by using the string method split.
    token_list = postfixExpr.split()
    # Scan the token list from left to right.
    for token in token_list:
    # If the token is an operand, convert it from a string to an integer and push the value onto the operandStack.
        if token in "0123456789":
            operandStack.push(int(token))
    # If the token is an operator, *, /, +, or -, it will need two operands. Pop the operandStack twice.
    # The first pop is the second operand and the second pop is the first operand.
    # Perform the arithmetic operation.
    # Push the result back on the operandStack.
        else:
            second_ope = operandStack.pop()
            first_ope = operandStack.pop()
            result = doMath(token, first_ope, second_ope)
            print(first_ope, token, second_ope, result)
            operandStack.push(result)
    # When the input expression has been completely processed, the result is on the stack. Pop the operandStack and return the value.
    return operandStack.pop()

def doMath(op, op1, op2):
    if op == "*": return op1 * op2
    elif op == "/": return op1 / op2
    elif op == "+": return op1 + op2
    else: return op1 - op2

# print(postfixEval('7 8 + 3 2 + /'))
# print(postfixEval('17 10 + 3 * 9 /'))
```


---


## Queue

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

#### queue as a list  (!!!!!!!!!!!!!)

```py
class Queue:
    def __init__(self): self.items = []
    def is_empty(self): return not bool(self.items)
    def enqueue(self, item): self.items.insert(0, item)
    def dequeue(self): return self.items.pop()
    def size(self): return len(self.items)
```

#### queue in java


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



#### Simulation: Hot Potato

[code](https://github.com/ocholuo/language/tree/master/0.code/leecode/Algorithms/queue-hot-potato.py)

![hotpotato](https://i.imgur.com/VB33sdN.png)

![namequeue](https://i.imgur.com/k57sicw.png)

> arranged themselves in a circle. One man was designated as number one, and proceeding clockwise they killed every seventh man.

```py
from pythonds.basic import Queue

def hotPotato(namelist, num):
    simqueue = Queue()
    for name in namelist:
        simqueue.enqueue(name)

    while simqueue.size() > 1:
        for i in range(num):
            simqueue.enqueue(simqueue.dequeue())
        simqueue.dequeue()
    return simqueue.dequeue()

print(hotPotato(["Bill","David","Susan","Jane","Kent","Brad"],7))
```

---

#### Simulation: Printing Tasks

![simulationsetup](https://i.imgur.com/2RoMcuf.png)

- students send printing tasks to the shared printer,
- the tasks are placed in a queue to be processed in a first-come first-served manner.
  - Many questions arise with this configuration. The most important of these might be whether the printer is capable of handling a certain amount of work. If it cannot, students will be waiting too long for printing and may miss their next class.

- On any average day about 10 students are working in the lab at any given hour.
- These students typically print up to twice during that time, and the length of these tasks ranges from 1 to 20 pages.
- The printer in the lab is older, capable of processing
  - `10 pages per minute` of draft quality.  
  - `five pages per minute` of better quality.
- The slower printing speed could make students wait too long. What page rate should be used?

building a simulation that models the laboratory.
- construct representations for students, printing tasks, and the printer (Figure 4).
- As students submit printing tasks, we will add them to a waiting list, a queue of print tasks attached to the printer.
- When the printer completes a task, it will look at the queue to see if there are any remaining tasks to process.
- Of interest for us is the `average amount of time students will wait` for their papers to be printed. This is equal to the `average amount of time a task waits in the queue`.

use some probabilities. For example,
- students may print a paper from 1 to 20 pages in length.
  - If each length from 1 to 20 is equally likely, the actual length for a print task can be simulated by using a random number between 1 and 20 inclusive.
  - This means that there is equal chance of any length from 1 to 20 appearing.

If there are 10 students in the lab and each prints twice, then there are 20 print tasks per hour on average.
- What is the chance that at any given second, a print task is going to be created? The way to answer this is to consider the ratio of tasks to time.
- Twenty tasks per hour means that on average there will be one task every 180 seconds:
- For every second we can simulate the chance that a print task occurs by generating a random number between 1 and 180 inclusive.
- If the number is 180, we say a task has been created.
- Note that it is possible that many tasks could be created in a row or we may wait quite a while for a task to appear. That is the nature of simulation. You want to simulate the real situation as closely as possible given that you know general parameters.



```py
# Create a queue of print tasks. Each task will be given a timestamp upon its arrival. The queue is empty to start.
# For each second (currentSecond):
# Does a new print task get created? If so, add it to the queue with the currentSecond as the timestamp.
# If the printer is not busy and if a task is waiting,
# Remove the next task from the print queue and assign it to the printer.
# Subtract the timestamp from the currentSecond to compute the waiting time for that task.
# Append the waiting time for that task to a list for later processing.
# Based on the number of pages in the print task, figure out how much time will be required.
# The printer now does one second of printing if necessary. It also subtracts one second from the time required for that task.
# If the task has been completed, in other words the time required has reached zero, the printer is no longer busy.
# After the simulation is complete, compute the average waiting time from the list of waiting times generated.


# create classes for the three real-world objects described above: Printer, Task, and PrintQueue.
class Printer:
    def __init__(self, ppm):
        self.pagerate = ppm
        self.currentTask = None
        self.timeRemaining = 0

    def tick(self):
        if self.currentTask != None:
            self.timeRemaining = self.timeRemaining - 1
            if self.timeRemaining <= 0:
                self.currentTask = None

    def busy(self):
        if self.currentTask != None: return True
        else: return False

    def startNext(self, newtask):
        self.currentTask = newtask
        self.timeRemaining = newtask.getPages() * 60/self.pagerate
        # the amount of time needed can be computed from the number of pages in the task.

import random

class Task:
    def __init__(self,time):
        self.timestamp = time
        # Each task need to keep a timestamp to compute waiting time.
        # This timestamp will represent the time that the task was created and placed in the printer queue.

        self.pages = random.randrange(1,21)

    def getStamp(self): return self.timestamp

    def getPages(self): return self.pages

    # The waitTime method can then be used to
    # retrieve the amount of time spent in the queue before printing begins.
    def waitTime(self, currenttime): return currenttime - self.timestamp



from pythonds.basic.queue Queue

import random

def newPrintTask():
    num = random.randrange(1,181)
    if num == 180: return True
    else: return False
# newPrintTask, decides whether a new printing task has been created.
# return a random integer between 1 and 180.
# Print tasks arrive once every 180 seconds. By arbitrarily choosing 180 from the range of random integers (line 32), we can simulate this random event.


# The simulation function
# set the total time and the pages per minute for the printer.

def simulation(numSeconds, pagesPerMinute):
    labprinter = Printer(pagesPerMinute)
    printQueue = Queue()
    waitingtimes = []

    for currentSecond in range(numSeconds):

      if newPrintTask():
         task = Task(currentSecond)
         printQueue.enqueue(task)

      if (not labprinter.busy()) and (not printQueue.isEmpty()):
        nexttask = printQueue.dequeue()
        waitingtimes.append(nexttask.waitTime(currentSecond))
        labprinter.startNext(nexttask)

      labprinter.tick()

    averageWait=sum(waitingtimes)/len(waitingtimes)
    print("Average Wait %6.2f secs %3d tasks remaining."%(averageWait,printQueue.size()))


for i in range(10):
    simulation(3600,5)


```



---

## Deque


- double-ended queue
- an ordered collection of items similar to the queue.
- It has two ends,
  - a front and a rear,
  - and the items remain positioned in the collection.

What makes a deque different is the unrestrictive nature of adding and removing items.
- New items can be added at either the front or the rear.
- Likewise, existing items can be removed from either end.
- provides the capabilities of stacks and queues in a single data structure.
- 可以模拟stack或者queue

it does not require the LIFO and FIFO orderings that are enforced by those data structures.
- It is up to you to make consistent use of the addition and removal operations.

![basicdeque](https://i.imgur.com/5dSSfi2.png)


### Deque - Abstract Data Type

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

---

### code

#### dequeue as a list in py (!!!!!!!!!!!!!)

```py
class Deque:
    def __init__(self): self.items = []
    def is_empty(self): return not bool(self.items)
    def add_front(self, item): self.items.append(item)
    def add_rear(self, item): self.items.insert(0, item)
    def remove_front(self): return self.items.pop()
    def remove_rear(self): return self.items.pop(0)
    def size(self): return len(self.items)
```


#### Palindrome-Checker 回文 对称的单词

![palindromesetup](https://i.imgur.com/BiBLdwq.png)


```py
def pal_checker(input_string):
    char_deque = Deque()

    for ch in input_string:
        char_deque.add_rear(ch)

    while char_deque.size() > 1:
        first = char_deque.remove_front()
        last = char_deque.remove_rear()
        if first != last: return False
    return True

palchecker("lsdkjfhd")
palchecker("radar")    
```

---

## Lists

- a collection of items
- each item holds a relative position with respect to the others.
- More specifically, we will refer to this type of list as an unordered list.
- [54, 26, 93, 17, 77, 31].

---

## Unordered List - Abstract Data Type

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

### Unordered List: Linked Lists


![idea2](https://i.imgur.com/SqXvGO8.png)


无序表： `unordered list`
- 一种数据按照相对位置存放的数据集
- (for easy, assum that no repeat)
- 无序存放，但是在数据相之间建立`链接指向`, 就可以保持其前后相对位置。
  - 显示标记 `head` `end`
- 每个节点 `node` 包含2信息：
  - 数据本身，指向下一个节点的引用信息`next`
  - `next=None` 没有下一个节点了

---


### Node Class

the constructor that a node is initially created with next set to `None`.
- sometimes referred to as “grounding the node,”
- use the standard ground symbol to denote a reference that is referring to `None`

![node](https://i.imgur.com/CK40mon.png)

![node2](https://i.imgur.com/b0X4X3K.png)


---

#### Node Class <- unordered linked list  (!!!!!!!!!!!!!)

```py
class Node:
    def __init__(self, node_data):
        self._data = node_data
        self._next = None
    def get_data(self): return self._data
    def set_data(self, node_data): self._data = node_data
    def get_next(self): return self._next
    def set_next(self, node_next): self._next = node_next
    def __str__(self): return str(self._data)

# create Node objects in the usual way.
>>> temp = Node(93)
>>> temp.data
93
```

---

### Unordered List Class <- unordered linked list (old)

A linked list
- nothing more than a single chain of nodes with a few well defined properties and methods such as:
- Head Pointer:
  - pointer to the origin, or first node in a linked list.
  - Only when the list has a length of 1 will it’s value be None.
- Tail Pointer:
  - pointer to the last node in a list.
  - When a list has a length of 1, the Head and the Tail refer to the same node.
  - By definition, the Tail will have a next value of None.
- Count*:
  - We’ll also be keeping track of the number of nodes we have in our linked list. Though this is not strictly necessary, I find it to be more efficient and convenient than iterating through the entire linked list when polling for size.

![1_73b9zu3H5pjLd8W0RZPeng](https://i.imgur.com/fFWCFCl.jpg)

```py
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

    def is_empty(self): return self.head == None


    # """
    # Add node to start of list
           # (Head) [2] -> [3] -> (Tail)
    # (Head) [1] -> [2] -> [3] -> (Tail)
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

    def size(self): return self.count

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
	def remove_last(self):
		if self.count > 0:
			if self.count == 1:
				self.head = None
				self.tail = None
			else:
				current = self.head
				while current.next != self.tail:
					current = current.next
				current.next = None
				self.tail = current
			self.count -= 1


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


---

### Unordered List Class <- unordered linked list (new)  (!!!!!!!!!!!!!)

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

#### `size`, `search`, and `remove`
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

---

#### `pop()`

```py
def pop(self, index):
    self.remove(self.getItem(index))
```

---

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

---

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


---

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

## Ordered List - Abstract Data Type

ordered list.
- For example, if the list of integers shown above were an ordered list (ascending order), then it could be written as `17, 26, 31, 54, 77, and 93`.
- Since 17 is the smallest item, it occupies the first position in the list.
- Likewise, since 93 is the largest, it occupies the last position.

The structure of an ordered list
- a collection of items where **each item** holds a `relative position that is based upon some underlying characteristic of the item`.
- The ordering is typically either ascending or descending and we assume that list items have a meaningful comparison operation that is already defined.
- Many of the ordered list operations are the same as those of the unordered list.

---

### Ordered List in py (!!!!!!!!!!!!!)

```py
class OrderedList:
    def __init__(self):
        self.head = None
        self.count = 0

    # 𝑂(1)
    def is_empty(self): return self.head == None

    # 𝑂(1)
    def size(self): return self.count

    # 𝑂(n)
    # require the traversal process. Although on average they may need to traverse only half of the nodes, these methods are all 𝑂(𝑛) since in the worst case each will process every node in the list.
    def remove(self, item):
        current = self.head
        previous = None
        # find the item
        while current is not None:
            if current.data == item:
                break
            previous = current
            current = current.next
        # if current == None (tail)
        if current is None:
            raise ValueError("{} is not in the list".format(item))
        # if current is the head
        if previous is None:   # remove the frist item
            self.head = current.next
        else:
            previous.next = current.next
```

```py
    # 𝑂(n)
    # require the traversal process. Although on average they may need to traverse only half of the nodes, these methods are all 𝑂(𝑛) since in the worst case each will process every node in the list.
    def search(self, item):
        current = self.head
        while (current is not None):
            if current.data > item: return False
            if current.data == item: return True
            current = current.next
        return False
```

![orderedsearch](https://i.imgur.com/cXdshUF.png)

```py
    # 𝑂(n)
    # require the traversal process. Although on average they may need to traverse only half of the nodes, these methods are all 𝑂(𝑛) since in the worst case each will process every node in the list.
    def add(self, item):
        temp = Node(item)
        current = self.head
        previous = None
        self.count += 1
        # keep finding
        while (current is not None) and current.data < item:
            previous = current
            current = current.next
        # current.data > item
        # current is head
        if previous is None:
            temp.next = self.head
            self.head = temp
        else:
            temp.next = current
            previous.next = temp
```

![linkedlistinsert](https://i.imgur.com/dZE3tzH.png)

```py
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
