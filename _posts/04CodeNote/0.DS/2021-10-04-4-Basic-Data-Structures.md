---
title: DS - pythonds3 - 4. Basic Data Structures
# author: Grace JyL
date: 2021-10-04 11:11:11 -0400
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
- [List](#list)
  - [Arrays](#arrays)
    - [Dynamic Array in Java](#dynamic-array-in-java)
      - [Analysis of Dynamic Arrays](#analysis-of-dynamic-arrays)
      - [ADT design Position](#adt-design-position)
    - [Arrays in Java](#arrays-in-java)
      - [declare Array](#declare-array)
        - [Instantiate Array](#instantiate-array)
        - [Array Literal](#array-literal)
      - [Access Array Elements](#access-array-elements)
        - [with for Loop](#with-for-loop)
        - [with foreach loops ???](#with-foreach-loops-)
      - [Arrays of Objects](#arrays-of-objects)
      - [error:](#error)
        - [access element outside the array size](#access-element-outside-the-array-size)
      - [Multidimensional Arrays](#multidimensional-arrays)
      - [Passing Arrays to Methods](#passing-arrays-to-methods)
      - [Returning Arrays from Methods](#returning-arrays-from-methods)
      - [Class Objects for Arrays](#class-objects-for-arrays)
    - [Array Members](#array-members)
    - [Cloning of arrays](#cloning-of-arrays)
  - [Linked List](#linked-list)
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
  - [Hashing](#hashing)
  - [Graph](#graph)


- ref:
  - https://runestone.academy/runestone/books/published/pythonds/BasicDS/toctree.html
  - [Problem Solving with Algorithms and Data Structures using Python](https://runestone.academy/runestone/books/published/pythonds3/BasicDS/ImplementinganOrderedList.html)

---

# DS - pythonds3 - 4. Basic Data Structures

Problem Solving with Algorithms and Data Structures using Python 4

---

# Linear Structures
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


# List

Locations within an array are easily described with an integer `index`.
- an index of an element `e` in a sequence is equal to the number of elements before `e` in that sequence.
- By this definition, the first element of a sequence has index `0`, and the last has index `n−1`,
- `n` denotes the total number of elements.

Functions:
- size(): Returns the number of elements in the list.
- isEmpty(): Returns a boolean indicating whether the list is empty.
- get(i):
  - Returns the element of the list having index i;
  - an error condition occurs if i is not in range [0, size( ) − 1].
- set(i,e):
  - Replaces th eelementat indexi with e, and returns the old element that was replaced;
  - an error condition occurs if i is not in range [0, size( ) − 1].
- add(i, e):
  - Inserts a new element `e` into the list so that it has index `i`,
  - moving all subsequent elements one index later in the list;
  - an error condition occurs if i is not in `range[0,size()]`.
- remove(i):
  - Removes and returns the element at index i,
  - moving all subsequent elements one index earlier in the list;
  - an error condition occurs if i is not in range [0, size( ) − 1].



---

## Arrays

array
- **fixed-capacity array**
  - more advanced technique that effectively allows an array-based list to have `unbounded capacity`.
  - Such an unbounded list is known as an **array list in Java**
- a collection of items stored at **contiguous 连续的 memory locations**
- The idea is to `store multiple items of the same type together`.
- easier to calculate the `position of each element` by simply adding an 抵消 offset to a base value
  - i.e., the memory location of the first element of the array (generally denoted by the name of the array).
- Each element can be uniquely identified by their `index` in the array.

![array-2](https://i.imgur.com/0lvYfk8.png)


**Advantages**
- `allow random access` of elements
  - makes **accessing elements by position** faster.
- have **better cache locality**
  - big difference in performance.
v
```java
// A character array in C/C++/Java
char arr1[] = {'g', 'e', 'e', 'k', 's'};

// An Integer array in C/C++/Java
int arr2[] = {10, 20, 30, 40, 50};
```

> Usually, an array of characters is called a ‘string’,
> whereas an array of ints or floats is called simply an array.


**running time**
- size( ): `O(1)`
- isEmpty( ): `O(1)`
- get(i): `O(1)`
- set(i, e): `O(1)`
- add(i, e): `O(n)`
- remove(i): `O(n)`

---

### Dynamic Array in Java


In reality, elements of an ArrayList are stored in a traditional array
- the precise size of that traditional array must be internally declared in order for `the system to properly allocate a consecutive piece of memory for its storage`.
  - For example,
  - an array with 12 cells
  - might be stored in memory locations 2146 through 2157 on a computer system.
- Because **the system may allocate neighboring memory locations to store other data**, the capacity of an array **cannot be increased** by expanding into subsequent cells.


serious limitation;
- it requires that a `fixed maximum capacity be declared, throwing an exception if attempting to add an element once full`.
- risk: either too large of an array will be requested, inefficient waste of memory,
- or that too small of an array will be requested, fatal error when exhausting that capacity.

**unbounded/dynamic array**
- `Java’s ArrayList class` provides a more robust abstraction, allowing a user to add elements to the list, with no apparent limit on the overall capacity.
- To provide this abstraction, Java relies on an algorithmic sleight of hand that is known as a **dynamic array**.
  - The first key to providing the semantics of an unbounded array is that an array list instance maintains an internal array that often has greater capacity than the current length of the list.
  - If a user continues to add elements to a list, all reserved capacity in the underlying array will eventually be exhausted.
  - In that case, the class requests a new, larger array from the system, and copies all references from the smaller array into the beginning of the new array.
  - At that point in time, the old array is no longer needed, so it can be reclaimed by the system.




#### Analysis of Dynamic Arrays

**amortization** 分期偿还
- an algorithmic design pattern
- amortized analysis,
  - view the computer as a coin-operated appliance that requires the payment of one cyber-dollar for a constant amount of computing time.
  - When an operation is executed, we should have enough cyber-dollars available in our current “bank account” to pay for that operation’s running time.
  - Thus, the total amount of cyber-dollars spent for any computation will be proportional to the total time spent on that computation.
  - The beauty of using this analysis method is that we can overcharge some operations in order to save up cyber-dollars to pay for others.



#### ADT design Position
- first( ): Returns the position of the first element of L (or null if empty).
- last(): Returns the position of the last element of L (or null if empty).
- before(p): Returns the position of L immediately before position p (or null if p is the first position).
- after(p): Returns the position of L immediately after position p (or null if p is the last position)
- isEmpty(): Returns true if list L does not contain any elements.
- size(): Returns the number of elements in list L.


Linked Positional List

running time:
- size( ): `O(1)`
- isEmpty( ): `O(1)`
- first(), last(): `O(1)`
- before(p), after(p): `O(1)`
- addFirst(e), addLast(e): `O(1)`
- addBefore(p, e), addAfter(p, e): `O(1)`
- set(p, e): `O(1)`
- remove( p): `O(1)`

---

### Arrays in Java 


`char arr1[] = {'g', 'e'}`

**array**
- a group of like-typed variables that are referred to by a common name.
- Arrays in Java work differently than they do in C/C++.  

- In Java all arrays are `dynamically allocated`.
- arrays are objects in Java, find length using member `length`.
  - different from C/C++, find length using `sizeof`.
  - The direct superclass of an array type is Object.
  - Every array type implements the interfaces `Cloneable` and `java.io.Serializable`.

- A Java array variable can also be declared like other variables with `[]` after the data type.
- The variables in the array are **ordered** and each have an `index` beginning from 0.
- Java array can be also be used as a static field, a local variable or a method parameter.
- The size of an array must be specified by an int value and not long or short.


Array can contains primitives (int, char, etc) as well as object (or non-primitives) references of a class depending on the definition of array.
- In case of **primitives data types**
  - the actual `values` are stored in `contiguous memory locations`.
- In case of **objects of a class**
  - the actual `objects` are stored in `heap segment`

Time
- `O(1)` to add/remove at end (amortized for allocations for more space), index, or update
- `O(n)` to insert/remove elsewhere

Space
- contiguous in memory, so proximity helps performance
- space needed = (array capacity, which is >= n) * size of item,
  - but even if 2n, still O(n)



---

#### declare Array

One-Dimensional Arrays :
- The general form of a one-dimensional array declaration is

```java
type var-name[];
OR
type[] var-name;
```

An array declaration has two components:
- the type: determines what type of data the array will hold.
- the name.



```java
// both are valid declarations
int intArray[];
int[] intArray;

byte byteArray[];
short shortsArray[];
boolean booleanArray[];
long longArray[];
float floatArray[];
double doubleArray[];
char charArray[];

// an array of references to objects of
// the class MyClass (a class created by
// user)
MyClass myClassArray[];

Object[]  ao,        // array of Object
Collection[] ca;     // array of Collection of unknown type
```


---

##### Instantiate Array

**allocating**
- **declaration**
  - establishes the fact that intArray is an array variable,
  - but no array actually exists.
  - When an array is declared, only a reference of array is created.
  - It simply tells to the compiler that `this(intArray) variable will hold an array of the integer type`.
- **allocate**
  - To link intArray with an actual, physical array of integers
  - must **allocate** one using `new` and assign it to intArray.

`var-name = new type [size];`

- `type` specifies the type of data being allocated
- `size` specifies the number of elements in the array
- `var-name` is the name of array variable that is linked to the array.
- to use new to allocate an array, must specify the type and number of elements to allocate.

```java
int intArray[];          //declaring array
intArray = new int[20];  // allocating memory to array

OR

int[] intArray = new int[20]; // combining both statements in one
```

1. The elements in the array allocated by `new` will automatically be initialized to `zero (for numeric types)`, `false (for boolean)`, or `null (for reference types)`.
Refer Default array values in Java
2. Obtaining an array is a two-step process.
   - First, must declare a variable of the desired array type.  
   - Second, must allocate the memory that will hold the array, using `new`, and assign it to the array variable. Thus, in Java all arrays are dynamically allocated.

---

##### Array Literal

where the `size` of the array and `variables` of array are already known, array literals can be used.

```java
int[] intArray = new int[]{ 1,2,3,4,5,6,7,8,9,10 };
// Declaring array literal
```

- The length of this array determines the length of the created array.
- no need to write the new int[] part in the latest versions of Java


---

#### Access Array Elements

##### with for Loop

Each element in the array is accessed via its index.
- The index begins with 0 and ends at `array.length-1`.
- All the elements of array can be accessed using Java for Loop.

```java
for (int i = 0; i < arr.length; i++){
  System.out.println(arr[i]);
}
```

---

##### with foreach loops ???

---


#### Arrays of Objects

An array of objects is created just like an array of primitive type data items in the following way.

```java
Class[] arr = new Class[7];
```

- The student Array contains seven memory spaces each of size of student class in which the address of seven Student objects can be stored.
- The Student objects have to be instantiated using the constructor of the `Student` class and their references should be assigned to the array elements in the following way.

```java
Student[] arr = new Student[5];

class Student {
    public int roll_no;
    public String name;

    Student(int roll_no, String name) {
        this.roll_no = roll_no;
        this.name = name;
    }
}

// Elements of array are objects of a class Student.
public class GFG {

    public static void main (String[] args) {
        Student[] arr;
        arr = new Student[5];

        // initialize the first elements of the array
        arr[0] = new Student(1,"aman");
        arr[1] = new Student(2,"vaibhav");
        arr[2] = new Student(3,"shikar");
        arr[3] = new Student(4,"dharmesh");
        arr[4] = new Student(5,"mohit");

        // accessing the elements of the specified array
        for (int i = 0; i < arr.length; i++)
            System.out.println("Element at " + i + " : " +
                        arr[i].roll_no +" "+ arr[i].name);
    }
}
```

---

#### error:

##### access element outside the array size

JVM throws `ArrayIndexOutOfBoundsException` to indicate that array has been accessed with an illegal index.
- The index is either negative or greater than or equal to size of array.

```java
Runtime error
Exception in thread "main" java.lang.ArrayIndexOutOfBoundsException: 2
    at GFG.main(File.java:12)
```


---


#### Multidimensional Arrays

**Multidimensional arrays** `Jagged Arrays`
- each element of the array holding the reference of other array.
- A multidimensional array is created by appending one set of square brackets ([]) per dimension.

```java
int[][] intArray = new int[10][20];       //a 2D array or matrix
int[][][] intArray = new int[10][20][10]; //a 3D array

class multiDimensional {
    public static void main(String args[]) {

        // declaring and initializing 2D array
        int arr[][] = { {2,7,9},{3,6,1},{7,4,2} };

        // printing 2D array
        for (int i=0; i< 3 ; i++) {
            for (int j=0; j < 3 ; j++)
                System.out.print(arr[i][j] + " ");
            System.out.println();
        }
    }
}
// Output:
2 7 9
3 6 1
7 4 2
```

---

#### Passing Arrays to Methods

```java
// pass array to method sum for calculating sum of array’s values.

class Test {     
    public static void sum(int[] arr)  {
        int sum = 0;
        for (int i = 0; i < arr.length; i++){
            sum+=arr[i];
        }
        System.out.println("sum of array values : " + sum);
    }
    public static void main(String args[])  {
        int arr[] = {3, 1, 2, 5, 4};
        sum(arr);
    }
}
```

---

#### Returning Arrays from Methods

```java
class Test {     

    // Driver method
    public static void main(String args[])  {
        int arr[] = m1();
        for (int i = 0; i < arr.length; i++)
            System.out.print(arr[i]+" ");
    }
    public static int[] m1()  {
        // returning  array
        return new int[]{1,2,3};
    }
}
// Output:
1 2 3
```

---

#### Class Objects for Arrays

Every array has an associated Class object, shared with all other arrays with the same component type.

```java
class Test
{  
    public static void main(String args[])  
    {
        int intArray[] = new int[3];
        byte byteArray[] = new byte[3];
        short shortsArray[] = new short[3];

        // array of Strings
        String[] strArray = new String[3];

        System.out.println(intArray.getClass());
        System.out.println(intArray.getClass().getSuperclass());
        System.out.println(byteArray.getClass());
        System.out.println(shortsArray.getClass());
        System.out.println(strArray.getClass());
    }
}
// Output:
// class [I
// class java.lang.Object
// class [B
// class [S
// class [Ljava.lang.String;

// Explanation :
// The string “[I” is the run-time type signature for the class object “array with component type int“.
// The only direct superclass of any array type is java.lang.Object.
// The string “[B” is the run-time type signature for the class object “array with component type byte“.
// The string “[S” is the run-time type signature for the class object “array with component type short“.
// The string “[L” is the run-time type signature for the class object “array with component type of a Class”. The Class name is then followed.
```

---


### Array Members

arrays are object of a class and direct superclass of arrays is class Object.

The members of an array type are all of the following:
- The public final field `length`, which contains the number of components of the array. length may be positive or zero.
- All the members inherited from class `Object`;
  - the only method of Object that is not inherited is its `clone` method.
  - The public method `clone()`, which overrides clone method in class Object and throws no checked exceptions.


---

### Cloning of arrays

When you clone a single dimensional array, such as `Object[]`
- a “deep copy” is performed with the new array containing copies of the original array’s elements as opposed to references.

```java
class Test{
    public static void main(String args[])  {
        int intArray[] = {1,2,3};
        int cloneArray[] = intArray.clone();
        // will print false as deep copy is created
        // for one-dimensional array
        System.out.println(intArray == cloneArray);

        for (int i = 0; i < cloneArray.length; i++) {
            System.out.print(cloneArray[i]+" ");
        }
    }
}
// Output:
// false
// 1 2 3
```

A clone of a `multidimensional array` (like `Object[][]`) is a “shallow copy”
- it creates only a single new array with each element array a reference to an original element array but subarrays are shared.

```
class Test{     
    public static void main(String args[])  {
        int intArray[][] = { {1,2,3} , {4,5} };
        int cloneArray[][] = intArray.clone();

        // will print false
        System.out.println(intArray == cloneArray);

        // will print true
        // as shallow copy is created
        // i.e. sub-arrays are shared
        System.out.println(intArray[0] == cloneArray[0]);
        System.out.println(intArray[1] == cloneArray[1]);
    }
}
```





---

## Linked List

- a collection of items
- each item holds a relative position with respect to the others.
- More specifically, we will refer to this type of list as an `unordered list`.
- [54, 26, 93, 17, 77, 31].
- a linear collection of data elements `nodes`, each pointing to the next node by means of a `pointer`.
- It is a data structure consisting of `a group of nodes` which together represent a sequence.

**linked list**
- Singly-linked list:
  - linked list
  - each node points to the next node
  - and the last node points to `null`
- Doubly-linked list:
  - linked list
  - each node has two pointers, `p` and `n`,
  - p points to the previous node
  - n points to the next node;
  - the last node's `n pointer` points to `null`
- Circular-linked list:
  - linked list
  - each node points to the next node
  - and the last node points back to the `first node`

**Time Complexity**
- Access: `O(n)`
- Search: `O(n)`
- Insert: `O(1)`
- Remove: `O(1)`


---

### Unordered List - Abstract Data Type

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

#### Unordered List: Linked Lists


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


#### Node Class

the constructor that a node is initially created with next set to `None`.
- sometimes referred to as “grounding the node,”
- use the standard ground symbol to denote a reference that is referring to `None`

![node](https://i.imgur.com/CK40mon.png)

![node2](https://i.imgur.com/b0X4X3K.png)


---

##### Node Class <- unordered linked list  (!!!!!!!!!!!!!)

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

#### Unordered List Class <- unordered linked list (old)

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

#### Unordered List Class <- unordered linked list (new)  (!!!!!!!!!!!!!)

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

##### `is_empty()`
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

##### `add()`
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

##### `size`, `search`, and `remove`
- all based on a technique known as linked list traversal
- Traversal refers to the process of systematically visiting each node.

###### `size()`
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

###### `search(item):`
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

###### `remove()`
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

##### `pop()`

```py
def pop(self, index):
    self.remove(self.getItem(index))
```

---

##### `append()`

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

##### `insert()`

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

##### `index()`

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

### Ordered List - Abstract Data Type

ordered list.
- For example, if the list of integers shown above were an ordered list (ascending order), then it could be written as `17, 26, 31, 54, 77, and 93`.
- Since 17 is the smallest item, it occupies the first position in the list.
- Likewise, since 93 is the largest, it occupies the last position.

The structure of an ordered list
- a collection of items where **each item** holds a `relative position that is based upon some underlying characteristic of the item`.
- The ordering is typically either ascending or descending and we assume that list items have a meaningful comparison operation that is already defined.
- Many of the ordered list operations are the same as those of the unordered list.

---

#### Ordered List in py (!!!!!!!!!!!!!)

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




---

## Stack

- a collection of elements, with two principle operations:
  - `push`, which adds to the collection,
  - `pop`, which removes the most recently added element
- “push-down stack”
- an `ordered collection of items`
- the addition and the removal always takes place at the same end.
  - This end is commonly referred to as the “top” and “base”

**Time Complexity**
- Access: O(n)
- Search: O(n)
- Insert: O(1)
- Remove: O(1)


<kbd>LIFO, last-in first-out</kbd>

- items stored closer to the `base`, been in the stack the longest.
- The most recently added item, be removed first.
- It provides an ordering `based on length of time` in the collection.
- Newer items are near the top, while older items are near the base.


Stacks are fundamentally important, as they can be used to `reverse the order of items`.
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


- a collection of elements,
- supporting two principle operations:
  - enqueue, which inserts an element into the queue,
  - dequeue, which removes an element from the queue

**Time Complexity**:
- Access: O(n)
- Search: O(n)
- Insert: O(1)
- Remove: O(1)

**Queue**

- FIFO: first in first out.
- only change from 2 side. **no insert!!**
  - front : rear

- used when `things don’t have to be processed immediatly`, but `have to be processed in First In First Out order` like Breadth First Search.
- This property of Queue makes it useful in following kind of scenarios.
  - printing queues,
  - operating systems use different queues to control processes
  - keystrokes are being placed in a queue-like buffer so that they can eventually be displayed on the screen in the proper order.
  - When a resource is shared among multiple consumers.
    - Examples include CPU scheduling, Disk Scheduling.
  - When data is transferred asynchronously (data not necessarily received at same rate as sent) between two processes.
    - Examples include IO Buffers, pipes, file IO, etc.

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

## Hashing

- Hashing is used to map data of an arbitrary size to data of a fixed size.
- The values returned by a hash function are called `hash values, hash codes, or simply hashes`.
- If two keys map to the same value, a `collision` occurs


**Hash Map**
- a hash map is a structure that can `map keys to values`.
- A hash map uses a hash function to compute an index into an array of buckets or slots, from which the desired value can be found.


**Collision Resolution**
- Separate Chaining:
  - each bucket is independent, and contains a list of entries for each index.
  - The time for hash map operations is the time to find the bucket (constant time), plus the time to iterate through the list
- Open Addressing:
  - when a new entry is inserted, the buckets are examined, starting with the hashed-to-slot and proceeding in some sequence, until an unoccupied slot is found.
  - The name open addressing refers to the fact that the location of an item is not always determined by its hash value

![hash](https://i.imgur.com/Rnf96Ip.png)

---



## Graph

- A Graph is an ordered pair of `G = (V, E)`
  - comprising a set `V of vertices or nodes` together with a `set E of edges or arcs`,
  - which are 2-element subsets of V
  - (i.e. an edge is associated with two vertices, and that association takes the form of the unordered pair comprising those two vertices)


- `Undirected Graph`:
  - a graph in which the 邻接关系 adjacency relation is **symmetric**.
  - So if there exists an `edge from node u to node v (u -> v)`,
  - then it is also the case that there exists an `edge from node v to node u (v -> u)`


- `Directed Graph`:
  - a graph in which the 邻接关系 adjacency relation is **not symmetric**.
  - So if there exists an edge from node u to node v (u -> v),
  - this **does not imply** that there exists an edge from node v to node u (v -> u)


![graph](https://i.imgur.com/Z4TDBoj.png)


---





。
