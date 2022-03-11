---
title: Data Structures - Basic 1 - Data Structures
# author: Grace JyL
date: 2021-10-10 11:11:11 -0400
description:
excerpt_separator:
categories: [04CodeNote, DS]
tags:
math: true
# pin: true
toc: true
# image: /assets/img/sample/devices-mockup.png
---

- [Data Structures - Basic 1 - Data Structures](#data-structures---basic-1---data-structures)
- [Linear Structures](#linear-structures)
- [String](#string)
- [StringBuilder](#stringbuilder)
- [Arrays 数组 (fixed size)](#arrays-数组-fixed-size)
    - [Create Array](#create-array)
      - [One-Dimensional Arrays](#one-dimensional-arrays)
      - [Multidimensional Arrays](#multidimensional-arrays)
      - [Instantiating an Array in Java](#instantiating-an-array-in-java)
    - [method](#method)
    - [Arrays of Objects](#arrays-of-objects)
    - [Java Array Error](#java-array-error)
    - [Class Objects for Arrays](#class-objects-for-arrays)
    - [Array Members](#array-members)
    - [Arrays Types, Allowed Element Types](#arrays-types-allowed-element-types)
    - [Cloning of arrays](#cloning-of-arrays)
- [LinkedList (array-based structure) (without fixed size)](#linkedlist-array-based-structure-without-fixed-size)
  - [basicc](#basicc)
  - [Abstract Data Type](#abstract-data-type)
    - [Unordered List - Abstract Data Type](#unordered-list---abstract-data-type)
    - [singly linked list](#singly-linked-list)
    - [Circularly Linked Lists](#circularly-linked-lists)
    - [doubly linked list](#doubly-linked-list)
  - [general method](#general-method)
    - [Equivalence Testing](#equivalence-testing)
    - [Cloning Data Structures](#cloning-data-structures)
  - [Node Class](#node-class)
  - [unordered Linked Lists: Unordered List](#unordered-linked-lists-unordered-list)
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
    - [Abstract Data Type (ADT)](#abstract-data-type-adt)
      - [java](#java)
      - [Python](#python)
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


---

# Data Structures - Basic 1 - Data Structures

source:
- https://runestone.academy/runestone/books/published/pythonds/BasicDS/toctree.html
- [Problem Solving with Algorithms and Data Structures using Python](https://runestone.academy/runestone/books/published/pythonds3/BasicDS/ImplementinganOrderedList.html)
- Data Structures and Algorithms in Java, 6th Edition.pdf




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



# String

**String**

- Because it is common to work with sequences of text characters in programs, Java provides support in the form of a String class.
  - The class provides extensive support for various text-processing tasks

- A **string** instance represents `a sequence of zero or more characters`.


- Java uses double quotes to designate string literals.
    - declare and initialize a String instance as follows: `String title = "Data Structures & Algorithms in Java"`

- Character Indexing
  - Each character within a string can be referenced by using an index

- Concatenation 级联 `P + Q`
  - The primary operation for combining strings is called concatenation,
  - P + Q, which consists of all the characters of P followed by all the characters of Q.
  - concatenation on two strings: `String term = "over" + "load";`



- **immutable**
  - An important trait, String instances are immutable;
    - once an instance is created and initialized, the value of that instance cannot be changed.
    - This is an intentional design, it allows for great efficiencies and optimizations within the Java Virtual Machine.
  - However, as String is a class, a reference type, `variables` of type String can be reassigned to another `string instance` (even if the current string instance cannot be changed)

    ```java
    String greeting = "Hello";
    greeting = "Ciao"; // we changed our mind

    greeting = greeting + '!'; // now it is ”Ciao!”
    ```

    - However, this operation **does create a new string instance**, copying all the characters of the existing string in the process.
    - For long string (such as DNA sequences), this can be very time consuming.




---




# StringBuilder

**StringBuilder**
- to support more efficient editing of character strings
- effectively a mutable version of a string.




---

# Arrays 数组 (fixed size)

- [https://www.geeksforgeeks.org/arrays-in-java/](https://www.geeksforgeeks.org/arrays-in-java/)
- [https://leetcode.com/explore/learn/card/array-and-string/201/introduction-to-array/1143/](https://leetcode.com/explore/learn/card/array-and-string/201/introduction-to-array/1143/)


![Arrays](https://media.geeksforgeeks.org/wp-content/uploads/Arrays1.png)


![Screen Shot 2022-03-02 at 00.02.56](https://i.imgur.com/jxXiikI.png)

- basic data structure

- In Java all arrays are `dynamically allocated`.


- The idea is to `store multiple items of the same type together`.


* Since arrays are objects in Java, we can find their length using the object property _length_. This is different from C/C++, where we find length using sizeof.
* A Java array variable can also be declared like other variables with [] after the data type.
* The variables in the array are ordered, and each has an index beginning from 0.
* Java array can be also be used as a static field, a local variable, or a method parameter.
* The **size** of an array must be specified by int or short value and not long.
* The direct superclass of an array type is [Object](https://www.geeksforgeeks.org/object-class-in-java/).
* Every array type implements the interfaces [Cloneable](https://www.geeksforgeeks.org/marker-interface-java/) and [java.io.Serializable](https://www.geeksforgeeks.org/serialization-in-java/).


An array can contain `primitives (int, char, etc.)` and `object (non-primitive) references of a class` depending on the definition of the array.
- primitive data types: the actual values are stored in contiguous memory locations.
- class objects, [the actual objects are stored in a heap segment](https://www.geeksforgeeks.org/g-fact-46/).  


- to store **a collection of elements sequentially**
  - keep track of an **ordered** sequence of related values or objects.
  - a collection of items stored at **contiguous 连续的 memory locations**


- **element**: Each value stored in an array


- **capacity**: the length of an array
  - the length of an array determines the maximum number of things that can be stored in the array
  - an array has a **fixed capacity**
  - he capacity of the array must be fixed when it is created, specify the size of the array when initialize it.
  - the precise size of array must be internally declared in order for `the system to properly allocate a consecutive piece of memory for its storage`.
    - For example,
      - an array with 12 cells
      - might be stored in memory locations 2146 through 2157 on a computer system.
  - Because **the system may allocate neighboring memory locations to store other data**, the capacity of an array **cannot be increased** by expanding into subsequent cells.
  - serious limitation;
    - it requires that a `fixed maximum capacity be declared, throwing an exception if attempting to add an element once full`.
    - risk: either too large of an array will be requested, inefficient waste of memory,
    - or that too small of an array will be requested, fatal error when exhausting that capacity.


- Array can contains primitives (int, char, etc) as well as object (or non-primitives) references of a class depending on the definition of array.
  - In case of **primitives data types**
    - the actual `values` are stored in `contiguous memory locations`.
  - In case of **objects of a class**
    - the actual `objects` are stored in `heap segment`




**Advantages**

- have **better cache locality**
  - big difference in performance.

- **index**:
  - elements can **be accessed randomly** as each element in the array can be identified by an array **index**.
  - easier to calculate the `position of each element` by simply adding an 抵消 offset to a base value
  - i.e., the memory location of the first element of the array (generally denoted by the name of the array).
  - Each element can be uniquely identified by their `index` in the array.
  - makes **accessing elements by position** faster.

![array-2](https://i.imgur.com/0lvYfk8.png)

- Out of Bounds Errors
  - attempt to index into an array a using a number outside the range.
  - Such a reference is said to be out of bounds.
  - **buffer overflow attack**
    - Out of bounds references have been exploited numerous times by hackers to compromise the security of computer systems written in languages other than Java.
  - As a safety feature, array indices are always checked in Java to see if they are ever out of bounds.
  - If an array index is out of bounds, the runtime Java environment signals an error condition. The name of this condition is the `ArrayIndexOutOfBoundsException`. This check helps Java avoid a number of security problems, such as buffer overflow attacks.


---


### Create Array


#### One-Dimensional Arrays

An array declaration has two components: the type and the name.
- _type_
  - declares the `element type` of the array.
  - determines the data type of each element that comprises the array.
  - determines what type of data the array will hold.
  - Like an array of integers, other primitive data types like char, float, double, etc., or user-defined data types (objects of a class).


```java

type var-name[];
type[] var-name;

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

// an array of references to objects of the class
MyClass myClassArray[];

Object[]  ao,        // array of Object
Collection[] ca;  // array of Collection of unknown type
```


Although the first declaration establishes that intArray is an array variable, **no actual array exists**.
- It merely tells the compiler that this variable (intArray) will hold an array of the integer type.
- To link intArray with an actual, physical array of integers, allocate one using **new** and assign it to intArray.



#### Multidimensional Arrays


- drawbacks.
  - insertions and deletions at interior positions of an array can be time consuming if many elements must be shifted.

  - an array has a **fixed capacity**, The capacity of the array must be fixed when it is created, need to specify the size of the array when initialize it.

**unbounded/dynamic array**
- Therefore, most programming languages offer built-in **dynamic array**
  - still a random access list data structure
  - but with variable size.
  - For example, we have `vector` in C++ and `ArrayList` in Java.

- `Java’s ArrayList class` provides a more robust abstraction, allowing a user to add elements to the list, with no apparent limit on the overall capacity.

- To provide this abstraction, Java relies on an algorithmic sleight of hand that is known as a **dynamic array**.
  -  an array list instance maintains an internal array that often has greater capacity than the current length of the list.
  - If a user continues to add elements to a list, all reserved capacity in the underlying array will eventually be exhausted.
  - In that case, the class requests a new, larger array from the system, and copies all references from the smaller array into the beginning of the new array.
  - At that point in time, the old array is no longer needed, so it can be reclaimed by the system.


Multidimensional arrays are **arrays of arrays** with each element of the array holding the reference of other arrays.
- two-dimensional array is sometimes also called a matrix.
- These are also known as [Jagged Arrays](https://www.geeksforgeeks.org/jagged-array-in-java/).
- A `multidimensional array` is created by appending one set of square brackets ([]) per dimension. Examples:



```java
int[] intArray = new int[10][20]; //a 2D array or matrix
int[] intArray = new int[10][20][10]; //a 3D array
```

![Blank Diagram - Page 1 (13)](https://media.geeksforgeeks.org/wp-content/cdn-uploads/Blank-Diagram-Page-1-13.jpeg)


**amortization** 分期偿还
- an algorithmic design pattern
- amortized analysis,
  - view the computer as a coin-operated appliance that requires the payment of one cyber-dollar for a constant amount of computing time.
  - When an operation is executed, we should have enough cyber-dollars available in our current “bank account” to pay for that operation’s running time.
  - Thus, the total amount of cyber-dollars spent for any computation will be proportional to the total time spent on that computation.
  - The beauty of using this analysis method is that we can overcharge some operations in order to save up cyber-dollars to pay for others.

```java
type var-name[];
type[] var-name;

List<Integer> v0 = new ArrayList<>();
List<Integer> v1;                           // v1 == null

Integer[] a = {0, 1, 2, 3, 4};
v1 = new ArrayList<>(Arrays.asList(a));

// 3. make a copy
List<Integer> v2 = v1;                      // another reference to v1
List<Integer> v3 = new ArrayList<>(v1);     // make an actual copy of v1  


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

**running time**:
- size( ): `O(1)`
- isEmpty( ): `O(1)`
- first(), last(): `O(1)`
- before(p), after(p): `O(1)`
- addFirst(e), addLast(e): `O(1)`
- addBefore(p, e), addAfter(p, e): `O(1)`
- set(p, e): `O(1)`
- remove( p): `O(1)`





---

#### Instantiating an Array in Java

When an array is `declared`, only a **reference** of an array is created.

To create or give memory to the array, you create an array like this: The general form of _new_ as it applies to one-dimensional arrays appears as follows:

- _type_ specifies the type of data being allocated,
- _size_ determines the number of elements in the array,
- _var-name_ is the name of the array variable that is linked to the array.
- use _new_ to allocate an array, **you must specify the type and number of elements to allocate.**

an instance of an array is treated as an object by Java, and variables of an array type are reference variables.

`var-name = new type [size];`
- `type` specifies the type of data being allocated
- `size` specifies the number of elements in the array
- `var-name` is the name of array variable that is linked to the array.
- to use new to allocate an array, must specify the type and number of elements to allocate.

**Instantiating**
- Obtaining an array is a two-step process.
  - First, must declare a variable of the desired array type.  
  - Second, must allocate the memory that will hold the array, using `new`, and assign it to the array variable. Thus, in Java all arrays are dynamically allocated.

- **declaration**
  - declare a variable of the desired array type.
  - establishes the fact that intArray is an array variable,
  - but no array actually exists.
  - When an array is declared, only a reference of array is created.
  - It simply tells to the compiler that `this(intArray) variable will hold an array of the integer type`.

- **allocate**
  - must **allocate** one using `new` and assign it to intArray.
  - allocate the memory to hold the array, using new, and assign it to the array variable. Thus, **in Java**, **all arrays are dynamically allocated.**
  - To link intArray with an actual, physical array of integers
  - The elements in the array allocated by `new` will automatically be initialized to `zero (for numeric types)`, `false (for boolean)`, or `null (for reference types)`.



```java
var-name = new type [size];

int intArray[];    //declaring array
intArray = new int[20];  // allocating memory to array
int[] intArray = new int[20]; // combining both statements in one
int[] intArray = new int[]{ 1,2,3,4,5,6,7,8,9,10 };
```

where the `size` of the array and `variables` of array are already known, array literals can be used.
- The length of this array determines the length of the created array.
- no need to write the new int[] part in the latest versions of Java


---


### method

- Because arrays are so important, Java provides a class, `java.util.Arrays`, with a number of built-in static methods for performing common tasks on arrays.


```java
elementType[] arrayName = {initialValue0, initialValue1, . . . , initialValueN−1};
elementType[] arrayName = new elementType[length]
// When arrays are created using the new operator, all of their elements are automatically assigned the default value for the element type.
// if the element type is numeric, all cells of the array are initialized to zero,
// if the element type is boolean, all cells are false,
// if the element type is a reference type, all cells are initialized to null.

int[] a0 = new int[5];
int[] a1 = {1, 2, 3};
a1.length;
a1[0];
a1[0] = 4;
for (int i = 0; i < a1.length; ++i) System.out.print(" " + a1[i]);
for (int item: a1) System.out.print(" " + item);
Arrays.sort(a1);

Arrays.equals(A, B)
Arrays.fill(A, x)
Arrays.copyOf(A, n)
// Returns an array of size n such that the first k elements of this array are copied from A, where k = min{n, A.length}. If n > A.length, then the last n − A.length elements in this array will be padded with default values, e.g., 0 for an array of int and null for an array of objects.
Arrays.copyOfRange(A, s, t)  // order from A[s] to A[t − 1]
Arrays.toString(A)
Arrays.sort(A)
Arrays.binarySearch(A, x)
```

**running time**
- size( ): `O(1)`
- isEmpty( ): `O(1)`
- get(i): `O(1)`
- set(i, e): `O(1)`
- add(i, e): `O(n)`
- remove(i): `O(n)`

Time
- `O(1)` to add/remove at end (amortized for allocations for more space), index, or update
- `O(n)` to insert/remove elsewhere

Space
- contiguous in memory, so proximity helps performance
- space needed = (array capacity, which is >= n) * size of item,
  - but even if 2n, still O(n)


---

### Arrays of Objects

An array of objects is created like an array of primitive type data items in the following way.

```java
Student[] arr = new Student[7]; //student is a user-defined class
```
The studentArray contains seven memory spaces each of the size of student class in which the address of seven Student objects can be stored. The Student objects have to be instantiated using the constructor of the Student class, and their references should be assigned to the array elements in the following way.


```java
Student[] arr = new Student[5];

// Java program to illustrate creating an array of objects`

class Student {
    public int roll_no;
    public String name;

    Student(int roll_no, String name) {
        this.roll_no = roll_no;
        this.name = name;
    }
}


// Elements of the array are objects of a class Student.`

public class GFG {
    public static void main (String[] args) {
        // declares an Array of integers.
        Student[] arr;

        // allocating memory for 5 objects of type Student.
        arr =new Student[5];

        arr[0] =new Student(1, "aman");
        arr[1] =new Student(2, "vaibhav");
        arr[2] =new Student(3, "shikar");
        arr[3] =new Student(4, "dharmesh");
        arr[4] =new Student(5, "mohit");
    }
}
```

---

### Java Array Error

JVM throws **ArrayIndexOutOfBoundsException** to indicate that the array has been accessed with an illegal index. The index is either negative or greater than or equal to the size of an array.
```java
Runtime error:
Exception in thread "main" java.lang.ArrayIndexOutOfBoundsException: 2 at GFG.main(File.java:12)
```


---

### Class Objects for Arrays

Every array has an associated Class object, shared with all other arrays with the same component type.

```java
class Test {
    public static void main(String args[]) {
        int intArray[] = new int[3];
        byte byteArray[] =new byte[3];
        short shortsArray[] =new short[3];
        String[] strArray =new String[3];
        System.out.println(intArray.getClass());
        System.out.println(intArray.getClass().getSuperclass());  
    }

}
```


**Explanation:**

1. The string “[I” is the run-time type signature for the class object “array with component type _int_.”
2. The only direct superclass of an array type is [java.lang.Object](https://www.geeksforgeeks.org/object-class-in-java/).
3. The string “[B” is the run-time type signature for the class object “array with component type _byte_.”
4. The string “[S” is the run-time type signature for the class object “array with component type _short_.”
5. The string “[L” is the run-time type signature for the class object “array with component type of a Class.” The Class name is then followed.


---

### Array Members

Now, as you know that arrays are objects of a class, and a direct superclass of arrays is a class Object. The members of an array type are all of the following:

* The public final field _length_, which contains the number of components of the array. Length may be positive or zero.
* All the members inherited from class Object; the only method of Object that is not inherited is its [clone](https://www.geeksforgeeks.org/clone-method-in-java-2/) method.
* The public method _clone()_, which overrides the clone method in class Object and throws no [checked exceptions](https://www.geeksforgeeks.org/checked-vs-unchecked-exceptions-in-java/).


---

### Arrays Types, Allowed Element Types

Array Types
- Primitive Type Arrays: Any type which can be implicitly promoted to declared type.
- Object Type Arrays: Either declared type objects or it’s child class objects.
- Abstract Class Type Arrays: Its child-class objects are allowed.
- Interface Type Arrays: Its implementation class objects are allowed.



---


### Cloning of arrays


**single-dimensional array**
clone a single-dimensional array, such as Object[],
- a “deep copy” is performed with the new array containing copies of the original array’s elements as opposed to references.

![Blank Diagram - Page 1 (11)](https://media.geeksforgeeks.org/wp-content/cdn-uploads/Blank-Diagram-Page-1-11.jpeg)


```java
int intArray[] = {1, 2, 3};
int cloneArray[] = intArray.clone();
System.out.println(intArray == cloneArray) // false
```

**multi-dimensional array**
A clone of a multi-dimensional array (like Object[])
- a “shallow copy,”
- it creates only a single new array with each element array a reference to an original element array
- **subarrays are shared**.

![Blank Diagram - Page 1 (12)](https://media.geeksforgeeks.org/wp-content/cdn-uploads/Blank-Diagram-Page-1-12.jpeg)


```java  
int intArray[][] = {4,5};  
int cloneArray[][] = intArray.clone();  
System.out.println(intArray == cloneArray); // false
System.out.println(intArray[0] == cloneArray[0]); // true
```

---




# LinkedList (array-based structure) (without fixed size)

- an alternative to an array-based structure.

- A linked list, in its simplest form, is a collection of nodes that collectively form a linear sequence.

- An important property of a linked list is that `it does not have a predetermined fixed size`;
- it uses space proportional to its current number of elements.

---


## basicc

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

## Abstract Data Type


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


### singly linked list

- In a **singly linked list**,
  - each node stores a reference to an object that is an element of the sequence,
  - as well as a reference to the next node of the list

- `head`
  - Minimally, the linked list instance must keep a reference to the first node of the list
  - Without an `explicit reference` to the head, there would be no way to locate that node (or indirectly, any others).

- `tail`
  - The last node of the list
  - can be found by traversing the linked list—starting at the head and moving from one node to another by following each node’s next reference. **link/pointer hopping**
  - identify the tail as the node having null as its next reference.
  - storing an `explicit reference` to the tail node is a common efficiency to avoid such a traversal. In similar regard, it is common for a linked list instance to keep a count of the total number of nodes that comprise the list (also known as the size of the list), to avoid traversing the list to count the nodes.


![Screen Shot 2022-03-03 at 21.26.04](https://i.imgur.com/t0PStKi.png)


**Inserting an Element at the Head of a Singly Linked List**

```java
Algorithm addFirst(e):
newest=Node(e);
newest.next = head;
head = newest;
size = size + 1;
```

**Inserting an Element at the Tail of a Singly Linked List**


```java
Algorithm addLast(e):
newest=Node(e);
newest.next = null;
tail.next = newest;
tail = newest;
size = size + 1;
```

**Removing an Element from a Singly Linked List**

```java
Algorithm removeFirst():
if head == null:
    the list is empty;
head = head.next;
size = size - 1;
```


**other**
- Unfortunately, we cannot easily delete the last node of a singly linked list.
- we must be able to access the node before the last node in order to remove the last node.
- The only way to access this node is to start from the head of the list and search all the way through the list.
- to support such an operation efficiently, we will need to make our list **doubly linked**


---


### Circularly Linked Lists

- there are many applications in which data can be more naturally viewed as having a cyclic order, with well-defined neighboring relationships, but no fixed beginning or end.

- esentially a singularly linked list, the `next reference of the tail node` is set to refer back to the head of the list (rather than null),

![Screen Shot 2022-03-03 at 22.17.09](https://i.imgur.com/4tzqpWi.png)


**Round-Robin Scheduling**
- One of the most important roles of an operating system is in managing the many processes that are currently active on a computer, including the scheduling of those processes on one or more central processing units (CPUs).
- In order to support the responsiveness of an arbitrary number of concurrent processes, most operating systems allow processes to effectively share use of the CPUs, using some form of an algorithm known as `round-robin scheduling`.
  - A process is given a short turn to execute, known as a `time slice`,
  - it is interrupted when the slice ends, even if its job is not yet complete.
  - Each active process is given its own time slice, taking turns in a cyclic order.
  - New processes can be added to the system, and processes that complete their work can be removed.

1. traditional linked list
   1. by repeatedly performing the following steps on linked list L
      1. process p = L.removeFirst( )
      2. Give a time slice to process p
      3. L.addLast(p)
   2. drawbacks: unnecessarily inefficient to repeatedly throw away a node from one end of the list, only to create a new node for the same element when reinserting it, not to mention the various updates that are performed to decrement and increment the list’s size and to unlink and relink nodes.

2. Circularly Linked List
   1. on a circularly linked list C:
      1. Give a time slice to process C.first()
      2. C.rotate()
   2. Implementing the new rotate method is quite trivial.
      1. do not move any nodes or elements
      2. simply advance the tail reference to point to the node that follows it (the implicit head of the list).


---



### doubly linked list

- there are limitations that stem from the asymmetry of a singly linked list.
  - can efficiently insert a node at either end of a singly linked list, and can delete a node at the head of a list,
  - cannot efficiently delete a node at the tail of the list.
  - cannot efficiently delete an arbitrary node from an interior position of the list if only given a reference to that node, because we cannot determine the node that immediately precedes the node to be deleted (yet, that node needs to have its next reference updated).

![Screen Shot 2022-03-04 at 09.56.42](https://i.imgur.com/dzUHpQI.png)

**doubly linked list**
- a linked list, each node keeps an explicit reference to the node before it and a reference to the node after it.
- These lists allow a greater variety of O(1)-time update operations, including insertions and deletions at arbitrary positions within the list.
- We continue to use the term “next” for the reference to the node that follows another, and we introduce the term “prev” for the reference to the node that precedes it.


**Header and Trailer Sentinels**
- to avoid some special cases when operating near the boundaries of a doubly linked list, it helps to add special nodes at both ends of the list: a `header` node at the beginning of the list, and a `trailer` node at the end of the list.
- These “dummy” nodes are known as `sentinels/guards`, and they do not store elements of the primary sequence.
- When using sentinel nodes, an empty list is initialized so that the `next field of the header points to the trailer`, and the `prev field of the trailer points to the header`; the remaining fields of the sentinels are irrelevant (presumably null, in Java).
- For a nonempty list, the header’s next will refer to a node containing the first real element of a sequence, just as the trailer’s prev references the node containing the last element of a sequence.


**Advantage of Using Sentinels**
- Although we could implement a doubly linked list without sentinel nodes, slight extra memory devoted to the `sentinels greatly simplifies the logic of the operations`.
  - the header and trailer nodes never change — only the nodes between them change.
  - treat all insertions in a unified manner, because a new node will always be placed between a pair of existing nodes.
  - every element that is to be deleted is guaranteed to be stored in a node that has neighbors on each side.
- contrast
  - SinglyLinkedList implementation addLast method required a conditional to manage the special case of inserting into an empty list.
  - In the general case, the new node was linked after the existing tail.
  - But when adding to an empty list, there is no existing tail; instead it is necessary to reassign head to reference the new node.
  - The use of a sentinel node in that implementation would eliminate the special case, as there would always be an existing node (possibly the header) before a new node.


## general method


### Equivalence Testing
- At the lowest level, if a and b are reference variables, then` expression a == b tests whether a and b refer to the same object` (or if both are set to the null value).
- higher-level notion of two variables being considered “equivalent” even if they do not actually refer to the same instance of the class. For example, we typically want to consider two String instances to be equivalent to each other if they represent the identical sequence of characters.
- To support a broader notion of equivalence, all object types support a method named equals.
- The author of each class has a responsibility to provide an implementation of the equals method, which overrides the one inherited from Object, if there is a more relevant definition for the equivalence of two instances

- Great care must be taken when overriding the notion of equality, as the consistency of Java’s libraries depends upon the **equals method defining** what is known as an **equivalence relation** in mathematics, satisfying the following properties:
  - `Treatment of null`:
    - For any nonnull reference variable x,  `x.equals(null) == false` (nothing equals null except null).
  - `Reflexivity`:
    - For any nonnull reference variablex, `x.equals(x) == true` (object should equal itself).
  - `Symmetry`:
    - For any nonnull reference variablesxandy, `x.equals(y) == y.equals(x)`, should return the same value.
  - `Transitivity`:
    - For any nonnull reference variables x, y, and z, if `x.equals(y) == y.equals(z) == true`, then `x.equals(z) == true` as well.



- Equivalence Testing with Arrays
  - a == b:
    - Tests if a and b refer to the same underlying array instance.
  - a.equals(b):
    - identical to a == b. Arrays are not a true class type and do not override the Object.equals method.
  - Arrays.equals(a,b):
    - This provides a more intuitive notion of equivalence, **returning true if the arrays have the same length and all pairs of corresponding elements are “equal” to each other**.
    - More specifically, if the array elements are primitives, then it uses the standard == to compare values.
    - If elements of the arrays are a reference type, then it makes pairwise `comparisons a[k].equals(b[k])` in evaluating the equivalence.

- compound objects
  - two-dimensional arrays in Java are really one-dimensional arrays nested inside a common one-dimensional array raises an interesting issue with respect to how we think about compound objects
  - two-dimensional array, b, that has the same entries as a
    - But the one-dimensional arrays, **the rows of a and b are stored in different memory locations**, even though they have the same internal content.
    - Therefore
      - `java.util.Arrays.equals(a,b) == false`
      - `Arrays.deepEquals(a,b) == true`

---

### Cloning Data Structures

- **abstraction** allows for a data structure to be treated as a single object, even though the encapsulated implementation of the structure might rely on a more complex combination of many objects.
- each class in Java is responsible for defining whether its instances can be copied, and if so, precisely how the copy is constructed.

- The universal `Object superclass` defines a method named `clone`
  - can be used to produce shallow copy of an object.
  - This uses the standard assignment semantics to assign the value of `each field of the new object` equal to the `corresponding field of the existing object` that is being copied.
  - The reason this is known as a shallow copy is because if the field is a reference type, then an initialization of the form `duplicate.field = original.field` causes the field of the new object to refer to the same underlying instance as the field of the original object.

- A `shallow copy` is not always appropriate for all classes
  - therefore, Java intentionally **disables use of the clone() method** by
    - declaring it as protected,
    - having it throw a CloneNotSupportedException when called.
  - The author of a class must explicitly declare support for cloning by
    - formally declaring that the class implements the `Cloneable interface`,
    - and by declaring a public version of the clone() method.
  - That public method can simply call the protected one to do the field-by-field assignment that results in a shallow copy, if appropriate. However, for many classes, the class may choose to implement a deeper version of cloning, in which some of the referenced objects are themselves cloned.


![Screen Shot 2022-03-04 at 11.13.02](https://i.imgur.com/5l3YSL1.png)

![Screen Shot 2022-03-04 at 11.13.41](https://i.imgur.com/gUZfkkP.png)


```java
int[ ] data = {2, 3, 5, 7, 11, 13, 17, 19};
int[ ] backup;

backup = data; // warning; not a copy
backup = data.clone();  // copy
```


**shallow copy**
- considerations when copying an array that stores `reference types` rather than `primitive types`.
  - The `clone()` method produces a shallow copy of the array
  - producing a new array whose cells refer to the same objects referenced by the first array.

![Screen Shot 2022-03-04 at 11.16.26](https://i.imgur.com/jzdkcuy.png)

**deep copy**
- A **deep copy** of the contact list can be created by iteratively cloning the individual elements, as follows, but only if the Person class is declared as Cloneable.

```java
Person[ ] guests = new Person[contacts.length];
for (int k=0; k < contacts.length; k++)
    guests[k] = (Person) contacts[k].clone(); // returns Object type
```

**clone on 2D Arrrays**
- two-dimensional array is really a one-dimensional array storing other one-dimensional arrays, the same distinction between a shallow and deep copy exists.
- Unfortunately, the java.util.Arrays class does not provide any “deepClone” method.

```java
// A method for creating a deep copy of a two-dimensional array of integers.
public static int[][] deepClone(int[][] original){
    int[][] backup = new int[original.length][];
    for(int k=0;k<original.length;k++){
        backup[k] = original[k].clone();
    }
    return backup;
}
```


**Cloning Linked Lists**
- to making a class cloneable in Java
  - declaring that it `implements the Cloneable interface`.
  - implementing a `public version of the clone() method` of the class
  - By convention, that method should begin by creating a new instance using a call to `super.clone()`, which in our case invokes the method from the Object class

> While the assignment of the size variable is correct, we cannot allow the new list to share the same head value (unless it is null).
> For a nonempty list to have an independent state, it must have an entirely new chain of nodes, each storing a reference to the corresponding element from the original list.
> We therefore create a new head node, and then perform a walk through the remainder of the original list while creating and linking new nodes for the new list.


---




## Node Class

the constructor that a node is initially created with next set to `None`.
- sometimes referred to as “grounding the node,”
- use the standard ground symbol to denote a reference that is referring to `None`

![node](https://i.imgur.com/CK40mon.png)

![node2](https://i.imgur.com/b0X4X3K.png)

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

## unordered Linked Lists: Unordered List


![idea2](https://i.imgur.com/SqXvGO8.png)


无序表： `unordered list`
- 一种数据按照相对位置存放的数据集
- (for easy, assum that no repeat)
- 无序存放，但是在数据相之间建立`链接指向`, 就可以保持其前后相对位置。
  - 显示标记 `head` `end`
- 每个节点 `node` 包含2信息：
  - 数据本身，指向下一个节点的引用信息`next`
  - `next=None` 没有下一个节点了


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
  - We also be keeping track of the number of nodes we have in our linked list. Though this is not strictly necessary, I find it to be more efficient and convenient than iterating through the entire linked list when polling for size.


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




## Ordered List - Abstract Data Type


ordered list
- a collection of items where **each item** holds a `relative position that is based upon some underlying characteristic of the item`.

- The ordering is typically either ascending or descending and we assume that list items have a meaningful comparison operation that is already defined.

- Many of the ordered list operations are the same as those of the unordered list.


- For example
  - the list of integers were an ordered list (ascending order),
  - then it could be written as `17, 26, 31, 54, 77, and 93`.
  - Since 17 is the smallest item, it occupies the first position in the list.
  - Likewise, since 93 is the largest, it occupies the last position.


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

![orderedsearch](https://i.imgur.com/cXdshUF.png)

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

![linkedlistinsert](https://i.imgur.com/dZE3tzH.png)

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









# Stack

- a collection of elements, with two principle operations:
  - `push`, which adds to the collection,
  - `pop`, which removes the most recently added element
- “push-down stack”
- an `ordered collection of items`
- the addition and the removal always takes place at the same end.
  - This end is commonly referred to as the “top” and “base”

Stack is a `linear data structure`
- Abstract Data Type
- follows a particular order in which the operations are performed.
- LIFO(Last In First Out), FILO(First In Last Out).


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



![Screen Shot 2020-05-26 at 14.28.21](https://i.imgur.com/viZ9E8J.png)

![Screen Shot 2020-05-26 at 14.29.19](https://i.imgur.com/rTL2FM6.png)

---

### Abstract Data Type (ADT)

---

#### java

```java
push(e)     // O(1)
pop()       // O(1)
top()       // O(1)
size()      // O(1)
isEmpty()   // O(1)
```

**Time Complexity**
- Access: O(n)
- Search: O(n)
- Insert: O(1)
- Remove: O(1)

**space usage**
- Performance of a stack realized by an array. 
- The space usage is O(N),


- In order to formalize abstraction of a stack, define its `application programming interface (API)` in the form of a Java `interface`, which describes the names of the methods that the ADT supports and how they are to be declared and used.


- rely on Java’s **generics framework**, allowing the elements stored in the stack to belong to any object `type <E>`. The formal type parameter is used as the parameter type for the push method, and the return type for both pop and top.

- Because of the importance of the stack ADT, Java has included a concrete class named `java.util.Stack` that implements the LIFO semantics of a stack.
  - However, Java’s Stack class remains only for historic reasons, and its interface is not consistent with most other data structures in the Java library.
  - the current documentation for the Stack class recommends that it not be used, as LIFO functionality (and more) is provided by a more general data strucure **double-ended queue**  


![Screen Shot 2022-03-11 at 00.40.59](https://i.imgur.com/YA3vGtX.png)


```java
public interface Stack<E> {
    int size();
    boolean isEmpty();
    viod push(E e);
    E top();
    E pop();
}

public class ArrayStack<E> implements Stack<E> {

    public static final int CAPACITY = 1000;
    private E[] data;
    private int t = -1;

    public ArrayStack() { this(CAPACITY); }
    public ArrayStack(int capacity) {
        data = (E[]) new Object[capacity];
    }

    public int size() {return (t+1);}
    public boolean isEmpty() {return t==-1;}

    public void push(E e) throws IllegalStateException {
        if(size()==data.length) throw new IllegalStateException("Stack is full");
        data[++t] = e;
    }

    public E top(){
        return isEmpty()? null: data[t];
    }

    public E pop(){
        if(isEmpty()) return null;
        E ans = data[t];
        data[t] = null;
        t--;
        return E;
    }

}
```

**Drawback of This Array-Based Stack Implementation**
- one negative aspect
- it relies on a fixed-capacity array, which limits the ultimate size of the stack.

- where a user has a good estimate on the number of items needing to go in the stack, the array-based implementation is hard to beat.


two approaches for implementing a stack without such a size limitation and with space always proportional to the actual number of elements stored in the stack. 
- One approach, singly linked list for storage;
- more advanced array-based approach that overcomes the limit of a fixed capacity.






----

#### Python

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
