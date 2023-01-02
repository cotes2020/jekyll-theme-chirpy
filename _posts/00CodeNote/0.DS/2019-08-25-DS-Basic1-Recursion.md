---
title: Data Structures - Basic 1 - Recursion
# author: Grace JyL
date: 2019-08-25 11:11:11 -0400
description:
excerpt_separator:
categories: [00CodeNote, DS]
tags:
math: true
# pin: true
toc: true
# image: /assets/img/sample/devices-mockup.png
---

- [Data Structures - Basic 1 - Recursion](#data-structures---basic-1---recursion)
  - [basic](#basic)
  - [The 3 Recursion Laws](#the-3-recursion-laws)
  - [Analyzing Recursive Algorithms](#analyzing-recursive-algorithms)
  - [Examples of Recursion](#examples-of-recursion)
    - [linear recursion](#linear-recursion)
      - [Summing the Elements of an Array Recursively](#summing-the-elements-of-an-array-recursively)
      - [Reversing a Sequence with Recursion](#reversing-a-sequence-with-recursion)
      - [Recursive Algorithms for Computing Powers](#recursive-algorithms-for-computing-powers)
      - [Binary Search](#binary-search)
      - [The Factorial Function](#the-factorial-function)
      - [fibonacci](#fibonacci)
    - [Binary Recursion](#binary-recursion)
      - [Drawing an English Ruler ????????????](#drawing-an-english-ruler-)
      - [summing the n integers of an array](#summing-the-n-integers-of-an-array)
    - [Multiple Recursion](#multiple-recursion)
      - [File Systems Computing Disk Space Usage](#file-systems-computing-disk-space-usage)
      - [Recursion Trace](#recursion-trace)
      - [summation puzzles](#summation-puzzles)
  - [examples of the use of recursion](#examples-of-the-use-of-recursion)
    - [Calculating the Sum of a List of Numbers](#calculating-the-sum-of-a-list-of-numbers)
    - [returns reverse string](#returns-reverse-string)
    - [check palindrome string](#check-palindrome-string)
    - [Int to Str in Any Base](#int-to-str-in-any-base)
    - [Int to Str in Any Base <- Stack + Recursion](#int-to-str-in-any-base---stack--recursion)
  - [Visualizing Recursion](#visualizing-recursion)
    - [turtle graphics basics.](#turtle-graphics-basics)
    - [fractal tree.](#fractal-tree)
    - [Sierpinski Triangle](#sierpinski-triangle)
  - [Complex Recursive Problems](#complex-recursive-problems)
    - [Tower of Hanoi](#tower-of-hanoi)
    - [Exploring a Maze 迷宫](#exploring-a-maze-迷宫)
  - [Dynamic Programming](#dynamic-programming)
    - [making change using the fewest coins](#making-change-using-the-fewest-coins)
      - [recursive solution.](#recursive-solution)
      - [memoization/caching](#memoizationcaching)
      - [dynamic programming](#dynamic-programming-1)


---

# Data Structures - Basic 1 - Recursion

source:
- DS - pythonds3 - 3. Analysis
- Problem Solving with Algorithms and Data Structures using Python 3
- Data Structures and Algorithms in Java, 6th Edition.pdf
- [youtube - CS Dojo](https://www.youtube.com/watch?v=vYquumk4nWw)


---

## basic

**Recursion**

- **A recursive function is a function that calls itself**.
  - Usually recursion involves a function calling itself.

- recursion provides an elegant and powerful alternative for performing repetitive tasks.
  - method of solving problems that involves breaking a problem down into smaller and smaller subproblems until you get to a small enough problem that it can be solved trivially.
  - While it may not seem like much on the surface, recursion allows us to write elegant solutions to problems that may otherwise be very difficult to program.

> Each time we make a recursive call we are solving a smaller problem, until we reach the point where the problem cannot get any smaller.


**A truly dynamic programming algorithm will take a more `systematic 系统的 approach` to the problem**
- build from the bottom to top

find:
- **recursive definition**
- **base case**: refer to fixed values of the function.
- **recursive case**: define the function in terms of itself.
- **recursion trace**:
  - illustrate the execution of a recursive method
  - mirrors a programming language’s execution of the recursion.
- **activation record/frame**
  - In Java, each time a method (recursive or otherwise) is called, a structure known as an `activation record or activation frame is created to store information`
    - about the progress of that invocation of the method.
    - stores the parameters and local variables specific to a given call of the method,
    - and information about which command in the body of the method is currently executing.
  - When the execution of a method leads to a nested method call
    - the execution of the former call is suspended
    - its frame stores the place in the source code at which the flow of control should continue upon return of the nested call.
    - A new frame is then created for the nested method call.
    - This process is used both in the standard case of one method calling a different method, or in the recursive case where a method invokes itself.
    - The key point is to **have a separate frame for each active call**.




**Parameterizing a Recursion**

- A successful recursive design sometimes requires that we redefine the origi- nal problem to facilitate similar-looking subproblems.
- `binarySearch(data, target)` -> `binarySearch(data, target, low, high)`
- using the additional parameters to demarcate subarrays as the recursion proceeds.


- If we wish to provide a cleaner public interface to an algorithm without exposing the user to the recursive parameterization, a standard technique is to **make the recursive version private**, and to introduce a cleaner public method (that calls the private one with appropriate parameters).

```java
// /∗∗ Returns true if the target value is found in the data array. ∗/
public static boolean binarySearch(int[ ] data, int target) {
  return binarySearch(data, target, 0, data.length − 1); // use parameterized version
}
```





---


## The 3 Recursion Laws

all recursive algorithms must obey three important laws:

- A recursive algorithm **must have a `base case`**.
  - a base case is the condition that allows the algorithm to stop recursing.
  - A base case is typically a problem that is small enough to solve directly.


- A recursive algorithm **must `change its state` and `move toward the base case`**.
  - A change of state means that some data that the algorithm is using is modified.
  - Usually `the data that represents problem gets smaller` in some way.


A recursive algorithm **must `call itself, recursively`**.
- recursion
  - add the result from a smaller problem
- store memoize
  - store the smaller problem result
  - result = func(n) + func(n-1)
- botton-up
  - use a list or array
  - result = list(n) + list(n-1)


---


## Analyzing Recursive Algorithms

- With a recursive algorithm, we will account for each operation that is performed based upon the particular activation of the method that manages the flow of control at the time it is executed.

**amortization**
- get a tighter bound on a series of operations by considering the cumulative effect, rather than assuming that each achieves a worst case。



---


## Examples of Recursion

- a recursive call starts at most one other: linear recursion.
- a recursive call may start two others: binary recursion.
- a recursive call may start three or more others: multiple recursion.



**Tail Recursion**
- the usefulness of recursion comes at a modest cost.
- Java Virtual Machine must maintain frames that keep track of the state of each nested call. When computer memory is at a premium, it can be beneficial to derive nonrecursive implementations of recursive algorithms.

- In general, we use the stack data structure to convert a recursive algorithm into a nonrecursive algorithm by managing the nesting of the recursive structure ourselves, rather than relying on the interpreter to do so.

- some forms of recursion can be eliminated without any use of auxiliary memory. One such form is known as `tail recursion`.
- A recursion is a tail recursion if any recursive call that is made from one context is the very last operation in that context, with the return value of the recursive call (if any) immediately returned by the enclosing recursion.
- By necessity, a tail recursion must be a linear recursion (since there is no way to make a second recursive call if you must immediately return the result of the first).

```java
return binarySearch(data, target, mid+1, high);
return n ∗ factorial(n−1);
```

```java
// /∗∗ Returns true if the target value is found in the data array. ∗/
public static boolean binarySearchIterative(int[ ] data, int target) {
  int low = 0;
  int high = data.length − 1;
  while (low <= high) {
    int mid = (low + high) / 2;
    if (target == data[mid]) return true;
    else if (target < data[mid]) high = mid − 1;
    else low = mid + 1;
  }
  return false;
}

// Reverses the contents of the given array. ∗/
public static void reverseIterative(int[ ] data) {
  int low = 0, high = data.length − 1;
  while (low < high) {
    int temp = data[low];
    data[low++] = data[high];
    data[high−−] = temp;
  }
}
```




---

### linear recursion

**linear recursion**
- a recursive method is designed so that `each invocation of the body` makes at most one new recursive call.

- A consequence of the definition of linear recursion is that any recursion trace will appear as a single sequence of calls

- linear recursion terminology reflects **the structure of the recursion trace**, not the asymptotic analysis of the **running time**; for example, we have seen that binary search runs in O(log n) time.


---


#### Summing the Elements of an Array Recursively

```java
// /∗∗ Returns the sum of the first n integers of the given array. ∗/
public static int linearSum(int[ ] data, int n) {
  if (n == 0) return 0;
  else return linearSum(data, n−1) + data[n−1];
}
```

---


#### Reversing a Sequence with Recursion

- reversing the n elements of an array, so that the first element becomes the last, the second element becomes second to the last, and so on.

```java
// Reverses the contents of subarray data[low] through data[high] inclusive.
public static void reverseArray(int[ ] data, int low, int high) {
    if(low<high){
        int temp = data[low];
        data[low] = data[high];
        data[high] = temp;
        reverseArray(data, low+1, high-1);
    }
}
```

> the algorithm is guaranteed to terminate after 1+ n/2 recursive calls,
> each call involves a constant amount of work, the entire process runs in O(n) time.



---

#### Recursive Algorithms for Computing Powers

- raising a number x to an arbitrary nonnegative integer n.


1. trivial recursive definition

x^n = x · x^(n−1) for n > 0.
- runs in O(n) time.

```java
public static double power(double x, int n) {
  if (n == 0) return 1;
  else return x * power(x, n−1);
}
```

1. 2nd

`2*2 *2*2* 2*2* 2*2` 2^8
= `(2*2 *2*2) * (2*2* 2*2)`
= (2^4)^2

k = n/2 -> (x^k)^2


```java
public static double power(double x, int n) {
  if(n == 0) return 1;
  double base = power(x,n/2);
  double res = base*base;
  if(n%2==1) res*x;
  return res;
}
```

> time:
> the number of times divide n by two before getting to one or less is O(log n).
> Therefore, power results in O(logn) recursive calls.
> so the total number of operations for computing power(x,n) is O(logn).
> a significant improvement over the original O(n) time algorithm.


> memory usage.
> The first version has a recursive depth of O(n), O(n) frames are simultaneously stored in memory.
> Because the recursive depth of the improved version is O(log n), its memory usage is O(log n) as well.


---


#### Binary Search

> linear recursion
> The code for binary search includes a case analysis, with two branches that lead to a further recursive call,
> but only one branch is followed during a particular execution of the body.


**sorted order**
- Values stored in sorted order within an array.
- The numbers at top are the indices.


**unsorted**
- the standard approach to search for a target value is to use a `loop to examine every element`, until either finding the target or exhausting the data set.
- This algorithm is known as **linear/sequential search**
- runs in O(n) time (i.e., linear time) since every element is inspected in the worst case.


**sorted and indexable**
- a more efficient algorithm.
- If we consider an arbitrary element of the sequence with value v
  - all elements prior to that in the sequence have values less than or equal to v,
  - all elements after that element in the sequence have values greater than or equal to v.
- This observation allows us to quickly “home in” on a search target using a variant of the children’s game “high-low.”
- We call an element of the sequence a candidate if, at the current stage of the search, we cannot rule out that this item matches the target.
- The algorithm maintains two parameters, low and high, such that all the candidate elements have index at least low and at most high.
- Initially, low = 0 and high = n − 1. We then compare the target value to the median candidate, that is, the element with index mid = ⌊(low + high)/2⌋ .


**binary search**
- a classic recursive algorithm
- to efficiently locate a target value within a sorted sequence of n elements stored in an array.

> a constant number of primitive operations are executed during each recursive call of the binary search method. Hence, the running time is proportional to the number of recursive calls performed.

> Initially, the number of candidates is n;
> after the first call in a binary search, it is at most n/2;
> after the second call, it is at most n/4; and so on.
> In general, after the jth call in a binary search, the number of candidate elements remaining is at most n/2^j .
> In the worst case (an unsuccessful search), the recursive calls stop when there are no more candidate elements.
> the maximum number of recursive calls performed, is the smallest integer r such that
> n/2^j < 1
> n < 2^j
>


```java
public static boolean binarySearch(int[] data, int target, int low, int high) {
    if(low>high) return false;
    int mid = (low + high)/2;
    if(data[mid]==target) return true;
    else if(data[mid]>target) return binarySearch(data, target, low, mid-1);
    else return binarySearch(data, target, mid+1, high);
}

```

---

#### The Factorial Function

- 5! = 5 · 4 · 3 · 2 · 1 = 120.
- The factorial function is important because it is known to equal the `number of ways in which n distinct items can be arranged` into a sequence
- the number of permutations of n items.

> the overall number of operations for computing factorial(n) is O(n)
> as there are n + 1 activations, each of which accounts for O(1) operations.

```java
// 5! = 5 · 4 · 3 · 2 · 1
// 2! = 2 · 1
// 1! = 1
public static int factorial(int n) throws IllegalArgumentException {
    if(n<0) throw new IllegalArgumentException();
    else if(n==0) return 1;
    else return factorial(n-1) * n;
}
```


---

#### fibonacci


```java
public static long[ ] fibonacciGood(int n) {
  if (n <= 1) {
    long[ ] answer = {n, 0};
    return answer;
  }
  else {
    long[ ] temp = fibonacciGood(n − 1);
    long[ ] answer = {temp[0] + temp[1], temp[0]};
    return answer;
  }
}
```



---


### Binary Recursion

- When a method makes two recursive calls

---



#### Drawing an English Ruler ????????????

- For each inch, we place a tick with a numeric label.
- We denote the length of the tick designating a whole inch as the major tick length.
- Between the marks for whole inches, the ruler contains a series of minor ticks, placed at intervals of 1/2 inch, 1/4 inch, and so on.
- As the size of the interval decreases by half, the tick length decreases by one.


- The English ruler pattern is a simple example of a fractal, that is, a shape that has a self-recursive structure at various levels of magnification.

![Screen Shot 2022-03-05 at 14.39.50](https://i.imgur.com/ztQl7AA.png)


```java
---- 0
-
--
-
---
-
--
-
---- 1
-
--
-
----
-
--
-
--- 2
```


---

#### summing the n integers of an array

Computing the sum of one or zero values is trivial. With two or more values, we can recursively compute the sum of the first half, and the sum of the second half, and add those sums together. Our implementation of such an algorithm, in Code Fragment 5.10, is initially invoked as binarySum(data, 0, n−1).


```java
// /∗∗ Returns the sum of the first n integers of the given array. ∗/
public static int binarySum(int[ ] data, int low, int high) {
  if(low>high) return 0;
  if(data[low] == data[high]) return data[low];
  int mid = (high + low) /2
  else return binarySum(data, low, mid) + binarySum(data, mid+1, high);
}
```

> The size of the range is divided in half at each recursive call, and so the depth of the recursion is 1 + log2 n.

> Therefore, binarySum uses O(log n) amount of additional space, which is a big improvement over the O(n) space used by the linearSum method

> However, the running time of binarySum is O(n), as there are 2n − 1 method calls, each requiring constant time.

---

### Multiple Recursion

- process in which a method may make more than two recursive calls.




---

#### File Systems Computing Disk Space Usage

- Modern operating systems define file-system directories in a recursive way.
- Given the recursive nature of the file-system representation, it should not come as a surprise that many common behaviors of an operating system:
  - copying a directory or deleting a directory, are implemented with recursive algorithms.
  - computing the total disk usage for all files and directories nested within a particular directory.
    - We differentiate between the `immediate` disk space used by each entry and the `cumulative` disk space used by that entry and all nested features.


Algorithm DiskUsage(path):
- Input: A string designating a path to a file-system entry
- Output: The cumulative disk space used by that entry and any nested entries total = size( path) {immediate disk space used by the entry}
  - if path represents a directory then
  - for each child entry stored within directory path do
  - total = total + DiskUsage( child) {recursive call}
  - return total

**java.io.File**
- To implement a recursive algorithm for computing disk usage in Java, we rely on the `java.io.File` class.
- An instance of this class represents an abstract pathname in the operating system and allows for properties of that operating system entry to be queried.
  - `new File(pathString) or new File(parentFile, childString)`
    - A new File instance can be constructed either by providing the full path as a string, or by providing an existing File instance that represents a directory and a string that designates the name of a child entry within that directory.
  - `file.length()`
    - Returns the **immediate disk usagE** (measured in bytes) for the operating system entry represented by the File instance (e.g., /user/rt/courses).
  - `file.isDirectory()`
    - Returns true if the File instance represents a directory;
    - false otherwise.
  - `file.list()`
    - Returns an array of strings designating the names of all entries within the given directory.
      - call this method on the File associated with path `/user/rt/courses/cs016`,
      - it returns an array with contents: {"grades","homeworks","programs"}.

> the number of recursive calls made during one invocation was equal to the number of entries within a given directory of the file system.

> there is a constant number of steps reflected in root.length() to compute the disk usage directly at that entry,
> when the entry is a directory, the body of the diskUsage method includes a for loop that iterates over all entries that are contained within that directory. In the worst case, it is possible that one entry includes n − 1 others.
> conclude that there are O(n) recursive calls, each of which runs in O(n) time, leading to an overall running time that is O(n^2)

> We therefore conclude that there are O(n) recursive calls, each of which uses O(1) time outside the loop, and that the overall number of operations due to the loop is O(n). Summing all of these bounds, the overall number of operations is O(n).


```java
public static long diskUsage(File root) {
    long disk_usage = root.length();
    if(root.isDirectory()) {
        for(String file: root.list()) {
            File child = new File(root, file);
            disk_usage += diskUsage(child);
        }
    }
    System.out.println(disk_usage + "\t" + root);
    return disk_usage;
}
```



---


#### Recursion Trace

a classic Unix/Linux utility named du (for “disk usage”).
- It reports the amount of disk space used by a directory and all contents nested within, and can produce a verbose report,

```bash
8 /user/rt/courses/cs016/grades
3 /user/rt/courses/cs016/homeworks/hw1
2 /user/rt/courses/cs016/homeworks/hw2
4 /user/rt/courses/cs016/homeworks/hw3
10 /user/rt/courses/cs016/homeworks
57 /user/rt/courses/cs016/programs/pr1
97 /user/rt/courses/cs016/programs/pr2
74 /user/rt/courses/cs016/programs/pr3
229 /user/rt/courses/cs016/programs
249 /user/rt/courses/cs016
26 /user/rt/courses/cs252/projects/papers/buylow 55 /user/rt/courses/cs252/projects/papers/sellhigh 82 /user/rt/courses/cs252/projects/papers
4786 /user/rt/courses/cs252/projects/demos/market
4787 /user/rt/courses/cs252/projects/demos
4870 /user/rt/courses/cs252/projects 3 /user/rt/courses/cs252/grades 4874 /user/rt/courses/cs252
5124 /user/rt/courses/
```

---

#### summation puzzles


pot + pan = bib
dog + cat = pig
boy + girl = baby


- If the number of possible configurations is not too large, however, we can use a computer to simply enumerate all the possibilities and test each one
- Such an algorithm can use multiple recursion to work through the configurations in a systematic way.


- To keep the description general enough to be used with other puzzles, we consider an algorithm that enumerates and tests all k-length sequences, without repetitions, chosen from a given universe U.
  - Recursively generating the sequences of k − 1 elements
  - Appending to each such sequence an element not already contained in it.
  - Throughout the execution of the algorithm, we use a set U to keep track of the elements not contained in the current sequence, so that an element e has not been used yet if and only if e is inU.

- enumerates every possible size-k ordered subset of U, and tests each subset for being a possible solution to our puzzle.
  - For summation puzzles, U = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} and each position in the sequence corresponds to a given letter


Recursion trace for an execution of PuzzleSolve(3, S, U)
- S is empty and U = {a, b, c}.
- This execution generates and **tests all permutations of a, b, and c**.


Algorithm PuzzleSolve(k, S, U):
- Input: An integer k, sequence S, and set U
- Output: An enumeration of all k-length extensions to S using elements in U without repetitions
- for eache in U do
  - Add e to the end of S
  - Remove e from U            {e is now being used}
  - if k == 1 then
    - Test whether S is a configuration that solves the puzzle
    - if S solves the puzzle then
      - add S to output               {a solution}
  - else
    - PuzzleSolve(k − 1, S, U )       {a recursive call}
  - Remove e from the end of S
  - Add e back to U

![Screen Shot 2022-03-10 at 11.24.59](https://i.imgur.com/2zRWa1a.png)


---

## examples of the use of recursion



---

### Calculating the Sum of a List of Numbers

calculate the sum of a list of numbers such as: [1,3,5,7,9]

**An iterative function**:

```py
def listsum(numList):
  theSum = 0
  for i in numList: theSum = theSum + i
  return theSum
print(listsum([1,3,5,7,9]))
```

do not have while loops or for loops. How would you compute the sum of a list of numbers?
- the sum of the list `numList` is the sum of the first element of the list `numList[0]`, and the sum of the numbers in the rest of the list `numList[1:]`


- A recursive algorithm must have a `base case`.
  - the base case is a list of length 1.
- A recursive algorithm must `change its state` and `move toward the base case`.
  - primary data structure is a list,
  - so we must focus state-changing efforts on the list.
  - Since the base case is a list of length 1, a natural progression toward the base case is to shorten the list.
  - call listsum with a shorter list.
- A recursive algorithm must `call itself, recursively`.


```py
def listsum(numList):
  # This check is crucial and is escape clause from the function. The sum of a list of length 1 is trivial; it is just the number in the list.
  if len(numList) == 1:
  return numList[0]
  else:
  return numList[0] + listsum(numList[1:])
print(listsum([1,3,5,7,9]))
```

---

### returns reverse string

```py
def reverse(s):
  # print(s)
  if len(s) <= 1:
    s = s
  elif len(s) <=2:
    s = s[1] + s[0]
  else:
    s = reverse(s[1:]) + s[0]
  # print(s)
  return s

print(reverse("hello")=="olleh")
print(reverse("l")=="l")
print(reverse("follow")=="wollof")
print(reverse("")=="")
```

----


### check palindrome string

takes a string as a parameter and returns True if the string is a palindrome, False otherwise.
- a string is a palindrome if it is spelled the same both forward and backward.
- For example:
  - radar is a palindrome.
- bonus points
  - palindromes can also be phrases,
  - need to remove the spaces and punctuation before checking.
  - `madam i’m adam` is a palindrome.

Other fun palindromes include:
```
kayak
aibohphobia
Live not on evil
Reviled did I live, said I, as evil I did deliver
Go hang a salami; I’m a lasagna hog.
Able was I ere I saw Elba
Kanakanak – a town in Alaska
Wassamassaw – a town in South Dakota
```

```py
def removeWhite(s):
  s = s.replace(" ", "").replace("'","").replace('"','')
  return s

def isPal(s):
  if len(s) <= 1:
    # print(s)
    return True
  if len(s) == 2:
    # print(s)
    return s[0] == s[-1]
  else:
    return isPal(s[0]+s[-1]) and isPal(s[1:-1])

print(isPal("x"))
print(isPal("radar"))
print(isPal("hello"))
print(isPal(""))
print(isPal("hannah"))
print(isPal(removeWhite("madam i'm adam")))
```

---



### Int to Str in Any Base

For example,
- convert the integer 10 to its string representation in decimal as "10",
- or to its string representation in binary as "1010".


three comp1nts:
- **Reduce the original number to a series of single-digit numbers**
- **Convert the single digit-number to a string using a lookup**
  - divide a number by the base we are trying to convert to.
- **Concatenate the single-digit strings together to form the final result**

```py
def toStr(n,base):
  convertString = "0123456789ABCDEF"
  if n < base:
  return convertString[n]
  else:
  return toStr(n//base,base) + convertString[n%base]
print(toStr(1453,16))
```


---


### Int to Str in Any Base <- Stack + Recursion


push the strings onto a stack instead of making the recursive call.

![newcallstack](https://i.imgur.com/kPOeI2V.png)

```py
from pythonds.basic import Stack

rStack = Stack()

def toStr(n,base):
  convertString = "0123456789ABCDEF"
  while n > 0:
    if n < base:
   rStack.push(convertString[n])
    else:
   rStack.push(convertString[n % base])
    n = n // base
  res = ""
  while not rStack.isEmpty():
    res = res + str(rStack.pop())
  return res

print(toStr(1453,16))

```


---

## Visualizing Recursion

using recursion to draw pictures


### turtle graphics basics.
- use the turtle module to draw a spiral recursively.
- importing the turtle module we create a turtle. When the turtle is created it also creates a window for itself to draw in.
- define the `drawSpiral` function.
  - The base case for this simple function is when the length of the line we want to draw, as given by the len parameter, is reduced to zero or less.
  - If the length of the line is longer than zero we instruct the turtle to go forward by len units and then turn right 90 degrees.
  - The recursive step is when we call drawSpiral again with a reduced length. At the end of ActiveCode 1 you will notice that we call the function myWin.exitonclick(), this is a handy little method of the window that puts the turtle into a wait mode until you click inside the window, after which the program cleans up and exits.

```py
import turtle

myTurtle = turtle.Turtle()
myWin = turtle.Screen()

def drawSpiral(myTurtle, lineLen):
  if lineLen > 0:
    myTurtle.forward(lineLen)
    myTurtle.right(90)
    drawSpiral(myTurtle,lineLen-5)

drawSpiral(myTurtle,100)
myWin.exitonclick()
```


### fractal tree.
- Fractals come from a branch of mathematics, and have much in common with recursion.
- -The definition of a fractal is that when you look at it the fractal has the same basic shape no matter how much you magnify it.
- Some examples from nature are the coastlines of continents, snowflakes, mountains, and even trees or shrubs
- fractal is something that looks the same at all different levels of magnification.
- If we translate this to trees and shrubs we might say that even a small twig has the same shape and characteristics as a whole tree. Using this idea we could say that a tree is a trunk, with a smaller tree going off to the right and another smaller tree going off to the left.
- If you think of this definition recursively it means that we will apply the recursive definition of a tree to both of the smaller left and right trees.


```py
import turtle

def tree(branchLen,t):
  if branchLen > 5:
    t.forward(branchLen)
    t.right(20)
    tree(branchLen-15,t)
    t.left(40)
    tree(branchLen-15,t)
    t.right(20)
    t.backward(branchLen)

def main():
  t = turtle.Turtle()
  myWin = turtle.Screen()
  t.left(90)
  t.up()
  t.backward(100)
  t.down()
  t.color("green")
  tree(75,t)
  myWin.exitonclick()
  main()
main()
```


Modify the recursive tree program using 1 or all of the following ideas:
- Modify the thickness of the branches so that as the `branchLen` gets smaller, the line gets thinner
- Modify the color of the branches so that as the `branchLen` gets very short it is colored like a leaf.
- Modify the angle used in turning the turtle so that at each branch point the angle is selected at random in some range. For example choose the angle between 15 and 45 degrees. Play around to see what looks good.
- Modify the branchLen recursively so that instead of always subtracting the same amount you subtract a random amount in some range.

```py
import turtle

def tree(branchLen,t):
  if branchLen > 5:
    t.forward(branchLen)
    t.right(20)
    tree(branchLen-15,t)
    t.left(40)
    tree(branchLen-15,t)
    t.right(20)
    t.backward(branchLen)

def main():
  t = turtle.Turtle()
  myWin = turtle.Screen()
  t.left(90)
  t.up()
  t.backward(100)
  t.down()
  t.color("green")
  tree(75,t)
  myWin.exitonclick()
main()
```

---

### Sierpinski Triangle

![sierpinski](https://i.imgur.com/uiLL6q9.png)

![stCallTree](https://i.imgur.com/N2o6xhE.png)


- the base case is set arbitrarily as the number of times we want to divide the triangle into pieces.
- Sometimes we call this number the “degree” of the fractal.
- Each time we make a recursive call, we subtract 1 from the degree until we reach 0.
- When we reach a degree of 0, we stop making recursive calls.



```py
import turtle

def drawTriangle(points,color,myTurtle):
  myTurtle.fillcolor(color)
  myTurtle.up()
  myTurtle.goto(points[0][0],points[0][1])
  myTurtle.down()
  myTurtle.begin_fill()
  myTurtle.goto(points[1][0],points[1][1])
  myTurtle.goto(points[2][0],points[2][1])
  myTurtle.goto(points[0][0],points[0][1])
  myTurtle.end_fill()

def getMid(p1,p2):
  return ( (p1[0]+p2[0]) / 2, (p1[1] + p2[1]) / 2)

def sierpinski(points,degree,myTurtle):
  colormap = ['blue','red','green','white','yellow',
     'violet','orange']
  drawTriangle(points,colormap[degree],myTurtle)
  if degree > 0:
    sierpinski( [ points[0],
      getMid(points[0], points[1]),
      getMid(points[0], points[2]) ],
       degree-1, myTurtle)
    sierpinski( [ points[1],
      getMid(points[0], points[1]),
      getMid(points[1], points[2])],
       degree-1, myTurtle)
    sierpinski( [ points[2],
      getMid(points[2], points[1]),
      getMid(points[0], points[2])],
       degree-1, myTurtle)

def main():
  myTurtle = turtle.Turtle()
  myWin = turtle.Screen()
  myPoints = [[-100,-50],[0,100],[100,-50]]
  sierpinski(myPoints,3,myTurtle)
  myWin.exitonclick()

main()
```

---

## Complex Recursive Problems

---

### Tower of Hanoi

high-level outline of how to move a tower from the starting pole, to the goal pole, using an intermediate pole:
- Move a tower of `height-1` to an intermediate pole, using the final pole.
- Move the remaining disk to the final pole.
- Move the tower of `height-1` from the intermediate pole to the final pole using the original pole

As long as we always obey the rule that the larger disks remain on the bottom of the stack, we can use the three steps above recursively, treating any larger disks as though they were not even there.

The only thing missing from the outline above is the identification of a base case. The simplest
- Tower of Hanoi problem is a tower of 1 disk.
- In this case, we need move only a single disk to its final destination.
- A tower of 1 disk will be base case.

![hanoi](https://i.imgur.com/ONnKass.png)

```py
def moveTower(height,fromPole, toPole, withPole):
  if height >= 1:
    moveTower(height-1,fromPole,withPole,toPole)
    moveDisk(fromPole,toPole)
    moveTower(height-1,withPole,toPole,fromPole)

def moveDisk(fp,tp):
  print("moving disk from",fp,"to",tp)

moveTower(3,"A","B","C")
# moving disk from A to B
# moving disk from A to C
# moving disk from B to C
# moving disk from A to B
# moving disk from C to A
# moving disk from C to B
# moving disk from A to B
```


---

### Exploring a Maze 迷宫

![maze](https://i.imgur.com/fA26d8F.png)

assume that maze is divided up into “squares.”
- Each square of the maze is either open or occupied by a section of wall.
- The turtle can only pass through the open squares of the maze.
- If the turtle bumps into a wall it must try a different direction.
- The turtle will require a **systematic procedure** to find its way out of the maze.


Here is the procedure:
- From starting position, try going `North` 1 square and then recursively try procedure from there.
- If `Northern` does not work, take a step to the `South` and recursively repeat procedure.
- If `South` does not work, try a step to the `West` and recursively apply procedure.
- If `North`, `South`, and `West` does not work, then apply the procedure recursively from a position 1 step to `East`.
- If n1 of these directions works then there is no way to get out of the maze and we fail.

If we apply the recursive procedure from there we will just go back 1 step to the North and be in an infinite loop.
- So, we must have a strategy to remember where we have been.
- In this case we will assume that we have a bag of bread crumbs we can drop along our way.
- If we take a step in a certain direction and find that there is a bread crumb already on that square, we know that we should immediately back up and try the next direction in our procedure.
- As we will see when we look at the code for this algorithm, backing up is as simple as returning from a recursive function call.


base cases
- In this algorithm, there are 4 base cases to consider:
- The turtle has `run into a wall`. Since the square is occupied by a wall no further exploration can take place.
- The turtle has `found a square that has already been explored`. We do not want to continue exploring from this position or we will get into a loop.
- We have `found an outside edge`, not occupied by a wall. In other words we have found an exit from the maze.
- We have `explored a square unsuccessfully in all 4 directions`.


use the turtle module to draw and explore our maze
- so we can watch this algorithm in action.
- The `maze` object will provide the following methods for us to use in writing our search algorithm:

- `__init__`
  - Reads in a data file representing a maze,
  - initializes the internal representation of the maze,
  - and finds the starting position for the turtle.
  - text file that represents a maze by using
    - `“+”` characters for walls,
    - spaces for open squares,
    - and the letter `“S”` to indicate the starting position.

```py
[ ['+','+','+','+',...,'+','+','+','+','+','+','+'],
  ['+',' ',' ',' ',...,' ',' ',' ','+',' ',' ',' '],
  ['+',' ','+',' ',...,'+','+',' ','+',' ','+','+'],
  ['+',' ','+',' ',...,' ',' ',' ','+',' ','+','+'],
  ['+','+','+',' ',...,'+','+',' ','+',' ',' ','+'],
  ['+',' ',' ',' ',...,'+','+',' ',' ',' ',' ','+'],
  ['+','+','+','+',...,'+','+','+','+','+',' ','+'],
  ['+',' ',' ',' ',...,'+','+',' ',' ','+',' ','+'],
  ['+',' ','+','+',...,' ',' ','+',' ',' ',' ','+'],
  ['+',' ',' ',' ',...,' ',' ','+',' ','+','+','+'],
  ['+','+','+','+',...,'+','+','+',' ','+','+','+']]
```


- `drawMaze`
  - Draws the maze in a window on the screen.

```py
++++++++++++++++++++++
+   +   ++ ++     +
+ +   +       +++ + ++
+ + +  ++  ++++   + ++
+++ ++++++    +++ +  +
+          ++  ++    +
+++++ ++++++   +++++ +
+     +   +++++++  + +
+ +++++++      S +   +
+                + +++
++++++++++++++++++ +++
```


- `updatePosition`
  - Updates the internal representation of the maze and changes the position of the turtle in the window.
  - uses the same internal representation to see if the turtle has run into a wall.
  - It also updates the internal representation with a `“.”` or `“-”` to indicate that the turtle has visited a particular square or if the square is part of a dead end.
  - In addition, the updatePosition method uses two helper methods, moveTurtle and dropBreadCrumb, to update the view on the screen.


- `isExit`
  - Checks to see if the current position is an exit from the maze.
  - uses the current position of the turtle to test for an exit condition.
  - An exit condition is whenever the turtle has navigated to the edge of the maze, either row zero or column zero, or the far right column or the bottom row.

- The `Maze` class also overloads the index operator `[]` so that our algorithm can easily access the status of any particular square.



```py
import turtle

PART_OF_PATH = 'O'
TRIED = '.'
OBSTACLE = '+'
DEAD_END = '-'

class Maze:
    def __init__(self,mazeFileName):
        rows_in_Maze = 0
        columns_in_Maze = 0
        self.mazelist = []

        mazeFile = open(mazeFileName,'r')
        for line in mazeFile:
            rowList = []
            col = 0
            for ch in line[:-1]:
                rowList.append(ch)
                if ch == 'S':
                    self.startRow = rows_in_Maze
                    self.startCol = col
                col = col + 1
            rows_in_Maze = rows_in_Maze + 1
            self.mazelist.append(rowList)
            columns_in_Maze = len(rowList)

        self.rows_in_Maze = rows_in_Maze
        self.columns_in_Maze = columns_in_Maze
        self.xTranslate = -columns_in_Maze/2
        self.yTranslate = rows_in_Maze/2
        self.t = turtle.Turtle(shape='turtle')
        setup(width=600,height=600)
        self.wn = turtle.Screen()
        self.wn.setworldcoordinates(
          -(columnsInMaze-1)/2-.5,
          -(rowsInMaze-1)/2-.5,
          (columnsInMaze-1)/2+.5,
          (rowsInMaze-1)/2+.5)
        # setworldcoordinates(-(columns_in_Maze-1)/2-.5,
        #                     -(rows_in_Maze-1)/2-.5,
        #                     (columns_in_Maze-1)/2+.5,
        #                     (rows_in_Maze-1)/2+.5)

def drawMaze(self):
    self.t.speed(10)
    self.wn.tracer(0)
    for y in range(self.rows_in_Maze):
        for x in range(self.columns_in_Maze):
            if self.mazelist[y][x] == OBSTACLE:
                self.drawCenteredBox(x+self.xTranslate,
                                     -y+self.yTranslate,
                                     'tan')
    self.t.color('black','blue')
    # self.t.color('black')
    # self.t.fillcolor('blue')
    # self.wn.update()
    # self.wn.tracer(1)

def drawCenteredBox(self,x,y,color):
    tracer(0)
    self.t.up()
    self.t.goto(x-.5,y-.5)
    self.t.color('black',color)
    self.t.setheading(90)
    self.t.down()
    self.t.begin_fill()
    for i in range(4):
      self.t.forward(1)
      self.t.right(90)
    self.t.end_fill()
    update()
    tracer(1)

def moveTurtle(self,x,y):
    self.t.up()
    self.t.setheading(self.t.towards(x+self.xTranslate, -y+self.yTranslate))
    self.t.goto(x+self.xTranslate,-y+self.yTranslate)

def dropBreadcrumb(self,color):
    self.t.dot(color)

def updatePosition(self,row,col,val=N1):
    if val: self.mazelist[row][col] = val
    self.moveTurtle(col,row)
    if val == PART_OF_PATH: color = 'green'
    elif val == OBSTACLE: color = 'red'
    elif val == TRIED: color = 'black'
    elif val == DEAD_END: color = 'red'
    else: color = N1
    if color: self.dropBreadcrumb(color)

def isExit(self,row,col):
     return (row == 0 or
             row == self.rows_in_Maze-1 or
             col == 0 or
             col == self.columns_in_Maze-1 )

def __getitem__(self,idx):
     return self.mazelist[idx]

# function takes three parameters:
# a maze object, the starting row, and the starting column.
# This is important because as a recursive function the search logically starts again with each recursive call.

def searchFrom(maze, startRow, startColumn):
  maze.updatePosition(startRow, startColumn)
  #  Check for base cases:
  #  1. We have run into an obstacle 障碍, return false
  if maze[startRow][startColumn] == OBSTACLE : return False
  #  2. We have found a square that has already been explored
  if maze[startRow][startColumn] == TRIED: return False
  #  3. Success, an outside edge not occupied by an obstacle
  if maze.isExit(startRow,startColumn):
    maze.updatePosition(startRow, startColumn, PART_OF_PATH)
    return True

  maze.updatePosition(startRow, startColumn, TRIED)

  # Otherwise, use logical short circuiting to try each
  # direction in turn (if needed)
  found = searchFrom(maze, startRow-1, startColumn) or \
          searchFrom(maze, startRow+1, startColumn) or \
          searchFrom(maze, startRow, startColumn-1) or \
          searchFrom(maze, startRow, startColumn+1)
  if found:
    maze.updatePosition(startRow, startColumn, PART_OF_PATH)
  else:
    maze.updatePosition(startRow, startColumn, DEAD_END)
  return found
```

---


## Dynamic Programming

Dynamic programming is 1 strategy for optimization problems.

A classic example of an optimization problem involves
- making change using the fewest coins.


### making change using the fewest coins

- giving out the fewest possible coins in change for each transaction.
- Suppose a customer puts in a dollar bill and purchases an item for 37 cents. What is the smallest number of coins you can use to make change?
- The answer is six coins: two quarters, 1 dime, and three pennies.
- start with the largest coin in our arsenal (a quarter) and use as many of those as possible, then we go to the next lowest coin value and use as many of those as possible.
- This first approach is called a `greedy method` because we try to solve as big a piece of the problem as possible right away.


The greedy method works fine when we are using U.S. coins,
- but suppose that your company decides to deploy its vending machines in Lower Elbonia where, in addition to the usual 1, 5, 10, and 25 cent coins they also have a 21 cent coin.
- In this instance our greedy method fails to find the optimal solution for 63 cents in change.
- With the addition of the 21 cent coin the greedy method would still find the solution to be six coins.
- However, the optimal answer is three 21 cent pieces.


#### recursive solution.
- identifying the base case.
- make change for the same amount as the value of 1 of our coins, the answer is easy, 1 coin.
- If the amount does not match we have several options. What we want is
  - the minimum of a penny plus the number of coins needed to make change for the original amount minus a penny,
  - or a nickel plus the number of coins needed to make change for the original amount minus 5 cents,
  - or a dime plus the number of coins needed to make change for the original amount minus ten cents, and so on.

```py
def recMC(coinValueList,change):
  minCoins = change
  # checking our base case
  if change in coinValueList: return 1
  # If we do not have a coin equal to the amount of change, we make recursive calls for each different coin value less than the amount of change we are trying to make.
  else:
    for i in [c for c in coinValueList if c <= change]:
      numCoins = 1 + recMC(coinValueList,change-i)
      if numCoins < minCoins:
        minCoins = numCoins
  return minCoins

print(recMC([1,5,10,25],63))
```


The trouble with the algorithm is that it is extremely inefficient.
- it takes `67,716,925` recursive calls to find the optimal solution to the 4 coins, 63 cents problem!
- To understand the fatal flaw in our approach

![callTree](https://i.imgur.com/uE3AQum.png)

- Each node in the graph corresponds to a call to recMC. The label on the node indicates the amount of change for which we are computing the number of coins.
- The label on the arrow indicates the coin that we just used.
- By following the graph we can see the combination of coins that got us to any point in the graph.
- The main problem is that we are `re-doing too many calculations`.
  - For example, the graph shows that the algorithm would recalculate the optimal number of coins to make change for 15 cents at least three times.
  - Each of these computations to find the optimal number of coins for 15 cents itself takes 52 function calls. Clearly we are wasting a lot of time and effort recalculating old results.


#### memoization/caching
- The key to cutting down on the amount of work: `remember some of the past results`
  - to avoid recomputing results we already know.
- A simple solution:
  - store the results for the minimum number of coins in a table when we find them.
  - Then before we compute a new minimum, we first check the table to see if a result is already known.
  - If there is already a result in the table, we use the value from the table rather than recomputing.

```py
def recDC(coinValueList,change,knownResults):
  minCoins = change
  if change in coinValueList:
     knownResults[change] = 1
     return 1
  elif knownResults[change] > 0:
     return knownResults[change]
    # see if our table contains the minimum number of coins for a certain amount of change.
    # If it does not, we compute the minimum recursively and store the computed minimum in the table.
  else:
      for i in [c for c in coinValueList if c <= change]:
        numCoins = 1 + recDC(coinValueList, change-i, knownResults)
        if numCoins < minCoins:
           minCoins = numCoins
           knownResults[change] = minCoins
  print(knownResults)
  return minCoins
print(recDC([1,5,10,25],63,[0]*64))
```

- look at the knownResults lists, there are some holes in the table.
- this is not dynamic programming but just improve the performance by using “memoization/caching”

```
[0, 1, 0, 0, 0, 1, 2, 0, 0, 0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
[0, 1, 0, 0, 0, 1, 2, 0, 0, 0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
[0, 1, 0, 0, 0, 1, 2, 3, 0, 0, 1, 2, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
[0, 1, 0, 0, 0, 1, 2, 3, 0, 0, 1, 2, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
[0, 1, 0, 0, 0, 1, 2, 3, 0, 0, 1, 2, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
[0, 1, 0, 0, 0, 1, 2, 3, 0, 0, 1, 2, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
[0, 1, 0, 0, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
[0, 1, 0, 0, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
[0, 1, 0, 0, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
[0, 1, 0, 0, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
[0, 1, 0, 0, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
[0, 1, 0, 0, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
[0, 1, 0, 0, 0, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
[0, 1, 0, 0, 0, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
[0, 1, 0, 0, 0, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
[0, 1, 0, 0, 0, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
[0, 1, 0, 0, 0, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
[0, 1, 0, 0, 0, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
[0, 1, 0, 0, 0, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
[0, 1, 0, 0, 0, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
[0, 1, 0, 0, 0, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
[0, 1, 0, 0, 0, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 2, 3, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
[0, 1, 0, 0, 0, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 2, 3, 4, 0, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
[0, 1, 0, 0, 0, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 2, 3, 4, 5, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
[0, 1, 0, 0, 0, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 2, 3, 4, 5, 6, 0, 0, 0, 0, 0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
[0, 1, 0, 0, 0, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 2, 3, 4, 5, 6, 2, 0, 0, 0, 0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
[0, 1, 0, 0, 0, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 2, 3, 4, 5, 6, 2, 3, 0, 0, 0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
[0, 1, 0, 0, 0, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 2, 3, 4, 5, 6, 2, 3, 0, 0, 0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
[0, 1, 0, 0, 0, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 2, 3, 4, 5, 6, 2, 3, 4, 0, 0, 1, 2, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
[0, 1, 0, 0, 0, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 2, 3, 4, 5, 6, 2, 3, 4, 0, 0, 1, 2, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
[0, 1, 0, 0, 0, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 2, 3, 4, 5, 6, 2, 3, 4, 0, 0, 1, 2, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
[0, 1, 0, 0, 0, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 2, 3, 4, 5, 6, 2, 3, 4, 5, 0, 1, 2, 3, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
[0, 1, 0, 0, 0, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 2, 3, 4, 5, 6, 2, 3, 4, 5, 0, 1, 2, 3, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
[0, 1, 0, 0, 0, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 2, 3, 4, 5, 6, 2, 3, 4, 5, 0, 1, 2, 3, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
[0, 1, 0, 0, 0, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 2, 3, 4, 5, 6, 2, 3, 4, 5, 0, 1, 2, 3, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
[0, 1, 0, 0, 0, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 2, 3, 4, 5, 6, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
[0, 1, 0, 0, 0, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 2, 3, 4, 5, 6, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
[0, 1, 0, 0, 0, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 2, 3, 4, 5, 6, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
[0, 1, 0, 0, 0, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 2, 3, 4, 5, 6, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
[0, 1, 0, 0, 0, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 2, 3, 4, 5, 6, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
[0, 1, 0, 0, 0, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 2, 3, 4, 5, 6, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
[0, 1, 0, 0, 0, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 2, 3, 4, 5, 6, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 2, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
[0, 1, 0, 0, 0, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 2, 3, 4, 5, 6, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 2, 3, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
[0, 1, 0, 0, 0, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 2, 3, 4, 5, 6, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 2, 3, 4, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
[0, 1, 0, 0, 0, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 2, 3, 4, 5, 6, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 2, 3, 4, 5, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
[0, 1, 0, 0, 0, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 2, 3, 4, 5, 6, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 2, 3, 4, 5, 6, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
[0, 1, 0, 0, 0, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 2, 3, 4, 5, 6, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 2, 3, 4, 5, 6, 2, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
[0, 1, 0, 0, 0, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 2, 3, 4, 5, 6, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 2, 3, 4, 5, 6, 2, 3, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
[0, 1, 0, 0, 0, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 2, 3, 4, 5, 6, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 2, 3, 4, 5, 6, 2, 3, 4, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
[0, 1, 0, 0, 0, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 2, 3, 4, 5, 6, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 2, 3, 4, 5, 6, 2, 3, 4, 5, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
[0, 1, 0, 0, 0, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 2, 3, 4, 5, 6, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 2, 3, 4, 5, 6, 2, 3, 4, 5, 6, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
[0, 1, 0, 0, 0, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 2, 3, 4, 5, 6, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 2, 3, 4, 5, 6, 2, 3, 4, 5, 6, 3, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
[0, 1, 0, 0, 0, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 2, 3, 4, 5, 6, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 2, 3, 4, 5, 6, 2, 3, 4, 5, 6, 3, 4, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
[0, 1, 0, 0, 0, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 2, 3, 4, 5, 6, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 2, 3, 4, 5, 6, 2, 3, 4, 5, 6, 3, 4, 5, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
[0, 1, 0, 0, 0, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 2, 3, 4, 5, 6, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 2, 3, 4, 5, 6, 2, 3, 4, 5, 6, 3, 4, 5, 6, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
[0, 1, 0, 0, 0, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 2, 3, 4, 5, 6, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 2, 3, 4, 5, 6, 2, 3, 4, 5, 6, 3, 4, 5, 6, 7, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
[0, 1, 0, 0, 0, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 2, 3, 4, 5, 6, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 2, 3, 4, 5, 6, 2, 3, 4, 5, 6, 3, 4, 5, 6, 7, 3, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
[0, 1, 0, 0, 0, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 2, 3, 4, 5, 6, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 2, 3, 4, 5, 6, 2, 3, 4, 5, 6, 3, 4, 5, 6, 7, 3, 4, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
[0, 1, 0, 0, 0, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 2, 3, 4, 5, 6, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 2, 3, 4, 5, 6, 2, 3, 4, 5, 6, 3, 4, 5, 6, 7, 3, 4, 5, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
[0, 1, 0, 0, 0, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 2, 3, 4, 5, 6, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 2, 3, 4, 5, 6, 2, 3, 4, 5, 6, 3, 4, 5, 6, 7, 3, 4, 5, 6, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
[0, 1, 0, 0, 0, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 2, 3, 4, 5, 6, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 2, 3, 4, 5, 6, 2, 3, 4, 5, 6, 3, 4, 5, 6, 7, 3, 4, 5, 6, 7, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
[0, 1, 0, 0, 0, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 2, 3, 4, 5, 6, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 2, 3, 4, 5, 6, 2, 3, 4, 5, 6, 3, 4, 5, 6, 7, 3, 4, 5, 6, 7, 2, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
[0, 1, 0, 0, 0, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 2, 3, 4, 5, 6, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 2, 3, 4, 5, 6, 2, 3, 4, 5, 6, 3, 4, 5, 6, 7, 3, 4, 5, 6, 7, 2, 3, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
[0, 1, 0, 0, 0, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 2, 3, 4, 5, 6, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 2, 3, 4, 5, 6, 2, 3, 4, 5, 6, 3, 4, 5, 6, 7, 3, 4, 5, 6, 7, 2, 3, 4, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
[0, 1, 0, 0, 0, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 2, 3, 4, 5, 6, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 2, 3, 4, 5, 6, 2, 3, 4, 5, 6, 3, 4, 5, 6, 7, 3, 4, 5, 6, 7, 2, 3, 4, 5, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0]
[0, 1, 0, 0, 0, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 2, 3, 4, 5, 6, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 2, 3, 4, 5, 6, 2, 3, 4, 5, 6, 3, 4, 5, 6, 7, 3, 4, 5, 6, 7, 2, 3, 4, 5, 6, 3, 0, 0, 0, 0, 0, 0, 0, 0]
[0, 1, 0, 0, 0, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 2, 3, 4, 5, 6, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 2, 3, 4, 5, 6, 2, 3, 4, 5, 6, 3, 4, 5, 6, 7, 3, 4, 5, 6, 7, 2, 3, 4, 5, 6, 3, 4, 0, 0, 0, 0, 0, 0, 0]
[0, 1, 0, 0, 0, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 2, 3, 4, 5, 6, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 2, 3, 4, 5, 6, 2, 3, 4, 5, 6, 3, 4, 5, 6, 7, 3, 4, 5, 6, 7, 2, 3, 4, 5, 6, 3, 4, 5, 0, 0, 0, 0, 0, 0]
[0, 1, 0, 0, 0, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 2, 3, 4, 5, 6, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 2, 3, 4, 5, 6, 2, 3, 4, 5, 6, 3, 4, 5, 6, 7, 3, 4, 5, 6, 7, 2, 3, 4, 5, 6, 3, 4, 5, 6, 0, 0, 0, 0, 0]
[0, 1, 0, 0, 0, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 2, 3, 4, 5, 6, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 2, 3, 4, 5, 6, 2, 3, 4, 5, 6, 3, 4, 5, 6, 7, 3, 4, 5, 6, 7, 2, 3, 4, 5, 6, 3, 4, 5, 6, 7, 0, 0, 0, 0]
[0, 1, 0, 0, 0, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 2, 3, 4, 5, 6, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 2, 3, 4, 5, 6, 2, 3, 4, 5, 6, 3, 4, 5, 6, 7, 3, 4, 5, 6, 7, 2, 3, 4, 5, 6, 3, 4, 5, 6, 7, 3, 0, 0, 0]
[0, 1, 0, 0, 0, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 2, 3, 4, 5, 6, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 2, 3, 4, 5, 6, 2, 3, 4, 5, 6, 3, 4, 5, 6, 7, 3, 4, 5, 6, 7, 2, 3, 4, 5, 6, 3, 4, 5, 6, 7, 3, 4, 0, 0]
[0, 1, 0, 0, 0, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 2, 3, 4, 5, 6, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 2, 3, 4, 5, 6, 2, 3, 4, 5, 6, 3, 4, 5, 6, 7, 3, 4, 5, 6, 7, 2, 3, 4, 5, 6, 3, 4, 5, 6, 7, 3, 4, 5, 0]
[0, 1, 0, 0, 0, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 2, 3, 4, 5, 6, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 2, 3, 4, 5, 6, 2, 3, 4, 5, 6, 3, 4, 5, 6, 7, 3, 4, 5, 6, 7, 2, 3, 4, 5, 6, 3, 4, 5, 6, 7, 3, 4, 5, 6]
6
```


#### dynamic programming

A truly dynamic programming algorithm will take a more `systematic 系统的 approach` to the problem.

- Our dynamic programming solution is going to
  - start with making change for 1 cent
  - and systematically work its way up to the amount of change we require.
  - This guarantees that at each step we already know the mincoins needed to make change for any smaller amount.


fill in a table of minimum coins for 11 cents.
- start with 1 cent. The only solution possible is 1 coin (a penny).
- The next row shows the minimum for 1 cent and two cents. Again, the only solution is two pennies.
- The fifth row is where things get interesting.
  - Now 2 options, 5 pennies or 1 nickel.
  - How do we decide which is best?
  - table: mincoins for 4 cents is 4, plus 1 more penny to make 5, equals 5 coins.
  - Or zero cents plus 1 more nickel to make 5 cents, equals 1 coin.
  - Since the minimum of 1 and 5 is 1, we store 1 in the table.
- consider 11 cents.
  - three options that we have to consider:
    - A penny plus the minimum number of coins to make change for 11−1=10 cents (1)
    - A nickel plus the minimum number of coins to make change for 11−5=6 cents (2)
    - A dime plus the minimum number of coins to make change for 11−10=1 cents (1)
    - Either option 1 or 3 will give us a total of two coins which is the minimum number of coins for 11 cents.

![changeTable](https://i.imgur.com/1OjlSc7.png)

![elevenCents](https://i.imgur.com/PbEZBA9.png)

```py
# dynamic programming algorithm
# dpMakeChange takes three parameters:
# a list of valid coin values,
# the amount of change we want to make,
# and a list of the minimum number of coins needed to make each value.
# When the function is done, minCoins will contain the solution for all values from 0 to the value of change.

# build the dic from 0 to range
def dpMakeChange(coinValueList,change,minCoins):
   for cents in range(change+1):
      coinCount = cents
      for j in [c for c in coinValueList if c <= cents]:
            if minCoins[cents-j] + 1 < coinCount:
               coinCount = minCoins[cents-j]+1
      minCoins[cents] = coinCount
   return minCoins[change]
```

This `dpMakeChange` is not a recursive function,
- important: `recursive solution` does not mean it is the **best or most efficient solution**.
- The bulk of the work in this function is done by the loop that starts on line 4.
  - using all possible coins to make change for the amount specified by cents.
  - Like we did for the 11 cent example above,
  - we remember the minimum value and store it in our minCoins list.

it does a good job of figuring out the minimum number of coins,
- but it does not help us make change since we do not keep track of the coins we use.

extend `dpMakeChange`
- keep track of the coins used by simply remembering the last coin we add for each entry in the `minCoins` table.
- If we know the last coin added,
  - we can simply subtract the value of the coin to find a previous entry in the table that tells us the last coin we added to make that amount.
- We can keep tracing back through the table until we get to the beginning.


```py
# keep track of the coins used
def dpMakeChange(coinValueList,change,minCoins,coinsUsed):
   for cents in range(change+1):
      coinCount = cents
      newCoin = 0
      for j in [c for c in coinValueList if c <= cents]:
            if minCoins[cents-j] + 1 <= coinCount:
               coinCount = minCoins[cents-j]+1
               newCoin = j
      minCoins[cents] = coinCount
      coinsUsed[cents] = newCoin
   print(minCoins)
   print(coinsUsed)
   return minCoins[change]
# Making change for 63 requires

# minCoins: change for 0, for 1, for 2 ....
# [0, 1, 2, 3, 4, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 2, 3, 4, 5, 6, 2, 1, 2, 3, 4, 1, 2, 3, 4, 5, 2, 2, 3, 4, 5, 2, 3, 4, 5, 6, 3, 3, 2, 3, 4, 3, 2, 3, 4, 5, 2, 3, 3, 4, 5, 3, 3, 4, 5, 6, 3, 4, 4, 3]
# coinsUsed: which coins to add,
# [0, 1, 1, 1, 1, 5, 5, 5, 5, 5, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 21, 21, 21, 21, 25, 25, 25, 25, 25, 25, 21, 21, 21, 21, 25, 25, 25, 25, 25, 25, 21, 21, 21, 21, 25, 25, 25, 25, 25, 25, 25, 21, 21, 21, 25, 25, 25, 25, 25, 25, 25, 21, 21]


# printCoins that walks backward through the table to print out the value of each coin used.
# This shows the algorithm in action solving the problem
def printCoins(coinsUsed,change):
   coin = change
   while coin > 0:
      thisCoin = coinsUsed[coin]
      print(thisCoin)
      coin = coin - thisCoin

def main():
    # the amount to be converted
    amnt = 63
    # create the list of coins used.
    clist = [1,5,10,21,25]
    # create the lists we need to store the results.
    # coinsUsed is a list of the coins used to make change
    coinsUsed = [0]*(amnt+1)
    # coinCount is the minimum number of coins used to make change for the amount corresponding to the position in the list.
    coinCount = [0]*(amnt+1)

    print("Making change for",amnt,"requires")
    print(dpMakeChange(clist,amnt,coinCount,coinsUsed),"coins")
    print("They are:")
    printCoins(coinsUsed,amnt)
    print("The used list is as follows:")
    print(coinsUsed)
```











.
