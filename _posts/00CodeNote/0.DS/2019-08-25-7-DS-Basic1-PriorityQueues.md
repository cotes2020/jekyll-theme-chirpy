---
title: Data Structures - Basic 1 - Priority Queues
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


- [Data Structures Basic 1 Priority Queues](#data-structures-basic-1-priority-queues)
  - [Priority Queues](#priority-queues)
    - [ADT: Priority Queue in java](#adt-priority-queue-in-java)
    - [Implementing a Priority Queue](#implementing-a-priority-queue)
      - [Entry **Interface**](#entry-interface)
      - [PriorityQueue **Interface**](#priorityqueue-interface)
      - [defining comparisons](#defining-comparisons)
        - [Comparable **Interface**](#comparable-interface)
        - [Comparator **Interface**](#comparator-interface)
      - [AbstractPriorityQueue **Abstract base class**](#abstractpriorityqueue-abstract-base-class)
      - [UnsortedPriorityQueue **class** Unsorted List](#unsortedpriorityqueue-class-unsorted-list)
      - [SortedPriorityQueue **class** sorted List](#sortedpriorityqueue-class-sorted-list)
      - [import java.util.PriorityQueue](#import-javautilpriorityqueue)
  - [Binary Heap 堆](#binary-heap-堆)
    - [The Heap Data Structure](#the-heap-data-structure)
      - [The Height of a Heap](#the-height-of-a-heap)
  - [Implement Priority Queue](#implement-priority-queue)
    - [Implementing a Priority Queue with a Heap](#implementing-a-priority-queue-with-a-heap)
      - [Complete Binary Tree **Array-Based**](#complete-binary-tree-array-based)
      - [priority queue **heap-based**](#priority-queue-heap-based)
        - [heap in java](#heap-in-java)
          - [analyze the binary heap](#analyze-the-binary-heap)
        - [max heap in java](#max-heap-in-java)
          - [min heap in python](#min-heap-in-python)
    - [Bottom-Up Heap Construction](#bottom-up-heap-construction)
      - [Implementation in Java](#implementation-in-java)
        - [Asymptotic Analysis of Bottom-Up Heap Construction](#asymptotic-analysis-of-bottom-up-heap-construction)
    - [java.util.PriorityQueue **Class**](#javautilpriorityqueue-class)
  - [Sorting with a Priority Queue](#sorting-with-a-priority-queue)
    - [Selection-Sort](#selection-sort)
    - [Insertion-Sort](#insertion-sort)
    - [Heap-Sort](#heap-sort)
    - [Implementing Heap-Sort In-Place](#implementing-heap-sort-in-place)
  - [Adaptable Priority Queues](#adaptable-priority-queues)
    - [The Adaptable Priority Queue ADT](#the-adaptable-priority-queue-adt)
      - [Location-Aware Entries](#location-aware-entries)
      - [Implementing an Adaptable Priority Queue](#implementing-an-adaptable-priority-queue)


- ref
  - DS - pythonds3 - 7. Binary Heap
  - Data Structures and Algorithms in Java, 6th Edition.pdf


---


# Data Structures Basic 1 Priority Queues



---

## Priority Queues


**queue**
- first-in first-out data structure


In practice, there are many applications in which a `queue-like structure` is used to manage objects that must be processed in some way, but `the first-in, first-out policy does not suffice`.
- It is unlikely that the landing decisions are based purely on a FIFO policy.
- “first come, first serve” policy might seem reasonable, yet for which other priorities come into play.



**priority queue**

- One important variation of Queue

- a collection of `prioritized elements` that
  - allows `arbitrary element insertion`,
  - and allows `the removal of the element that has first priority`.

- A priority queue acts like a queue that dequeue an item by removing it from the front.
  - However, in a priority queue the `logical order of items` inside a queue is determined by their `priority`.
  - The highest priority items are at the front of the queue and the lowest priority items are at the back.
  - Thus when you enqueue an item on a priority queue, the new item may move all the way to the front.


When an element is added to a priority queue
- the user designates its priority by providing an associated **key**.
- `The element with the minimal key will be the next to be removed` from the queue
  - an element with key 1 will be given priority over an element with key 2
- Although it is quite common for priorities to be expressed numerically, any Java object may be used as a key, as long as there exists means to compare any two instances a and b, in a way that defines a natural order of the keys.
- With such generality, applications may develop their own notion of priority for each element.
  - For example
  - different financial analysts may assign different ratings (i.e., priorities) to a particular asset, such as a share of stock.


Implement a priority queue using `sorting` functions and `lists`.
- However, inserting into a list is `𝑂(𝑛)` and sorting a list is `𝑂(𝑛log𝑛)`
- We can do better.
- The classic way to implement a priority queue is using a data structure called a `binary heap`.
  - A binary heap will allow us both enqueue and dequeue items in `𝑂(log𝑛)`.




---

### ADT: Priority Queue in java

- When an element is added to a priority queue, the user designates its priority by providing an associated key.
- The element with the minimal key will be the next to be removed from the queue (thus, an element with key 1 will be given priority over an element with key 2).
- Although it is quite common for priorities to be expressed numerically, any Java object may be used as a key, as long as there exists means to compare any two instances a and b, in a way that defines a natural order of the keys.
- With such generality, applications may develop their own notion of priority for each element.
  - For example,
  - different financial analysts may assign different ratings (i.e., priorities) to a particular asset,
  - such as a share of stock.


`insert(k, v)`:
- Creates an entry with key k and value v in the priority queue.

`min()`:
- Returns (but does not remove) a priority queue entry (k,v) having minimal key;
- returns null if the priority queue is empty.

`removeMin()`:
- Removes and returns an entry (k,v) having minimal key fromm the priority queue;
- returns null if the priority queue is empty.

`size()`:
- Returns the number of entries in the priority queue.

`isEmpty()`:
- Returns a boolean indicating whether the priority queue is empty.


total order relation
- it satisfies the following properties for any keys k1, k2, and k3:
- **Comparability property**: k1 ≤ k2 or k2 ≤ k1.
- **Antisymmetric property**: if k1 ≤ k2 and k2 ≤ k1, then k1 = k2.
- **Transitive property**: if k1 ≤ k2 and k2 ≤ k3, then k1 ≤ k3.


Method | Unsorted List | Sorted List
---|---|---
size | O(1) | O(1)
isEmpty | O(1) | O(1)
insert | O(1) | O(n)
min | O(n) | O(1)
removeMin | O(n) | O(1)
space requirement | O(n) |

---




### Implementing a Priority Queue



---



#### Entry **Interface**




implementing a priority queue
- we must keep track of both `an element and its key`
  - even as entries are relocated within a data structure.
- This is reminiscent of maintain a list of elements with access frequencies.
  - use **Design Pattern: Composite**
  - defining an Item class that paired each element with its associated count in our primary data structure.

  - For priority queues
    - we use composition to `pair a key k and a value v as a single object`.
    - To formalize this, we define the public interface, `Entry`


```java
// ∗ Interface for a key-value pair. ∗/
public interface Entry<K, V> {
    K getKey();
    V getV();
}
```


---

#### PriorityQueue **Interface**

use the Entry interface for the priority queue
- This allows us to `return both a key and value as a single object` from methods such as min and removeMin.

define the insert method to return an entry;
- in a more **advanced adaptable priority queue**, that entry can be subsequently updated or removed.

```java
// ∗∗ Interface for the priority queue ADT. ∗/
public interface PriorityQueue<K, V> {
    int size();
    boolean isEmpty();
    Entry<K,V> insert(K key, V value) throws IllegalArgumentException;
    Entry<K,V> mim();
    Entry<K,V> removeMim();
}
```


---

#### defining comparisons

**Comparing Keys with Total Orders**
- we can allow any type of object to serve as a key
  - but we must be able to compare keys to each other in a meaningful way.
  - More so, the results of the comparisons must not be contradictory.

- For a comparison rule, which we denote by ≤, to be self-consistent, it must define a total order relation, which is to say that it satisfies the following properties for any keys k1, k2, and k3:

  - **Comparability property**: k1 ≤ k2 or k2 ≤ k1.
    - The comparability property states that comparison rule is defined for every pair of keys.
    - Note that this property implies the following one:
      - **Reflexive property**: k ≤ k.

  - **Antisymmetric property**: if k1 ≤ k2 and k2 ≤ k1, then k1 = k2.
  - **Transitive property**: if k1 ≤ k2 and k2 ≤ k3, then k1 ≤ k3.


- A comparison rule, ≤, that defines a total order relation will never lead to a contradiction.
- Such a rule defines a linear ordering among a set of keys;
  - hence, if a (finite) set of elements has a total order defined for it, then the notion of a minimal key, kmin, is well defined, as a key in which kmin ≤ k, for any other key k in our set.



- Java provides two means for defining comparisons between object types.


---

##### Comparable **Interface**

a class may define what is known as the natural ordering of its instances by formally implementing the `java.lang.Comparable` interface -> method, `compareTo`.

- For example,
  - the **compareTo method of the String class** defines the natural ordering of strings to be lexicographic, which is a case-sensitive extension of the alphabetic ordering to Unicode.

The syntax `a.compareTo(b)` must return an integer i with the following meaning:
- i<0 designates that `a<b`.
- i=0 designates that `a=b`.
- i>0 designates that `a>b`.


---



##### Comparator **Interface**


- to compare objects according to some notion other than their natural ordering.
- For example
  - which of two strings is the shortest
  - defining our own complex rules for judging which of two stocks is more promising.

- To support generality, Java defines the `java.util.Comparator interface`.
  - A comparator is an object that is external to the class of the keys it compares.
  - It provides a method with the signature `compare(a, b)` that returns an integer with similar meaning to the `compareTo` method described above.


**Example**
- a **comparator** that evaluates strings based on their `length` (rather than their natural lexicographic order).


```java
// /∗∗ Compares two strings according to their lengths. ∗/
public class StringLengthComparator implements Comparator<String> {
  public int compare(String a, String b) {
    if (a.length() < b.length()) return −1;
    else if (a.length() == b.length()) return 0;
    else return 1;
  }
}
```


**Comparators and the Priority Queue ADT**

For a general and reusable form of a priority queue
1. it allow a user to `choose any key type` and to `send an appropriate comparator instance as a parameter to the priority queue constructor`.

   - The priority queue will use that comparator anytime it needs to compare two keys to each other.

2. it allow a `default priority queue to instead rely on the natural ordering` for the given keys (assuming those keys come from a comparable class).
   - In that case, we build our own instance of a DefaultComparator class
   - implements a comparator based upon the natural ordering of its element type.

```java
public class DefaultComparator<E> implements Comparator<String> {
    public int compare(String a, String b) throws ClassCastException {
      return ( (Comparable<E>) a ).compareTo(b);
    }
}
```


---



#### AbstractPriorityQueue **Abstract base class**

- To manage technical issues common to all our priority queue implementations, we define an **abstract base class** named `AbstractPriorityQueue`
- This includes a nested `PQEntry` class that implements the public Entry interface.
- Our abstract class also declares and initializes
  - an instance variable, comp, that stores the comparator being used for the priority queue.
  - a protected method, compare, that invokes the comparator on the keys of two given entries.

```java
package pq;

public abstract class AbstractPriorityQueue<K,V> implements PriorityQueue<K,V> {

    //---------------- nested PQEntry class ----------------
    protected static class PQEntry<K,V> implements Entry<K,V> {
        private K k;
        private V v;
        public PQEntry(K key, V value){
            k=key;
            v=value;
        }
        // methods of the Entry interface
        public K getKey(){return k;}
        public V getV(){return v;}
        // utilities not exposed as part of the Entry interface
        protected void setKey(K key){k=key;}
        protected void setValue(V value){v=value;}

    }

    // instance variable for an AbstractPriorityQueue
    private Comparator<K> comp;
    protected AbstractPriorityQueue(Comparator<K> c){comp = c;}
    protected AbstractPriorityQueue(){ this(new DefaultComparator<K>()); }

    // /∗∗ Method for comparing two entries according to key ∗/
    protected int compare(Entry<K,V> a, Entry<K,V> b) { return comp.compare(a.getKey(), b.getKey());}

    // /∗∗ Determines whether a key is valid. ∗/
    protected boolean checkKey(K key) throws IllegalArgumentException {
        try {
            return comp.compare(key, key)==0;
        } catch (ClassCastException e) {
            throw new IllegalArgumentException("Incompatible key.");
        }
    }

    // /∗∗ Tests whether the priority queue is empty. ∗/
    public boolean isEmpty(){return size()==0;}
}
```



---



#### UnsortedPriorityQueue **class** Unsorted List

store entries within an unsorted linked list.


```java
package pq;
import list.*;

// ** An implementation of a priority queue with an unsorted list. */
public class UnsortedPriorityQueue<K,V> extends AbstractPriorityQueue<K,V> {

    // ∗∗ primary collection of priority queue entries ∗/
    private PositionalList<Entry<K,V>> list = new LinkedPositionalList<>();

    /** Creates an empty priority queue based on the natural ordering of its keys. */
    public UnsortedPriorityQueue() { super(); }
    /** Creates an empty priority queue using the given comparator to order keys. */
    public UnsortedPriorityQueue(Comparator<K> comp) { super(comp); }

    /** Returns the Position of an entry having minimal key. */
    private Position<Entry<K,V>> findMin() { // only called when nonempty
        Position<Entry<K,V>> small = list.first();
        for (Position<Entry<K,V>> walk : list.positions())
        if (compare(walk.getElement(), small.getElement()) < 0) small = walk; // found an even smaller key
        return small;
    }

    /** Inserts a key-value pair and returns the entry created. */
    public Entry<K,V> insert(K key, V value) throws IllegalArgumentException {
        checkKey(key); // auxiliary key-checking method (could throw exception)
        Entry<K,V> newest = new PQEntry<>(key, value);
        list.addLast(newest);
        return newest;
    }
    /** Returns (but does not remove) an entry with minimal key. */
    public Entry<K,V> min() {
        if (list.isEmpty()) return null;
        return findMin().getElement();
    }

    /** Removes and returns an entry with minimal key. */
    public Entry<K,V> removeMin() {
        if (list.isEmpty()) return null;
        return list.remove(findMin()); }

    /** Returns the number of items in the priority queue. */
    public int size() { return list.size(); }
}
```










---



#### SortedPriorityQueue **class** sorted List


- implementation of a priority queue also uses a positional list, yet maintains entries sorted by nondecreasing keys.
- This ensures that the first element of the list is an entry with the smallest key.

```java
package pq;

import list.*;

// ** An implementation of a priority queue with an unsorted list. */
public class SortedPriorityQueue<K,V> extends AbstractPriorityQueue<K,V> {

    // ∗∗ primary collection of priority queue entries ∗/
    private PositionalList<Entry<K,V>> list = new LinkedPositionalList<>();

    /** Creates an empty priority queue based on the natural ordering of its keys. */
    public SortedPriorityQueue() { super(); }
    /** Creates an empty priority queue using the given comparator to order keys. */
    public SortedPriorityQueue(Comparator<K> comp) { super(comp); }

    /** Inserts a key-value pair and returns the entry created. */
    public Entry<K,V> insert(K key, V value) throws IllegalArgumentException {
        checkKey(key); // auxiliary key-checking method (could throw exception)
        Entry<K,V> newest = new PQEntry<>(key, value);
        Position<Entry<K,V>> walk = list.last();
        while(walk != null && compare(newest, walk.getElement())<0 ) walk = list.before(walk);
        if(walk==null) list.addFirst(newest);
        else list.addAfter(walk, newest);
        return newest;
    }

    /** Returns the Position of an entry having minimal key. */
    private Position<Entry<K,V>> findMin() { // only called when nonempty
        return list.first();
    }

    /** Returns (but does not remove) an entry with minimal key. */
    public Entry<K,V> min() {
        if (list.isEmpty()) return null;
        return list.first().getElement();
    }

    /** Removes and returns an entry with minimal key. */
    public Entry<K,V> removeMin() {
        if (list.isEmpty()) return null;
        return list.remove(list.first()); }

    /** Returns the number of items in the priority queue. */
    public int size() { return list.size(); }
}
```


---




#### import java.util.PriorityQueue

https://docs.oracle.com/javase/7/docs/api/java/util/PriorityQueue.html#add(E)









---


## Binary Heap 堆

two strategies for implementing a priority queue ADT demonstrate an trade-off.
- using an `unsorted` list to store entries
  - `insertions` in `O(1)` time,
  - but `finding or removing` an element with minimal key `requires an O(n)-time loop` through the entire collection.

- using a `sorted` list,
  - `find or remove` the minimal element in O(1) time,
  - but `adding` a new element to the queue `may require O(n) time` to restore the sorted order.



**Heap**
- looks a lot like a tree,
- but we implement it only need a single list as an internal representation.



**binary heap**
- a more efficient realization of a priority queue using a data structure
- This data structure allows both insertions and removals in logarithmic time, a significant improvement over the list-based implementations

- The fundamental way the heap achieves this improvement is to `use the structure of a binary tree` to find a compromise between elements being entirely unsorted and perfectly sorted.

take advantage of the logarithmic nature of the binary tree
- In order to guarantee logarithmic performance, we must keep our tree balanced.
- A balanced binary tree has roughly the same number of nodes in the left and right subtrees of the root.
- keep the tree balanced by creating a <font color=red> complete binary tree </font>.
  - A complete binary tree is a tree in which each level has all of its nodes.
  - The exception to this is the bottom level of the tree, which we fill in from left to right.

![heapOrder](https://i.imgur.com/FzGkeOJ.png)


- The binary heap has two common variations:
  - min heap, the <font color=red> smallest key is always at the front </font>,
  - max heap, the <font color=red> largest key value is always at the front </font>.




---

### The Heap Data Structure

- A heap is a binary tree T that stores `entries` at its `positions`, and that satisfies two additional properties:
  - a **relational property** defined in `keys` stored in T
  - and a **structural property** defined in `the shape of T` itself.


- **relational property**:
  - **Heap-Order Property**:
    - `The method to store items in a heap` relies on maintaining the **heap order property**.
    - In a heap T , for every position p other than the root, `the key stored at p >= the key stored at p’s parent`.
    - consequence:
      - the keys encountered on a path from the root to a leaf of T are in nondecreasing order.
      - Also, a minimal key is always stored at the root of T.
        - easy to locate such an entry when min or removeMin is called

- **structural property**:
  - For the sake of efficiency, `want the heap T to have as small a height as possible`.
  - enforce this requirement by insisting that the heap T satisfy an additional structural property; it must be what we term complete.
  - **Complete Binary Tree Property**:
    - A `heap T with height h` is a complete binary tree if
      - levels `0,1,2,...,h−1` of T have the maximal number of nodes possible (level i has 2i nodes, for 0 ≤ i ≤ h − 1)
      - and the remaining nodes at level h reside in the leftmost possible positions at that level.

![Screen Shot 2022-03-31 at 23.57.55](https://i.imgur.com/3SWsjqX.png)

leftmost possible positions,
- level numbering in array-based representation of a binary tree.
- A complete binary tree with n elements is one that has positions with level numbering 0 through n − 1.
- For example, in an array-based representation of the above tree, its 13 entries would be stored consecutively `from A[0] to A[12]`



**example**
a complete binary tree that has the heap order property.

![percUp](https://i.imgur.com/xWvuclU.png)



---

#### The Height of a Heap

Proposition 9.2: **A heap T storing n entries has height h = ⌊log n⌋**
Justification:
- From the fact that T is complete
- Let h denote the height of T
- the number of nodes in **levels 0 through h−1** of T is precisely `1+2+4+···+2^(h−1) = 2^h −1`,
- the number of nodes in **level h** is `at least 1 and at most 2^h`.
- Therefore `n≥2^h−1+1=2^h` and `n≤2^h−1+2^h =2^(h+1)−1`.
- By taking the logarithm of both sides of inequality n ≥ 2^h, we see that height `h ≤ log n`.
- By taking the logarithm of both sides of inequality n ≤ 2^(h+1) −1, we see that `h ≥ log(n+1)−1`.
- Since h is an integer, these two inequalities imply that `h = ⌊log n⌋`.


**if we can perform update operations on a heap in time proportional to its height, then those operations will run in logarithmic time.**


---



## Implement Priority Queue


### Implementing a Priority Queue with a Heap


to efficiently perform various priority queue methods using a heap.
- use the composition pattern to `store key-value pairs as entries in the heap`.
- The `size` and `isEmpty` methods can be implemented based on examination of the tree
- the `min` operation is equally trivial because the heap property assures that the element at the root of the tree has a minimal key.
- The interesting algorithms are the `insert` and `removeMin` methods.



**Adding an Entry to the Heap**
- to perform `insert(k, v)` on a priority queue implemented with a heap T .
- store the pair (k, v) as an entry at a new node of the tree.
- To maintain the complete binary tree property
  - new node should be placed at a position p just beyond the `rightmost node at the bottom level of the tree`, or as the `leftmost position of a new level` (if the bottom level is already full or if the heap is empty).


**Up-Heap Bubbling After an Insertion**
- After this action, the tree T is complete
- but it may violate the **heap-order property**.
- Hence, unless position p is the root of T (the priority queue was empty before the insertion), we compare the key at position p to that of p’s parent, which we denote as q.
- If key `kp ≥ kq`, the heap-order property is satisfied and the algorithm terminates.
- If instead `kp < kq`
  - then we need to restore the heap-order property, which can be locally achieved by `swapping the entries stored at positions p and q`.
  - This swap causes the new entry to move up one level.
  - Again, the heap-order property may be violated, so we repeat the process,
  - going up in T until no violation of the heap-order property occurs.
- The upward movement of the newly inserted entry by means of swaps is conventionally called **up-heap bubbling**.
- A swap either resolves the violation of the heap-order property or propagates it one level up in the heap.
- In the worst case, up-heap bubbling causes the `new entry to move all the way up to the root of heap T` .
  - Thus the number of swaps performed in insert == the height of T, that bound is `⌊log n⌋`.



**Removing the Entry with Minimal Key**
- to method removeMin of the priority queue ADT.
- an entry with the smallest key is stored at the root r of T (even if there is more than one entry with smallest key).
  - However, cannot simply delete node r,
  - because this would leave two disconnected subtrees.
- to ensure that the shape of the heap respects the complete binary tree property
  - deleting the leaf at the last position p of T (the rightmost position at the bottommost level of the tree).
  - To preserve the entry from the `last position p`, we copy it to the root r (the entry with minimal key that is being removed by the operation).
  - with minimal entry being removed from the root and replaced by entry from the last position.
  - The node at the last position is removed from the tree.

![Screen Shot 2022-04-01 at 09.43.23](https://i.imgur.com/cjkPQ2P.png)



**Down-Heap Bubbling After a Removal**
- even though T is now complete, it likely violates the heap-order property.
- If T has only one node (the root), then the heap-order property is trivially satisfied and the algorithm terminates.
- Otherwise, two cases, where p initially denotes the root of T:
  - If p has no right child, let c be the left child of p.
  - Otherwise (p has both children), let c be a child of p with minimal key.
    - If key kp ≤ kc, the heap-order property is satisfied and the algorithm terminates.
    - If instead kp > kc, then we need to restore the heap-order property. This can be locally achieved by swapping the entries stored at p and c.
    - It is worth noting that when p has two children, we intentionally consider the smaller key of the two children.
    - Not only is the key of c smaller than that of p, it is at least as small as the key at c’s sibling.
    - This ensures that the heap-order property is locally restored when that smaller key is promoted above the key that had been at p and that at c’s sibling.
- Having restored the heap-order property for node p relative to its children, there may be a violation of this property at c;
- hence, we may have to continue swapping down T until no violation of the heap-order property occurs.
- This downward swapping process is called **down-heap bubbling**.
- A swap either resolves the violation of the heap-order property or propagates it one level down in the heap.
- In the worst case, an entry moves all the way down to the bottom level.
  - the number of swaps performed in removeMin == the height of heap T, `log n⌋`



---


#### Complete Binary Tree **Array-Based**
- array-based binary tree is especially suitable for a complete binary tree.
- the elements of the tree are stored in an `array-based list A` such that the element at position p is stored in A with index equal to the level number f(p) of p, defined as follows:
  - If p is the root,then f(p)=0.
  - If p is the left child of position q, then f(p) = 2f(q)+1.
  - If p is the right child of position q, then f(p) = 2f(q)+2.
- For a tree with of size n, the elements have contiguous indices in the range [0, n − 1] and the last position of is always at index n − 1.


![Screen Shot 2022-04-01 at 11.56.09](https://i.imgur.com/04PxEIi.png)


The **array-based heap representation**
- avoids some complexities of a linked tree structure.
  - methods `insert` and `removeMin` depend on locating the last position of a heap.
  - With the array-based representation of a heap of size n, the last position is simply at index `n − 1`.
  - Locating the last position in a heap implemented with a linked tree structure requires more effort.

- If the size of a priority queue is not known in advance,
  - use of an array-based representation does introduce the need to dynamically resize the array on occasion, done with ArrayList.
- The space usage of such an array-based representation of a complete binary tree with n nodes is `O(n)`
- the time bounds of methods for `adding` or `removing` elements become amortized


---


#### priority queue **heap-based**

- Java implementation of a heap-based priority queue.

- Although we think of our heap as a binary tree, we do not formally think of our heap as a binary tree, do not formally use the binary tree ADT.
  - prefer to use the more efficient array-based representation of a tree,
  - maintaining a Java `ArrayList` of entry composites.
  - To allow us to formalize our algorithms using tree-like terminology of parent, left, and right, the class includes protected utility methods that compute the **level numbering** of a parent or child of another position
  - However, the “positions” in this representation are simply integer indices into the array-list.

- Our class also has protected utilities swap, upheap, and downheap for the low-level movement of entries within the array-list.
  - A new entry is added the end of the array-list, and then repositioned as needed with **upheap**.
  - To remove the entry with minimal key (which resides at index 0), we move the last entry of the array-list from index n − 1 to index 0, and then invoke **downheap** to reposition it.





---

##### heap in java



```java
package pq;

/** An implementation of a priority queue using an array-based heap. */
public class HeapPriorityQueue<K,V> extends AbstractPriorityQueue<K,V> {

    /** primary collection of priority queue entries */
    protected ArrayList<Entry<K,V>> heap = new ArrayList<>();

    /** Creates an empty priority queue based on the natural ordering of its keys. */
    public HeapPriorityQueue() { super(); }
    /** Creates an empty priority queue using the given comparator to order keys. */
    public HeapPriorityQueue(Comparator<K> comp) { super(comp); }

    // protected utilities
    protected int parent(int j) { return (j-1) / 2; } // truncating division
    protected int left(int j) { return 2*j + 1; }
    protected int right(int j) { return 2*j + 2; }
    protected boolean hasLeft(int j) { return left(j) < heap.size(); }
    protected boolean hasRight(int j) { return right(j) < heap.size(); }

    /** Exchanges the entries at indices i and j of the array list. */
    protected void swap(int i, int j) {
        Entry<K,V> temp = heap.get(i);
        heap.set(i, heap.get(j));
        heap.set(j, temp);
    }

    /** Moves the entry at index j higher, if necessary, to restore the heap property. */
    protected void upheap(int j) {
        while (j > 0) { // continue until reaching root (or break statement)
            int p = parent(j);
            if (compare(heap.get(j), heap.get(p)) >= 0) break; // heap property verified
            swap(j, p);
            j = p; // continue from the parent's location
        }
    }

    /** Moves the entry at index j lower, if necessary, to restore the heap property. */
    protected void downheap(int j) {
        while (hasLeft(j)) { // continue to bottom (or break statement)
            int leftIndex = left(j);
            int smallChildIndex = leftIndex;  // although right may be smaller
            if (hasRight(j)) {
                int rightIndex = right(j);
                if (compare(heap.get(leftIndex), heap.get(rightIndex)) > 0) smallChildIndex = rightIndex; // right child is smaller
            }
            if (compare(heap.get(smallChildIndex), heap.get(j)) >= 0) break; // heap property has been restored
            swap(j, smallChildIndex);
            j = smallChildIndex;  // continue at position of the child
        }
    }

    // public methods
    /** Returns the number of items in the priority queue. */
    public int size( ) { return heap.size( ); }
    /** Returns (but does not remove) an entry with minimal key (if any). */
    public Entry<K,V> min( ) {
        if (heap.isEmpty()) return null;
        return heap.get(0);
    }

    /** Inserts a key-value pair and returns the entry created. */
    public Entry<K,V> insert(K key, V value) throws IllegalArgumentException {
        checkKey(key); // auxiliary key-checking method (could throw exception)
        Entry<K,V> newest = new PQEntry<>(key, value);
        heap.add(newest); // add to the end of the list
        upheap(heap.size() - 1); // upheap newly added entry
        return newest;
    }

    /** Removes and returns an entry with minimal key (if any). */
    public Entry<K,V> removeMin( ) {
        if (heap.isEmpty()) return null;
        Entry<K,V> answer = heap.get(0);
        swap(0, heap.size() - 1); // put minimum item at the end
        heap.remove(heap.size() - 1); // and remove it from the list;
        downheap(0); // then fix new root
        return answer;
    }
}
```



---


###### analyze the binary heap

- assuming that
  - two keys can be compared in O(1) time
  - and the heap T is implemented with an array-based or linked-based tree representation.

In short, each of the priority queue ADT methods can be performed in O(1) or in O(logn) time, where n is the number of entries at the time the method is executed.


The analysis of the running time of the methods is based on the following:
- The heap T has n nodes, each storing a reference to a key-value entry.
- The `height of heap T is O(log n)`, since T is complete
- The **min** operation runs in `O(1)`: the root of the tree contains such an element.
- for **insert** and **removeMin**: Locating the last position of a heap can be performed in `O(1)` time for an array-based representation, or `O(log n)` time for a linked-tree representation
- In the worst case, **up-heap and down-heap bubbling** perform a number of swaps equal to the height of T `O(log n)`



- size, isEmpty, min: `𝑂(1)`
- insert: `𝑂(log𝑛)`
- removeMin: `𝑂(log𝑛)`
- up-heap and down-heap bubbling: `𝑂(log𝑛)`





---









---



##### max heap in java

`swim(int k)`

![swim](https://i.imgur.com/ITg3gBR.gif)


`sink(int k)`

![sink](https://i.imgur.com/1yOPINm.gif)


```java
public class MaxPQ
    <Key extends Comparable<Key>> {

    private Key[] pq;    // 存储元素的数组
    private int N = 0;   // 当前 Priority Queue 中的元素个数

    public MaxPQ(int cap) {
        // 索引 0 不用，所以多分配一个空间
        pq = (Key[]) new Comparable[cap + 1];
    }

    /* 返回当前队列中最大元素 */
    public Key max() {
        return pq[1];
    }

    // /* 插入元素 e */ 插入和删除元素的时间复杂度为 O(logK)
    public void insert(Key e)
      N++;
      // 先把新元素加到最后
      pq[N] = e;
      // 然后让它上浮到正确的位置
      swim(N);
    }

    // /* 删除并返回当前队列中最大元素 */ 插入和删除元素的时间复杂度为 O(logK)
    public Key delMax() {
      // 最大堆的堆顶就是最大元素
      Key max = pq[1];
      // 把这个最大元素换到最后，删除之
      exch(1, N);
      pq[N] = null;
      N--;
      // 让 pq[1] 下沉到正确位置
      sink(1);
      return max;
    }

    /* 上浮第 k 个元素，以维护最大堆性质 */
    private void swim(int k) {
      // 如果浮到堆顶，就不能再上浮了
      while(k>1 && less(parent(k), k)){
        // 如果第 k 个元素比上层大
        // 将 k 换上去
        exch(k, parent(k));
        k = parent(k);
      }
    }

    /* 下沉第 k 个元素，以维护最大堆性质 */
    private void sink(int k) {
        // 如果沉到堆底，就沉不下去了
        while (left(k) <= N) {
            // 先假设左边节点较大
            int older = left(k);
            // 如果右边节点存在，比一下大小
            if (right(k) <= N && less(older, right(k)))
                older = right(k);
            // 结点 k 比俩孩子都大，就不必下沉了
            if (less(older, k)) break;
            // 否则，不符合最大堆的结构，下沉 k 结点
            exch(k, older);
            k = older;
        }
    }

    /* 交换数组的两个元素 */
    private void exch(int i, int j) {
        Key temp = pq[i];
        pq[i] = pq[j];
        pq[j] = temp;
    }

    /* pq[i] 是否比 pq[j] 小？ */
    private boolean less(int i, int j) {
        return pq[i].compareTo(pq[j]) < 0;
    }

    /* 还有 left, right, parent 三个方法 */



}


```


---


heap in python

```py
# an empty binary heap has a single zero as the first element of heapList and that this zero is not used, but is there so that simple integer division can be used in later methods.
class BinHeap:
    def __init__(self):
        self.heapList = [0]
        self.currentSize = 0
```



`insert`
- most efficient way to add an item to a list is to simply append the item to the end of the list.
- The good news about appending is that it guarantees that we will maintain the complete tree property.
- The bad news about appending is that we will very likely violate the heap structure property.
- However, it is possible to write a method to regain the **heap structure property** by comparing the newly added item with its parent.
  - If the newly added item is less than its parent, then we can swap the item with its parent.

![percUp](https://i.imgur.com/xWvuclU.png)


when we percolate 扩散 an item up
- we are restoring the heap property between the newly added item and the parent.
- We are also preserving the heap property for any siblings.
- Of course, if the newly added item is very small, we may still need to swap it up another level.
- In fact, we may need to keep swapping until we get to the top of the tree.

`percUp`
- percolates a new item as far up in the tree as it needs to go to maintain the heap property.
- Here is where our wasted element in heapList is important.
- Notice that we can compute the parent of any node by using simple integer division.
- The parent of the current node can be computed by dividing the index of the current node by 2.


```py
def percUp(self,i):
    while i // 2 > 0:
      if self.heapList[i] < self.heapList[i // 2]:
         tmp = self.heapList[i // 2]
         self.heapList[i // 2] = self.heapList[i]
         self.heapList[i] = tmp
      i = i // 2

def insert(self,k):
    self.heapList.append(k)
    self.currentSize = self.currentSize + 1
    self.percUp(self.currentSize)
```



`delMin`

> to keep complete binary tree, replace the last item with the root

- Since the heap property requires that the root of the tree be the smallest item in the tree, finding the minimum item is easy.
- The hard part of delMin is restoring full compliance with the heap structure and heap order properties after the root has been removed.

restore our heap in two steps.
- First, restore the root item by taking the last item in the list and moving it to the root position.
  - It maintains our heap structure property.
  - But we have probably destroyed the heap order property of our binary heap.
- Second, restore the heap order property by pushing the new root node down the tree to its proper position.
  - to maintain the heap order property,
  - swap the root with its smallest child less than the root.
  - After the initial swap, we may repeat the swapping process with a `node` and `its children` until the node is swapped into a position on the tree where it is already less than both children.

![percDown](https://i.imgur.com/I1kHltA.png)

```py
def percDown(self,i):
    while (i * 2) <= self.currentSize:
        mc = self.minChild(i)
        if self.heapList[i] > self.heapList[mc]:
            tmp = self.heapList[i]
            self.heapList[i] = self.heapList[mc]
            self.heapList[mc] = tmp
        i = mc

def minChild(self,i):
    if i * 2 + 1 > self.currentSize:
        return i * 2
    # i * 2 + 1 <= self.currentSize
    else:
        if self.heapList[i*2] < self.heapList[i*2+1]:
            return i * 2
        else:
            return i * 2 + 1
```



`delMin`
- the hard work is handled by a helper function, in this case percDown.

```py
def delMin(self):
    retval = self.heapList[1]
    self.heapList[1] = self.heapList[self.currentSize]
    self.currentSize = self.currentSize - 1
    self.heapList.pop()
    self.percDown(1)
    return retval
```



build an entire heap from a list of keys.

build a heap by inserting each key one at a time.
- a list of one item, the list is sorted
- use binary search to find the right position to insert the next key at a cost of approximately `𝑂(log𝑛)` operations.
- However, inserting an item in the middle of the list may require `𝑂(𝑛)` operations to shift the rest of the list over to make room for the new key.
- Therefore, to insert `𝑛` keys into the heap would require a total of `𝑂(𝑛log𝑛)` operations.


if we start with an entire list then we can build the whole heap in `𝑂(𝑛)` operations.

```py
def buildHeap(self, alist):
    i = len(alist) // 2
    self.currentSize = len(alist)
    self.heapList = [0] + alist[:]
    while (i > 0):
        self.percDown(i)
        i = i - 1
```

![buildheap](https://i.imgur.com/KjU539s.png)


`percDown` method ensures that the largest child is always moved down the tree.
- Because the heap is a complete binary tree, any nodes past the halfway point will be leaves and therefore have no children.
- when `i=1`, we are percolating down from the root of the tree, so this may require multiple swaps.

As you can see in the rightmost two trees of Figure 4,
- first the 9 is moved out of the root position,
- but after 9 is moved down one level in the tree,
- percDown ensures that we check the next set of children farther down in the tree to ensure that it is pushed as low as it can go.
- In this case it results in a second swap with 3. Now that 9 has been moved to the lowest level of the tree, no further swapping can be done.
- It is useful to compare the list representation of this series of swaps as shown in Figure 4 with the tree representation.

```py
i = 2  [0, 9, 5, 6, 2, 3]
i = 1  [0, 9, 2, 6, 5, 3]
i = 0  [0, 2, 3, 6, 5, 9]
```


The assertion that we can build the heap in `𝑂(𝑛)` is beyond the scope of this book. However, the key to understanding that you can build the heap in `𝑂(𝑛)` is to remember that the `log𝑛` factor is derived from the height of the tree.

For most of the work in buildHeap, the tree is shorter than `log𝑛`

Using the fact that you can build a heap from a list in `𝑂(𝑛)` time, you will construct a sorting algorithm that uses a heap and sorts a list in `𝑂(𝑛log𝑛))` as an exercise at the end of this chapter.





---


###### min heap in python

- `BinaryHeap()`
  - creates a new, empty, binary heap.
- `insert(k)`
  - adds a new item to the heap.
- `findMin()`
  - returns the item with the minimum key value, leaving item in the heap.
- `delMin()`
  - returns the item with the minimum key value, removing the item from the heap.
- `isEmpty()`
  - returns true if the heap is empty, false otherwise.
- `size()`
  - returns the number of items in the heap.
- `buildHeap(list)`
  - builds a new heap from a list of keys.


```py
from pythonds.trees import BinHeap

class BinHeap:
    def __init__(self):
        self.heapList = [0]
        self.currentSize = 0

    def percUp(self,i):
        while i // 2 > 0:
          if self.heapList[i] < self.heapList[i // 2]:
             tmp = self.heapList[i // 2]
             self.heapList[i // 2] = self.heapList[i]
             self.heapList[i] = tmp
          i = i // 2

    def insert(self,k):
      self.heapList.append(k)
      self.currentSize = self.currentSize + 1
      self.percUp(self.currentSize)

    def percDown(self,i):
      while (i * 2) <= self.currentSize:
          mc = self.minChild(i)
          if self.heapList[i] > self.heapList[mc]:
              tmp = self.heapList[i]
              self.heapList[i] = self.heapList[mc]
              self.heapList[mc] = tmp
          i = mc

    def minChild(self,i):
      if i * 2 + 1 > self.currentSize:
          return i * 2
      else:
          if self.heapList[i*2] < self.heapList[i*2+1]:
              return i * 2
          else:
              return i * 2 + 1

    def delMin(self):
      retval = self.heapList[1]
      self.heapList[1] = self.heapList[self.currentSize]
      self.currentSize = self.currentSize - 1
      self.heapList.pop()
      self.percDown(1)
      return retval

    def buildHeap(self,alist):
      i = len(alist) // 2
      self.currentSize = len(alist)
      self.heapList = [0] + alist[:]
      while (i > 0):
          self.percDown(i)
          i = i - 1

bh = BinHeap()
bh.buildHeap([9,5,6,2,3])

print(bh.delMin())
print(bh.delMin())
print(bh.delMin())
print(bh.delMin())
print(bh.delMin())
```

Notice that no matter the order that we add items to the heap, the smallest is removed each time.

---





### Bottom-Up Heap Construction


- If we start with an initially empty heap, n successive calls to the insert operation will run in `O(nlogn)` time in the worst case.
- However, if `all n key-value pairs to be stored in the heap are given in advance`, such as during the first phase of the heap-sort algorithm, there is an alternative bottom-up construction method that runs in `O(n)` time.


In this section, we describe the bottom-up heap construction, and provide an implementation that can be used by the constructor of a heap-based priority queue.


For simplicity of exposition, we describe this bottom-up heap construction assuming
- the number of keys, n, is an integer that `n = 2^(h+1) − 1`
  - the heap is a complete binary tree with every level being full,
- so the heap has height `h = log(n + 1) − 1`.
- Viewed non recursively, bottom-up heap construction consists of the following `h + 1 = log(n + 1)` steps:



![Screen Shot 2022-04-01 at 13.20.09](https://i.imgur.com/nJevOoi.png)

1. construct `(n + 1)/2` elementary heaps storing one entry each.
2. form `(n+1)/4` heaps, each storing three entries, by joining pairs of elementary heaps and adding a new entry.
   1. The new entry is placed at the root and may have to be swapped with the entry stored at a child to preserve the heap-order property.
3. form `(n + 1)/8` heaps, each storing 7 entries, by joining pairs of 3-entry heaps (constructed in the previous step) and adding a new entry. The new entry is placed initially at the root, but may have to move down with a down-heap bubbling to preserve the heap-order property.
4. In the generic i^th step, 2 ≤ i ≤ h, we form `(n+1)/2^i` heaps, each storing 2i −1 entries, by joining pairs of heaps storing (2i−1 − 1) entries (constructed in the previous step) and adding a new entry. The new entry is placed initially at the root, but may have to move down with a down-heap bubbling to preserve the heap-order property.
5. In the last ste, form the final heap,
   1. storing all the n entries, by joining two heaps storing `(n − 1)/2` entries (constructed in the previous step)
   2. and adding a new entry.
   3. The new entry is placed initially at the root, but may have to move down with a down-heap bubbling to preserve the heap-order property.


---



#### Implementation in Java


Implementing a bottom-up heap construction is quite easy, given the existence of a “down-heap” utility method.

- The “merging” of two equally sized heaps that are subtrees of a common position p can be accomplished simply by down-heaping p’s entry.
  - For example, that is what happened to the key 14 in going from Figure 9.5(f ) to (g).

- With our array-based representation of a heap
  - if we initially store all n entries in arbitrary order within the array,
  - we can implement the bottom-up heap construction process with a single loop that makes a call to downheap from each position of the tree, as long as those calls are ordered starting with the deepest level and ending with the root of the tree.
- In fact, that loop can start with the deepest internal position, since there is no effect when down-heap is called at an external position.


- augment the original HeapPriorityQueue class to provide support for the **bottom-up construction** of an initial collection.
- a nonpublic utility method, `heapify`, that calls downheap on each nonleaf position, beginning with the deepest and concluding with a call at the root of the tree.
- an additional constructor for the class that accepts an initial sequence of keys and values, parameterized as two coordinate arrays that are presumed to have the same length.
  - create new entries,
  - pairing the first key with the first value, the second key with the second value, and so on.
  - then call the heapify utility to establish the heap ordering.
- For brevity, we omit a similar constructor that accepts a nondefault comparator for the priority queue.



##### Asymptotic Analysis of Bottom-Up Heap Construction
- `Bottom-up heap construction` is **asymptotically faster** than incrementally inserting n entries into an initially empty heap.
- it performing a single down-heap operation at each position in the tree, rather than a single up-heap operation from each.
- Since more nodes are closer to the bottom of a tree than the top, the sum of the downward paths is linear


**Proposition 9.3**:
- Bottom-up construction of a heap with n entries takes O(n) time, assuming two keys can be compared in O(1) time.
**Justification**:
- The primary cost of the construction is due to the **down-heap steps** performed at `each nonleaf position`.
- Let `πv` denote the path of T from nonleaf node v to its “inorder successor” leaf
  - the path that starts at v, goes to the right child of v, and then goes down leftward until it reaches a leaf.
- Although, `πv` is not necessarily the path followed by the down-heap bubbling step from v, its number of edges `∥πv∥` is proportional to the height of the subtree rooted at v, and thus a bound on the complexity of the down-heap operation at v.
- The total running time of the bottom-up heap construction algorithm is therefore bounded by the `sum ∑v ∥πv∥`.
- For intuition, Figure 9.6 illustrates the justification “visually,” marking each edge with the label of the nonleaf node v whose path `πv` contains that edge.




- We claim that the paths `πv` for all nonleaf v are edge-disjoint, and thus the sum of the path lengths is bounded by the number of total edges in the tree, hence O(n). To show this, we consider what we term “right-leaning” and “left-leaning” edges (i.e., those going from a parent to a right, respectively left, child). A particular right- leaning edge e can only be part of the path `πv` for node v that is the parent in the relationship represented by e. Left-leaning edges can be partitioned by considering the leaf that is reached if continuing down leftward until reaching a leaf. Each nonleaf node only uses left-leaning edges in the group leading to that nonleaf node’s inorder successor. Since each nonleaf node must have a different inorder successor, no two such paths can contain the same left-leaning edge.
- We conclude that the bottom-up construction of heap T takes `O(n)` time.



---


### java.util.PriorityQueue **Class**

There is no priority queue interface built into Java, but Java does include a **class**, `java.util.PriorityQueue`, which implements the `java.util.Queue` **interface**.


- Instead of adding and removing elements according to the standard FIFO policy used by most queues
- `java.util.PriorityQueue` **class** processes its entries according to a priority
  - The “front” of the queue will always be a minimal element
  - with priorities based either on the natural ordering of the elements, or in accordance with a `comparator` **object** sent as a parameter when constructing the priority queue.
  - java.util.PriorityQueue class relies on a single element type. That element is effectively treated as a key.
  - If a user wishes to insert distinct keys and values, the burden is on the user to define and insert appropriate composite objects, and to ensure that those objects can be compared based on their keys.

- The java.util.PriorityQueue class is implemented with a heap
  - so it guarantees `O(logn)` time performance for methods `add` and `remove`,
  - and `O(1)` constant-time performance for accessors `peek`, `size`, and `isEmpty`.
  - In addition, it provides a parameterized method, `remove(e)` that removes a specific element e from the priority queue runs in `O(n)` time, performing a sequential search to locate the element within the heap.

![Screen Shot 2022-04-08 at 01.02.45](https://i.imgur.com/IJfWqbT.png)


---




## Sorting with a Priority Queue


The algorithm for sorting a `sequence S` with a `priority queue P` is quite simple and consists of the following two phases:
1. insert the elements of `S` as keys into an initially empty `priority queue P` by means of a series of n insert operations, one for each element.
2. extract the elements from P in nondecreasing order by means of a series of n removeMin operations, putting them back into S in that order.


The algorithm works correctly for any priority queue P, no matter how P is implemented.
- but the running time of the algorithm is determined by running times of operations insert and removeMin, which do depend on how P is implemented.
- Indeed, pqSort should be considered more a sorting “scheme” than a sorting “algorithm,” because it does not specify how the priority queue P is implemented.
- The pqSort scheme is the paradigm of several popular sorting algorithms, including selection-sort, insertion-sort, and heap-sort.


---


### Selection-Sort


![Screen Shot 2022-04-08 at 01.17.26](https://i.imgur.com/Ortlmyb.png)

> If we implement P with an unsorted list


- In Phase 1 of the pqSort scheme, we insert all elements into a priority queue P;
  - takes `O(n)` time, insert each element in O(1) time.
- in Phase 2 we repeatedly `remove the minimal element from P` using the **removeMin** method.
  - the running time of each **removeMin** operation is proportional to the size of P.
    - the bottleneck computation is the <font color=red> repeated “selection” of the minimum element in Phase 2. </font>
    - For this reason, this algorithm is better known as **selection-sort**
    - the bottleneck is in Phase 2: `repeatedly remove an entry with smallest key from the priority queue P`.
      - The size of P starts at n and incrementally decreases with each removeMin until it becomes 0.
      - Thus, the first removeMin operation takes time O(n),
      - the second one takes time O(n − 1), and so on,
      - until the last (nth) operation takes time O(1).
      - Therefore, the total time needed for the second phase is `􏰎∑n 􏰏 O(n+(n−1)+···+2+1)`
      - ∑ni=1 i = n(n + 1)/2.
    - Phase 2 takes time` O(n^2)`, as does the entire selection-sort algorithm.

---


### Insertion-Sort

> implement the priority queue P using a sorted list


- then the running time of Phase 2 improves to O(n), for each operation removeMin on P now takes O(1) time.
- Unfortunately, Phase 1 now becomes the bottleneck for the running time,
  - in the worst case, each insert operation takes time proportional to the size of P.
  - This sorting algorithm is therefore better known as **insertion-sort**
  - for the bottleneck in this sorting algorithm involves the <font color=red> repeated “insertion” of a new element at the appropriate position in a sorted list </font>.


- In Phase 1, we repeatedly remove the first element of S and insert it into P.

- In Phase 2, we repeatedly perform the removeMin operation on P and add the returned element to the end of S.

- Analyzing the running time of Phase 1 of insertion-sort, we note that it is `􏰎∑n 􏰏 O(1+2+...+(n−1)+n)`
- Phase 1 runs in `O(n^2)` time, and hence, so does the entire insertion-sort algorithm.


- Alternatively, we could change our definition of insertion-sort so that we `insert elements starting from the end of the priority-queue list` in Phase 1, in which case performing insertion-sort on a sequence that is already sorted would run in O(n) time.
- Indeed, the running time of insertion-sort in this case is `O(n+I)`, where I is the number of inversions in the sequence, that is, the number of pairs of elements that start out in the input sequence in the wrong relative order.


---

### Heap-Sort

- priority queue with heap: all the methods in the priority queue ADT run in logarithmic time or better.
  - Hence, this realization is suitable for applications where fast running times are sought for all the priority queue methods.


consider the pqSort scheme using a heap-based implementation of the priority queue.

- During Phase 1, `O(n log n)` time.
  - since the heap has i entries after the operation is performed, the i th **insert** operation takes `O(log i)` time.
  - It could be improved to O(n) with the bottom-up heap construction

- During the second phase of method pqSort
  - the j th **removeMin** operation runs in `O(log(n − j + 1))`, since the heap has `n − j + 1` entries at the time the operation is performed.
  - Summing over all j, this phase takes `O(nlogn)` time,
  - so the entire priority-queue sorting algorithm runs in `O(nlogn)` time when we use a heap to implement the priority queue.
- This sorting algorithm is better known as **heap-sort**


**Proposition 9.4**:
- The heap-sort algorithm sorts a sequence S of n elements in O(n log n) time, assuming two elements of S can be compared in O(1) time.
- the O(n log n) running time of heap-sort is considerably better than the O(n^2) selection-sort and insertion-sort.




---

### Implementing Heap-Sort In-Place

- If the `sequence S` to be sorted is implemented by means of an array-based sequence, such as an ArrayList in Java, we can speed up heap-sort and reduce its space requirement by a constant factor by using a portion of the array itself to store the heap, thus avoiding the use of an auxiliary heap data structure.

This is accomplished by modifying the algorithm as follows:
1. redefine the heap operations to be a maximum-oriented heap
   1. each position key >= its children.
   2. This can be done by recoding the algorithm/providing a new comparator that reverses the outcome of each comparison.
   3. At any time during the execution of the algorithm,
      1. we use the left portion of `S`, up to a certain index i − 1, to store the `entries of the heap`,
      2. and the right portion of `S`, from index i to n − 1, to store the `elements of the sequence`.
   4. Thus, the first i elements of S (at indices 0,...,i−1) provide the array-list representation of the heap.


Implementing
1. In the first phase of the algorithm, we start with an empty heap and move the boundary between the heap and the sequence from left to right, one step at a time.
   1. In step i, for i = 1,...,n, we expand the heap by adding the element at index i − 1.

2. In the second phase of the algorithm, we start with an empty sequence and move the boundary between the heap and the sequence from right to left, one step at a time.
   1. At step i, for i = 1,...,n, we remove a maximal element from the heap and store it at index n − i.


In general, we say that a sorting algorithm is **in-place** if it uses only a `small amount of memory` in addition to the sequence storing the objects to be sorted.
- The variation of heap-sort above qualifies as in-place;
- instead of transferring elements out of the sequence and then back in, we simply rearrange them.


![Screen Shot 2022-04-08 at 01.51.31](https://i.imgur.com/pXjmBUs.png)


---


## Adaptable Priority Queues


The methods of the priority queue ADT are sufficient for most basic applications of priority queues, such as sorting. However, there are situations in which additional methods would be useful


Example: the standby airline passenger application.
- A standby passenger with a pessimistic attitude may become tired of waiting and decide to leave ahead of the boarding time, requesting to be removed from the waiting list.
  - Thus, we would like to remove from the priority queue the entry associated with this passenger.
  - Operation `removeMin` does not suffice since the passenger leaving does not necessarily have first priority.
  - In- stead, we want a new operation, remove, that removes an arbitrary entry.
- Another standby passenger finds her gold frequent-flyer card and shows it to the agent.
  - Thus, her priority has to be modified accordingly.
  - To achieve this change of priority, have a new operation `replaceKey` to replace the key of an existing entry with a new key.
- Finally, a third standby passenger notices her name is misspelled on the ticket and asks it to be corrected.
  - To update the passenger’s record. have a new operation `replaceValue`, allowing us to replace the value of an existing entry with a new value.



### The Adaptable Priority Queue ADT

The above scenarios motivate the definition of a new adaptable priority queue ADT that extends the priority queue ADT with additional functionality.


To implement methods `remove`, `replaceKey`, and `replaceValue` efficiently
- need a mechanism for finding a user’s element within a priority queue, ideally in a way that avoids performing a linear search through the entire collection.
- in priority queue ADT, `insert(k, v)` formally returns an instance of type Entry to the user.
- to be able to update or remove an entry in our new adaptable priority queue ADT, the user must retain that Entry object as a token that can be sent back as a parameter to identify the relevant entry. Formally, the adaptable priority queue ADT includes the following methods:
  - `remove(e)`: Removes entry e from the priority queue.
  - `replaceKey(e, k)`: Replaces the key of existing entry e with k.
  - `replaceValue(e, v)`: Replaces the value of existing entry e with v.


#### Location-Aware Entries

To allow an entry instance to encode a location within a priority queue
- extend the PQEntry class, adding a third field that designates the current index of an entry within the array-based representation of the heap
- When perform priority queue operations on our heap, causing entries to be relocated within structure, make sure to update the third field of each affected entry to reflect its new index within the array.
- example,
  - after a call to removeMin().
  - The heap operation causes the minimal entry, (4,C), to be removed,
  - and the last entry, (16,X), to be temporarily moved from the last position to the root, followed by a down-heap bubble phase.
  - During the down-heap, element (16,X) is swapped with its left child, (5,A), at index 1 of the list, then swapped with its right child, (9,F), at index 4 of the list.
  - In the final configuration, the last field for all affected entries has been modified to reflect their new location.


![Screen Shot 2022-04-08 at 02.05.53](https://i.imgur.com/ZjEu5GT.png)


![Screen Shot 2022-04-08 at 02.03.07](https://i.imgur.com/XGPiRjh.png)


![Screen Shot 2022-04-08 at 02.08.26](https://i.imgur.com/OKkQjxI.png)



#### Implementing an Adaptable Priority Queue


Java implementation of an adaptable priority queue as a subclass of the HeapPriorityQueue class

- defining a nested `AdaptablePQEntry` class extends the inherited PQEntry class, augmenting it with an additional index field.
  - The inherited insert method is overridden, create and initialize an instance of the AdaptablePQEntry class (not the original PQEntry class).

- original `HeapPriorityQueue` class relies exclusively on a protected swap method for all low-level data movement during up-heap or down-heap operations.

- The `AdaptablePriorityQueue` class overrides that utility in order to update the stored indices of our location-aware entries when they are relocated (as discussed on the previous page).

- When an entry is sent as a parameter to remove, replaceKey, or replaceValue, we rely on the new index field of that entry to designate where the element resides in the heap (a fact that is easily validated).
  - When a key of an existing entry is replaced, that new key may violate the heap-order property by being either too big or too small. We provide a new bubble utility that determines whether an up- heap or down-heap bubbling step is warranted. When removing an arbitrary entry, we replace it with the last entry in the heap (to maintain the complete binary tree property) and perform the bubbling step, since the displaced element may have a key that is too large or too small for its new location.

- Performance of Adaptable Priority Queue Implementations
  - The performance of an adaptable priority queue by means of our location-aware heap structure is summarized in Table 9.5. The new class provides the same asymp- totic efficiency and space usage as the nonadaptive version, and provides logarith- mic performance for the new locator-based remove and replaceKey methods, and constant-time performance for the new replaceValuemethod.

  - `size, isEmpty, min`: O(1)
  - `insert`: O(log n)
  - `remove`: O(log n)
  - `removeMin`: O(log n)
  - `replaceKey`: O(log n)
  - `replaceValue`: O(1)


```java
package pq;

import java.util.Comparator;

// /∗∗ An implementation of an adaptable priority queue using an array-based heap. ∗/
public class HeapAdaptablePriorityQueue<K,V> extends HeapPriorityQueue<K,V> implements AdaptablePriorityQueue{

    //---------------- nested AdaptablePQEntry class ----------------
    // /∗∗ Extension of the PQEntry to include location information. ∗/
    protected static class AdaptablePQEntry<K,V> extends PQEntry<K,V> {
        private int index;
        public AdaptablePQEntry(K key, V value, int j){
            super(key, value);
            index = j;
        }
        public int getIndex(){return index;}
        public void setIndex(int j) {index = j;}
    }


    public HeapAdaptablePriorityQueue(){ super(); }
    public HeapAdaptablePriorityQueue(Comparator<K> comp){ super(comp); }

    // protected utilites
    // /∗∗ Validates an entry to ensure it is location-aware. ∗/
    protected AdaptablePQEntry<K,V> validate(Entry<K,V> entry) throws IllegalArgumentException {
        if (!(entry instanceof AdaptablePQEntry)) throw new IllegalArgumentException("Invalid entry"); AdaptablePQEntry<K,V> locator = (AdaptablePQEntry<K,V>) entry;
        int j = locator.getIndex();
        if (j >= heap.size() || heap.get(j) != locator) throw new IllegalArgumentException("Invalid entry");
        return locator;
    }

    // /∗∗ Exchanges the entries at indices i and j of the array list. ∗/
    // safe
    protected void swap(int i, int j) {
        super.swap(i,j);
        ((AdaptablePQEntry<K,V>) heap.get(i)).setIndex(i);
        ((AdaptablePQEntry<K,V>) heap.get(j)).setIndex(j);
    }


    // /∗∗ Restores the heap property by moving the entry at index j upward/downward.∗/
    protected void bubble(int j) {
        if (j > 0 && compare(heap.get(j), heap.get(parent(j))) < 0) upheap(j);
        else downheap(j); // although it might not need to move
    }

    // /∗∗ Inserts a key-value pair and returns the entry created. ∗/
    public Entry<K,V> insert(K key, V value) throws IllegalArgumentException {
        checkKey(key); // might throw an exception
        Entry<K,V> newest = new AdaptablePQEntry<>(key, value, heap.size());
        heap.add(newest); // add to the end of the list
        upheap(heap.size() - 1); // upheap newly added entry
        return newest;
    }


    // /∗∗ Removes the given entry from the priority queue. ∗/
    public void remove(Entry<K,V> entry) throws IllegalArgumentException {
        AdaptablePQEntry<K,V> locator = validate(entry);
        int j = locator.getIndex( );
        if (j == heap.size( ) - 1) heap.remove(heap.size() - 1);
        else {
            swap(j, heap.size( ) - 1);
            heap.remove(heap.size( ) - 1);
            bubble(j);
        }
    }


    // /∗∗ Replaces the key of an entry. ∗/
    public void replaceKey(Entry<K,V> entry, K key) throws IllegalArgumentException {
        AdaptablePQEntry<K,V> locator = validate(entry);
        checkKey(key);
        locator.setKey(key);
        bubble(locator.getIndex());
    }


    // /∗∗ Replaces the value of an entry. ∗/
    public void replaceValue(Entry<K,V> entry, V value) throws IllegalArgumentException {
        AdaptablePQEntry<K,V> locator = validate(entry);
        locator.setValue(value); // method inherited from PQEntry
    }
}
```














.
