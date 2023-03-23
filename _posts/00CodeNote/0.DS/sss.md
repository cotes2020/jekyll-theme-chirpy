

### LinkedList (array-based structure) (without fixed size) **class**

- an alternative to an array-based structure.

- A linked list, in its simplest form, is a collection of nodes that collectively form a linear sequence.

- An important property of a linked list is that `it does not have a predetermined fixed size`;
- it uses space proportional to its current number of elements.

---


#### basicc

- a collection of items
- each item holds a relative position with respect to the others.
- More specifically, will refer to this type of list as an `unordered list`.
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

#### Abstract Data Type


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

##### Unordered List - Abstract Data Type

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


##### singly linked list

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
- Unfortunately, cannot easily delete the last node of a singly linked list.
- must be able to access the node before the last node in order to remove the last node.
- The only way to access this node is to start from the head of the list and search all the way through the list.
- to support such an operation efficiently, will need to make our list **doubly linked**


---


##### Circularly Linked Lists

- there are many applications in which data can be more naturally viewed as having a cyclic order, with well-defined neighboring relationships, but no fixed beginning or end.

- essentially a singularly linked list, the `next reference of the tail node` is set to refer back to the head of the list (rather than null),

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



##### doubly linked list

- there are limitations that stem from the asymmetry of a singly linked list.
  - can efficiently insert a node at either end of a singly linked list, and can delete a node at the head of a list,
  - cannot efficiently delete a node at the tail of the list.
  - cannot efficiently delete an arbitrary node from an interior position of the list if only given a reference to that node, because cannot determine the node that immediately precedes the node to be deleted (yet, that node needs to have its next reference updated).

![Screen Shot 2022-03-04 at 09.56.42](https://i.imgur.com/dzUHpQI.png)

**doubly linked list**
- a linked list, each node keeps an explicit reference to the node before it and a reference to the node after it.
- These lists allow a greater variety of O(1)-time update operations, including insertions and deletions at arbitrary positions within the list.
- continue to use the term “next” for the reference to the node that follows another, and introduce the term “prev” for the reference to the node that precedes it.


**Header and Trailer Sentinels**
- to avoid some special cases when operating near the boundaries of a doubly linked list, it helps to add special nodes at both ends of the list: a `header` node at the beginning of the list, and a `trailer` node at the end of the list.
- These “dummy” nodes are known as `sentinels/guards`, and they do not store elements of the primary sequence.
- When using sentinel nodes, an empty list is initialized so that the `next field of the header points to the trailer`, and the `prev field of the trailer points to the header`; the remaining fields of the sentinels are irrelevant (presumably null, in Java).
- For a nonempty list, the header’s next will refer to a node containing the first real element of a sequence, just as the trailer’s prev references the node containing the last element of a sequence.


**Advantage of Using Sentinels**
- Although could implement a doubly linked list without sentinel nodes, slight extra memory devoted to the `sentinels greatly simplifies the logic of the operations`.
  - the header and trailer nodes never change — only the nodes between them change.
  - treat all insertions in a unified manner, because a new node will always be placed between a pair of existing nodes.
  - every element that is to be deleted is guaranteed to be stored in a node that has neighbors on each side.
- contrast
  - SinglyLinkedList implementation addLast method required a conditional to manage the special case of inserting into an empty list.
  - In the general case, the new node was linked after the existing tail.
  - But when adding to an empty list, there is no existing tail; instead it is necessary to reassign head to reference the new node.
  - The use of a sentinel node in that implementation would eliminate the special case, as there would always be an existing node (possibly the header) before a new node.


#### general method


##### Equivalence Testing
- At the lowest level, if a and b are reference variables, then` expression a == b tests whether a and b refer to the same object` (or if both are set to the null value).
- higher-level notion of two variables being considered “equivalent” even if they do not actually refer to the same instance of the class. For example, typically want to consider two String instances to be equivalent to each other if they represent the identical sequence of characters.
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
  - two-dimensional arrays in Java are really one-dimensional arrays nested inside a common one-dimensional array raises an interesting issue with respect to how think about compound objects
  - two-dimensional array, b, that has the same entries as a
    - But the one-dimensional arrays, **the rows of a and b are stored in different memory locations**, even though they have the same internal content.
    - Therefore
      - `java.util.Arrays.equals(a,b) == false`
      - `Arrays.deepEquals(a,b) == true`

---

##### Cloning Data Structures

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
int[] data = {2, 3, 5, 7, 11, 13, 17, 19};
int[] backup;

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
Person[] guests = new Person[contacts.length];
for (int k=0; k < contacts.length; k++)
    guests[k] = (Person) contacts[k].clone(); // returns Object type
```

**clone on 2D Arrays**
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

> While the assignment of the size variable is correct, cannot allow the new list to share the same head value (unless it is null).
> For a nonempty list to have an independent state, it must have an entirely new chain of nodes, each storing a reference to the corresponding element from the original list.
> therefore create a new head node, and then perform a walk through the remainder of the original list while creating and linking new nodes for the new list.


---




#### Node Class

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

## create Node objects in the usual way.
>>> temp = Node(93)
>>> temp.data
93
```







---

#### unordered Linked Lists: Unordered List


![idea2](https://i.imgur.com/SqXvGO8.png)


无序表： `unordered list`
- 一种数据按照相对位置存放的数据集
- (for easy, assume that no repeat)
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
  - also be keeping track of the number of nodes have in our linked list. Though this is not strictly necessary, I find it to be more efficient and convenient than iterating through the entire linked list when polling for size.


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

###### Unordered List Class <- unordered linked list (new)  (!!!!!!!!!!!!!)

- 无序表必须要有对第一个节点的引用信息
- 设立属性head，保存对第一个节点的引用空表的head为None
- the unordered list will be built from a collection of nodes, each linked to the next by explicit references.
- As long as know where to find the first node (containing the first item), each item after that can be found by successively following the next links.
- the UnorderedList class must maintain a reference to the first node.
- each `list` object will maintain a single reference to the head of the list.


```py
class UnorderedList:
    def __init__(self):
        self.head = None

## Initially when construct a list, there are no items.
mylist = UnorderedList()
print(mylist.head)
## None
```

---

####### `is_empty()`
- the special reference `None` will again be used to state that the head of the list does not refer to anything.
- Eventually, the example list given earlier will be represented by a linked list as below

![initlinkedlist](https://i.imgur.com/HugjffZ.png)

```py
## checks to see if the head of the list is a reference to None.
## The result of the boolean expression self.head == None will only be true if there are no nodes in the linked list.
def is_empty(self):
    return self.head == None
```

![linkedlist](https://i.imgur.com/t0sWHTx.png)

- The `head` of the list refers to the `first node` which contains the `first item of the list`.
- In turn, that node holds a reference to the next node (the next item) and so on.
- **the list class itself does not contain any node objects**.
- Instead it contains `a single reference to the first node in the linked structure`.

---

####### `add()`
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

####### `size`, `search`, and `remove`
- all based on a technique known as linked list traversal
- Traversal refers to the process of systematically visiting each node.

######## `size()`
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

######## `search(item):`
- Searching for a value in a linked list implementation of an unordered list also uses the traversal technique.
- visit each node in the linked list, ask whether the data matches the item
- may not have to traverse all the way to the end of the list.
  - if get to the end of the list, that means that the item are looking for must not be present.
  - if do find the item, there is no need to continue.

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

######## `remove()`
- requires two logical steps.
- traverse the list for the item to remove.
  - Once find the item , must remove it.
  - If item is not in the list, raise a ValueError.
  - The first step is very similar to search.
    - Starting with an external reference set to the head of the list,
    - traverse the links until discover the item
    - When the item is found, break out of the loop
- use two external references as traverse down the linked list.
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
    if previous is None:   # remove the first item
        self.head = current.next
    else:
        previous.next = current.next
```

---

####### `pop()`

```py
def pop(self, index):
    self.remove(self.getItem(index))
```

---

####### `append()`

```py
## 1. 𝑂(𝑛)
def append(self, item):
    current = self.head
    while current.set_next() is not None:
        current = current.set_next()

    temp = Node(item)
    temp.set_next(current.set_next())
	current.set_next(temp)

## 2. 𝑂(1)
## use tail point & head point
```

---

####### `insert()`

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

####### `index()`

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




#### Ordered List - Abstract Data Type


ordered list
- a collection of items where **each item** holds a `relative position that is based upon some underlying characteristic of the item`.

- The ordering is typically either ascending or descending and assume that list items have a meaningful comparison operation that is already defined.

- Many of the ordered list operations are the same as those of the unordered list.


- For example
  - the list of integers were an ordered list (ascending order),
  - then it could be written as `17, 26, 31, 54, 77, and 93`.
  - Since 17 is the smallest item, it occupies the first position in the list.
  - Likewise, since 93 is the largest, it occupies the last position.


---

###### Ordered List in py (!!!!!!!!!!!!!)

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
        if previous is None:   # remove the first item
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
