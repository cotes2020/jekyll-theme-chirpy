---
title: Python - Data Structure
date: 2020-09-10 11:11:11 -0400
description:
categories: [1CodeNote, PythonNote]
tags:
---

# Python - Data Structure

[toc]

---

## stack

```java
s.is_empty()
s.push('dog')
s.peek()
s.size()
s.is_empty()
s.pop()
```

## queue

```java
q = Queue()
q.enqueue(item)
q.dequeue()
q.is_empty()
q.size()
```

## deque

```java
dq = Deque()
dq.add_front(item)
dq.add_rear(item)
dq.remove_front()
dq.remove_rear()
dq.is_empty()
dq.size()
```


## list

```java
List<Integer> newlist = new ArrayList<>();
newlist.add(i);
newlist.remove(new Integer(i));
newlist.get(0);

l = List()
l.add(item)
l.remove(item)
l.search(item)
l.is_empty()
l.size()
l.append(item)
l.index(item)
l.insert(pos, item)
l.pop()
l.pop(pos)
```

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

---

## ArrayList
- in order
- index

```java
arr.add();

Student[] arr;
arr = new Student[5];

arr[0] = new Student(1,"aman");
arr[1] = new Student(2,"vaibhav");
arr[2] = new Student(3,"shikar");
arr[3] = new Student(4,"dharmesh");
arr[4] = new Student(5,"mohit");
```

---

## linked list

```java
Deque<Node> d = new LinkedList<Node>();
d.offerFirst(root)
d.pollFirst();


FIFO 从上往下垒 上面先走
Queue<Node> queue = new LinkedList<Node>();
q.add()
q.poll()


LIFO  从下往上垒 上面先走
Stack<Node> s = new Stack<Node>();
s.push()
s.pop()
```

---

## HashMap
```java
HashMap<Integer, Integer> ht = new HashMap<>();

Map.Entry<Integer, Integer> entry;
hm.entrySet()
entry.getValue()
entry.getKey()

ht.put(i, ht.getOrDefault(i, 0) + 1);
ht.getOrDefault(i, 0)
ht.get(i)

// Show all hts in hash table.
names = ht.keys();

ht.get("Zara")
```

---

## Set

no duplicate

---

## HashSet

- not in order
- not index

```java
mySet.add();

```

---

# sort

## SelectionSort - keep finding the smallest one

keep finding the smallest one, and put in the `arr[0]`


![Selection-sort-1](https://i.imgur.com/Ok74yxY.png)


```java
public void SelectionSort(int[] num) {
    for (int k=0; k < in.size(); j++) {
        int minIndex = k;
        for (int i=k+1; i < num.length; i++) {
            if (num[minIndex] > num[i]) {
                minIndex = i;
            }
        }
        int temp = num[k];
        num[k] = num[minIndex];
        num[minIndex] = temp;    
    }
}
```



## BubbleSort - if bigger, back

![1_7QsZkfrRGhAu5yxxeDdzsA](https://i.imgur.com/qoeN7kJ.png)

- Worst and Average Case Time Complexity: O(n*n). Worst case occurs when array is reverse sorted.
- Best Case Time Complexity: O(n). Best case occurs when array is already sorted.
- Auxiliary Space: O(1)
- Boundary Cases: Bubble sort takes minimum time (Order of n) when elements are already sorted.
- Sorting In Place: Yes
- Stable: Yes


## InsertionSort - if smaller, forward

![download](https://i.imgur.com/xTvF4Yn.png)


## QuickSort

![download-2](https://i.imgur.com/joPmStj.png)


## MergeSort

![download-3](https://i.imgur.com/f90CphJ.png)








.
