---
title: Zipping each of python data structures
date: "2024-12-29T13:10:43+09:00"
categories: [Programming, Python]
tags: [zip]
description: combine multiple iterables (such as lists, tuples, or other sequences) into a single iterable
author: hoon
---

The `zip()` function in Python is used to **combine multiple iterables (such as lists, tuples, or other sequences) into a single iterable**. 

It pairs the elements from each iterable based on their positions (index), creating tuples of corresponding elements. 

It stops when the shortest input iterable is exhausted.

The result of `zip()` is an iterator of tuples, where each tuple contains elements from the input iterables at the same index.

---

## 1. Zipping Lists
```python
list1 = [1, 2, 3]
list2 = ['a', 'b', 'c']

zipped_list = zip(list1, list2)
print(list(zipped_list))  # Output: [(1, 'a'), (2, 'b'), (3, 'c')]
```

---

## 2. Zipping Tuples
```python
tuple1 = (4, 5, 6)
tuple2 = ('x', 'y', 'z')

zipped_tuple = zip(tuple1, tuple2)
print(list(zipped_tuple))  # Output: [(4, 'x'), (5, 'y'), (6, 'z')]
```

---

## 3. Zipping Sets
Sets are unordered, so the results may vary:
```python
set1 = {7, 8, 9}
set2 = {'p', 'q', 'r'}

zipped_set = zip(set1, set2)
print(list(zipped_set))  # Example Output: [(7, 'p'), (8, 'q'), (9, 'r')]
```

---

## 4. Zipping Dictionaries
When zipping dictionaries, only the keys are zipped by default:
```python
dict1 = {'key1': 10, 'key2': 20}
dict2 = {'keyA': 'A', 'keyB': 'B'}

# To zip keys:
zipped_dict_keys = zip(dict1, dict2)
print(list(zipped_dict_keys))  # Output: [('key1', 'keyA'), ('key2', 'keyB')]

# To zip values:
zipped_dict_values = zip(dict1.values(), dict2.values())
print(list(zipped_dict_values))  # Output: [(10, 'A'), (20, 'B')]

# To zip items (key-value pairs):
zipped_dict_items = zip(dict1.items(), dict2.items())
print(list(zipped_dict_items))  # Output: [(('key1', 10), ('keyA', 'A')), (('key2', 20), ('keyB', 'B'))]
```

---

## 5. Zipping Different Data Structures Together
You can mix data structures:
```python
list1 = [1, 2, 3]
tuple1 = ('a', 'b', 'c')
set1 = {100, 200, 300}

zipped_mixed = zip(list1, tuple1, set1)
print(list(zipped_mixed))  # Example Output: [(1, 'a', 100), (2, 'b', 200), (3, 'c', 300)]
```

## 6. Key Notes:
- The `zip()` function stops zipping when the shortest iterable is exhausted.
- If you need a list, tuple, or set as output, wrap the result in the appropriate constructor (e.g., `list(zip(...))`).
