---
title: Packing and Unpacking
date: "2024-12-29T18:52:59+0900"
categories: [Programming, Python]
tags: [zip, packing, unpacking]
description: ''
author: hoon
---
## 1. Multiple assignment


```python
a, b = 1, 2
print(a,b)

# x, y, z = 1, 2
# ValueError: too many values to unpack (expected 3, got 2)
```

    1 2


## 2. Unpacking

In Python, it is possible to unpack the elements of list/tuple/dictionary into distinct variables. 

Since values appear within lists/tuples in a specific order, they are unpacked into variables in the same order:


```python
fruits = ["apple", "banana", "cherry"]

x, y, z = fruits
print(x,y,z)

# If there are values that are not needed then you can use _ to flag them:
_, _, z = fruits
print(z)
```

    apple banana cherry
    cherry



```python
fruits_vegetables = [["apple", "banana"], ["carrot", "potato"]]

[[a, b], [c, d]] = fruits_vegetables
print(a,b,c,d)

[a, [c, d]] = fruits_vegetables
print(a,c,d)

# [[a], [c, d]] = fruits_vegetables
# ValueError: too many values to unpack (expected 1)
```

    apple banana carrot potato
    ['apple', 'banana'] carrot potato



```python
fruits_vegetables = [["apple", "banana", "melon"], ["carrot", "potato", "tomato"]]

[[a, *rest], b] = fruits_vegetables
print(a)
print(rest)
print(b)
```

    apple
    ['banana', 'melon']
    ['carrot', 'potato', 'tomato']



```python
fruits_inventory = {"apple": 6, "banana": 2, "cherry": 3}

x, y, z = fruits_inventory
print(x,y,z)

x, y, z = fruits_inventory.values()
print(x,y,z)

x, y, z = fruits_inventory.items()
print(x,y,z)
```

    apple banana cherry
    6 2 3
    ('apple', 6) ('banana', 2) ('cherry', 3)


## 3. Packing (Unpack first, and then pack again)

Packing is the ability to group multiple values into one list that is assigned to a variable. 

This is useful when you want to unpack values, make changes, and then pack the results back into a variable.

### 3.1. list/tuple with *


```python
fruits = ("apple", "banana", "cherry")
more_fruits = ["orange", "kiwi", "melon", "mango","mango","mango","mango"]

# fruits and more_fruits are unpacked and then their elements are packed into combined_fruits tuple.
combined_fruits = *fruits, *more_fruits
print(combined_fruits)

# into list.
# Note the trailing comma. 
*combined_fruits, = *fruits, *more_fruits
print(combined_fruits)

# A list literal can be used instead, but might not be as readable.
[*combined_fruits] = *fruits, *more_fruits
print(combined_fruits)
```

    ('apple', 'banana', 'cherry', 'orange', 'kiwi', 'melon', 'mango', 'mango', 'mango', 'mango')
    ['apple', 'banana', 'cherry', 'orange', 'kiwi', 'melon', 'mango', 'mango', 'mango', 'mango']
    ['apple', 'banana', 'cherry', 'orange', 'kiwi', 'melon', 'mango', 'mango', 'mango', 'mango']


### 3.2. dictionary with **


```python
fruits_inventory = {"apple": 6, "banana": 2, "cherry": 3}
more_fruits_inventory = {"orange": 4, "kiwi": 1, "melon": 2, "mango": 3, "cherry": 3, "cherry": 3, "cherry": 3}

# fruits_inventory and more_fruits_inventory are unpacked into key-values pairs and combined.
combined_fruits_inventory = {**fruits_inventory, **more_fruits_inventory}

# then the pairs are packed into combined_fruits_inventory
print(combined_fruits_inventory)
```

    {'apple': 6, 'banana': 2, 'cherry': 3, 'orange': 4, 'kiwi': 1, 'melon': 2, 'mango': 3}


## 4. Usage of * and ** with functions

### 4.1. Packing with function parameters

When we don’t know how many arguments need to be passed to a python function, we can use Packing to pack all arguments in a tuple. 

For example, if we preset the no of arguments for a function and pass a wrong no of arguments, it'll throw an error


```python
def my_function(a, b, c):
    print(a,b,c)

# calling function with a wrong no of arguments
# my_function(0, 1, 4, 9)
# TypeError: my_function() takes 3 positional arguments but 4 were given
```

To avoid this, use *args or **kwargs in the function definition. 
- *args is used to pack an arbitrary number of positional (non-keyworded) arguments
- **kwargs is used to pack an arbitrary number of keyword arguments.

#### 4.1.1. *args


```python
# This function is defined to take any number of positional arguments
def my_function(*args):
    print(args)

# Arguments given to the function are packed into a tuple
my_function(1)
my_function(1, 2, 3)
my_function(1, 2, 3, "Hello")

tuple1 = (1, 2, 3)
my_function(tuple1, 1, 2, 3)
my_function(*tuple1,*tuple1)
```

    (1,)
    (1, 2, 3)
    (1, 2, 3, 'Hello')
    ((1, 2, 3), 1, 2, 3)
    (1, 2, 3, 1, 2, 3)


#### 4.1.2. **kwargs

This way the function will receive a dictionary of arguments, and can access the items accordingly


```python
# This function is defined to take any number of keyword arguments
def my_function(**kwargs):
    print(kwargs)
    for k,v in kwargs.items():
        print("key = " + str(k) + ", value = " + str(v))

# Arguments given to the function are packed into a dictionary
result = my_function(a=1, b=2, c=3)
result
```

    {'a': 1, 'b': 2, 'c': 3}
    key = a, value = 1
    key = b, value = 2
    key = c, value = 3


different approach


```python
dict1 = {'a':1, 'b':2, 'c':3}
result = {}

for k,v in dict1.items():
    result[k]=v

print(result)
```

    {'a': 1, 'b': 2, 'c': 3}


#### 4.1.3. Combination of *args, **kwargs


```python
def my_function(a, b, *args, **kwargs):
    print("Normal arguments:", a,b)
    print("Positional arguments:", args)
    print("Keyword arguments:", kwargs)

my_function('hello', 'world', 1, 2, 3, x=4, y=5)
```

    Normal arguments: hello world
    Positional arguments: (1, 2, 3)
    Keyword arguments: {'x': 4, 'y': 5}


### 4.2. Unpacking into function calls

What if a function does not accept an iterable?

Example of that is in the [4.1. Packing with function parameters](#-41.-Packing-with-function-parameters)

In this case, We can unpack the iterable with * operator to match the no of arguments that the function requires.


```python
def my_function(a, b, c):
    print(a,b,c)

my_function(1, 2, 3)

numbers = [1, 2, 3]
my_function(*numbers)

numbers = {'a':1,'b':2,'c':3}
my_function(**numbers)

# my_function(numbers)
# my_function() missing 2 required positional arguments: 'b' and 'c'
```

    1 2 3
    1 2 3


#### 4.2.1. Using * unpacking with the zip() function


```python
values = (['x', 'y', 'z'], [1, 2, 3], [True, False, True])
```


```python
a,b,c = zip(*values)
print(a)
print(b)
print(c)
```

    ('x', 1, True)
    ('y', 2, False)
    ('z', 3, True)



```python
a,*rest = zip(*values)
print(a)
print(rest)
```

    ('x', 1, True)
    [('y', 2, False), ('z', 3, True)]

