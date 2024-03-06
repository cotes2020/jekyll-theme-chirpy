---
title: DS - pythonds3 - 1. Introduction
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

- [DS - pythonds3 - 1. Introduction](#ds---pythonds3---1-introduction)
  - [1.5. Why Study Data Structures and Abstract Data Types?](#15-why-study-data-structures-and-abstract-data-types)
  - [1.8. Getting Started with Data](#18-getting-started-with-data)
    - [1.8.1. Built-in Atomic Data Types](#181-built-in-atomic-data-types)
    - [1.8.2. Built-in Collection Data Types](#182-built-in-collection-data-types)
  - [1.9. Input and Output](#19-input-and-output)
    - [1.9.0. input](#190-input)
    - [1.9.1. String Formatting](#191-string-formatting)
  - [1.10. Control Structures](#110-control-structures)
  - [1.13. Object-Oriented Programming in Python: Defining Classes](#113-object-oriented-programming-in-python-defining-classes)
    - [1.13.1. A Fraction Class](#1131-a-fraction-class)
    - [1.13.2. Inheritance: Logic Gates and Circuits](#1132-inheritance-logic-gates-and-circuits)
      - [1. build a representation for logic gates.](#1-build-a-representation-for-logic-gates)
      - [2. have the basic gates working, now build circuits.](#2-have-the-basic-gates-working-now-build-circuits)
  - [1.14. Summary](#114-summary)

---

# DS - pythonds3 - 1. Introduction

Problem Solving with Algorithms and Data Structures using Python 1


## 1.5. Why Study Data Structures and Abstract Data Types?

1. procedural abstraction:
   - process that hides the details of a particular function to allow the user or client to view it at a very high level.
2. data abstraction:
   - An abstract data type, ADT, a logical description of how we view the data and the operations that are allowed without regard to how they will be implemented.
   - concerned only with what the data is representing and not with how it will eventually be constructed
   - providing this level of abstraction, creating an `encapsulation` around the data.
3. information hiding
   - by encapsulating the details of the implementation, we are hiding them from the user’s view.

- The implementation of an abstract data type, `data structure`, will require provide a **physical view of the data** using some **collection of programming constructs and primitive data types**.
- the separation of these two perspectives define the complex data models for problems without giving any indication as to the details of how the model will actually be built.
- This provides an implementation-independent view of the data.
- Since there will usually be many different ways to implement an abstract data type, this implementation independence allows the programmer to switch the details of the implementation without changing the way the user of the data interacts with it. The user can remain focused on the problem-solving process.


a | b
---|---
抽象 | 实现
逻辑 | 物理
接口 | 实现 implement

<kbd>算法+数据结果=编程</kbd>

build in function: procedural abstraction

primitive data typr: int, str....

ADT: abstract data type: 抽象数据 (user descript / cover the data)

![Screen Shot 2020-05-25 at 23.25.42](https://i.imgur.com/cJRDWnA.png)

逻辑层面稳定，操作接口不一样，物理不一样

```
coding:
- c
  - compile
  - link
  - execute
- python
  - run
```

---

## 1.8. Getting Started with Data

### 1.8.1. Built-in Atomic Data Types

### 1.8.2. Built-in Collection Data Types

1. `Lists`  **mutable**

```py
>>> my_list = [1, 2, 3, 4]
>>> big_list = [my_list] * 3
# [[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]]

>>> (54).__add__(21)
75
# asking the integer object 54 to execute its add method (called __add__ in Python) and passing it 21 as the value to add.
```

Method Name | Use | Explanation
---|---|---
append | a_list.append(item) | Adds item to the end
insert | a_list.insert(i,item) | Inserts an item at the ith position in a list
pop | a_list.pop() | Removes and returns the last item in a list
pop | a_list.pop(i) | Removes and returns the ith item in a list
sort | a_list.sort() | Modifies a list to be sorted
reverse | a_list.reverse() | Modifies a list to be in reverse order
del | `del a_list[i]` | Deletes the item in the ith position
index | a_list.index(item) | Returns the index of the first occurrence of item
count | a_list.count(item) | Returns the number of occurrences of item
remove | a_list.remove(item) | Removes the first occurrence of item


2. `Strings`  **immutable**

Method Name | Use | Explanation
---|---|---
center | a_string.center(w) | Returns a string centered in a field of size w
count | a_string.count(item) | Returns the number of occurrences of item in the string
ljust | a_string.ljust(w) | Returns a string left-justified in a field of size w
rjust | a_string.rjust(w) | Returns a string right-justified in a field of size w
lower | a_string.lower() | Returns a string in all lowercase
find | `a_string.find(item)` | Returns the index of the first occurrence of item
split | `a_string.split(s_char)` | Splits a string into substrings at s_char

3. `Tuples` **immutable**

```py
>>> my_tuple = (2, True, 4.96)
>>> my_tuple[0]
2
>>> my_tuple * 3
(2, True, 4.96, 2, True, 4.96, 2, True, 4.96)
```

4. `set`  **unordered, non-duplicates**

```py
>>> my_set = {3, 6, "cat", 4.5, False}
>>> my_set
{False, 3, 4.5, 6, 'cat'}
```

Operation Name | Operator | Explanation
---|---|---
membership `in` | "dog" in my_set | Set membership
length | len | Returns the cardinality of the set
`|` | `a_set | other_set` | Returns a new set with all elements from both sets
`&` | `a_set & other_set` | Returns a new set with only those elements common to both sets
`-` | `a_set - other_set` | Returns a new set with all items from the first set not in second
`<=` | `a_set <= other_set` | Asks whether all elements of the first set are in the second


Method Name | Use | Explanation
---|---|---
union | `a_set.union(other_set)` | Returns a new set with all elements from both sets
intersection | a_set.intersection(other_set) | Returns a new set with only those elements common to both sets
difference | a_set.difference(other_set) | Returns a new set with all items from first set not in second
issubset | a_set.issubset(othe_rset) | Asks whether all elements of one set are in the other
add | a_set.add(item) | Adds item to the set
remove | a_set.remove(item) | Removes item from the set
pop | a_set.pop() | Removes an arbitrary element from the set
clear | a_set.clear() | Removes all elements from the set

5. `dictionary`

```py
>>> capitals = {"Iowa": "Des Moines", "Wisconsin": "Madison"}
>>> capitals
{'Iowa': 'Des Moines', 'Wisconsin': 'Madison'}
```

Operator | Use | Explanation
---|---|---
`[]` | `a_dict[k]` | Returns the value associated with k, otherwise its an error
in | `key in a_dict` | Returns True if key is in the dictionary, False otherwise
del | `del a_dict[key]` | Removes the entry from the dictionary


Method Name | Use | Explanation
---|---|---
keys | `a_dict.keys()` | Returns the **keys of the dictionary** in a dict_keys object
values | `a_dict.values()` | Returns the **values of the dictionary** in a dict_values object
items | `a_dict.items()` | Returns the **key-value pairs** in a dict_items object
get | `a_dict.get(k)` | Returns the value associated with k, None otherwise
get | `a_dict.get(k, alt)` | Returns the value associated with k, alt otherwise


6. `list comprehension`

```py
>>> sq_list=[x * x for x in range(1,11) if x % 2 != 0]
# [1, 9, 25, 49, 81]
>>>[ch.upper() for ch in 'comprehension' if ch not in 'aeiou']
# ['C', 'M', 'P', 'R', 'H', 'N', 'S', 'N']
```

---

## 1.9. Input and Output

### 1.9.0. input

```py
a_name = input("Please enter your name ")
print("Your name in all capitals is",a_name.upper(),
      "and has length", len(a_name))
```

### 1.9.1. String Formatting

```py
>>> print("%s is %d years old." % (a_name, age))
>>> print("The {:s} costs {:d} cents".format(item, price))
>>> print(f"The {item:10} costs {price:10.2f} cents")
```


Character | Output Format
---|---
d, i | Integer
u | Unsigned integer
f | Floating point as m.ddddd
e | Floating point as m.ddddde+/-xx
E | Floating point as m.dddddE+/-xx
g | Use `%e` for exponents less than −4 or greater than +5, otherwise use `%f`
c | Single character
s | String, or any Python data object that can be converted to a string by using the str function.
% | Insert a literal % character

```py
>>> print("Hello")
# Hello
>>> print("Hello", "World")
# Hello World
>>> print("Hello", "World", sep="***")
# Hello***World
>>> print("Hello", "World", end="***")
# Hello World***

>>> print("%s is %d years old." % (a_name, age))
```


Modifier | Example | Description
---|---|---
`number` | `%20d` | Put the value in a field width of 20
`-` | `%-20d` | Put the value in a field 20 characters wide, left-justified
`+` | `%+20d` | Put the value in a field 20 characters wide, right-justified
`0` | `%020d` | Put the value in a field 20 characters wide, fill in with leading zeros.
`.` | `%20.2f` | Put the value in a field 20 characters wide with 2 characters to the right of the decimal point.
`(name)` | `%(name)d` | Get the value from the supplied dictionary using name as the key.


```py
>>> price = 24
>>> item = "banana"

>>> print("The %+10s costs %10.2f cents" % (item, price))
# The     banana costs      24.00 cents

>>> itemdict = {"item": "banana", "cost": 24}
>>> print("The %(item)s costs %(cost)7.1f cents" % itemdict)
# The banana costs    24.0 cents

>>> print("The {} costs {} cents".format(item, price))
# The banana costs 24 cents
>>> print("The {:s} costs {:d} cents".format(item, price))
# The banana costs 24 cents
```

Modifier | Example | Description
---|---|---
`number` | `:20d` | Put the value in a field width of 20
`<` | `:<20d` | Put the value in a field 20 characters wide, left-aligned
`>` | `:>20d` | Put the value in a field 20 characters wide, right-aligned
`^` | `:^20d` | Put the value in a field 20 characters wide, middle-aligned
`0` | `:020d` | Put the value in a field 20 characters wide, fill in with leading zeros.
`.` | `:20.2f` | Put the value in a field 20 characters wide with 2 characters to the right of the decimal point.


```py
>>> price = 24
>>> item = "banana"

>>> print(f"The {item:10} costs {price:10.2f} cents")
# The banana     costs      24.00 cents

>>> print(f"The {item:<10} costs {price:<10.2f} cents")
# The banana     costs 24.00      cents
>>> print(f"The {item:^10} costs {price:^10.2f} cents")
# The   banana   costs   24.00    cents
>>> print(f"The {item:>10} costs {price:>10.2f} cents")
# The     banana costs      24.00 cents

>>> print(f"The {item:>10} costs {price:>010.2f} cents")
The     banana costs 0000024.00 cents

>>> itemdict = {"item": "banana", "price": 24}
>>> print(f"Item:{itemdict['item']:.>10}\n" + f"Price:{'$':.>4}{itemdict['price']:5.2f}")
Item:....banana
Price:...$24.00
```

---

## 1.10. Control Structures


```py
while counter <= 10 and not done:

>>> for item in [1, 3, 6, 2, 5]:
      print(item)

>>> if n < 0:
      print("Sorry, value is negative")
    elif:
      print(math.sqrt(n))
    else:
      print("nothing")

```

---

## 1.13. Object-Oriented Programming in Python: Defining Classes
One of the most powerful features in an object-oriented programming language is the ability to allow a programmer (problem solver) to create new classes that model data that is needed to solve the problem.

use `abstract data types` to provide the `logical description` of what a data object looks like (its state) and what it can do (its methods).
- By building a `class` that implements an abstract data type, a programmer can take advantage of the abstraction process and at the same time provide the details necessary to actually use the abstraction in a program.
- Whenever we want to implement an abstract data type, we will do so with a new class.

---

### 1.13.1. A Fraction Class

The first method that all classes should provide is the `constructor`.
- The `constructor` defines the way in which data objects are created.

```py
class Fraction:
    """Class Fraction"""
    def __init__(self, top, bottom):
        """Constructor definition"""
        self.num = top
        self.den = bottom

# To create an instance of the Fraction class, must invoke the constructor.
# - using the name of the class and passing actual values for the necessary state
# - (note that we never directly invoke __init__). For example,
my_fraction = Fraction(3, 5)
# creates an object called my_fraction representing the fraction 35 (three-fifths). Figure 5 shows this object as it is now implemented.
```

1. To implement the behavior that the abstract data type requires.

1.1. To begin, consider what happens when we try to print a Fraction object.

```py
>>> my_fraction = Fraction(3, 5)
>>> print(my_fraction)
<__main__.Fraction object at 0x103203eb8>
>>>
# The Fraction object, my_fraction, does not know how to respond to this request to print.
# The print function requires that the object convert itself into a string so that the string can be written to the output.
# The only choice my_fraction has is to show the actual reference that is stored in the variable (the address itself). This is not what we want.

------------------------------------------

# 2 ways to solve this problem.

# 1. define a method called show, allow the Fraction object to print itself as a string.
def show(self):
        print(f"{self.num}/{self.den}")

>>> my_fraction = Fraction(3, 5)
>>> my_fraction.show()
3/5
>>> print(my_fraction)
<__main__.Fraction object at 0x40bce9ac>
>>>

# 2. all classes have a set of standard methods,
# __str__: the method to convert an object into a string.
# The default implementation for this method is to return the instance address string as we have already seen.
# define a method with the name __str__ and give it a new implementation
# the method will build a string representation by converting each piece of internal state data to a string and then placing a / character in between the strings using string concatenation.
# The resulting string will be returned any time a Fraction object is asked to convert itself to a string.
def __str__(self):
    return f"{self.num}/{self.den}"

>>> my_fraction = Fraction(3, 5)
>>> print(my_fraction)
3/5
>>> my_fraction.__str__()
'3/5'
>>> str(my_fraction)
'3/5'
```

2. create two Fraction objects and then add them together using the standard “+” notation.

```py
>>> f1 = Fraction(1, 4)
>>> f2 = Fraction(1, 2)
>>> f1 + f2
Traceback (most recent call last):
File "<stdin>", line 1, in <module>
TypeError: unsupported operand type(s) for +: 'Fraction' and 'Fraction'
>>>
# If you look closely at the error, you see that the problem is that the “+” operator does not understand the Fraction operands.
# providing the Fraction class with a method that overrides the addition method.
# In Python, this method is called __add__ and it requires two parameters.
def __add__(self, other_fraction):
     new_num = self.num * other_fraction.den + self.den * other_fraction.num
     new_den = self.den * other_fraction.den
     return Fraction(new_num, new_den)
>>> f1 = Fraction(1, 4)
>>> f2 = Fraction(1, 2)
>>> f3 = f1 + f2
>>> print(f3)
6/8

# 6/8 is the correct result (14+12) but that it is not in the “lowest terms” representation.
# The best representation would be 3/4.
# to sure the results are always in the lowest terms, need to reduce fractions,
# look for the greatest common divisor, or GCD.
# then divide the numerator and the denominator by the GCD and the result will be reduced to lowest terms.
# Euclid’s Algorithm
# the greatest common divisor of two integers 𝑚 and 𝑛 is 𝑛 if 𝑛 divides 𝑚 evenly. However, if 𝑛 does not divide 𝑚 evenly, then the answer is the greatest common divisor of 𝑛 and the remainder of 𝑚 divided by 𝑛.
# only works when the denominator is positive. This is acceptable for our fraction class because we have said that a negative fraction will be represented by a negative numerator.

def gcd(m, n):
   while m % n != 0:
      m, n = n, m % n
   return n
print(gcd(20, 10))
10

def __add__(self, other_fraction):
    new_num = self.num * other_fraction.den + self.den*other_fraction.num
    new_den = self.den * other_fraction.den
    common = gcd(new_num, new_den)
    return Fraction(new_num // common, new_den // common)
>>> f1 = Fraction(1, 4)
>>> f2 = Fraction(1, 2)
>>> f3 = f1 + f2
>>> print(f3)
3/4
```

1. allow two fractions to compare themselves to one another

![fraction3](https://i.imgur.com/slSOGiw.png)

```py
# two Fraction objects, f1 and f2.


# shallow equality
# f1==f2 will only be True if they are references to the same object.
# Two different objects with the same numerators and denominators would not be equal under this implementation.


# deep equality –equality by the same value, not the same reference–by overriding the __eq__ method.
# The __eq__ method is another standard method available in any class.
# The __eq__ method compares two objects and returns True if their values are the same, False otherwise.
# implement the __eq__ method by again putting the two fractions in common terms and then comparing the numerators
# other relational operators that can be overridden:
# For example, the __le__ method provides the less than or equal functionality.
def __eq__(self, other_fraction):
    first_num = self.num * other_fraction.den
    second_num = other_fraction.num * self.den
    return first_num == second_num
```

---

### 1.13.2. Inheritance: Logic Gates and Circuits
Inheritance is the ability for one class to be related to another class in much the same way that people can be related to one another.
- Python child classes can inherit characteristic data and behavior from a parent class.
- These classes are often referred to as `subclasses` and `superclasses`.


![inheritance1](https://i.imgur.com/grFDDxH.png)
> The built-in Python collections and their relationships to one another.

**inheritance hierarchy**: relationship structure such as this
- the list is a child of the sequential collection.
- list is the child and the sequence the parent (or subclass list and superclass sequence).
- an `IS-A Relationship` (the list IS-A sequential collection).
- implies that `lists inherit important characteristics from sequences`, namely the ordering of the underlying data and operations such as concatenation, repetition, and indexing.

`Lists, tuples, and strings` are all types of **sequential collections**.
- They all inherit common data organization and operations.
- However, each of them is distinct based on whether the data is homogeneous and whether the collection is immutable.
- The children all gain from their parents but distinguish themselves by adding additional characteristics.

By organizing classes in this hierarchical fashion, `object-oriented programming languages` allow **previously written code** to be extended to meet the needs of a **new situation**.

by organizing data in this `hierarchical manner`, can better understand the relationships and more efficient in building abstract representations.

```py
# construct a simulation,
# an application to simulate digital circuits.
# The basic building block for this simulation will be the logic gate. These electronic switches represent boolean algebra relationships between their input and their output. The value of the output is dependent on the values given on the input lines.

# AND gates: only 1 and 1 = 1
# OR gates: has 1 = 1
# NOT gates: The output value is simply the opposite of the input value.

# Each gate has a truth table of values showing the input-to-output mapping that is performed by the gate.
```

top of the hierarchy, the LogicGate class
- represents the most general characteristics of logic gates: namely, a label for the gate and an output line.

The next level of subclasses
- breaks the logic gates into two families, those that have one input line and those that have two. Below that, the specific logic functions of each appear.

![gates](https://i.imgur.com/CTqVJGo.png)


#### 1. build a representation for logic gates.

```py
# LogicGate.
# each gate has a label for identification and a single output line.
# In addition, we need methods to allow a user of a gate to ask the gate for its label.
# The other behavior that every logic gate needs is the ability to know its output value. This will require that the gate perform the appropriate logic based on the current input. In order to produce output, the gate needs to know specifically what that logic is. This means calling a method to perform the logic computation.

class LogicGate:
    def __init__(self, lbl):
        self.label = lbl
        self.output = None

    def get_label(self):
        return self.label

    def get_output(self):
        self.output = self.perform_gate_logic()
        return self.output


# the perform_gate_logic function details will be included by each individual gate that is added to the hierarchy.

# The parameter self is a reference to the actual gate object invoking the method. Any new logic gate that gets added to the hierarchy will simply need to implement the perform_gate_logic function and it will be used at the appropriate time. Once done, the gate can provide its output value. This ability to extend a hierarchy that currently exists and provide the specific functions that the hierarchy needs to use the new class is extremely important for reusing existing code.

# We categorized the logic gates based on the number of input lines.
# - The BinaryGate class will be a subclass of LogicGate and will add two input lines.
# - The UnaryGate class will also subclass LogicGate but will have only a single input line.
# - In computer circuit design, these lines are sometimes called “pins”

class BinaryGate(LogicGate):
    def __init__(self, lbl):
        LogicGate.__init__(self, lbl)
        self.pin_a = None
        self.pin_b = None

    def get_pin_a(self):
        return int(input(f"Enter pin A input for gate {self.get_label()}: "))

    def get_pin_b(self):
        return int(input(f"Enter pin B input for gate {self.get_label()}: "))

class UnaryGate(LogicGate):
    def __init__(self, lbl):
        LogicGate.__init__(self, lbl)
        self.pin = None

    def get_pin(self):
        return int(input(f"Enter pin input for gate {self.get_label()}: "))

# The constructors in both classes start with an explicit call to the constructor of the parent class, using the parent’s __init__ method.
# When creating an instance of the BinaryGate class, we first want to initialize any data items that are inherited from LogicGate.
# Child class constructors need to call parent class constructors and then move on to their own distinguishing data.


# Python has a function super
# can be used in place of explicitly naming the parent class.
# more general mechanism, and is widely used, especially when a class has more than one parent.
# In our example above
LogicGate.__init__(self, lbl)
super().__init__(lbl),
super(UnaryGate, self).__init__(lbl)
super().__init__("UnaryGate", lbl).


# Now that we have a general class for gates depending on the number of input lines, build specific gates that have unique behavior.
# AndGate class
# subclass of BinaryGate
# the first line of the constructor calls upon the parent class constructor (BinaryGate), which in turn calls its parent class constructor (LogicGate).
# Note that the AndGate class does not provide any new data since it inherits two input lines, one output line, and a label.


class AndGate(BinaryGate):
    def __init__(self, lbl):
        super().__init__(lbl)

    def perform_gate_logic(self):
        a = self.get_pin_a()
        b = self.get_pin_b()
        if a == 1 and b == 1:
            return 1
        else:
            return 0

# The only thing AndGate needs to add is the specific behavior that performs the boolean operation that was described earlier.
# This is the place where we can provide the perform_gate_logic method.

# create an AndGate object, g1, that has an internal label "G1".
# When invoke the get_output method, the object must first call its perform_gate_logic method which in turn queries the two input lines.
>>> g1 = AndGate("G1")
>>> g1.get_output()
Enter pin A input for gate G1: 1
Enter pin B input for gate G1: 0


>>> g2 = OrGate("G2")
>>> g2.get_output()
Enter pin A input for gate G2: 1
Enter pin B input for gate G2: 1
1
>>> g2.get_output()
Enter pin A input for gate G2: 0
Enter pin B input for gate G2: 0
0

>>> g3 = NotGate("G3")
>>> g3.get_output()
Enter pin input for gate G3: 0
```

#### 2. have the basic gates working, now build circuits.
- to create a circuit, need to connect gates together, the output of one flowing into the input of another.
- To do this, implement a new class called `Connector`.

> The `Connector` class will not reside in the **gate hierarchy**.
>
> It will use the **gate hierarchy** in that each `connector` will have two gates, one on either end.

This relationship: `HAS-A Relationship`.
- `“IS-A Relationship”`:
  - a child class is related to a parent class,
  - for example
    - `UnaryGate` **IS-A** `LogicGate`.
- `HAS-A Relationship`:
  - Now, with the Connector class,
    - a `Connector` **HAS-A** `LogicGate`
    - connectors will have instances of the LogicGate class within them
    - but are not part of the hierarchy.

> When designing classes, it is very important to distinguish between those that have the `IS-A relationship (which requires inheritance)` and those that have `HAS-A relationships (with no inheritance)`.

![circuit1](https://i.imgur.com/gNLPAqZ.png)

```py
# the Connector class.
# The two gate instances within each connector object will be referred to as the from_gate and the to_gate
# data values will “flow” from the output of one gate into input line of the next.
# The call to set_next_pin is very important for making connections.
# add this method to gate classes so that each to_gate can choose the proper input line for the connection.

class Connector:
    def __init__(self, fgate, tgate):
        self.from_gate = fgate
        self.to_gate = tgate
        tgate.set_next_pin(self)

    def get_from(self):
        return self.from_gate

    def get_to(self):
        return self.to_gate

# In the BinaryGate class, for gates with two possible input lines, the connector must be connected to only one line. If both of them are available, we will choose pin_a by default. If pin_a is already connected, then we will choose pin_b. It is not possible to connect to a gate with no available input lines.

def set_next_pin(self, source):
    if self.pin_a == None:
        self.pin_a = source
    else:
        if self.pin_b == None:
            self.pin_b = source
        else:
            raise RuntimeError("Error: NO EMPTY PINS")

# Now it is possible to get input from two places:
# externally, as before, and from the output of a gate that is connected to that input line. This requires a change to the get_pin_a and get_pin_b methods.
# - If the input line is nothing (None), ask the user externally as before.
# - if there is a connection, the connection is accessed and from_gate’s output value is retrieved.
# This in turn causes that gate to process its logic. This continues until all input is available and the final output value becomes the required input for the gate in question. In a sense, the circuit works backwards to find the input necessary to finally produce output.

def get_pin_a(self):
    if self.pin_a == None:
        return input(f"Enter pin A input for gate {self.get_label()}:")
    else:
        return self.pin_a.get_from().get_output()
```

3. The following fragment constructs the circuit

```py
>>> g1 = AndGate("G1")
>>> g2 = AndGate("G2")
>>> g3 = OrGate("G3")
>>> g4 = NotGate("G4")
>>> c1 = Connector(g1, g3)
>>> c2 = Connector(g2, g3)
>>> c3 = Connector(g3, g4)

# The outputs from the two AND gates (g1 and g2) are connected to the OR gate (g3) and that output is connected to the NOT gate (g4). The output from the NOT gate is the output of the entire circuit. For example:

>>> g4.get_output()
Enter pin A input for gate G1: 0
Enter pin B input for gate G1: 1
Enter pin A input for gate G2: 1
Enter pin B input for gate G2: 1
0
```

ActiveCode:

```py

# -------------------------------------------------
class LogicGate:
    def __init__(self, lbl):
        self.name = lbl
        self.output = None

    def get_label(self):
        return self.name

    def get_output(self):
        self.output = self.perform_gate_logic()
        return self.output


# -------------------------------------------------
class BinaryGate(LogicGate):
    def __init__(self, lbl):
        super(BinaryGate, self).__init__(lbl)
        self.pin_a = None
        self.pin_b = None

    def get_pin_a(self):
        if self.pin_a == None:
            return int(input("Enter pin A input for gate " + self.get_label() + ": "))
        else:
            return self.pin_a.get_from().get_output()

    def get_pin_b(self):
        if self.pin_b == None:
            return int(input("Enter pin B input for gate " + self.get_label() + ": "))
        else:
            return self.pin_b.get_from().get_output()

    def set_next_pin(self, source):
        if self.pin_a == None:
            self.pin_a = source
        else:
            if self.pin_b == None:
                self.pin_b = source
            else:
                print("Cannot Connect: NO EMPTY PINS on this gate")


# -------------------------------------------------
class AndGate(BinaryGate):
    def __init__(self, lbl):
        BinaryGate.__init__(self, lbl)

    def perform_gate_logic(self):
        a = self.get_pin_a()
        b = self.get_pin_b()
        if a == 1 and b == 1:
            return 1
        else:
            return 0

class OrGate(BinaryGate):
    def __init__(self, lbl):
        BinaryGate.__init__(self, lbl)

    def perform_gate_logic(self):
        a = self.get_pin_a()
        b = self.get_pin_b()
        if a == 1 or b == 1:
            return 1
        else:
            return 0

class NotGate(UnaryGate):
    def __init__(self, nlbl):
        UnaryGate.__init__(self, lbl)

    def perform_gate_logic(self):
        if self.get_pin():
            return 0
        else:
            return 1


# -------------------------------------------------
class UnaryGate(LogicGate):
    def __init__(self, lbl):
        LogicGate.__init__(self, lbl)
        self.pin = None

    def get_pin(self):
        if self.pin == None:
            return int(input("Enter pin input for gate " + self.get_label() + ": "))
        else:
            return self.pin.get_from().get_output()

    def set_next_pin(self, source):
        if self.pin == None:
            self.pin = source
        else:
            print("Cannot Connect: NO EMPTY PINS on this gate")


# -------------------------------------------------
class Connector:
    def __init__(self, fgate, tgate):
        self.from_gate = fgate
        self.to_gate = tgate
        tgate.set_next_pin(self)

    def get_from(self):
        return self.from_gate

    def get_to(self):
        return self.to_gate


# -------------------------------------------------
def main():
    g1 = AndGate("G1")
    g2 = AndGate("G2")
    g3 = OrGate("G3")
    g4 = NotGate("G4")
    c1 = Connector(g1, g3)
    c2 = Connector(g2, g3)
    c3 = Connector(g3, g4)
    print(g4.get_output())

main()

```

## 1.14. Summary
- Computer science is the study of problem solving.
- Computer science uses abstraction as a tool for representing both processes and data.
- Abstract data types allow programmers to manage the complexity of a problem domain by hiding the details of the data.
- Python is a powerful, yet easy-to-use, object-oriented language.
- `Lists, tuples, and strings` are built in Python sequential collections.
- `Dictionaries` and `sets` are nonsequential collections of data.
- `Classes` allow programmers to implement abstract data types.
- Programmers can override standard methods as well as create new `methods`.
- Classes can be organized into `hierarchies`.
- A class `constructor` should always invoke the constructor of its parent before continuing on with its own data and behavior.


---


















.
