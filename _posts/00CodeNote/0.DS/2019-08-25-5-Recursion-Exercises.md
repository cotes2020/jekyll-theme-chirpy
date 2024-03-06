---
title: DS - pythonds3 - 5. Recursion - Exercises
# author: Grace JyL
date: 2019-08-25 11:11:11 -0400
description:
excerpt_separator:
categories: [00CodeNote, PythonNote]
tags:
math: true
# pin: true
toc: true
# image: /assets/img/sample/devices-mockup.png
---

- [DS - pythonds3 - 5. Recursion - Exercises](#ds---pythonds3---5-recursion---exercises)
  - [check the reverse words](#check-the-reverse-words)
  - [check the mirror words](#check-the-mirror-words)
  - [exchange the coins](#exchange-the-coins)
  - [factorial of a number](#factorial-of-a-number)
  - [recursive tree](#recursive-tree)
  - [Fibonacci sequence](#fibonacci-sequence)
  - [water jugs](#water-jugs)
  - [missionaries and cannibals](#missionaries-and-cannibals)
  - [python tree](#python-tree)

---


# DS - pythonds3 - 5. Recursion - Exercises

---

## check the reverse words

```py
# --------------------------------------------------------------------------------------------------
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

# print(reverse("hello")=="olleh")
# print(reverse("l")=="l")
# print(reverse("follow")=="wollof")
# print(reverse("")=="")
```

---


---

## check the mirror words

```py
# --------------------------------------------------------------------------------------------------
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

# print(isPal("x"))
# print(isPal("radar"))
# print(isPal("hello"))
# print(isPal(""))
# print(isPal("hannah"))
# print(isPal(removeWhite("madam i'm adam")))

# testEqual(isPal(removeWhite("x")),True)
# testEqual(isPal(removeWhite("radar")),True)
# testEqual(isPal(removeWhite("hello")),False)
# testEqual(isPal(removeWhite("")),True)
# testEqual(isPal(removeWhite("hannah")),True)
# testEqual(isPal(removeWhite("madam i'm adam")),True)

```

---

## exchange the coins

```py
# -------------------------------------- Exercises -------------------------------------------------

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

# amnt = 63
# clist = [1,5,10,21,25]
# coinsUsed = [0]*(amnt+1)
# coinCount = [0]*(amnt+1)
# print("Making change for",amnt,"requires")
# print(dpMakeChange(clist,amnt,coinCount,coinsUsed),"coins")

# minCoins: change for 0, for 1, for 2 ....
# [0, 1, 2, 3, 4, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 2, 3, 4, 5, 6, 2, 1, 2, 3, 4, 1, 2, 3, 4, 5, 2, 2, 3, 4, 5, 2, 3, 4, 5, 6, 3, 3, 2, 3, 4, 3, 2, 3, 4, 5, 2, 3, 3, 4, 5, 3, 3, 4, 5, 6, 3, 4, 4, 3]

# [1, 1, 1, 1, 1, 5, 1, 1, 1, 1, 10, 1, 1, 1, 1, 5, 1, 1, 1, 1, 10, 21, 1, 1, 1, 25, 1, 1, 1, 1, 5, 10, 1, 1, 1, 10, 1, 1, 1, 1, 5, 10, 21, 1, 1, 10, 21, 1, 1, 1, 25, 1, 10, 1, 1, 5, 10, 1, 1, 1, 10, 1, 10, 21]

# printCoins that walks backward through the table to print out the value of each coin used. This shows the algorithm in action solving the problem for our friends in Lower Elbonia. The first two lines of main set the amount to be converted and create the list of coins used. The next two lines create the lists we need to store the results. coinsUsed is a list of the coins used to make change, and coinCount is the minimum number of coins used to make change for the amount corresponding to the position in the list.
def printCoins(coinsUsed,change):
  coin = change
  while coin > 0:
     thisCoin = coinsUsed[coin]
     print(thisCoin)
     coin = coin - thisCoin

def main():
   amnt = 63
   clist = [1,5,10,21,25]
   coinsUsed = [0]*(amnt+1)
   coinCount = [0]*(amnt+1)

   print("Making change for",amnt,"requires")
   print(dpMakeChange(clist,amnt,coinCount,coinsUsed),"coins")
   print("They are:")
   printCoins(coinsUsed,amnt)
   print("The used list is as follows:")
   print(coinsUsed)

# main()
```

---

## factorial of a number

```py
# -------------------------------------- Exercises -------------------------------------------------
# Write a recursive function to compute the factorial of a number.
# Factorial of a non-negative integer, is multiplication of all integers smaller than or equal to n.
# For example factorial of 6 is 6*5*4*3*2*1 which is 720.

# recursion:
def factorial_recursion(number):
   if number == 1:
       factorial = 1
   else:
       factorial = number * factorial_recursion(number-1)
   print(factorial)
   return factorial

def factorial_recursion(number):
   if number <= 1: return 1
   return number * factorial_recursion(number-1)

def factorial_recursion(number):
   return 1 if (number == 1 or number == 0) else number*factorial_recursion(number-1)

# Iterative Solution: O(n)
# Factorial can also be calculated iteratively as recursion can be costly for large numbers.
# Here we have shown the iterative approach using both for and while loop.
def factorial(n):
   res = 1
   for i in range(2, n+1):
       res *= i
   return res

def factorial(n):
   if(n == 0): return 1
   i = n
   fact = 1
   while(n / i != n):
       fact = fact * i
       i -= 1
   return fact
# print(factorial_recursion(6))


# Factorial of a large number
# not possible to store these many digits even if we use long long int.
# Input : 100
# Output : 933262154439441526816992388562667004-
#          907159682643816214685929638952175999-
#          932299156089414639761565182862536979-
#          208272237582511852109168640000000000-
#          00000000000000

# Input :50
# Output : 3041409320171337804361260816606476884-
#          4377641568960512000000000000
# 1. use an array to store individual digits of the result. The idea is to use basic mathematics for multiplication.

```

---


## recursive tree

![Screen Shot 2021-09-25 at 4.19.43 PM](https://i.imgur.com/3R9u0AH.png)



```py
# -------------------------------------- Exercises -------------------------------------------------
# Modify the recursive tree program using one or all of the following ideas:
# Modify the thickness of the branches so that as the branchLen gets smaller, the line gets thinner.
# Modify the color of the branches so that as the branchLen gets very short it is colored like a leaf.
# Modify the angle used in turning the turtle so that at each branch point the angle is selected at random in some range. For example choose the angle between 15 and 45 degrees. Play around to see what looks good.
# Modify the branchLen recursively so that instead of always subtracting the same amount you subtract a random amount in some range.
# If you implement all of the above ideas you will have a very realistic looking tree.
import turtle
import random
def tree(branchLen,t, wid, color):
   min_len = random.randint(14,17)
   # min_len = 15
   if branchLen > 5:
       angle = random.randint(15,45)
       t.width(wid)
       t.color(color)
       r,g,b = color
       t.forward(branchLen)
       t.right(angle)
       tree(branchLen-min_len, t, wid-5, (r+10,g+20,b+10))
       t.left(angle*2)
       tree(branchLen-min_len, t, wid-5, (r+10,g+20,b+10))
       # back to center
       t.right(angle)
       t.backward(branchLen)

def main():
 t = turtle.Turtle()
 myWin = turtle.Screen()
 myWin.colormode(255)
 t.left(90)
 t.up()
 t.backward(100)
 t.down()
 t.color((50,100,20))
 tree(85, t, 25, (50,100,20))
 myWin.exitonclick()
# main()


```

---



## Fibonacci sequence

```py
# -------------------------------------- Exercises -------------------------------------------------
# Write a recursive function to compute the Fibonacci sequence.
# How does the performance of the recursive function compare to that of an iterative version?

def Fibonacci(number):
   arr = {}
   arr[0] = 0
   arr[1] = 1
   arr[2] = 1
   arr[3] = 3
   arr[4] = 5
   if number in arr.keys():
       return arr[number]
   for i in range(5, number+1):
       arr[i] = i-1 + i-2
   return arr[number]
# print(Fibonacci(8))


```

---


## water jugs


![Screen Shot 2021-09-25 at 10.38.52 PM](https://i.imgur.com/wr2BbqN.png)

```py
# -------------------------------------- Exercises -------------------------------------------------
# Write a program to solve the following problem:
# You have two jugs: a 4-gallon jug and a 3-gallon jug.
# Neither of the jugs have markings on them.
# There is a pump that can be used to fill the jugs with water.
# How can you get exactly two gallons of water in the 4-gallon jug?

# The operations you can perform are:
# Empty a Jug
# Fill a Jug
# Pour water from one jug to the other until one of the jugs is either empty or full.

# m < n
# Solution 1 (Always pour from m liter jug into n liter jug)
# Fill the m litre jug and empty it into n liter jug.
# Whenever the m liter jug becomes empty fill it.
# Whenever the n liter jug becomes full empty it.
# Repeat steps 1,2,3 till either n liter jug or the m liter jug contains d litres of water.

def add_water(big, small, big_size, small_size):
   # Whenever the m liter jug becomes empty fill it.
   if small == 0:
       print("refill small")
       small = small_size
   # Whenever the n liter jug becomes full empty it.
   if big == big_size:
       print("empty big")
       big = 0
   # Fill the m litre jug and empty it into n liter jug.
   if big+small < big_size:
       print("big <- all small")
       big = small
       small = 0
   else:
       print("big <- small, small still have some")
       small = big+small - big_size
       big = big_size
   print("end:", big, small)
   return (big, small)

def jugs(target, jug_list):
   big, small = (0,0)
   big_size, small_size = jug_list
   number_dic = {}

   i = 0
   while target != big:
       print(" ============= step:", i)
       print("start:", big, small)
       big, small = add_water(big, small, big_size, small_size)

       if big not in number_dic.keys():
           number_dic[big] = i
       target_index = i
       i += 1

   for i in number_dic.keys():
       print("for number: ", i, "need step: ", number_dic[i])

   print(" ============= for target: ", target, "need step: ", target_index)

jugs(1, (4, 3))
```

---


## missionaries and cannibals

```py
# -------------------------------------- Exercises -------------------------------------------------
# Write a program that solves the following problem:
# Three missionaries and three cannibals come to a river and find a boat that holds two people.
# Everyone must get across the river to continue on the journey.
# However, if the cannibals ever outnumber > the missionaries on either bank, the missionaries will be eaten.
# Find a series of crossings that will get everyone safely to the other side of the river.

```

---


## python tree

```py

class Tree:
   def __init__(self, data=None):
     self.child = []
     self.parent = None
     self.data = data

   def printTree(self):
       if self.child == []:
           print("root ", self.data, "have no child")
       else:
           root = self
           print("root ", root.data, "have child")
           leaf = {}
           child_1st_list = []
           child_1st = root.child
           for i in child_1st:
               print("1st child:", i.data)
               self.printChild(i)

   def printNode(self, Node):
       print(Node.data)

   def printChild(self, node):
       if node.child == []:
           print(node.data, "has no child")
       else:
           child_list = []
           for i in node.child:
               child_list.append(i.data)
           leaf = {"parent": node.data, "child": child_list}
           print(leaf)

   def insert(self, parent_node, child_node):
       if parent_node.data != None:
           child_node.parent = parent_node
           parent_node.child.append(child_node)
       else:
           print("not tree yet")
           print("setup tree")
           self.data = child_node.data

# t = Tree((0,0))
# # root.printTree()

# child1 = Tree((0,1))
# print(" +++++++ add child 1")
# t.insert(t, child1)
# # t.printTree()
# # t.printChild(t)

# child2 = Tree((0,2))
# print(" +++++++ add child 2")
# t.insert(t, child2)
# # t.printChild(child1)

# child3 = Tree((0,3))
# print(" +++++++ add child 3")
# t.insert(child1, child3)
# t.printChild(child1)

# t.printTree()
```
