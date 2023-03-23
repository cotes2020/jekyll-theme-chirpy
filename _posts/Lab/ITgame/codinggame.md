# CodingGame

Similarities are not intended.

## Puzzles

### The Descent
(https://www.codingame.com/training/easy/the-descent)

```
import sys

#Game loop.
while True:
    max = 0
    maxIndex = -1

    for i in range(8):
        #Read inputs.
        mountainH = int(input())

        #Set highest mountain.
        if mountainH > max:
            max = mountainH
            maxIndex = i

    #Output highest mountain.
    print(maxIndex)
```


#### solution from

Ynoht:

```
while True:
    print(  max(  [(int(input()),x) for x in range(8)]  )    [1])
```

( int( `input()` )  ,x):  [(1, 0), (3, 1), (8, 2), (2, 3), (3, 4), (9, 5), (5, 6), (3, 7)]

print(`(A,B)`[1]): B


Gulzt:

```
while True:
    print(max(range(8), key=lambda _: input()))
```

```
def winner():
    w = max(players, key=lambda p: p.totalScore)
```

---

`lambda`
an anonymous function, it is equivalent to:

```
def func(p):
   return p.totalScore
```

Now max becomes:

max(players, key=func)

But as def statements are compound statements they can't be used where an expression is required, that's why sometimes lambda's are used.

Note that lambda is equivalent to what you'd put in a return statement of a def. Thus, you can't use statements inside a lambda, only expressions are allowed.

---

`max`

max(a, b, c, ...[, key=func]) -> value

With a single iterable argument, return its largest item. With two or more arguments, return the largest argument.

simply returns the object that is the largest.


[![捕获的数据包]](https://wizardforcel.gitbooks.io/daxueba-kali-linux-tutorial/content/ "58")
