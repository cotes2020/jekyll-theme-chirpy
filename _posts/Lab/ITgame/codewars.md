
# solution

| No. | Puzzle Name             | From     | State                       | Difficulty | Link                                                                                                                                                                                                                                                                                                                                                                                                 |
|-----|-------------------------|----------|-----------------------------|------------|---------------------------------------------------------------------|
| 1   | Duplicate Encoder       | obnounce | ✔️Completed | 6          | https://www.codewars.com/kata/54b42f9314d9229fd6000d9c/train/python                                                                                                                                                                                                                                              |
| 2   | count_sheeps            | obnounce | ✔️Completed | 8          | https://www.codewars.com/kata/54edbc7200b811e956000556                                                                                                                                                                                                                                                                                |
| 2   | Persistent Bugger.      | obnounce | :x: Not Completed           | 6          | https://www.codewars.com/kata/55bf01e5a717a0d57e0000ec/python
| 2   | Persistent Bugger.      | obnounce | :x: Not Completed           | 6          | https://www.codewars.com/kata/55bf01e5a717a0d57e0000ec/python



# words

## 1. Duplicate Encoder

```py
def duplicate_encode(word):
    import re
    answer=[]
    word=word.lower()
    for i in range(len(word)):
        filterlist=word[0:i]+word[i+1:]
#        print(filterlist)
        if word[i] in filterlist:
            answer.append(')')
        else:
            answer.append('(')
    answer=''.join(answer)
    return answer


def duplicate_encode(word):
    a = ["(" if word.lower().count(c) == 1 else ")" for c in word.lower()]
    return "".join(a)


def duplicate_encode(word):
    word = word.upper()
    result = ""
    for char in word:
        if word.count(char) > 1:
            result += ")"
        else:
            result += "("
    return result


def duplicate_encode(word):
    word = word.lower()
    return ''.join([')' if word.count(char) > 1 else '(' for char in word])
```

## 2. count_sheeps
```py
1.
def count_sheeps(sheep):
  return (sheep.count(True))

2.
def count_sheeps(sheep):
    a=0
    for i in sheep:
        if i:
            a+=1
    return a

3.
def count_sheeps(sheep):
    return len([i for i in sheep if i])

def count_sheeps(sheep):
    return sum(1 for i in sheep if i)

4.
def count_sheeps(a):
    t = 0
    for i in a:
        if i == True:
            t += 1
    return t
```

## 3. Persistent Bugger
```py
1.
def persistence(n):
    res = 0
    temp = 1
    while (n > 9):
        if temp == 1:
          temp = n % 10
        n -= n %10
        n /= 10
        temp *= (n % 10)
        if (n <= 9) :
          n = temp
          temp = 1
          res += 1
    return res

2.
def persistence(n):
    n=str(n)
    count=0
    while len(n)>1:
        p=1
        for i in n:
            p *= int(i)
        count+=1
        n=str(p)
    return count

```
