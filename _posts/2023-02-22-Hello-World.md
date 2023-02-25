---
title: Hello World
author: Nick_Post
date: 2023-02-22 14:10:00 +0800
categories: [HelloWorld, NewTest]
tags: [postnick]
render_with_liquid: false
pin: true
---
# Hello World

I'll figure out what to do with this GitHub static web page someday. As for now welcome, come back soon, maybe I'll have some new content. 


## Python Code
I wrote this after watching a youtube video. 

[Veritasium Video Link](https://www.youtube.com/watch?v=094y1Z2wpJg)

```python
StartNumb = int (input ("Give me any positive integer: "))
print(f'You entered {StartNumb}')

result = StartNumb
X = 1
maxnum = int(1)
while result != 1:
    if (result % 2) == 0:
        #print ("(result} is Even")
        result = result/ 2
    else:
        #print ("(result} is odd")
       result = (result * 3) + 1
    print(X, ":", int (result) )
    X=X+1
    if result > maxnum:
        maxnum = int(result)

print (f"It took {X-1} cycles to hit 1 with a Maximum of {maxnum}")
```

Adding Code to avoid Run Errors