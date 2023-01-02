---
title: Python Crash
date: 2019-10-11 11:11:11 -0400
description:
categories: [00CodeNote, PythonNote]
img: /assets/img/sample/rabbit.png
tags: [Python]
---

- [Python Crash](#python-crash)
  - [String](#string)
  - [List](#list)
  - [dictionary](#dictionary)
  - [Tuple](#tuple)
  - [Sorted](#sorted)
  - [Functions](#functions)
  - [lambda](#lambda)
  - [zip, map filter](#zip-map-filter)
  - [test](#test)
  - [except](#except)
  - [RegularExpression](#regularexpression)
  - [Data collect](#data-collect)
  - [Network with PY](#network-with-py)
  - [class](#class)
- [images](#images)
  - [ku example](#ku-example)
    - [filter() function](#filter-function)

---


# Python Crash

---

## String
str.count( sub, start= 0, end=len(string) )


## List
```py
1.	list.append(obj): 在列表末尾添加新的对象
2.	list.count(obj): 统计某个元素在列表中出现的次数
3.	list.extend(seq): 在列表末尾一次性追加另一个序列中的多个值（用新列表扩展原来的列表）
4.	list.index(obj): 从列表中找出某个值第一个匹配项的索引位置
5.	list.insert(index, obj): 将对象插入列表
6.	list.pop([index=-1]): 移除列表中的一个元素（默认最后一个元素），并且返回该元素的值
7.	list.remove(obj): 移除列表中某个值的第一个匹配项
8.	list.reverse(): 反向列表中元素
9.	list.sort( key=None, reverse=False): 对原列表进行排序
10.	list.clear(): 清空列表
11.	list.copy(): 复制列表

# a list of numbers from 0 to 67:
for i in range(68)

# return values
list.sort()  =None
sorted(list) =list

# separate
inventory = ["shoes, 12, 29.99", "shirts, 20, 9.99", "sweatpants, 25, 15.00", "scarves, 13, 7.75"]
for item in inventory:
    a=item.split(", ")
    name=a[0]
    num=a[1]
    price=a[2]
    print("The store has {} {}, each for {} USD.".format(num,name,price))
```


## dictionary

```py

############### dictionary mechanics ###############

sports = {'baseball': 9, 'basketball': 4, 'soccer': 4, 'cricket': 2}
sports['hockey']=3
sport=list(sports.keys())


############### dictionary accumulation ###############
credits=0
for i in sports:
    credits+= sports[i]

str1 = "peter piper picked a peck of pickled peppers"
freq = {}                 # check frequency
for i in str1:
    if i not in freq:
        freq[i]=0
    freq[i]+=1
best_char=freq.keys()[0]         # check bigest value & key
best_value=freq[freq.keys()[0]]
for j in freq:
    if freq[j]>freq[freq.keys()[0]]:
        best_char=j
        best_value=freq[j]
```



## Tuple

```py
tuples_lst = [('Beijing', 'China', 2008), ('London', 'England', 2012), ('Rio', 'Brazil', 2016, 'Current'), ('Tokyo', 'Japan', 2020, 'Future')]
country=[]
for i,j in enumerate(tuples_lst):
    country.append(j[1])

# Tuple Unpacking
(variable names) = (values)

julia = "Julia", "Roberts", 1967, "Duplicity", 2009, "Actress", "Atlanta, Georgia"

name, surname, birth_year, movie, movie_year, profession, birth_place = julia
name, surname, birth_year, movie, movie_year, profession, birth_place="Julia", "Roberts", 1967, "Duplicity", 2009, "Actress", "Atlanta, Georgia"

print(name)   # julia

# variable names on the left side!!
# "Julia", "Roberts", 1967, "Duplicity", 2009, "Actress", "Atlanta, Georgia" = name, surname, birth_year, movie, movie_year, profession, birth_place
# SyntaxError: can't assign to literal on line 7

(a, b, c, d) = (1, 2, 3)  # ValueError: need more than 3 values to unpack
(a, b, c, d) = (1, 2, 3, 4)

students = [('Tommy', 95), ('Linda', 63), ('Carl', 70), ('Bob', 100), ('Raymond', 50), ('Sue', 75)]
passed = [ name for (name,grade) in students if grade>=70 ]
```

---

## Sorted
```py
letters = "alwnfiwaksuezlaeiajsdl"
sorted_letters=sorted(letters, reverse=True)

# according to the dic value
medals = {'Japan':41, 'Russia':56, 'South Korea':21, 'United States':121, 'Germany':42, 'China':70}
sort_list = sorted(medals, reverse=True, key=lambda key:medals[key])
top_three = sorted(medals, reverse=True, key=lambda key:medals[key])[:3]

# according second letter
list = ['hi', 'how are you', 'bye', 'BigBlueberry', 'zebra', 'dance']
lambda_sort=sorted(ex_lst, key=lambda str: str[1])


# case-insensitive string comparison:
sorted("This is a test string from Andrew".split(), key=str.lower)
# ['a', 'Andrew', 'from', 'is', 'string', 'test', 'This']



# Operator Module Functions
# Python provides convenience functions to make accessor functions easier and faster. The operator module has itemgetter(), attrgetter(), and a methodcaller() function.
from operator import itemgetter, attrgetter
class Student:
    def __init__(self, name, grade, age):
        self.name = name
        self.grade = grade
        self.age = age
    def __repr__(self):
        return repr((self.name, self.grade, self.age))
student_objects = [
    Student('john', 'A', 15),
    Student('jane', 'B', 12),
    Student('dave', 'B', 10),
]
# sort by age:
sorted(student_objects, key=lambda student: student.age)
# sort by age:
sorted(student_tuples, key=itemgetter(2))
[('dave', 'B', 10), ('jane', 'B', 12), ('john', 'A', 15)]
# sort by grade then by age:
sorted(student_tuples, key=itemgetter(1,2))
[('john', 'A', 15), ('dave', 'B', 10), ('jane', 'B', 12)]
# sort by age:
sorted(student_objects, key=attrgetter('age'))
[('dave', 'B', 10), ('jane', 'B', 12), ('john', 'A', 15)]
# sort by grade then by age:
sorted(student_objects, key=attrgetter('grade', 'age'))
[('john', 'A', 15), ('dave', 'B', 10), ('jane', 'B', 12)]


# Sort Stability and Complex Sorts
s = sorted(student_objects, key=attrgetter('age'))     # sort first on secondary key
sorted(s, key=attrgetter('grade'), reverse=True)       # now sort on primary key, descending
# [('dave', 'B', 10), ('jane', 'B', 12), ('john', 'A', 15)]
def multisort(xs, specs):
    for key, reverse in reversed(specs):
        xs.sort(key=attrgetter(key), reverse=reverse)
    return xs
multisort(list(student_objects), (('grade', True), ('age', False)))



# The Old Way Using Decorate-Sort-Undecorate¶
decorated = [(student.grade, i, student) for i, student in enumerate(student_objects)]
decorated.sort()
[student for grade, i, student in decorated]               # undecorate
[('john', 'A', 15), ('jane', 'B', 12), ('dave', 'B', 10)]



```

---

## Functions
```py
def length(inlist):
    if len(inlist) >= 5:
        return "Longer than 5"
    else:
        return "Less than 5"

def test(x, abool = True, dict1 = {2:3, 4:5, 6:8}):
    return abool and dict1.get(x, False)
test(5, dict1 = {5:4, 2:1})


def strip_punctuation(string):   # move the punctuation
    for i in string:
        if i in ["'", '"', ",", ".", "!", ":", ";", '#', '@']:
            string=string.replace(i,'')
    return string
```



## lambda

```py
def fname(arguments):
    return value

fname = lambda arguments: value


mult = lambda int,x=6: int*x

greeting = lambda name, greeting="Hello ", excl="!": greeting + name + excl


lst_check = ['plums', 'watermelon', 'kiwi', 'strawberries', 'blueberries', 'peaches', 'BigBlueberry', 'mangos', 'papaya']
map_testing= map( lambda s: 'Fruit: '+s, lst_check)

countries = ['Canada', 'Mexico', 'Brazil', 'Chile', 'Denmark', 'Botswana', 'Spain', 'Britain', 'Portugal', 'Russia', 'Thailand', 'Bangladesh', 'Nigeria', 'Argentina', 'Belarus', 'Laos', 'Australia', 'Panama', 'Egypt', 'Morocco', 'Switzerland', 'Belgium']
b_countries= filter( lambda s: s[0]=='B' , countries)

def get_related_titles(lst):
    titlelst=[]
    for name in lst:
        [titlelst.append(name) for name in extract_movie_titles(get_movies_from_tastedive(name)) if name not in titlelst]
    return titlelst


sum = lambda arg1, arg2: arg1 + arg2
print ("sum= ", sum( 10, 20 ))
sum=30

together= lambda num,abc,x=" ":x.join([str(num),abc])
```


## zip, map filter

```py
l1 = ['left', 'up', 'front']
l2 = ['right', 'down', 'back']
zip(l1,l2) = [('left', 'right'), ('up', 'down'), ('front', 'back')]

opposites= [ (x1,x2) for (x1,x2) in zip(l1,l2) if len(x1)>3 and len(x2)>3 ]


def square(x) : return x ** 2
map(square, [1,2,3,4,5])
map(lambda x: x ** 2, [1, 2, 3, 4, 5])

filter(function, sequence)

lst_check = ['plums', 'watermelon', 'kiwi', 'strawberries', 'blueberries', 'peaches', 'BigBlueberry', 'mangos', 'papaya']
# elements in lst_check that have a w
filter_testing=list( filter( lambda value: 'w' in value, lst_check) )
```

---

## test
```py
assert sorted([1, 7, 4]) == [1, 4, 7]
assert sorted([1, 7, 4], reverse=True) == [7, 4, 1]
```

---

## except

```py
try:
    items = ['a', 'b']
    third = items[3]
    print("This won't print")
except Exception:
    print("got an error")


full_lst = ["ab", 'cde', 'fgh', 'i', 'jkml', 'nop', 'qr', 's', 'tv', 'wxy', 'z']
attempt = []

for elem in full_lst:
    try:
        attempt.append(elem[1])
    except:
        attempt.append("Error")
```


---

## RegularExpression
```py
# ^	Matches the beginning of a line
# $	Matches the end of the line
# .	Matches any character
# \s	Matches whitespace
# \S	Matches any non-whitespace character
# *	    Repeats a character zero or more times
# *?	Repeats a character zero or more times (non-greedy)
# +	    Repeats a character one or more times
# +?	Repeats a character one or more times (non-greedy)
# [aeiou]	Matches a single character in the listed set
# [^XYZ]	Matches a single character not in the listed set
# [a-z0-9]	The set of characters can include a range
# (	Indicates where string extraction is to start
# )	Indicates where string extraction is to end

import re

hand = open('file.txt')
    for line in hand:
        line = line.rstrip()
        if re.search('From:', line):
            print(line)


x='my 2 favoriate numbers are 19 and 42'
y=re.findall('[0-9]+',x)
print(y)     # ['2', '19', '42']


# greedy match
x='From: sing the : character.'
y=re.findall('^F.+:',x)   # greedy: it will give back the largest one.
print(y)                  # ['From: sing the : ']
y=re.findall('^F.+?:',x)  # dont be greedy: add '?'
print(y)                  # ['From:']
```



## Data collect
```py
import requests
import json

page = requests.get("https://api.datamuse.com/words?rel_rhy=funny")

print(type(page))       #<class 'requests.Response'>

print(page.text[:150])  # print the first 150 characters
#[{"word":"money","score":4417,"numSyllables":2},{"word":"honey","score":1208,"numSyllables":2},{"word":"sunny","score":720,"numSyllables":2},{"word":"

print(page.url)          # print the url that was fetched
#https://api.datamuse.com/words?rel_rhy=funny

x = page.json()            # turn page.text into a python object
x = jsno.loads(page, text) # list

------------------------------------

d = {'q': '"violins and guitars"', 'tbm': 'isch'}
results = requests.get("https://google.com/search", params=d)
results.url  # https://www.google.com/search?q=%22violins+and+guitars%22&tbm=isch
results.text

------------------------------------

import requests_with_caching
import json

parameters = {"term":"ann arbor", "entity": "podcast"}
response = requests_with_caching.get("https://api.datamuse.com/words?rel_rhy=happy", permanent_cache_file="data_cache.txt")

py_data = json.loads(data_cache.txt)
for r in py_data['results']:
    print(r['trackName'])

------------------------------------

import requests

def requestURL(baseurl, params = {}):
    # accepts a URL path and a params diction as inputs.
    # calls requests.get() with those inputs,
    # and returns the full URL of the data you want to get.
    req = requests.Request(method = 'GET', url = baseurl, params = params)
    prepped = req.prepare()
    return prepped.url

print(requestURL(some_base_url, some_params_dictionary))

https://api.datamuse.com/words?rel_rhy=funny

print(requestURL("https://api.datamuse.com/words", {"rel_rhy":"funny"}) )

```


## Network with PY
```py
# make a socket, a connection
import socket
mysock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
mysock.connect( ('data.pr4e.org', 80))   # host and port

# HTTP
# http:// www.goo.com/ index.html
# protocol+host+document
import socket
mysock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
mysock.connect( ('data.pr4e.org', 80))   # host and port
cmd = 'GET http://data.pr4e.org/romeo.txt HTTP/1.0\r\n\r\n'.encode()
mysock.send(cmd)

while True:
    data = mysock.recv(512)
    if (len(data) < 1):
        break
    print(data.decode())
mysock.close()
```


---


## class

```py

class Car:
    def __init__(self, make, model, year):
        self.make = make
        self.model = model self.year = year self.odometer_reading = 0

    def get_descriptive_name(self):
        long_name = f"{self.year} {self.manufacturer} {self.model}"
        return long_name.title()

    def read_odometer(self):
        print(f"This car has {self.odometer_reading} miles on it.")

    def update_odometer(self, mileage):
        if mileage >= self.odometer_reading:
            self.odometer_reading = mileage else:
            print("You can't roll back an odometer!")

    def increment_odometer(self, miles):
        self.odometer_reading += miles

class ElectricCar(Car):

    def __init__(self, make, model, year):
        super().__init__(make, model, year)
        self.battery_size = 75

    # Overriding Methods from the Parent Class
    def fill_gas_tank(self):
        print("This car doesn't need a gas tank!")

```



---


# images


```py

# face detect
import cv2 as cv
face_cascade = cv.CascadeClassifier('readonly/haarcascade_frontalface_default.xml')
eye_cascade = cv.CascadeClassifier('readonly/haarcascade_eye.xml')

img = cv.imread('readonly/floyd.jpg')

# convert it to grayscale using the cvtColor image
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(gray)
# array([[158, 75, 176, 176]], dtype=int32)

faces = face_cascade.detectMultiScale(cv_img, 1.05)
show_rects(faces)


rec = faces.tolist()[0]
# [158, 75, 176, 176]


from PIL import Image
from PIL import ImageDraw


#  draw
pil_img=Image.fromarray(gray,mode="L")
drawing=ImageDraw.Draw(pil_img)
drawing.rectangle( (rec[0],rec[1], rec[0]+rec[2], rec[1]+rec[3]), outline="white")
display(pil_img)


pil_img = Image.open('readonly/msi_recruitment.gif')
pil_img.mode       # "P"


open_cv_version = pil_img.convert("L")
open_cv_version.save("msi_recruitment.png")


pil_img = pil_img.convert("RGB")
pil_img.mode       # "RBG"





# list all files in the current directory using os.listdir:
import os
for filename in os.listdir(os.getcwd()):
   with open(os.path.join(os.cwd(), filename), 'r') as f: # open in readonly mode
      # do your stuff

# list only some files, depending on the file pattern using the glob module:
import glob
for filename in glob.glob('*.txt'):
   with open(os.path.join(os.cwd(), filename), 'r') as f: # open in readonly mode
      # do your stuff
```


---


## ku example


### filter() function

- filter out contents from a given sequence that can be a `list, string or tuple` etc.

`filter(function, iterable)`

Arguments:
- An iterable sequence to be filtered.
- a function that accepts an argument and returns bool
  - i.e. True or False based on it’s logic.

Returns:
- A new sequence of filtered contents.

Logic:
- filter() iterates over all elements in the sequence and for each element it calls the given callback function.
- If this function returns False then that element is skipped
- returned True, elements are added into a new list.
- In the end it returns a new list with filtered contents based on the function passed to it as argument.

```py
def isOfLengthFour(strObj):
    if len(strObj) == 2:
        return True
    else:
        return False


# Filter a list of strings in Python using filter()
listOfStr = ['hi', 'this' , 'is', 'a', 'very', 'simple', 'string' , 'for', 'us']
filteredList = list(filter(isOfLengthFour , listOfStr))
# called isOfLengthFour() for each string element.
# filteredList:  ['hi', 'is', 'us']


filteredList = list(filter(lambda x : len(x) == 2 , listOfStr))
# filteredList:  ['hi', 'is', 'us']


strObj = 'Hi this is a sample string, a very sample string'
filteredChars = ''.join((filter(lambda x: x not in ['a', 's'], strObj)))
# Filtered Characters  :  Hi thi i  mple tring,  very mple tring


array1 = [1,3,4,5,21,33,45,66,77,88,99,5,3,32,55,66,77,22,3,4,5]
array2 = [5,3,66]
filteredArray = list(filter(lambda x : x not in array2, array1))
# Filtered Array  :  [1, 4, 21, 33, 45, 77, 88, 99, 32, 55, 77, 22, 4]
```




















.
