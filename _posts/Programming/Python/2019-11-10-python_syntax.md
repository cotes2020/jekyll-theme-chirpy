---
title: Python Syntax
categories: [Programming, Python]
---

## Output
<hr style="border-top: 1px solid;"><br>

+ ```print(*objects, sep=' ', end='\n', file=sys.stdout, flush=False)```

  + objects : 출력할 값들

  + sep : 구분자 

  + end : 마지막에 출력할 값

  + file : 출력 결과를 보여줄 곳, 기본 값은 sys.stdout (screen)

<br>

+ ex) ```print(1, 2, 3, 4, sep='#', end='&') >> 1#2#3#4&```

<br>

+ inline if문
  
  + print ("aa" if True else "bb") #aa 

  + print ("aa" if False else "bb") #bb

<br>

inline for문이나 if문 등 압축해서 표현하는 것은 기본적으로 짠 구조를 거꾸로 적어놓은 것 같음.

<br>

```python
for i in check_nums :
    if i in cnt :
        print(cnt[i])
    else :
        print('0')
```

<br>

한줄로 표현하면 다음과 같음.
: ```print( ' '.join( str( cnt[i] ) if i in cnt else '0' for i in check_nums ) )```

<br>

**Output Formatting**

+ ```str.format()```을 이용해서 출력 가능하며, {}은 placeholders로 사용.

<br>

```python
print('The value of x is {} and y is {}'.format(x,y))

>> The value of x is 5 and y is 10
```

<br>

```python
print('I love {0} and {1}'.format('bread','butter'))

>> I love bread and butter
```

<br>

```python
print('Hello {name}, {greeting}'.format(greeting = 'good', name = 'John'))

>> Hello John, good
```

<br>

+ 추가로 format을 이용해 bin, oct, hex 값 출력 가능

<br>

```python
print(format(value,'b')  # bin

print(format(value,'o')  # oct

print(format(value,'x')  # hex
```

<br>

+ Python 3.6 이상의 버젼에서는 f-string이 있음.

  + 문자열 앞에 f를 붙이고 {}안에 변수명을 넣어주면 됨.

<br>

```python
animals = 'eels'

print(f'My hovercraft is full of {animals}.')

>>> My hovercraft is full of eels.


print(f'My hovercraft is full of {animals!r}.')

>>> My hovercraft is full of 'eels'.
```

<br>

+ 또한 C언어처럼 출력타입을 통해 출력할 수 있음. 차이점은 변수 앞에 %를 붙여준다는 점.

<br>

```python
x = 12.3456789
print('The value of x is %3.2f' %x)

>> The value of x is 12.35
```

<br><br>
<hr style="border: 2px solid;">
<br><br>

## Input
<hr style="border-top: 1px solid;"><br>

사용자가 어떤 값을 입력하게 하고, 그 값을 변수에 저장함.  

함수로는 **input(), sys.stdin.readline()** 이 있음. readline()이 input()보다 빠르다고 함.

<br>

+ 입력된 값들은 모두 문자열로 저장되고, 정수형이나 실수형으로도 저장 가능함.  

<br>

```python
x = int(input('number : '))     # 정수

x = float(input('number : '))   # 실수
```

<br>

+ 여러 개의 값 입력받기

  + 문자열
    + ```a,b,c=input().split()``` : split 함수는 문자열을 분리시켜줌

  + 숫자
    + ```a,b,c=map(int,input().split())``` : map 함수는 일종의 반복처리를 해주는 함수

<br>

+ input과 readline 차이점

  + input() : built-in function

  + sys.stdin : file object

<br>

sys.stdin을 보면 파일 오브젝트임. 파일에는 read() 메소드가 있음.

+ read() : 파일 전체의 내용을 하나의 문자열로 읽어옴. Binary 파일도 읽을 수 있음.

+ readline() : 한번에 하나의 라인을 읽어오는 메소드.

+ readlines() : 파일 전체를 한라인씩 읽어와서 리스트를 만들어주는 메소드.

<br>

**readline(), readlines()는 개행문자인 "\n"도 같이 읽어옴.**

<br><br>
<hr style="border: 2px solid;">
<br><br>

## Python Built-in Functions
<hr style="border-top: 1px solid;"><br>

Link 
: <a href="https://docs.python.org/ko/3/library/functions.html" target="_blank">Built-in Functions</a>

<br><br>
<hr style="border: 2px solid;">
<br><br>

## List
<hr style="border-top: 1px solid;"><br>

```list[start:end]``` 포맷은 리스트의 start인덱스부터 end-1인덱스까지의 요소를 선택함.

슬라이싱에 값을 대입하는 것도 가능함. 
: ex) ```letters[2:5]=[‘C’,‘D’,‘E’]```

슬라이싱을 하면 요구된 요소를 포함하는 부분 리스트를 반환함. 

즉, 새로운 복사본을 얻음.

<br><br>

### 합병과 반복
<hr style="border-top: 1px solid;"><br>

두 개의 리스트를 합칠 때는 ```+```연산자 이용 (extend 메소드와 동일한 기능)
: ex) ```list3=[1,2,3]+[4,5,6] -------> [1,2,3,4,5,6]```

리스트를 반복할 땐 ```*```연산자 이용
: ex) ```list3=[1,2,3]*3 -----> [1,2,3,1,2,3,1,2,3]```

<br><br>

### 요소 삽입/삭제 
<hr style="border-top: 1px solid;"><br>

+ 삽입

  + ```{list}.append()``` : 리스트의 끝에 새로운 요소를 추가함.

  + ```{list}.extend()``` : 여러 개의 요소를 추가 가능. 
    ex) ```{list}.extend([9, 11, 13])```

  + ```{list}.insert(인덱스 위치, 값)``` : 특정한 위치에 새로운 요소를 추가함.

<br>

+ 삭제

  + ```del {list}``` 
    
    특정 요소를 삭제하거나, 리스트 전체를 삭제 가능(리스트 변수 자체가 없어짐).

  + ```{list}.pop(index)```
    
    특정한 위치에 있는 항목 반환 후 삭제, 아무 값 없으면 맨 끝 요소 반환 후 삭제.

  + ```{list}.remove(값)```
    
    삭제하고자 하는 값을 적으면 리스트 항목에서 찾아서 일치하는 항목을 삭제.

  + ```{list}.clear()```
    
    리스트의 모든 요소를 삭제 (del과 달리 변수는 남음)

<br>

+ 요소의 개수 세기

  + ```{list}.count(element)``` : element의 개수를 반환, 없으면 0 리턴.

<br><br>

### 정렬
<hr style="border-top: 1px solid;"><br>

+ ```<list>.sort([key = <function>], [reverse = True|False])```

  + sort()는 리스트를 제자리에서 정렬, 원본 리스트가 변경되며 리턴값은 None임.

<br>

+ ```sorted(<iterable>, [key = <function>], [reverse = True|False])```

  + sorted()는 새로운 정렬된 리스트를 리턴함.

<br>

sort() 메소드와 sorted() 함수는 매개변수를 가지는데 key와 reverse가 있음.

key 매개변수를 이용하면 요소들을 비교하기 전에 정렬 기준이 되는 함수를 적용시킬 수 있음.

<br>

```python

sorted(“I am iron man.”.split(), key=str.lower) // 대소문자 구분없이 정렬

-> ['am', 'I', 'iron', 'man.']
```

<br>

또는 사용자 정의 함수를 작성한 후 적용시킬 수 있음.
: ```sorted_list = sorted(l, key=cmp_to_key(comp))```

<br>

reverse 매개변수는 정렬 방향을 지정하는데 사용
: ex) ```sorted([5,2,3,1,4], reverse=True) -----> [5,4,3,2,1]```

<br><br>

### 문자열에서 리스트 만들기
<hr style="border-top: 1px solid;"><br>

split() 메소드는 문자열을 분리하고 이것을 리스트로 만들어서 반환함.

이때 문자열을 분리하는 분리자를 지정할 수 있으며, 지정되지 않으면 스페이스를 이용하여 문자열을 분리함.

<br><br>

### 리스트 복사
<hr style="border-top: 1px solid;"><br>

score라는 리스트가 있을 때, value라는 변수를 value=score라 하면 value는 단지 score 리스트가 가리키고 있는 리스트를 가리키는 score과 이름만 다른 변수가 됨.
즉, c언어의 포인터 변수

<br>

올바른 복사 방법은 list() 함수를 이용하거나 deepcopy() 함수를 이용해야 함. 단, deepcopy() 함수는 copy 모듈을 import 해야 함.

<br><br>

### 함축(comprehension)
<hr style="border-top: 1px solid;"><br>

```변수 = [ expression for i in 리스트 if 조건]```, 조건은 생략 가능 
: ex) ```S = [ x**2 for x in range(10) if x%2 == 0] -> [0,2,4,6,8]```

<br.

두 개의 리스트를 합쳐서 만들수도 있음.

<br>

```python
colors = [‘white’,‘silver’], cars=[‘bmw’,‘sonata’],

color_cars = [(x,y) for x in colors for y in cars]
```

<br><br>

### List built-in functions
<hr style="border-top: 1px solid;"><br>

+ cmp(list1, list2)

+ len(list)

+ max(list)

+ min(list)

<br><br>
<hr style="border: 2px solid;">
<br><br>

## Tuple
<hr style="border-top: 1px solid;"><br>

튜플은 ```tuple=()```로 생성하며, **리스트와 거의 비슷하나 리스트와 달리 요소 값을 변경할 수 없음.**

하지만 튜플의 요소 중 mutable data(리스트, 딕셔너리 등)는 변경 가능함.

<br>

```python
my_tuple = (4, 2, 3, [6, 5])

my_tuple[3][0] = 9    

print(my_tuple) # Output: (4, 2, 3, [9, 5])
```

<br>

+ mutable data
  list, dictionary, set and user-defined classes 

<br>

+ immutable data
  int, float, decimal, bool, string, tuple, and range

<br><br>

### 요소 삭제
<hr style="border-top: 1px solid;"><br>

**tuple은 immutable data로 list처럼 특정 요소를 삭제하는 것이 불가능.**

따라서 tuple은 아예 변수를 삭제하는 것만 가능함. 이 때 del 를 사용하면 됨.

<br><br>

### tuple methods
<hr style="border-top: 1px solid;"><br>

튜플은 두 가지의 메소드(count, index)만 사용 가능함.

<br>

```python
my_tuple = ('a', 'p', 'p', 'l', 'e',)

print(my_tuple.count('p'))  # Output: 2

print(my_tuple.index('l'))  # Output: 3
```

<br><br>

### Tuple built-in functions
<hr style="border-top: 1px solid;"><br>

+ cmp(list1, list2)

+ len(list)

+ max(list)

+ min(list)

<br><br>

### Tuple vs List
<hr style="border-top: 1px solid;"><br>

+ 서로 다른 데이터 유형에 대해선 튜플을 사용하고, 같은 데이터 유형에는 리스트 사용.

+ 튜플은 immutable data로 튜플을 통해 반복하는 속도가 list보다 빠름.

+ 튜플은 dictionary의 key 값으로 사용 가능함.

<br><br>
<hr style="border: 2px solid;">
<br><br>

## Dictionary
<hr style="border-top: 1px solid;"><br>

+ ```변수 = {키1 : 값1, 키2 : 값2 ...}```

  + key는 immutable object(int, str, tuple, etc) 이며 unique한 값이어야 함.

  + value는 repeat 가능하며, any data type 허용함.

<br>

공백 딕셔너리는 변수={}로 생성

<br>

+ key() 메소드 : 딕셔너리 key 값 만 저장된 dict_keys 타입 (클래스 객체)을 반환합니다.

+ values() 메소드 : 딕셔너리에 저장된 value들을 dict_values 타입으로 반환합니다.

<br><br>

### 항목 접근, 추가, 삭제
<hr style="border-top: 1px solid;"><br>

+ Access : 변수[키] 또는 변수.get(키 값) 메소드를 이용

+ Update / Add : 변수[키] = 값으로 변경 및 추가 가능.

+ Remove 
  
  + pop(키) : key 값으로 찾아서 삭제, return value

  + popitem() : 임의의 값 삭제, return (key, value) pair

  + del 변수[키]

  + clear()

<br><br>

### 함축
<hr style="border-top: 1px solid;"><br>

```python
triples={ x : x*x*x for x in range(3) }
>>> { 0 : 0, 1 : 1, 2 : 8 }

odd_squares = {x: x*x for x in range(11) if x % 2 == 1}
>>> {1: 1, 3: 9, 5: 25, 7: 49, 9: 81}
```

<br><br>

### 정렬
<hr style="border-top: 1px solid;"><br>

근본적으로 요소들을 특정 순서대로 저장하지 않아서 입력순서와 다르게 저장될 수 있음. 그래서 sorted()함수 이용

키를 정렬하고 싶으면 sorted()를 쓰고 값을 정렬하고 싶으면 sorted(변수.values())

<br><br>
<hr style="border: 2px solid;">
<br><br>

## Set
<hr style="border-top: 1px solid;"><br>

Set는 {}로 감싸며, 정렬되지 않고, 요소의 값도 중복을 허용하지 않음.

또한 set는 mutable data이나 set의 요소(item)에는 mutable data를 허용하지 않음.

따라서 set에는 slicing, indexing을 통해 변경, 접근 불가능(사용도 불가능).

공백 Set는 ```set()```로 생성.

<br>

```python
# set cannot have duplicates

my_set = {1, 2, 3, 4, 3, 2}

print(my_set) # Output: {1, 2, 3, 4}

s2 = set("Hello") # {'e', 'H', 'l', 'o'}
```

<br>

```python
# we can make set from a list

my_set = set([1, 2, 3, 2])

print(my_set) # Output: {1, 2, 3}
```

<br>

```python
# set cannot have mutable items

my_set = {1, 2, [3, 4]} # TypeError: unhashable type: 'list'
```

<br><br>

### Set methods
<hr style="border-top: 1px solid;"><br>

+ add() : 요소 추가

+ update() : 여러 개의 요소 추가, (인자로는 list, tuple, string, set 등)

<br>

단, 두 개의 메소드 모두 중복된 값은 추가되지 않음.

```python
my_set = {1, 3}

my_set.add(2) # add an element
print(my_set) # Output: {1, 2, 3}


my_set.update([2, 3, 4]) # add multiple elements
print(my_set) # Output: {1, 2, 3, 4}


my_set.update([4, 5], {1, 6, 8}) # add list and set
print(my_set) # Output: {1, 2, 3, 4, 5, 6, 8}
```

<br>

+ discard(item), remove(item)  
  
  두 함수의 차이점은 discard는 제거할 item이 set에 없어도 오류가 안나지만 remove 함수는 오류를 raise함.

+ pop() : 랜덤 element를 삭제

+ clear() : empty set으로 만듦

<br><br>

### set operations
<hr style="border-top: 1px solid;"><br>

+ 합집합(union) : set1.union(set2), set1 | set2

+ 교집합(intersection) : set1.intersection(set2), set1 & set2

+ 차집합(difference) : set1(or 2).difference(set2(or 1)), set1(or 2) - set2(or 1)

<br><br>
<hr style="border: 2px solid;">
<br><br>

## Function
<hr style="border-top: 1px solid;"><br>

keyword arguments는 반드시 positional arguments 보다 뒤에 있어야 함.

<br>

``` python
# def greet(name, msg)

# 2 keyword arguments
greet(name = "Bruce", msg = "How do you do?")


# 2 keyword arguments (out of order)
greet(msg = "How do you do?", name = "Bruce") 


# 1 positional, 1 keyword argument
greet("Bruce", msg = "How do you do?") 
```

<br>

```python
greet(name="Bruce","How do you do?")

>> SyntaxError: non-keyword arg after keyword arg
```

<br>

함수 사용 시 때로는 인수의 개수를 미리 알 수 없는 경우가 있음.

이 때 매개변수 앞에 ```*```를 붙이면 됨. (약간 C언어의 포인터 같은 역할)  

```python
def greet(*names):
    """This function greets all
    the person in the names tuple."""

    # names is a tuple with arguments
    for name in names:
        print("Hello", name)

greet("Monica", "Luke", "Steve", "John")

'''
Hello Monica
Hello Luke
Hello Steve
Hello John
'''
```

<br><br>
<hr style="border: 2px solid;">
<br><br>

## 예외 처리
<hr style="border-top: 1px solid;"><br>

+ try .. except 형태

```python
try:
    ...
except [발생 오류[as 오류 메시지 변수]]:
    ...
```

<br>

+ try .. except만 사용

```python
try:
    ...
except:
    ...
```

<br>

+ try .. except 발생오류

```python
try:
    ...
except 발생 오류:
    ...
```

<br>

+ try .. except 발생오류 as 오류 메시지 변수

```python
try:
    ...
except 발생 오류 as 오류 메시지 변수:
    ...

////////////////////////////////////
# example

try:
    4 / 0
except ZeroDivisionError as e:
    print(e) # division by zero
```

<br>

+ try .. finally

  + finally절은 try문 수행 도중 예외 발생 여부에 상관없이 항상 수행됨.

  + 보통 finally절은 사용한 리소스를 close해야 할 때에 많이 사용함.

```
f = open('foo.txt', 'w')
try:
    ...
finally:
    f.close()
```

<br>

+ 여러개의 오류 처리

```
try:
    ...
except 발생 오류1:
   ... 
except 발생 오류2:
   ...
   
한줄로 표현 가능함.
except (error1, error2, ..., errorN):
   ...
```

```
# example

try:
    a = [1,2]
    print(a[3])
    4/0
except (ZeroDivisionError, IndexError) as e:
    print(e)
```

<br>

+  try .. except .. else

```
try:
    ...
except [발생 오류[as 오류 메시지 변수]]:
    ...
else:  # 오류가 없을 경우에만 수행된다.
    ...
```

<br>

+ 오류 회피하기

```
try:
    f = open("나없는파일", 'r')
except FileNotFoundError:
    pass
```

<br>

+ 오류 일부러 발생시키기
  + raise 발생오류 -> raise문을 이용해 오류를 일부러 발생

```python
# example

class Bird:
    def fly(self):
        raise NotImplementedError
        
class Eagle(Bird):
    pass

eagle = Eagle()
eagle.fly()

'''
Traceback (most recent call last):
  File "...", line 33, in <module>
    eagle.fly()
  File "...", line 26, in fly
    raise NotImplementedError
NotImplementedError
'''
```

<br><br>
<hr style="border: 2px solid;">
<br><br>

## lambda
<hr style="border-top: 1px solid;"><br>

anonymous function으로 이름이 없는 함수임. 보통 함수는 def를 통해 이름을 갖게 됨.

<br>

```
lambda arguments: expression

인수는 여러개 사용 가능하나, 표현식은 하나만 사용 가능함.
ex) ex = lambda x, y: x * y

lambda 함수는 보통 filter(), map() 함수와 사용한다고 함.
```

<br><br>

### filter()
<hr style="border-top: 1px solid;"><br>

```
filter(function, iterable)

filter() 함수는 두번째 인자로 넘어온 데이터 중에서 
첫번째 인자로 넘어온 조건 함수를 만족하는 데이터만을 반환함.

filter 함수는 filter 타입으로 결과 값을 반환하므로
list() 또는 tuple()로 값을 받아와야 함.
```

<br>

```python
my_list = [1, 5, 4, 6, 8, 11, 3, 12]
new_list = list(filter(lambda x: (x%2 == 0) , my_list))

print(new_list) # [4, 6, 8, 12]
```

### map()
<hr style="border-top: 1px solid;"><br>

```
map(function, iterable)

두 번째 인자로 들어온 iterable을 첫 번째 인자로 들어온 함수에 
하나씩 집어넣어서 데이터를 반환함.

map 함수는 map 타입으로 결과 값을 반환하므로 
iterable(list, tuple 등) 으로 값을 받아와야 함.
```

<br>

```python
my_list = [1, 5, 4, 6, 8, 11, 3, 12]
new_list = list(map(lambda x: x * 2 , my_list))

print(new_list) # [2, 10, 8, 12, 16, 22, 6, 24]
```

<br><br>
<hr style="border: 2px solid;">
<br><br>

## map과 filter 차이점
<hr style="border-top: 1px solid;"><br>

filter 함수는 list의 element에서 어떤 함수의 조건에 일치하는 값만 반환하고 싶을 때 사용

map 함수는 list의 element에 함수를 적용시켜 결과를 반환하고 싶을 때 사용

<br><br>
<hr style="border: 2px solid;">
<br><br>

## global, nonlocal 변수 선언
<hr style="border-top: 1px solid;"><br>

global 변수 : global, 전역변수로 선언

<br>

```python
x = "global "

def foo():
    global x
    y = "local"
    x = x * 2
    print(x)
    print(y)

foo()

'''
global global 
local
'''
```

<br>

nonlocal 변수 : nonlocal, 지역변수가 아님을 선언

<br>

```python
def outer():
    x = "local"

    def inner():
        nonlocal x
        x = "nonlocal"
        print("inner:", x)

    inner()
    print("outer:", x)


outer()

'''
inner: nonlocal
outer: nonlocal
'''
```

<br>

두 개의 차이점은 global은 어느 함수에서나 사용 가능하지만 nonlocal은 사용된 함수 바로 한단계 바깥쪽에 위치한 변수와 바인딩을 할 수 있음.

즉, 함수 한개를 정의하고 전역변수에 영향을 주게 하는 것은 안됌.

<br><br>
<hr style="border: 2px solid;">
<br><br>

## String methods

### enumerate(iterable, start=0)
<hr style="border-top: 1px solid;"><br>

enumerate 객체를 리턴하며, 리턴 값으로 index, value 쌍을 리턴함.

<br>

```python
seasons = ['Spring', 'Summer', 'Fall', 'Winter']

list(enumerate(seasons))

>>> [(0, 'Spring'), (1, 'Summer'), (2, 'Fall'), (3, 'Winter')]
```

<br><br>

### join()
<hr style="border-top: 1px solid;"><br>

```python
string.join(iterable) // List, Tuple, String, Dictionary and Set.

# join()은 iterable의 요소를 string seperator로 엮은 문자열을 리턴함.

# 단 dictionary와 사용할 경우 key 값만 join 시키며 key 값이 string이어야 함.

# string이 아니면 TypeError 발생
```

<br>

```python
# ex)

>> ''.join(['a','b','c'])
>> abc

>> ' '.join(['a','b','c'])
>> a b c
```

<br><br>
<hr style="border: 2px solid;">
<br><br>

## class and object
<hr style="border-top: 1px solid;"><br>

참고
: <a href="https://duwjdtn11.tistory.com/95" target="_blank">duwjdtn11.tistory.com/95</a>
: <a href="https://duwjdtn11.tistory.com/96" target="_blank">duwjdtn11.tistory.com/96</a>

<br>

python에서는 모든 것이 클래스이다?
: <a href="https://gist.github.com/shoark7/fb388e6494350442a2d649a154f69a3a" target="_blank">gist.github.com/shoark7/fb388e6494350442a2d649a154f69a3a</a>

<br>

+ class : 틀, 종류, 형식을 의미
  + ```ex)  human, dog, food 등등```

+ object(객체) : 틀, 형식(class)으로 만들어진 것

+ instance : 객체가 실체화 된 것. 즉, class의 실체를 의미
  + ```ex) apple = Food() # apple은 객체!, 객체 apple은 Food 클래스의 인스턴스 즉, 실체를 의미```

+ 필드 : 클래스에 속해 있는 변수를 의미

+ 메소드 : 클래스에 속해 있는 함수 즉, 기능을 의미
  + ```ex) ski 클래스가 있을 때, ride() 라는 기능을 설정하면 ride()는 ski 클래스의 메소드가 되는 것``` 

+ 속성 : 메소드, 필드 등을 통틀어 클래스의 속성이라고 한다. 

+ 클래스 변수 : 클래스에 속해 있는 변수

+ 인스턴스 변수 : 클래스의 인스턴스에 내장되어 있는 변수

<br>

+ ```__init__```
  + 생성자, 객체(class instance)가 생성되면 가장 먼저 실행되는 메소드
 
+ ```instance.__class__``` : 클래스 인스턴스가 속한 클래스 리턴
  + ```ex) ''.__class__ --> <class 'str'>```

<br>

객체에서 새로운 attribute 생성 가능(단, 생성한 객체에 한해서만)

<br>

```python
num1 = ComplexNumber(2, 3)
num2 = ComplexNumber(5)
num2.attr = 10

print(num2.attr) # Output: 10

print(num1.attr) 
# AttributeError: 'ComplexNumber' object has no attribute 'attr'
```

<br>

del keyword를 통해 attribute, object 삭제 가능.

class에서 메소드나 변수를 private으로 설정하고자 하면 변수, 메소드 명 앞에 ```_``` 또는 ```__```를 붙임.


<br><br>
<hr style="border: 2px solid;">
<br><br>

## inheritance
<hr style="border-top: 1px solid;"><br>

+ 클래스 상속(inheritance)

  + child class에 생성자(```__init__``` 메소드)에 ```super().__init__()``` 추가

    + --> super()는 parent class를 뜻함. 즉, parent class의 생성자를 사용하겠다는 뜻.

<br><br>
<hr style="border: 2px solid;">
<br><br>
