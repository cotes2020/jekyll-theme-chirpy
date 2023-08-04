---
title: Python Standard Library
categories: [Programming, Python]
---

## Python Built-In methods
<hr style="border-top: 1px solid;"><br>

Link
: <a href="https://docs.python.org/3/library/stdtypes.html" target="_blank">docs.python.org/3/library/stdtypes.html</a>

<br><br>
<hr style="border: 2px solid;">
<br><br>

## Python Standard Library List
<hr style="border-top: 1px solid;"><br>

List 
: <a href="https://docs.python.org/ko/3/library/index.html" target="_blank">docs.python.org/ko/3/library/index.html</a>

<br><br>
<hr style="border: 2px solid;">
<br><br>

## sys
<hr style="border-top: 1px solid;"><br>

sys.stdin.readline()
: ```input()``` 대신 사용, time out 발생 시 input 대신 사용하면 됨.

<br>

sys.setrecursionlimit(10**4)
: 재귀횟수 설정, 기본은 1000번임. 재귀 횟수를 늘리고자 할 때 사용

<br><br>
<hr style="border: 2px solid;">
<br><br>

## itertools
<hr style="border-top: 1px solid;"><br>

itertools 
: 효율적인 루핑을 위한 이터레이터를 만드는 함수
: <a href="https://docs.python.org/ko/3/library/itertools.html" target="_blank">docs.python.org/ko/3/library/itertools.html</a> 

<br><br>

### 짧은 입력에 관한 시퀀스
<hr style="border-top: 1px solid;"><br>

+ ```chain(*iterables)```

  + 첫 번째 iterable에서 소진될 때까지 요소를 반환한 다음 iterable로 넘어가고 이런 식으로 iterables의 모든 이터러블이 소진될 때까지 진행하는 이터레이터를 만듦. 

  + 여러 시퀀스를 단일 시퀀스처럼 처리하는 데 사용됨.

<br>

```python
# example

chain('ABC', 'DEF') : A B C D E F

my_list = [[1, 2], [3, 4], [5, 6]]
list(itertools.chain(*my_list)) : [1,2,3,4,5,6]
```

<br>

+ ```chain.from_iterable(iterable)``` : chain()의 대체 생성자

  + ```ex) chain.from_iterable(['ABC', 'DEF']) : A B C D E F```

<br><br>

### 무한 iterator
<hr style="border-top: 1px solid;"><br>

```python
count(start,[step])

ex) count(10) : 10 11 12 . . .
```

<br>

```python
cycle(iterable)

ex) cycle('ABCD') : A B C D A B C D . . . .
```

<br>

```python
repeat(object, [times])

# object를 반복해서 반환하는 이터레이터 생성

# times 인자가 지정되지 않으면 무기한 실행


# ex) repeat(10, 3) : 10 10 10 
```

<br><br>

### 조합형 iterator
<hr style="border-top: 1px solid;"><br>

```product(p, q, .. , [repeat=1])```
: 주어진 iterable 객체들에 대해 곱 집합을 리턴

```python
# ex) 
product('ABCD', repeat=2) : AA AB AC AD BA BB BC BD CA CB CC CD DA DB DC DD

product('ABCD', 'xy') --> Ax Ay Bx By Cx Cy Dx Dy

product(range(2), repeat=3) --> 000 001 010 011 100 101 110 111

product(A, repeat=4) --> product(A, A, A, A) -> A A A A
```

<br>

```permutations(iterable, r=None)```
: iterable에서 요소의 연속된 길이 r 순열을 반환
: r이 지정되지 않았거나 None이면, r의 기본값은 iterable의 길이이며 가능한 모든 최대 길이 순열이 생성됨.

```python
# ex) 
permutations('ABCD', 2) --> AB AC AD BA BC BD CA CB CD DA DB DC

permutations(range(3)) --> 012 021 102 120 201 210
```

<br>

```combinations(iterable, r)```
: 입력 iterable에서 요소의 길이 r 서브 시퀀스들을 반환

```python
combinations('ABCD', 2) --> AB AC AD BC BD CD

combinations(range(4), 3) --> 012 013 023 123
```

<br>

```combinations_with_replacement(iterable, r)```
: combinations 기능에서 추가로 개별 요소를 두 번 이상 반복할 수 있음

```python
combinations_with_replacement('ABCD', 2) --> AA AB AC AD BB BC BD CC CD DD
```

<br><br>
<hr style="border: 2px solid;">
<br><br>

## collections
<hr style="border-top: 1px solid;"><br>

Docs 
: <a href="https://docs.python.org/ko/3/library/collections.html" target="_blank">docs.python.org/ko/3/library/collections.html</a>  

<br><br>

### deque
<hr style="border-top: 1px solid;"><br>

+ 양쪽 끝에서 빠르게 추가와 삭제를 할 수 있는 리스트류 컨테이너

<br>

+ ```deque([iterable[, maxlen]])```

  + iterable의 데이터로 왼쪽에서 오른쪽으로 초기화된 새 deque 객체를 반환, iterable을 지정하지 않으면 빈 데크 반환.

  + maxlen이 지정되지 않거나 None이면, 데크는 임의의 길이로 커질 수 있음. 그렇지 않으면, 데크는 지정된 최대 길이로 제한됨. 

  + 일단 제한된 길이의 데크가 가득 차면 새 항목이 추가될 때, 해당하는 수의 항목이 반대쪽 끝에서 삭제됨.

<br>

deque는 append, pop 연산 시 거의 O(1) 성능으로 지원함.

list는 append, pop 연산 시 O(n)

<br><br>

+ deque methods

  + append(x) : 데크의 오른쪽에 x를 추가.

  + appendleft(x) : 데크의 왼쪽에 x를 추가.

  + clear() : 데크에서 모든 요소를 제거하고 길이가 0인 상태로 만듦.

  + copy() : 데크의 얕은 복사본을 만듦.

  + len(deque) : deque 길이 반환

  + count(x) : x 와 같은 데크 요소의 수를 count.

  + extend(iterable) : iterable 인자에서 온 요소를 추가하여 데크의 오른쪽을 확장.

  + extendleft(iterable)
    + iterable에서 온 요소를 추가하여 데크의 왼쪽을 확장. 
    + 일련의 왼쪽 추가는 iterable 인자에 있는 요소의 순서를 뒤집는 결과를 줌.

  + ```index(x[, start[, stop]])```
    + 데크에 있는 x의 위치를 반환 (인덱스 start 또는 그 이후, 그리고 인덱스 stop 이전).
    + 첫 번째 일치를 반환하거나 찾을 수 없으면 ValueError를 발생.

  + insert(i, x) : x를 데크의 i 위치에 삽입합니다.
    + 삽입으로 인해 제한된 길이의 데크가 maxlen 이상으로 커지면, IndexError가 발생.

  + pop() : 데크의 오른쪽에서 요소를 제거하고 반환. 요소가 없으면, IndexError를 발생.

  + popleft() : 데크의 왼쪽에서 요소를 제거하고 반환. 요소가 없으면, IndexError를 발생.

  + remove(value) : value의 첫 번째 항목을 제거. 찾을 수 없으면, ValueError를 발생.

  + reverse() : 데크의 요소들을 제자리에서 순서를 뒤집고 None을 반환.
    + ```reversed(deque)```는 파이썬 내장함수 이므로 지원함.

  + rotate(n=1) : 데크를 n 단계 오른쪽으로 회전. n이 음수이면, 왼쪽으로 회전.

  + maxlen : 데크의 최대 크기 또는 제한이 없으면 None. (read-only attribute)

<br><br>

### Counter
<hr style="border-top: 1px solid;"><br>

+ 해시 가능한 객체를 세는 데 사용하는 딕셔너리 서브 클래스

<br>

+ ```Counter([iterable-or-mapping])```

  + 요소가 딕셔너리 키로 저장되고 개수가 딕셔너리값으로 저장되는 컬렉션

  + 개수는 0이나 음수를 포함하는 임의의 정숫값이 될 수 있음

<br>

```python
c = Counter()                           # a new, empty counter
c = Counter('gallahad')                 # a new counter from an iterable
c = Counter({'red': 4, 'blue': 2})      # a new counter from a mapping
c = Counter(cats=4, dogs=8)             # a new counter from keyword args
```

<br>

```python
c = Counter(['eggs', 'ham'])
c['bacon']  # 0                         # count of a missing element is zero
```

<br>

```python
완전히 제거하려면 del 사용

c['sausage'] = 0                        # counter entry with a zero count
del c['sausage']                        # del actually removes the entry
```

<br><br>

**Counter methods**

elements()
: 반복되는 요소에 대한 이터레이터를 만나는 순서대로 반환. 
: 요소의 개수가 1보다 작으면 무시.

```python
# ex)

>> c = Counter(a=4, b=2, c=0, d=-2)
>> sorted(c.elements()) # ['a', 'a', 'a', 'a', 'b', 'b']
```

<br>

most_common(n)
: n개의 최빈값을 순서대로 나열
: n이 생략되거나 n=None이면 모든 요소 반환
: 개수가 같은 요소는 처음 발견된(또는 입력된) 순서를 유지함. 

<br>

```python
# ex)
>> Counter('abracadabra').most_common(3)
>> [('a', 5), ('b', 2), ('r', 2)]
```

<br>

substract(iterable-or-mapping)
: iterable이나 다른 mapping에서 온 요소에 대해서 같은 요소끼리의 개수를 뺌.

<br>

```python
>> c = Counter(a=4, b=2, c=0, d=-2)
>> d = Counter(a=1, b=2, c=3, d=4)
>> c.subtract(d)
>> c # Counter({'a': 3, 'b': 0, 'c': -3, 'd': -6})
```

<br>

total()
: 개수 총합 반환.

<br>

```python
>> c = Counter(a=10, b=5, c=0)
>> c.total() # 15
```

<br><br>
<hr style="border: 2px solid;">
<br><br>

## binascii
<hr style="border-top: 1px solid;"><br>

Link 
: <a href="https://docs.python.org/ko/3/library/binascii.html" target="_blank">docs.python.org/ko/3/library/binascii.html</a>

<br>

+ ```binascii.a2b_hex(hexstr)``` or ```binascii.unhexlify(hexstr)``` : hex to string
  + ```hexstr``` 인자는 짝수개의 16진수 값으로 int가 아닌 bytes, buffer or ascii 의 형태여야 함.
  + 변수에 저장하고자 하면 ```str(binascii.unhexlify(hexstr), 'ascii')```로 저장하면 됨.

<br><br>
<hr style="border: 2px solid;">
<br><br>

## threading
<hr style="border-top: 1px solid;"><br>

Link 
: <a href="https://docs.python.org/ko/3/library/threading.html" target="_blank">docs.python.org/ko/3/library/threading.html</a>

<br>



<br><br>
<hr style="border: 2px solid;">
<br><br>
