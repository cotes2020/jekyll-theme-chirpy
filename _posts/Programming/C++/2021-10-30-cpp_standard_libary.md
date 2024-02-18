---
title : C++ Standard Library
categories : [Programming, C++]
tags : [C++ Standard Library]
---

## C++ Standard Library Docs
<hr style="border-top: 1px solid;"><br>

Link 
: <a href="https://docs.microsoft.com/ko-kr/cpp/standard-library/cpp-standard-library-header-files?view=msvc-160" target="_blank">C++ 표준 라이브러리 헤더 파일</a> 

<br>

함수, 레퍼런스 검색 
: <a href="http://cplusplus.com/" target="_blank">cplusplus.com/</a>  

<br><br>
<hr style="border: 2px solid;">
<br><br>


## string
<hr style="border-top: 1px solid;"><br>

```cpp
#include <string>

string str;

str.front(), back()

str.begin(), end(), rbegin(), rend()

str.size(), length()

str.c_str() /* string -> char로 변환 */

str.clear()

str.empty()

str.compare(str2) /* str2와 비교 */

str.compare(pos, len, str2) /* str[pos] ~ str[pos+len-1]을 str2와 비교 */

str.compare(pos, len, str2, pos2, len2) 
/* str[pos] ~ str[pos+len-1]을 str2[pos2] ~ str2[pos2+len2-1]와 비교 */

str.substr(pos,len) /* str[pos] ~ str[pos+len-1] 까지 substr */

str.find(str)
```

<br><br>
<hr style="border: 2px solid;">
<br><br>


## vector
<hr style="border-top: 1px solid;"><br>

**가변길이 배열을 사용할 수 있는 라이브러리.**

<br>

```cpp
template <class Type, class Allocator = allocator<Type>>
class vector
```

<br>

```cpp
#include <vector>

vector<DataType> varName;

vector<DataType> varName(size, value) : /* size만큼 value로 초기화 */

varName.size() /* 벡터 요소 개수 반환 */

varName.capacity() /* 벡터에 할당된 메모리 길이 */

/* 
메모리는 2배씩 증가함.
현재 할당한 메모리에 최대로 들어갈 수 있는 원소의 개수를 capacity, 
현재 삽입된 원소의 개수를 size
*/

varName.resize(n) /* 벡터 길이 수정 */

varName.resize(n, value) -> /* 벡터 길이를 n으로 수정 시 기존보다 크면 */
                            /* 추가된 요소의 초기화 값을 value로 설정 */

varName.push_back(value); -> /* 벡터 마지막 원소 뒤에 삽입 */

varName.pop_back(); -> /* 벡터 마지막 값 삭제 */

varName.front(); -> /* 첫번째 원소를 참조 */

varName.back(); -> /* 마지막 원소를 참조*/ 

varName.clear(); -> /* 모든 원소를 제거 */
                 -> /* 메모리는 그대로(size는 줄지만 capacity는 그대로) */

varName.begin() -> /* 첫번째 원소 idx 가리킴, iterator사용 */

varName.end() -> /* 마지막 원소 다음 idx 가리킴, iterator 사용 */

varName.rbegin(), rend() -> /* 역으로 가리킴, reverse_iterator 사용 */

varName.empty() -> /* 비었으면 true, 아니면 false */

varName.erase(position) -> /* position에 위치한 값 제거 */ 
                          /* const_iterator position */

varName.erase(first, last) -> /* first부터 last-1에 위치한 값 제거 */
                              /* const_iterator first, last */

varName.insert(position, value) -> /* position에 value 삽입 */

varName[key] -> /* operator[], 요소 접근 */

varName = vector -> /* operator=, 백터의 요소를 다른 벡터의 복사본으로 변경 */
```

<br>

**연산자는 비교연산자만 사용 가능!!! (>, >=, <, <=, ==, !=)**

<br><br>
<hr style="border: 2px solid;">
<br><br>

## stack
<hr style="border-top: 1px solid;"><br>

연산자에는 ```!=, <, >, <=, >=, ==``` 이 있음.

<br>

```cpp
template <class Type, class Container= deque <Type>>
class stack
```

<br>

```cpp
#include <stack>

//Explicitly declares a stack with deque base container
stack <char, deque<char> > dsc2;

// Declares a stack with vector base containers
stack <int, vector<int> > vsi1;

// Declares a stack with list base container
stack <int, list<int> > lsi;

vector<int> v1;
v1.push_back( 1 );
stack <int, vector<int> > vsi2( v1 );     // vsi2.top() == 1

-> 비어 있거나 기본 컨테이너 개체의 복사본인 스택을 생성
```

<br>

```cpp
stack<DataType> varName;

varName.empty()     // -> 비었으면 true, 아니면 false

pop()     // -> pop

push()    // -> push

size()    // -> size 반환

top()     // -> 젤 윗 값 반환
```

<br><br>
<hr style="border: 2px solid;">
<br><br>

## queue
<hr style="border-top: 1px solid;"><br>

queue는 stack에서 몇가지만 더 추가하면 됨.

<br>

```cpp
template <class Type, class Container = deque <Type>>
class queue
```

<br>

```cpp
#include <queue>

queue<DataType> varName;

push()

pop()

empty()

size()

back()    // -> 마지막 요소 반환

front()   // -> 첫번째 요소 반환
```

<br>

**queue의 연산자에는 비교연산자만 사용 가능.**

<br><br>
<hr style="border: 2px solid;">
<br><br>


## priority_queue (heap)
<hr style="border-top: 1px solid;"><br>

가장 큰 요소가 항상 최상위 위치에 있도록 요소를 정렬

<br>

```cpp
template <
class Type, 
class Container= vector <Type>,   // default : vector
class Compare= less <typename Container ::value_type>   // default : 오름차순
>
class priority_queue
```

<br>

```cpp
#include <queue>

priority_queue<DataType> varName;

push()

pop()   // -> 가장 큰(작은) 값 pop

size()

empty()

top()   // -> 가장 큰 요소 반환
```

<br>

디폴트는 max heap이므로 min heap으로 사용 시 2가지 방법이 있음.

<br>

+ ```priority_queue<int,vector<int>, greater<int>) pq;```
  
  + ```greater<DataType>``` -> 오름차순, ```less<DataType>``` -> 내림차순
  
  + priority queue에서는 그 특징때문에 반대가 됨.

<br>

+ 값을 넣을 때 음수로 바꿔서 넣은 뒤 ```-pq.top()```으로 출력.
  
<br>

```cpp
int main(){

  priority_queue<int> pq;
  for(int i=0; i<5; ++i){
    pq.push(-i);
  }

  while(!pq.empty()) {
    cout << -pq.top() << '\n';
    pq.pop();
  }
}
```

<br><br>
<hr style="border: 2px solid;">
<br><br>


## deque
<hr style="border-top: 1px solid;"><br>

```cpp
template <class Type, class Allocator =allocator<Type>>
class deque
```

<br>

```cpp
deque<DataType> varName;
deque<DataType>::iterator varName2;

varName.begin(), end()    // -> 처음, 마지막+1 인덱스 반환, iterator 사용

varName.rbegin(), rend()  // -> 역순, reverse_iterator 사용

varName.front(), back()   // -> 처음, 마지막 값 반환

varName.clear()

varName.empty()

varName.erase(position)   // -> position에 위치한 값 제거

varName.erase(first, last)  // -> first부터 last-1에 위치한 값 제거

varName.push_front(val), push_back(val)

varName.pop_front(), pop_back()

varName.resize(size, [val])   // : deque 새 크기 지정
                              // val 값 지정 시 val 값 추가, 생략 시 0 추가
```

<br><br>
<hr style="border: 2px solid;">
<br><br>

## pair
<hr style="border-top: 1px solid;"><br>

```cpp
// #include <utility> -> vector, algorithm 헤더에 추가되어 있음.
#include <vector>

pair<int, int> p1;
cout << p1.first << ' ' << p1.second << '\n'; // 0 0 

p1 = make_pair(1, 2);
cout << p1.first << ' ' << p1.second << '\n'; // 1 2 

pair<int,int> p2(1,2); // p2.first == 1, p2.second == 2

// Pair 속에 Pair 를 중첩해서 사용 가능
pair<pair<int,int>, pair<int,int>>  
p = make_pair(make_pair(1,2), make_pair(3,4));

cout << p.first.first << ' ' << p.first.second << ' '; // 1 2
cout << p.second.first << ' ' << p.second.second << '\n'; // 3 4 \n
```

<br><br>
<hr style="border: 2px solid;">
<br><br>

## heap

<br>

### map
<hr style="border-top: 1px solid;"><br>

연결된 키 값을 기반으로 **요소 값을 효율적으로 검색**하는 다양한 크기의 컨테이너

각 요소가 **데이터 값**과 **정렬 키**를 갖고 있는 쌍인 컬렉션에서 **데이터의 저장과 검색에 사용**됨.

**키 값은 고유**하며 **데이터를 자동으로 정렬하는 데 사용**됨.

<br>

map에 있는 요소의 값은 직접 변경할 수 있음.

요소가 지정된 비교 함수에 따른 ```키 값``` 으로 정렬되므로 정렬되어 있음.
: 자동 정렬된다는 뜻

키 값은 중복되면 안됨.
: 중복되면 마지막에 넣은 값으로 갱신.

<br>

```cpp
template <class Key,
    class Type,
    class Traits = less<Key>,
    class Allocator=allocator<pair <const Key, Type>>>
class map;
```

<br>

map은 key, value 의 형태로 되어 있음. (cpp의 pair, python의 dict)

<br>

```
Key : key DataType 
Type : value DataType

Traits : 두 요소 값을 키로 정렬, 비교하여 
         상대적 순서를 결정할 수 있는 함수 개체를 제공.
         이 인수는 선택 사항이며 기본값은 이진 조건자 less<Key>.
```

<br>

```cpp
#include <map>

map<DataType> var;

iterator --> pair<const Key, Type>
/*
iter->first, (*Iter).first     ---> key
iter->second, (*Iter).second   ---> value
*/

var.size()

var.begin(), var.end() -> iterator 사용

var.rbegin(), rend() -> 역순, reverse_iterator 

var.clear()

var.count(key) -> key에 해당하는 요수의 개수 반환 >> (1 or 0)

var.erase(key) -> key에 위치한 값 제거

var.erase(key) -> iterator key에 위치한 값 제거

var.erase(first, last) -> iterator first ~ iterator last-1에 위치한 값 삭제

var.contains(key) -> 지정된 key의 값이 있는지 확인 -> 있으면 true, 없으면 false
                  /* -> C++20에서 새롭게 추가된 내용 */

var.empty()

var.equal_range(key) -> lower_bound(x), upper_bound(x) 값을 pair 형태로 리턴.

/* Usage
pair<map<DataType,DataType>::iterator,map<DataType,DataType>::iterator> ret;
ret=equal_range(key);

ret.first->first     -------> lower_bound(key)
ret.first->second    -------> lower_bound(value)

ret.second->first    -------> upper_bound(key)
ret.second->second   -------> upper_bound(value)
*/

var.find(key) -> key 값에 해당하는 위치 반환 (iterator 리턴형)
                 없으면 end() 리턴
                 
var.insert(pair<DataType,DataType>(key, value))
var.insert(make_pair(key,value))

var.lower_bound(key), upper_bound(key)

var[key] -> value, operator[]

var2 =  var -> copy, operator=
```

<br><br>

### set
<hr style="border-top: 1px solid;"><br>

연관된 키 값을 기준으로 하며 **요소 값의 효율적인 검색을 지원**하는 가변 크기 컨테이너인 연관 컨테이너. 

해당 요소가 컨테이너 내에서 지정된 비교 함수에 따라 ```키 값```을 기준으로 정렬됨.
: 자동 정렬 된다는 뜻.

각각의 요소가 반드시 고유한 키를 가지고 있어야 함. 
: 키 값은 중복되지 않음.

균형 이진 트리로 구현

<br>

```cpp
template <class Key,
    class Traits=less<Key>,
    class Allocator=allocator<Key>>
class set
```

<br>

```
Key -> set에 저장되는 요소 데이터 형식

Traits -> 두 요소 값을 정렬 키로 비교하여 set에서 상대적인 순서를 결정할 수 있는 
          함수 개체를 제공하는 형식. 
          이 인수는 선택 사항이며 이진 조건자 less<Key>가 기본값.        
```

<br>

```cpp
#include <set>

set<DataType> var;

var.begin(), var.end() -> iterator 사용

var.rbegin(), rend() -> 역순, reverse_iterator 사용

var.clear() -> set에 있는 모든 값 제거

var.erase(x) -> set에 있는 모든 x 값을 제거

var.erase(x) -> iterator x가 가리키는 값 제거

var.erase(x,y) -> iterator x,y가 가리키는 위치 사이에 있는 요소들 제거

var.contains(key) -> key 값이 요소에 있으면 true, 없으면 false

var.count(key) -> key 값이 요소에 있으면 1 (key는 중복되지 않으므로 1개), 없으면 0

var.empty()

var.equal_range(key) -> lower_bound(key), upper_bound(key) 값을 pair 형태로 리턴

/* >>>>>> Usage  
pair<set<DataType>::iterator, set<DataType>::iterator> res = equal_range(key);
res.first --> lower_bound(key)
res.second --> upper_bound(key)
*/

var.find(key) -> key 값을 찾아서 위치를 반환 (iterator 리턴형)
                 없으면 end()  

var.insert(key) -> key 값을 set에 삽입

var.size()

var.lower_bound(key), key.upper_bound()
```
```
insert : O(logN)

erase : O(logN)

find : P(logN)
```

<br><br>
<hr style="border: 2px solid;">
<br><br>

## algorithm

<br>

### sort
<hr style="border-top: 1px solid;"><br>

```cpp
#include <algorithm>

sort(start, end, compare)
stable_sort(start, end, compare)

/*

start, end는 주소, end는 마지막 다음 원소 주소

compare는 user custom function으로 내림차순 등 원하는 방식으로 정렬 
단, 리턴형은 bool형

*/
```

<br>

sort는 ```unstable sort```임. 

```stable sort```는 중복된 값들의 순서를 유지, ```unstable sort```는 순서 보장 X.

<br>

+ sort user custom function

<br>

```cpp
auto cmp = [](pair<int, int> a, pair<int, int> b) {
  if(a.first == b.first) return a.second < b.second;
  return a.first < b.first;
} // 변수
```

<br>

```cpp
bool cmp(pair<int, int> a, pair<int, int> b) {
  if(a.first == b.first) return a.second < b.second;
  return a.first < b.first;
} // 함수

sort(arr, arr + N, cmp);
sort(arr, arr + N, greater<DataType>); // greater 구조체 -> 내림차순
```

<br><br>

### binary_search, lower_bound, upper_bound
<hr style="border-top: 1px solid;"><br>

+ ```binary_search```

```cpp
#include <algorithm>

binary_search(first, end, val)

/*
정렬된 범위에서 지정한 값을 찾는 함수

first, end는 주소이며 end는 마지막 값 다음 위치 주소
val을 찾으면 true, 없으면 false
*/
```

<br>

+ ```lower_bound```

```
int *p = lower_bound(first, end, val)

/*
정렬된 범위에서 지정된 값보다 << 크거나 같은 값 >>을 갖는 
첫 번째 요소의 위치를 찾는 함수
*/
```

<br>

+ ```upper_bound```

```
int *p = upper_bound(first, end, val)

/*
정렬된 범위에서 지정된 값보다 << 큰 값을 >>갖는 
첫 번째 요소의 위치를 찾는 함수
*/

/* 만약 범위 안에 값이 없다면 end+1 리턴 */
```

<br>

```c++
#include <iostream>
#include <algorithm>
using namespace std;

int main() {
    int arr[10]={1,2,3,4,5,6,7,8,9,10};
    for(int i=0; i<10; ++i){
        cout << arr[i] << ' ';
    }
    cout << '\n';
    
    int val; cin >> val;
    cout << (binary_search(arr, arr + 10, val) ? "True":"False") << '\n';
    while(true) {
        int val;
        cin >> val;
        int *pos = lower_bound(arr, arr+10, val);
        int *pos2 = upper_bound(arr, arr+10, val); 
        cout << pos-arr << ' ' << pos2-arr << '\n'; 
    }
}
```

<br>


```lower_bound```, ```upper_bound```를 이용해서 특정 범위 내 원소의 개수, 특정한 값의 개수를 구할 수 있음.

```cpp
int main() {
    vector<int> arr = { 1,3,5,5,7,8,8,10,10,11,13 };
    cout << "5 이상 11 이하의 갯수 : " 
    << upper_bound(arr.begin(), arr.end(), 11) - lower_bound(arr.begin(), arr.end(), 5);
    
    cout << "5의 갯수 : " 
    << upper_bound(arr.begin(), arr.end(), 5) - lower_bound(arr.begin(), arr.end(), 5);
    return 0;
}
```

<br><br>
<hr style="border: 2px solid;">
<br><br>

