---
title: "C++"
# description: ""
categories: [컴퓨터, 프로그래밍]
tags: []
image: "/assets/img/background/kururu-lab.jpg"
hidden: true

date: 2023-02-22. 02:35
# last_modified_at: 2024-02-17. 20:13
# last_modified_at: 2024-02-21. 18:09
# last_modified_at: 2024-02-23. 03:50
# last_modified_at: 2024-02-24. 00:23
# last_modified_at: 2024-02-24. 19:30
# last_modified_at: 2024-02-25. 01:17
# last_modified_at: 2024-03-01. 02:02
# last_modified_at: 2024-04-27. 21:52
# last_modified_at: 2024-07-12. 22:06
# last_modified_at: 2024-08-19. 14:07
# last_modified_at: 2024-09-16. 16:35
# last_modified_at: 2025-04-28. 19:17 # 메모
last_modified_at: 2025-05-28. 21:59 # +Q
---

## Q

---

- 엔진에서 쓰인 C++에 대한 지식
  - OOP 개념
  - 템플릿
  - STL

## Modern C++ (VS Old C++)

---

- 자동 타입 추론 (auto 키워드 사용)
- 범위 기반 루프
- 람다식
- 스마트 포인터
- 벡터, 목록 및 맵과 같은 표준 템플릿 라이브러리 (STL) 컨테이너
- STL 알고리듬
- std::string 및 std::wstring 형식
- 오류 조건을 보고하고 처리하는 예외
- STL std::atomic\<\> 를 사용하여 잠금 없는 스레드 간 통신

## 메모

---

### Func, Hack

[iter find](https://modoocode.com/261)  
[string find](https://naakjii.tistory.com/104)  

find(v.begin(), b.end(), )  
s.find(c)  

```cpp
if (int condition = get_status())

to_string(10);

string::npos == -1  
find 실패 시 리턴  

pow sqrt -> cmath  
min max -> algorithm  

[rotate](https://notepad96.tistory.com/59)  

toupper, tolower  
isupper, islower, isdigit, isalpha  

// split 대신
#include <sstream>
stringstream ss(s);  
string word;
while (ss >> word)
{
    cout << word << endl;
}

ceil ceiling 올림  
round 반올림  
floor floor 내림  

vector<T> v2(v1)  

memset(ptr, value, size)  

오름차순 정렬되어 있는 자료구조에 대하여,  
lower_bound: k <= 요소가 배열 몇 번째에 처음 등장하는 지  
upper_bound: k < 요소가 배열 몇 번째에 처음 등장하는 지  
iterator로 반환되므로 자료구조 주소를 빼주면 인덱스가 나옴  

while(!cin.eof())
    cin >> temp;

while(cin >> temp);
```

### Function Prototype

```cpp
// 함수의 시그니처 (매개변수개수, 타입, 순서)  
// 함수 원형 Function Prototype
int spuare(int n);
int spuare(int);
```

```cpp
// & 별명

// 문자열 \> \< 사전순 비교  
// 크면 앞에 있는  

cin.ignore(); // 엔터키 없애기
getline(cin, s);
get(cin, s);

class Circle
{
    public:
    double calcArea();

    int radius;
    // ...
}

double Circle::calcArea()
{
    // ...
}
```

멤버 함수 이름 관례적으로 소문자, 동사 - 명사/형용사  

멤버 함수를 클래스 외부에 저장할 수 있는 기능  
멤버 함수를 클래스 외부에서 정의하려면 함수들의 원형(프로토타입)만 정의한다.  

:: 연산자는 이름공간 (namespace)를 지정하는 연산자  

```cpp
int value; // 선언과 정의
double sqrt (double); // 선언
double sqrt (double) { /* ... */ } // 선언과 정의
```

멤버 함수들을 외부에 정의하는것은 그 자체로도 상당한 의미가 있다.  
우리가 클래스를 작성하는 이유는 여러 소스 파일에서 이 클래스를 사용하기 위해서이다.  
다른 소스 파일에서 클래스를 사용하려면 클래스 선언을 포함하여야 한다.  

만약 하나의 파일에 클래스에 대한 모든 것이 들어 있다면, 상당한 양이 될 수 있다.  
따라서 대부분의 프로그래머들이 선호하는 방법을 클래스를 헤더 파일과 소스 파일로 나누어서 작성하는 방법이다.  

\+ 클래스를 사용하는 사람들은 클래스의 자세한 구현에는 관심이 없다  
클래스를 사용하는데 필요한 최소한의 정보만 있으면 된다.  
개발자 자신도 헤더를 여러 소스 파일에서 사용할 수도 있다  

```cpp
class SomeClass
{
    int hour;
    int m;
    SomeClass(int h, int n = 0): hour(h), minute(m); // Initializer List 초기화 리스트
    {

    }
    SomeClass(int h, int n = 0): hour{h}, minute{m}; // Initializer List 초기화 리스트
    {
        
    }
};

//생성자
SomeClass a; // X
SomeClass a(10, 25);// Old, 함수 생성과 혼동 가능성
SomeClass a { 10, 25 };
SomeClass a = { 10, 25 };

int i { 5 };
```

객체복사된다  
C#은 ref으로 들어가는데  
C++은 & 참조자 연산자 함수 인수에다 붙여야 한다  
C처럼 주소를 넘기지 않아도 된다  

객체 배열, 객체에 기본 생성자가 정의 되어 있어야 한다  

`int* p` 포인터 선언용 *  
`p = &num;` 주소 연산자 (참조자 연산자 아님)  
`a = *p;` 간접 참조 연산자 Dereferecing, Indirection  

NULL == 0 (정수로 인식)  
nullptr  

동적 메모리 할당 Dynamic Memory Allocation  
히프에 할당받는 메모리  

요청 시 메모리가 부족하면 bad_alloc 오류 exception  

Smart Pointer  
동적 메모리 할당 후에 해제를 잊어도 자동으로 해제  
자동으로 nullptr 초기화  

```cpp
unique_ptr<int> p(new int); // new int: 포인터 초기화
unique_ptr<int[]> p(new int[]);
```

unique_ptr  
스마트 포인터의 일종  
기존 포인터를 감싼 객체  
객체가 삭제되면, 포인터가 가리키는 메모리 공간도 해제  
실행 시간의 부담이 전혀 없어서 타 언어 GC에 비하면 성능의 향상을 꾀할 수 있다  

@ TODO  
\* unique_ptr  
포인터에 대해 오직 하나의 소유자만 허용한다. shared_ptr이 필요하다는 점을 확실히 알지 못하는 경우에만 사용한다. 새 소유자로 이동할 수 있지만 복사하거나 공유할 수 없다. 노후된 auto_ptr을 대체한다.  

\* shared_ptr  
참조 횟수가 계산되는 스마트 포인터. 원시 포인터 하나를 여러 소유자에게 할당하려고 할 경우 사용한다. 원시 포인터는 모든 shared_ptr 소유자가 범위를 벗어나거나 소유권을 포기할 때까지 삭제되지 않는다.  

```cpp
const int *p1; // 포인터를 통해 참조되는 값이 바뀔 수 없음
int* const p2; // 상수 포인터, 참조하는 값이 바뀔 수는 있지만, 참조하는 주소가 바뀔 수 없음
const int* const p3; // 참조 값도, 참조하는 주소고 바뀔 수 없음

// const ~ * 객체 불변
// * ~ const 포인터 자체 불변
```

```cpp
int getRadius() const
{
    return radius;
}
// 함수안에서 멤버 변수를 변경하는 것이 금지됨

const SomeClass *pConstObj = new SomeClass();
pConstObj.getRaius();
// const 객체를 가리키는 포인터는, const 함수만 호출할 수 있다. (일반함수 호출 불가능)
}
```

```cpp
#include <utility>
pair<int, int> p;

{
    pair<int, int> p = make_pair(10, 13);
    pair<int, int> p = {4, 6}; // C++11
}

p.first
p.second

if (p1 < p2>) // 앞쪽 뒤쪽 비교
```

```cpp
void Some(Pizza *p) { /* ... */ }
Some(&newP);

void Some(Pizza &p) { /* ... */ }
Some(newP);

// 이거나 저거나
```

매개변수로 객체를 넘기면, 함수 파트에서 객체를 생성할 때 해당 객체의 내용을 복사해서 생성된다  
따라서 일반적인 생성자가 호출되는 것이 아니라 복사 생성자 Copy Constructor 라는 특별한 생성자가 호출된다  
기본적인 버전은 컴파일러가 만들어서 사용  

함수에서 객체를 반환할 때도 마찬가지  
return 으로 넘긴 객체를 복사하여 반환 객체를 생성한다.  

- 같은 종류의 객체로 초기화 하는 경우
  - `MyClass obj(obj2);`
- 객체를 함수에 전달하는 경우
- 함수가 객체를 반환하는 경우

```cpp
// 복사 생성자
MyClass ( const MyClass& other)
{ // other로 현재 객체를 초기화 }

MyClass (MyClass other)
// 이건 무한 루프를 생성하는 관계로 사용하면 안된다
```

```cpp
class MyArray
{
    public:
    int size;
    int* data;

    MyArray(int size)
    {
        this->size = size;
        data = new int[size];
    }

    ~MyArrat()
    {
        if (data != NULL) delete[] this->data;
    }
};

int main()
{
    MyArray buffer(10);
    buffer.data[0] = 1;
    
    {
        MyArray clone = buffer;
    } 
    buffer.data[0] = 2; // 이때 오류 발생
    // 기본 복사 생성자로 buffer의 값이 clone에 복사가 되는데 (얕은 복사 Shallow Copy)
    // 이때 data의 경우 똑같은 주소가 복사됨 (새로 공간이 할당되는 것이 아니라, 동일한 공간을 buffer와 clone이 공유하며 동시에 가리키는)
    // clone이 { } 을 넘어 파괴자가 호출될때 data를 할당 해제(반납)하는데
    // buffer와 clone이 가리키는 data 주소가 똑같음으로
    // buffer에서 data에 접근하면 메모리가 이미 해제된 주소라 오류가 생김

    return 0;
}
```

```cpp
// 이런 경우 직접 구현해주면 됨
MyArray::MyArrat(const MyArray& other)
{
    this->size = other.size;
    this->data = new int[other.size];
    for (int i = 0; i < size; i ++)
        this->data[i] = other.data[i];
}
```

최신 버전 C++에서는 `shared_ptr`을 이용해서 이 문제를 해결  
이를 사용하면 얼마나 많은 객체들이 동일한 동적 데이터를 참조하고 있는지 알 수 있다  
만약 카운트 값이 0이 되면 동적 데이터가 삭제된다

```cpp
// 복사 생성자 vs 대입 연산자
// 이미 생성된 객체를 다른 객체에 복사하는 경우에는 호출되지 않음, 이때는 대입 연산자가 적용된다

MyArray buffer1(20);
MyArray buffer2(30);
buffer2 = buffer1; // 이것은 대입 연산

MyArray s1; // 일반 생성자
MyArray s2 = s1; // 복사 생성자
MyArray s2(s1); // 복사 생성자
MyArray s2{s1}; // 복사 생성자
```

대입 연산자 역시 모든 멤버 변수의 값이 복사된다  
같은 타입의 객체 끼리는 대입 연산이 가능하다  
C++에서는 대입 연산자를 개발자가 재정의하여 사용할 수 있다  

`==` 연산자는 연산자 중복 (Operator Overloading)이라는 메카니즘을 통해 재정의해줘야 한다  

객체 지향에서 코드를 재사용하는 방법  

1. is-a 관계: 객체 지향 프로그래밍에서 is-a의 개념은 상속을 기반으로 한다. "A는 B유형의 물건" 이라고 말하는 것과 같다. Apple은 과일의 일종, Car는 자동차의 일정
2. has-a 관계: has-a는 하나의 객체가 다른 객체를 가지고 있는 관계이다. Car에는 Engine이 있고, House에는 Bathroom이 있다.

정적 변수 Static Variable  

```cpp
class Circle
{
    static int count;
    const static int MAX_CIRCLES = 300;
    static int getCount()
    {
        return count;
    }
}

// 초기화는 클래스 외부에서
int Circle::count = 0;
Circle::getCount();
```

## 연산자 중복

---

연산자 중복 (Operator Overloading)  
일종의 다형성 기법  
string에서 연산자 중복을 사용하고 있다 `+,-, &, /, =, ==, !=, ++, --` 등  

중복할 수 없는 연산자  
`:: 범위지정 연산자, . 멤버 선택 연산자, .* 멤버 포인터 연산자, ?: 조건연산자`

```cpp
반환형 operator연산자(멤버 변수 목록)
{
    // ...
}

MyVector MyVector::operator+(const MyVector& v2)
{
    MyVector v;
    v.x = this->x + v2.x;
    v.y = this->y + v2.y;
    return v;
}

bool operator== (Time &t2)
{
    return blabla;
}

bool operator!= (Time &t2)
{
    return !(*this == t2);
}

// ++counter
Counter& operator++()
{
    // blabla
    return *this;
}

// counter++
const Counter operator++(int i)
{
    Counter temp = { *this }; // 현재 상태 저장
    // blabla
    return temp;
} 
// const 객체를 반환하기에 반환된 임시 객체는 변경이 불가능
// (v++)++; 오류
// ++(++v); OK
```

대입 연산자도 연산자 중복을 통하여 이루어진다  
대입 연산자의 경우, 개발자가 중복하지 않았더라도 기본 대입 연산자가 자동으로 생성되고 이것을 통하여 객체 간의 대입 여산이 이루어진다.  
기본 대입 연산자는 단순히 한 객체의 모든 멤버들을 다른 객체로 복사한다  

주의  
대입 연산자의 매개변수는 일반적으로는 객체에 대한 상수 참조자이지만 코드에 따라서 그냥 객체이거나 상수가 아닌 참조자일 수 있다  
이 경우에는 컴파일러에 의하여 매개 변수가 적절하게 변한된다  
컴파일러가 매개 변수를 변환할 수 있으려면 대입 연산자는 반드시 멤버 함수이어야 한다  

```cpp
Box& operator=(const Box& b2)
{
    this->length = b2.length;
    this->width = b2.width;
    this->height = b2.height;
    return *this;
}
```

반환값에 주의.  
대입 연산자는 참조자를 반환하여야 한다. 대입 연산자는 연속하여 적용될 수 있기 때문이다  
`b3 = b2 = b1;`  

인덱스 연산자 []은 번호를 가지고 해당되는 요소를 찾는 연산자  
인덱스 중복 정의하여 배열의 경계에서 벗어나는 것을 막는다던지?  

```cpp
someType &operator[](int i)
{
    return blabla;
}
```

포인터 연산자의 중복  
포인터와 관련된 두 가지의 연산자인 *과 ->도 중복이 가능하다  
특히 이 두 연산자를 중복하면 포인터와 비슷한 클래스를 작성하여 사용할 수 있다  
두 연산자 모두 멤버 함수로 작성하는 것이 좋다  

```cpp
int* operator->() const { return p; }
int& operator*() const { return *p; }
```

이런 포인터가 그냥 포인터보다 나은 점  
소멸자에서 만약 포인터가 소멸되면 동적 할당 받은 공간도 반납하게 되어 있다  
사용자가 동적 메모리 공간을 반납할 필요가 없어지는 것이다  

이렇게 포인터 연산 중복 정의를 이용해 만들어진 향상된 포인터를 스마트 포인터 라고 한다  
최신 C++에서는 이러한 연산자 중복 정의와 템플릿을 이용하여 스마트 포인터들을 정식으로 제공하고 있다  
스마트 포인터는 포인터와 같이 동작하면서 객체가 스마트 포인터를 통하여 접근될 때마다 필요한 어떤 동작을 수행하는 객체이다  

unique_ptr는 연산자 중복과 템플릿을 사용해서 만든 스마트 포인터  
C++11 도입  
기존 포인터 감싸서 객체로 만든다  
객체에 소멸자를 추가하여서 객체가 소멸될 때, 포인터가 가리키는 메모리 공간도 해제한다  
스마트 포인터는 실행 시간의 부담이 전혀 없어서 자바나 C#의 쓰레기 수집기에 비하면 성능의 향상을 꾀할 수 있다  

```cpp
int main()
[
    unique_ptr<int> p(new int);
    *p = 99;
    // 여기서 지역변수인 p가 삭제되면서 소멸자가 호출되고 소멸자에서
    // 동적 메모리도 함께 삭제하기 때문에 메모리 누수가 발생하지 않는다
]
```

unique_ptr에는 템플릿 기술이 추가되어 있다  
템플릿은 자료형도 변수처럼 만들어서 바꿀 수 있도록 하는 기법  
스마트 포인터는 근본적으로 객체로 포인터를 감싸는 것이기 때문에 객체가 소멸되면 소멸자가 호출되기 때문에 소멸자에서 할당받은 동적 메모리 공간을 삭제할 수 있다  
자바와 같은 언어와 차이점이라면 C++에서는 별도의 GC가 실행되지 않는다는 것  
동적 메모리는 컴파일 단계에서 모두 안전하게 처리되기 때문에 실행될 때는 쓰레기 수집기가 없어도 되고 실행 속도가 빨라진다  

## 프렌드

---

프렌드  
외부의 클래스나 함수가 자신의 내부 데이터를 사용하도록 허가할 수 있다  
프렌드는 전역함수일수도 잇고 어떤 클래스의 멤버 함수 일수도 있고 아니면 전체 클래스일 수도 있다  
특정한 클래스를 프렌드로 지정하면 그 클래스의 모든 멤버 함수는 내부 데이터를 참조할 수 있다  
하나의 예로 특정한 함수를 프렌드로 지정하여 보자  
프렌드를 선언하기 위해서는 클래스 안에 프렌드로 지정하고 싶은 함수의 원형을 적고 원형 앞에 friend 라는 키워드를 붙인다  

```cpp
class MyClass
{
    friend void sub();
};
```

프렌드 함수 선언은 클래스 안의 어떤 위치에서도 가능하지만 일반적으로 시작 부분에 두는 것이 바람직하다  
프렌드 함수 선언은 클래스의 멤버가 아니므로 public이나 private의 영향을 받지 않는다  
프렌드 함수의 원형은 비록 클래스 안에 포함되어 있으니 멤버 함수는 아니면 프렌드 함수의 본체는 외부에서 따로 정의된다  
프렌드 함수는 클래스 내부의 모든 멤버 변수를 사용할 수 있으면 어떤 멤버 함수도 호출할 수있다  

```cpp
class MyClass
{
    int someVar;
    public:
    friend class SomeClass; // SomeClass는 MyClass의 친구가 된다
    friend void SomeFunc(MyClass myClass); // 프렌드 선언
};
void SomeFunc(MyClass myClass) // 프렌드 정의
{
    cout << myClass.someVar;
}

class SomeClass
{
    public:
    void print(MyClass myClass)
    {
        cout << myClass.someVar;
    }
}
```

프렌드 함수는 두 개의 객체를 비교하거나 연산하는 경우에 많이 사용된다  
프렌드를 사용하지 않으면 약간은 이해하기 어려운 멤버 함수 형태를 사용하여야 하기 때문이다  

멤버함수와 프렌드 함수는 비슷한 역할을 한다  
함수가 수행하는 작업이 오직 하나의 객체에만 관련된다면 멤버 함수로 정의  
함수가 수행하는 작업이 두 개 이상의 객체에 관련된다면 프렌드 함수로 정의  

물론 함수가 수행하는 작업이 두 개 이상의 객체에 관련되더라도 접근자와 설정자를 사용하면 똑같이 수행할 수 있다  
하지만 효율성을 생각하면 프렌드 함수로도 정의할 수 있는 것이다  

프렌드 함수나 프렌드 클래스는 객체 지향 프로그래밍의 중요한 원칙, 정보 은닉을 손상하는 것으로 꼭 필요한 경우가 아니면 사용을 자제하여야 한다  

`<<, >>` 연산자 중복 정의  
`cout << 객체`;  
`cin >> 객체`;  

```cpp
friend ostream& operator<<(ostream& os, const MyVector& v)
{
    // ...
    os << "(" << v.x << "," << v.y << ")" << endl;
    return os;
}

friend istream& operator>>(ostream& in, MyVector& v)
{
    // ...
    in >> v.x >> v.y;
    if (!in)
        v = MyVector(0,0); // 입력 오류 처리
    return in;
}
```

cout은 ostream 클래스의 객체  
따라서 멤버 함수로 추가하려면 ostream 클래스 안에 operator<<() 함수를 추가하여야 한다  
하지만 ostream 클래스는 컴파일러가 제공하는 라이브러리에 속하므로 우리가 변경할 수 없다  
따라서 프렌드 함수 형태로 객체 클래스 안에 연산자를 중복 정의하는 수밖에 없다  

또 하나 주의할 점은 연산의 결과로 반드신 ostream 참조자를 반환해야 한다는 점  
ostream 참조자를 반환하지 않으면 다음과 같이 << 연산자가 연속적으로 사용되었을 경우에 제대로 작동하지 않음  
`cout << 객체 << 객체;`  

`>>`, cin  
주의해야 할 점은 두 번째 매개 변수를 const로 선언하면 안된다는 것  
입력 연산자는 입력을 받아서 객체에 저장되어야 하기 때문  

입력 연산자는 입력 시에 발생되는 오류에 대하여 대비를 해야 한다  
잘못된 데이터를 입력받았을 경우 오류가 발생하게 되고, 이때는 변수를 초기화 상태로 만들어 주는 것이 좋다  
입력 단계에서 오류가 발생하면 in는 0이 아닌 값을 반환하게 된다  
따라서 in이 0이 아닌 값을 반환하면 변수에 초기화한 임시 객체를 복사하여 준다  

## 연산자 중복 시에 유의할 점

---

- 새로운 연산자를 만드는 것은 허용되지 않는다, i.e. 지수승을 나타내기 위한 `^` 연산자를 새롭게 정의할 수 없다
- 거의 모든 연산자가 중복이 가능. 하지만 `:: .* , ?:`는 불가능
- 중복된 연산자는 클래스 타입의 피연산자를 반드시 가져야 한다. 즉 내장된 int형이나 double형에 대한 연산자의 의미를 변경할 수는 없다.
- 연산자들의 우선순위나 결합 법칙은 변경되지 않는다
- 만약 +연산자를 중복하였다면 일관성을 위해서 - += -= 연산자도 중복하는 것이 좋다

## 상속

---

```cpp
class ChildClass: 접근지정자 ParentClass { }

// 접근지정자가
// public이면 그대로
// protected면 상속 받는 public 멤버들이 protected로
// private이면 상속 받는 모든 멤버들이 private으로

class ChildClass: 접근지정자 ParentClass, 접근지정자 ParentClass { }
// 똑같은 이름의 멤버가 있으면 자식객체.ParentClass:멤버
```

상속을 나타낼 때 확장 Extend 또는 파생 Derive 이라는 용어를 사용하는 이유도 상속 시 멤버가 증가하기 때문  

부모 클래스 == 수퍼 클래스 == 베이스 클래스  
자식 클래스 == 서브 클래스 == 파생된 클래스 (Derived)  

공통 부분을 한 번에  
중복되는 부분을 최소화  
-> 하나로 정리되어서 관리하기 쉽고 유지 보수와 변경도 쉬워진다  

상속에서 자식과 부모는 "~은 ~이다" 와 같은 is-a 관계가 있다  
따라서 상속의 계층 구조를 올바르게 설계하였는지를 알려면 is-a 관계가 성립하는지를 생각해보면 된다  

만약 "~은 ~을 가지고 있다" 와 같은 has-a (포함) 관계가 성립되면 이 관계는 상속으로 모델링하면 안된다  
이 경우 하나의 클래스에 다른 클래스의 객체를 포함시키면 된다  

자식의 객체는 부모의 객체를 포함하고 있다  
자식 클래스 객체 안의 부모 클래스 부분을 초기화 하기 위해서는 부모 클래스의 생성자도 호출되어야 하는 것이 논리적이다  
실제로 자식의 생성자에서 제일 먼저 하는 일이 부모의 생성자를 호출하는 일이다  
특별히 지정하지 않으면 부모의 `기본` 생성자가 호출된다  

소멸자의 경우 역순으로 호출된다  
즉 자식의 소멸자가 먼저 호출되고 이어서 부모 클래스의 소멸자가 호출된다  

매개변수가 있는 다른 생성자를 호출하려면  
자식의 생성자 헤더의 뒤에 콜론을 추가한 후에 원하는 부모 클래스의생성자를 적어주면 된다  

```cpp
자식생성자(): 부모생성자()
{

}

자식생성자(int x = 0, int y = 0): 부모생성자(x, y)
{

}

자식생성자(int x = 0, int y = 0): 부모생성자(x, y), width(x), height(y)
{

}
```

C#과 달리 그냥 함수 시그니처만 똑같으면 재정의 된다  

C#의 base.  
C++에서 부모클래스::  

## 다형성

---

중복 정의나 연산자 중복 정의도 크게 분류하면 다형성에 포함  

- 다형성
  - 컴파일 시간 다형성
    - 함수 중복 정의
    - 연산자 중복 정의
  - 실행 시간 다형성
    - 가상 함수

실행 시간 다형성은 객체들의 타입이 다르면 똑같은 메시지가 전달되더라도 서로 다른 동작을 하는 것  
한 곳에 모인 동물들이 각자의 소리를 내게 하고 싶으면 어떤 동물인지 신경쓰지 말고 무조건 speak 메시지를 보내면 된다  
메시지를 받는 동물은 자신이 낼 수 있는 소리를 낼 것이다  
똑같은 메시지를 보내지만 객체의 타입이 다르면 서로 다른 결과를 얻는 것이 다형성이다  
중요한 것은 메시지를 보내는 측에서는 객체가 어떤 타입인지 알 필요가 없다는 것  
실행시간에 객체의 타입에 따라서 자동적으로 적합한 동작이 결정된다  
다형성은 객체 지향 기법의 특징 중의 하나로서 동일한 코드로 다양한 타입의 객체를 처리하는 기술  

상향 형변환 Up-Casting  
`Animal *pa = new Dog();`  
자식 객체는 부모 객체를 포함하고 있기 때문에 자식 객체는 부모 객체이기도 하다  
다형성의 핵심 -> **부모 포인터로 자식 객체를 가리킬 수 있다**  

기본적으로 부모로부터 상속받은 부분만 포인터를 통해서 사용할 수 있다.  
부모의 함수를 가상 함수 Virtual Function으로 정의하면 상향 형변환을 통한 포인터라도 객체의 종류에 따라 서로 다른 함수가 호출된다  

```cpp
virtual void speak() { /* ... */ } // 부모에서만 해주면 된다,자식은 그냥 void speak()써도 되는데 이왕이면 똑같이virtual 표시해주면 좋겠지
```

HOW?  
함수 호출을 함수의 몸체와 연결하는 것을 바인딩 Binding 이라고 한다.  
바인딩에는 정적 바인딩과 동적 바인딩이 존재한다  
컴파일 단계에서 모든 바인딩이 완료되는 것을 정적 바인딩이라고 한다 (빠르다, 일반 함수를 대상)  
반대로 바인딩이 실행시까지 연기되고 실행시간에 호출되는 함수를 결정하는 것을 동적 바인딩, 또는 지연 Late 바인딩이라고 한다 (테이블을 사용하여서 실제 호출되는 함수를 결정해야 하므로 느리다, 가상 함수를 대상)  

동적 바인딩을 사용하면 객체 지향의 중요한특징 중의 하나인 다형성을 구현할 수 있다  
객체에 메시지를 보내면 객체가 메시지를 해석하여서 가장 적절한 동작을 하게 한다  

C++에서 가상 함수가 아니면 모든 함수가 정적 바인딩으로 호출된다  
가상 함수 기능은 포인터와 참조자를 통해서만 가능하다. 객체를 함수의 값으로 전달하는 경우에는 동작하지 않느다  

하향 형변환? Down-Casting  

```cpp
BassClass *pb = new DerivedClass();
DerivedClass *pd = (DerivedClass *) pb;
// or
DerivedClass *pd = dynamic_cast<DerivedClass*>(pb);
```

포인터 뿐만 아니라 참조자도 다형성이 동일하게 적용된다  

```cpp
Dog d;
Animal &a1 = d;
```

다형성을 사용하는 과정에서 소멸자를 virtual로 해주지 않으면 자식 소멸자가 호출되지 않는다  
왜냐하면 실제로는 자식 객체지만 부모 포인터로 가리키고 있기 때문에, 컴파일러는 부모 객체로 생각해서 부모 소멸자한 호출하는 것이다  
부모의 소멸자를 virtual(가상함수로) 선언하면 된다  

순수 가상 함수 pure virtual function는 함수 헤더만 존재하고 함수의 몸체는 없는 함수  
`virtual 반환형 함수이름(매개변수리스트) = 0;`  
순수 가상 함수를 하나라도 가지고 있는 클래스를 추상 클래스라고 한다  
추상 클래스로는 객체를 생성할 수 없다  

객체를 생성할 수는 없지만, 포인터 변수로 써서 자식 객체를 가리키고, 추상 클래스에 정의된 함수들을 호출할 수있다  

## 입출력 Stream

---

입출력 장치에 따라 서로 다르게 코드를 짜야 한다면 아주 불편한 일이 될 것이다  
이를 위해 Stream이란 개념을 사용하고 잇다  
입출력은 모두 Stream으로 이루어진다  
Stream이란 입출력을 바이트들의 흐름으로 생각하는 것이다  
스트림의 최대 장점은 장치 독립성이다  
입출력 장치에 상관없이 프로그램을 작성할 수 있다는 것  

cin 이 키보드와 연결된 입력 스트림이다  
cout은 콘솔과 연결된 출력 스트림  

디스크 <- I/O System -> 메인 메모리  
파일(바이트의 시퀸스) <- iostreams -> 객체  

- ofstream 출력파일 스트림 클래스. 출력 파일을 생성하고 파일에 데이터를 쓸 때 상ㅇ
- ifstream 입력 파일 스트림 클래스. 파일에서 데이터를 읽을 때 사용
- fstream 일반적인 파일 스트림

C++에서 파일 처리를 수행할 때는 `<iostream>` `<fstream>` 헤더 파일을 포함시켜야 한다  

파일 입출력도 스트림을 통해 이루어진다  
파일에 데이터를 쓸 때 사용되는 스트림은 클래스 ofstream의 객체이다  
먼저 객체를 나타내는 변수를 선언한 후에 이 변수를 파일과 연결하면 된다  
파일에 연결하려면 open() 멤버 함수를 사용하거나 생성자를 사용하면 된다  

```cpp
int main()
{
    // 객체가 생성되면서 자동으로 파일이 열린다 open()
    ofstream os("numbers.txt"); // 파일이름만 지정하면 현재 프로젝트 위치에서
    if (!os) // os.fail()
    {
        cerr << "파일 오픈 실패" << endl;
        exit(1);
    }
    for (int i = 0; i < 100; i++)
    {
        os << i << " ";
    }
    return 0;

    // 객체 os가 범위를 벗어나면 ofstream 소멸자가 파일을 닫는다. close()
}
```

파일에서 데이터를 읽을 때 사용되는 스트림은 클래스 ifstream의 객체이다  
먼저 객체를 나타내는 변수를 선언한 후에 이 변수를 파일과 연결하면 된다  
파일에 연결하려면 open() 멤버 함수를 사용하거나 생성자를 사용하면 된다  

```cpp
int main()
{
    // 객체가 생성되면서 자동으로 파일이 열린다 open()
    ifstream os("numbers.txt"); // 파일이름만 지정하면 현재 프로젝트 위치에서
    if (!is) // is.fail()
    {
        cerr << "파일 오픈 실패" << endl;
        exit(1);
    }
    int number;
    while (is)
    {
        is >> number;
        cout << number << " ";
    }
    cout << endl;
    return 0;

    // 객체 is가 범위를 벗어나면 ifstream 소멸자가 파일을 닫는다. close()
}
```

파일 모드  
isstream 생성자를 호출할 때 2번째 인수로 넘길 수있다  

- ios::in 입력을 위하여 파일은 연다
- ios::out 출력을 위하여 파일을 연다
- ios::binary 이진 파일 입출력을 위하여 파일을 연다
- ios::ate 파일의 끝을 초기 위치로 한다
- ios::app 파일의 끝에 추가된다
- ios::trunc 새로운 내용으로 교체된다

이들은 \| 연산자를 통해 합쳐질 수 있다  

```cpp
ifstraem is("someText.txt");
if (!is) // ! 연산자 오버로딩
{
    cerr << "파일 오픈에 실패하였습니다" << endl;
    exit(1);
}

char c;
is.get(c);
while (!is.eof())
{
    cout << c;
    is.get(c);
}
```

```cpp
ofstream os("someText.txt");
char c;
while (cin.get(c))
{
    os.put(c);
}

// Ctrl + Z = End of File
```

- 출력 방식 지정, 플래그 설정, `|` 연산 가능
  - `cout.precision(3)`: 소수점 자리 제한
  - `cout.width(10)`: 출력 필드의 너비 지정
  - `cout.setf(ios::fixed)`: 고정 소수점 표기법
  - `cout.setf(ios::scientific)`: 과학적 표기법 (지수 이용)
  - `cout.setf(ios::showpoint)`: 소수점 항상 표시
  - `cout.setf(ios::showpos)`: 양수 부호를 반드시 출력
  - `cout.setf(ios::left)`: 왼쪽 정렬
  - `cout.setf(ios::right)`: 오른쪽 정렬
  - `cout.setf(ios::dec)`: 10진법
  - `cout.setf(ios::oct)`: 8진법
  - `cout.setf(ios::hex)`: 16진법
  - `cout.setf(ios::uppercase)`: 지수나 16진법으로 표시할 때 대문자
  - `cout.setf(ios::show)`: 8진수면 앞에 0, 16진수면 앞에 0x
- `os.unsetf(ios::uppercase)`: 플래그 해제

- 텍스트 파일
  - 사람이 읽을 수 있는 문자들로 구성
  - 연속적인 라인으로 구성
  - 정수 `123456` 을, 문자 `1`, `2`, `3`, ... 으로 저장
- 이진 파일
  - 사람이 읽을 수 없는 이진 데이터로 구성
  - 문자열로 변환되지 않고 입출력하기 때문에 라인의 끝을 표시할 필요가 없음
  - NULL, CR, LF 같은 문자열들은 특별한 의미를 가지지 않고 데이터로 취급
  - 특정 프로그램에 의해서만 판독이 가능 (실행 파일, 사운드 파일, 이미지 파일 등)
  - 정수 `123456` 을, 바이트 `00000000`, `00000001`, `11100010`, `01000000` 으로 저장
    - 정수나 실수를 저장하는 방식이 시스템마다 다를 수 있기 때문에 이식성이 떨어짐

```cpp
// 이진 파일 입출력
ofstream os("text.dat", ofstream::binary);
int x = 5;
os.write((char*)&x, sizeof(int));

ifstream is("text.dat", ofstream::binary);
int x = 5;
is.read((char*)&x, sizeof(int));
```

```cpp
// 이진 파일 복사

// 1. istream 객체가 사용하는 버퍼 객체의 포인터를 반환
// 이를 << 연산자를 이용하여 ostream으로 연결하면 문장 하나로 파일의 전체 내용을 복사할 수 있음
dest << source.rdbuf();

// 2. get, put (read, write)
if (source.is_open() && dest.is_open())
{
    while (!source.eof())
    {
        dest.put(source.get());
    }
}
```

- 순차 접근
  - 전에 읽었던 데이터를 다시 읽으려면 현재의 파일을 닫고 파일을 다시 열어야 한다
  - 앞부분을 읽지 않고 중간이나 마지막으로 건너뛸 수도 없다
- 임의 접근 파일
  - 파일의 어느 위치에서든지 읽기와 쓰기가 가능하다

- 원리
  - 모든 파일에는 파일 위치 표시자라는 것이 존재
  - 읽기와 쓰기 동작이 현재 어떤 위치에서 이루어지는 지를 나타낸다
  - i.e. 새 파일의 위치 표시자는 0, 이것이 파일의 시작 부분을 가리킨다
  - 기존 파일의 경우, 추가 모드에서 열렸을 경우 파일의 끝, 다른 모드인 경우에는 시작부분을 가리킨다
  - 파일에서 읽기나 쓰기가 수행되면 파일 위치 표시자가 갱신된다
  - 예를 들어 읽기 모드로 파일을 열고, 100 바이트를 읽었다면 파일 위치 표시자의 값이 100이 된다
  - 입출력 함수를 사용하면 그 함수의 내부에서 파일 위치 표시자의 값이 변경된다
  - 보통 순차적으로 데이터를 읽게 되면 파일 위치 표시자는 파일의 시작 위치에서 순차적으로 증가하여 파일의 끝으로 이동한다
  - 임의 접근을 하고 싶다면, 파일 위치 표시자를 이동시켜서 임의 파일 액세스를 할 수 있다

- `seekg(long offset, seekdir way);`
  - way
    - `ios::beg`: 처음부터의 offset
    - `ios::cur`: 현재 위치부터의 offset
    - `ios::end`: 파일 끝에서부터의 offset

- `tellg()` 현재 파일 위치 표시자 값

영상을 중간 건너뛰고 본다던지  
파일의 크기를 계산한다던지  

Raw 파일은 이미지 헤더 없이 바로 픽셀 값부터 시작  

```cpp
#include <windows.h>
HDC hdc = GetWindowDC(GetForegroundWindow());
SetPixel(hdc, c, r, RGB(red, green, blue));
```

## 예외

---

```cpp
catch(...)
{
    // ... 으로 모든 예외를 잡을 수 있다
    // 처리 될 수 있는 예외를 먼저 잡으니까, 구체적인 예외처리를 먼저, ... 는 나중에
}
```

예외가 처리되지 않으면, 이전 함수에 예외가 전달되고, 어떤 함수에서도 예외가 처리되지 않으면 시스템 라이브러리 함수 abort()가 호출되어서 프로그램을 종료  

## 함수 템플릿, 매크로

---

함수 템플릿, 매크로  
둘 다 타입과 문관하게 작동되므로 일반화 프로그래밍의 한 형태  
매크로는 매개 변수가 여러 번 계산될 수 있고, 타입의 변환이 부적절한데도 불구하고 타입을 혼합하여 쓸 수 있다.  

```cpp
#define GET_MAX(x,y) ((x)>(y) ? (x): y)

template<typename T>
T get_Max(T x, T y)
{
    if (x > y) return x;
    else return y;
}

template<> // 함수 템플릿의 특수화 (template specialization)
T get_Max(float x, float y)
{
    if (x > y) return x;
    else return y;
}

// 중복 정의
// 함수 템플릿 형태보다 함수 중복 정의의 우선 순위가 높음

// 여러 타입도 가능
template<typename T1, typename T2>
```

클래스 템플릿  

```cpp
template<typename T>
class Box
{
    // ...
}

// 멤버 함수 외부 정의
template<typename T>
Box<T>::Box()
{
    // ...
}

// 기본 값
template<typename T = int>

// 기본 값이 설정되어 있으면 생략 가능
Box<> box;

// typedef 가능
typedef Box<int> iBox;
iBox box;
```

## STL, Standard Template Library | 표준 템플릿 라이브러리

---

프로그래머들이 공통적으로 사용하는 자료 구조와 알고리듬을 구현한 클래스  
STL은 템플릿 기법을 사용하였기 때문에 어떤 자료형에 대해서도 사용할 수 있다  

- 이미 만들어진 검증된 컨테이너와 알고리듬 -> 시간 절약, 안정성 확보
- STL은 객체 지향 기법과 일반화 프로그래밍 기법을 적용하여서 만들어졌으므로 컨테이너 또는 알고리듬을 어떤 자료형에서도 사용할 수 있다. -> 타입마다 다시 만들 필요가 없다.

### Container | 컨테이너

데이터들을 저장하는 클래스.  
복잡한 데이터 구조를 쉽게 구현하는데 도움이 된다.  
템플릿으로 구현되므로 컨테이너가 서로 다른 종류의 객체를 저장하는 데 사용될 수 있다.  

#### Container 분류

- 순차 (Sequence) 컨테이너
  - 자료를 순차적으로 가지고 있다.
  - 순차적인 컨테이너는 자료의 추가는 빠르지만 탐색할 때는 시간이 많이 걸린다

- 연관(연관 Sequence) 컨테이너
  - 사전과 같은 구조를 사용하여서 자료를 저장한다.
  - 원소들을 검색하기 위한 키 key를 가지고 있다. 자료들은 정렬되어 있다.
  - 자료의 추가에는 시간이 걸리지만 자료의 탐색은 매우 빠르다.

#### Container 공통 메서드

- `Container()`: 기본 생성자
  - `Container(size)`: 크기가 size인 컨텐이너 생성
  - `Container(size, value)`: 크기가 size이고 초기값이 value인 컨테이너 생성
  - `Container(iterator, iterator)`: 다른 컨테이너부터 초기값의 범위를 받아서 생성
- `begin()`: 첫 번째 요소의 반복자 위치
- `end()`: 반복자가 마지막 요소를 지난 위치
- `rbegin()`: 끝을 나타내느 역반복자
- `rend()`: 역반복자가 처음을 지난 위치
- `front()` 컨테이너의 첫 번째 요소 반환
- `insert(iterator, value)`: 컨테이너의 중간에 요소 삽입
- `pop_back()`: 컨테이너의 마지막 요소를 삭제
- `push_back(value)`: 컨테이너의 끝에 데이터를 추가
- `erase(iterator)`: 컨테이너의 중간 요소를 삭제
- `erase(iterator, iterator)`: 컨테이너의 지정된 범위를 삭제
- `clear()`: 모든 요소를 삭제
- `size()`: 컨테이너의 크기
- `empty()`: 비어있는지를 검사

참고: `std::begin()`, `std::end()` 함수 -> 컨테이너의 `begin()`, `end()` 메서드 호출  

#### Container Adapter | 컨테이너 어댑터

이미 존재하는 컨테이너를 변경하여 새로운 기능을 제공하는 클래스  
즉 기존의 컨테이너의 기능을 그대로 이용하면서 새로운 기능이나 인터페이스를 제공하는 것  
스택, 큐, 우선순위 큐 등이 있다  

`stack`은 선형적인 자료구조만 있으면 된다  
중간에서 데이터를 추가하거나 삭제하는 기능은 필요 없다, 오히려 순차 컨테이너의 기능이 너무 많아서 기능을 제약해야 스택을 만들 수 있다  

`stack`은 `deque`를 디폴트로 만들어졌다  
다른 컨테이너로 만들고 싶으면  
`stack<string, vector<string>> st;` 처럼 두 번째 인자를 넘겨주면 된다  

우선순위 큐는 히프를 내부적으로 사용한다.  

### Iterator | 반복자

컨테이너의 요소를 가리키는 데 사용된다  
컨테이너에 저장된 요소에 접근하는 방법.  

데이터(컨테이너)와 알고리듬의 연결자.  
예를 들어 sort() 알고리듬은 시작 반복자와 종료 반복자라는 매개변수를 가지고 있다  
sort() 알고리듬은 반복자를 이용하여 컨테이너의 요소들을 비교하여서 정렬을 수행할 수 있다.  
반복자를 사용하면 컨테이너의 유형에 상관없이 동일한 정렬 알고리듬을 적용할 수 있다  

예전 방법  
배열과 벡터는 인덱스를 사용하여 요소에 접근 가능  
그러나 랜덤 접근을 허용하지 않는 연결 리스트에서는 인덱스는 사용할 수 없고 아마 포인터를 사용하여야 할 것  
문제는 컨테이너의 종류에 따라서 요소에 접근하는 방법이 상당히 다르다는 것  
따라서 일반적인 방법을 찾아야 한다  

STL을 작성한 사람들은 컨테이너의 종류에 관계없이 요소들에 접근하게 하기 위하여 반복자라는 방식을 제안하였다  
반복자는 컨테이너의 요소를 가리키는 객체이다  
기존의 포인터와 비슷하여서 반복자를 흔히 일반화된 포인터 Generalized Pointer 라고 한다  
반복자를 사용하게 되면 컨테이너의 종류에 상관없이 일관된 방법으로 컨테이너의 요소에 접근할 수 잇따  

알고리듬은 컨테이너의 요소에 접근하여서 읽거나 써야 한다  
이때 반복자가 사용된다  
반복자를 사용하여서 컨테이너의 첫 번째 요소를 가리키게 한 후에 작업을 하고 작업이 끝나면 반복자를 증가하여서 다음 요소를 가리키게 한다  
반복자가 마지막 요소를 벗어나게 되면 작업을 끝내면 된다  

STL의 핵심 개념은 시퀸스 Sequence이다  
시퀸스는 어떤 순서를 가지고 있는 일련의 데이터이다  
시퀸스에는 시작과 끝이 있다  
시퀸스는 처음부터 끝까지 순회할 수 있으며 중간 요소 읽기 또는 쓰기가 가능하다  
반복자는 시퀸스의 요소를 식별하는 객체이다  
예를 들어서 begin()과 end()는 반복자로서 시퀸스의 시작과 끝을 식별한다  

begin()으로 식별되는 요소는 시퀸스의 첫 번째 요소이고 end()는 시퀸스의 끝을 하나 지난 위치를 가리킨다  

반복자에서는 다음과 같은 연산자들을 사용할 수 있다. 반복자는 다음의 연산자가 중복 정의되어 있는 객체라고 생각하면 된다.  

- 컨테이너에서 다음 요소를 가리키기 위한 ++연산자
- 이전 요소 -- 연산자
- 두 개의 반복자가 같은 요소를 가리키고 있는 지를 확인하기 위한 ==와 != 연산자
- 반복자가 가리키는 요소의 값을 추출하기 위한 역참조 연산자 *

각 컨테이너는 특별한 위치의 반복자를 얻는 함수를 지원한다  

- v.begin() 함수는 컨테이너 v에서 첫 번째 요소를 반환한다
- v.end() 함수는 컨테이너 v에서 마지막 요소를 하나 지난 값을 반환한다. v.end()는 마지막 요소가 아니라 컨테이너의 끝을 나타내는 보초값 Sentinel을 반환한다. v.end()가 반환하는 값은 포인터에서의 NULL값과 같은 의미를 지닌다

예전 방법  

```cpp
// old
for (vector<int>::iterator it = v.begin(); it != c.end(); it++)

// new 1
for (auto it = v.begin(); it != c.end(); it++)

// new 2
for (auto& n: v)
```

반복자는 코드(알고리듬)을 데이터에 연결하는 데 사용된다  
코드 작성자는 반복자에 대해 알고 있지만 반복자가 실제로 데이터에 어떻게 접근하는지에 대해서는 자세히 알지 않으며 데이터 공급자는 데이터가 저장되는 방식에 대한 세부 정보를 표시하지 않고 단순히 사용자에게 반복자를 제공한다  

따라서 반복자는 알고리듬과 컨테이너 사이에 독립성을 제공한다  
STL을 작성하였던 Alex Stepanov는 다음과 같이 말햇다  
STL 알고리듬과 컨테이너가 잘 작동하는 이유는 서로에 대해 알지 못하기 때문입니다  

#### Iterator 종류

- 전향 반복자 | Forward Iterator: ++연산자
- 양방향 반복자 | Bidirectional Iterator: ++연산자, --연산자
- 무작위 접근 반복자 | Random Access Iterator: ++연산자, -- 연산자, []연산자
  - i.e. 벡터

C++11 - 범위기반루프(Range-Based Loop) 로 대신 할 수 있다.  
컨테이너의 중간에 삽입하는 경우에는 반복자를 사용하여야 한다  

### STL 알고리듬

일반적으로 반복자를 사용하여 주어진 타입으로 컨테이너에 접근  
반복자는 알고리듬과 컨테이너를 연결하는 역할을 한다  

쓰러면 `<algorithm>` 헤더  

- 탐색 find() 컨테이너 안에서 특정한 자료를 찾는다
- 정렬 sort() 자료들을 크기순으로 정렬한다
- binary_search() 이진 탐색으로 자료를 찾는다
- 반전 reverse() 자료들의 순서를 역순으로 한다
- 삭제 remove() 조건이 만족되는 자료를 삭제한다
- 변환 transform() 컨테이너의 요소들을 사용자가 제공하는 변환 함수에 따라서 변환한다

- 불변경 알고리듬 (컨테이너가 안변경되는)
  - 개수
    - count 값과 일치하는 요소 수
    - count_if 조건에 맞는 요소 수
  - 탐색
    - search 값과 일치하는 첫 번째 요소
    - search_n 값과 일치하는 n개의 요소
    - find 값과 일치하는 첫번째요소
    - find_if 조건에 일치하는 첫 번째 요소
    - find_end 조건에 일치하는 마지막 요소
    - binary_search 정렬된 컨테이너에 대해 이진 탐색
  - 비교
    - equal 두 요소가 같은지
    - mismatch 두 컨테이너를 비교해 일치하지 않는 첫 번째 요소
    - lexicographical_compare 두 순차 컨테이너를 비교하여서 사전적으로 어떤 컨테이너가 작은 지 반환
- 변경 알고리듬
  - 초기화
    - fill 모든 요소를 지정된 값으로
    - generate 지정된 함수의 반환값을 할당
  - 변경
    - for_each 지정된 범위의 모든 요소에 대하여 연산 수행
    - transform 지정된 범위의 모든 요소에 대하여 함수를 적용
  - 복사
    - copy 하나의 구간을 다른 구간으로 복사
  - 삭제
    - remove 지정된 구간에서 지정된 값을 가지는 요소들을 삭제
    - unique 구간에서 중복된 요소들을 삭제
  - 대치
    - replace 지정된 구간에서 요소가 지정된 값과 일치하면 대치값으로 변경
  - 정렬
    - sort 지정된 정렬 기분에 따라서 구간의 요소들을 정렬
      - 합병정렬 O(NLogN) 거의 정렬된 리스트에 대해서는 상당히 빠르단
  - 분할
    - partition 지정된 구간의 요소들을 조건에 따라서 두 개의 집합으로 나눈다

## Lambda-Expression

---

C++에서 함수는 정식 객체 (일급객체)가 아니다  
따라서 함수를 독립적으로 정의할 수 있는 방법이 없었다  
또 함수를 다른 함수의 인수로 전달하거나 함수 몸체를 반환할 수 있는 방법이 없었다  

하지만 함수형 프로그래밍 언어에서는 함수가 아주 중요시 된다  
함수형 프로그래밍 언어에서는 함수가 객체로 존재할 수 있다  
함수를 변수에 할당할 수 있으면 다른 함수의 인수로 함수를 전달할 수 있다  
람다식은 함수형 프로그래밍을 C++에 도입한 것이다  

```cpp
[](arg1, arg2, ...) 반환형 { body }
auto sum = [](int x, int y) { return x + y; }
```

## TODO

---

[size_t, unsigned int](https://love-every-moment.tistory.com/38)  
[size_t, unsigned int](https://pvs-studio.com/en/blog/posts/cpp/a0050/)  
[size_t, unsigned int](http://mwultong.blogspot.com/2007/06/c-sizet-unsigned-int.html)  
[특정 문자 제거 erase remove](https://wooono.tistory.com/475)  
[특정 문자 제거 erase remove](https://cho001.tistory.com/164)  
[string 생성자](https://modoocode.com/237)  
[배열 특정 요소 개수](https://codechacha.com/ko/cpp-check-if-element-is-in-array/)  
[Vector 탐색](https://notepad96.tistory.com/41)  
[size_t, unsigned int](https://pvs-studio.com/en/blog/posts/cpp/a0050/)  

- ++ ++ a O, a ++ ++ X  
  - 오른쪽에서 왼쪽으로 실행되기 대문에  
  - 무튼 연산자 만들 때 그래서 위쪽은 되게, 아래쪽은 안되게 만들어야함  
  - 이걸 간단히 구현하게 해주는 방법이 있음  
  - 객체& operator++() 에서 &을 붙여주는 이유  
  - 예를 들어 Temp a = v1 + v2 를 봤을 때  
  - v1 + v1 의 결과값은 임시 객체임, v1이나 v2의 실체 객체가 아니라는 것  
  - 이를 바탕으로 ++ ++ a 를 보면  
  - 객체를 반환값으로 보내면  
  - a = a + 1 에서 a(왼쪽) 를 반환하지 않고 a + 1(오른쪽) 의 임시 객체를 보내버림  
  - 그래서 자기 자신을 반환하기 위해 레퍼런스를 보냄  

- C++ time  

```cpp
#include <time.h>

using namespace std;

int main()
{
    time_t cur;
    time(&cur);
    tm* gmTM = gmtime(&cur);
    printf("%d\n%02d\n%02d", 1900 + gmTM->tm_year, gmTM->tm_mon + 1, gmTM->tm_mday);
}
```

- [[C++] 코딩테스트를 위한 C++ 기본](https://suyeoniii.tistory.com/13)
- [코딩 테스트 입문 (with C++)](https://gamedevlog.tistory.com/6?category=892157)

> [C++ C3861](https://docs.microsoft.com/ko-kr/cpp/error-messages/compiler-errors-2/compiler-error-c3861?view=msvc-170)  
> 에러: '뭐시깽' 식별자를 찾을 수 없습니다.  
> 해결: 함수 위치 밑 선언 확인  

---

> [C++ C2360](https://docs.microsoft.com/ko-kr/cpp/error-messages/compiler-errors-1/compiler-error-c2360?view=msvc-170), [C++ C2361](https://docs.microsoft.com/ko-kr/cpp/error-messages/compiler-errors-1/compiler-error-c2361?view=msvc-170)  
> 에러: '뭐시깽' 초기화가 'case'/'default' 레이블에 의해 생략되었습니다.  
> 해결: 변수 선언하는 case에 스코프 {} 달아주기  
> [설명](https://ansohxxn.github.io/cpp/chapter5-1/): case/default 이전 공간에서는 메모리 할당 안됨

포인터 delete 후 어떤 코드가 없더라도 = nullptr 대입  

- c 스타일 형변화
  - char 형변환
  - (char)num
  - char(num)
  - <https://boycoding.tistory.com/177>
  - <https://softwareengineering.stackexchange.com/questions/50442/c-style-casts-or-c-style-casts>
- ccp 소수점 자리수 고정 출력
  - `cout << fixed; // 아래 함수를 소수점에 대해서만 쓸건지`
  - `cout.precision(30); // 숫자 최대 길이 출력`
- stl
  - map: key정렬
  - unordered_map: 걍 넣음
