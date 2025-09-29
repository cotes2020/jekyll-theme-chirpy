---
title: "C++11 auto"
# description: ""
categories: [컴퓨터, 프로그래밍]
tags: []
image: "/assets/img/background/kururu-lab.jpg"

date: 2025-09-29. 18:33 # Init
# last_modified_at: 2025-09-29. 18:33
---

## 말머리

---

## auto

---

```cpp
int i = 0;
// 대신
auto j = i;
// 로 선언 가능
// 컴파일러가 타입을 자동 추론
```

타입 막 바꿔도 되는 js랑 다른 점은,  
c++ auto는 타입이 고정된다는 점. (타입 안정성 유지)  

이것의 장점?  
타입 하나 바꿀 때 연관된 것 일일이 찾아가면서 타입 다 바꿔야 하는데, auto 쓰면 안바꿔도 된다 (코드 일관성 유지)  

iterator 같이 **읽기** 복잡한 것들, 기다란 변수명 대신 auto 사용 가능  
타이핑을 줄여준다.  
// 가장 많이 쓰이는 부분

단점은..  
코드 읽다가 중간에 auto 나오면 이게 뭔지 모른다는 건데  
크게 문제되지 않는다  
변수에 마우스만 올려주면 어떤 타입인지 ide가 알려주니까  

헷갈리는 거  
auto에 3가지를 대입할 수 있음  

```cpp
int i = 0;
auto j = i; // 변수 값

int *i;
auto j = i; // 주소 값. pointer도 auto로
// auto *j = i; // 이건 아니다

int& sans() {}
auto &j = sans(); // 레퍼런스. auto& 사용해야 참조가 유지됨. 단순 auto는 복사 발생 가능. &는 펭귄
```

```cpp
auto add(int x, int y) // 반환형도 가능하다.
{
    return x + y;
}
auto sum = add(5, 6);
```

```cpp
std::vector<int> vec = {1, 2, 3, 4, 5};
for (auto it = vec.begin(); it != vec.end(); ++it) // iterator도
{
	std::cout << *it << " ";
}
```

```cpp
for (auto &value : vec) // range-based for문
{
	std::cout << value << " ";
}
```

```cpp
std::map<std::string, int> myMap = {{"one", 1}, {"two", 2}, {"three", 3}};
for (const auto &[key, value] : myMap) // 구조분해도 가능
{
	std::cout << key << ": " << value << std::endl;
}
```

```cpp
auto lambda = [](int x, int y) { return x + y; }; // 람다 표현식
int result = lambda(3, 4);
std::cout << "Result: " << result << std::endl;
```

```cpp
template <typename T, typename U>
auto multiply(T a, U b) // 템플릿 함수
{
	return a * b;
}
```

```cpp
std::tuple<int, std::string, double> myTuple = {1, "Hello", 3.14};
auto [num, str, pi] = myTuple; // 구조분해
std::cout << "Number: " << num << ", String: " << str << ", Pi: " << pi << std::endl;
```

```cpp
std::variant<int, std::string> myVariant = "Hello";
std::visit([](auto&& arg) { // std::variant와 std::visit
	std::cout << arg << std::endl;
}, myVariant);
```

```cpp
auto factorial(int n) -> int // 반환형 후치 지정
{
	if (n <= 1) return 1;
	return n * factorial(n - 1);
}
```

```cpp

auto ptr = std::make_unique<int>(42); // 스마트 포인터
std::cout << *ptr << std::endl;
```

```cpp
auto gcd = [](int a, int b) { // 람다 캡처
	while (b != 0) {
		int temp = b;
		b = a % b;
		a = temp;
	}
	return a;
};
std::cout << "GCD: " << gcd(48, 18) << std::endl;
```

```cpp
std::vector<int> numbers = {1, 2, 3, 4, 5};
auto evenNumbers = std::views::filter(numbers, [](int n) { return n % 2 == 0; }); // C++20의 범위 라이브러리
for (auto n : evenNumbers) {
	std::cout << n << " ";
}
```

```cpp
auto [x, y] = std::pair<int, int>{10, 20}; // std::pair 구조분해
std::cout << "x: " << x << ", y: " << y << std::endl;
```

```cpp
auto isEven = [](int n) { return n % 2 == 0; }; // 람다 표현식
std::cout << "Is 4 even? " << (isEven(4) ? "Yes" : "No") << std::endl;
```
