---
title: Ps를 위한 C++정리
date: 2024-1-13 16:16:00 +0900
categories: [Ps]
tags: [cpp, c++, ps]
math: true
img_path: /assets/img/post6/
---

이 포스트는 Ps에 쓰기 위해 c++ 공부한 것들을 정리한다. 

## String
```cpp
#include <iostream>
#include <string>

using namespace std;
int main ()
{
    string str;
    getline(cin, str);
    //이렇게 하면 한 줄의 string을 입력을 받을 수 있다.
}
```

## Vector
vector는 가변 배열 같은 느낌으로 보면 될 듯 하다.
```cpp
#include <iostream>
#include <vector>

using namespace std;
int main ()
{
    vector<long> V(n, 0); //길이 n짜리 vector, 다 0으로 초기화
    for (int i = 0; i < V.size(); i++)
    {
        V[i] = i; //이렇게 index로 접근도 가능하고
    }
    for (auto& elem:V)
    {
        elem = i; //이렇게 reference로 접근도 가능하고
    }
    for (auto it = V.begin(); it != V.end(); ++it)
    {
        *it *= 2; //이렇게 iterator로 접근도 가능하다.
    }

}
```

## fast input
```cpp
    std::ios::sync_with_stdio(false);
    std::cin.tie(0);
    //이렇게 하면 c언어의 scanf와의 연결을 끊어서 입력을 더욱 빠르게 받을 수 있다. 
```

## array (길이가 정해진 배열)
```cpp
    int alphaArray[26] = {0, }; //0으로 초기화
```

```cpp
    #include <array>
    array<int, 3> betaArray{}; //길이 3짜리 int 배열 0으로 초기화해서 생성
    array<array<int, 1000>, 1000> betaArray{}; // 1000*1000 int 이차원 배열 
```

## vector (길이가 정해지지 않은 배열)
```cpp
    #include <vector>
    using namespace std;
    vector<int> gammaArray(n, 0); //길이 n짜리 int 배열 0으로 초기화해서 생성

    vector<vector<int>> gammaArray(n);
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            int temp;
            cin >> temp;
            gammaArray[i].push_back(temp);
        }
    }
    //n*n 이차원 배열 입력 받기

    vector<int> gammaArray = {3, 5, 4, 1, 2};
    vector<int> deltaArray(gammaArray.begin(), gammaArray.begin()+3);
    //deltaArray = {3, 5, 4};
    //vector 배열 일부 복제하기
```