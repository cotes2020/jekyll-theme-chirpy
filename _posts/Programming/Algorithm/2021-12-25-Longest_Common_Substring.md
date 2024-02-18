---
title : 최장 공통 부분 문자열 알고리즘 [Longest Common Substring]
categories : [Programming, Algorithm]
tags : [다이나믹 프로그래밍, Dynamic Programming, Longest Common Substring, 최장 공통 부분 문자열, LCS]
---

## Longest Common Substring
<hr style="border-top: 1px solid;"><br>

Link 
: <a href="https://www.acmicpc.net/problem/5582" target="_blank">백준 5582번 공통 부분 문자열</a>

![image](https://user-images.githubusercontent.com/52172169/148672661-ab5d3925-2543-4322-a0c7-ea3e5f8d262a.png)

<br>

공통 부분 수열과의 차이점은 문자열은 연속되어야 한다는 점임.

따라서 달라져야 하는 부분은 **공통 수열에서는 ```arr[i] != arr[j]``` 일 때, 이전까지 구한 부분수열 길이의 최대값을 가져왔지만, 부분 문자열에서는 같지 않을 때, ```dp[i][j]=0```이 되어야 함.**

같을 때는 부분 수열과 동일하게 이전까지 구한 부분 문자열에 현재 문자를 추가해줘야 함.

따라서 아래와 같이 점화식이 구성이 됨.

<br>

```
arr[i] == arr[j] => dp[i][j]=dp[i-1][j-1]+1

arr[i] != arr[j] => dp[i][j]=0
```

<br>

문제 코드는 아래와 같음.

```cpp
#include <iostream>
#include <algorithm>
#include <string>
using namespace std;

int dp[4001][4001];

int main()
{
  string s1, s2; cin >> s1 >> s2;
  int i=0, j=0, res=-1;
  
  for(; i<s1.size(); i++)
  {
      for(j=0; j<s2.size(); j++)
      {
          if(s1[i] == s2[j]) { dp[i+1][j+1]=dp[i][j]+1; }
          else { dp[i+1][j+1]=0; }
          res=max(res,dp[i+1][j+1]);
      }
  }
  cout << res << '\n';
}
```

<br><br>
<hr style="border: 2px solid;">
<br><br>

## 정리글
<hr style="border-top: 1px solid;"><br>

Link 
: <a href="https://velog.io/@emplam27/알고리즘-그림으로-알아보는-LCS-알고리즘-Longest-Common-Substring와-Longest-Common-Subsequence" target="_blank">velog.io/@emplam27/알고리즘-그림으로-알아보는-LCS-알고리즘-Longest-Common-Substring와-Longest-Common-Subsequence</a>

<br><br>
<hr style="border: 2px solid;">
<br><br>
