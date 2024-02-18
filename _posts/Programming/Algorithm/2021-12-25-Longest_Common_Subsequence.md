---
title : 최장 공통 부분 수열 알고리즘 [Longest Common Subsequence]
categories : [Programming, Algorithm]
tags : [다이나믹 프로그래밍, Dynamic Programming, Longest Common Subsequence, LCS, 최장 공통 부분 수열]
---

## LCS (Longest Common Subsequence)
<hr style="border-top: 1px solid;"><br>

![image](https://user-images.githubusercontent.com/52172169/148669138-47614794-7851-44cb-8e8b-5006e5faa0fa.png)

<br>

ACAYKP, CAPCAK이 있을 때 최장 공통 부분 수열을 구하려면 다음과 같다. 

|      |  A   |  C   |  A   |  Y   |  K   |  P   |
| :--: | :--: | :--: | :--: | :--: | :--: | :--: |
|  C   |  0   |  1   |  1   |  1   |  1   |  1   |
|  A   |  1   |  1   |  2   |  2   |  2   |  2   |
|  P   |  1   |  1   |  2   |  2   |  2   |  3   |
|  C   |  1   |  2   |  2   |  2   |  2   |  3   |
|  A   |  1   |  2   |  3   |  3   |  3   |  3   |
|  K   |  1   |  2   |  3   |  3   |  4   |  4   |

<br>

arr1 = ACAYKP, arr2 = CAPCAK 일 때

```i, j >= 1일 때, arr1[i-1] == arr2[j-1] 일 때 dp[i][j] = dp[i-1][j-1] + 1;``` 
: 각 문자열의 i, j번째까지 보았을 때 같다면, 그 전까지의 문자열에서 나온 공통 부분수열 길이에 +1 해주는 것임.

```arr1[i-1] != arr2[j-1] 일 때 dp[i][j] = max(dp[i][j-1], dp[i-1][j]);```
: 다르다면, 여태까지 나온 부분수열의 길이 중 최고 길이를 넣어주는 것임.

<br>

Link 
: <a href="https://www.acmicpc.net/problem/9251" target="_blank">백준 9251번 : LCS</a>

<br>

```cpp
#include <iostream>
#include <string>
using namespace std;

typedef long int li;

li dp[1001][1001];

int main() {
	string arr1, arr2;
	cin >> arr1 >> arr2;
	li i=0, j=0;
	for(; i<arr1.size(); i++) {
		for(j=0; j < arr2.size(); j++) {
			if(arr1[i] == arr2[j]) {
				dp[i+1][j+1] = dp[i][j]+1;
			}
			else {
				dp[i+1][j+1] = max(dp[i][j+1], dp[i+1][j]);
			}
		}
	}
	cout << dp[i][j] << '\n';
	return 0;
}
```

<br><br>
<hr style="border: 2px solid;">
<br><br>

## 정리글
<hr style="border-top: 1px solid;"><br>

Link 
: <a href="https://velog.io/@emplam27/알고리즘-그림으로-알아보는-LCS-알고리즘-Longest-Common-Substring와-Longest-Common-Subsequence" target="_blank">velog.io/@emplam27/알고리즘-그림으로-알아보는-LCS-알고리즘-Longest-Common-Substring와-Longest-Common-Subsequence</a>

위 링크에서 최장 공통 부분 수열 찾는 방법에 대해서도 알려줌.

<br><br>
<hr style="border: 2px solid;">
<br><br>

