---
layout: post
title: 백준 9012번 문제
categories : 백준
tag : 백준
---
# [Silver IV] 괄호 - 9012 

[문제 링크](https://www.acmicpc.net/problem/9012) 

### 성능 요약

메모리: 2024 KB, 시간: 4 ms

### 분류

자료 구조(data_structures), 스택(stack), 문자열(string)

### 문제 설명

<p>괄호 문자열(Parenthesis String, PS)은 두 개의 괄호 기호인 ‘(’ 와 ‘)’ 만으로 구성되어 있는 문자열이다. 그 중에서 괄호의 모양이 바르게 구성된 문자열을 올바른 괄호 문자열(Valid PS, VPS)이라고 부른다. 한 쌍의 괄호 기호로 된 “( )” 문자열은 기본 VPS 이라고 부른다. 만일 x 가 VPS 라면 이것을 하나의 괄호에 넣은 새로운 문자열 “(x)”도 VPS 가 된다. 그리고 두 VPS x 와 y를 접합(concatenation)시킨 새로운 문자열 xy도 VPS 가 된다. 예를 들어 “(())()”와 “((()))” 는 VPS 이지만 “(()(”, “(())()))” , 그리고 “(()” 는 모두 VPS 가 아닌 문자열이다. </p>

<p>여러분은 입력으로 주어진 괄호 문자열이 VPS 인지 아닌지를 판단해서 그 결과를 YES 와 NO 로 나타내어야 한다. </p>

### 입력 

 <p>입력 데이터는 표준 입력을 사용한다. 입력은 T개의 테스트 데이터로 주어진다. 입력의 첫 번째 줄에는 입력 데이터의 수를 나타내는 정수 T가 주어진다. 각 테스트 데이터의 첫째 줄에는 괄호 문자열이 한 줄에 주어진다. 하나의 괄호 문자열의 길이는 2 이상 50 이하이다. </p>

### 출력 

 <p>출력은 표준 출력을 사용한다. 만일 입력 괄호 문자열이 올바른 괄호 문자열(VPS)이면 “YES”, 아니면 “NO”를 한 줄에 하나씩 차례대로 출력해야 한다. </p>

 56  
백준/Silver/9012. 괄호/괄호.cc
@@ -0,0 +1,56 @@

#include <iostream>
#include <vector>
using namespace std;
```c
int main()
{
	string str;
	vector<string> answer;
	int n;
	cin >> n;

	for (int i = 0; i < n; i++)
	{
		cin >> str;
		int a = 0, b = a + 1;
		while (true)
		{
			if (str[a] == '(')
			{
				if (str[b] == ')')
				{
					str.erase(str.begin() + a);
					str.erase(str.begin() + b-1);
					a = 0;
					b = 0;
				}
				b++;
				if (str == "\0") {

					answer.push_back("YES");
					break;
				}
				if (str.length() < b) {

					answer.push_back("NO");
					break;
				}

			}
			if (str[a] != '(') {
				a++;
				if (str.length() <= a) {
					answer.push_back("NO");
					break;
				}
			}
		}
		str.clear();
	}

	for (int i = 0; i < n; i++)
	{
		cout <<answer[i] << endl;
	}
return 0;
}
```