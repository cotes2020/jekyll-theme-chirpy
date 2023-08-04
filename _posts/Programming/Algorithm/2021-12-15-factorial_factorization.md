---
title : 팩토리얼 소인수분해 알고리즘 [Factorial Factorization Algorithm]
categories: [Programming, Algorithm]
tags : [Factorial Factorization Algorithm, 소인수분해, 팩토리얼 소인수분해]
---

## 팩토리얼 소인수분해
<hr style="border-top: 1px solid;"><br>

```100! = 2^a + 3^b+ 5^c``` ~~~ 로 나타내는 방법은 아래와 같음.

+ 100까지의 숫자에 2를 인수로 가지고 있는 숫자의 개수는 100/2 = 50 개임.

+ 100까지의 숫자에 4를 인수로 가지고 있는 숫자의 개수는 100/4 = 25 개임.

+ 100까지의 숫자에 8를 인수로 가지고 있는 숫자의 개수는 100/8 = 12 개임.

+ 16은 100/16 = 6개

+ 32는 100/32 = 3개

+ 64는 100/64 = 1개

<br>

따라서 100!에 2가 들어가 있는 개수는 50+25+12+6+3+1 = 97개

마찬가지의 방법으로 b와 c 등을 구할 수 있음.

<br><br>
<hr style="border: 2px solid;">
<br><br>

## 백준 7806번 : gcd!
<hr style="border-top: 1px solid;"><br>

팩토리얼의 소인수분해를 구한 뒤 k와의 최대공약수를 구해야 하는 문제.

처음에 접근한 방법은 그냥 gcd 알고리즘을 이용하였으나, 100! 부터는 값이 너무나 커져버려서 담을 수가 없었음.

**따라서 팩토리얼의 소인수를 구하고, k의 소인수를 구해서 최대공약수를 찾아야 함.**

아래 풀이는 팩토리얼의 소인수는 구하지 않지만, k의 소인수를 구한 뒤 그 소인수들로 팩토리얼의 소인수인지 판단 후 지수를 비교하여 지수가 더 작은 값을 gcd에 곱해주었음.

<br>

```cpp
#include <iostream>
#include <algorithm>
#include <vector>
#include <cmath>
using namespace std;
typedef unsigned long long int uli;

int main() {
    ios::sync_with_stdio(false); 
    cin.tie(NULL);
    
    uli n,k;
    
    while(cin >> n >> k) {
        uli gcd=1;
        vector<pair<uli,uli>> fk;
        for(uli i=2; i*i <= k; i++) {
            uli cnt=0;
            while(k%i == 0) {
                k/=i;
                cnt++;
            }
            if(cnt != 0) {
                fk.push_back(make_pair(i,cnt));
            }
        }
        if(k != 1) {
            fk.push_back(make_pair(k,1));
        }
        
        for(uli i=0; i<fk.size(); i++) {
            if(fk[i].first <= n) {
                uli cnt=0;
                for(uli j=fk[i].first; j <= n; j*=fk[i].first) {
                    cnt+=n/j;
                }
                if(cnt > fk[i].second) {
                    while(fk[i].second--) {
                        gcd*=fk[i].first;
                    }
                }
                else {
                    while(cnt--) {
                        gcd*=fk[i].first;
                    }
                }
            }
        }
        cout << gcd << '\n';
    }
    return 0;
}
```

<br><br>
<hr style="border: 2px solid;">
<br><br>

## 참고
<hr style="border-top: 1px solid;"><br>

Link 
: <a href="https://m.blog.naver.com/shalska1234/50087466089" target="_blank">m.blog.naver.com/shalska1234/50087466089</a>  

<br><br>
<hr style="border: 2px solid;">
<br><br>
