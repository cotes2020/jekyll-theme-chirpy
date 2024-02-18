---
title : "플로이드-와샬 알고리즘 [Floyd-Warshall Algorithim]"
categories : [Programming, Algorithm]
tags: [Dynamic Programming, 다이나믹 프로그래밍, Graph, 그래프, Floyd-Warshall, 플로이드-와샬]
---

## Floyd-Warshall
<hr style="border-top: 1px solid;"><br>

Link 
: <a href="https://www.acmicpc.net/problem/11404" target="_blank">11404번 : 플로이드</a>

![image](https://user-images.githubusercontent.com/52172169/148632668-08748145-ee60-420e-911c-6c00f38fe8f4.png)

<br>

```cpp
#include <iostream>
using namespace std;
int dp[100][100];
const int INF = 1e9;

int main(){
    int N, M; cin >> N >> M;
    for(int i=0; i<N; ++i){
        for(int j=0; j<M; ++j){
            dp[i][j] = INF;
        }
        dp[i][i] = 0;
    }
    // 초기화 : 자기 자신은 0, 나머지는 INF(무한)
    
    
    // 입력단계에서 dp[i][j] : i에서 j까지의 거리
    for(int i=0; i<M; ++i){
        int u, v, w; cin >> u >> v >> w;
        u--, v--;
        dp[u][v] = min(dp[u][v], w);
    }

    for(int k=0; k<N; ++k){
        for(int i=0; i<N; ++i){
            for(int j=0; j<N; ++j){
                dp[i][j] = min(dp[i][j], dp[i][k] + dp[k][j]);
            }
        }
    }
    // dp갱신 후 dp[i][j] : i에서 j까지의 최단거리

    for(int i=0; i<N; ++i){
        for(int j=0; j<N; ++j){
            if(dp[i][j] == INF) cout << 0 << ' ';
            else cout << dp[i][j] << ' ';
        }
        cout << '\n';
    }
}
```

<br><br>
<hr style="border: 2px solid;">
<br><br>
