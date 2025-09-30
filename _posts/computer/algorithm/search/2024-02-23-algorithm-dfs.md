---
title: "DFS"
# description: ""
categories: [컴퓨터, 알고리즘]
tags: [알고리듬, Search]
image: "/assets/img/background/kururu-lab.jpg"
hidden: true

date: 2024-02-23. 17:42
last_modified_at: 2024-07-15. 06:52
---

{% include embed/youtube.html id='93jy2yUYfVE' %}

## 정의

---

BFS에서 `Queue` 대신 `Stack`을 쓰는  

DFS | Depth-First-Search | 깊이 우선 탐색  
다차원 배열에서 각 칸을 방문할 때 깊이를 우선으로 방문하는 알고리듬  

깊이를 우선으로 방문?  
설명할 방법이 없다  

원래 DFS는 `Graph`에서 모든 노드를 방문하기 위한 알고리듬  

## 구현

---

### 다차원 배열에서의 DFS

1. DFS에는 자료를 담을 `Stack`가 필요
2. 시작 좌표에 방문 표시를 남기고 `Stack`에 넣기
3. `Stack`가 빌 때까지 원소를 `pop`하고 4번 처리
4. 해당 좌표와 그 상하좌우에 대해 방문 표시를 남기고 해당 칸을 `Stack`에 삽입 (이미 방문했었다면 pass)

방문 표시를 남기기 때문에 모든 칸이 `Queue`에 1번씩 들어가므로,  
시간복잡도는 칸이 N개일 때 O(N), = 행 R개 열 C개일 때 O(RC)  

BFS와 최종 결과는 똑같더라도, 방문 순서에 아주 큰 차이가 있다  
BFS는 파문 퍼지듯 상하좌우로 퍼져나가는, 거리 순을 방문  
DFS는 마인크래프트 땅굴 파듯 뭔가 한 방향으로 막힐 때까지 쭉 직진  

BFS에서 현재 보는 칸으로부터 추가되는 인접한 칸은 거리가 현재 보는 칸보다 1만큼 떨어져있다는 성질이  
DFS에서는 성립하지 않음  
그래서 거리를 계산할 때는 DFS를 쓸 수 없음  

그래서 다차원 배열에서 굳이 BFS 대신 DFS를 써야하는 일이 없다  
Flood Fill은 어느것을 써도 되는데, 거리 측정은 BFS만 할 수 있으니 DFS를 쓸 일이 없다.  
그래서 다차원 배열에서 순회하는 문제를 풀 때 BFS만 쓰게 된다  

DFS는 나중에 그래프와 트리라는 자료구조를 배울 때 필요하게 된다  

어느정도 정석적인 구현  

```cpp
#define X first
#define Y second // pair에서 first, second를 줄여서 쓰기 위해서 사용

int board[502][502] = {
    { 1, 1, 1, 0, 1, 0, 0, 0, 0, 0} ,
    { 1, 0, 0, 0, 1, 0, 0, 0, 0, 0} ,
    { 1, 1, 1, 0, 1, 0, 0, 0, 0, 0} ,
    { 1, 1, 0, 0, 1, 0, 0, 0, 0, 0} ,
    { 0, 1, 0, 0, 0, 0, 0, 0, 0, 0} ,
    { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0} ,
    { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0} }; // 1이 파란 칸, 0이 빨간 칸에 대응
bool vis[502][502]; // visit 해당 칸을 방문했는지 여부를 저장
int n = 7, m = 10; // n = 행의 수, m = 열의 수

int dx[4] = { 1, 0, -1, 0 };
int dy[4] = { 0, 1, 0, -1 }; // 상하좌우 네 방향을 의미

int main(void)
{
    stack<pair<int,int>> S;

    vis[0][0] = 1; // @ (0, 0)을 방문했다고 명시
    S.push({ 0, 0 }); // 큐에 시작점인 (0, 0)을 삽입.
    
    while(!S.empty())
    {
        pair<int,int> cur = S.top();
        S.pop();
        
        cout << '(' << cur.X << ", " << cur.Y << ") -> ";
        
        // 상하좌우 칸을 살펴볼 것이다.
        for (int dir = 0; dir < 4; dir++)
        { 
            // nx, ny에 dir에서 정한 방향의 인접한 칸의 좌표가 들어감
            int nx = cur.X + dx[dir];
            int ny = cur.Y + dy[dir];

            // @ 아랫조건보다 먼저, 범위 밖일 경우 넘어감
            if (nx < 0 || nx >= n || ny < 0 || ny >= m)
                continue; 
            // 이미 방문한 칸이거나 파란 칸이 아닐 경우
            if (vis[nx][ny] || board[nx][ny] != 1)
                continue;

            // (nx, ny)를 방문했다고 명시
            // @ 넣을때 표시하지 않고, 뺄 때 표시한다면 중복된 요소가 스택에 들어갈 수 있어서 메모리 초과, 시간 초과가 날 수 있다
            vis[nx][ny] = 1; 
            S.push({ nx, ny });
        }
    }
}
```

### 트리에서의 DFS

임의의 시작점을 잡고 BFS를 돌리면, 그 시작점을 루트로 정해서 트리를 재배치했을 때의 순으로 방문.  
이때, 루트가 아닌 아무 정점이나 잡고 생각을 해보면, 인접한 정점들 중에 자신의 부모를 제외하고는 전부 자신의 자식이라 아직 방문을 하지 않은 상태.  

즉, 트리에서는 DFS 과정 속에서 자신의 자식들을 전부 스택에 넣어주기만 하면 된다.  
이 말은 곧 자신과 이웃한 정점들에 대해 부모만 빼고 나머지는 전부 스택에 넣으면 된다.  

그렇기 때문에 vis 배열을 들고갈 필요가 없이, 그냥 부모가 누구인지만 저장하고 있으면 된다.  
부모의 정보는 DFS를 돌리면서 자식 정점을 스택에 집어넣어줄 때 채워줄 수 있다.  
이렇게 DFS를 돌리면, 각 정점의 부모 정보를 알아낼 수 있다. (필요하다면 활용 가능)  

```cpp
vector<int> adj[8]; // Adjacency 인접 리스트
// bool vis[8]; // 방문 여부를 저장 (parent로 대체)
int parent[8]; // 부모 정점을 저장 (루트의 부모는 자연스럽게 0)
int depth[8]; // 깊이를 저장 (부가적인 정보)

void dfs(int start)
{
    stack<int> s;
    s.push(start);
    vis[start] = true;
    while (s.empty() == false)
    {
        int cur = s.top();
        s.pop();
        cout << cur << ' ';
        for (int next: adj[cur])
        {
            // if (vis[next])
            if (parent[cur] == next)
                continue;

            s.push(next);
            // vis[next] = true;
            parent[next] = cur;
            depth[next] = depth[cur] + 1;
        }
    }
}
```

재귀함수로 구현할 수도 있다.  

```cpp
vector<int> adj[8]; // Adjacency 인접 리스트
int parent[8]; // 부모 정점을 저장 (루트의 부모는 자연스럽게 0)
int depth[8]; // 깊이를 저장 (부가적인 정보)

void dfs(int cur)
{
    cout << cur << ' ';
    for (int next: adj[cur])
    {
        if (parent[cur] == next)
            continue;
        parent[next] = cur;
        depth[next] = depth[cur] + 1;
        dfs(next);
    }
}

// or
// 부모나 깊이를 저장할 필요가 없다면

void dfs(int cur, int parent)
{
    cout << cur << ' ';
    for (int next: adj[cur])
    {
        if (parent == next)
            continue;
        dfs(next, cur);
    }
}
```
