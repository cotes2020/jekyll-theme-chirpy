---
title: "BFS"
# description: ""
categories: [컴퓨터, 알고리즘]
tags: [알고리듬, Search]
image: "/assets/img/background/kururu-lab.jpg"
hidden: true

date: 2024-02-23. 03:30
# last_modified_at: 2024-02-23. 17:29
# last_modified_at: 2024-03-22. 01:03
last_modified_at: 2024-07-15. 06:15
---

{% include embed/youtube.html id='ftOmGdm95XI' %}

## 정의

---

DFS에서 `Stack` 대신 `Queue`를 쓰는  

BFS | Breadth First Search | 너비 우선 탐색  
다차원 배열에서 각 칸을 방문할 때 너비를 우선으로 방문하는 알고리듬  

너비를 우선으로 방문?  
설명할 방법이 없다  

원래 BFS는 `Graph`에서 모든 노드를 방문하기 위한 알고리듬  

## 구현

---

### 다차원 배열에서의 BFS

1. BFS에는 자료를 담을 `Queue`가 필요
2. 시작 좌표에 방문 표시를 남기고 `Queue`에 넣기
3. `Queue`가 빌 때까지 원소를 `pop`하고 4번 처리
4. 해당 좌표와 그 상하좌우에 대해 방문 표시를 남기고 해당 칸을 `Queue`에 삽입 (이미 방문했었다면 pass)

방문 표시를 남기기 때문에 모든 칸이 `Queue`에 1번씩 들어가므로,  
시간복잡도는 칸이 N개일 때 O(N), = 행 R개 열 C개일 때 O(RC)  

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
    queue<pair<int,int>> Q;

    vis[0][0] = 1; // @ (0, 0)을 방문했다고 명시
    Q.push({ 0, 0 }); // 큐에 시작점인 (0, 0)을 삽입.
    
    while(!Q.empty())
    {
        pair<int,int> cur = Q.front();
        Q.pop();
        
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
            // @ 넣을때 표시하지 않고, 뺄 때 표시한다면 중복된 요소가 큐에 들어갈 수 있어서 메모리 초과, 시간 초과가 날 수 있다
            vis[nx][ny] = 1;
            Q.push({ nx, ny });
        }
    }
}
```

### 트리에서의 BFS

![트리에서의 BFS](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FcJXaCd%2FbtrnP6QRW8z%2FrKKUP02Tb95iML46jwdgP1%2Fimg.png)

임의의 시작점을 잡고 BFS를 돌리면, 그 시작점을 루트로 정해서 트리를 재배치했을 때의 높이 순으로 방문.  
이때, 루트가 아닌 아무 정점이나 잡고 생각을 해보면, 인접한 정점들 중에 자신의 부모를 제외하고는 전부 자신의 자식이라 아직 방문을 하지 않은 상태.  

즉, 트리에서는 BFS 과정 속에서 자신의 자식들을 전부 큐에 넣어주기만 하면 된다.  
이 말은 곧 자신과 이웃한 정점들에 대해 부모만 빼고 나머지는 전부 큐에 넣으면 된다.  

그렇기 때문에 vis 배열을 들고갈 필요가 없이, 그냥 부모가 누구인지만 저장하고 있으면 된다.  
부모의 정보는 BFS를 돌리면서 자식 정점을 큐에 집어넣어줄 때 채워줄 수 있다.  
이렇게 BFS를 돌리면, 각 정점의 부모 정보를 알아낼 수 있다. (필요하다면 활용 가능)  

또한 각 정점의 깊이도 알 수 있다.  

시간 복잡도는 O(V+E), 이때 트리에서 E = V-1 이므로, O(V)  

```cpp
vector<int> adj[8]; // Adjacency 인접 리스트
// bool vis[8]; // 방문 여부를 저장 (parent로 대체)
int parent[8]; // 부모 정점을 저장 (루트의 부모는 자연스럽게 0)
int depth[8]; // 깊이를 저장 (부가적인 정보)

void bfs(int start)
{
    queue<int> q;
    q.push(start);
    vis[start] = true;
    while (q.empty() == false)
    {
        int cur = q.front();
        q.pop();
        cout << cur << ' ';
        for (int next: adj[cur])
        {
            // if (vis[next])
            if (parent[cur] == next)
                continue;

            q.push(next);
            // vis[next] = true;
            parent[next] = cur;
            depth[next] = depth[cur] + 1;
        }
    }
}
```

## 메모

---

- 그림판에서 페인트(통) 기능
- 외부 윤곽석을 따라 구분되는 영역의 색을 한 번에 바꾸는 기능

- Flood Fill
- 거리측정 (최단거리)
  - 방문 표시 대신 시작점과의 거리
  - -1로 초기화하면 방문 표시 여부도 알 수 있다
- 시작점이 여러 개
  - 여러 시작점을 큐에 다 넣으면 됨
  - BFS 성질 -> 큐에는 요소가 거리(시간) 순서대로 들어감 (큐에 요소가 쌓이는 순서는 거리 순)
- 시작점이 두 종류
  - 모두 BFS를 돌림
  - 하나를 먼저 BFS. 순서를 다르게 할 수 있다면 (전파가 서로 영향을 주지 않으면, 한쪽만 영향을 준다면)
  - 전파가 서로 영향을 준다면 -> 백 트래킹
- 1차원에서의 BFS
  - 단순히 전파가 좌우로만 이루어지는
