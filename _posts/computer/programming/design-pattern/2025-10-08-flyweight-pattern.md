---
title: "경량 패턴"
# description: ""
categories: [컴퓨터, 프로그래밍]
tags: [Design-Pattern]
image: "/assets/img/background/kururu-lab.jpg"

date: 2025-10-08. 14:32 # Init
# last_modified_at: 2025-10-08. 17:27 # E 초고
last_modified_at: 2025-10-08. 17:56 # E + 예제: 유니티 스크립터블 오브젝트를 이용한 경량 패턴 구현
---

## 머리말

---

- **참고:**
  - ['게임 프로그래밍 패턴: 더 빠르고 깔끔한 게임 코드를 구현하는 13가지 디자인 패턴' - 로버트 나이스트롬](https://gameprogrammingpatterns.com/flyweight.html)
- **핵심:**
  - 중복되는 것은 분리하여 공유. (= 최적화)
  - 공유를 통해 많은 수의 소립 객체들을 효과적으로 지원한다.

내가 생각하기에, 특별히 어떤 문제에 대해 '경량 패턴을 사용했다!' 라고 말하기는 어렵고, 개발하면서 자연히 체득하게 되는 기술을 구체적으로 개념화 시킨 것에 가까운 것 같다.  

## 예제-숲 렌더링

---

울창한 숲을 렌더링한다고 가정해보자.  

```cpp
// #1 나무마다 필요한 데이터
class Tree {
private:
    Mesh mesh_;
    Texture bark_;
    Texture leaves_;
    Vector position_;
    double height_;
    double thickness_;
    Color bartTint_;
    Color leafTint_;
};
```

우선 나무마다 필요한 데이터는 위와 같다.  
데이터가 많은 데다가 메쉬와 텍스처는 크기도 크다.  

핵심은 숲에 나무가 수천 그루 넘게 있다고 해도 대부분 비슷해 보인다는 점이다. 그렇다면 모든 나무를 같은 메시와 텍스처로 표현할 수 있을 것 같다. 즉, 나무 객체에 들어 있는 데이터 대부분이 인스턴스별로 다르지 않다는 뜻이다.  

객체를 반으로 쪼개어 이런 점을 명시적으로 모델링할 수 있다. 모든 나무가 다 같이 사용하는 데이터를 뽑아내 새로운 클래스에 모아보자.  

```cpp
// #2
class TreeModel {
private:
    Mesh mesh_;
    Texture bark_;
    Texture leaves_;
}

class Tree {
private:
    TreeModel* model_;

    Vector position_;
    double height_;
    double thickness_;
    Color bartTint_;
    Color leafTint_;
};
```

게임 내에서 같은 메시와 텍스처를 여러 번 메모리에 올릴 필요가 전혀 없기 때문에 `TreeModel` 객체는 하나만 존재하게 된다. 이제 각 나무 인스턴스는 공유 객체인 `TreeModel`을 참조하기만 한다. `Tree` 클래스에는 인스턴스별로 다른 상태 값만 남겨둔다.  

### 덤-GPU 인스턴싱

메모리에 객체를 저장하기 위해서라면 이 정도로 충분하다.  

하지만 렌더링은 또 다른 얘기다. 화면에 숲을 그리기 위해서는 먼저 데이터를 GPU로 전달해야 한다. 어떤 식으로 자원을 공유하고 있는지를 그래픽 카드도 이해할 수 있는 방식으로 표현해야 한다.  

GPU로 보내는 데이터 양을 최소화하기 위해서는 공유 데이터인 `TreeModel`을 딱 한 번만 보낼 수 있어야 한다. 그런 후에 나무마다 값이 다른 위치, 색, 크기를 전달하고, 마지막으로 GPU에 '전체 나무 인스턴스를 그릴 때 공유 데이터를 사용해'라고 말하면 된다.  

다행히, 최신 GPU나 API에서는 이런 기능을 제공한다. Direct3D, OpenGL 모두 **인스턴스 렌더링**을 지원한다.  

이런 API에서 인스턴스 렌더링을 하려면 데이터 스트림이 두 개 필요하다. 첫 번쨰 스트림에는 숲 렌더링 예제의 메시나 텍스처처럼 여러 번 렌더링되어야 하는 공유 데이터가 들어간다. 두 번째 스트림에는 인스턴스 목록과, 이들 인스턴스를 첫 번째 스트림 데이터를 이용해 그릴 때 각기 다르게 보이기 위해 필요한 매개변수들이 들어간다. 이때 `draw` 호출 한 번만으로 전체 숲을 다 그릴 수 있다.  

## Flyweight Pattern

---

경량 패턴은 어떤 객체의 개수가 너무 많아서 좀 더 가볍게 만들고 싶을 때 사용한다.  

인스턴스 렌더링에서는 메모리 크기보다는 렌더링할 나무 데이터를 하나씩 GPU 버스로 보내는 데 걸리는 **시간**이 중요하지만, 기본 개념은 경량 패턴과 같다.  

이런 문제를 해결하기 위해 경량 패턴은 객체 데이터를 두 종류로 나눈다. 먼저 모든 객체의 데이터 값이 같아서 공유할 수 있는 데이터를 모은다. 이런 데이터를 GoF는 **고유 상태**라고 했다. (책에서는 **자유 문액** 상태라고 한다.) 예제에서는 나무 형태나 텍스처가 이에 해당한다.  

나머지 데이터는 인스턴스별로 값이 다른 **외부 상태**에 해당한다. 예제에서는 나무의 위치, 크기, 색 등이 이에 해당한다. 예제 코드에서 봤듯이, 경량 패턴은 한 개의 고유 상태를 다른 객체에서 공유하게 만들어 메모리 사용량을 줄이고 있다.  

여기까지만 보면 기초적인 자원 공유 기법이지 패턴이라고 부를 정도는 아닌 것처럼 보인다. 이 예제에서는 공유 상태를 `TreeModel` 클래스로 깔끔하게 분리할 수 있어서 그렇게 보이는 측면도 있다.  

공유 객체가 명확하지 않은 경우 경량 패턴은 잘 드러나 보이지 않는다(그만큼 더 교모하다). 그런 경우에는 하나의 객체가 신기하게도 여러 곳에 동시에 존재하는 것처럼 보인다. 이런 예를 하나 들어보겠다.  

## 예제-지형 정보

---

나무를 심을 땅도 게임에서 표현해야 한다. 보통 풀, 흙, 언덕, 호수, 강 같은 다양한 지형을 이어 붙여서 땅을 만든다. 여기에서는 땅을 타일 기반으로 만들 것이다. 즉, 땅은 작은 타일들이 모여 있는 거대한 격자인 셈이다. 모든 타일은 지형 종류 중 하나로 덮여 있다.  

지형 종류에는 게임플레이에 영향을 주는 여러 속성이 들어 있다.  

- 플레이어가 얼마나 빠르게 이동할 수 있는지를 결정하는 이동 비용 값
- 강이나 바다처럼 보트로 건너갈 수 있는 곳인지 여부
- 렌더링할 때 사용할 텍스처

```cpp
enum Terrain {
    TERRAIN_GRASS,
    TERRAIN_HILL,
    TERRAIN_RIVER
    // ...
};

class World {
private:
   Terrain tiles_[WIDTH][HEIGHT];
   // C/C++은 2차원 배열 데이터가 전부 메모리에 같이 붙어 있어 효과적 (i.e. int[2][3]과 int[6] 메모리 구조 똑같음)
   // C#/Java 같은 매니지드 언어는 가로 배열이 각자 세로 배열을 참조하는 형태라 그다지 메모리 친화적이지 않음.
   // 실제 코드에서는 상세 구현을 잘 만든 2차원 격자 자료구조 안에 숨기는 게 좋다.
};

int World::getMovementCost(int x, int y) {
    switch (tiles_[x][y]) {
        case TERRAIN_GRASS: return 1;
        case TERRAIN_HILL: return 3;
        case TERRAIN_RIVER: return 2;
        // ...
    }
}

bool World::isWater(int x, int y) {
    switch (tiles_[x][y]) {
        case TERRAIN_GRASS: return false;
        case TERRAIN_HILL: return false;
        case TERRAIN_RIVER: return true;
        // ...
    }
}
```

이 코드는 동작하긴 하지만, 지저분하다. 이용 비용이나 물인지 땅인지 여부는 지형에 관한 데이터인데 이 코드에서는 하드코딩되어 있다. 게다가 같은 지형 종류에 대한 데이터가 여러 메서드에 나뉘어 있다. 이런 데이터는 하나로 합쳐서 캡슐화하는 게 좋다. 그러라고 객체가 있는 것이니 말이다.  

```cpp
class Terrain
{
public:
    Terrain(int movementCost, bool isWater, Texture texture)
    : movementCost_(movementCost),
      isWater_(isWater),
      texture_(texture) {
    }

    int getMovementCost() const { return movementCost_; }
    bool isWater() const { return isWater_; }
    const Texture& getTexture() const { return texture_; }
    // const Method
    // 같은 Terrain 객체를 여러 곳에서 공유해서 쓰기 때문에, 한 곳에서 값을 바꾼다면 그 결과가 여러 군데에서 동시에 나타나게 된다.
    // 이건 원하는 바가 아니다. 메모리를 줄여보겠다고 객체를 공유했는데 그게 코드 기능에 영향을 미쳐서는 안 된다. 이런 이유로 경량 객체는 변경 불가능한(immutable) 상태로 만드는 게 보통이다.

private:
    int movementCost_;
    bool isWater_;
    Texture texture_;
};
```

하지만 타일마다 `Terrain` 인스턴스를 하나씩 만드는 비용은 피하고 싶다. `Terrain` 클래스에는 타일 위치와 관련된 내용은 전혀 없는 것을 볼 수 있다. 경량 패턴식으로 얘기하자면 모든 지형 상태는 '고유'하다. 즉 '자유문맥'에 해당한다.  

따라서 지형 종류별로 `Terrain` 객체가 여러 개 있을 필요가 없다. 지형에 들어가는 모든 강 타일은 전부 동일하다. 즉, `World` 클래스 격자 멤버 변수에 열거형이나 `Terrain` 객체 대신 `Terrain` 객체의 포인터를 넣을 수 있다.  

```cpp
class World {
private:
    Terrain* tiles_[WIDTH][HEIGHT];
}
```

지형 종류가 같은 타일들은 모두 같인 `Terrain` 인스턴스 포인터를 갖는다.  

`Terrain` 인스턴스는 여러 곳에서 사용되다 보니, 동적으로 할당하면 생명주기를 관리하기 좀 더 어렵다. 그래서 World 클래스에 멤버변수로 저장한다.  

```cpp
class World {
public:
    World()
    : grassTerrain_(1, false, GRASS_TEXTURE),
      hillTerrain_(3, false, HILL_TEXTURE),
      riverTerrain_(2, true, RIVER_TEXTURE)
    {}

private:
    Terrain grassTerrain_;
    Terrain hillTerrain_;
    Terrain riverTerrain_;
    // ...
}
```

이렇게 구현하면 다음과 같이 땅 위를 채울 수 있게된다.  

```cpp
void World::generateTerrain() {
    // 땅에 풀을 채운다.
    for(int x = 0; x < WIDTH; x++) {
        for(int y = 0; y < HEIGHT; y++) {
            // 언덕을 몇 개 놓는다.
            if (random(10) == 0) {
                tiles_[x][y] = &hillTerrain_;
            } else {
                tiles_[x][y] = &grassTerrain_;
            }
        }
    }
    
    // 강을 놓는다.
    int x = random(WIDTH);
    for(int y = 0; y < HEIGHT; y++) {
        tiles_[x][y] = &riverTerrain_;
    }
}
```

이제 지형 속성 값을 `World`의 메서드 대신 `Terrain` 객체에서 바로 얻을 수 있다.  

```cpp
const Terrain& World::getTile(int x, int y) const {
    return *tiles_[x][y];
}
```

`World` 클래스는 더 이상 지형 세부 정보와 커플링되지 않는다. 타일 속성을 `Terrain` 객체에서 바로 얻을 수 있다.  

```cpp
int cost = world.getTile(2, 3).getMovementCost();
```

이제 객체로 작동하는 근사한 API가 되었다. 게다가 포인터는 열거형과 비교해도 성능 면에서 거의 뒤지지 않는다.  

### 성능에 대해서

지형 데이터를 포인터로 접근한다는 것은 간접 조회한다는 뜻이다. 이동 비용 같은 지형 데이터 값을 얻으려면 먼저 격자 데이터로부터 지형 객체 포인터를 얻은 다음에, 포인터를 통해서 이동 비용 값을 얻어야 한다. 이렇게 포인터를 따라가면 캐시 미스가 발생할 수 있어 성능이 조금 떨어질 수는 있다.  

최적화의 황금률은 언제나 **먼저 측정**하는 것이다. 최신 컴퓨터 하드웨어는 너무 복잡해서 더이상 추측만으로는 최적화하기 어려워졌다. 측정해본 결과로는 경량 패턴을 써도 열거형을 쓴 것과 비교해서 성능이 나빠지지 않았따. 오히려 열거형에 비해 훨씬 빨랐다. 하지만 이건 객체가 메모리에 어떤 식으로 배치되느냐에 따라 달라질 수 있다.  

**확실한** 것은 경량 객체를 한 번은 고려해봐야 한다는 점이다. 경량 패턴을 사용하면 객체를 마구 늘리지 않으면서도 객체지향 방식의 장점을 취할 수 있다. 열거형을 선언해 수많은 `switch`문을 만들 생각이라면, 경량 패턴을 고려해보자. 성능이 걱정된다면, 유지보수하기 어려운 형태의 코드를 고치기 전에 적어도 프로파일링이라도 먼저 해보자.

## 예제-유니티 스크립터블 오브젝트

---

유니티에서 스크립터블 오브젝트를 활용해 경량 패턴을 구현할 수도 있다. (꼭 스크립터블 오브젝트를 이용해야 하는 건 아니지만, 스크립터블 오브젝트를 이용하면 쉽고 효과적으로 구현할 수 있다.)  

```cs
public class Monster : MonoBehaviour
{
    [SerializeField] private string monsterName;
    [SerializeField] private string monsterDescription;
    [SerializeField] private int hpMax;
    // ...
}
```

몬스터들의 정보를 각 `MonoBehaviour`에 변수로 구현할 수 있다 (좋은 방법은 아니다). `monsterName`, `monsterDescription`, `hpMAX` 같은 값들을 보통 런타임 중에 변하지 않는 정보다. 만약 똑같은 몬스터를 여러 번 생성하는 경우, 이런 값들은 메모리에 중복되어 올라가게 되고, 이는 낭비가 될 수 있다.  

이떄, 스크립터블 오브젝트를 이용하면 쉽고 효과적으로 경량 패턴을 구현할 수 있다.  

```cs
public class MonsterData : ScriptableObject
{
    [field: SerializeField] public string Name { get; private set; }
    [field: SerializeField] public string Description { get; private set; }
    [field: SerializeField] public int HpMax { get; private set; }
    // ...
}

public class Monster : MonoBehaviour
{
    [SerializeField] private MonsterData data;
}
```

`monsterName`, `monsterDescription`, `hpMAX`처럼 공용으로 사용하는 데이터는 `MonsterData` 같은 스크립터블 오브젝트를 만들어 분리한다. 그리고 `Monster`는 스크립터블 오브젝트를 참조하도록 수정한다.  

이렇게 되면 똑같은 몬스터를 몇 백, 몇 천 마리 생성하더라도 공용 데이터는 메모리에 스크립터블 오브젝트 하나만 올라간다. (= 메모리 최적화)  

## 메모

---

- **관련 자료:**
  - 타일 예제에서는 지형 종류별로 `Terrain` 인스턴스를 미리 만들어 `World`에 저장했다. 덕분에 공유 객체를 찾고 재사용하기 쉬웠다. 하지만 경량 객체를 미리 전부 만들고 싶지 않은 경우도 많다. 어떤 경량 객체가 실제로 필요할지를 예측할 수 없다면, 필요할 때 만드는 게 낫다. 공유 기능을 유지하고 싶다면, 인스턴스를 요청받았을 때 이전에 같은 걸 만들어놓은 게 있는지 확인해보고, 있다면 그걸 반환하면 된다. 이러려면 객체를 생성할 때 기존 객체가 있는지를 먼저 확인하게 할 수 있도록 생성 코드를 인터페이스 밑으로 숨겨둬야 한다. 이런 식으로 생성자를 숨기는 방식은 GoF의 **팩토리 메서드 패턴**의 한 예이기도 하다.
  - 이전에 만들어놓은 경량 패턴 객체를 반환하려면, 이미 생성해놓은 객체를 찾을 수 있도록 풀을 관리해야 한다. 이름에서 알 수 있듯이 **객체 풀 패턴**이 이런 데 유용하다.
  - **상태 패턴**을 쓸 때는, 상태의 아이디와 메서드만으로 충분해서 상태 기계에서 사용되는 '상태' 객체에는 멤버 변수가 하나도 없는 경우가 종종 있다. 이럴 때 경량 패턴을 적용하면 상태 인스턴스 하나를 여러 상태 기계에서 동시에 재사용할 수 있다.
- **키워드:**
  - [Design Pattern (디자인 패턴)](/posts/design-pattern/)
  - [Flyweight Pattern (경량 패턴, 플라이급 패턴, 플라이웨이트 패턴)](/posts/flyweight-pattern/)
  - fine-grained (소림)
  - 타입 객체 패턴
    - [예제-숲 렌더링](#예제-숲-렌더링)
      - 객체 상태 일부를 여러 인스턴스가 공유하는 다른 객체에 위임한다는 점에서 타입 객체 패턴과 비슷해 보인다. 하지만 의도가 다르다. 타입 객체 패턴의 목표는 '타입'을 직접 만든 객ㅊ 모델에 위임함으로써 정의해야 하는 클래스 개수를 줄이는 것이다. 메모리 공유는 어디까지나 덤이다. 경량 패턴은 순수하게 최적화가 목표다.
  - Instanced Rendering (인스턴스 렌더링)
  - Intrinsic State (고유 상태), Context-Free (자유 문맥)
  - Extrinsic State (외부 상태)
  - C++ Const Method, Const Instance
  - Indirect Lookup (간접 조회)
  - 데이터 지역성 패턴
    - 포인터 따라가기, 캐시 미스
  - 팩토리 메서드 패턴
  - 객체 풀 패턴
    - Pool 풀
  - 상태 패턴
