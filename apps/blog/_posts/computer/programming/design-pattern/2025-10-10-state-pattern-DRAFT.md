---
title: "상태 패턴"
# description: ""
categories: [컴퓨터, 프로그래밍]
tags: [Design-Pattern]
image: "/assets/img/background/kururu-lab.jpg"
hidden: true

date: 2025-10-10. 14:52 # Init
last_modified_at: 2025-10-11. 00:08 # E 초고 (단순 복붙 및 글 모양 정리)
---

## 머리말

---

- **참고:**
  - ['게임 프로그래밍 패턴: 더 빠르고 깔끔한 게임 코드를 구현하는 13가지 디자인 패턴' - 로버트 나이스트롬](https://gameprogrammingpatterns.com/state.html)
  - [Wikipedia](https://en.wikipedia.org/wiki/State_pattern)
- **핵심:**
  - 객체의 내부 상태에 따라 스스로 행동을 변경할 수 있게 허가하는 패턴으로, 이렇게 하면 객체는 마치 자신의 클래스를 바꾸는 것처럼 보인다.

겉으로 보기에는 상태 패턴을 다루지만 이를 위해서는 좀 더 근본 개념은 **유한 상태 기계 (FSM)**을 언급할 수 밖에 없었다. 그러다 보니 **계층형 상태 기계 (Hierarchical State Machine)**와 **푸시다운 오토마타 (Pushdown Automata)**까지 이어졌다.  

## 예제-플랫포머

---

간단한 횡스크롤 플랫포머를 만든다고 해보자. 게임 월드의 캐릭터가 사용자 입력에 따라 반응하도록 구현해야 한다. B버튼을 누르면 점프하는 것부터 간단하게 만들어 보자.  

```cpp
void Heroine::handleInput(Input input) {
    if (input == PRESS_B) {
        yVelocity = JUMP_VELOCITY;
        setGraphics(IMAGE_JUMP);
    }
}
```

버그를 눈치챘는가? '공중 점프'를 막는 코드가 없다. 주인공이 공중에 있는 동안 B를 연타하면 계속 떠 있을 수 있다. 이 버그는 `Heroine` 클래스에 `isJumping_` 불타입 필드를 추가해 점프 중인지 검사하면 간단하게 고칠 수 있다.  

```cpp
void Heroine::handleInput(Input input) {
    if (input == PRESS_B) {
        if (!isJumping_) {
            isJumping_ = true;
            // ...
        }
    }
}
```

주인공이 땅에 있을 때 아래 버튼을 누르면 엎드리고, 버튼을 떼면 다시 일어서는 기능을 추가해보자.  

```cpp
void Heroine::handleInput(Input input) {
    if (input == PRESS_B) {
        // 점프 중이 아니라면 점프한다.
    } else if (input == PRESS_DOWN) {
        if (!isJumping_) {
            setGraphics(IMAGE_DUCK);
        }
    } else if (input == RELEASE_DOWN) {
        setGraphics(IMAGE_STAND);
    }
}
```

이번에도 버그를 찾아냈는가?  

1. 엎드리기 위해 아래 버튼을 누른 뒤
2. B 버튼을 눌러 엎드린 상태에서 점프하고 나서
3. 공중에서 아래 버튼을 떼면

점프 중인데도 땅에 서 있는 모습으로 보인다. 플래스 변수가 더 필요한다.  

```cpp
void Heroine::handleInput(Input input) {
    if (input == PRESS_B) {
        if (!isJumping_ && !isDucking_) {
            // 점프 코드.
        }
    } else if (input == PRESS_DOWN) {
        if (!isJumping_) {
            isDucking_ = true;
            setGraphics(IMAGE_DUCK);
        }
    } else if (input == RELEASE_DOWN) {
        if (isDucking_) {
            isDucking_ = false;
            setGraphics(IMAGE_STAND);
        }
    }
}
```

이번에는 점프 중에 아래 버튼을 눌러 내려찍기 공격을 할 수 있게 해보자.  

```cpp
void Heroine::handleInput(Input input) {
    if (input == PRESS_B) {
        if (!isJumping_ && !isDucking_) {
            // 점프 코드.
        }
    } else if (input == PRESS_DOWN) {
        if (!isJumping_) {
            isDucking_ = true;
            setGraphics(IMAGE_DUCK);
        } else {
            isJumping_ = false;
            setGraphics(IMAGE_DIVE);
        }
    } else if (input == RELEASE_DOWN) {
        if (isDucking_) {
            // 일어서기 코드.
        }
    }
}
```

이번에도 버그를 찾았는가? 공중 점프를 막기 위해서 점프 중인지는 검사하지만, 내려찍기 중인지는 검사하지 않는다. 또 플래그 변수를...

뭔가 방향을 잘못 잡은게 분명하다. 코드가 얼마 없는데도 조금만 건드리면 망가진다. 심지어 걷기 구현을 시작하지도 않았다. 이동과 관련해서 할 게 많은데 이런 식이면 버그에 파묻혀서 구현을 못 끝낼 것이다.  

### FSM이 우리르 구원하리라

꽉 막혔다 싶으면 컴퓨터를 끄고 펜과 종이만으로 플로차트를 그려보자. 주인공이 할 수 있는 동작 (서 있기, 점프, 엎드리기, 내려찍기)을 각각 네모칸에 적어 넣는다. 어떤 버튼을 눌렀을 때 상태가 바뀐다면 이전 상태에서 다음 상태로 도착하는 화살표를 그린 뒤 눌렀던 버튼을 선에 적는다.  

축하한다. 방금 여러분은 **유한 상태 기계 (FSM)**을 만들었다. FSM은 컴퓨터 과학 분야 중의 하나인 **오토마타 이론**에서 나왔다. 오토마다 중에는 유명한 튜링 기계도 있다. FSM은 이 분야 주제 중에서 가장 간단한 축에 속한다.  

요점은 이렇다.  

- 가질 수 있는 '상태'가 한정된다.
  - 예제에서는 서기, 점프, 엎드리기, 내려찍기가 있다.
- 한 번에 '한 가지' 상태만 될 수 있다.
  - 주인공은 점프와 동시에 서 있을 수 없다. 동시에 두 가지 상태가 되지 못하도록 막는 게 FSM을 쓰는 이유 중 하나다.
- '입력'이나 '이벤트'가 기계에 전달된다.
  - 예제로 치면 버튼 누르기와 버튼 떼기가 이에 해당한다.
- 각 상태에는 입력에 따라 다음 상태로 바뀌는 '전이 (Transition)'가 있다.
  - 입력이 들어왔을 때 현재 상태에 해당하는 전이가 있다면 전이가 가리키는 다음 상태로 변경한다.

예를 들어, 서 있는 동안 아래 버튼을 누르면 엎드리기 상태로 전이한다. 점프하는 동안 아래 버튼을 누르면 내려찍기 상태로 전이한다. 현재 상태에서 들어온 입력에 대한 전이가 없을 경우 입력을 무시한다.  

순수하게 형식만 놓고 보면 상태, 입력, 전이가 FSM의 전부다. 컴파일러는 우리가 끄적거린 플로차트를 이해하지 못하니 구현해야 한다. GoF의 상태 패턴이 FSM을 **구현하는** 방법 중 하나이지만 뒤에서 다루기로 하고 먼저 간단한 방법부터 알아보자.

### 열거형과 다중 선택문

`Heroin` 클래스의 문제점 하나는 불리언 변수 값 조합이 유효하지 않을 수 있다는 점이다. 예를 들어 `isJumping_`과 `isDucking_`은 동시에 참이 될 수 없다. 여러 플래그 변수 중에서 하나만 참일 떄가 많다면 열거명 (`enum`)이 필요하다는 신호다.  

우리 예제에서는 FSM 상태를 열거형으로 정의할 수 있다.  

```cpp
enum State
{
    STATE_STANDING,
    STATE_JUMPING,
    STATE_DUCKING,
    STATE_DIVING
};
```

이제 `Heroine`에는 플래그 변수 여러 개 대신 `state_` 필드 하나만 있으면 된다. 분기 순서도 바뀐다. 이전에는 입력에 따라 먼저 분기한 **뒤에** 상태에 따라 분기했다. 따라서 하나의 버튼 입력에 대한 코드는 모아둘 수 있었으나 하나의 상태에 대한 코드는 흩어져 있었다. 상태 관련 코드를 한 곳에 모아두기 위해 먼저 상태에 따라 분기하게 하자.  

```cpp
void Heroine::handleInput(Input input) {
    switch (state_) {
        case STATE_STANDING:
            if (input == PRESS_B) {
                state_ = STATE_JUMPING;
                yVelocity = JUMP_VELOCITY;
                setGraphics(IMAGE_JUMP);
            } else if (input == PRESS_DOWN) {
                state_ = STATE_DUCKING;
                setGraphics(IMAGE_DUCK);
            }
            break;
        case STATE_JUMPING:
            if (input == PRESS_DOWN) {
                state_ = STATE_DIVING;
                setGraphics(IMAGE_DIVE);
            }
            break;
        case STATE_DUCKING:
            if (input == RELEASE_DOWN) {
                state_ = STATE_STANDING;
                setGraphics(IMAGE_STAND);
            }
        break;
    }
}
```

사소해 보여도 코드가 훨씬 나아졌다. 분기문을 다 없애진 못했지만 업데이트해야 할 상태 변수를 하나로 줄였고, 하나의 상태를 관리하는 코드는 깔끔하게 한곳에 모았다. 열거형은 상태 기계를 구현하는 가장 간단한 방법이고, 이 정도만으로 충분할 때도 꽤 있다. (특히 주인공은 이제 **유효하지 않은** 상태가 될 수 없게 되었다. 불리언 변수 여러 개로 상태를 관리하다 보면 일부 값 조합은 유효하지 않을 수 있다. 열거형을 그럴 일이 없다.)  

열거형 만으로는 부족할 수도 있다. 이동을 구현하되, 엎드려 있으면 기가 모여서 놓는 순간 특수 공격을 쏠 수 있게 만든다고 해보자. 엎드려서 기를 모으는 시간을 기록해야 한다.  

이를 위해 `Heroine`에 `chargeTime_` 멤버 변수를 추가하자. 매 프레임마다 호출되는 `update()` 메서드는 이미 있었다고 가정하고 다음 코드를 추가해보자. (업데이트 메서드 패턴)  

```cpp
void Heroine::update() {
    if (state_ == STATE_DUCKING) {
        chargeTime_++;
        if (chargeTime_ > MAX_CHARGE) s{
            superBomb();
        }
    }
}
```

엎드릴 때마다 시간을 초기화 해야 하니 `handleInput()`을 조금 바꿔보자.  

```cpp
void Heroine::handleInput(Input input) {
    switch (state_) {
        case STATE_STANDING:
            if (input == PRESS_DOWN) {
                state_ = STATE_DUCKING;
                chargeTime_ = 0;
                setGraphics(IMAGE_DUCK);
            }

            // 다른 입력 코드...
            break;

            // 다른 상태 처리...
    }
}
```

기 모으기 공격을 추가하기 위해 함수 두 개를 수정하기 위해 엎드리기 상태에서만 의미있는 `changeTime_`이라는 필드를 `Heroine`에 추가해야 했다. 이것보다는 모든 코드와 데이터를 한 곳에 모아둘 수 있는게 낫다. GoF가 나설 차례이다.  

### 상태 패턴

객체지향에 푹 빠진 나머지 모든 분기문을 동적 디스패치 (C++에서는 가상 함수)로 바꾸려 하는 사람들이 있다. 그건 너무 과하다. 때로는 `if`문만으로도 충분한다.  

(여기에는 역사적인 근거가 있다. GoF나 '리펙토링' 저자인 마틴 파울러 같은 OOP 거장들은 스몰토크 개발자 출신이다. 스몰토크에서 `ifTrue:`는 `true` 객체와 `false` 객체에서 각기 다르게 구현한 메서드에 불과하다. (스몰토크에 다음과 같은 식으로 표현된다.)

```smalltalk
(x = y)
    ifTrue: [^x]
    ifFalse: [^y].
```

여기서 `(x = y)`는 결과에 따라 `True` 객체 또른 `False` 객체를 생성한 후 `ifTrue:`와 `ifFalse:` 메시지를 던진다. 예를 들어 조건이 참이라면 `true`의 `ifFalse:` 메서드는 `^nil`(`nil`을 반환)로 구현되어 있기 때문에 아무것도 하지 않고 `ifTrue:`에 정의된 블록만 실행한다. `False` 객체는 그 반대다.)  

하지만 `Heroine` 예제 정도라면 객체지향, 즉 상태 패턴을 쓰는 게 더 낫다. GoF가 설명한 상태 패턴을 `Heroin` 클래스에 적용해보면 다음과 같다.  

#### 상태 인터페이스

상태 인터페이스부터 정의하자. 상태에 의존하는 모든 코드, 즉 다중 선택문에 있던 동작을 인터페이스의 가상 메서드로 만든다. 예제에서는 `handleInput()`과 `update()`가 해당된다.  

```cpp
class HeroineState
{
public:
    virtual ~HeroineState() {}
    virtual void handleInput(Heroine& heroine, Input input) {}
    virtual void update(Heroine& heroine) {}
};
```

#### 상태별 클래스 만들기

상태별로 인터페이스를 구현하는 클래스도 정의한다. 메서드에는 정해진 상태가 되었을 때 주인공이 어떤 행동을 할지를 정의한다. 다중 선택문에 있는 `case`별로 클래스를 만들어 코드를 옮기면 된다.  

```cpp
class DuckingState : public HeroineState {
public:
    DuckingState() : chargeTime_(0) {}

    virtual void handleInput(Heroine& heroine, Input input) {
        if (input == RELEASE_DOWN) {
            // 일어선 상태로 바꾼다...
            heroine.setGraphics(IMAGE_STAND);
        }
    }

    virtual void update(Heroine& heroine) {
        chargeTime_++;
        if (chargeTime_ > MAX_CHARGE) {
            heroine.superBomb();
        }
    }

private:
    int chargeTime_;
};
```

`changeTime_` 변수를 `Heroine`에서 `DuckingState`로 옮겼다는 점을 놓치지 말자. `chargeTime_`은 엎드린 상태에서만 의미가 있다는 점을 객체 모델링을 통해서 분명하게 보여준다는 점에서 훨씬 개선되었다.

#### 동작을 상태에 위임하기

이번에는 `Heroine` 클래스 자신의 현재 상태 객체를 포인터로 추가해, 거대한 다중 선택문은 제거하고 대신 상태 객체에 위임한다.

```cpp
class `Heroine`
{
public:
    virtual void handleInput(Input input) {
        state_->handleInput(*this, input);
    }
    virtual void update() { state_->update(*this); }
    // 다른 메서드들...

private:
    HeroineState* state_;
};
```

'상태를 바꾸려면' `state_` 포인터에 `HeroineState`를 상속받는 다른 객체를 할당하기만 하면 된다. 이게 상태 패턴의 전부이다.  

(이런 점은 GoF의 전략 패턴, 타입 객체 패턴과도 비슷해 보인다. 셋 다 주요 클래스가 여러 하위 객체에 동작을 위임한다는 게 공통점이다. 차이점은 **의도**에 있다.  

- 전략 패턴은 주요 클래스를 일부 동작으로부터 **디커플링**하는 게 목표다.
- 타입 객체 패턴은 같은 타입 객체의 레퍼런스를 **공유**함으로써 **여러** 객체를 비슷하게 동작시키는 게 목표다.
- 상태 패턴은 동작을 위임하는 객체를 **변경**함으로써 주요 클래스의 동작을 **변경**하는 게 목표다.
)  

### 상태 객체는 어디에 둬야 할까?

앞에서 얼버무리고 넘어간 것이 있다. 상태를 바꾸려면 `state_`에 새롱누 상태 객체를 할당해야 한다. 그렇다면 이 객체는 어디에서 온 것일까? 열거형은 숫자처럼 기본 자료형이기 때문에 신경 쓸 게 없지만 상태 패턴은 클래스를 쓰기 때문에 포인터에 저장할 실제 인스턴스가 필요하다. 두 가지 방법을 알아보자.  

#### 정적 객체

상태 객체에 필드가 따로 없다면 가상 메서드 호출에 필요한 `virtual` 포인터만 있는 셈이다. 이럴 경우 모든 인스턴스가 같기 때문에 인스턴스는 하나만 있으면 된다.  

(상태 클래스에 필드가 없고 가상 메서드도 하나밖에 없다면 더욱 더 단순화해서 상태 클래스를 정적 함수로 바꿀 수도 있다. 이럴 경우 `state_` 필드는 함수 포인터가 된다.)  

이제 **정적** 인스턴스 하나만 만들면 된다. 여러 FSM이 동시에 돌더라도 상태 기계는 다 같으므로 인스턴스 하나를 같이 사용하면 된다. (이런게 경량 패턴이다.)  

정적 인스턴스를 **원하는** 곳에 두면 된다. 특별히 다른 곳이 없다면 상위 상태 클래스에 두자.

```cpp
class HeroineState
{
public:
    static StandingState standing;
    static DuckingState ducking;
    static JumpingState jumping;
    static DivingState diving;
    // 다른 코드들...
};
```

각각의 정적 변수가 게임에서 사용하는 상태 인스턴스다. 서 있는 상태에서 점프하게 하려면 이렇게 한다.

```cpp
if (input == PRESS_B) {
    heroine.state_ = &HeroineState::jumping;
    heroine.setGraphics(IMAGE_JUMP);
}
```

#### 상태 객체 만들기

정적 객체만으로 부족할 때도 있다. 엎드리기 상태에는 `chargeTime_` 필드가 있는데 이 값이 주인공마다 다르다 보니 정적 객체로 만들 수 없다. 주인공이 하나라면 어떻게든 되겠지만, 협동 플레이 기능을 추가해 두 주인공이 한 화면에 보여야 한다면 문제가 된다.  

이럴 때는 전이할 때마다 상태 객체를 만들어야 한다. 이러면 FSM이 상태별로 인스턴스를 갖게 된다. **새로** 상태를 할당했기 때문에 **이전** 상태를 해제해야 한다. 상태를 바꾸는 코드가 현제 상태 메서드에 있기 떄문에 삭제할 때 **this**를 스스로 지우지 않도록 주의해야 한다.

이를 위해 **handleInput()**에서 상태가 바뀔 때에만 새로운 상태를 반환하고, 밖에서는 반환값에 따라 예전 상태를 삭제하고 새로운 상태를 저장하도록 바꿔보자.

```cpp
void Heroine::handleInput(Input input) {
    HeroineState* state = state_->handleInput(*this, input);
    if (state != nullptr) {
        delete state_;
        state_ = nullptr;
    }
}
```

`handleInput` 메서드가 새로운 상태를 반환하지 않는다면 현재 상태를 삭제하지 않는다. 서 있기 상태에서 엎드리기 상태로 전이하려면 새로운 인스턴스를 생성해 반환한다.  

```cpp
HeroineState* StandingState::handleInput(
    Heroine& heroine, Input input) {
    if (input == PRESS_DOWN) {
        // 다른 코드들...
        return new DuckingState();
    }
    // 지금 상태를 유지한다.
    return NULL;
}
```

이렇게 하는 경우 매번 객체 할당을 위해 메모리와 CPU를 낭비하지 않아도 되는 정적 상태를 쓰려고 하는 편이다. (상태를 동적으로 할당하면서 생길 수 있는 메모리 단편화가 부담스럽다면 객체 풀 패턴을 고려해보자.) 지금부터는 상태 패턴을 좀 더, 음, 말하자면 **상태스럽게** 만들 방법을 살펴본다.  

### 입장과 퇴장

상태 패턴의 목표는 같은 상태에 대한 모든 동작과 데이터를 클래스 하나에 캡슐화하는 것이다. 이런 면에서는 우리의 예제 코드는 아직 부족한 면이 있다.  

주인공은 상태를 변경하면서 주인공의 스프라이트도 같이 바꾼다. 지금까지는 이전 상태에서 스프라이트를 변경했다. 예를 들어 엎드리기에서 서기로 넘어갈 때에는 엎드리기 상태에서 주인공 이미지를 변경했다.  

```cpp
HeroineState* DuckingState::handleInput(
    Heroine& heroine, Input input) {
    if (input == RELEASE_DOWN) {
        heroine.setGraphics(IMAGE_STAND);
        return new StandingState();
    }
    // 다른 코드들...
}
```

이렇게 하는 것보다 상태에서 그래픽까지 제어하는게 더 바람직하다. 이를 위해 **입장 기능**을 추가하자.  

```cpp
class StandingState : public HeroineState
{
public:
    virtual void enter(Heroine& heroine) {
        heroine.setGraphics(IMAGE_STAND);
    }
    // 다른 코드들...
};
```

`Heroine` 클래스에서는 새로운 상태에 들어 있는 `enter` 함수를 호출하도록 상태 변경 코드를 수정한다.  

```cpp
void Heroine::handleInput(Input input) {
    HeroineState* state = state_->handleInput(*this, input);
    if (nullptr != state) {
        delete state_;
        state_ = state;

        // 새로운 상태의 입장 함수를 호출한다.
        state_->enter(*this);
    }
}
```

이제 엎드리기 코드를 더 단순하게 만들 수 있다.  

```cpp
HeroineState* DuckingState::handleInput(
    Heroine& heroine, Input input) {
    if (input == RELEASE_DOWN) {
        return new StandingState();
    }
    // 다른 코드들...
}
```

`Heroine` 클래스에서는 서기 상태로 변경하기만 하면 서기 상태가 알아서 그래픽까지 챙긴다. 이래야 상태가 제대로 캡슐화되었다고 할 수 있다. 그 전 상태와는 상관없이 항상 같은 입장 코드가 실행된다는 것도 장점이다.  

실제 게임 상태 그래프라면 점프 후 착지 혹은 내려찍기 후 착지하는 식으로 같은 상태에 여러 전이가 들어올 수 있다. 그냥 두면 전이가 일어나는 모든 곳에 중복 코드를 넣었겠지만 이제는 입장 기능 한곳에 코드를 모아두면 된다.  

상태가 새로운 상태로 **교체**되기 직전에 호출되는 **퇴장 코드**도 이런 식으로 활용할 수 있다.  

### 단점은?

FSM의 장점만 얘기했지만, 단점이 없을 리 없다. 이제껏 한 얘기는 다 사실이고, FSM으로 해결할 수 있는 문제도 많다. 하지만 FSM의 장점은 동시에 단점이기도 하다.  

상태 기계는 엄격하게 제한된 구조를 강제함으로써 복잡하게 얽힌 코드를 정리할 수 있게 새준다. FSM은 미리 정해놓은 여러 상태와 현재 상태 하나, 그리고 하드코딩되어 있는 전이만이 존재한다.

(FSM은 **튜링 완전 (Turing complete**)하지조차 않다. 오토마타 이론은 일련의 추상 모델을 이용해 이보다 더 복잡한 문제를 계산하는 방법을 다룬다. 튜링 기계는 이들 모델 중에서도 표현력이 풍부한 모델에 속한다.  
'튜링 완전'하다는 뜻은 (보통은 프로그래밍 언어인) 시스템이 **튜링 기계**를 구현할 수 있을 정도로 충분히 강력하다는 의미다. 즉, 모든 튜링 완전 언어는 어떤 의미에서는 표현력이 동일하다. FSM은 그 정도까지는 아니다.)  

상태 기계를 인공지능같이 더 복잡한 곳에 적용하다 보면 한계에 부딪히게 된다. 다행히 이전 세대 개발자들이 한계를 빠져나갈 방법을 먼저 찾아놨다. 이 중 몇가지 방법을 살펴보면서 이 장을 마무리하자.  

### 병행 상태 기계

주인공이 총을 들 수 있게 만든다고 해보자. 총을 장착한 후에도 이전에 할 수 있었던 모든 동작을 할 수 있어야 한다. 그러면서 동시에 총도 쏠 수 있어야 한다.  

FSM 방식을 고수하겠다면 모든 상태를 서기, 무장한 채로 서기, 점프, 무장한 채로 점프 같은 식으로 **두 개씩** 만들어야 한다.  

무기가 추가되면 상태 조합이 폭발적으로 늘어난다. 상태가 많아지는 것도 문제지만, 무장 상태와 비무장 상태는 총 쏘기 코드 약간 외에는 거의 같아서 중복이 많아진다는 점이 더 큰 문제다.  

두 종류의 상태, 즉 무엇을 **하는가**와 무엇을 **들고 있는가**를 한 상태 기계에 욱여넣다 보니 생긴 문제다. 모든 가능한 조합에 대해 모델링하려다 보니 모든 **쌍**에 대해 상태를 만들어야 한다. 해결법은 간단하다. 상태 기계를 둘로 나누면 된다.  

(무엇을 하는가에 대한 상태 n개와 무엇을 들고 있는가에 대한 상태 m을 한 상태 기계에 욱여넣으면 n * m개 상태가 필요하다. 상태 기계를 두 개로 만들면 n + m개 상태만 있으면 된다.)  

무엇을 하는가에 대한 상태 기계는 그대로 두고, 무엇을 들고 있는가에 대한 상태 기계를 따로 정의한다. `Heroine` 클래스는 이들 '상태'를 **각각** 참조한다.  

(예시를 위해 무기 장착에 대해서도 제대로 된 상태 기계를 사용했다. 사실 무기 장착은 '장착했다', '장착 안 했다' 두 가지 상태밖에 없기 때문에 불리언 플래그만으로도 충분하다.)  

```cpp
class Heroine
{
    // 다른 코드들...

private:
    HeroineState* state_;
    HeroineState* equipment_;
};
```

`Heroine`에서 입력을 상태에 위임할 때에는 입력을 상태 기계 양쪽에 다 전달한다.  

(시스템을 더 정교하게 만든다면 필요에 따라 첫 번째 상태 기계에서 입력을 **씹어서** (consume) 다음 상태 기계까지 입력이 가지 않도록 할 수 있다. 이러면 똑같은 입력을 두 기계가 같이 처리하면서 잘못 반응하는 위험을 방지할 수 있다.)  

```cpp
void Heroine::handleInput(Input input) {
    state_->handleInput(*this, input);
    equipment_->handleInput(*this, input);
}
```

각각의 상태 기계는 입력에 따라 동작을 실행하고 독립적으로 상태를 변경할 수 있다. 두 상태 기계가 서로 전혀 연관이 없다면 이 방법이 잘 들어 맞는다.  

현실적으로 점프 도중에 총을 못 쏜다든가, 무장한 상태에서는 내려찍기를 못한다든가 하는 식으로 복수의 상태 기계가 상호작용해야 할 수도 있다. 이를 위해 어떤 상태 코드에서 **다른** 상태 기계의 상태가 무엇인지를 검사하는 지저분한 코드를 만들 일이 생길 수도 있다. 거림직하지만 문제를 해결할 수는 있을 것이다.  

### 계층형 상태 기계

주인공 동작에 살을 덧붙이다 보면 서기, 걷기, 달리기 미끄러지기 같이 비슷한 상태가 많이 생기기 마련이다. 이들 상태에선 모두 B 버튼을 누르면 점프하고, 아래 버튼을 누르면 엎드려야 한다.  

단순한 상태 기계 구현에서는 이런 코드를 모든 상태마다 중복해 넣어야 한다. 그보다는 한 번만 구현하고 다른 상태에서 재사용하는 게 낫다.  

상태 기계가 아니라 객체지향 코드라고 생각해보면, 상속으로 여러 상태가 코드를 공유할 수 있다. 점프와 엎드리기는 '땅 위에 있는' 상태 클래스를 정의해 처리한다. 서기, 걷기, 달리기, 미끄러지기는 '땅 위에 있는' 상태 클래스를 상속받아 고유 동작을 추가하면 된다.  

(상속은 강력한 코드 재사용 방법이지만 두 코드가 강하게 커플링된다는 단점도 있다. 상속은 거대한 망치이니 조심해서 휘두르자.)  

이런 구조를 **계층형 상태 기계**라고 한다. 어떤 상태를 **상위 상태** (superstate)를 가질 수 있고, 그 경우 그 상태 자신은 **하위 상태** (substate)가 된다. 이벤트가 들어올 때 하위 상태에서 처리하지 않으면 tkd위 상태로 넘어간다. 말하자면 상속받은 메서드를 오버라이드하는 것과 같ek.

예제 FSM을 상태 패턴으로 만든다면 클래스 상속으로 계층을 구현할 수 있다. 상위 상태용 클래스를 하나 정의하자.  

```cpp
class OnGroundState : public HeroineState {
public:
    virtual void handleInput(
        Heroine& heroine, Input input) {
        if (input == PRESS_B) { // 점프...
        } else if (input == PRESS_DOWN) { // 엎드리기...
        }
    }
};
```

그 다음 각각의 하위 상태가 상위 상태를 상속 받는다.  

```cpp
class DuckingState : public OnGroundState
{
public:
    virtual void handleInput(
        Heroine& heroine, Input input) {
        if (input == PRESS_DOWN) {
            // 서기...
        } else {
            // 따로 입력을 처리하지 않고, 상위 상태로 보낸다.
            OnGroundState::handleInput(heroine, input);
        }
    }
};
```

계층형을 꼭 이렇게 구현해야 하는 건 아니다. 클래스를 사용하는 GoF식 상태 패턴을 쓰지 않는다면 이런 구현이 불가능할 수 있다. 그럴 땐 주 클래스에 상태를 하나만 두지 않고 상태 **스택**을 만들어 명식적으로 현재 상태의 상위 상태 연쇄를 모델링할 수도 있다.  

현재 상태가 스택 최상위에 있고 밑에는 바로 위 상위 상태가 있으며, 그 상위 상태 밑에는 **그** 상위 상태의 상위 상태가 있는 식이다. 상태 관련 동작이 들어오면 어느 상태든 동작을 처리할 때까지 스택 위에서부터 미틍로 전달한다. (아무도 처리하지 않는다면 무시하면 된다.)  

### 푸시다운 오토마타

상태 스택을 활용하여 FSM을 확장하는 다른 방법도 있다. 계층형 FSM에서 봤던 스택과는 상태를 담는 방식도 다르고 해결하려는 문제도 다르다.  

FSM에는 **이력**(History) 개념이 없다는 문제가 있다. **현재** 상태는 알 수 없지만 **직전** 상태가 무엇인지를 따로 저장하지 않기 때문에 이전 상태로 쉽게 돌아갈 수 없다.  

옐를 들어보자. 앞에서 우리는 용감한 주인공이 완전무장할 수 있게 했다. 주인공이 총을 쏘면 발사 애니메이션 재생과 함께 총알과 시각 이펙트를 생성하는 새로운 상태가 필요하다. 총을 쏠 수 있는 모든 상태에서 발사 버튼을 눌렀을 때 전이할 `FiringState`라는 상태를 대충 만들어 보자.  

(`FiringState`로 옮겨지는 동작도 여러 상태에서 중복되기 때문에 계층형 상태 기계를 활용해 코드를 재사용할 수도 있다.)  

이때 어려운 부분은 총을 쏜 **뒤에** 어는 상태로 돌아가야 하는가 하는 점이다. 서기, 달리기, 점프, 엎드리기 상태에서 총을 쏠 수 있는데 총 쏘는 동작이 끝난 후에는 다시 이전 상태로 돌아가야 한다.  

일반적인 FSM에서는 이전 상태를 알 수 없다. 이전 상태를 알려면 상태마다 서 있는 상태에서 총쏘기, 달려가면서 총 쏘기, 점프하며넛 총 쏘기 같은 식으로 상태마다 새로운 상태를 하나씩 더 만들어 총 쏘기가 끝났을때 되돌아갈 상태를 하드코딩해야 한다.  

이것보다는 총 쏘기 전 상태를 **저장해놨다가** 나중에 불러와 써먹는 게 훨씬 낫다. 다시 오토마타 이론이 도움을 줄 차례다. 이럴 때 써먹을 만한 것으로 **푸시다운 오토마타**가 있다.  

FSM이 **한 개**의 상태를 포인터로 관리했다면 푸시다운 오토마타에서는 상태를 **스택**으로 관리한다. FSM은 이전 상태를 **덮어쓰고** 새로운 상태로 전이하는 방식이었다. 푸시다운 오토마타에서는 이외에도 부가적인 명령이 두 가지 있다.  

- 새로운 상태를 스택에 **넣는다.** (push) 스택의 최상위 상태가 '현재' 상태이기 때문에, 새로 추가된 상태가 현재 상태가 된다. 다만, 이전 상태는 버리지 않고 최신 상태 밑에 있게 된다.
- 최상위 상태를 스택에서 **뺀다.** (pop) 빠진 상태는 제거되고, 바로 밑에 있던 상태가 새롭게 '현재' 상태가 된다.

이것은 총 쏘기 살태를 구현할 때 딱 좋다. 먼저 총 쏘기 상태를 하나 만든다. 어떤 상태에서든지 간에 발사 버튼을 누르면 총 쏘기 상태를 스택에 넣는다. 총 쏘기 애니메이션이 끝날 때 총 쏘기 상태를 스택에서 빼면, 푸시다운 오토마타가 알아서 이전 상태로 보내준다.  

### 얼마나 유요안가?

FSM에는 몇 가지 확장판이 나와 있지만 FSM만으로는 한계가 있다. 요즘 게임 AI는 **행동 트리 (Behaviour Tree)**나 **계획 시스템 (Planning System)**을 더 많이 쓰는 추세다. 복잡한 AI에 관심이 있다면 이번 장은 맛보기 정도로 생각하자. 제대로 하려면 다른 책을 더 읽어볼 필요가 있다.  

FSM이나 푸시다운 오토마다, 그 외 간단한 시스템들이 쓸모없다는 얘기는 아니다. 이것만으로도 특정 문제 해결을 위한 모델링으로선 충분하다. FSM은 다음 경우에 사용하면 좋다.  

- 내부 상태에 따라 객체 동작이 바뀔 때
- 이런 상태가 그다지 많지 않은 선택지로 분명하게 구분될 수 있을 때
- 객체가 입력이나 이벤트에 따라 반응할때

게임에서는 FSM이 AI에서 사용되는 걸로 가장 잘 알려져 있지만, 입력 처리나 메뉴 화면 전환, 문자 해석, 네트워크 프로토콜, 비동기 동작을 구현하는 데에도 많이 사용되고 있다.  

## 메모

---

- **관련 자료:**
- **키워드:**
  - [Design Pattern (디자인 패턴)](/posts/design-pattern/)
  - [State Pattern (상태 패턴)](/posts/state-pattern/)
  - FSM, Finite State Machine (유한 상태 기계)
  - Hierarchical State Machine (계층형 상태 기계)
  - Pushdown Automata (푸시다운 오토마타)
  - Update Method Pattern (업데이트 메서드 패턴)
