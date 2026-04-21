---
title: "커맨드 패턴"
# description: ""
categories: [컴퓨터, 프로그래밍]
tags: [Design-Pattern]
image: "/assets/img/background/kururu-lab.jpg"

date: 2025-10-04. 14:54 # S Init
# last_modified_at: 2025-10-06. 03:35 # E 초고
last_modified_at: 2025-10-08. 14:19 # E ~정리
---

## 머리말

---

- **참고:**
  - ['게임 프로그래밍 패턴: 더 빠르고 깔끔한 게임 코드를 구현하는 13가지 디자인 패턴' - 로버트 나이스트롬](https://gameprogrammingpatterns.com/command.html)
- **핵심:**
  - **명령 교체:**
    - 메서드 호출을 **실체화**한 것이다.
    - 콜백을 객체지향적으로 표현한 것이다.
  - **이점:**
    - 코드 유연성/확장성을 극대화.
    - Undo/Redo, 매크로, 리플레이 같은 복잡한 기능을 우아하고 체계적으로 구현할 수 있게 해줌.

## 실체화

---

- 실제하는 것으로 만든다. (rei-fy, thing-ify)
- 어떤 개념을 변수에 저장하거나 함수에 전달할 수 있도록 **데이터**. 즉 객체로 바꿀 수 있다는 것을 의미한다.
  - 일급으로 만든다. (first-class)
    - like 콜백, 일급 함수, 함수 포인터, 클로저, 부분 적용 함수

## 예제-입력키 변경

---

```cpp
// #1
void InputHandler::handleInput() {
    if (isPressed(BUTTON_X)) jump();
    else if (isPressed(BUTTON_Y)) fireGun();
    else if (isPressed(BUTTON_A)) swapWeapon();
    else if (isPressed(BUTTON_B)) lurchIneffectively();
}
```

일반적으로, 많은 게임에서 키를 **바꿀** 수 있는 기능을 제공한다.  
키 변경을 지원하려면, 함수를 직접 호출하지 않고 **교체** 가능한 무언가로 바꿔야 한다.  

**교체**. 어떤 게임 행동을 나타내는 **객체**를 만들어서 이를 변수에 할당해야 할 것 같다.  

```cpp
// #2 공통 상위 클래스 `Command` 정의, 각 행동 별로 하위 클래스 구현
class Command {
    public:
        virtual ~Command() {}
        virtual void execute() = 0; // = 0; -> C# abstract
};

class JumpCommand : public Command {
    public:
        virtual void execute() { jump(); }
};
// ...
```

공통 상위 클래스를 정의하고, 각 행동 별로 하위 클래스를 만든다.  

```cpp
// #3
class InputHandler  {
    public:
        void handleInput():
        // 명령을 바인드 할 메서드들...
    private:
        Command* buttonX_;
        Command* buttonY_;
        Command* buttonA_;
        Command* buttonB_;
};

void InputHandler::handleInput() {
    if (isPressed(BUTTON_X)) buttonX_->execute();
    else if (isPressed(BUTTON_Y)) buttonY_->execute();
    else if (isPressed(BUTTON_A)) buttonA_->execute();
    else if (isPressed(BUTTON_B)) buttonB_->execute();
}
```

직접 함수를 호출하는 대신, 한 겹 우회하는 계층이 생겼다.  

### 액터에게 지시하기

현재 `JumpCommand` 클래스는 오직 플레이어 캐릭터만 점프하게 만들 수 있다.  

`JumpCommand`의 `jump()` 같은 전역 함수가 캐릭터 객체를 암시적으로 찾아서 사용.  
커플링이 가정에 깔려있다보니, `Command` 클래스의 유용성이 떨어지고 제한적이다.  

```cpp
// #4
class Command {
    public:
        virtual ~Command() {}
        virtual void execute(GameActor& actor) = 0;
};

class JumpCommand : public Command {
    public:
        virtual void execute(GameActor& actor) { actor.jump(); }
};
```

이런 제약을 유연하게 만들기 위해 제어하려는 객체를 함수에서 직접 찾게 하지 말고 밖에서 전달해주자.  
위처럼 `execute()`에서 액터를 받도록하면, 어떤 캐릭터든 폴짝거리게 할 수 있다.  

남은 것은 입력 핸들러에서 입력을 받아 적당한 객체의 메서드를 호출하는 명령 객체를 연결하는 것.  
`handleInput()`에서 명령 객체를 **반환**하도록 변경.  

```cpp
// #5
Command* InputHandler::handleInput() {
    if (isPressed(BUTTON_X)) return buttonX_;
    if (isPressed(BUTTON_Y)) return buttonY_;
    if (isPressed(BUTTON_A)) return buttonA_;
    if (isPressed(BUTTON_B)) return buttonB_;

    // 아무것도 누르지 않았다면, 아무것도 하지 않는다
    return NULL;
}
```

어떤 액터를 매개변수로 넘겨줘야 할지 모르기 때문에 `handleInput()`에서는 명령을 실행할 수 없다. (책임의 분리)  
여기에서는 명령이 '실체화된 함수 호출'이라는 점을 활용해서, 함수 호출 시점을 **지연**한다.  

```cpp
// #6
Command* command = inputHandler.handleInput();
if (command) {
    command->execute(actor);
}
```

플레이어 기능은 변함없지만, 명령과 액터 사이에 추상 계층을 한 단계 더 둔 덕분에,  
**명령을 실행할 때 액터만 바꾸면 (플레이어가) 게임에 있는 어떤 액터라도 제어할 수 있게 되었다**  

액터를 제어하는 `Command`를 일급 객체로 만든 덕분에, 메서드를 직접 호출하는 형태의 강한 커플링을 제거할 수 있었다.  

활용: AI 엔진과 액터 사이의 인터페이스용으로 사용할 수 있다. 즉, AI 코드에서 원하는 `Command` 객체를 이용하는 식이다.  
AI 엔진 (or 플레이어) -> AI 행동 (or 입력) -> 액터 -> `Command`  

### 명령을 큐나 스트림으로 만들기

명령을 큐나 스트림에 넣는다.  
큐나 스트림을 생산자와 소비자 사이에 끼워 넣음으로써, **서로를 디커플링** 할 수 있다.  
생산자 -> 큐/스트림 -> 소비자  

생산자는 소비자를 전혀 몰라도 된다. '명령을 큐에 넣는다'는 책임만 다하면 끝이다. 명령을 누가 언제 어떻게 실행할지는 신경쓰지 않아도 된다.  
소비자는 생산자를 전혀 몰라도 된다. '큐에 명령이 있으면 하나 꺼내서 실행한다'는 책임만 다하면 끝이다. 명령이 어떻게 만들어졌는지는 신경쓰지 않아도 된다.  

입력 핸들러나 AI 같은 코드에서는 명령 객체를 만들어 스트림에 밀어 넣는다.  
(명령들을 직렬화할 수 있다면, 네트워크로도 전달할 수 있다. 즉 플레이어로부터 입력을 받아 네트워크를 통해 상대방에게 전달해 재현할 수 있다. 네트워크 플레이에서 중요)

디스패터나 액터에서는 명령 객체를 받아서 호출한다.  

## 예제-실행취소와 재실행

명령 패턴 사용 예 중에서도 가장 잘 알려진 것.  
명령 객체가 어떤 작업을 실행할 수 있다면, 이를 실행취소 할 수 있게 만들수도 있다.  

게임 개발 툴에는 필수. (Like 레벨 에디터)  

```cpp
class MoveUnitCommand : public Command {
    public:
        MoveUnitCommand(Unit* unit, int x, int y)
        : unit_(unit),
          x_(x),
          y_(y) {
        }

    virtual void execute() {
        unit_->moveTo(x_, y_);
    }

    private:
        Unit* unit_;
        int x_;
        int y_;
};
```

이전 예제와는 다르다.  

이전 예제에서는 명령에서 변경하려는 액터와 명령 사이를 **추상화**로 격리시켰다.  
**어떤 일을 하는지**를 정의한 명령 객체 하나가 매번 재사용된다.  
입력 핸들러 코드에서는 특정 버튼이 눌릴 때마다 여기에 연결된 명령 객체의 `execute()`를 호출했었다.  

이번 예제는 이동하려는 유닛과 위치 값을 생성자에서 받아서 명령과 명시적으로 **바인드**했다.  
`MoveUnitCommand` 명령 인스턴스는 '무엇인가를 움직이는' (언제든지 써먹을 수 있는) 보편적인 작업이 아니라, 게임에서의 구체적인 실제 이동을 담고 있다.  
이 명령 클래스는 특정 시점이 발생될 일을 표현한다는 점에서 좀 더 구체적이다.  
이를테면, 입력 핸들러 코드는 플레이어가 이동을 선택할 때마다 명령 인스턴스를 생성해야 한다.  
(C++ 같이 GC 없는 언매니지드 언어에서는 `Command` 객체를 실행하는 코드가 메모리 해제까지 챙겨야 한다.)  

```cpp
Command* handleInput() {
    Unit* unit = getSelectedUnit();
    if (isPressed(BUTTON_UP)) {
        // 유닛을 한 칸 위로 이동한다.
        int destY = unit->y() - 1;
        return new MoveUnitCommand(unit, unit->x(), destY);
    }
    // 다른 이동들...
    return NULL;
}
```

`Command` 클래스가 일회용이라는 게 장점이다.  
(각 명령이 독립적인 상태를 가지는 별개의 객체로 생성되기 때문에, 실행 취소에 필요한 과거 상태를 저장하고 관리하기에 매우 적합하다.)  

명령을 취소할 수 있도록 순수 가상 함수 `undo()`를 정의한다.  

```cpp
class Command {
    public:
        virtual ~Command() {}
        virtual void execute() = 0;
        virtual void undo() = 0;
}

class MoveUnitCommand : public Command {
    public:
        MoveUnitCommand(Unit* unit, int x, int y)
        : unit_(unit), x_(x), y_(y),
          xBefore_(0), yBefore_(0) {
        }

    virtual void execute() {
        // 나중에 이동을 취소할 수 있도록 원래 유닛 위치를 저장한다.
        xBefore_ = unit_->x();
        yBefore_ = unit_->y();
        unit_->moveTo(x_, y_);
    }

    virtual void undo() {
        unit_->moveTo(xBefore_, yBefore_);
    }

    private:
        Unit* unit_;
        int x_;
        int y_;
};
```

상태 몇 개가 추가되었다. (`xBefore_`, `yBefore_`)  

여러 단계의 실행취소를 구현하는 것도 그다지 어렵지 않다. 가장 최근 명령만 기억하는 대신, 명령 리스트를 유지하고 '현재' 명령이 무엇인지만 알고 있으면 된다. (current = 현재 명령 인덱스)  

### 클래스 대신 함수형

C++은 일급 함수를 제대로 지원하지 않는다. 함수 포인터에는 상태를 저장할 수 없고, 펑터는 이상한 데다가(?) 여전히 클래스를 정의해야 한다. C++11에 도입된 람다는 메모리를 직접 관리해야 하기 때문에 쓰기가 까다롭다.  

언어에서 클로저를 제대로 지원해준다면 안 쓸 이유가 없다! 어떻게 보면 명령 패턴은 클로저를 지원하지 않는 언어에서 클로저를 흉내 내는 방법 중 하나일 뿐이다. (어떻게 보면이라고 쓴 이유는 클로저를 지원하는 언어에서조차 명령 패턴을 구현하기 위해 클래스나 구조체를 사용하는 게 좋을 때도 있기 때문이다. 명령에 실행취소 가능한 명령같이 여러 기능이 함꼐 들어 있을 때에는 함수 하나로 치환하기가 쉽지 않다. 멤버 변수가 들어 있는 클래스로 정의하면 코드를 읽을 때 명령에 어떤 데이터가 들어 있는지 알아보기가 쉽다. 클로저는 어떤 상태를 자동으로 래핑하는 굉장히 간단한 방법을 제공하지만, 너무 자동으로 해주다 보니 클로저가 어떤 상태를 들고 있는지를 알아보기가 어렵다.)  

```js
function makeMoveUnitCommand(unit, x, y) {
    // 아래 function이 명령 객체에 해당한다:
    return function() {
        unit.moveTo(x, y);
    }
}

// 클로저를 여러 개 이용하는 예제
function makeMoveUnitCommand(unit, x, y) {
    var xBefore, yBefore;
    return {
        execute: function() {
            xBefore = unit.x();
            yBefore = unit.y();
            unit.moveTo(x, y);
        },
        undo: function() {
            unit.moveTo(xBefore, yBefore);
        }
    };
}
```

명령 패턴의 유용성은 함수형 패러다임이 얼마나 많은 문제에 효과적인지를 보여주는 예이기도 하다.  

## 메모

---

- **관련 자료:**
  - 명령 패턴을 쓰다 보면 수많은 `Command` 클래스를 만들어야 할 수 있다. 이럴 때에는 구체 상위 클래스에 여러 가지 편의를 제공하는 상위 레벨 메서드를 만들어놓은 뒤에 필요하면 하위 클래스에서 원하는 작동을 재정의할 수 있게 하면 좋다. 이러면 명령 클래스의 `execute` 메서드가 하위 클래스 샌드박스 패턴으로 발전하게 된다.
  - 예제에서는 어떤 엑터가 명령을 처리할지를 명시적으로 지정했다. 하지만 계층 구조 객체 모델에서처럼 누가 명령을 처리할지가 그다지 명시적이지 않을 수도 있다. 객체가 명령에 반응할 수도 있고 종속 객체에 명령 처리를 떠넘길 수도 있다면 GoF의 책임 연쇄-chain of responsibility 패턴이라고도 볼 수 있다.
  - 어떤 명령은 처음 예제에 등장한 `JumpCommand` 클래스처럼 상태 없이 순수하게 행위만 정의되어 있을 수 있다. 이런 클래스는 모든 인스터스가 같기 때문에 인스턴스를 여러 개 만들어봐야 메모리만 낭비된다. 이 문제는 경량 패턴으로 해결할 수 있다. (싱글턴으로 만들어도 가능하지만, 권하지는 않는다.)
- **키워드:**
  - [Design Pattern (디자인 패턴)](/posts/design-pattern/)
  - [Command Pattern (커맨드 패턴, 명령 패턴)](/posts/command-pattern/)
  - [Callback (콜백)](/posts/callback/)
  - First Class (일급 객체)
  - Producer, Consumer (생산자, 소비자)
  - Dispatcher (디스패처)
  - Event Queue Pattern (이벤트 큐 패턴)
  - Memento Pattern (메멘토 패턴)
    - [예제-실행취소와 재실행](#예제-실행취소와-재실행)
      - **메멘토 패턴**이란 것도 있지만, 제대로 활용하는 걸 본 적이 없다. 메멘토 패턴처럼 객체 상태 전부를 저장하는 것보다는 명령 패턴처럼 객체 상태 중에서 변경한 데이터만 따로 저장하는 것이 메모리 효율 면에서 훨씬 낫다.
  - President Data Structure (지속 자료구조)
    - [예제-실행취소와 재실행](#예제-실행취소와-재실행)
      - **지속 자료구조**도 써볼만 하다. 이 자료구조에서는 어떤 객체를 변경하면 원래 객체는 그대로 두고 새로운 객체를 반환한다. 새로 만들어진 객체가 이전 객체와 데이터를 공유하도록 잘 구현하면 완전히 새로운 객체를 복제하는 것보다는 메모리를 훨씬 적게 쓸 수 있다.
      - 지속 자료구조에서는 명령 객체마다 명령을 실행하기 전 객체를 참조하고 있다가, 실행취소할 떄 객체를 예전 객체로 되돌려주기만 하면 된다.
  - Closer (클로저)
  - Functor (펑터)
  - Concrete Base Class (구체 상위 클래스)
  - Sandbox Pattern (샌드박스 패턴)
  - [Flyweight Pattern (경량 패턴)](/posts/flyweight-pattern/)
  - Singleton Pattern (싱글턴 패턴)
