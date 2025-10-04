---
title: "커맨드 패턴"
# description: ""
categories: [컴퓨터, 프로그래밍]
tags: [Design-Pattern]
image: "/assets/img/background/kururu-lab.jpg"
hidden: true

date: 2025-10-04. 14:54 # S Init
# last_modified_at: 2025-10-04. 14:54 # E
---

## 말머리

---

- 커맨드 패턴:
  - 메서드 호출을 실체화 한 것이다.
  - 콜백을 객체지향적으로 표현한 것.
- 키워드:
  - 명령, 교체

## 실체화

---

- 실제하는 것으로 만든다. (rei-fy, thing-ify)
- in 프로그래밍, 일급으로 만든다. (first-class)
  - like 콜백, 일급 함수, 함수 포인터, 클로저, 부분 적용 함수
- 어떤 개념을 변수에 저장하거나 함수에 전달할 수 있도록 **데이터**. 즉 객체로 바꿀 수 있다는 것을 의미한다.

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

일반적으로, 이런 함수는 게임 루프에서 매 프레임 호출되고, 많은 게임이 키를 **바꿀** 수 있게 해준다.  
키 변경을 지원하려면, 함수를 직접 호출하지 않고 **교체** 가능한 무언가로 바꿔야 한다.  

**교체**, 어떤 게임 행동을 나타내는 **객체**가 있어서 이를 변수에 할당해야 할 것 같다.  

```cpp
// #2
class Command {
    public:
        virtual ~Command() {}
        virtual void execute() = 0;
};

class JumpCommand : public Command {
    public:
        virtual void execute() { jump(); }
};
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

한계: `jump()`, `fireGun()` 같은 전역 함수가 캐릭터 객체를 암시적으로 찾아서 사용 -> 상당히 제한적.  
커플링이 가정에 깔려있다보니, Command 클래스의 유용성이 떨어진다.  
현재 `JumpCommand` 클래스는 오직 플레이어 캐릭터만 점프하게 만들 수 있다.  

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

어떤 캐릭터든 폴짝거리게 할 수 있다.  

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

어떤 액터를 매개변수로 넘겨줘야 할지 모르기 때문에 `handleInput()`에서는 명령을 실행할 수 없다.  
여기에서는 명령이 '실체화된 함수 호출'이라는 점을 활용해서, 함수 호출 시점을 **지연**한다.  

```cpp
// #6
Command* command = inputHandler.handleInput();
if (command) {
    command->execute(actor);
}
```

플레이어 기능은 변함없지만,  
명령과 액터 사이에 추상 계층을 한 단계 더 둔 덕분에,  
**명령을 실행할 때 액터만 바꾸면 (플레이어가) 게임에 있는 어떤 액터라도 제어할 수 있게 되었다**

AI 엔진과 액터 사이의 인터페이스용으로 사용할 수 있다.  
즉, AI 코드에서 원하는 Command 객체를 이용하는 식이다.  

액터를 제어하는 Command를 일급 객체로 만든 덕분에, 메서드를 직업 호출하는 형태의 강한 커플링을 제거할 수 있었다.  

### 명령을 큐나 스트림으로 만들기  

## 메모

---

- WM: Effect, Criteria
