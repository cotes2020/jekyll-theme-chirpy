---
layout: post
title: PeachTri 두번째 세미나 - 클린코드
date: 2022-06-03 11:43:00 +0900
description: PeachTri 두번째 세미나 # Add post description (optional)
published: true
img : Seminar_banner_small.png
tags: [세미나, PeachTri]
---
# 클린코드 🧑‍💻
### 깨끗한 코드란?
깨끗한 코드가 무엇이냐는 질문에 내가 생각하는 가장 좋은 대답은, **누구나 이해하기 쉬운 코드**라고 생각한다. 읽는 사람이 중학교 수준이더라도, 언어의 이해나 for같은 어법을 전부 덜어내고서라도 흐름을 설명하고 각 메소드나 객체가 어떤 역할인지 말했을 떄 쉽게 이해하는 정말 깨끗한 코드라고 생각한다.
## 깨끗한 코드가 되기 위한 조건 💁
- 이해하기 쉬운 이름
- 단순한 함수
- 명령과 조회의 분리
- 오류보다는 예외
- Test-Driven
- 단순한 클래스
- 높은 클래스 응집도, 낮은 결합도
- 변경하기 쉬운 클래스
- 디미터 법칙

다음과 같은 조건들을 생각하면, 코드를 작성하는데 있어서 초기에는 시간이 더 들지만, 우리가 목표하고자 하는 **살아있는 프로그램**을 위해서는 초기 투자라고 생각할 수 있다.
___

### 이해하기 쉬운 이름
우리가 만드는 수많은 클래스, 함수, 변수들의 이름들은 무심하게 짓기 십상이다. 하지만, 이러한 네이밍은 나중에 유지보수에서 엄청나게 애를 먹을 뿐만 아니라. 협업에 있어서 문제를 일으키는 원인이라는 것을 알고 있어야 한다.
- 의도가 분명한 이름
- 잘못된 정보가 없는 이름
- 검색하기 쉬운 단어
- 상수 사용
- 클래스 이름은 명사, 메서드 이름은 동사

~~~ dart
int num = 10;

int totalClientNumber = 10;
~~~
다음과 같은 네이밍은 이 num이라는 변수가 어떤 곳에 쓰이고 어떤식으로 코드내에서 역할을 하는지 이름으로는 알 수 없게 된다. 아래와 같은 경우는 총 클라이언트의 숫자를 뜻하는 변수임을 알 수 있는 상대적으로 직관적인 네이밍이라고 할 수 있다.

상수의 사용에 있어서도 비슷하게 적용 가능하다.
~~~dart
for (int i = 0; i < 10 ; i++){
    //code
}
~~~
만약 다음과 같은 loop가 존재할 때, 10이라는 숫자는 어떤것으로 확정되어 있는지, 왜 10번 돌아야 하는지 알 수 없기 때문에 이런 경우 **constant(상수)** 를 사용하여서, 값 자체가 이름을 가지도록 하는 것이 좋다.

또한 이름에 전체적으로 검색하기 쉽고, 함수에서 동작이 같다면 같은 이름으로 적용해주는게 좋다.
~~~dart
int getStudentGrade(){
    //code
}

String takeStudentName(){
    //code
}
~~~
두 메서드 전부 학생의 이름 학년을 가져온다고 했을 때, get과 take라는 비슷한 의미의 다른 단어를 사용해서, 코드가 나중에 유지보수될 때 혼란을 초래할 수 있다. **이름이 다르면 의도도 달라야한다**
___
### 단순한 함수
우리는 객체지향에서 SOLID 법칙에 대해서 들어봤다. 여기서 첫번째로 나오는 **Single Resposibility Principle**은 작성된 클래스가 하나의 기능을 가지고 있어야 한다는 원칙인데, 이와 유사하게, 함수또한 하나의 함수에 하나의 기능이 존재해야한다.
 - 함수는 하나의 기능만 존재
 - 최대한 작게 만들기
 - 함수 인수 최대한 적게
 - 작게 만들기

~~~dart
void infoStudent(case){
    switch(case){
        case 'name'
            print(this.name);
        case 'grade'
            print(this.grade);
        case 'score'
            print(this.score);
    }
}
~~~
만약에 위와 같은 infoStudent 메서드를 만들었다고 가정할 경우. SRP를 위반하고, OCP또한 위반하여서 만약에 학생 정보가 업데이트 된다면, infoStudent라는 메서드까지 전부 변경해야하게 됩니다.

### 명령과 조회의 분리
흔히 데이터를 조회하고 데이터를 modify하는 과정은 앱에 있어서 자주 일어나는 일이다. 이 때, 함수가 명령, 조회를 전부 처리한다면 함수를 유지 보수 하는 과정에서 이해하기 어려운 코드가 될 확률이 높아진다.
~~~dart
bool set(String attribute, int value);

if(set("영찬", 2)){
    //code
}
~~~
다음과 같은 예시에서, set이라는 메서드는 "영찬"이 존재하는지, 존재한다면 overwirte하는지, 아니면 새로 작성하는지 알 수 없다. 또한 그 결과를 bool타입으로 반환까지하는 아주 복합적인 함수다.
~~~dart
isAttributeExist()
setAttribute()
~~~
다음과 같이 조회와 명령을 분리하여서, 값이 존재하는지에 대해, 그리고 그 값을 설정하는 두 가지로 나눠서 정의하면 좀 더 유지보수도 편하고 human-readable한 함수가된다.
___
### 오류보다는 예외
~~~dart
Status deletePage(Page page) {
    if(deletePage(page) == E_OK) {
        if(registry.deleteReference(page.name) == E_OK) {
            if(configKeys.deleteKey(page.name.makeKey()) == E_OK) {
                log.info("page deleted");
                return E_OK;
            } else {
                log.error("config key not deleted");
            }
        } else {
            log.error("reference not deleted");
        }
    } else {
        log.error("page not deleted");
    }
    return E_ERROR;
}
~~~

다음과 같이 페이지 삭제 메소드에서 각각의 분기에서 에러를 출력한다고 한다면, 코드도 너무 길고, 구조적으로 가장 좋지않은 if-else문의 남발로 읽기 힘든 코드가 된다.

~~~dart
void deletePage(Page page) {
    try {
        deletePage(page);
        registry.deleteReference(page.name);
        configKeys.deleteKey(page.name.makeKey());
    } catch (Exception e) {
        log.error(e.getMessage());
    }
}
~~~
다음과 같이 try-catch로 묶어준다면, 훨씬 가독성이 좋은 코드가 완성된다.
___
### Test-Driven-Development
테스트 코드는 우리가 작성할 메인 코드 만큼이나 중요하다. TDD는 우리가 실제 코드를 짜기 전부터 단위 테스트를 먼저 작성하는 기법이다. 테스트 코드를 먼저 작성해보는 것은, 코드의 전체적인 로직을 파악하고 구성하는데 있어서 유연함을 제공한다.
1. 실패하는 단위테스트를 작성하기 전 까지 실제 코드를 작성하지 않는다.
2. 컴파일은 실패하지 않으면서, 실행이 실패하는 정도의 단위 테스트를 작성한다.
3. 현재 실패하는 테스트를 통과할 정도로만 실제 코드를 작성한다.

또한 SRP처럼, 1개의 테스트 코드는 1가지 개념만을 테스트하는 것이 바람직하다.

또한 깨끗한 테스트 코드를 위한 FISRT규칙도 존재한다.
- Fast          빠르게
- Independent   독립적으로
- Repeatable    재사용 가능하게
- Self-Validation   자체 검증
- Timely    적시에
___
### 단순한 클래스
클래스 또한 함수와 마찬가지로 최대한 간결하게 작성하는 것이 좋다. SRP에 따라서 1가지 책임만 가져야 한다.
___
### 높은 클래스 응집도, 낮은 결합도
응집도는 클래스의 메소드와 변수가 클래스에 얼마나 의존되는지 이고, 결합도가 높으면 다른 클래스간의 요소들이 얼마나 의존하고 있는지이다.
응집도는 높혀야 하고, 결합도는 낮아야 한다. 응집도가 너무 낮은 클래스는 다른 클래스의 변화에 민감해지고, 유지보수하기 힘들며, 재사용과 유지보수가 힘들다.
___
### 변경하기 쉬운 클래스
코드를 작성하면, 요구사항이 계속해서 변하기 때문에, 클래스도 변경하기 쉬워야 한다. SRP는 이를 위해서 기본적으로 지켜져야할 원칙이며, 추상체를 통한 다형성을 통해 클래스가 변경하기 쉬워져야 한다. OCP를 통해서 기존의 코드가 확장되어 변경하는건 쉽지만, 이미 작성되어 있는 코드가 변경되는 일은 적어야 한다
___
### 디미터 법칙
객체지향에서 중요한 법칙으로, 객체내의 자료를 공개하는 것이 아니라, 함수를 공개하는 법칙을 말한다. 
>객체가 어떤 메세지를 주고 받는가?

~~~dart
class User {
    String _email;
    String _name;
    Address _address;
}

class Address {
    String _region;
    String _details;
}
~~~

다음과 같은 코드가 있다고 했을 때.
~~~dart
class NotificationService {

    void sendMessageForSeoulUser(User user) {
        if("서울".equals(user.getAddress().getRegion())) {
            sendNotification(user);
        }
    }
}
~~~
다음과 같이 코드를 짜게 되면 , class 객체가 가지고 있는 region을 확인한다.

~~~dart
class Address {

    String region;
    String details;

    bool isSeoulRegion() {
        return "서울".equals(region);
    }
}

class User {

    String email;
    String name;
    Address address;

    bool isSeoulUser() {
        return address.isSeoulRegion();
    }
}
~~~
이런식으로 메세지를 구현하면, 내부 데이터를 모르는 채 메세지를 보낼 수 있게 된다.

~~~dart
class NotificationService {

    void sendMessageForSeoulUser(User user) {
        if(user.isSeoulUser()) {
            sendNotification(user);
        }
    }
}
~~~
