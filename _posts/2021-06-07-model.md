```
title: Model and View
author: dongee_seo
date: 2021-06-07
categories: [Blogging, Tutorial]
tags: [google analytics, pageviews]
```

## 들어가며

---

앞서 클린아키텍처에서 살펴보았듯이 설계적 관점에서의 좋은 코드는 확장과 수정에 용이하며, 설계 이후에도 추가적인 유지 보수에 적은 비용이 들어가는 코드를 말한다. 이를 실현하기 위해 추구해야 할 설계 방향은 다음과 같다.

**“ 객체 간 응집도는 높이고, 결합도는 낮게. 요구 사항 변경 시, 코드 변경을 최소화 하는 방향으로.”**

의존성을 고려해본다면 UI(view)는 Domain(Model)에 의존성이 높다.

이러한 의존성을 해소하고 코드의 재활용성을 높이기위해(불필요한 중복을 막기 위해) MVC, MVP, MVVM 패턴등의 프레임 워크가 나오게 되었다고 본다.

## 본문

---

디자인패턴은 Model과 View의 의존성을 어떻게 제어하느냐에 따라 분류된다. 디자인패턴에 공통적으로 쓰이는 Model과 View에 대해서 간략히 설명하고, MVC, MVP, MVVM 각 디자인패턴의 차이점을 설명하겠다.

### 1. Model과 View

![](https://velog.velcdn.com/images%2Fseod0209%2Fpost%2Fee5fb76a-88fa-43f2-95a5-78c53dff6663%2Fimage.png)

(1) Model: 일종의 데이터(Data).
데이터 이외에 데이터를 조작하는 간단한 로직이 추가 되기도 한다. 이때 주의할 점은 모델이 데이터의 집합이라고 생각하면 안되다는 것이다. 모델은 UI와 Presenter를 제외한 모든 부분을 의미한다.

(2) View: 사용자에게 제공되어지는 UI Layer를 뜻한다.보통 Application에서 View CSS/HTML/XML/XAML 등으로 렌더링 된 화면을 가르킨다.

MVP 와 MVVM 은 MVC 에서 파생되었기에, MVC 를 먼저 다룬 후, 세가지의 패턴을 비교해보도록 하겠다.

### 2. MVC(Model View Controller)

![](https://velog.velcdn.com/images%2Fseod0209%2Fpost%2Fb3fbf6c1-14b9-414d-bf5a-c155696e6899%2FMVC%20%E1%84%80%E1%85%AE%E1%84%8C%E1%85%A9.png)
(1) 구조

- Model, View, Controller
- Controller: 사용자의 입력(Action)을 받고 처리하는 부분.
- View 는 웹 사이트, Controller 는 서버, Model 은 데이터베이스라고 생각할 수도 있다.

(2) MVC 패턴의 동작 순서

1. 입력은 Controller 로 들어온다.
2. Controller는 사용자의 Action을 확인하고, Action을 업데이트 한다.
3. Controller는 Model으르 나타내줄 View를 선택한다.
4. View는 Model을 이용해 업데이트 된 후 화면을 나타낸다.

(3) 특징

- View-Controller 관계: One-to-Many(일대다 관계)이다. Many-to-Many(다대다) 가 될 수도 있다.( Controller 는 view를 선택할 수 있기에 view를 여러개 관리할 수 있다. )
- View 는 Controller 의 존재를 모른다. (=Controller 가 원하는 View 를 선택)
- View 는 Model 의 변화에 대해 직접적으로 알지 못한다. 또한 Controller 는 view를 선택하지만 view를 직접 업데이트하지 않는다.
- 하지만 MVC에서 View가 업데이트 되는 방법을 살펴보면,

  ```null
   1) View가 Model을 이용하여 직접 업데이트 하는 방법

   2) Model에서 View에게 Notify 하여 업데이트 하는 방법

   3) View가 Polling으로 주기적으로 Model의 변경을 감지하여 업데이트 하는 방법.
  ```

(4) 한계점

MVC 패턴은 단순하다는 장점이 있다.

하지만, Model이 업데이트 되면 View도 업데이트가 된다.

즉, 의존성문제를 완전히 해결하지 못한다. 이러한 controller의 한계점을 극복하고자 mvp, mvvm이 등장하게되었다.

### 3. MVP(Model View)

![](https://velog.velcdn.com/images%2Fseod0209%2Fpost%2F54ded783-b0e7-4b8a-8093-88e6dc3ee019%2FMVP%20%E1%84%80%E1%85%AE%E1%84%8C%E1%85%A9.png)

(1) 구조

- Model, View, Presenter.
- Presenter: View 에서 요청한 정보로 Model을 가공하여 View에게 전달. View와 Model사이의 가교역할.

(2) MVP 패턴의 동작 순서

```null
 1. 입력은 View 로 들어온다.

 2. View는 데이터를 Presenter에 요청한다.

 3.  Presenter는 Model에 데이터를 요청한다.

 4. Model은 요청받은 데이터를 응답한다.

 5. Presenter는 View에게 데이터를 응답한다.

 6. View는 Presenter에게 받은 데이터를 이용하여 화면을 나타낸다.
```

(3) 특징

- View-Presenter 는 One-to-One(일대일) 관계이다.
- View 는 Presenter 를 참조하고, Presenter 는 View 의 존재를 알고 있다. (강한 결합 = 서로 의존성이 높다)
- View 는 Model 의 존재를 모른다.= > view 와 model의 의존성이 사라진다.

(4) 한계점

- View와 Model 은 오로지 Presenter 에 의해서만 상호작용을 하게 된다. 그로 인해 View-Model 관계는 분리되었지만, View-Presenter 관계는 서로 강하게 의존한다.
- 코드 상으로는 View-Presenter 일대일 관계로 인해, 각 View 를 위한 각 Presenter 가 필요하게된다. 이로인해 코드의 수가 상당히 증가하게 된다.

### 4. MVVM

![](https://velog.velcdn.com/images%2Fseod0209%2Fpost%2Fc0a21e1e-acea-4a15-bd86-e065cdab7d8f%2FMVVP%20%E1%84%80%E1%85%AE%E1%84%8C%E1%85%A9.png)
(1) 구조

- MVVM 은 MVC 와 유사하다.
- ViewModel : Controller 역할을 한다. 이름 그대로 View 를 위한 Model 이라고 보면 된다. 여기서 'View 를 위한 Model' 은 일반적인 Model 이 아닌 특정 View 에게 맞춰진 Model 을 의미한다.

(2) MVP 패턴의 동작 순서

```null
1. 사용자의 Action들은 View를 통해 들어온다.

2. View에 Action이 들어오면, Command 패턴으로 View Model에 Action을 전달한다.

3. View Model은 Model에 데이터를 요청한다.

4. Model 은 요청받은 데이터를 View Model에 응답한다.

5. View Model은 응답받은 데이터를 가공하여 저장한다.

6. View는 View Model과 Data Binding하여 화면을 나타낸다.
```

(3) 특징

- Command 패턴: 객체의 행위( 메서드 )를 클래스로 만들어 캡슐화함으로써 주어진 여러 기능을 실행할 수 있는 재사용성이 높은 클래스를 설계하는 패턴
  즉, 이벤트가 발생했을 때 실행될 기능이 다양하면서도 변경이 필요한 경우에 이벤트를 발생시키는 클래스를 변경하지 않고 재사용하고자 할 때 유용하다.
- Data Binding(데이터 연결): 두 데이터 혹은 정보의 소스를 모두 일치시키는 기법이다. 즉 화면에 보이는 데이터와 브라우저 메모리에 있는 데이터를 일치시키는 기법이다. 많은 자바스크립트 프레임워크가 이러한 데이터 바인딩 기술을 제공하고 있다. 하지만 대다수의 자바스크립트 프레임워크가 단방향 데이터 바인딩을 지원하는 반면 AngularJS는 양방향 데이터 바인딩을 제공한다.
- View-ViewModel 관계는 One-to-Many 관계이다.
- ViewModel 은 View 의 존재를 모른다.
- View 는 Model 의 존재를 모른다. View 는 ViewModel 만을 고려한다. 결과적으로 View-Model 관계는 분리된다. 또한 현재 많은 변화를 요구하는 사용자 인터페이스에 있어, View 를 위한 View 가 중심이 되는 패턴이라고 볼 수 있다.
- MVP 의 문제점인 View-Presenter 관계의 의존성은 **데이터 바인딩과 명령을 통해 해결**하게 된다.
- 그래픽 사용자 인터페이스(GUI; Graphic User Interface)의 개발을 [비즈니스 로직](https://ko.wikipedia.org/wiki/%EB%B9%84%EC%A6%88%EB%8B%88%EC%8A%A4_%EB%A1%9C%EC%A7%81) 또는 [백-엔드 로직](https://ko.wikipedia.org/wiki/%ED%94%84%EB%9F%B0%ED%8A%B8%EC%97%94%EB%93%9C%EC%99%80_%EB%B0%B1%EC%97%94%EB%93%9C)(모델)로부터 분리시켜서, View가 어느 특정한 모델 플랫폼에 종속되지 않도록 해준다
- 바인더(연결자), 뷰 모델등 을 사용하여 들어오는 데이터를 검증. 결과적으로 모델과 프레임워크가 가능한 많은 작업을 수행하며, 뷰를 직접 조작하는 응용 프로그램 로직은 최소화하거나 아예 없애버린다.

⇒ 서로가 각자(view- model)의 역할에 충실할수 있게됨.

(4) 한계점

- View Model의 설계가 쉽지 않다. 데이터 바인딩은 자동으로 이루어 지지 않는다.
- 데이터 바인딩을 도와주는 라이브러리(ex. injector)를 함께 사용하지 않으면, 많은 기반 코드를 작성해야 함.

## 나가며

---

디자인 패턴은 많은 측면에서 개발에 도움을 주기 위한 수단으로써 필수이지만 어떤 패턴을 선택할지는 상황과 구조가 고려되어야한다.

예를 들면, 개발환경 특성상 View 와 Controller 의 역할을 분리할 수 없는 경우, MVP 또는 MVVM 을 선호한다고 한다. 하지만 굳이 역할분리가 필요없는 상황에 도입하는것은 오히려 자원낭비이다. 따라서 어느 것이 더 좋은 개발 방법론이라고 확언하여 말할 수 없다. 어떤 언어가 더 좋다고 말할 수 없는 것과 같은 맥락이다.

이번 세미나를 준비하면서 클린아키텍처가 필요한 이유, 의존성을 왜고려해야하는지를 알게되었고, ‘좋으면 무조건 도입이 해야겠다.’가 아니라 **목적과 환경에 따라 어떤 패턴이 어울릴 지 고민해야 한다**는 생각이드는 뜻깊은 시간이 되었다.

> ### 참고자료

[Command Pattern]

- [https://gmlwjd9405.github.io/2018/07/07/command-pattern.html](https://gmlwjd9405.github.io/2018/07/07/command-pattern.html)
- [https://gmlwjd9405.github.io/2018/07/07/command-pattern.html](https://gmlwjd9405.github.io/2018/07/07/command-pattern.html)
- [https://victorydntmd.tistory.com/295](https://victorydntmd.tistory.com/295)

[Business Logic]
-[[https://genesis8.tistory.com/233](https://genesis8.tistory.com/233)

[DAO]

- [https://genesis8.tistory.com/214#:~:text=DAO란](https://genesis8.tistory.com/214#:~:text=DAO%EB%9E%80)
- [https://genesis8.tistory.com/214#:~:text=DAO%EB%9E%80%20Data%20Access%20Object,%EC%A0%91%EA%B7%BC%ED%95%98%EB%8A%94%20%EA%B0%9D%EC%B2%B4%EB%A5%BC%20%EB%A7%90%ED%95%9C%EB%8B%A4.&amp;text=DAO(Data%20Access%20Object)%EB%8A%94,%ED%95%98%EB%8F%84%EB%A1%9D%20%EB%A7%8C%EB%93%A0%20%EC%98%A4%EB%B8%8C%EC%A0%9D%ED%8A%B8%EB%A5%BC%20%EB%A7%90%ED%95%9C%EB%8B%A4](<https://genesis8.tistory.com/214#:~:text=DAO%EB%9E%80%20Data%20Access%20Object,%EC%A0%91%EA%B7%BC%ED%95%98%EB%8A%94%20%EA%B0%9D%EC%B2%B4%EB%A5%BC%20%EB%A7%90%ED%95%9C%EB%8B%A4.&text=DAO(Data%20Access%20Object)%EB%8A%94,%ED%95%98%EB%8F%84%EB%A1%9D%20%EB%A7%8C%EB%93%A0%20%EC%98%A4%EB%B8%8C%EC%A0%9D%ED%8A%B8%EB%A5%BC%20%EB%A7%90%ED%95%9C%EB%8B%A4>)).)

[Design Pattern]

- [https://dailyheumsi.tistory.com/148](https://dailyheumsi.tistory.com/148)

[Clean Architecture]

- [https://suhwan.dev/2019/10/06/review-clean-architecture/](https://suhwan.dev/2019/10/06/review-clean-architecture/)
- [https://pusher.com/tutorials/clean-architecture-introduction](https://pusher.com/tutorials/clean-architecture-introduction)
- [https://dailyheumsi.tistory.com/239](https://dailyheumsi.tistory.com/239)
- [https://velog.io/@trequartista/TIL-Clean-Architecture-Design-Pattern1](https://velog.io/@trequartista/TIL-Clean-Architecture-Design-Pattern1)

[Clean Architecture and MVVM]

- [https://tech.olx.com/clean-architecture-and-mvvm-on-ios-c9d167d9f5b3](https://tech.olx.com/clean-architecture-and-mvvm-on-ios-c9d167d9f5b3)
- [https://velog.io/@dlrmsghks7/whatismvvmpattern](https://velog.io/@dlrmsghks7/whatismvvmpattern)

[MVC, MVP, MVVM]

- [https://mygumi.tistory.com/304](https://mygumi.tistory.com/304)
- [https://beomy.tistory.com/43](https://beomy.tistory.com/43)
- [https://stfalcon.com/en/blog/post/android-mvvm](https://stfalcon.com/en/blog/post/android-mvvm)
