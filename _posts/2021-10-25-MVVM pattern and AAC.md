---
title: MVVM 패턴과 AAC
author: Beanie
date: 2021-10-25 20:55:00 +0800
categories: [Mobile frontend, Android]
tags: [Design patterns, MVVM, AAC]
layout: post
current: post
class: post-template
subclass: 'post'
navigation: true
cover:  assets/img/post_images/mvvm_cover.jpg
---

이전 글에서 디자인 패턴 중 MVC 패턴을 다뤄보았다. 이번 글에서는 안드로이드 개발에서 더욱 선호되는 MVVM 패턴과 MVVM 패턴 적용을 더 쉽게 해주는 AAC에 대해서 살펴보려고 한다.
\
&nbsp;
## 앱 아키텍쳐의 원칙
&nbsp;

안드로이드 앱에는 여러 앱 구성요소가 포함된다.

1. 액티비티 : 사용자와 상호작용하기 위한 진입점. (Activity)
2. 서비스 : 백그라운드에서 앱을 계속 실행하기 위한 다목적 진입점. (Service)
3. 콘텐츠 제공자 : 파일 시스템, SQLite, 데이터베이스, 웹이나 앱이 액세스할 수 있는 저장 가능한 앱 데이터의
    공유 가능한 형태들을 관리한다. (ContentProvider)
4. 브로드캐스트 수신자 : 시스템이 사용자 플로우 밖에서 이벤트를 앱에 전달하도록 지원하는 구성 요소.
    앱이 시스템 전체의 브로드캐스트 알림에 응답할 수 있도록 해준다.
    가 대표가 된다. (BroadcastReceiver)

개발자는 앱 매니페스트에서 앱 구성요소의 대부분을 선언하고 안드로이드 OS에서는 매니페스트를 확인해서 기기의 사용자 환경에 앱을 통합하는 방법을 결정한다.

휴대폰에서는 앱 실행 중 전화가 오거나, 램 공간이 부족해서 OS에서 새로운 앱 실행을 위해 일부 앱을 강제 종료 시킬 수 있다.
이러한 환경 조건을 고려해 볼 때 앱 구성요소는 개별적이고 비순차적으로 실행될 수 있고, 운영체제나 사용자가 언제든지 앱 구성요소를 제거할 수 있게 되는데, 이런 경우들은 직접 제어할 수 없기 때문에

1. 앱 구성요소에 데이터나 상태를 저장해서는 안되며,
2. 앱 구성요소가 서로 종속되어서는 안된다.

따라서 아래의 아키텍쳐 원칙이 제안되고 있다.
\
&nbsp;
### 관심사 분리

코드를 작성할 때 Activity나 Fragment와 같은 UI 기반의 클래스는 UI 및 OS 상호작용을 처리하는 로집만 포함해야 한다. 이는 UI 클래스를 최대한 가볍게 유지하여 Lifecycle 관련 문제를 피하기 위함이다.

UI 클래스는 무언가를 소유하는 것이 아닌 구현된 앱의 내용을 띄워주는 클래스 일 뿐이며 따라서 OS는 메모리 부족과 같은 특정한 상황이 발생되면 언제든지 UI 클래스를 제거할 수 있다. (제거하더라도 앱 기반이 되는 내용이 복구가 되면 **UI클래스는 단지 내용을 띄우는 수단에 불구하기 때문에** 같은 내용이 화면에 표시될 수 있다.)

따라서 UI 클래스로부터 UI, OS 상호작용을 제외한 다은 로직을 분리하여 UI 클래스에 대한 의존성을 최소화하는 것이 앱 관리 측면에서 좋다.
\
&nbsp;
### 모델에서 UI 도출

UI는 Model에서 만들어져야 한다. Model은 앱의 데이터 처리를 담당하는 컴포넌트로, 앱의 View 객체 및 앱 컴포넌트와 독립되어 있으므로 앱의 Lifecycle에 영향을 받지 않는다.
(네트워크가 끊어져도 앱이 죽지 않고, 메모리 확보를 위해 앱이 강제종료되어도 데이터가 살아있는 앱)
\
&nbsp;
\
&nbsp;
## MVVM 개요
&nbsp;

이런 앱 아키텍쳐의 원칙에 따라 **MVC 패턴의 단점을 보완**하고자 등장한 디자인 패턴이 바로 **MVVM** 이다. **MVVM** 은 기존 MVC 에서처럼 **Controller 에게 막중한 역할을 부여하기보다,** 이 **동작 자체를 분리하여** 동작의 흐름을 더욱 **체계적으로 만들어주고 유지보수를 편리하게** 할 수 있도록 해주는 디자인 패턴이다.
**MVVM 은 Model, View, ViewModel** 로 이루어져 있다. 대충 이름으로 구성을 파악해보면 MVC 패턴이 Model, View, Controller로 이루어져 있던 것과 비교하여 MVVM 패턴에서는 Controller가 사라지고 대신 ViewModel이 생겼다. 그렇다고 ViewModel이 Controller의 역할을 가져갔다고 볼 순 없고, Controller의 역할이 축소되며 View와 통합되었고 ViewModel이 새롭게 등장하였다고 이해하는 게 좋다.

MVC 패턴과 비교하여 MVVM 패턴을 자세히 살펴보자. 아래 이미지는 이해하기 좋아서 https://aonee.tistory.com/48 에서 가져왔다.

<div style="text-align: left">
   <img src="/assets/img/post_images/mvvm1.png" width="100%"/>
</div>

* **View**
    * 사용자의 Action을 받는 곳
    * Activity/Fragment가 View의 역할
    * ViewModel의 데이터를 관찰하여 UI 갱신
* **ViewModel**
    * View의 데이터 요청을 Model로 전달
    * Model이 전달해주는 데이터를 받음
    * View에서는 ViewModel을 알고 있지만, ViewModel은 View를 알지 못함
* **Model**
    * 앱의 데이터 처리를 담당
    * ViewModel이 요청한 데이터를 반환
    * Room, Realm 등의 DB 사용이나 Retrofit을 통한 백엔드 API 호출(네트워킹)이 보편적으로 이루어짐

ViewModel은 View와 Model 사이에서 데이터를 관리하고 바인딩해주는 요소이다. View가 원하는 데이터를 ViewModel이 들고있는데 따라서 View에서 ViewModel가 가지고 있는 데이터를 관찰(Observing)한다. 이를 통해 View가 데이터에 직접 접근하는 것이 아니라 UI 업데이트에만 집중할 수 있다.

일반적으로 ViewModel과 View는 1:n의 관계이다. 따라서 View는 자신이 이용할 ViewModel을 선택하여 상태 변화 알림을 받게 된다. ViewModel은 View가 쉽게 사용할 수 있도록 Model의 데이터를 가공하여 View에게 제공한다.
\
&nbsp;
\
&nbsp;
## AAC
&nbsp;

MVVM 패턴은 많은 장점이 있지만 초보자가 쉽게 사용하기 다소 어렵다. 따러서 MVVM 패턴이 진입 장벽이 크다는 문제를 해결하기 위해서 구글에서 AAC라는 것을 제공한다.

> **AAC**(Android Architecture Components)는 테스트와 유지보수가 쉬운 앱을 디자인할 수 있도록 돕는 라이브러리의 모음

![Untitled](/assets/img/post_images/mvvm3.png)

구조도를 확인하면 각 구성요소가 아래 계층의 구성요소에만 종속됨을 알 수 있다.
- Activity / Fragment -> ViewModel
- ViewModel -> Repository,
- Repository는 유일하게 여러 다른 클래스에 종속되는데, Local DB(Room) / Server DB(Retrofit)에 종속된다.

이는 ViewModel에 어디에서 가져온 데이터든지 일관성있는 데이터를 제공해주기 위함이다. 위와 같이 설계할 경우 네트워크 연결과 관계없이, 얼마나 오랜만에 앱을 켰든지간에 앱에서는 Room DB에 저장해놓았던 데이터를 통해 UI를 미리 표시해준다. 그리고 이 데이터가 오래된 경우에는 Repository를 통해 데이터를 Update하게된다.

이제 AAC의 각 구성요소를 자세히 확인해보자.

### View

UI Controller을 담당하는 Activity, Fragment이다. 화면에 무엇을 그릴 지 결정하고, 사용자와 상호작용한다. 데이터의 변화를 감지하기 위한 옵저버를 가지고 있다.
\
&nbsp;
### ViewModel

ViewModel은 앱의 Lifecycle을 고려하여 **UI 관련 데이터를 저장하고 관리하는 컴포넌트**이다. AAC의 ViewModel은 안드로이드에서 자체적으로 **안드로이드의 생명주기를 고려해서** 만든 ViewModel로, MVVM에서의 ViewModel과는 전혀 다른 개념이다. Activity/Fragment 당 하나의 ViewModel만 생성 가능하다. 다음 그림을 살펴보자.

![Untitled](/assets/img/post_images/mvvm2.png)

이처럼 메모리 누수, 화면 회전과 같은 상황에서도 Data를 잘 저장할 수 있다.
UI Controller로부터 UI 관련 데이터 저장 및 관리를 분리하여 ViewModel이 담당하도록 하면 다음과 같은 문제를 해결할 수 있다.
* 안드로이드 프레임워크는 특정 작업이나 완전히 통제할 수 없는 기기 이벤트에 대한 응답으로 UI Controller를 제거하거나 다시 만들 수 있는데, 이런 경우 **UI Controller에 저장된 모든 일시적인 UI 관련 데이터가 삭제**된다. 단순한 데이터의 경우 `onSaveInstanceState()` 메서드를 사용하여 복구할 수 있지만 대용량의 데이터의 경우엔 불가능하다.
* UI Controller에서 데이터를 위한 비동기 호출을 한다면 **메모리 누수 가능성을 방지하기 위한 많은 유지 관리**가 필요하며, 위에서와 같이 데이터를 복귀해야 하는 경우 **비동기 호출을 다시해야 해서 리소스가 낭비**된다.
* UI Controller에서 DB나 네트워크로부터 데이터를 로드하도록 하면 구분된 다른 클래스로 역할이 분담되지 않고 단일 클래스가 혼자서 앱의 모든 작업을 처리하려고 할 수 있다. 이 경우 **테스트가 훨씬 더 어려워진다.**
\
&nbsp;

### Live Data

LiveData는 **식별 가능한 데이터 홀더 클래스**로 다른 앱 컴포넌트의 Lifecycle을 인식하며, 이를 통해 **활성 상태에 있는 앱 컴포넌트 옵저버에게만 업데이트 정보를 알린다.**

LiveData를 사용하면 다음과 같은 이점이 있다.
* **UI와 데이터 상태의 일치 보장**: LiveData는 `Observer Pattern`을 따르며 Lifecycle 상태가 변경될 때마다 `Observer` 객체에 알린다. 또 앱 데이터의 변경이 발생할 때마다 관찰자에게 알려 UI를 업데이트할 수 있도록 한다.
* **메모리 누수 없음**: `Observer`는 Lifecycle 객체에 결합되어 있으며 연결된 객체의 Lifecycle이 끝나면 자동으로 삭제된다.
* **중지된 활동으로 인한 비정상 종료 없음**: 활동이 백 스택에 있을 때를 비롯하여 `Observer`가 비활성 상태에 있으면 어떤 LiveData 이벤트도 받지 않는다.
* **Lifecycle을 더 이상 수동으로 처리하지 않음**: UI 컴포넌트는 관련 데이터를 관찰하기만 할 뿐 관찰을 중지하거나 다시 시작하지 않으며, LiveData가 이를 자동으로 관리한다.
* **최신 데이터 유지**: 컴포넌트가 비활성화되면 다시 활성화될 때 최신 데이터를 수신한다.
* **적절한 구성 변경**: 기기 회전과 같은 구성 변경으로 인해 액티비티나 프래그먼트가 다시 생성되면 최신 데이터를 즉시 받게 된다.
* **리소스 공유**: 앱에서 시스템 서비스를 공유할 수 있도록 싱글톤 패턴을 사용하는 LiveData 객체를 확장하여 시스템 서비스를 래핑할 수 있다.

LiveData 객체는 다음과 같은 순서로 사용된다.
1. `ViewModel` 클래스 내에서 특정 유형의 데이터를 보유할 `LiveData`의 인스턴스를 만든다.
2. `onChanged()` 메서드를 정의하는 `Observer` 객체를 UI Controller에 만든다. `onChanged()` 메서드는 `LiveData` 객체가 보유한 데이터가 변경될 경우 발생하는 작업을 제어한다.
3. `observe()` 메서드를 사용하여 `LiveData` 객체에 `Observer` 객체를 연결한다.
4. `LiveData` 객체를 업데이트하는 경우 `MutableLiveData` 클래스는 `setValue(T)` 또는 `postValue(T)` 메서드로 `LiveData` 객체에 저장된 값을 수정한다.


### Repository

ViewModel과 상호작용하기 위해 잘 정리된 데이터 API를 들고 있는 클래스이다. 앱에 필요한 데이터 (내장 DB or 외부 DB)를 가져온다. ViewModel은 DB나 서버에 직접 접근하지 않고, Repository에 접근하는 것으로 앱의 데이터를 관리한다.
\
&nbsp;
### RoomDatabase

Room 라이브러리는 SQLite에 추상화 레이어를 제공하여 원활한 DB 액세스를 지원하고 SQLite를 완벽히 활용할 수 있게 하는 라이브러리이다. Room 라이브러리를 사용하면 앱을 실행하는 기기에서 앱 데이터의 캐시를 만들 수 있으며, 이 캐시를 통해 사용자는 인터넷 연결 여부와 관계없이 앱에 있는 주요 정보를 일관된 형태로 볼 수 있다. Room에 대해서는 다음 포스팅에서 더 자세히 설명할 예정이다.
\
&nbsp;
\
&nbsp;
## MVVM 장단점
&nbsp;

**장점**

* 계속 데이터를 관찰하고 있기 때문에 UI 업데이트가 간편하다.
* 모듈화 되어 있어 유지보수에 용이하다.
* View가 직접 Model에 접근하지 않아 Activity/Fragment 라이프 사이클에 의존하지 않는다.

**단점**

* 아무래도 MVC 패턴보다는 처음 익숙해지는 데 시간이 많이 걸린다. (어렵다)

\
&nbsp;

***

참고 내용 출처 :
 * [https://kimyunseok.tistory.com/152](https://kimyunseok.tistory.com/152)
 * [https://velog.io/@hwi\_chance/Android-%EC%95%88%EB%93%9C%EB%A1%9C%EC%9D%B4%EB%93%9C-AAC](https://velog.io/@hwi\_chance/Android-%EC%95%88%EB%93%9C%EB%A1%9C%EC%9D%B4%EB%93%9C-AAC)
 * [https://aonee.tistory.com/48](https://aonee.tistory.com/48)
