```
title: Test
author: dongee_seo
date: 2023-04-25
categories: [Blogging, Tutorial]
tags: [google analytics, pageviews]

```

1. data-model` 사용 이유가 있나요? 그렇다면 DTO와의 차이는 무엇인가요? 해당 모델을 사용한 목적과 도입한 결과에 대해서도 설명해주세요.
   [](https://emewjin.github.io/model/)[https://emewjin.github.io/model/](https://emewjin.github.io/model/) > [](https://yozm.wishket.com/magazine/detail/1663/)[https://yozm.wishket.com/magazine/detail/1663/](https://yozm.wishket.com/magazine/detail/1663/) > [](https://velog.io/@jjunyjjuny/%ED%94%84%EB%A1%A0%ED%8A%B8%EC%97%94%EB%93%9C-%EB%8D%B0%EC%9D%B4%ED%84%B0%EB%A5%BC-%EA%B0%80%EA%B3%B5%ED%95%98%EB%9D%BC)[https://velog.io/@jjunyjjuny/프론트엔드-데이터를-가공하라](https://velog.io/@jjunyjjuny/%ED%94%84%EB%A1%A0%ED%8A%B8%EC%97%94%EB%93%9C-%EB%8D%B0%EC%9D%B4%ED%84%B0%EB%A5%BC-%EA%B0%80%EA%B3%B5%ED%95%98%EB%9D%BC) > [](https://codingmoondoll.tistory.com/entry/%EB%8D%B0%EC%9D%B4%ED%84%B0%EB%B2%A0%EC%9D%B4%EC%8A%A4-%EB%AA%A8%EB%8D%B8%EB%A7%81)[https://codingmoondoll.tistory.com/entry/데이터베이스-모델링](https://codingmoondoll.tistory.com/entry/%EB%8D%B0%EC%9D%B4%ED%84%B0%EB%B2%A0%EC%9D%B4%EC%8A%A4-%EB%AA%A8%EB%8D%B8%EB%A7%81) > [](https://blog.soomgo.com/blog/using-repository-pattern-at-frontend/)[https://blog.soomgo.com/blog/using-repository-pattern-at-frontend/](https://blog.soomgo.com/blog/using-repository-pattern-at-frontend/) > [](https://tech.kakao.com/2022/03/15/2022-newkrew-onboarding-fe/)[https://tech.kakao.com/2022/03/15/2022-newkrew-onboarding-fe/](https://tech.kakao.com/2022/03/15/2022-newkrew-onboarding-fe/)




1. `micro-frontend` 아키텍쳐에 대해서 설명해주세요. 해당 개념을 실무에 도입해본 경험이 있나요? 있다면 장점, 단점, 어려웠던점, 해결방안에 대해서 설명해주세요.
   [](https://velog.io/@kylexid/%EB%A7%88%EC%9D%B4%ED%81%AC%EB%A1%9C%ED%94%84%EB%A1%A0%ED%8A%B8%EC%97%94%EB%93%9C-%EC%95%84%ED%82%A4%ED%85%8D%EC%B3%90)[https://velog.io/@kylexid/마이크로프론트엔드-아키텍쳐](https://velog.io/@kylexid/%EB%A7%88%EC%9D%B4%ED%81%AC%EB%A1%9C%ED%94%84%EB%A1%A0%ED%8A%B8%EC%97%94%EB%93%9C-%EC%95%84%ED%82%A4%ED%85%8D%EC%B3%90)

# **프론트엔드 개발에서 서버 데이터를 모델로 관리한다는 것**

2021.09-2022.03 백엔드에서 보내준 데이터를 Class를 이용해 Model로 먼저 정의하고, 그 후에 UI개발
( `UI를 렌더링하기 위해 필요한 데이터`를 받아서 이 데이터를 기반으로 클라이언트에서 직접 UI를 렌더링)

관리자 3.0 개발을 시작하면서 서버측에서 보내주는 데이터 전체를 가지고 있다가 화면서 보여주는 것이 아니라→ 서버측에서 보내주는 데이터를 “화면에 필요한 데이터”만 으로 구성된 해당 화면만을 위한 데이터모델을 도입하게 되었다.

(**데이터는 그것을 소비하는 주체에 따라 모습을 달리해야한다.**
여기서 한 가지 문제점이 생기는데, 일반적으로 **서버에서 내려주는 데이터의 형태**
와 **UI를 렌더링하기 위한 최적의 데이터의 형태**
가 **😱 항상 일치하지는 않는다** )
`REST API`
의 응답값을 **Class Instance로 변환해서 사용하는 방식**
과 , **decode/encode 함수를 사용하는 방식**
을 소개하고자한다.

1. 응답값에 **불필요한 필드**가 있다.
   - `updated_at`, `tags`, `url_slug` 등..
   - 물론 실제 비즈니스 로직을 본 건 아니라서 사용될지 모르겠지만, 슬쩍봐선 불필요하다.
2. 백과 프론트의 **변수 명명법이 다르다**
   - 필드명이 스네이크\_케이스로 내려왔지만, 프론트엔드는 주로 파스칼/카멜케이스를 사용한다.
   - 같은 데이터도 달리 표현할 수 있다
     - `user` / `writer`
     - `released_at` / `createdDate`
3. **바로 쓰일 수 없는** 값이 있다
   - `released_at`은 UTC 표기법으로 되어 있는데, `yyyy년 mm월 dd일`의 형태로 변환해주어야한다.
4. **기존 데이터를 기반으로 연산한 새로운 데이터**가 필요할 수 있다.
   - 혹시나 `recommended_posts`가 비어있을 때 다른 UI를 표시하려면 `recommended_posts.length === 0`와 같은 로직이 추가되어야 한다. `isEmpty` 같은 `boolean`값이 있다면 더 깔끔해질 것이다.
   - `title`과 `short_description`을 합쳐야한다거나, `likes`와 `comments_count`의 비율을 보여준다거나 하는 기획이 추가된다면, 그 연산 결과가 필요하다.

1번 문제의 경우 사용하지 않으면 그만이지만, 나머지 2~4의 경우 **추가적인 가공 로직**이 필요하다.
그리고 보통 이 **추가적인 가공 로직**을 컴포넌트 내부에서 작성하곤 한다. 따라서

- 컴포넌트의 관심사에 맞지 않는 불필요한 비즈니스 로직이 생기며
- 유지보수가 어려운 코드가 된다

프로젝트가 커지면 서버는 **도메인 단위로 코드를 관리해야할 필요성이 생긴다.** **서버는 어느 한 도메인에 맞추지 않고 어떤 요청에도 유연하게 대응할 수 있는 구조를 갖추어야한다.**

⇒ 클라이언트의 모든 사소한 차이에 대응하는 API를 만들어 두는 것은 리소스 낭비고, 유지보수도 힘들어진다.

⇒ 또한, **해당 데이터가 클라이언트에서 어떻게 렌더링 될 것인가**는 서버의 관심사가 아니다.
서버는 데이터의 CRUD 관점에서 요청과 데이터를 처리하는 것에 집중한다.

따라서 `프론트엔드` 입장에서, 해당 API를 여러 페이지에서 호출하는 경우
페이지마다 디자인/기획에 따라 미묘하게 필요한 데이터가 달라질 수 있다.
페이지를 만들다보면 어떤 추가적인 가공이 필요할지 모르기 때문에 그 때마다 서버에 요청하기보다 스스로 컨트롤 할 수 있는게 좋다.

## 뭐가 좋았나?

모델을 정의하여 사용할 때 다음과 같은 이점이 있었다.

- 서버로부터 받아온 값과 (필드) 각 필드에 대한, 뷰 로직과는 독립적인 로직을 모아두어 한 곳에서 관리할 수 있다.
- 공통되는 필드, 로직의 경우 재활용할 수 있다.
- 뷰와 분리되어 있기에 테스트가 용이하다.
- 백엔드에서 내려주는 데이터 스키마가 변경되었을 경우, 모델만 수정하면 된다는 유지보수의 용이함도 있다.
  :비즈니스전체를 목적으로 만들어진 데이터를 그대로 화면으로 내려보낼 경우,
  서비스의 규모가 커지면 코드가 여기저기 퍼지게 되어 유지보수에 상당한 어려움을 겪게 된다.
- 모델 인스턴스를 만들 때에는 class를 썼지만, 위의 이점들이 꼭 class를 썼기 때문에 가능했다고 할 수는 없다. class가 아닌 일반 객체였어도 가능하지 않았을까?
- API가 미완성일지라도 레이아웃 작업을 진행할 수..
  :서버의 API 응답 스키마가 확정되지 않았더라도 **클라이언트에서 사용할 데이터의 인터페이스를 우선 정의하고**
  그에 맞게 레이아웃 작업을 진행해두면, 향후 API에서 예상과 다른 응답값이 내려왔을지라도 **데이터 가공 레이어에서 미리 정의해두었던 클라이언트 인터페이스에 맞게 변경시켜주면**
  클라이언트 코드를 수정할 필요가 없어진다!

## 아쉬웠던 점

**프로젝트 개발 때에는 정신없이 개발하느라 몰랐는데 끝나고 나니 보이는 점들이 있었다.**

### 1. 이 로직이 꼭 모델 안에 있어야 할까? 모델이 너무 많은 일을 하는 것 같다

사실은 역할과 책임이 여럿인 만큼 모델도 여럿으로 분리가 되어있어야 하는데 지금은 무조건 도메인 하나당 한 모델로 정리가 되어있다보니 복잡도를 높이는 원인이 되고 있었다.

더 자세히 이야기하면 현재 프로젝트는 Model을 DTO처럼 쓰고있고, 그러면서도 단순히 DTO의 역할만 해주는 것이 아니라 비즈니스로직에 가까운 유틸성 메서드들이 함께 들어있다.

게다가 이 모델은 레이어마다 각각 존재하는 것이 아니라 하나의 모델로 여러 레이어의 DTO 역할을 하고 있다보니 하는 일이 많아 문제가 많다.

가장 대표적인 문제는 A 레이어에서 하는 일과 B 레이어에서 하는 일이 충돌되는데, 모델은 하나라는 것이다.
해당 모델의 문제는 다음과 같다.

- 코드가 굉장히 방대하여, 레이어마다 DTO를 각각 만들어주면 엄청나게 많은 코드가 생긴다

설계했던 모델의 메서드들이 사실은 모델 안에 있으면 안될 것 같다는 생각을 하게 되었다.

예를들어 지금은 각 필드에 관한 로직이고 여러 뷰에서 쓰인다면, 모델안에 메서드로 그 로직을 들고 있게 되어있다.

가령, 이런식이다.

```jsx
/* 결제금액: 부가세포함여부, 공금가액 변동 실시간 반영값 */
   get supplyPrice  () {
        if (this.isApplyReturn) {
            let total: number = this.totalPrice - this.returnPrice + this.spendInStoreCredit;
            return total < 0 ? 0 : total;
        } else {
            return this.totalPrice + this.spendInStoreCredit;
        }
    };
```

---

이 부분이 우리 서비스에서만 사용되는 비즈니스 로직이라고 생각해서 메서드로 달아두었던 건데 사실은 뷰 로직이었다. 숫자를 3자리 씩 ’,‘를 찍는다와 같이 뷰에서 처리해줘야 하는 로직이었던 것이다.
그럼 재사용은? 지금은 모델안에 메서드로 로직이 존재하기 때문에 어떤 뷰에서든 재사용하기 쉽지만, 꼭 모델 안에 있지 않아도 아마도 훅으로 만들거나 유틸 함수로 만들거나 해서 재사용 할 수 있을 것이다.

![https://user-images.githubusercontent.com/76927618/171328203-64ba709e-6af2-44b8-9572-452207617376.png](https://user-images.githubusercontent.com/76927618/171328203-64ba709e-6af2-44b8-9572-452207617376.png)

### 2. 해당 클래스의 데이터를 변경해도 동일한 객체로 보기때문에 얕은복사가 필요하다.

뷰에서 해당 데이터를 변경하더라도 리액트가 렌더되지 않았다. 왜냐하면 같은 객체로 인식하기 때문. 변경된 데이터로 화면이 보여야 하는데 계속 기존값으로 보임.

리액트에선 초기에 한번 렌더링을 진행하고, 그 이후에 특정 조건이 발생하면 다시 렌더링을 진행하는 **리렌더링**이있습니다.

- 내부 상태(state) 변경시
- 부모에게 전달받은 값(props) 변경시
- 중앙 상태값(Context value 혹은 redux store) 변경시
- 부모 컴포넌트가 리렌더링 되는 경우

위의 경우가 컴포넌트가 리렌더링 되는 조건입니다.

리액트가 아무리 최적화가 잘 되어있다고해도, 무분별하게 렌더링이 일어날 경우 성능 저하가 일어나게 되기 때문에, 이러한 조건들을 기준을 두고 코드를 작성하여 무분별하게 렌더링이 일어나지 않도록 주의하여야 합니다.

- **redux store 변경**시 자동으로 리렌더링이 되는 이유는, 리덕스 스토어가 `<Provider store={store}>`로 컴포넌트를 감싸주었을 때, 스토어 상태가 변경될 때마다 이를 참조하는 컴포넌트들이 리렌더링이 될 수 있도록 react-redux 라이브러리가 자동적으로 컴포넌트 들의 렌더 함수들을 subscribe 해주기 때문입니다.

1. 위의 조건을 통해 컴포넌트 리렌더링
2. 구현부 실행 = props 취득, hook 실행, 내부 변수 및 함수 재 생성
3. return 실행, 렌더링 시작
4. 렌더 단계(Render Phase): 새로운 가상 DOM 생성 후 이전 가상 DOM과 비교해 달라진 부분을 탐색하고 실제 DOM에 반영할 부분을 결정
5. 커밋 단계(Commit Phase): 달라진 부분만 실제 DOM에 반영
6. useLayoutEffect: 브라우저가 화면에 Paint하기 전에 useLayoutEffect에 등록해둔 effect(부수 효과 함수)가 동기적으로 실행되며, 이때 state, redux store 등의 변경이 있다면 한번 더 리렌더링
7. Paint: 브라우저가 실제 DOM을 화면에 그림. didUpdate 완료.
8. useEffect: update되어 화면에 그려진 직후, useEffect에 등록해둔 effect(부수 효과 함수)가 비동기로 실행

### 3. Redux의 경우: 인스턴스를 풀었다가 만들었다가, 왔다갔다 하는 불편함

리덕스 스토어에 저장시 인스턴스는 저장할 수 없다. 때문에 매번 `classToPlain`으로 class를 json객체로 풀어서 넣어주고, 리덕스에서 나온 데이터는 다시 class로 만들어주어야 했다.
그 뿐인가? json으로 필요로 하는 곳인데 `classToPlain`을 쓸 수 없는 상황에서는 대비하기 위해 `toJson` 이라는 메서드를 모든 모델 안에 넣어주어야 했다.
또, react의 state로 데이터를 두고 유저 인터랙션에 의해 업데이트 하기 위해서는 class를 json으로 변환해서 state에 넣어주어야 했다.

**이게 너무 번거롭고, 왜 이렇게 해야 하는지 스스로도 납득이 잘 되지 않았다. 나중에는 급기야 ‘음… 왜 class로 만들어야 하는거지? 어차피 다시 풀건데…’ 하는 생각도 들었다**

```jsx
JSON.parse(JSON.stringify(store.getState().feedSlice.feed.map((f) => new Feed(f)))),
```

## 내가 생각하는 해결방안
1. 내부 함수를 utill로 뺀다
2. class 내부 요소를 직접 변환할 경우(state를 직접 업데이트 할 경우) useState와 동시에 업데이트 해준다.
