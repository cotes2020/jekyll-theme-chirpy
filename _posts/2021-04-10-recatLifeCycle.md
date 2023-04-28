> 컴포넌트가 브라우저에서 나타날때, 사라질때, 그리고 업데이트 될 때, 호출되는 API 이다.
> '컴포넌트 초기 생성-업데이트-제거' 순서대로 살펴보겠다.

## 1. 컴포넌트 초기생성

컴포넌트가 브라우저에 나타나기 전, 후에 호출되는 API 들

---

### 1-1 constructor

```jsx
constructor(props) {
  super(props);
  // 컴포넌트 생성자 함수.
  // 컴포넌트가 새로 만들어질 때마다 이 함수가 호출된다.
}
```

### 1-2 componentDidMount

컴포넌트가 화면에 나타나게 됐을 때 호출

```jsx
componentDidMount() {
  // 외부 라이브러리 연동: D3, masonry, etc
  // 컴포넌트에서 필요한 데이터 요청: Ajax, GraphQL, etc
  // DOM 에 관련된 작업: 스크롤 설정, 크기 읽어오기 등
}
```

## 2. 컴포넌트 업데이트

:: props 의 변화, 그리고 state 의 변화에 따라 결정.
업데이트가 되기 전/후에 어떠한 API 가 호출 되는지 살펴보겠다.

### 2-1 static getDerivedStateFromProps()

//v16.3 이후에 만들어진 라이프사이클 API.
//props 로 받아온 값을 state 로 동기화 하는 작업을 해줘야 하는 경우에 사용.

```jsx
static getDerivedStateFromProps(nextProps, prevState) {
  // 여기서는 setState 를 하는 것이 아니라
  // 특정 props 가 바뀔 때 설정하고 설정하고 싶은 state 값을 리턴하는 형태로 사용.
  /*
  if (nextProps.value !== prevState.value) {
    return { value: nextProps.value };
  }
  return null;
  // null 을 리턴하면 따로 업데이트 할 것은 없다라는 의미
  */
}
```

### 2-2 shouldComponentUpdate

//컴포넌트를 최적화하는 작업에서 매우 유용하게 사용됨.

```jsx
shouldComponentUpdate(nextProps, nextState) {
  // return false 하면 업데이트를 안함
  // return this.props.checked !== nextProps.checked
  return true;
}
```

리액트는 **변화가 발생하는 부분만 업데이트**를 해줘서 성능이 꽤 잘 나온다. 하지만, **변화가 발생한 부분만 감지해내기 위해서는 Virtual DOM** 에 한번 그려줘야한다.

즉, 현재 컴포넌트의 상태가 업데이트되지 않아도, 부모 컴포넌트가 리렌더링되면, 자식 컴포넌트들도 렌더링 됩니다. (여기서 “렌더링” 된다는건, render() 함수가 호출된다는 의미)
변화가 없으면 물론 DOM 조작은 하지 않게 됩니다. 그저 Virutal DOM 에만 렌더링 할 뿐이죠.이 작업은 그렇게 부하가 많은 작업은 아니지만, **컴포넌트가 무수히 많이 렌더링된다면** 쓸대없이 CPU가 낭비된다.

이처럼 쓸대없이 낭비되고 있는 이 CPU 처리량을 줄여주기 위해서,** Virtual DOM 에 리렌더링 하는것마저도 불필요할경우엔 방지하기 위해**서 shouldComponentUpdate 를 작성합니다.

이 함수는 기본적으로 true 를 반환한다.
우리가 따로 작성을 해주어서 **조건에 따라 false 를 반환**하면 해당 조건에는 render 함수를 호출하지 않는다.

### 2-3 getSnapshotBeforeUpdate()

발생하는 시점은 다음과 같다.

```null
render()
getSnapshotBeforeUpdate()
실제 DOM 에 변화 발생
componentDidUpdate
```

이 API를 통해서 DOM 변화가 일어나기 직전의 DOM 상태를 가져오고, 여기서 리턴하는 값은 componentDidUpdate 에서 **3번째 파라미터**로 받아올 수 있게 된다.

```jsx
getSnapshotBeforeUpdate(prevProps, prevState) {
    // DOM 업데이트가 일어나기 직전의 시점.
    // 새 데이터가 상단에 추가되어도 스크롤바를 유지해보겠습니다.
    // scrollHeight 는 전 후를 비교해서 스크롤 위치를 설정하기 위함이고,
    // scrollTop 은, 이 기능이 크롬에 이미 구현이 되어있는데,
    // 이미 구현이 되어있다면 처리하지 않도록 하기 위함입니다.
    if (prevState.array !== this.state.array) {
      const {
        scrollTop, scrollHeight
      } = this.list;

      // 여기서 반환 하는 값은 componentDidMount 에서
      // snapshot 값으로 받아올 수 있습니다.
      return {
        scrollTop, scrollHeight
      };
    }
  }

  componentDidUpdate(prevProps, prevState, snapshot) {
    if (snapshot) {
      const { scrollTop } = this.list;
      if (scrollTop !== snapshot.scrollTop) return;
      // 기능이 이미 구현되어있다면 처리하지 않습니다.
      const diff = this.list.scrollHeight - snapshot.scrollHeight;
      this.list.scrollTop += diff;
    }
  }
```

### 2-4 componentDidUpdate

:: 컴포넌트에서 render() 를 호출한 다음에 발생.

```jsx
componentDidUpdate(prevProps, prevState, snapshot) {
}
```

이 시점에선 this.props 와 this.state 가 바뀌어있다. 그리고 파라미터를 통해 이전의 값인 prevProps 와 prevState 를 조회 할 수 있다. 그리고, getSnapshotBeforeUpdate 에서 반환한 snapshot 값은 세번째 값으로 받아온다.

## 3. 컴포넌트 제거

컴포넌트가 더 이상 필요하지 않게 되면 단 하나의 API 가 호출된다.

```jsx
componentWillUnmount
componentWillUnmount() {
  // 이벤트, setTimeout, 외부 라이브러리 인스턴스 제거
}
```

- 등록했었던 이벤트를 제거
- 만약에 setTimeout 을 걸은것이 있다면 clearTimeout 을 통하여 제거
- 외부 라이브러리를 사용한게 있고, 해당 라이브러리에 dispose 기능이 있다면 여기서 호출해주면 됨.

> 참고자료
>
> - [https://velopert.com/3631](https://velopert.com/3631)
> - [https://developer.mozilla.org/ko/docs/Web/JavaScript/Reference/Classes/constructor](https://developer.mozilla.org/ko/docs/Web/JavaScript/Reference/Classes/constructor)
