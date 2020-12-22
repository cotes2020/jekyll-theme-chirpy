---
title: react hook useEffect()
author: juyoung
date: 2020-11-19 18:46:00 +0800
categories: [react, tutorial]
tags: [react]
---

 # [1] Mount: 컴포넌트가 처음 실행될 때   
 1. 컴포넌트가 시작되면 우선 context, defaultProps와 state를 저장합니다. 
 2. 그 후에 componentWillMount 메소드를 호출합니다. 
 3. 그리고 render로 컴포넌트를 DOM에 부착한 후 Mount가 완료된 후 componentDidMount가 호출됩니다.
---


# [2] props가 업데이트될 때
1. 업데이트되기 전에 업데이트가 발생하였음을 감지하고, componentWillReceiveProps 메소드가 호출됩니다. 
2. 그 후 shouldComponentUpdate, componentWillUpdate가 차례대로 호출된 후,
3. 업데이트가 완료(render)되면 componentDidUpdate가 됩니다.   
* 이 메소드들은 첫 번째 인자로 바뀔 props에 대한 정보를 가지고 있습니다. componentDidUpdate만 이미 업데이트되었기 때문에 바뀌기 이전의 props에 대한 정보를 가지고 있습니다.


# 라이프사이클
class 컴포넌트 때는 라이프사이클이 컴포넌트에 중심이 맞춰져 있었습니다.  
함수 컴포넌트에서는 특정 데이터에 대해서 라이프사이클이 진행됩니다.   
클래스 컴포넌트에서는 componentWillMount, componentDidMount, componentDidUpdate, componentWillUnmount를 컴포넌트 당 한 번씩만 사용했다면,  
useEffect는 데이터의 개수에 따라 여러 번 사용하게 됩니다.


메모리 누수를 방지하기 위해 이전 이벤트를 정리해야할 경우 return 안에 함수 호출

```javascript
useEffect(() => {
  const subscription = props.source.subscribe();
  return () => {
    // Clean up the subscription
    subscription.unsubscribe();
  };
});
```

```console
Unhandled Runtime Error
Error: Rendered fewer hooks than expected. This may be caused by an accidental early return statement.
```

```javascript
 useEffect(() => {
        if (!(me && me.id)) {
            alert('로그인이 필요합니다.');
            Router.push('/')
        };
    }, [me && me.id]);
    if (!me) { return null };
    const loadMoreFollowings = useCallback(() => { }, []);
    const loadMoreFollowers = useCallback(() => { }, []);
```
이런 경우 만약 !me인 경우(로그인 하지 않은 경우) 뒤의 useCallback함수들이 실현되지 못한다.
같은 수의 함수들이 실행되지 않으면 위와 같은 에러가 발생한다


```javascript
 const loadMoreFollowings = useCallback(() => { }, []);
    const loadMoreFollowers = useCallback(() => { }, []);
 useEffect(() => {
 
        if (!(me && me.id)) {
            alert('로그인이 필요합니다.');
            Router.push('/')
        };
    }, [me && me.id]);
    if (!me) { return null };
   
```
return이 가장 아래 오도록 하면 에러가 해결된다

출처:zeroCho blog <https://www.zerocho.com/category/React/``>,  

react Docs <https://reactjs.org/docs/hooks-reference.html#useeffect>

