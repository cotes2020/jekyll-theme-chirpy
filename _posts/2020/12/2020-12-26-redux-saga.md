---
title: "redux-saga管理异步"
url: "2020-12-26-redux-saga"
date: 2020-12-26
---

## redux-saga 是做什么？


redux-saga 基于 yield 语法，能够创建+管理更加复杂的异步操作。


比如有时候点击前端按钮，发起异步请求。为了防止频繁点击请求，需要进行节流防抖。除了可以在发起请求的时候，进行节流防抖。**还可以把节流防抖的时机提前，在状态更新的时候节流防抖**。


怎么理解呢？就是一定时间内的点击，只考虑最近一次的点击。最近这次点击才会触发回调函数，发起异步，更新状态。


redux-saga 就提供了很多这样的操作符来控制更复杂的异步流程。


## redux-saga 的简单使用


先看代码，假设要更新用户状态：


`sagas.js`文件，定义 saga：


```typescript
/*
 * @Author: dongyuanxin
 * @Date: 2020-12-25 18:51:11
 * @Github: https://github.com/dongyuanxin/blog
 * @Blog: https://0x98k.com/
 * @Description: 定义saga
 */
// effects中，有处理异步副作用的操作符
const { put, takeLatest, delay, call } = require("redux-saga/effects");

/**
 * 异步请求函数：这里简单模拟拿到用户数据
 */
function* fetchUser(userId) {
    yield delay(100); // 模拟异步
    return {
        sex: 1,
        userId,
    };
}

/**
 * saga函数：负责用户异步数据的状态维护
 */
function* watchUserFetchRequested() {
    console.log("watch action.type: USER_FETCH_REQUESTED");
    yield takeLatest("USER_FETCH_REQUESTED", function* (action) {
        console.log("invoke action.type: USER_FETCH_REQUESTED");
        try {
            // 通过call调用异步函数，第一个参数是函数，之后都是函数参数
            const userInfo = yield call(fetchUser, action.payload);
            yield put({ type: "USER_FETCH_SUCCEEDED", payload: userInfo });
        } catch (error) {
            yield put({ type: "USER_FETCH_FAILED", payload: error.message });
        }
    });
}

module.exports = { watchUserFetchRequested };
```


`index.js`文件，使用 saga：


```typescript
/*
 * @Author: dongyuanxin
 * @Date: 2020-12-25 18:42:57
 * @Github: https://github.com/dongyuanxin/blog
 * @Blog: https://0x98k.com/
 * @Description: redux-saga学习
 */
const { watchUserFetchRequested } = require("./sagas");
const { createStore, applyMiddleware } = require("redux");

// step1: 创建一个saga的中间件
const createSagaMiddleware = require("redux-saga").default;
const sagaMiddleware = createSagaMiddleware();

// 这只是一个普普通通的更新状态的 reducer
function startUserReq(state = {}, action) {
    switch (action.type) {
        // USER_FETCH_REQUESTED 是一个不改动 state 的 action.type
        // 它主要是用来触发 saga 函数中的事件监听
        case "USER_FETCH_REQUESTED":
            console.log("reducer USER_FETCH_REQUESTED");
            return state;
        case "USER_FETCH_SUCCEEDED":
            console.log("reducer USER_FETCH_SUCCEEDED");
            return action.payload;
        case "USER_FETCH_FAILED":
            console.log("reducer USER_FETCH_FAILED");
            return action.payload;
        default:
            return state;
    }
}

// step2: 使用saga中间件
const store = createStore(startUserReq, applyMiddleware(sagaMiddleware));

// step3: 调用run(),使用封装的saga的函数(watchUserFetchRequested)
// 在watchUserFetchRequested中，做了什么呢？
//  1、通过事件监听(effects/takeLatest函数)，监听action.type(USER_FETCH_REQUESTED)，然后发起请求
//  2、发起请求后，成功和创建则更新状态。在saga中不通过dispatch发起更新，通过effects/put发起更新
sagaMiddleware.run(watchUserFetchRequested);

// step4: 发起获取用户的异步请求
//  1、会先挨个触发reducer，和redux原生一样
//  2、触发完redcuer之后，由于step3.1中，saga中监听了USER_FETCH_REQUESTED，所以会执行回调函数
//  3、剩下的步骤和step3.2描述的一样
store.dispatch({ type: "USER_FETCH_REQUESTED" });
```


在注释中，展示了整体调用流程。除此之外，还有几点要注意：

- saga 是基于事件的（例如 take、takeLatest 等等）
- store.dispatch 还是会先触发 reducer，reducer 执行之后，才会触发 saga 的事件监听回调
- saga 中，通过 put 而不是 dispatch 来更新触发 reducer，更新状态
- sage 中，如果事件监听回调中，put 触发 reducer 传入的 action.type 和事件监听的 action.type 一样，就可能会陷入死循环

对于第 2 点的顺序，上述代码的输出是：


```shell
watch action.type: USER_FETCH_REQUESTED
reducer USER_FETCH_REQUESTED
invoke action.type: USER_FETCH_REQUESTED
reducer USER_FETCH_SUCCEEDED
```


对于第 4 点，代码换成以下的样子，就会死循环：


```typescript
function* watchUserFetchRequested() {
    console.log("watch action.type: USER_FETCH_REQUESTED");
    yield takeLatest("USER_FETCH_REQUESTED", function* (action) {
        console.log("invoke action.type: USER_FETCH_REQUESTED");
        try {
            const userInfo = yield call(fetchUser, action.payload);
            // 这里会触发reducer，action.type为USER_FETCH_REQUESTED
            // 然后外层事件监听又会监听到，死循环
            yield put({ type: "USER_FETCH_REQUESTED", payload: userInfo });
        } catch (error) {
            yield put({ type: "USER_FETCH_FAILED", payload: error.message });
        }
    });
}
```


## effects 深入学习


### 并发任务：all、race


前面多个`yield call(...)`是串行的，如果想并行怎么写呢？使用`all`操作符。


```typescript
const [users, repos] = yield all([
  call(fetchUser, { role: 'user' }),
  call(fetchStudent, { role: 'student' })
])
```


`effects/all`和`Promise.all`的行为类似，`effects/race`和`Promise.race`的行为类似。


### 异步任务：fork、spawan


前面`yield call(...)`是阻塞的，等待 call 中的异步任务完成后，才会向下执行。


如果想异步执行，那么需要使用`fork(...)`，返回异步标识，然后通过`effects/cancel`来取消。


上面的`sagas.js`改造下：


```typescript
function* fetchUser(userId) {
    yield delay(100);
    console.log(">>> 触发fetchUser");

    if (yield cancelled()) {
        // 如果fetchUser是异步任务，并且被取消了，这里可以捕获到
    }

    return {
        sex: 1,
        userId,
    };
}

function* watchUserFetchRequested() {
    yield takeLatest("USER_FETCH_REQUESTED", function* (action) {
        try {
            const task = yield fork(fetchUser, action.payload);
            console.log(">>> fork完成");
            // 调用 cancel 取消任务
            yield cancel(task);
            yield put({ type: "USER_FETCH_SUCCEEDED", payload: userInfo });
        } catch (error) {
            yield put({ type: "USER_FETCH_FAILED", payload: error.message });
        }
    });
}
```


上面代码输出是：


```shell
>>> fork完成
// 100ms后输出
>>> 触发fetchUser
```


可以调用`cancel(task)`，来取消 task 任务。这个和`setInterval`、`clearInterval`接口设计相似。


**那么 fork、spawan 有啥区别呢？**


这里借用操作系统的进程概念，fork 出来的任务会阻塞父任务；spawan 出来的任务不会阻塞父任务，同理，也不受父任务取消的影响。


### 事件处理：take、takeEvery、takeLatest


takeEvery、takeLatest 的区别好理解，就是响应 action.type，触发回调函数。


它们和 take 的区别呢？take 可以主动地等待用户操作；takeEvery 和 takeLatest 是被动的收到消息。


例如登录和登出的代码，用 take 可以写成：


```typescript
function* loginFlow() {
    while (true) {
        const { user, password } = yield take("LOGIN_REQUEST");
        // fork return a Task object
        const task = yield fork(authorize, user, password);
        const action = yield take(["LOGOUT", "LOGIN_ERROR"]);
        if (action.type === "LOGOUT") yield cancel(task);
        yield call(Api.clearItem("token"));
    }
}
```


如果用 takeLatest，则写成：


```typescript
function* watchLoginRequest() {
    yield takeLatest("LOGIN_REQUEST", function* (action) {
        // 进行登录
    });
}

function* watchLoginError() {
    yield takeLatest("LOGIN_ERROR", function* (action) {
        // 登录出错
    });
}

function* watchLogout() {
    yield takeLatest("LOGOUT", function* (action) {
        // 登出
    });
}
```


## 参考

- [redux-saga 中文文档](https://redux-saga-in-chinese.js.org/): 有点旧，但流畅
- [redux-saga docs](https://redux-saga.js.org/): 以英文文档为准

