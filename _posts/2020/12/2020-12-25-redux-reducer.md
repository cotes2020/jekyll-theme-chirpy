---
title: "redux的reducer管理"
url: "2020-12-25-redux-reducer"
date: 2020-12-25
---

## 如何封装多个 reducer？


可以将多个 reducer 合并到一个 reducer 中。例如将`setTodoList`和`setNum`，都收归到`setAppState`中。


注意：

- setAppState 中的子 reducer 都会触发，所以 type 不要重复
- setAppState 中的子 reducer 和要更新的 key 一一对应。

```typescript
/*
 * @Author: dongyuanxin
 * @Date: 2020-12-25 23:39:48
 * @Github: https://github.com/dongyuanxin/blog
 * @Blog: https://0x98k.com/
 * @Description: 如何维护reducer
 */
const { createStore } = require("redux");

// action的types：type抽成常量，便于维护
const types = {
    ADD_TODO_LIST: "ADD_TODO_LIST",
    CLEAR_TODO_LIST: "CLEAR_TODO_LIST",
    INCR_NUM: "INCR_NUM",
    DECR_NUM: "DECR_NUM",
};

// reducer：负责更新 todoList 字段的reducer
function setTodoList(state = [], action) {
    switch (action.type) {
        case types.ADD_TODO_LIST:
            return [...state, action.payload];
        case types.CLEAR_TODO_LIST:
            return [];
        default:
            return state;
    }
}

// reducer：负责更新 num 字段的reducer
function setNum(state = 0, action) {
    switch (action.type) {
        case types.INCR_NUM:
            return state + 1;
        case types.DECR_NUM:
            return state - 1;
        default:
            return state;
    }
}

// reducer：全局的，收到一起统一管理
function setAppState(state = {}, action) {
    return {
        todoList: setTodoList(state.todoList, action),
        num: setNum(state.num, action),
    };
}

// state：初始状态
const initState = {
    todoList: [],
    num: 0,
};

const store = createStore(setAppState, initState);

store.subscribe(() => {
    console.log(">>> state is", store.getState());
});
// 通过 dispatch 触发更新：一般通过payload来携带要更新的数据
store.dispatch({ type: types.INCR_NUM });
store.dispatch({ type: types.ADD_TODO_LIST, payload: 1 });
```


## 利用 combineReducers 组织多个 reducer


前面组织多个 reducer，redux 也提供了`combineReducers()`方法来帮助开发者快速组合。


```typescript
const initState2 = {
    todoList: [],
    num: 0,
};

// 通过 combineReducers 组合 reducer
// key和reducer接收到的state参数，是一一对应的
const store2 = createStore(
    combineReducers({
        todoList: setTodoList,
        num: setNum,
    }),
    initState2
);

store2.subscribe(() => {
    console.log(">>> state2 is", store2.getState());
});

store2.dispatch({ type: types.INCR_NUM });
store2.dispatch({ type: types.ADD_TODO_LIST, payload: 1 });
```


