---
title: "处理Redux异步状态更新"
date: 2020-12-25
permalink: /2020-12-25-redux-async/
tags: [状态管理, 异步编程]
---

## 纯 redux 如何处理异步状态


按照 redux 的编程方式，其实可以不借助任何中间件实现异步状态更新。


假设是异步读取文件，大概代码如下：


```typescript
dispatch({ type: "START_REQUEST" });
fs.readFile(filename, (err, content) => {
    if (err) {
        return dispatch({ type: "ERROR_REQUEST", payload: err });
    }
    return dispatch({
        type: "FINISH_REQUEST",
        payload: content.toString("utf-8"),
    });
});
```


在开始前，触发 START_REQUEST，成功则触发 FINISH_REQUEST，失败触发 ERROR_REQUEST。


但是这么写的问题有 2 个：

- 无法复用，没封装函数
- 即使封装函数，函数入参必须传入`store.dispatch`，函数内部才能调用 dispatch 更新状态

## redux-thunk：支持 dispatch 传入函数


使用 redux-think，就可以在 dispatch 中传入一个函数。那么就解决了前面说的 2 个问题。


完整代码如下：


```typescript
/*
 * @Author: dongyuanxin
 * @Date: 2020-12-25 14:24:57
 * @Github: https://github.com/dongyuanxin/blog
 * @Blog: https://0x98k.com/
 * @Description: redux-thunk处理异步action
 */

const fs = require("fs");
const { createStore, applyMiddleware, combineReducers } = require("redux");
const thunkMiddleware = require("redux-thunk").default;
const { createLogger } = require("redux-logger");
const loggerMiddleware = createLogger();

// step1: 定义action.type

// step2: 定义reducer
// 第一步和第二步，与同步状态更新都一样

function setNum(state = 0, action) {
    switch (action.type) {
        case "INCR":
            return state + 1;
        case "DECR":
            return state - 1;
        default:
            return state;
    }
}

function setReadFile(
    state = {
        status: null,
        data: null,
        error: null,
    },
    action
) {
    switch (action.type) {
        case "START_REQUEST":
            return { ...state, status: "pending" };
        case "FINISH_REQUEST":
            return { ...state, status: "resolved", data: action.payload };
        case "ERROR_REQUEST":
            return { ...state, status: "rejected", error: action.payload };
        default:
            return state;
    }
}

// step3: 定义异步action
//  异步action是个函数，借助redux-thunk，可以在dispatch中传入函数（异步action）。
//  在函数中，通过dispatch触发状态更新。

// 对比：在没有redux-thunk的时候，dispatch只能传入对象。
function readFile(filename) {
    return (dispatch) => {
        dispatch({ type: "START_REQUEST" });
        fs.readFile(filename, (err, content) => {
            if (err) {
                return dispatch({ type: "ERROR_REQUEST", payload: err });
            }
            return dispatch({
                type: "FINISH_REQUEST",
                payload: content.toString("utf-8"),
            });
        });
    };
}

// step4: 通过applyMiddleware引入中间件，支持dispatch()传入函数
const store = createStore(
    combineReducers({
        num: setNum,
        readFile: setReadFile,
    }),
    // 使用中间件，参考koajs的中间件
    applyMiddleware(
        thunkMiddleware,
        loggerMiddleware // loggerMiddler放在最后，在更新状态后，打印action、prevState、nextState，方便调试
    )
);

store.dispatch({ type: "INCR" });
// step5: 触发读取文件的action
store.dispatch(readFile(`${process.cwd()}/package.json`));
```


## 参考

- [redux 中文文档](https://www.redux.org.cn/docs/basics/Actions.html)

