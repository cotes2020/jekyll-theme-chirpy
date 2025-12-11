---
title: "React Hooks 实现双向绑定"
date: 2023-02-16
permalink: /2023-02-16-react-hooks-bind/
categories: ["B源码精读", "React"]
tags: ["函数式编程", "双向绑定设计"]
---

## 双向绑定


Vue.js 就是典型的双向绑定设计。除此之外，新的框架比如 Solid.js 、Preact.js，以及第三方库 ahooks、redux/toolkit、mbox 都采用了双向绑定设计。


## 特点与好处

- 无需开发者手动调用 `setState` 函数来触发底层视图的更新
- 开发者只需像写JS代码那样，对值进行增删改查即可，由框架来监听value更新，触发视图更新

## 实现原理

- 需要对数据进行「劫持」，从而能够修改默认的行为。这就需要通过 `Proxy` 来代理，外界不是操作对象，而是操作被代理后的对象
- 需要对「缓存」被代理的对象，使其在React组件生命周期不被重复代理。这就需要配合 `ref`。
- 在 `set` 和 `delete` 属性的时候，需要内部自动触发视图更新。配合 React Hook 的更新原理，创建一个强制更新的 Hook。
- 在 `get` 属性时，如果被访问的值是复杂对象，那么就需要访问其代理（和第一步一样），而不是直接返回它。

	不要一开始就采用 `DFS` 给对象的所有深层属性都挂上代理，影响性能。


## 实现代码


```typescript
const proxyMap = new WeakMap();

const observer = (initialState, cb) = >{
    const existing = proxyMap.get(initialState);
    if (existing) return existing;
    const proxy = new Proxy(initialState, {
        get(target, key, receiver) {
            const val = Reflect.get(target, key, receiver);
            return typeof val === "object" && val !== null ? observer(val, cb) : val; // 递归处理object类型
        },
        set(target, key, val) {
            const ret = Reflect.set(target, key, val);
            cb() return ret;
        },
        deleteProperty(target, key) {
            const ret = Reflect.deleteProperty(target, key);
            cb();
            return ret;
        },
    });
    return proxyMap.set(initialState, proxy) && proxy;
};

function useReactive(initialState) {
    const refState = useRef(initialState);
    const[, setUpdate] = useState({});
    const refProxy = useRef({
        data: null,
        initialized: false,
    });
    if (refProxy.current.initialized === false) {
        refProxy.current.data = observer(refState.current, () = >{
            setUpdate({});
        });
        refProxy.current.initialized = true;
        return refProxy.current.data;
    }
    return refProxy.current.data;
}
```


## 参考资料

- ahooks 提供了 `useReactive` 钩子，可以直接使用

[bookmark](https://github.com/alibaba/hooks/blob/master/packages/hooks/src/useReactive/index.ts)

- 讲解文章

[bookmark](https://blog.csdn.net/Android062005/article/details/124836176)


