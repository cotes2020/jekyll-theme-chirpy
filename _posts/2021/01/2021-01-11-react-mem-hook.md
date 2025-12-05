---
title: "React 性能优化 Hooks：React.memo & useMemo"
url: "2021-01-11-react-mem-hook"
date: 2021-01-11
---

_**Hook 实现原理请参考**_ [一文彻底搞懂react hooks的原理和实现](https://www.notion.so/a3380898e55e49b98a7ec7aae960cb0a) 


## 原：不使用 React.memo 和 useMemo


```typescript
function C() {
    console.log(">>> C: render");
    return <div>C</div>;
}

function B() {
    console.log(">>> B: render");
    return <C />;
}

function App() {
    const [arr, setArr] = useState([1, 2, 3]);
    const [times, setTimes] = useState(1);

    useEffect(() => {
        console.log(">>> App: render");
    });

    useEffect(() => {
        setTimeout(() => {
            console.log(">>> App: update arr");
            setArr([1, 2, 3, 4]);
        }, 500);
    }, []);

    return (
        <div>
            <B arr={times} />
            <br />
            arr is: {arr}
        </div>
    );
}

const rootElement = document.getElementById("root");
ReactDOM.render(<App />, rootElement);
```


上面的 demo 没有使用 useMemo 和 React.useMemo，纯函数组件。那么当 500ms 后，更新 arr 状态后，虽然传递给组件 B 的 times 属性没变，但组件 B 也会重新渲染。


控制台输出：


```shell
>>> B: render
>>> C: render
>>> App: render
>>> App: update arr
>>> B: render
>>> C: render
>>> App: render
```


**为什么 B 和 C 会输出 2 次“render”相关信息？“**


App 状态更新后，对于其子组件以及子组件的子组件（依次类推），react 会进行重新 render，然后进行 Diff DOM 算法比较，再决定是否更新对应的 DOM 结构


关键点就在于，react 需要重新进行 render，才能和之前的 dom 进行比较。所以触发了所有下层组件的渲染。


## 对比 1: 使用 React.memo()


**怎么去解决任意状态更新，都会造成子函数组件的重新执行呢？**


使用 React.memo(component, equqlFunction) 来包装子组件，并且可以定义属性比较函数。


当 equqlFunction 返回 true，代表前后组件属性相同，不会重新执行执行函数组件；返回 false，会重新执行。


**关于 React.memo()：**


1、功能和 class 组件的 shouldComponentUpdate 方法类似，自定义属性比较函数，避免组件的重复渲染


2、使用 react.memo()包装的组件，如果不传入 equqlFunction，默认是浅比较。
传入 equqlFunction 函数，来代替 react 的浅比较，自定义 prevProps 和 nextProps 的比较。


**代码示例**：


假设 A 使用了 B(被 React.memo 包装过)，B 使用了 C。


当在 500ms 后，组件 A 中调用 setArr([1, 2, 3]) 更新 arr 时，会与原来的 arr 进行比较。


由于使用了 react.memo()，并且没有使用默认的浅比较，所以不会触发 B 的重新执行。


```typescript
/*
 * @Author: dongyuanxin
 * @Date: 2021-01-06 00:18:34
 * @Github: https://github.com/dongyuanxin/blog
 * @Blog: https://xin-tan.com/
 * @Description: React.memo() 使用
 */
/**
 * 比价数组1和数组2是否相同
 */
function compareArray(arr1, arr2) {
    if (!Array.isArray(arr1) || !Array.isArray(arr2)) {
        return false;
    }

    if (arr1.length !== arr2.length) {
        return false;
    }

    const everyEqual = arr1.every((_, index) => {
        return arr1[index] === arr2[index];
    });

    return everyEqual;
}

function C() {
    console.log(">>> C: render");
    return <div>C</div>;
}

function B({ arr }) {
    console.log(">>> B: render");
    console.log(">>> B: props.arr is", arr);
    return <C />;
}

const BComponent = React.memo(B, (prevProps, nextProps) => {
    if (compareArray(prevProps.arr, nextProps.arr)) {
        return true;
    }
    return false;
});

function App() {
    const [arr, setArr] = useState([1, 2, 3]);

    useEffect(() => {
        console.log(">>> App: render");
    });

    useEffect(() => {
        setTimeout(() => {
            console.log(">>> App: update arr");
            setArr([1, 2, 3]);
        }, 500);
    }, []);

    return (
        <div>
            <BComponent arr={arr} />
        </div>
    );
}

const rootElement = document.getElementById("root");
ReactDOM.render(<App />, rootElement);
```


输出是：


```typescript
>>> B: render
>>> C: render
>>> App: render
>>> App: update arr
>>> App: render
```


如果直接使用 B，而不是 React.memo()包装后的 B 组件；或者使用 React.memo()包装的 B，但是不传入第二个比较函数。那么结果输出和上面的输出是一样的。


**原因分析**：


2 种情况组件 B 都会重新执行。


第 1 种，是因为父组件状态更新，会重新执行子组件逻辑。


第 2 种，虽然会进行比较判断，再决定是否重新执行子组件逻辑，但是默认比较判断函数是浅比较。


## 对比 2: 使用 useMemo()


**关于 useMemo()：**


除了 React.memo()，还可以使用 useMemo Hooks，在函数内部自己控制子组件。


useMemo 的第二个参数和 useEffect 第二个参数类似，只有其中的值发生变化时，才会重新生成组件。


这里比较值的变化，采用的也是默认的浅比较。


**和 React.memo()的区别：**


除了控制组件的角度不同，关于 useMemo 无法像 React.memo()那样，自定义比较函数。所以对于复杂对象（例如数组），浅比较结果是变化的，那么就会导致重新生成组件。


**代码示例：**


```typescript
/*
 * @Author: dongyuanxin
 * @Date: 2021-01-07 20:14:21
 * @Github: https://github.com/dongyuanxin/blog
 * @Blog: https://xin-tan.com/
 * @Description: useMemo()
 */

function C() {
    console.log(">>> C: render");
    return <div>C</div>;
}

function B() {
    console.log(">>> B: render");
    return <C />;
}

function App() {
    // 缺点：无法像React.memo()那样，自定义比较函数。
    //     所以对于复杂对象（例如数组），浅比较结果是变化的，那么就会导致重新生成组件。
    //     例如下面代码，B: render 会输出多次

    // const [arr, setArr] = useState([1, 2, 3]);
    // const MemoB = useMemo(() => <B arr={arr} />, [arr]);

    const [times, setTimes] = useState(1);
    const MemoB = useMemo(() => <B times={times} />, [times]);

    useEffect(() => {
        console.log(">>> App: render");
    });

    useEffect(() => {
        setTimeout(() => {
            console.log(">>> App: update arr");
            setTimes(1);
        }, 500);
    }, []);

    return <div>{MemoB}</div>;
}

const rootElement = document.getElementById("root");
ReactDOM.render(<App />, rootElement);
```


**上述代码输出：**


```text
>>> B: render
>>> C: render
>>> App: render
>>> App: update arr
```


## 参考链接

- [十个案例学会 React Hooks](https://zhuanlan.zhihu.com/p/60925430)
- [Web 性能优化](https://segmentfault.com/a/1190000018444604)
- [React.memo 与 useMemo](https://zhuanlan.zhihu.com/p/105940433)

