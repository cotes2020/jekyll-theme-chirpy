---
title: "React Hooks 实现状态管理"
date: 2023-02-03
permalink: /2023-02-03-react-hooks-state/
---

- 主要api：useContext和useReducer
- 用途：都是为了进行状态管理。
	- 一般useContext更常用。
	- 如果使用useReducer不如使用redux或者其他管理库提供的更高级的hook api。
- 坑点：根据官方文档 [Hook API 索引 – React](https://zh-hans.reactjs.org/docs/hooks-reference.html#usecontext) ，也就是用到context的组件，都会由于context的变化导致re-render（就离谱。。）

	![Untitled.png](https://raw.githubusercontent.com/dongyuanxin/static/main/blog/imgs/2023-02-03-react-hooks-state/bee094f5d04c3bcd016ad7b89d48953e.png)

- useContext的正确使用姿势：排除掉网上各类说法，[直接看 github 上作者的建议即可](https://github.com/facebook/react/issues/15156#issuecomment-474590693) ：
	- 【推荐】拆分context，保证context的变动不会影响过多组件
	- 使用 范式memo（React.memo && useMemo）
- 拆分context做法示例。


以[全局Log组件为例](https://www.51cto.com/article/630362.html?u_atoken=84610f16-99c7-408b-8e3a-b9f846387eba&u_asession=01znUZrh7C3idq64_q-ThjY_xYZvR0tdXB1bM8a8UaamyBUhGyphFRy-jZIl3EHPNPX0KNBwm7Lovlpxjd_P_q4JsKWYrT3W_NKPr8w6oU7K8nnLKg8ynsdYVoq0PbrC86Ue3R9QHfzEvknA4dzJmVTGBkFo3NEHBv0PZUm6pbxQU&u_asig=05FK9I6z1Cl1R4tzutay0zKTvfAScjsn9SNr5NrgBxHloIquIsuQxnYq5meqgKU1dYzciszZ-4s5g5YWHcxNmhke6cDpp6A9EKTssTYM8_My0mpNQOrRCcK5E6KXhACw-A0B8iNDCiwYonkEOD43qrfB1wWd0Uo_0AxwudsNZ8eEr9JS7q8ZD7Xtz2Ly-b0kmuyAKRFSVJkkdwVUnyHAIJzf71r16cZbeFVVtB9MmU55gpBem96xLqNn0FQq1FIJSlH_8T8uYGNepqxdb-gLe1IO3h9VXwMyh6PgyDIVSG1W_VHHHLSS3YReTPPaLLxcwIfX5y-uYeLDQishV-vt1GW0-rYI4ZS3jgdyFCUUhU2fN0Rr_BaKvGm8Uyij_HgnOQmWspDxyAEEo4kbsryBKb9Q&u_aref=do2ntTdKyx4MUR4eZE35G7u2ahE%3D)，实现「读写分离」：

- Before: 

定义方代码👇

```typescript
const LogContext = React.createContext();

function LogProvider({children }) {
  const [logs, setLogs] = useState([]);
  const addLog = (log) => setLogs((prevLogs) => [...prevLogs,log]);
  return (
    <LogContext.Providervalue={{ logs, addLog }}>
      {children}
    </LogContext.Provider>
  );
}
```

使用方代码：必须使用 LogContext 上下文。

- After: 

定义方代码👇

```typescript
const LogStateContext = React.createContext();
const LogDispatcherContext = React.createContext();

function LogProvider({children }) {
  const [logs, setLogs] = useState([]);
  const addLog = useCallback(log => {
    setLogs(prevLogs => [...prevLogs,log]);
  }, []);
  return (
    <LogDispatcherContext.Providervalue={addLog}>
      <LogStateContext.Providervalue={logs}>
        {children}
      </LogStateContext.Provider>
    </LogDispatcherContext.Provider>
  );
}
```

使用方代码：根据读、写的需要，只使用 LogStateContext、LogDispatcherContext 即可。
![Untitled.png](https://raw.githubusercontent.com/dongyuanxin/static/main/blog/imgs/2023-02-03-react-hooks-state/72f95e6b5c68751ac7b9e24884a03806.png)

- **范式memo的做法示例：**
	- 核心思路：
		- 给子组件包一层类似Wrapper的东西，在这个Wrapper内，从context上读取属性，并且将其作为prop传递给子组件。
		- 子组件使用React.memo或者 useMemo包裹
	- 效果：context的更新，只会造成Wrapper的re-render。由于它只是一层包装，性能损耗几乎为0。
	- React.memo 实现（如果你是上层业务开发者，想引用底层组件，并且将context作为prop传递过去）：

		```typescript
		function Button() {
		  const appContextValue = useContext(AppContext);
		  const { theme } = appContextValue;// Your "selector"  
			return <ThemedButton theme={theme} />;
		}
		
		const ThemedButton = React.memo(({ theme }) =>
			// The rest of your rendering logic  
			<ExpensiveTreeclassName={theme} />
		);
		```

	- useMemo实现（如果你是底层/通用组件开发者，不想让外层使用者每次使用时都用React.memo包裹一次那么麻烦）：

		```typescript
		function Button() {
		  const appContextValue = useContext(AppContext);
		  const { theme } = appContextValue;
			// Your "selector"
		  return useMemo(() =>
				// The rest of your rendering logic    
				<ExpensiveTreeclassName={theme} />
		  , [theme]);
		}
		```
