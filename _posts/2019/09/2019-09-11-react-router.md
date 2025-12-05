---
title: "React Router 实战技巧"
date: 2019-09-11
permalink: /2019-09-11-react-router/
---
本文介绍在工程中经常用到的 react-router 的技巧：

- 🤔️ 如何在 TypeScript 中使用？
- 🤔️`exact`和`strict`的区别？
- 🤔️ 如何封装路由配置组件？
- 🤔️ 如何响应路由变化?

## 如何在 TypeScript 中使用？


有时候会需要使用编程式导航，比如上方导航栏里面选项，响应按钮事件，进行路由跳转。react 的做法是通过高阶函数，函数体内部向组件的 props 注册一些路由的方法，最后返回一个新的组件。


下面是一个结合 TypeScript 使用 withRouter 的例子：


```typescript
interface NavigationState {
  routes: Array<{
    path: string;
    name: string;
    key: string;
  }>;
  selectedKey: string;
}
interface NavigationProps {
  name: string;
}
class Navigation extends Component<
  RouteComponentProps & NavigationProps, // 使用「交叉类型」来处理Props的关系
  NavigationState
> {
  state = {
    routes: [],
    selectedKey: "1"
  };
  toggleRoute = (event: ClickParam) => {
    this.props.history.push(path); // route的方法已经被注册到了Props上
  };
  render() {
      // ...
    );
  }
}
export default withRouter(Navigation);

```


## `exact`和`strict`的区别？


为了方便说明，假设路由为`/a`：

- 若将`exact`设置为 true，路由相同（包括有斜杠）即可匹配。路由`/a`可以和`/a/`、`/a`匹配。
- 若将`strict`设置为 true，路由相同（不包括斜杠）可匹配。路由`/a`可以和`/a`匹配，不能和`/a/`匹配。

两者相比，`strict`匹配更严格。但一般常将`exact`设置为 true。


## 如何封装路由配置组件？


可以直接使用 `react-router-config` 组件。也可以根据原理，简单封装，代码如下：


```typescript
import { Route, Switch, SwitchProps, RouteProps } from "react-router-dom";
function renderRoutes(params: {
    routes: RouteProps[];
    switchProps?: SwitchProps;
}) {
    const { switchProps, routes } = params;
    return (
        <Switch {...switchProps}>
            {routes.map((route, index) => (
                <Route
                    key={index}
                    path={route.path}
                    component={route.component}
                    exact={route.exact || true}
                    strict={route.strict || false}
                ></Route>
            ))}
        </Switch>
    );
}

```


假设我们的路由配置声明如下：


```text
import { RouteProps } from "react-router-dom";
const config: RouteProps[] = [
    {
        path: "/",
        component: HomePage
    },
    {
        path: "/user",
        component: UserPage
    }
];

```


直接将路由声明传给`renderRoutes()`即可，使用如下：


```typescript
import React, { Component } from "react";
import { BrowserRouter } from "react-router-dom";
const routes = renderRoutes({
    routes: config
});
class App extends Component {
    render() {
        return <BrowserRouter>{routes}</BrowserRouter>;
    }
}
export default App;
```


如此一来，再增加新的页面，仅需要修改路由配置即可。改造成本低，拆卸方便。


## 如何响应路由变化?


在 VueJS 技术栈中，vue-router 是提供路由响应的钩子函数，例如：`beforeEach`、`afterEach`等等。


但是在 React 中，react-router 并不提供相关的钩子函数。**那么如果有顶部导航栏，不同页面切换时，高亮不同的标签，那么应该怎么实现响应路由变化呢**？


首先即使是路由，在 React 中，它也是一个组件对象。因此，如果要更新试图，必须触发组件的 render。而触发组件的关键在于，props 发生改变。


第一步：需要使用`withRouter`来包装对应的组件，将路由的信息作为 props 注入组件，比如顶部导航栏。


第二步：下面是 React17 前后的简单例子。


React17 之前：


```typescript
import { withRouter, RouteComponentProps } from "react-router-dom";
class Navigation extends Component<RouteComponentProps, any> {
    state = {
        selectedPath: "/"
    };
    // 在componentWillReceiveProps中接受新的props
    // 决定是否更新state
    componentWillReceiveProps(nextProps: RouteComponentProps) {
        if (nextProps.location.pathname === this.props.location.pathname) {
            this.setState({ selectedPath: nextProps.location.pathname });
        }
    }
    render() {
        // 这里的render渲染，取决于state是否更新
        const { selectedPath } = this.state;
        return <div>导航栏选中信息：{selectedPath}</div>;
    }
}
export default withRouter(Navigation);
```


在 React17 之后，不推荐使用`componentWillReceiveProps`等不确定的生命周期。处理的思路是：`render()`返回的视图中，变量的变化依赖 props 属性的值。


```typescript
import { withRouter, RouteComponentProps } from "react-router-dom";
class Navigation extends Component<RouteComponentProps, any> {
    state = {
        paths: ["/", "/a"]
    };
    render() {
        const { pathname } = this.props.location;
        const { paths } = this.state;
        // 切换路由, this.props 上的属性会变化
        let selectedPath = "";
        paths.some(path => {
            if (path === pathname) {
                selectedPath = path;
                return true;
            }
            return false;
        });
        return <div>导航栏选中信息：{selectedPath}</div>;
    }
}
export default withRouter(Navigation);

```


