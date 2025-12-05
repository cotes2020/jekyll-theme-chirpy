---
title: "styled-components 工作原理"
date: 2023-03-16
permalink: /2023-03-16-style-components/
---
## 写法demo


```typescript
const Button = styled.button`
  color: coral;
  padding: 0.25rem 1rem;
  border: solid 2px coral;
  border-radius: 3px;
  margin: 0.5rem;
  font-size: 1rem;
`;
// 以上写法和下面写法是等效的，上面的写法是JS支持的快捷方法
const Button = styled('button')([
  'color: coral;' +
  'padding: 0.25rem 1rem;' +
  'border: solid 2px coral;' +
  'border-radius: 3px;' +
  'margin: 0.5rem;' +
  'font-size: 1rem;'
]);
```


## 原理分析


### 生成带有样式的标签


难点在于理解这种看起来很高级的写法，本质上就是函数调用。


```typescript
const myStyled = (TargetComponent) => ([style]) => class extends React.Component {
  componentDidMount() {
    this.element.setAttribute('style', style);
  }
  render() {
    return (
      <TargetComponent {...this.props} ref={element => this.element = element } />
    );
  }
};

const Button = myStyled('button')`
  color: coral;
  padding: 0.25rem 1rem;
  border: solid 2px coral;
  border-radius: 3px;
  margin: 0.5rem;
  font-size: 1rem;
`;
```


### 支持获取props


难点在于如何识别带有动态参数的写法。


```typescript
const myStyled = (TargetComponent) => (strs, ...exprs) => class extends React.Component {
  interpolateStyle() {
    const style = exprs.reduce((result, expr, index) => {
			// 可以判断每一行是不是函数，如果是函数，那么就将compoent的props传递给它
      const isFunc = typeof expr === 'function';
      const value = isFunc ? expr(this.props) : expr;

      return result + value + strs[index + 1];
    }, strs[0]);

    this.element.setAttribute('style', style);
  }

  componentDidMount() {
    this.interpolateStyle();
  }

  componentDidUpdate() {
    this.interpolateStyle();
  }

  render() {
    return <TargetComponent {...this.props} ref={element => this.element = element } />
  }
};

const primaryColor = 'coral';

const Button = myStyled('button')`
  background: ${({ primary }) => primary ? primaryColor : 'white'};
  color: ${({ primary }) => primary ? 'white' : primaryColor};
  padding: 0.25rem 1rem;
  border: solid 2px ${primaryColor};
  border-radius: 3px;
  margin: 0.5rem;
  font-size: 1rem;
`;
```


### 计算类名和css


前面是把style直接赋给了元素。实际上，这里是给元素挂入了一个hash后的className，classname对应的样式是style。


具体流程如下：

- 解析具体的 style，比如前面的 `interpolateStyle` 函数
- 利用 murmurhash 算法，将 style + 一个全局递增的id拼接传入，生成对应className
- 利用 [stylis](https://github.com/thysultan/stylis) 提取对应的 css 。传入的参数就是 className 以及 style，会自动生成css的标准写法
- 将css注入到页面的 `<style>` 标签中

	```typescript
	// 往 head 中插入 css
	function insertCss(css: string, index = 0) {
	  const { styleSheets } = document;
	  styleSheets[0].insertRule(css, index);
	}
	```

- 讲classname给到元素上即可

## 参考文档

- [https://juejin.cn/post/6905166914234875911](https://juejin.cn/post/6905166914234875911)
- [https://github.com/thysultan/stylis](https://github.com/thysultan/stylis)
- [https://juejin.cn/post/7040229858189770782#heading-6](https://juejin.cn/post/7040229858189770782#heading-6)

