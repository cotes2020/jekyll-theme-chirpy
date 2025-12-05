---
title: "React父子组件实现：Form(Item)/Tab(Item)"
date: 2023-01-14
permalink: /2023-01-14-children-comp/
---
## Form(Item)


参考：[https://juejin.cn/book/6945998773818490884/section/6950659615675645990](https://juejin.cn/book/6945998773818490884/section/6950659615675645990)


### 示例demo


```typescript
export default  () => {
    const form =  React.useRef(null)
    const submit =()=>{
        /* 表单提交 */
        form.current.submitForm((formValue)=>{
            console.log(formValue)
        })
    }
    const reset = ()=>{
        /* 表单重置 */
        form.current.resetForm()
    }
    return <div className='box' >
        <Form ref={ form } >
            <FormItem name="name" label="我是"  >
                <Input   />
            </FormItem>
            <FormItem name="mes" label="我想对大家说"  >
                <Input   />
            </FormItem>
        </Form>
        <div className="btns" >
            <button className="searchbtn"  onClick={ submit } >提交</button>
            <button className="concellbtn" onClick={ reset } >重置</button>
        </div>
    </div>
}
```


### Form


Form 目标：

- 基于class component实现
- 负责管理 form store（表单数据存储对象）
- 能够识别 children 上的 FormItem 组件，并且只渲染它们
- 内置一些方法，外部能通过 ref 拿到，并且进行表单提交、验证等操作

实现：

- 无
- 内置一个object，或者搞个专门的class都行
- 通过FormItem的displayName或者其它标识，来判断是否为 FormItem；

	并且重写 children 节点，只拿出来 FormItem，并且将改变 FormItem 绑定的表单字段的值的方法传给 FormItem。

- 基于 class 实现的，外界使用时ref可以直接拿到 class 上的方法。内部支持表单提交、验证等操作即可。

### FormItem


FormItem 目标：

- 基于class component实现
- 能识别内置的 Input / Checkbox / Radio 等组件，并且在状态改变时，将改变传给 Form

实现：

- 无
- 给 Input  / Checkbox / Radio 等组件挂上 dispalyName 标识属性

## Tab(Item)


### 示例


```typescript
<TabButtonList
  onChange={value => {
	  setTab(value);
	  submit();
}}
  activeKey={tab}
>
  <TabButton key={TAB_TYPE.ALL}>{TAB_MAP[TAB_TYPE.ALL]}</TabButton>
  <TabButton key={TAB_TYPE.INWORK}>{TAB_MAP[TAB_TYPE.INWORK]}</TabButton>
  <TabButton key={TAB_TYPE.HASWORK}>{TAB_MAP[TAB_TYPE.HASWORK]}</TabButton>
</TabButtonList>
```


### 区别


和Form、FormItem相比，行为没有那么复杂，同时也不需要像FormItem那样，内部还得判断Input、Select等组件。


### TabItem实现

1. 响应点击选中样式
2. 相应点击事件
3. 需要通过 `displayName` 挂入标识，方便父级 `TabButtonList` 判断
<details>
<summary>代码如下：</summary>

```typescript
import React from 'react';
import classnames from 'classnames';
import styles from './index.module.scss';
import { TabButtonProps, TabButtonListProps } from './types';

const TYPE_NAME = {
  tabButton: '__tab_button',
} as const;

/**
 * 用于标签切换的按钮
 */
export const TabButton = React.memo((props: TabButtonProps) => {
  const { children, clicked } = props;

  /**
   * 渲染子元素
   * @returns
   */
  const renderChildren = () => {
    if (typeof children === 'function') {
      return children();
    }
    return children;
  };

  return (
    <div
      className={classnames(styles.tabButton, {
      [styles.active]: clicked,
    })}
      onClick={(props as any)?.onClick || (() => {})}
    >
      {renderChildren()}
    </div>
  );
});
TabButton.displayName = TYPE_NAME.tabButton;
```


</details>


### TabButtonList

1. 遍历所有的子元素，挑选 `TabButton` 组件，跳过其他组件
2. 根据外面传入的 `activeKey` ：判定当前的 `TabButton` 组件是否为选中状态，并且选中组件跳过 `click` 回diao
<details>
<summary>代码如下</summary>

```typescript
/**
 * 标签切换的按钮容器。可自动识别底层的切换按钮组件。
 */
export const TabButtonList = React.memo((props: TabButtonListProps) => {
  const { className, children, activeKey, onChange, style } = props;
  /**
   * 动态计算子组件
   */
  const childList: ReturnType<typeof React.cloneElement>[] = [];
  React.Children.forEach(children, (child: any) => {
    if (child.type.displayName === TYPE_NAME.tabButton) { // 判断是否为 TabButton 组件
      const { props: { children: cc }, key: cKey } = child; // 读取子组件的key，以及从props上读取子组件的children
      const childCopy = React.cloneElement(child, {
        key: cKey,
        clicked: cKey === activeKey,
        onClick: () => {
          if (typeof onChange === 'function' && cKey !== activeKey) { // 对于已选中的标签，不触发click回调
            onChange(cKey);
          }
        },
      }, cc);
      childList.push(childCopy);
    }
  });

  return (
    <div className={className} style={style}>
      {childList}
    </div>
  );
});
```


</details>


