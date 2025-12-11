---
title: "磁贴布局 react-grid-layout"
date: 2022-12-20
permalink: /2022-12-20-grid-layout/
categories: ["B源码精读", "React"]
tags: [组件开发, 碰撞检测]
---

## 描述


![Untitled.png](https://raw.githubusercontent.com/dongyuanxin/static/main/blog/imgs/2022-12-20-grid-layout/8f050d14ac580affd39981411e049c9a.png)


![Untitled.png](https://raw.githubusercontent.com/dongyuanxin/static/main/blog/imgs/2022-12-20-grid-layout/7e95b0b9ed8b53187f5754143dfc8784.png)


## 使用方法


直接使用 `react-grid-layout` 即可。使用方式简单，直接定义组件的 `layout` 即可，然后将其将给 `react-grid-layout` 渲染：


在低代码引擎，或者支持前端拖拽的前端应用中，额外在前端更新Schema、在后端存储每个组件的 i、x、y、w、h 这5个属性即可。


**同时，这个更新和存储Schema的Action，可以将其封装到一个** **`RGLContainer`** **容器组件中。**代码中，不直接使用 `react-grid-layout` 而是直接使用 `RGLContainer` 容器组件。


## 实现原理


### 记录组件和容器


布局组件首先要收集到有哪些可拖拽组件与容器，假设业务层将这些 DOM 生成好传给了布局：


```typescript
const elementMap: Record<
  string,
  {
    dom: HTMLElement;
    x: number;
    y: number;
    width: number;
    height: number;
  }
> = {};
const containerMap: Record<
  string,
  {
    dom: HTMLElement;
    rectX: number;
    rectY: number;
    width: number;
    height: number;
  }
> = {};
```

- `elementMap` 表示可拖拽的组件信息，包括其 DOM 实例，以及相对于父容器的 `x`、`y`、`width`、`height`。
- `containerMap` 表示容器组件信息，之所以存储 `rectX` 与 `rectY` 这两个相对浏览器绝对定位，是因为容器的直接父组件可能是 `element`，比如 `Card` 组件可以同时渲染 `Header` 与 `Footer`，这两个位置都可以拖入 `element`，所以这两个位置都是 `container`，它们是相对父 `element` `Card` 定位的，所以存储绝对定位方便计算。

### 拖拽行为


考虑到兼容性，整体可以用鼠标 `mousedown/mousemove/mouseup`等事件模拟拖拽。


给每个 `elementMap` 的每个组件都绑定对应的 `onmousedown` 事件，作为拖拽开始的时机。


```typescript
Object.keys(elementMap).forEach((componentId) => {
  elementMap[componentId].dom.onmousedown = () => {
    // 记录拖拽开始
  };
});
```


拖拽行为分3种：

- 开始拖拽：记录拖拽的组件。
- 拓拽中：监听 `document` 的 `mousemove` ，检测context上是否有拖拽组件，如果有，则进入主体逻辑。
- 结束拖拽：监听  `document` 的 `onmousedown` ，将 context 上拖拽组件挂成 null 。

```typescript
function onDragStart(context, componentId) {
  context.dragComponent = componentId;
}

function onDrag(context, event) {
  // 根据 context.dragComponent 响应组件的拖动
  // 将 element x、y 改为 event.clientX、event.clientY 即可
}

function onDragEnd(context) {
  context.dragComponent = undefined;
}
```


### 位置计算


**在网格布局中，计算位置和元素大小的基本单位不是px，而是网格栅栏。**比如，12栏网格布局，那么每一栏的宽度等于：容器物理像素 / 12。


而组件位置、大小一定等于一栏宽度的N倍，这也为计算带来方便。比如，用户在将组件从5号栅栏，横向拖入10号栅栏时，组件中心移动的长度，一定等于 5x一栏的宽度。


![Untitled.png](https://raw.githubusercontent.com/dongyuanxin/static/main/blog/imgs/2022-12-20-grid-layout/aa407175eb7e75d89c4a3217ffd8cae7.png)


### 非碰撞拖拽


定义：被拖组件，不和其它组件发生碰撞。


举例：上图3号被向右拖拽，拖拽后，3号向上吸附，底下的组件均向上吸附。


处理逻辑：

- x 坐标计算：根据拖拽后的坐标，找到其在网格布局所属的栅栏位置。x的计算比较简单，跟着鼠标走。
- y 坐标计算：
	- 根据x轴的「重叠关系」，找到新位置下的3号组件上方的所有组件。
	- 对这些组件的y轴进行排序，找到最下面组件（5号组件）的bottom边所处的y坐标，即为3号组件的top边的y坐标。
- 目标位置渲染：根据x和y，渲染目标位置，即红色方块；当用户松手后，3号组件的位置就会被修改到红色方块。
- 吸附渲染：
	- 遍历找到3号组件下面的所有组件，并且从大到小排序
	- 从大到小，依次重新计算每个组件的y坐标（此时忽略原来的3号组件，将红色方块所处的目标位置，作为3号组件位置）

### 碰撞检测


![Untitled.png](https://raw.githubusercontent.com/dongyuanxin/static/main/blog/imgs/2022-12-20-grid-layout/32e58a8eb9d9f3fd12412e2403ae12f3.png)


定义：移动过程中，与其它组建碰撞。


举例：11号组件向左下方移动，移动后，左侧组件均被向下挤；右侧组件向上吸附。


处理逻辑：

- 碰撞检测：得到移动后的最新位置，遍历所有元素，检测最新位置矩形和任一元素位置矩形是否相交。
	- 矩形不相交：不碰撞，处理逻辑回退「非碰撞检测」
	- 矩形相交：碰撞
- x 坐标计算：同「非碰撞检测-x 坐标计算」
- y 坐标计算：
	- 获取移动方向 ⬆️ or ⬇️
	- 如果是⬆️：
		- 按照「非碰撞检测-y坐标计算」找到碰撞组件的上方的组件，以及它的bottom
		- 此bttom配合组件的高度，就能计算出组件的y坐标
	- 如果是⬇️：
		- 移动到碰撞元素底部，也就是碰撞组件就是上方组件
		- 碰撞组件的bottom配合组件的高度，就能计算出组件的y坐标
- 目标位置渲染：同「非碰撞检测」
- 吸附渲染：同「非碰撞检测」

## 参考链接


[bookmark](https://developer.aliyun.com/article/883433)


[bookmark](https://github.com/ascoders/weekly/blob/master/%E5%89%8D%E6%B2%BF%E6%8A%80%E6%9C%AF/266.%E7%B2%BE%E8%AF%BB%E3%80%8A%E7%A3%81%E8%B4%B4%E5%B8%83%E5%B1%80%20-%20%E5%8A%9F%E8%83%BD%E5%AE%9E%E7%8E%B0%E3%80%8B.md)


