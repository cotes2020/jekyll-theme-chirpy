---
title: "HTML5原生拖放事件的学习与实践"
date: 2019-04-10
permalink: /2019-04-10-html5-drag-drop/
---

## 前言


之前学习了 HTML5 的拖放事件，开发中也用到了拖拽组件。为了厘清整体的逻辑，专门做了一个小例子。


具体实现的效果也很简单：元素可以在容器中任意拖动，元素被移入容器的时候，还会有相关样式的改变已达到更好的展示效果。


例子基本运用了拖放事件的全部事件，并且尽量简洁的展示了出来。特此记录。


## 拖放事件介绍


由名字可以看出来，拖放事件由 2 部分组成：拖动和释放。


而拖动又由 2 部分组成，分别是被拖动元素的相关事件和元素容器的相关事件。


**1、被拖动元素的相关事件** ：


child_database


**2、容器的相关事件** ：


child_database


**3、释放事件** ：


child_database


## 效果展示


为了方便说明，先看代码实现的效果。请前往  [Github 仓库](https://github.com/dongyuanxin/html5-drag-drop)  下载 `demo.html`  和 `demo.js`  到本地，然后用 Chrome 打开 html文件，初始效果如下图：


![name=image.png](https://cdn.nlark.com/yuque/0/2019/png/233327/1554824440934-6b099e29-5d03-47c8-9fcd-358dfac034ce.png#align=left&display=inline&height=489&name=image.png&originHeight=612&originWidth=329&size=8385&status=done&width=263)


将图中的可拖拽元素，拖放到下面的容器中，这个过程的效果如下所示。箭头表示拖拽方向，方框代表动态改变的容器样式。


![name=image.png](https://cdn.nlark.com/yuque/0/2019/png/233327/1554824491746-4b790630-bb53-479a-9f5d-eaafef60cb23.png#align=left&display=inline&height=501&name=image.png&originHeight=627&originWidth=364&size=15595&status=done&width=291)


最后，松开鼠标，将元素放入到下面的容器中，整个过程完成。


![name=image.png](https://cdn.nlark.com/yuque/0/2019/png/233327/1554824562321-b6b1a435-27b5-43ca-abb8-7e63265b839e.png#align=left&display=inline&height=477&name=image.png&originHeight=596&originWidth=315&size=8094&status=done&width=252)


## 代码实现


首先，先编写 html 代码。因为元素可以在两个容器之间任意拖动，因此这两个容器都需要监听 drapenter、dragover、dragleave、drop 这四个事件。


被拖拽元素的 `draggable`  属性需要指明为 `true` ，才可以被拖拽。同时为了记录一些信息，需要监听 dragstart 事件。


```html
<body>
    <script src="./demo.js"></script>
    <div
        class="container"
        ondragenter="onDragEnter(event)"
        ondragover="onDragOver(event)"
        ondragleave="onDragLeave(event)"
        ondrop="onDrop(event)"
    >
        <div id="target" draggable="true" ondragstart="onDragStart(event)">
            被拖拽元素
        </div>
    </div>
    <div
        class="container"
        ondragenter="onDragEnter(event)"
        ondragover="onDragOver(event)"
        ondragleave="onDragLeave(event)"
        ondrop="onDrop(event)"
    ></div>
</body>
```


为了让拖拽效果更明显，实现效果展示->第二部分的，拖拽元素进入一个新的容器的时候，新容器展示阴影效果。编写阴影效果样式：


```html
<style>
  .container {
    width: 200px;
    height: 200px;
    padding: 10px;
    border: 1px solid #aaaaaa;
    margin-bottom: 10px;
    transition: box-shadow .3s ease;
  }
  #target {
    width: 50px;
    height: 50px;
    border: 1px solid black;
    margin: 0 auto;
  }
  .container.active {
    border-bottom-width: 0;
    box-shadow: 0 10px 6px -6px #777;
  }
</style>
```


最后，编写 `demo.js`  代码。具体逻辑请看代码中的注释信息：


```typescript
let target = null,
    container = null;
// 寻找拖拽元素的容器类
function findParentContainer(node) {
    if (!node || node === document) {
        return null;
    }
    if (node.classList.contains("container")) {
        return node;
    }
    return findParentContainer(node.parentNode);
}
// 元素开始被拖拽时, 标记元素原生的容器类
function onDragStart(event) {
    target = event.target;
    container = findParentContainer(target);
}
// 元素进入目的容器时, 如果不是原来的容器, 则可以放置
// 此时更改样式, 以更好向用户展示
function onDragEnter(event) {
    event.preventDefault();
    if (event.target !== container) {
        event.target.classList.add("active");
    }
}
// 元素在目的容器内时触发
function onDragOver(event) {
    event.preventDefault();
}
// 元素离开目的容器, 需要移除相关样式
function onDragEnter(event) {
    event.preventDefault();
    event.target.classList.remove("active");
}
// 元素被放置在目的容器, 添加DOM节点, 移除相关样式
function onDrop(event) {
    event.preventDefault();
    event.target.appendChild(target);
    event.target.classList.remove("active");
    target = null;
    container = null;
}
```


## 参考链接

- 代码地址: [Github](https://github.com/dongyuanxin/html5-drag-drop)
- [《HTML5 拖放》](http://www.w3school.com.cn/html5/html_5_draganddrop.asp)
- [《HTML5 原生拖拽/拖放》](https://juejin.im/post/5a169d08518825592c07c666)

