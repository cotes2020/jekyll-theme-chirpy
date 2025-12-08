---
title: "canvas离屏技术与放大镜实现"
date: 2018-08-30
permalink: /2018-08-30-canvas-off-screen/
categories: ["实战分享"]
---

利用`canvas`除了可以实现滤镜，还可以利用**离屏技术**实现放大镜功能。为了方便讲解，本文分为 2 个应用部分：

1. 实现水印和中心缩放
2. 实现放大镜

## 什么是离屏技术？


[canvas实现前端滤镜](https://www.notion.so/623fe50e6b534106ba16861d07a79132)  介绍过`drawImage`接口。除了绘制图像，这个接口还可以：**将一个****`canvas`****对象绘制到另一个****`canvas`****对象上**。这就是离屏技术。


## 实现水印和中心缩放


在代码中，有两个 canvas 标签。分别是可见与不可见。**不可见的 canvas 对象上的 Context 对象，就是我们放置图像水印的地方。**


更多详解，请看代码注释：


```typescript
<!DOCTYPE html>
<html lang="en">
    <head>
        <meta charset="UTF-8" />
        <title>Learn Canvas</title>
        <style>
            canvas {
                display: block;
                margin: 0 auto;
                border: 1px solid #222;
            }
            input {
                display: block;
                margin: 20px auto;
                width: 800px;
            }
        </style>
    </head>
    <body>
        <div id="app">
            <canvas id="my-canvas"></canvas>
            <input type="range" value="1.0" min="0.5" max="3.0" step="0.1" />
            <canvas id="watermark-canvas" style="display: none;"></canvas>
        </div>
        <script type="text/javascript">
            window.onload = function() {
                var canvas = document.querySelector("#my-canvas");
                var watermarkCanvas = document.querySelector(
                    "#watermark-canvas"
                );
                var slider = document.querySelector("input");
                var scale = slider.value;
                var ctx = canvas.getContext("2d");
                var watermarkCtx = watermarkCanvas.getContext("2d");
                /* 给第二个canvas获取的Context对象添加水印 */
                watermarkCanvas.width = 300;
                watermarkCanvas.height = 100;
                watermarkCtx.font = "bold 20px Arial";
                watermarkCtx.lineWidth = "1";
                watermarkCtx.fillStyle = "rgba(255 , 255 , 255, 0.5)";
                watermarkCtx.fillText("=== dongyuanxin.github.io ===", 50, 50);
                /****************************************/
                var img = new Image();
                img.src = "./img/photo.jpg";
                /* 加载图片后执行操作 */
                img.onload = function() {
                    canvas.width = img.width;
                    canvas.height = img.height;
                    drawImageByScale(canvas, ctx, img, scale, watermarkCanvas);
                    // 监听input标签的mousemove事件
                    // 注意：mousemove实时监听值的变化，内存消耗较大
                    slider.onmousemove = function() {
                        scale = slider.value;
                        drawImageByScale(
                            canvas,
                            ctx,
                            img,
                            scale,
                            watermarkCanvas
                        );
                    };
                };
                /******************/
            };
            /**
             *
             * @param {Object} canvas 画布对象
             * @param {Object} ctx
             * @param {Object} img
             * @param {Number} scale 缩放比例
             * @param {Object} watermark 水印对象
             */
            function drawImageByScale(canvas, ctx, img, scale, watermark) {
                // 图像按照比例进行缩放
                var width = img.width * scale,
                    height = img.height * scale;
                // (dx, dy): 画布上绘制img的起始坐标
                var dx = canvas.width / 2 - width / 2,
                    dy = canvas.height / 2 - height / 2;
                ctx.clearRect(0, 0, canvas.width, canvas.height); // No1 清空画布
                ctx.drawImage(img, dx, dy, width, height); // No2 重新绘制图像
                if (watermark) {
                    // No3 判断是否有水印: 有, 绘制水印
                    ctx.drawImage(
                        watermark,
                        canvas.width - watermark.width,
                        canvas.height - watermark.height
                    );
                }
            }
        </script>
    </body>
</html>

```


实现效果如下图所示：


![Untitled.png](https://raw.githubusercontent.com/dongyuanxin/static/main/blog/imgs/2018-08-30-canvas-off-screen/6326c164e68a28b751721d54e4565727.png)


拖动滑竿，即可放大和缩小图像。然后右键保存图像。保存后的图像，就有已经有了水印，如下图所示：


![Untitled.png](https://raw.githubusercontent.com/dongyuanxin/static/main/blog/imgs/2018-08-30-canvas-off-screen/4d3dd690065da2e20a9460f3a75b5ffd.png)


## 实现放大镜


在上述中心缩放的基础上，实现放大镜主需要注意以下 2 个部分：

1. 细化处理`canvas`的鼠标响应事件：滑入、滑出、点击和松开
2. 重新计算离屏坐标（详细公式计算思路请见代码注释）
3. 重新计算鼠标相对于 canvas 标签的坐标（详细公式计算思路请见代码注释）

代码如下：


```javascript
<!DOCTYPE html>
<html lang="en">
    <head>
        <meta charset="UTF-8" />
        <title>Document</title>
        <style>
            canvas {
                display: block;
                margin: 0 auto;
                border: 1px solid #222;
            }
        </style>
    </head>
    <body>
        <canvas id="my-canvas"></canvas>
        <canvas id="off-canvas" style="display: none;"></canvas>
        <script>
            var isMouseDown = false,
                scale = 1.0;
            var canvas = document.querySelector("#my-canvas");
            var offCanvas = document.querySelector("#off-canvas"); // 离屏 canvas
            var ctx = canvas.getContext("2d");
            var offCtx = offCanvas.getContext("2d"); // 离屏 canvas 的 Context对象
            var img = new Image();
            window.onload = function() {
                img.src = "./img/photo.jpg";
                img.onload = function() {
                    canvas.width = img.width;
                    canvas.height = img.height;
                    offCanvas.width = img.width;
                    offCanvas.height = img.height;
                    // 计算缩放比例
                    scale = offCanvas.width / canvas.width;
                    // 初识状态下, 两个canvas均绘制Image
                    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
                    offCtx.drawImage(img, 0, 0, canvas.width, canvas.height);
                };
                // 鼠标按下
                canvas.onmousedown = function(event) {
                    event.preventDefault(); // 禁用默认事件
                    var point = windowToCanvas(event.clientX, event.clientY); // 获取鼠标相对于 canvas 标签的坐标
                    isMouseDown = true;
                    drawCanvasWithMagnifier(true, point); // 绘制在离屏canvas上绘制放大后的图像
                };
                // 鼠标移动
                canvas.onmousemove = function(event) {
                    event.preventDefault(); // 禁用默认事件
                    if (isMouseDown === true) {
                        var point = windowToCanvas(
                            event.clientX,
                            event.clientY
                        );
                        drawCanvasWithMagnifier(true, point);
                    }
                };
                // 鼠标松开
                canvas.onmouseup = function(event) {
                    event.preventDefault(); // 禁用默认事件
                    isMouseDown = false;
                    drawCanvasWithMagnifier(false); // 不绘制离屏放大镜
                };
                // 鼠标移出canvas标签
                canvas.onmouseout = function(event) {
                    event.preventDefault(); // 禁用默认事件
                    isMouseDown = false;
                    drawCanvasWithMagnifier(false); // 不绘制离屏放大镜
                };
            };
            /**
             * 返回鼠标相对于canvas左上角的坐标
             * @param {Number} x 鼠标的屏幕坐标x
             * @param {Number} y 鼠标的屏幕坐标y
             */
            function windowToCanvas(x, y) {
                var bbox = canvas.getBoundingClientRect(); // bbox中存储的是canvas相对于屏幕的坐标
                return {
                    x: x - bbox.x,
                    y: y - bbox.y
                };
            }
            function drawCanvasWithMagnifier(isShow, point) {
                ctx.clearRect(0, 0, canvas.width, canvas.height); // 清空画布
                ctx.drawImage(img, 0, 0, canvas.width, canvas.height); // 在画布上绘制图像
                /* 利用离屏，绘制放大镜 */
                if (isShow) {
                    var { x, y } = point;
                    var mr = 50; // 正方形放大镜边长
                    // (sx, sy): 待放大图像的开始坐标
                    var sx = x - mr / 2,
                        sy = y - mr / 2;
                    // (dx, dy): 已放大图像的开始坐标
                    var dx = x - mr,
                        dy = y - mr;
                    // 将offCanvas上的(sx,sy)开始的长宽均为mr的正方形区域
                    // 放大到
                    // canvas上的(dx,dy)开始的长宽均为 2 * mr 的正方形可视区域
                    // 由此实现放大效果
                    ctx.drawImage(
                        offCanvas,
                        sx,
                        sy,
                        mr,
                        mr,
                        dx,
                        dy,
                        2 * mr,
                        2 * mr
                    );
                }
                /*********************/
            }
        </script>
    </body>
</html>

```


放大镜效果如下图所示(被红笔标出的区域就是我们的正方形放大镜)：


![Untitled.png](https://raw.githubusercontent.com/dongyuanxin/static/main/blog/imgs/2018-08-30-canvas-off-screen/2578aed1a503c9f8120e0abd274e09ea.png)


