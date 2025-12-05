---
title: "canvas实现前端滤镜"
date: 2018-08-20
permalink: /2018-08-20-canvas-beauty-filter/
---
### 常用 API 接口


关于图像处理的 API，主要有 4 个：

- 绘制图像： `drawImage(img,x,y,width,height)` 或 `drawImage(img,sx,sy,swidth,sheight,x,y,width,height)`
- 获取图像数据： `getImageData(x,y,width,height)`
- 重写图像数据： `putImageData(imgData,x,y[,dirtyX,dirtyY,dirtyWidth,dirtyHeight])`
- 导出图像： `toDataURL([type, encoderOptions])`

更详细的 API 和参数说明请看：[canvas 图像处理 API 参数讲解](https://www.jb51.net/article/123995.htm)


### 绘制图像


在此些 API 的基础上，我们就可以在`canvas`元素中绘制我们的图片。假设我们图片是`./img/photo.jpg`。


```javascript
<script>
    window.onload = function() {
        var img = new Image(); // 声明新的Image对象
        img.src = "./img/photo.jpg";
        // 图片加载后
        img.onload = function() {
            var canvas = document.querySelector("#my-canvas");
            var ctx = canvas.getContext("2d");
            // 根据image大小，指定canvas大小
            canvas.width = img.width;
            canvas.height = img.height;
            // 绘制图像
            ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
        };
    };
</script>
```


如下图所示，图片被画入了canvas：


![Untitled.png](https://raw.githubusercontent.com/dongyuanxin/static/main/blog/imgs/2018-08-20-canvas-beauty-filter/e302161acdcab38d6b5f8f08fac06925.png)


## 实现滤镜


> 这里我们主要借用getImageData函数，他返回每个像素的 RGBA 值。借助图像处理公式，操作像素进行相应的、数学运算即可。


[什么是 RGBA？](http://www.css88.com/book/css/values/color/rgba.htm)


[更多滤镜实现](https://www.cnblogs.com/st-leslie/p/8317850.html?utm_source=debugrun&utm_medium=referral)


### 去色效果


去色效果相当于就是老旧相机拍出来的黑白照片。人们根据人眼的敏感程度，给出了如下公式：


`gray = red * 0.3 + green * 0.59 + blue * 0.11`


代码如下：


```javascript
<script>
    window.onload = function() {
        var img = new Image();
        img.src = "./img/photo.jpg";
        img.onload = function() {
            var canvas = document.querySelector("#my-canvas");
            var ctx = canvas.getContext("2d");
            canvas.width = img.width;
            canvas.height = img.height;
            ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
            // 开始滤镜处理
            var imgData = ctx.getImageData(0, 0, canvas.width, canvas.height);
            for (var i = 0; i < imgData.data.length / 4; ++i) {
                var red = imgData.data[i * 4],
                    green = imgData.data[i * 4 + 1],
                    blue = imgData.data[i * 4 + 2];
                var gray = 0.3 * red + 0.59 * green + 0.11 * blue; // 计算gray
                // 刷新RGB，注意：
                // imgData.data[i * 4 + 3]存放的是alpha，不需要改动
                imgData.data[i * 4] = gray;
                imgData.data[i * 4 + 1] = gray;
                imgData.data[i * 4 + 2] = gray;
            }
            ctx.putImageData(imgData, 0, 0); // 重写图像数据
        };
    };
</script>
```


效果如下图所示：


![Untitled.png](https://raw.githubusercontent.com/dongyuanxin/static/main/blog/imgs/2018-08-20-canvas-beauty-filter/7c75b3791c53c021ea5893ff7a6f24b5.png)


### 负色效果


负色效果就是用最大值减去当前值。而 getImageData 获得的 RGB 中的数值理论最大值是：255。所以，公式如下：


`new_val = 255 - val`


代码如下：


```javascript
<script>
    window.onload = function() {
        var img = new Image();
        img.src = "./img/photo.jpg";
        img.onload = function() {
            var canvas = document.querySelector("#my-canvas");
            var ctx = canvas.getContext("2d");
            canvas.width = img.width;
            canvas.height = img.height;
            ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
            // 开始滤镜处理
            var imgData = ctx.getImageData(0, 0, canvas.width, canvas.height);
            for (var i = 0; i < imgData.data.length / 4; ++i) {
                var red = imgData.data[i * 4],
                    green = imgData.data[i * 4 + 1],
                    blue = imgData.data[i * 4 + 2];
                // 刷新RGB，注意：
                // imgData.data[i * 4 + 3]存放的是alpha，不需要改动
                imgData.data[i * 4] = 255 - imgData.data[i * 4];
                imgData.data[i * 4 + 1] = 255 - imgData.data[i * 4 + 1];
                imgData.data[i * 4 + 2] = 255 - imgData.data[i * 4 + 2];
            }
            ctx.putImageData(imgData, 0, 0); // 重写图像数据
        };
    };
</script>
```


效果图如下：


![Untitled.png](https://raw.githubusercontent.com/dongyuanxin/static/main/blog/imgs/2018-08-20-canvas-beauty-filter/bccbeae0239c64c375239f753b8344a1.png)


