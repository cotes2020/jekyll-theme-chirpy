---
title: Java - DukeJava - 1-1-1 Programming Exercise 1 Modifying Images Pixel
date: 2020-09-02 11:11:11 -0400
description:
categories: [00CodeNote, JavaNote]
tags: [Java]
toc: true
---


# DukeJava - 1-1-1 Programming Foundations with JS, HTML and CSS

Java-Programming-and-Software-Engineering-Fundamentals-Specialization
- 1.Programming-Foundations-with-JavaScript-HTML-and-CSS
  - Programming Exercise: Modifying Images Pixel.

ProjectCode: https://github.com/ocholuo/language/tree/master/0.project/javademo

---

## 1

Write a JavaScript program that modifies an image by putting three vertical stripes on it - a red stripe on the left one third, a green stripe in the middle, and a blue stripe on the right one third. For example, if your program ran on Drew’s picture shown on the left, the resulting image would have red, green and blue vertical stripes as shown in the image on the right.

![3color](https://github.com/ocholuo/ocholuo.github.io/tree/master/assets/img/Javaimg/3color.png)

![3color](/assets/img/Javaimg/3color.png)


```js
// 1
var img = new SimpleImage("hilton.jpg");
print(img);

for(var pixel of img.values()) {
    if( pixel.getX() <= img.getWidth()/3) {
        pixel.setRed(255);
    }
    else if (pixel.getX() <= 2*img.getWidth()/3) {
        pixel.setGreen(255);
    }
    else {
        pixel.setBlue(255);
    }
}
print(img);
```

---

## 2.

Write code to change the Duke blue devil (the image below on the left) to be yellow (as in the image below on the right)

![Dukebluedevil](https://github.com/ocholuo/ocholuo.github.io/tree/master/assets/img/Javaimg/Dukebluedevil.png)

![Dukebluedevil](/assets/img/Javaimg/Dukebluedevil.png)

```js
var img = new SimpleImage("duke_blue_devil.png");
print(img);

for(var pixel of img.values()) {
    // if( pixel.getBlue() == 227){
    if(pixel.getRed()<255){
        pixel.setBlue(0);
        pixel.setRed(255);
        pixel.setGreen(255);
    }
}
print(img);
```

---

## 3.

![drewRobert](https://github.com/ocholuo/ocholuo.github.io/tree/master/assets/img/Javaimg/drewRobert.png)

![drewRobert](/assets/img/Javaimg/drewRobert.png)

![dinos](https://github.com/ocholuo/ocholuo.github.io/tree/master/assets/img/Javaimg/dinos.png)

![dinos](/assets/img/Javaimg/dinos.png)

![outimg](https://github.com/ocholuo/ocholuo.github.io/tree/master/assets/img/Javaimg/outimg.png)

![outimg](/assets/img/Javaimg/outimg.png)

```js
var img = new SimpleImage("drewRobert.png");
var bgimg = new SimpleImage("dinos.png");
var outimg = new SimpleImage(img.getWidth(), img.getHeight());

for(var pixel of img.values()) {
    if(pixel.getGreen() > pixel.getRed() + pixel.getBlue()){
    // if(pixel.getGreen() > 200){
        var x = pixel.getX();
        var y = pixel.getY();
        var bgpixel = bgimg.getPixel(x,y);
        outimg.setPixel(x,y,bgpixel);
    }
    else {
        outimg.setPixel(pixel.getX(),pixel.getY(),pixel);
    }
}
print(outimg);
```


---

## 4.

Your friend is trying to write a program that draws a square 200 pixels by 200 pixels and that looks like this square with colors red (red value 255), green (green value 255), blue (blue value 255) and magenta (red value 255 and blue value 255). All other RGB values are set to 0.


![4colorpixel](https://github.com/ocholuo/ocholuo.github.io/tree/master/assets/img/Javaimg/4colorpixel.png)

![4colorpixel](/assets/img/Javaimg/4colorpixel.png)

```java
var img = new SimpleImage(200,200);

for (var px of img.values()){
  var x = px.getX();
  var y = px.getY();
  if (x < img.getWidth()/2){
    px.setRed(255);
  }
  if (y>img.getHeight()/2){
    px.setBlue(255);
  }
  if(x > img.getWidth()/2 && y < img.getHeight()/2) {
    px.setGreen(255);
  }
}
print (img);
```


---

## 5.

write another function named addBorder. This function will add a black border to an image, such as in the following example:

![panda](https://github.com/ocholuo/ocholuo.github.io/tree/master/assets/img/Javaimg/panda.png)

![panda](/assets/img/Javaimg/panda.png)

black border that is 10 pixels thick

```java

1.

function setBlack(pixel) {
    pixel.setRed(0);
    pixel.setGreen(0);
    pixel.setBlue(0);
}

var img = new SimpleImage("smallpanda.png");
print(img);
for(var pixel of img.values()) {
    var x = pixel.getX();
    var y = pixel.getY();
    if(y < 10 || y > img.getHeight()-10){
        setBlack(pixel);
    }
    if(x < 10 || x > img.getWidth()-10){
        setBlack(pixel);
    }
}
print(img);


2.

function pixelOnEdge(image,pixel,horizontalThick, verticalThick){
    var x = pixel.getX();
    var y = pixel.getY();
    if (x < verticalThick || x > image.getWidth() - verticalThick){
        return true;
    }
    if (y < horizontalThick || y > image.getHeight() - horizontalThick){
        return true;
    }
    return false;
}

function addBorders(image,horizontalThick, verticalThick){
    for (var px of image.values()){
        if (pixelOnEdge(image,px,horizontalThick,verticalThick)){
            px = setBlack(px);
        }
    }
    return image;
}

var img = new SimpleImage("skyline.png");
img = addBorders(img,40,20);
print(img);
```






.
