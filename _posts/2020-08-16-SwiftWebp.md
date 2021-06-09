---
layout: post
title: "Swift使用Webp动图"
date: 2020-08-16 23:54:00.000000000 +09:00
categories: [Swift]
tags: [Swift, Webp, YYImage, SDWebImageWebPCoder]
---

## 关于Webp

`WebP`是一种同时提供了`有损压缩`与`无损压缩`（可逆压缩）的图片文件格式，是Google在2010发布的，`WebP` 的优势体现在它具有更优的图像数据压缩算法，能带来更小的图片体积，而且拥有肉眼识别无差异的图像质量。具体规范参见: [WebP](https://developers.google.com/speed/webp/)

**Webp 编码参数**

+ `lossless`: YES 为有损编码， NO 为无损编码。WebP 主要优势在于有损编码，其无损编码的性能和压缩比表现一般。
+ `quality`: `0~100` 图像质量，0表示最差质量，文件体积最小，细节损失严重，100表示最高图像质量，文件体积较大。该参数只针对有损压缩有明显效果，Google 官方的建议是 75。
+ `method`: `0~6`压缩比，0表示快速压缩，耗时短，压缩质量一般，6表示极限压缩，耗时长，压缩质量好。该参数也只针对有损压缩有明显效果，调节该参数最高能带来 `20% ～ 40%` 的更高压缩比，但相应的编码时间会增加 `5～20` 倍。Google 官方推荐的值是 4。

**Webp 解码参数**

+ `use_threads`: 是否启用 `pthread` 多线程解码。该参数只对宽度大于 512 的有损图片起作用，开启后内部会用多线程解码，CPU 占用会更高，解码时间平均能缩短 10%～20%。
+ `bypass_filtering`: 是否禁用滤波。该参数只对有损图片起作用，开启后大约能缩短 5%～10% 的解码时间，但会造成一些颜色过渡平滑的区域产生色带（banding）。
+  `no_fancy_upsampling`: 是否禁用上采样。该参数只对有损图片起作用。

## Webp展示

`Webp`动图的展示是播放帧图片，即解码出来的一张张图片就是需要绘制播放的，关键数据就是 `Image & Duration`。

**1.YYImage展示**

用`YYImage`框架来展示Webp动图，是使用了一个`UIImageView`的子类`YYAnimatedImageView`，通过直接插入了一个`CALayer`来作为图片的渲染，并用`CADisplayLink`这个帧定时器来刷新动图帧，通过异步线程处理解码，还有一些C的动态分配和回收内存来避免非常高的内存占用，保证了性能。

```swift
@interface YYAnimatedImageView : UIImageView
// 如果 image 为多帧组成时，自动赋值为 YES，可以在显示和隐藏时自动播放和停止动画
@property (nonatomic) BOOL autoPlayAnimatedImage;
// 当前显示的帧（从 0 起始），设置新值后会立即显示对应帧，如果新值无效则此方法无效
@property (nonatomic) NSUInteger currentAnimatedImageIndex;
// 当前是否在播放动画
@property (nonatomic, readonly) BOOL currentIsPlayingAnimation;
// 动画定时器所在的 runloop mode，默认为 NSRunLoopCommonModes，关乎动画定时器的触发
@property (nonatomic, copy) NSString *runloopMode;
// 内部缓存区的最大值（in bytes），默认为 0（动态），如果有值将会把缓存区限制为值大小，当收到内存警告或者 App 进入后台时，缓存区将会立即释放并且在适时的时候回复原状
@property (nonatomic) NSUInteger maxBufferSize;
@end
```

`YYAnimatedImageView` 作为 YYImage 框架中的图片视图层，上接图像层，下启编/解码底层。

```
pod 'YYWebImage'
pod 'YYImage/WebP'
```

```swift
// 初始化
private lazy var webpImageView: YYAnimatedImageView = {
        
   let imageView = YYAnimatedImageView()
   imageView.contentMode = .scaleAspectFill
   imageView.backgroundColor = UIColor.arcColor()
   imageView.autoPlayAnimatedImage = true
   return imageView
}()
// Webp动图展示
self.webpImageView.yy_setImage(with: URL(string: model.webpURL), options: [.progressiveBlur, .setImageWithFadeAnimation])
if !self.webpImageView.currentIsPlayingAnimation {
	 self.webpImageView.startAnimating()
}
```

**2.SDWebImageWebPCoder展示**

```swift
@implementation SDImageWebPCoder {
    WebPIDecoder *_idec; // 增量解码器
    WebPDemuxer *_demux; // 分离器
    WebPData *_webpdata; // Copied for progressive animation demuxer
    NSData *_imageData;
    NSUInteger _loopCount; // 动画循环次数
    NSUInteger _frameCount; // 动画帧数
    NSArray<SDWebPCoderFrame *> *_frames; // 动画帧数据集合
    CGContextRef _canvas; // 图片画布
    CGColorSpaceRef _colorSpace; // 图片 icc 彩色空间
    BOOL _hasAlpha;
    CGFloat _canvasWidth; // 图片画布宽度
    CGFloat _canvasHeight; // 图片画布高度
    NSUInteger _currentBlendIndex; //动画的当前混合帧率
}
```

`SDWebImageWebPCoder`基于`libwebp`库支持`webp`格式解码，而且一个线程一次性解压一张`Webp`动图，根据测试每帧画面通过`libwebp`提供的解码API需要`0.05-0.01s`的时间，如果`Webp`动图帧数比较多时解码会比较耗时，这点注意！而`YYImage`则是边解码边显示，就显示速度而言，优于`SDWebImage`。

```
pod 'SDWebImageWebPCoder'
```

```swift
// 初始化
private lazy var webpImageView: SDAnimatedImageView = {
        
   let imageView = SDAnimatedImageView()
   imageView.contentMode = .scaleAspectFill
   imageView.backgroundColor = UIColor.arcColor()
   imageView.maxBufferSize = UInt.max
   imageView.shouldIncrementalLoad = true
   return imageView
}()

// Webp动图展示
imageView.sd_setImage(with: URL(string: model.img), placeholderImage: nil, options: [.progressiveLoad])
```