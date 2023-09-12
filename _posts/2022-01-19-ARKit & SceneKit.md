---
title: ARKit 和 SceneKit 的简单使用
date: 2022-01-19 10:17:02
categories: iOS
tags: AR 相关
---

## 常用类的解释

![AR_SCNNode](/assets/img/AR_SCNNode.jpeg)

<br>

![ARKit](/assets/img/ARKit.png)

<br>

* `SceneKit`：是一个创建 3D 动画场景和效果的高级 3D 图形框架。
* `SCNView`：用来显示 3D 场景内容。
* `SCNScene`：场景，是节点层次结构和全局属性的容器，它们共同构成可显示的 3D 场景。
* `SCNNode`：节点对象的基类，节点是场景图的结构元素，表示 3D 坐标空间中的位置和变换，您可以将几何体、灯光、相机或其他可显示内容附加到节点上。场景的节点必须添加到根结点 `self.sceneView.scene.rootNode` 上，也可以给节点添加子节点。
* `SCNPhysicaBody`：附加到场景图节点的物理模拟属性，定义一个实体的类别和碰撞。
* `SCNGeometry`：一个可以在场景中显示的三维形状（也称为模型或网格），并带有定义其外观（也就是`.materials`属性）的附加素材。
* `SCNMaterial`：表面素材，也称贴图，是一组着色属性，用于定义渲染时几何体表面的外观。
* `SCNAction`：一个简单的、可重用的动画，可以更改您附加到的任何节点的属性，比如我们要让一个地球围绕太阳旋转，一个气球从一个地方移动另外一个地方。
* `SCNCamera`：一组相机属性，可以附加到节点以提供用于显示场景的视点。我们可以通过照相机捕捉到你想要观察的画面。
* `SCNLight`：可以附加到节点以照亮场景的光源。。
* `SCNAudioSource`：主要负责给场景中添加声音，一个简单的、可重复使用的音频源（从文件加载的音乐或音效），用于位置音频播放。

* `ARKit`：集成 iOS 设备相机和运动功能，在您的应用或游戏中产生增强现实体验。摄像机就是手机摄像头，通过摄像头和陀螺仪来采集数据。

附加到 `SCNNode` 节点对象的 `SCNGeometry` 几何体对象构成了场景的可见元素，而附加到几何体的 `SCNMaterial` 对象确定了它的外观。
    
``` objc
// 创建几何体
SCNBox *box = [SCNBox boxWithWidth:1 height:1 length:1 chamferRadius:0];

// 根据几何体来创建节点
SCNNode *boxNode = [[SCNNode alloc] init];
boxNode.geometry = box;

// 把 boxNode 放在摄像头正前方
boxNode.position = SCNVector3Make(0, 0, -1);

// 或者
SCNNode *boxNode = [SCNNode nodeWithGeometry:box];

// 设置节点的材质（贴图？）
SCNMaterial *material = [[SCNMaterial alloc] init];
material.diffuse.contents = [UIImage imageNamed:@"galaxy"];
cNode.geometry.materials  = @[material, material, material, material, material, material];
```

![坐标系](/assets/img/%E5%9D%90%E6%A0%87%E7%B3%BB.png)


<br>

## ARSessionDelegate

实现 ARSessionDelegate 中的方法，从 AR 会话接收捕获的视频帧图像和跟踪状态。
<br>

``` objc
// 提供新捕获的相机图像和随附的 AR 信息
- (void)session:(ARSession *)session didUpdateFrame:(ARFrame *)frame;

// 锚点已被添加到 session 中
- (void)session:(ARSession *)session didAddAnchors:(NSArray<ARAnchor *> *)anchors;

// session 已更新锚点的属性。
- (void)session:(ARSession *)session didUpdateAnchors:(NSArray<ARAnchor *> *)anchors;

// 从 session 中移除锚点
- (void)session:(ARSession *)session didRemoveAnchors:(NSArray<ARAnchor *> *)anchors;
```

<br>

## ARSCNViewDelegate
实现该代理的方法来让 `SceneKit` 内容与 `AR 会话`自动同步
<br>

``` objc
// 提供一个与新添加的锚点相对应的节点，如果返回 nil，则锚点被忽略；如果未实现这个代理方法，
// 则 ARKit 会自动创建一个空节点，您可以实现 renderer:didAddNode:forAnchor: 方法，通
// 过将其附加到该节点来提供可视内容。
- (SCNNode *)renderer:(id<SCNSceneRenderer>)renderer nodeForAnchor:(ARAnchor *)anchor;

// 与 ARAnchor 对象相对应的 SCNNode 对象已被添加到场景中
- (void)renderer:(id<SCNSceneRenderer>)renderer didAddNode:(SCNNode *)node forAnchor:(ARAnchor *)anchor;

// 将要更新与 ARAnchor 对象对应的 SCNNode 对象的属性
- (void)renderer:(id<SCNSceneRenderer>)renderer willUpdateNode:(SCNNode *)node forAnchor:(ARAnchor *)anchor;

// 已更新与 ARAnchor 对象对应的 SCNNode 对象的属性。
- (void)renderer:(id<SCNSceneRenderer>)renderer didUpdateNode:(SCNNode *)node forAnchor:(ARAnchor *)anchor;

// 从场景中移除与 ARAnchor 对象对应的 SCNNode 对象。
- (void)renderer:(id<SCNSceneRenderer>)renderer didRemoveNode:(SCNNode *)node forAnchor:(ARAnchor *)anchor;
```

<br>

## 基本使用

``` objc
// 第一步 创建场景视图
self.sceneView = [[ARSCNView alloc] initWithFrame:self.view.bounds];
self.sceneView.backgroundColor = [UIColor blackColor];
self.sceneView.preferredFramesPerSecond = 60;
self.sceneView.showsStatistics = YES;
[self.view addSubview:self.sceneView];
    
// 第二步 加载场景文件创建场景，注意 sceneView 默认没有 scene，必须给它设置一个非空场景才能显示
self.sceneView.scene = [SCNScene sceneNamed:@"ship.scn"];

// 第三步 创建一个正方体的几何模型
SCNBox *box = [SCNBox boxWithWidth:1 height:1 length:1 chamferRadius:0];
box.firstMaterial.diffuse.contents = [UIImage imageNamed:@"1.png"];
    
// 第四步 创建一个节点，将几何模型绑定到这个节点上去
SCNNode *boxNode = [[SCNNode alloc] init];
boxNode.geometry = box;
    
// 第五步 将绑定了几何模型的节点添加到场景的根节点上去，rootNode 是一个特殊 node，它是所有 node 的起始点
[self.sceneView.scene.rootNode addChildNode:boxNode];
    
// 第六步 运行操作摄像机，开启了这个功能，你就可以使用手势改变场景中摄像机的位置和方向了
self.sceneView.allowsCameraControl = YES;
    
// 第七步 开启抗锯齿，如果场景边缘有锯齿状的现象，可使用这个属性让锯齿减弱，但会消耗更多手机资源，谨慎使用。
self.sceneView.antialiasingMode = SCNAntialiasingModeMultisampling4X;
    
// 场景视图截屏
UIImage *image = self.sceneView.snapshot;
    
// 将场景写入到本地保存
NSString *path = NSSearchPathForDirectoriesInDomains(NSDocumentDirectory, NSUserDomainMask, YES).firstObject;
path = [path stringByAppendingPathComponent:@"test.scn"];
[self.sceneView.scene writeToURL:[NSURL URLWithString:path] options:nil delegate:nil progressHandler:NULL];
    
// 给节点添加子节点
// 创建子节点 给子节点添加几何形状
SCNNode *childNode = [SCNNode node];
    
// 设置子节点的位置
childNode.position = SCNVector3Make(-0.5, 0, 1);
   
// 设置几何形状，我们选择立体字体
SCNText *text = [SCNText textWithString:@"让学习成为一种习惯" extrusionDepth:0.03];
    
// 设置字体颜色
text.firstMaterial.diffuse.contents = [UIColor redColor];
    
// 设置字体大小
text.font = [UIFont systemFontOfSize:0.15];
   
// 给子节点绑定几何物体
childNode.geometry = text;
[boxNode addChildNode:childNode];
```

<br>

``` objc
// 点击屏幕添加节点
- (void)handleTap:(UITapGestureRecognizer *)sender {
    CGPoint touchPoint = [sender locationInView:self.sceneView];
    
    // 检测点击的地方有没有经过一些符合指定要求的点/面，返回一个数组，取第一个元素
    ARHitTestResult *result = [self.sceneView hitTest:touchPoint types:ARHitTestResultTypeFeaturePoint].firstObject;
    
    if (result) {
        SCNBox *box = [SCNBox boxWithWidth:0.1 height:0.1 length:0.1 chamferRadius:0];
        SCNNode *boxNode = [SCNNode nodeWithGeometry:box];
        
        // ARHitTestResult 信息转换成三维坐标，用到了 result.worldTransform.columns[3].x（y，z）的信息
        boxNode.position = SCNVector3Make(result.worldTransform.columns[3].x,
                                          result.worldTransform.columns[3].y,
                                          result.worldTransform.columns[3].z);
        [self.sceneView.scene.rootNode addChildNode:boxNode];
    }
}
```

<br>

## 注意
根据会话配置，ARKit 会自动将特殊的锚点添加到会话中，例如：World-tracking 会话可以添加 `ARPlaneAnchor`、`ARObjectAnchor` 和 `ARImageAnchor` 对象，如果您启用相应的功能；面部跟踪会话添加 `ARFaceAnchor` 对象。
