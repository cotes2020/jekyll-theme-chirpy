---
layout: post
title: "初识 Texture"
date: 2020-06-05 23:34:00.000000000 +09:00
categories: [Swift]
tags: [Swift, Texture, AsyncDisplayKit]
---

## 为什么使用Texture

`UITableView/UICollectionView`的优化一直是iOS应用性能优化重要的一块，在App列表滚动时仔细观察还是会感觉到有一定的掉帧现象，在一些复杂的列表上是无法达到`60fps`帧率的。造成掉帧的原因有很多，网上分析如下:

+ CPU(主要是主线程)/GPU负担过重或者不均衡。
+ Autolayout布局性能瓶颈，约束计算时间会随着数量呈指数级增长，并且必须在主线程执行。
+ 大多数app来说，多线程协作并没有被充分利用，在app卡顿（主线程所占用的核心满负荷）时，往往CPU的其他核心几乎无事可做。由于主线程承担了绝大部分的工作，如果能把主线程的任务转移一部给其他线程进行异步处理，就可以马上享受到并发带来的性能提升。

如何能将主线程的压力尽可能减轻成为优化的首要目标，**网上比较好的优化例子:**

+ [YY大神的博客](https://blog.ibireme.com/2015/11/12/smooth_user_interfaces_for_ios/)
+ 针对Autolayout性能优化，提前计算并缓存cell的layout。[forkingdog](https://github.com/forkingdog/UITableView-FDTemplateLayoutCell)
+ 省去中间滑动过程中的计算，直接计算目标区域cell。[VVebo](https://github.com/johnil/VVeboTableViewDemo)
+ 弃用Autolayout，采用手动布局计算。这样虽然可以换来最高的性能，但是代价是编写和维护的不便，对于经常改动或者性能要求不高的场景并不一定值得。

## 初识Texture

[Texture](https://github.com/TextureGroup/Texture)

`AsyncDisplayKit(ASDK)`是2012年由Facebook开始着手开发，现已经改名为`Texture`，并于2014年出品的高性能显示类库，主要作者是[Scott Goodson](https://github.com/appleguy)，在Scott介绍ASDK的视频中，总结了一下三点占用大量CPU时间的『元凶』（虽然仍然可能有以上提到的其他原因，但ASDK最主要集中于这三点进行优化）：

+ **渲染**，对于大量图片，或者大量文字(尤其是CJK字符)混合在一起时。而文字区域的大小和布局，恰恰依赖着渲染的结果。`Texture`尽可能后台线程进行渲染，完成后再同步回主线程相应的UIView。
+ **布局**，`Texture`完全弃用了`Autolayout`，另辟蹊径实现了自己的布局和缓存机制。
+ 系统objects的**创建与销毁**，由于UIKit封装了CALayer以支持触摸等显示以外的操作，耗时也相应增加。而这些同样也需要在主线程上操作。ASDK基于Node的设计，突破了UIKit线程的限制。

所以在主线程上同步执行，那么就意味着线程阻塞，把耗时的工作放到异步处理，等到需要主线程时再同步回来。对于一般`UIView`或`CALayer`来说，由于不是线程安全，任何相关操作都需要在主线程进行，为了弥补这些不足，`Texture`引入了`Node`的概念来解决`UIView/CALayer`只能在主线程上操作的限制。

![](/assets/images/texture01.png)

## Texture控件

`Texture`几乎涵盖了常用的控件，下面是 `Texture` 和 `UIKit` 的对应关系。

**Nodes**:

| Texture            | UIKit                                |
| ------------------ | ------------------------------------ |
| ASDisplayNode      | UIView                               |
| ASCellNode         | UITableViewCell/UICollectionViewCell |
| ASTextNode         | UILabel                              |
| ASImageNode        | UIImageView                          |
| ASNetworkImageNode | UIImageView                          |
| ASVideoNode        | AVPlayerLayer                        |
| ASScrollNode       | UIScrollView                         |
| ASEditableTextNode | UITextView                           |
| ASControlNode      | UIControl                            |

**Node Containers:**

| Texture          | UIKit            |
| ---------------- | ---------------- |
| ASViewController | UIViewController |
| ASTableNode      | UITableView      |
| ASCollectionNode | UICollectionView |
| ASPagerNode      | UICollectionView |

+ ASDisplayNode
  + 作用同等于`UIView`，是所有 Node 的父类，而且 `ASDisplayNode` 有一个`view`属性，所以`ASDisplayNode`及其子类都可以通过这个`view`来添加`UIKit`控件，所以`Texture` 和 `UIKit`混用是完全没问题的。
+ ASNetworkImageNode
  + 作用同等于 `UIImageView`，如果使用网络图片请使用此类，`Texture` 用的是第三方的图片加载库[PINRemoteImage](https://github.com/pinterest/PINRemoteImage)，`ASNetworkImageNode` 其实并不支持 gif，如果需要显示 gif 推荐使用[FLAnimatedImage](https://github.com/Flipboard/FLAnimatedImage)。

## Texture 布局

Auto Layout 不止在复杂 UI 界面布局的表现不佳，它还会强制视图在主线程上布局；所以在`Texture`中提供了另一种可以在后台线程中运行的布局引擎，它的结构大致是这样的：

![](/assets/images/texture02.png)

`Texture`有三种布局方式:

+ 手动布局

  + 在`Texture`中的手动布局与UIKit类似，只是方法名从`sizeThatFits`变成了`calculatedSizeThatFits`，`layoutSubviews`方法变成了`layout`方法。`Texture`会将布局结果异步预先计算，并缓存下来以提升性能。但是可读性和维护性比较差。

    ```swift
    func calculateSizeThatFits(_ constrainedSize: CGSize) -> CGSize {
        return _preferredFrameSize  
    }
    ```

+ Unified layout

  + 一个`ASLayout`对象包含以下元素有布局元素、元素尺寸、元素位置和sublayouts。当一个`node`具备了确定的`ASLayout`对象后，只要能计算出自己的`ASLayout`，父元素就可以完成对其的布局，但是缺点跟手动布局类似。

+ Automatic Layout

  + `Automatic Layout`是ASDK最推荐也是最为高效的布局方式，它引入了ASLayoutSpec的概念，可以理解为一个抽象的容器（并不需要创建对应node或者view来装载子元素），只需要为它制定一系列的布局规则，它就能对其负责的子元素进行布局。它们的关系如下图:

  ![](/assets/images/texture03.png)

可以看到`ASLayoutSpec`和`ASDisplayNode`都实现了ASLayoutable接口，因此他们都具备生成ASLayout的能力，这样就能唯一确定自身的大小。

- 对于以上提到的Unified布局，当我们实现了`calculateLayoutThatFits`，`Texture`会在布局过程中调用`measureWithSizeRange`(如果没有缓存过再调用`calculateLayoutThatFits`)来算出ASLayout。
- 如果`ASDisplayNode`选择实现`layoutSpecThatFits`，由更为抽象的`ASLayoutSpec`来负责指定布局规则，由于`ASLayoutSpec`也实现了`ASLayoutable`接口，同样也可以通过调用`measureWithSizeRange`来获得`ASLayout`。

受`CSS Flexbox`启发，`Texture layout` 和 `Flexbox` 有众多相似之处，`Texture` 的布局系统围绕布局规范 `Layout Specs` 和 布局元素 `Layout Elements`两个基本概念实现布局功能。

在自定义的ASDisplayNode中重写`override func layoutSpecThatFits(_ constrainedSize: ASSizeRange) -> ASLayoutSpec`方法进行布局，如果不是自定义，那么也可以使用`self.node.layoutSpecBlock = { [weak self] node, size in}`方法实现。

```swift
override func layoutSpecThatFits(_ constrainedSize: ASSizeRange) -> ASLayoutSpec {
    let stackLayout = ASStackLayoutSpec.horizontal()
    stackLayout.justifyContent = .start
    stackLayout.alignItems = .start
    stackLayout.style.flexShrink = 1.0
    stackLayout.children = [imagePlace]
    return  ASInsetLayoutSpec(insets: UIEdgeInsets.zero, child: stackLayout)
}
```

```swift
self.node.layoutSpecBlock = { [weak self] node, size in
    guard let `self` = self else { return ASStackLayoutSpec() }
    let titleInset = ASInsetLayoutSpec(insets: UIEdgeInsets(top: 0, left: 16, bottom: 0, right: 16), child: self.titleLabel)
    let layout = ASStackLayoutSpec(direction: .vertical, spacing: 4, justifyContent: .start, alignItems: .start, children: [titleInset, self.topicsCollectionView])
    return ASInsetLayoutSpec(insets: .zero, child: layout)
}
```

`ASLayoutSpec`相关API:

**1.`ASStackLayoutSpec`: 栈布局**

+ `direction`: 主轴的方向
  + `ASStackLayoutDirectionHorizontal`: 水平
  + `ASStackLayoutDirectionVertical`: 竖直
+ `spacing`: 主轴上子视图的间距
+ `justifyContent`:  子视图在主轴上的排列方式
  + `ASStackLayoutJustifyContentStart`: 从左往右排列
  + `ASStackLayoutJustifyContentCenter`: 居中排列
  + `ASStackLayoutJustifyContentEnd`: 从后往前排列
  + `ASStackLayoutJustifyContentSpaceBetween`: 间隔排列，两端没有间隙
  + `ASStackLayoutJustifyContentSpaceAround`: 间隔排列，两端有间隙
+ `alignItems`: 交叉轴排列方式
  + `ASStackLayoutAlignItemsStart`: 起点对齐
  + `ASStackLayoutAlignItemsEnd`: 终点对齐
  + `ASStackLayoutAlignItemsCenter`: 居中对齐
  + `ASStackLayoutAlignItemsStretch`: 没有设置高度前提下，会去拉伸直到填满整个父视图
  + `ASStackLayoutAlignItemsBaselineFirst`: (水平布局专有)第一个子视图的文字内容作为基线对齐
  + `ASStackLayoutAlignItemsBaselineLast`: (水平布局专有)最后一个子视图的文字内容作为基线对齐
+ `children`: 添加约束的子视图

```swift
override func layoutSpecThatFits(_ constrainedSize: ASSizeRange) -> ASLayoutSpec {

   let layout = ASStackLayoutSpec(direction: .vertical, spacing: 10, justifyContent: .start, alignItems: .center, children: [displayBg, titleNode])
   return ASInsetLayoutSpec(insets: UIEdgeInsets.zero, child: layout)
}
```

**2.`ASAbsoluteLayoutSpec`: 绝对布局**

+ 可以设置视图的绝对位置，但比较固定，官方文档里不建议使用
+ `layoutPosition`：ASAbsoluteLayoutSpec布局中，设置的起点位置

**3.ASBackgroundLayoutSpec: 背景布局 & ASOverlayLayoutSpec: 覆盖布局**

+ 背景布局：把一个视图拉伸作为另外一个视图的背景
+ 覆盖布局：和背景布局是对立的

```swift
override func layoutSpecThatFits(_ constrainedSize: ASSizeRange) -> ASLayoutSpec {
	
   let userLayout = ASStackLayoutSpec(direction: .horizontal, spacing: 4, justifyContent: .start, alignItems: .center, children: [userAvatar, userName])
   let userInset = ASInsetLayoutSpec(insets: UIEdgeInsets(top: 2, left: 4, bottom: 2, right: 4), child: userLayout)
   let userSpec = ASBackgroundLayoutSpec(child: userInset, background: userNode)
	 return userSpec
}
```

```swift
override func layoutSpecThatFits(_ constrainedSize: ASSizeRange) -> ASLayoutSpec {
        
   let layout = ASOverlayLayoutSpec(child: followBg, overlay: followButton)
   return ASInsetLayoutSpec(insets: .zero, child: layout)
}
```

**4.`ASInsetLayoutSpec`: 边距布局**

+ 设置子视图边距

```swift
override func layoutSpecThatFits(_ constrainedSize: ASSizeRange) -> ASLayoutSpec {

   return ASInsetLayoutSpec(insets: UIEdgeInsets.zero, child: tagNode)
}
```

**5.`ASRatioLayoutSpec`: 比例布局**

+ 根据自身的高度或者宽度来设置自身比例(高度 : 宽度)，所以必须先设置自身的高度或者宽度。 (ratio > 1 == 高>宽，ratio < 1 == 高<宽)，常见图片设置比例。

```swift
override func layoutSpecThatFits(_ constrainedSize: ASSizeRange) -> ASLayoutSpec {
    var imageRatio: CGFloat = 0.5
    if imageNode.image != nil {
      imageRatio = (imageNode.image?.size.height)! / (imageNode.image?.size.width)!
    }
    
    let imagePlace = ASRatioLayoutSpec(ratio: imageRatio, child: imageNode)
    
    let stackLayout = ASStackLayoutSpec.horizontal()
    stackLayout.justifyContent = .start
    stackLayout.alignItems = .start
    stackLayout.style.flexShrink = 1.0
    stackLayout.children = [imagePlace]
    
    return  ASInsetLayoutSpec(insets: UIEdgeInsets.zero, child: stackLayout)
}
```

**6.`ASRelativeLayoutSpec`: 相对布局**

+ 设置类似九宫格上任意一个区域的位置，需要水平和竖直方向组合使用
+ `horizontalPosition`: 水平方向位置
  + `ASRelativeLayoutSpecPositionStart`: 左边
  + `ASRelativeLayoutSpecPositionCenter`: 中间
  + `ASRelativeLayoutSpecPositionEnd`: 右边
+ `verticalPosition`: 竖直方向位置
  + `ASRelativeLayoutSpecPositionStart`: 上边
  + `ASRelativeLayoutSpecPositionCenter`: 中间
  + `ASRelativeLayoutSpecPositionEnd`: 下边

```swift
override func layoutSpecThatFits(_ constrainedSize: ASSizeRange) -> ASLayoutSpec {
        
   self.imageNode.style.preferredSize = CGSize(width: 60, height: 60)
   let layout = ASRelativeLayoutSpec()
   layout.horizontalPosition = .end
   layout.verticalPosition = .center
   layout.child = self.imageNode
   return ASInsetLayoutSpec(insets: .zero, child: layout)
}
```

**7.`ASCenterLayoutSpec`: 居中布局**

+ 设置主轴或者交叉轴上的居中布局
+ `centeringOptions`: 居中方式(X，Y，XY轴)
+ `sizingOptions`: 这个中心布局会占据多少空间(minimum X, minimum Y, minimum XY)

```swift
override func layoutSpecThatFits(_ constrainedSize: ASSizeRange) -> ASLayoutSpec {
  
   self.imageNode.style.preferredSize = CGSize(width: 60, height: 60)
   return ASCenterLayoutSpec(centeringOptions: .XY, sizingOptions: [], child: self.imageNode)
}
```

**8.`ASWrapperLayoutSpec`: 填充布局**

+ 根据布局视图上设置的大小来包装和计算子视图的布局

```swift
override func layoutSpecThatFits(_ constrainedSize: ASSizeRange) -> ASLayoutSpec {
    return ASWrapperLayoutSpec(layoutElement: self.collectionNode)
}
```

**9.`ASCornerLayoutSpec`: 圆角布局**

+ 设置角标

```swift
override func layoutSpecThatFits(_ constrainedSize: ASSizeRange) -> ASLayoutSpec {

    let cornerSpec = ASCornerLayoutSpec(child: avatarNode, corner: iconNode, location: .bottomRight)
   cornerSpec.offset = self.isBrand ? CGPoint(x: -12, y: -12) : CGPoint(x: -6, y: -6)
    return ASInsetLayoutSpec(insets: .zero, child: cornerSpec)
}
```

**10.`ASStackLayoutElement`元素属性**

+ `spacingBefore`: 栈布局中，第一个子视图在主轴上的间隙
+ `spacingAfter`: 栈布局中，最后一个子视图在主轴上的间隙
+ `flexGrow`: 定义子视图的放大比例，如果为0，即使有剩余空间也不会放大。如果所有子视图都设置，那么就按照设置的比例分配，如果都设置为1，那么是均分剩余空间，如果有一个设置为2，那么该视图分到的空间将是其他视图的2倍
+ `flexShrink`: 缩小比例：当空间不足的时候，缩小子视图，如果所有的子视图都设置为1就是等比例缩小，如果有一个子视图设置为0，表示该视图不缩小。
+ `flexBasis`: 可以指定某个**约束**的初始大小
+ `alignSelf`: 这个属性和alignItems效果一样，设置了这个，那么就会覆盖alignItems的效果
+ `ascender`: 在baseline选项中生效，文字基准线的顶部距离
+ `descender`: 在baseline选项中生效，文字基准线的底部距离

**11.`ASLayoutElement` 属性**

+ `preferredSize`: 设置视图的大小，如果设置了maxSize或者minSize，那么maxsize和minSize的优先级高。
+ `width`: 设置ASLayoutElement的内容宽度，默认ASDimensionAuto，minWidth和maxWidth会覆盖width。
+ `height`: 设置spec的内容高度
+ `minWidth`: 设置spec最小宽度
+ `minHeight`: 设置spec最小高度
+ `maxWidth`: 设置spec最大宽度
+ `maxHeight`: 设置spec最大高度
+ `preferredLayoutSize`:设置建议的相对尺寸，也就是一般推荐写百分比
+ `minLayoutSize`: 设置最小的相对尺寸
+ `maxLayoutSize`: 设置最大的相对尺寸
+ `minSize`: 设置最小尺寸
+ `maxSize`: 设置最大尺寸

`Layout API`是为了替代 `UIKit Auto Layout` 的高性能布局方案，具有许多优点：

- 快速：与手动布局一样快，比 auto layout 快很多。
- 异步和并发：布局在后台线程计算，不影响用户交互。
- 声明性（declarative）：布局用不可变数据结构声明。这使布局代码更易于开发、文档编写、代码审查、测试、调试、配置和维护。
- 可缓存：布局结果是不可变的数据结构，因此可以在后台对其预先计算并缓存，以提高用户感知性能。
- 可扩展：易于在类之间共享代码。

## Texture 使用

**1.Hit Test Slop**

`ASDisplayNode`有类型为`UIEdgeInsets`的`hitTestSlop`属性时，将其设置为正值时，缩小点击范围；设置为负值时，扩大点击范围。所有 node 均继承自`ASDisplayNode`，因此所有 node 均可以使用`hitTestSlop`属性。

`ASDisplayNode`获取触摸事件的能力受父Node的尺寸、`hitTestSlop`限制，如果子 node 想要超出父 node 尺寸，则需要扩大父 node `hitTestSlop`以包含 child node 需要响应触摸事件区域，例如:

```swift
let textNode = ASTextNode()
textNode..hitTestSlop = UIEdgeInsets(top: -22, left: 0, bottom: -22, right: 0)
```

**2.批量拉取数据`batch fetching`**

`Texture` 的批量拉取`batch fetching api`功能可以很方便的拉取数据，当用户滑动到距离 `ASTableNode`、`ASCollectionNode` 内容末尾两屏幕时，将尝试拉取更多数据。如果需要配置触发拉取的距离，只需设置`ASTableNode`、`ASCollectionNode`的`leadingScreensForBatching`属性。

```swift
tableNode.leadingScreensForBatching = 3
```

批量拉取`delegate`方法中决定决定是否执行批量拉取:

```swift
// ASTableNode
func shouldBatchFetch(for tableNode: ASTableNode) -> Bool {
    return currentPage < pageCount ? true : false
}
// ASCollectionNode
func shouldBatchFetch(for collectionNode: ASCollectionNode) -> Bool {
    return currentPage < pageCount ? true : false
}
```

当滚动到需要拉取更多数据时会触发上面的方法，如果有更多数据则返回`true`进行拉取，反之不拉取。

> 如果未实现上述方法，在进入拉取区域时会通知其`asyncDelegate`。

当返回`true`时，则会调用下面的方法:

```swift
// ASTableNode
func tableNode(_ tableNode: ASTableNode, willBeginBatchFetchWith context: ASBatchContext) {
    self.loadData(.more)
    context.completeBatchFetching(true)
}
func collectionNode(_ collectionNode: ASCollectionNode, willBeginBatchFetchWith context: ASBatchContext) {
    self.loadData(.more)
    context.completeBatchFetching(true)
}
```

当拉取完数据后则必须告知`Texture`这个过程已经完成，调用`completeBatchFetching:`方法，参数为`true`，只有传入`true`，再次需要拉取时才会尝试拉取更多数据。

> 这里后台返回的数据需要注意，数据有可能会相同。

## Texture优化列表性能

说到视图列表性能问题，我们不得不提到`UITableView/UICollectionView`，对于它的滚动性能的讨论和优化从未停止。

+ `cell reuse` 重用
+ `estimated cell height` 预估Cell高度，iOS8开始原生支持
+ 手动将计算完成的height缓存(或使用FDTemplateLayoutCell等框架自动计算)
+ 异步加载cell内容，文字图片等

`UITableView`加载Cell的过程如下:

+ cellForRowAtIndexPath，读取model
+ 从reuse池中dequeue一个cell，调用prepareForReuse重置其状态
+ 将model装配到UITableViewCell中
+ 布局（耗时且无法缓存的Autolayout），渲染（文字和图片都很耗时）

上面这些操作都在`cell`进入window刹那发生的，在短短的16ms里`60fps`是很难完成这些任务的，尤其是当用户快速滚动的时候，大量任务堆积在runloop。

**ASTableNode/ASCollectionNode开辟的新用法**

`ASTableNode/ASCollectionNode`中的`ASCellNode`已经具备了异步布局和异步渲染的能力，即使没有做额外优化，仅仅利用`Texture`通用的异步机制将耗时操作延后，相对于一般UITableView已经有了显著的提升。但是还是会出现一些问题，在进入屏幕之后才开始渲染，会有短暂的白屏现象（等待渲染完成）再显示内容。所以在显示之前的一段时间，把布局和渲染的工作预先完成。

`ASTableNode/ASCollectionNode`的`ASInterfaceState`有5中状态:

+ None，该node在一段时间内不会进入屏幕
+ MeasureLayout，可能会在一段时间后进入屏幕，应该准备layout和size计算
+ Preload，加载所需要的数据，如下载图片，缓存读取等
+ Display，马上将要进入屏幕，开始进行渲染操作，显示包含的文字或者图片
+ Visible，该node（所对应的view）至少有1个像素已经在屏幕内，正在显示

