---
layout: post
title: "Texture 笔记"
date: 2020-08-06 22:34:00.000000000 +09:00
categories: [Swift]
tags: [Swift, Texture, ASDK]
---

## 闪烁问题

**1.ASNetworkImageNode reload闪烁**

**原因**: `ASCellNode`中包含有`ASNetworkImageNode`时，当这个`cell reload`，`ASNetworkImageNode`会异步从本地缓存或者网络请求图片，请求到图片后再设置`ASNetworkImageNode`展示图片，但在异步过程中`ASNetworkImageNode`会先展示`PlaceholderImage`，从`PlaceholderImage`--->`fetched image`的展示替换导致闪烁发生，即使`cell`的数据没有任何变化，只是简单的`reload`，`ASNetworkImageNode`的图片加载逻辑依然不变，仍然会闪烁。

而对于`UIImageView`，`YYWebImage`或者`SDWebImage`对`UIImageView`的`image`设置逻辑是，先同步检查有无内存缓存，有的话直接显示，没有的话再先显示`PlaceholderImage`，等待加载完成后再显示加载的图片，逻辑是`memory cached image`--->`PlaceholderImage`--->`fetched image`的逻辑，刷新当前`cell`时，如果数据没有变化`memory cached image`一般都会有，因此不会闪烁。

`Texture`官方给的修复方案:

```swift
let node = ASNetworkImageNode()
node.placeholderColor = UIColor.white
node.placeholderFadeDuration = 3
```

上面修改方案确实没有看到闪烁，但是方案是将`PlaceholderImage-->fetched image`图片替换导致的闪烁拉长到3秒，并没有从根本上解决问题。

按照上述reload闪烁的原因，先检查有无缓存，有缓存的话直接设置Image，继承一个`ASNetworkImageNode`的子类，复写`url`设置逻辑：

```swift
class NetworkImageNode: ASDisplayNode {
  private var networkImageNode = ASNetworkImageNode.imageNode()
  private var imageNode = ASImageNode()

  var placeholderColor: UIColor? {
    didSet {
      networkImageNode.placeholderColor = placeholderColor
    }
  }

  var image: UIImage? {
    didSet {
      networkImageNode.image = image
    }
  }

  override var placeholderFadeDuration: TimeInterval {
    didSet {
      networkImageNode.placeholderFadeDuration = placeholderFadeDuration
    }
  }

  var url: URL? {
    didSet {
      // 这里用到SDWebImage缓存机制
      guard let u = url,
        let image = UIImage.cachedImage(with: u) else {
          networkImageNode.url = url
          return
      }

      imageNode.image = image
    }
  }

  override init() {
    super.init()
    addSubnode(networkImageNode)
    addSubnode(imageNode)
  }

  override func layoutSpecThatFits(_ constrainedSize: ASSizeRange) -> ASLayoutSpec {
    return ASInsetLayoutSpec(insets: .zero,
                             child: networkImageNode.url == nil ? imageNode : networkImageNode)
  }

  func addTarget(_ target: Any?, action: Selector, forControlEvents controlEvents: ASControlNodeEvent) {
    networkImageNode.addTarget(target, action: action, forControlEvents: controlEvents)
    imageNode.addTarget(target, action: action, forControlEvents: controlEvents)
  }
}
```

**2.reload 单个cell时的闪烁**

当`ASTableNode/ASCollectionNode`reload某个`indexPath`的`cell`时也会闪烁，原因都是跟`ASNetworkImageNode`差不多，都是异步的问题。当异步计算`cell`的布局时，`cell`使用`placeholder`占位（通常是白图），布局完成时，才用渲染好的内容填充`cell`，`placeholder`到渲染好的内容切换引起闪烁。

官方修复方案:

```swift
func tableNode(_ tableNode: ASTableNode, nodeForRowAt indexPath: IndexPath) -> ASCellNode {
  let node = ASCellNode()
  ... 
  node.neverShowPlaceholders = true
  return node
}
```

设置`node.neverShowPlaceholders = true`，会让`cell`从异步状态衰退回同步状态，若`reload`某个`indexPath`的`cell`，在渲染完成之前，主线程是卡死的，这与`UITableView`的机制一样，但速度会比`UITableView`快很多，因为`UITableView`的布局计算、资源解压、视图合成等都是在主线程进行，而`ASTableNode`则是多个线程并发进行，而且布局等还有缓存。

**3.减缓卡顿**

设置`ASTableNode`的`leadingScreensForBatching = 3`可以减缓列表滚动时卡顿问题，即提前计算3个屏幕的内容:

```swift
tableNode.leadingScreensForBatching = 3
```

**4.reloadData时闪烁**

当下拉列表刷新数据时，调用`ASTableNode/ASCollectionNode`的`reloadData`方法，列表会出现很明显的闪烁现象。修复方案是每次刷新算出需要添加的，删除/刷新的 `indexPath` 或者 `section`，再对这部分调用对应的局部刷新。

```swift
// ASTableNode
// Rows
- (void)insertRowsAtIndexPaths:(NSArray *)indexPaths withRowAnimation:(UITableViewRowAnimation)animation
- (void)deleteRowsAtIndexPaths:(NSArray *)indexPaths withRowAnimation:(UITableViewRowAnimation)animation
- (void)reloadRowsAtIndexPaths:(NSArray *)indexPaths withRowAnimation:(UITableViewRowAnimation)animation
// Sections
- (void)insertSections:(NSIndexSet *)sections withRowAnimation:(UITableViewRowAnimation)animation
- (void)deleteSections:(NSIndexSet *)sections withRowAnimation:(UITableViewRowAnimation)animation
- (void)reloadSections:(NSIndexSet *)sections withRowAnimation:(UITableViewRowAnimation)animation

// ASCollectionNode
// Rows
- (void)insertItemsAtIndexPaths:(NSArray<NSIndexPath *> *)indexPaths
- (void)deleteItemsAtIndexPaths:(NSArray<NSIndexPath *> *)indexPaths
- (void)reloadItemsAtIndexPaths:(NSArray<NSIndexPath *> *)indexPaths
// Sections
- (void)insertSections:(NSIndexSet *)sections
- (void)deleteSections:(NSIndexSet *)sections
- (void)reloadSections:(NSIndexSet *)sections
```

## 关于布局

**1.`flexGrow`**

+ 定义子视图的放大比例，`flexGrow`是指当有多余空间时，拉伸谁以及相应的拉伸比例。
+ 该属性来设置，当父元素的宽度大于所有子元素的宽度的和时（即父元素会有剩余空间），子元素如何分配父元素的剩余空间。
+ `flex-grow`的默认值为0，意思是该元素不索取父元素的剩余空间，如果值大于0，表示索取。值越大，索取的越厉害。

```swift
import UIKit
import AsyncDisplayKit

class ContainerNode: ASDisplayNode {
    
    let nodeA = ASDisplayNode()
    let nodeB = ASDisplayNode()
    override init() {
        super.init()
        self.backgroundColor = .purple
        self.cornerRadius = 16
        
        nodeA.backgroundColor = .orange
        nodeB.backgroundColor = .green
        nodeA.style.preferredSize = CGSize(width: 64, height: 64)
        nodeB.style.preferredSize = CGSize(width: 64, height: 64)
        self.addSubnode(nodeA)
        self.addSubnode(nodeB)
    }
    
    override func layoutSpecThatFits(_ constrainedSize: ASSizeRange) -> ASLayoutSpec {
        let spec1 = ASLayoutSpec()
        spec1.style.flexGrow = 1
        let spec2 = ASLayoutSpec()
        spec2.style.flexGrow = 1
        let spec3 = ASLayoutSpec()
        spec3.style.flexGrow = 1
        
        return ASStackLayoutSpec(direction: .horizontal, spacing: 0, justifyContent: .start, alignItems: .center, children: [spec1, nodeA, spec2, nodeB, spec3])
    }
}
```

![](/assets/images/texture04.png)

如果`spec`的`flexGrow`不同就可以实现指定比例的布局，再结合`width`样式可以实现以下布局。

![](/assets/images/texture05.png)

```swift
override func layoutSpecThatFits(_ constrainedSize: ASSizeRange) -> ASLayoutSpec {
   let spec1 = ASLayoutSpec()
   spec1.style.flexGrow = 2 // 间距比例
   let spec2 = ASLayoutSpec()
   spec2.style.width = ASDimensionMake(20) // 间距宽20
   let spec3 = ASLayoutSpec()
   spec3.style.flexGrow = 1 // 比例
   return ASStackLayoutSpec(direction: .horizontal, spacing: 0, justifyContent: .start, alignItems: .center, children: [spec1, nodeA, spec2, nodeB, spec3])
}
```

**2.`flexShrink`**

+ 缩小比例，当空间不足的时候，缩小子视图，如果所有的子视图都设置为1就是等比例缩小，如果有一个子视图设置为0，表示该视图不缩小。
+ 该属性来设置，当父元素的宽度小于所有子元素的宽度的和时（即子元素会超出父元素），子元素如何缩小自己的宽度的。
+ `flex-shrink`的默认值为1，当父元素的宽度小于所有子元素的宽度的和时，子元素的宽度会减小。值越大，减小的越厉害。如果值为0，表示不减小。

> 举个例子: 父元素宽400px，有两子元素：A和B。A宽为200px，B宽为300px。则A，B总共超出父元素的宽度为(200+300)- 400 = 100px。
>
> 如果A，B都不减小宽度，即都设置flex-shrink为0，则会有100px的宽度超出父元素。如果A不减小宽度:设置flex-shrink为0，B减小。则最终B的大小为 自身宽度(300px)- 总共超出父元素的宽度(100px)= 200px如果A，B都减小宽度，A设置flex-shirk为3，B设置flex-shirk为2。则最终A的大小为 自身宽度(200px)- A减小的宽度(100px * (200px * 3/(200 * 3 + 300 * 2))) = 150px,最终B的大小为 自身宽度(300px)- B减小的宽度(100px * (300px * 2/(200 * 3 + 300 * 2))) = 250px

**3.frame布局**

如果`ASDisplayNode`采用`frame`布局方式，那么它的动画跟`UIView`一样。

```swift
func animateContainer() {    
   let kwidth = UIScreen.main.bounds.width
   DispatchQueue.main.asyncAfter(deadline: .now() + 1) {    
     UIView.animate(withDuration: 0.5) {
       self.containerNode.frame = CGRect(x: (kwidth - 300)/2, y: 160, width: 300, height: 200)
     }
   }
}
```

对于`flexbox`布局，需要复写`Texture`动画API `func animateLayoutTransition(_ context: ASContextTransitioning)`，在动画上下文`context`获取animate前后布局信息，然后自定义动画。

**3.子线程崩溃问题**

由于`Texture`的性能优势来源于异步绘制，异步的意思是有时候`node`会在子线程创建，如果继承了一个`ASDisplayNode`，一不小心在初始化时调用了`UIKit`的相关方法，则会出现子线程崩溃。

```swift
class TestNode {
  let imageNode: ASDisplayNode
  override init() {
    imageNode = ASImageNode()
    // UIImage(named:)并不是线程安全，会崩溃
    imageNode.image = UIImage(named: "test.png") 
    super.init()
  }
}
```

**4.`ASLayoutSpec`**

当 spaceBetween 没有达到两端对齐的效果，尝试设置当前 layoutSpec 的 width，或它的上一级布局对象的 alignItems，在例子中就是 stackLayout.alignItems = .stretch。

```swift
let spec2 = ASLayoutSpec()
spec2.style.width = ASDimensionMake(20) // 间距宽20
```

## 其他问题

**1.关于tintColor**

`ASImageNode` 不支持直接设置图片的 `tintColor` ，如果需要设置，则需要通过 `imageModificationBlock` 进行设置。

```swift
imageNode.imageModificationBlock = ASImageNodeTintColorModificationBlock(.orange)
```

如果是直接向改变`ASButtonNode`中的`tintColor`，是不可以修改的，而是要通过设置`ASButtonNode`不同的状态图片才可以改变的。

```swift
buttonNode.setImage(image, for: .normal)
buttonNode.setImage(image.tinted(with: .orange), for: .selected)
```

**2.关于高度**

`ASDisplayNode`高度计算问题，如果在UIView或者`UITableViewCell/UICollectionViewCell`中添加`ASDisplayNode`，而需要计算`ASDisplayNode`的高度，一种方法本事`ASDisplayNode`是用frame布局的可以直接获取高度；另一种的`ASDisplayNode` 实现了 `- (ASLayoutSpec *)layoutSpecThatFits:(ASSizeRange)constrainedSize` 情况下，可以调用 `- (ASLayout *)layoutThatFits:(ASSizeRange)constrainedSize` 方法计算出大小。

```swift
let layout = node.layoutThatFits(ASSizeRangeMake(CGSize.zero,
						  CGSize(width: view.frame.width,								  		 height: CGFloat.greatestFiniteMagnitude)))
print(node.calculatedSize.height)
```

**3.`ASDisplayNode`问题**

给`ASDisplayNode` 的 `View `添加手势，则需要在`didLoad`方法中添加，这个方法类似于`viewDidLoad`方法，它被调用一次，并且是后台视图被加载的地方。它保证在主线路上被调用，并且是做任何UIKit事情的合适的地方(比如添加手势识别器，触摸视图/层，初始化UIKit对象)。

关于`initWithViewBlock`初始化`ASDisplayNode`时，需要注意`retain cycle`循环引用问题，持有self的变量初始化时需要设置weak弱化。

**4.关于`UITableViewCell`不能点击**

`ASCellNode`里面再嵌套`UITableViewCell`时，在不同版本的手机会出现无法响应点击事件，解决方法:

```swift
let cellNode = ASCellNode { () -> UIView in
	let cell = UITableViewCell(style: .default, reuseIdentifier: nil)
	// ios 10加上
	cell.isUserInteractionEnabled = false
	return cell
}
```

**5.`ASEditableTextNode`显示中文问题**

`ASEditableTextNode` 默认高度只是适应英文字母，如果输入中文会被裁了一截，在初始化时需要指定一下`ASEditableTextNode`的高度。

```swift
let editableTextNode = ASEditableTextNode()
editableTextNode.style.height = ASDimensionMake(44)
```