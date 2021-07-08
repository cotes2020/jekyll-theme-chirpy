---
layout: post
title: "IGListKit 框架学习"
date: 2020-09-21 22:23:00.000000000 +09:00
categories: [Swift]
tags: [Swift, IGListKit, UICollectionView]
---

## 前言

在我们日常的iOS开发中，经常会用到`UITableView/UICollectionView`，由于复用缓存池的存在，在处理庞大数据的情况下，列表还是能保持较低的内存占用，而且简单的`DataSource/Delegate`设计方式，可以通过几行的大卖就可以完成对`UITableView/UICollectionView`的数据和交互的相关配置。但是由于`UITableView`对界面布局的局限性，我们无法在`UITableView`中自定义布局，比如瀑布流之类的样式布局。所以苹果在iOS6.0时推出了`UICollectionView`， 虽然 `UICollectionView` 通过把 `Layout` 抽出来的方式以提供自定义布局，但是在处理数据时本质上和 `UITableView` 还是相同的。 日常使用 `UICollectionView/UITableView` 时都会遇到下面几个问题：

+ 复用流程问题。如果你想在 `UITableView/UICollectionView` 中复用 `Cell` 。那么必须先调用`register(_:forCellReuseIdentifier:)` 来注册对应的 `Cell` 类型。然后在对应的方法中调用 `dequeueReusableCell(withIdentifier:)` 从复用池中取出或者生成一个新的 `Cell` 来使用。如果在获取 `Cell` 前忘记调用 `register` 进行注册，那么你就会收获一个应用崩溃。为什么这样做？ [register(_:forCellReuseIdentifier:)](https://developer.apple.com/documentation/uikit/uitableview/1614888-register) 有简单说明。
+ 不支持`diff`。在数据源变化需要更新对应的界面时，苹果并没提供相关的 API 实现删除、插入或者更新的位置`indexPath`，需要去计算找出更新的数据源，然后删除、插入或者更新对应的 `Cell` 。由于有复用机制的存在，直接使用 `reloadData` 也不会有什么问题，复用机制使得在 `reloadData` 时只会对显示在屏幕上的 `Cell` 进行处理，需要处理视图数量并不多，但是应该有更好的办法来支持 `diff`。（ PS ：苹果在 WWDC19 上推出 Diffable Data Source 功能，支持局部刷新）
+ 模块的隔离。在复杂的界面处理数据的获取、处理 `Cell` 的配置、处理埋点统计之类的功能时，对于这些模块的隔离，系统本身的框架并没有提供这么一套东西给我们，所以很多时候会写在 `ViewController` 里，于是变成了 `Massive View Controller` 。如果自行设计一套方案，很容易弄出一套不但使用起来非常难受，而且还难以迁移的框架。

## IGListKit介绍

`IGListKit` 是 Instagram 在2016年年底推出的一套数据驱动的 `UICollectionView` 框架，使用数据驱动去创造更为快速灵活的列表界面。为什么选择 `UICollectionView` 而不是 `UITableView` ？因为 `UICollectionView` 支持自定义布局，比 `UITableView` 更加灵活。 `IGListKit` 的主要特性如下：

+ 数据驱动 (数据改变 -> Diff算法 -> update界面)
+ 支持多种数据类型
+ 支持对自己的数据模型进行自定义 `diff` 操作
+ 与 `diff` 算法解耦
+ 使用 `UICollectionView` 更加简单
+ 可扩展的 API 
+ 更好的架构以复用 `Cell` 和组件
+ 不再需要调用 `performBatchUpdates(_:, completion:)` 或者 `reloadData()` 

`IGListKit`数据处理流程:

![](/assets/images/swift-iglistkit-01.png)

由上图可以看出数据经由 `Adapter` 进行处理后转给对应的 `SectionController` ，而 `SectionController` 会根据不同的数据类型返回给不同的 `Cell` 。在这个过程中， `UIViewController` 只和 `Adapter` 、 `Updater` 进行交互，根据不同的数据类型返回不同的 `SectionController` 。而对于 `Cell` 的处理则完全交给 `SectionController` ，这一步的好处是 `SectionController` 可以进行复用。因为我们可能需要对不同的 `UIViewController` 进行配置，但是有很大可能它们的 `Cell` 显示方式都是相同，只是其它一些逻辑或者 UI 不同。我们也可以经由 `SectionController` 进行封装，合成不同的界面。

## IGListKit使用

通过使用`IGListKit`创建一个列表界面就会变得非常容易。只需要创建一个`IGListSectionController`子类，重写下面两个方法就可以:

+ `cellForItemAtIndex:`
+ `sizeForItemAtIndex`

```swift
class TestSectionController: ListSectionController {
  
  override func sizeForItem(at index: Int) -> CGSize {
    return CGSize(width: collectionContext!.containerSize.width, height: 50)
  }

  override func cellForItem(at index: Int) -> UICollectionViewCell {
    return collectionContext!.dequeueReusableCell(of: TestCell.self, for: self, at: index)
  }
}
```

 `Cell` 的配置和数据的处理都交给了 `SectionController` 来进行。完成 `TestSectionController` 的配置后，需要在 `UIViewController` 或者其它什么地方把各个模块串联起来。

```swift
let layout = UICollectionViewFlowLayout()
let collectionView = UICollectionView(frame: .zero, collectionViewLayout: layout)

let updater = ListAdapterUpdater()
let adapter = ListAdapter(updater: updater, viewController: self)
adapter.collectionView = collectionView
```

从上面可以看出 `UICollectionView` 设置布局 `Layout` 部分没有变动，但是使用了 `ListAdapter` 和 `ListAdapterUpdater` 将 `UIViewController` 和 `UICollectionView` 串起来。这里使用了默认的 `UICollectionViewFlowLayout` 和 `IGListAdapterUpdater` ，也可以通过配置自定义的类来使用一些高级特性。

通过给 `adapter` 设置 `dataSource` 

```swift
adapter.dataSource = self

func objects(for listAdapter: ListAdapter) -> [ListDiffable] {
  return [ "Foo", "Bar", 42, "Biz" ]
}

func listAdapter(_ listAdapter: ListAdapter, sectionControllerFor object: Any) -> ListSectionController {
  if object is String {
    return LabelSectionController()
  } else {
    return NumberSectionController()
  }
}
/// 可以返回任何数据类型, 需支持 IGListDiffable 协议
func emptyView(for listAdapter: ListAdapter) -> UIView? {
  return nil
}
```

注意: 返回的数据是不可变的，当返回可变的对象且会对它进行编辑时，`IGListKit` 就无法正确地计算出它们之间的差异。原因是对象已经发生了改变，对象的改变就会丢失。所以可以返回一个新的不可变的对象，而且支持`IGListDiffable`协议。

## IGListSectionController

`IGListSectionController`作为`IGListLit`的基石，`IGListSectionController` 跟 `Object` 是一一对应的关系，在 `IGListAdapterDataSource` 的 `listAdapter:sectionControllerForObject:` 方法中，会根据不同的 `Object` 返回不同的 `IGListSectionController` 。这跟`UICollectinoView` 的 `Section` 是不同，因为无法将`Object` 的数组和 `IGListSectionController` 绑定，如果要将数组绑定到一个 `IGListSectionController` 中，需要将 `Object` 的数组用一个 `Wrapper` 封装起来，再将其和 `IGListSectionController` 绑定。

一般情况下需要对 `IGListSectionController` 进行自定义，通过 `IGListSectionController` 的 `didUpdateToObject:` 方法更新自身的`Object` 。同时 `IGListSectionController` 也提供了与 `Cell` 进行交互的相关方法：

```swift
- (void)didSelectItemAtIndex:(NSInteger)index;
- (void)didDeselectItemAtIndex:(NSInteger)index;
- (void)didHighlightItemAtIndex:(NSInteger)index;
- (void)didUnhighlightItemAtIndex:(NSInteger)index;
- (void)didUnhighlightItemAtIndex:(NSInteger)index;
```

 `IGListSectionController` 已经包含了 `Cell` 的配置和交互，所以可以作为一个模块来复用。对于相同的 `Object` 交互和界面都一致，则可以返回相同的 `IGListSectionController` ，即便是在不同的 `ViewController` 中。同时 `IGListSectionController` 也提供了一些 `Delegate` 和 `DataSource` ，在对应的时机会进行调用。

```swift
- (NSArray<NSString *> *)supportedElementKinds;
- (__kindof UICollectionReusableView *)viewForSupplementaryElementOfKind:(NSString *)elementKind  atIndex:(NSInteger)index;
- (CGSize)sizeForSupplementaryViewOfKind:(NSString *)elementKind atIndex:(NSInteger)index;
```

`id <IGListSupplementaryViewSource> supplementaryViewSource` ，用于配置 `UICollectionView` 每个 `section` 的 `supplementary views`。同时也可以设置 `supplementaryViewSource`作为 为 `IGListSectionController` 自己的，在 `IGListSectionController` 就内部就可以完成 `supplementary views` 的配置，复用时更加方便。

### IGListSectionControllerThreadContext

`IGListSectionController` 在初始化时需要获取到当前的 `UIViewController *viewController` 和 `id <IGListCollectionContext> collectionContext` 。

+ `viewController` 可用于 `push` 、 `pop` 、 `present` 或者自定义转场，对于 `IGListSectionController` 来说，它只知道这是一个 `UIViewController` ，不知道它的具体类型，因为 `IGListSectionController` 是可复用的，我们有可能将其和不同的 `UIViewController` 连接起来，所以其 `viewController` 的类型是不确定的。
+ `collectionContext` 为了限制可以调用的接口，使用 `protocol` 对其进行抽象， `collectionContext` 本质上是一个 `IGListAdapter` ，`IGListAdapter` 可能有一些不需要或者不想提供给外界访问的接口，所以通过 `protocol` 来进行抽象。

```swift
- (instancetype)init {
    if (self = [super init]) {
        IGListSectionControllerThreadContext *context = [threadContextStack() lastObject];
        _viewController = context.viewController;
        _collectionContext = context.collectionContext;
        if (_collectionContext == nil) {
            IGLKLog(@"Warning: Creating %@ outside of -[IGListAdapterDataSource listAdapter:sectionControllerForObject:]. Collection context and view controller will be set later.",
                    NSStringFromClass([self class]));
        }
		  /// ...
    }
    return self;
}
```

通过上面`IGListSectionController`的初始化方法是交给调用方来定义的，`IGListKit` 无法确定 `IGListSectionController` 的初始化参数，为了使得初始化方法尽量简洁，不包含太多参数，使用了 `Thread Dictionary` 来存储所对应的对象，通过 `threadContextStack()` 来获取最新的 `IGListSectionControllerThreadContext` ：

```swift
static NSString * const kIGListSectionControllerThreadKey = @"kIGListSectionControllerThreadKey";
static NSMutableArray<IGListSectionControllerThreadContext *> *threadContextStack(void) {
    IGAssertMainThread();
    NSMutableDictionary *threadDictionary = [[NSThread currentThread] threadDictionary];
    NSMutableArray *stack = threadDictionary[kIGListSectionControllerThreadKey];
    if (stack == nil) {
        stack = [NSMutableArray new];
        threadDictionary[kIGListSectionControllerThreadKey] = stack;
    }
    return stack;
}
```

`threadContextStack()` 通过 `[[NSThread currentThread] threadDictionary]` 来获取对应的 `stack` ，而 `push` 和 `pop` 方法则是在 `IGListAdapter` 初始化 `IGListSectionController` 前后调用：

```swift
void IGListSectionControllerPushThread(UIViewController *viewController, id<IGListCollectionContext> collectionContext) {
    IGListSectionControllerThreadContext *context = [IGListSectionControllerThreadContext new];
    context.viewController = viewController;
    context.collectionContext = collectionContext;

    [threadContextStack() addObject:context];
}
void IGListSectionControllerPopThread(void) {
    NSMutableArray *stack = threadContextStack();
    IGAssert(stack.count > 0, @"IGListSectionController thread stack is empty");
    [stack removeLastObject];
}
```

### IGListGenericSectionController

`IGListSectionController` 提供了特定的类型`IGListGenericSectionController`，作用相当于一个强类型的 `IGListSectionController`，有关[问题](https://github.com/Instagram/IGListKit/issues/682)。

```swift
@interface IGListGenericSectionController<__covariant ObjectType> : IGListSectionController
@property (nonatomic, strong, nullable, readonly) ObjectType object;
- (void)didUpdateToObject:(ObjectType)object NS_REQUIRES_SUPER;
@end
```

### IGListSingleSectionController

对于单个 `Cell` 的 `Section ViewController` ， `IGListKit` 提供了 `IGListSingleSectionController` 来使用，它通过 `block` 配置 `Cell` 和通过 `Delegate` 来获取点击事件的回调。

```swift
@protocol IGListSingleSectionControllerDelegate <NSObject>
- (void)didSelectSectionController:(IGListSingleSectionController *)sectionController
                        withObject:(id)object;
@optional

- (void)didDeselectSectionController:(IGListSingleSectionController *)sectionController
                          withObject:(id)object;
@end

@interface IGListSingleSectionController: IGListSectionController
- (instancetype)initWithCellClass:(Class)cellClass
                   configureBlock:(IGListSingleSectionCellConfigureBlock)configureBlock
                        sizeBlock:(IGListSingleSectionCellSizeBlock)sizeBlock;
- (instancetype)initWithNibName:(NSString *)nibName
                         bundle:(nullable NSBundle *)bundle
                 configureBlock:(IGListSingleSectionCellConfigureBlock)configureBlock
                      sizeBlock:(IGListSingleSectionCellSizeBlock)sizeBlock;
- (instancetype)initWithStoryboardCellIdentifier:(NSString *)identifier
                                  configureBlock:(IGListSingleSectionCellConfigureBlock)configureBlock
                                       sizeBlock:(IGListSingleSectionCellSizeBlock)sizeBlock;
@property (nonatomic, weak, nullable) id<IGListSingleSectionControllerDelegate> selectionDelegate;
```

### IGListBindingSectionController

对于 `Section` 的数据流绑定， `IGListKit` 则提供了 `IGListBindingSectionController` 。 `IGListBindingSectionController` 通过 `id<IGListBindingSectionControllerDataSource> dataSource` 的方式把顶层的 `Object` 转换为 `NSArray<id<IGListDiffable>> viewModels` ，然后调用支持 `IGListBindable` 协议的 `Cell` 中的 `bindViewModel:` 方法来刷新 `Cell` 。

如果 `Object` 是跟 `IGListBindingSectionController` 匹配，那么在处理 `IGListDiffable` 协议的方法时需要非常小心。 `IGListDiffable` 通过 `-diffIdentifier` 来判断两个 `Object` 是否为同一个，再通过 `-isEqualToDiffableObject:` 方法来判断 `Object` 是否有更新。由于 `IGListBindingSectionController` 已经在内部消化了 `Object` 的更新逻辑，所以如果是跟 `IGListBindingSectionController` 绑定的 `Object` ，其对应的 `-isEqualToDiffableObject:` 方法应该一直返回 `YES` 。

```swift
- (BOOL)isEqualToDiffableObject:(id)object {
  return YES;
}

func isEqual(toDiffableObject object: IGListDiffable?) -> Bool { 
  return true 
}
```

`IGListBindingSectionController` 内部重写了一些方法，通过 `IGListBindingSectionControllerDataSource` 和 `IGListBindingSectionControllerSelectionDelegate` 来和外部进行交互。

```swift
- (void)didUpdateToObject:(id)object {
    id oldObject = self.object;
    self.object = object;
    if (oldObject == nil) {
        NSArray *viewModels = [self.dataSource sectionController:self viewModelsForObject:object];
        self.viewModels = objectsWithDuplicateIdentifiersRemoved(viewModels);
    } else {
#if defined(IGLK_LOGGING_ENABLED) && IGLK_LOGGING_ENABLED
        if (![oldObject isEqualToDiffableObject:object]) {
            IGLKLog(@"Warning: Unequal objects %@ and %@ will cause IGListBindingSectionController to reload the entire section",
                    oldObject, object);
        }
#endif
        [self updateAnimated:YES completion:nil];
    }
}
```

在 `didUpdateToObject:` 方法中，通过 `dataSource` 返回 `viewModels` ，移除重复部分后设置为 `self.viewModels` 。由于 `IGListBindingSectionController` 在内部处理了 `Object` 的更新逻辑，所以 `didUpdateToObject:` 只会调用一次，如果多次调用且新旧 `Object` 不相等，则表示设置错误，打印一个 `warning` ，但是同时也调用 `updateAnimated:completion:` 更新 `Cell` 。

当对 `Object` 进行修改后，如果需要更新 `Cell` ，调用 `IGListBindingSectionController` 的 `updateAnimated:completion:` 方法：

```swift
- (void)updateAnimated:(BOOL)animated completion:(void (^)(BOOL))completion {
    // 1. 如果不是空闲状态，则直接返回，调用 `completion(NO)` ；
    if (self.state != IGListDiffingSectionStateIdle) {
        if (completion != nil) {
            completion(NO);
        }
        return;
    }
    self.state = IGListDiffingSectionStateUpdateQueued;

    __block IGListIndexSetResult *result = nil;
    __block NSArray<id<IGListDiffable>> *oldViewModels = nil;

    id<IGListCollectionContext> collectionContext = self.collectionContext;
    [self.collectionContext performBatchAnimated:animated updates:^(id<IGListBatchContext> batchContext) {
        if (self.state != IGListDiffingSectionStateUpdateQueued) {
            return;
        }
        oldViewModels = self.viewModels;
        id<IGListDiffable> object = self.object;
        NSArray *newViewModels = [self.dataSource sectionController:self viewModelsForObject:object];
        self.viewModels = objectsWithDuplicateIdentifiersRemoved(newViewModels);
        // 2. 通过 IGListDiff 的算法计算出需要操作的位置
        result = IGListDiff(oldViewModels, self.viewModels, IGListDiffEquality);
        // 3. 遍历 `updates` ，首先获取 `Cell` 所对应的 `index` ，
        // 也就是旧的 `viewModels` 中的 `index` ，通过 `index` 获取到 `Cell` ，
        // 然后使用新的 `index` 获取到新的 `viewModel` ，`Cell` 通过新的 `viewModel` 进行更新
        [result.updates enumerateIndexesUsingBlock:^(NSUInteger oldUpdatedIndex, BOOL *stop) {
            id identifier = [oldViewModels[oldUpdatedIndex] diffIdentifier];
            const NSInteger indexAfterUpdate = [result newIndexForIdentifier:identifier];
            if (indexAfterUpdate != NSNotFound) {
                UICollectionViewCell<IGListBindable> *cell = [collectionContext cellForItemAtIndex:oldUpdatedIndex
                                                                                 sectionController:self];
                [cell bindViewModel:self.viewModels[indexAfterUpdate]];
            }
        }];
        if (IGListExperimentEnabled(self.collectionContext.experiments, IGListExperimentInvalidateLayoutForUpdates)) {
            [batchContext invalidateLayoutInSectionController:self atIndexes:result.updates];
        }
        // 4. 通过 result 进行 `Cell` 的删除/插入
        [batchContext deleteInSectionController:self atIndexes:result.deletes];
        [batchContext insertInSectionController:self atIndexes:result.inserts];
        for (IGListMoveIndex *move in result.moves) {
            [batchContext moveInSectionController:self fromIndex:move.from toIndex:move.to];
        }
        self.state = IGListDiffingSectionStateUpdateApplied;
    } completion:^(BOOL finished) {
        self.state = IGListDiffingSectionStateIdle;
        if (completion != nil) {
            completion(YES);
        }
    }];
}
```

在获取 `Cell` 时， `IGListBindingSectionController` 也会自行进行 `Cell` 和 `Object` 的绑定， `Cell` 需要遵循 `IGListBindable` 协议：

```swift
// protocol IGListBindable
@protocol IGListBindable <NSObject>
- (void)bindViewModel:(id)viewModel;
@end
- (UICollectionViewCell *)cellForItemAtIndex:(NSInteger)index {
    id<IGListDiffable> viewModel = self.viewModels[index];
    UICollectionViewCell<IGListBindable> *cell = [self.dataSource sectionController:self cellForViewModel:viewModel atIndex:index];
    [cell bindViewModel:viewModel];
    return cell;
}
```

从上面的代码可以看到 `IGListBindingSectionController` 也提供了基于 `Cell` 进行局部刷新的能力。

### IGListSectionMap

`IGListSectionMap` 提供了一种在常数时间内对 `Object` 和 `SectionController` 进行互相映射的方式。 主要方法有以下几种:

1.根据 `section` 返回对应的 `IGListSectionController`:

```swift
/// 根据 section 返回对应的 IGListSectionController
- (nullable IGListSectionController *)sectionControllerForSection:(NSInteger)section;
/// 根据 section 返回对应的 Object
- (nullable id)objectForSection:(NSInteger)section;
/// 根据 object 返回对应的 IGListSectionController
- (nullable id)sectionControllerForObject:(id)object;
/// 根据 sectionController 返回对应的 Section Index
- (NSInteger)sectionForSectionController:(id)sectionController;
/// 根据 object 返回对应的 Section Index
- (NSInteger)sectionForObject:(id)object;
```

`IGListSectionMap` 内部实现：

```swift
@interface IGListSectionMap ()

@property (nonatomic, strong, readonly, nonnull) NSMapTable<id, IGListSectionController *> *objectToSectionControllerMap;
@property (nonatomic, strong, readonly, nonnull) NSMapTable<IGListSectionController *, NSNumber *> *sectionControllerToSectionMap;
@property (nonatomic, strong, nonnull) NSMutableArray *mObjects;

@end
```

`objectToSectionControllerMap` 提供了 `Object` 到 `sectionController` 的映射， `sectionControllerToSectionMap` 提供了 `sectionController` 到 `section` 的映射， `mObjects` 提供了根据 `section` 返回 `index` 的方法。

2.`IGListSectionMap` 也提供了接口对属性进行批量更新：

```swift
- (void)updateWithObjects:(NSArray *)objects sectionControllers:(NSArray *)sectionControllers {
    
    [self reset];
    self.mObjects = [objects mutableCopy];
    id firstObject = objects.firstObject;
    id lastObject = objects.lastObject;

    [objects enumerateObjectsUsingBlock:^(id object, NSUInteger idx, BOOL *stop) {
        IGListSectionController *sectionController = sectionControllers[idx];

        [self.sectionControllerToSectionMap setObject:@(idx) forKey:sectionController];
        [self.objectToSectionControllerMap setObject:sectionController forKey:object];
        sectionController.isFirstSection = (object == firstObject);
        sectionController.isLastSection = (object == lastObject);
        sectionController.section = (NSInteger)idx;
    }];
}
```

3.重置现有数据主要是对 `IGListSectionController` 相关属性的清理：

```swift
- (void)reset {
    [self enumerateUsingBlock:^(id  _Nonnull object, IGListSectionController * _Nonnull sectionController, NSInteger section, BOOL * _Nonnull stop) {
        sectionController.section = NSNotFound;
        sectionController.isFirstSection = NO;
        sectionController.isLastSection = NO;
    }];
    [self.sectionControllerToSectionMap removeAllObjects];
    [self.objectToSectionControllerMap removeAllObjects];
}
```

### WorkingRange

![](/assets/images/swift-iglistkit-02.png)

`Working range` 表示还没出现在屏幕上，但是已经在附近的 `IGListSectionController` ， `IGListSectionController` 可以在进入或者退出 `Working range` 时获取对应的通知，借此可以进行一些准备工作，比如说预先下载图片。 `IGListAdapter` 可以在初始化时指定 `Working range` 的大小：

```swift
let adapter = ListAdapter(updater: ListAdapterUpdater(), viewController: self, workingRangeSize: 1)
```

可以通过给 `IGListSectionController` 设置 `workingRangeDelegate` 来获取对应的回调。 `IGListKit` 内部提供了一个 `IGListWorkingRangeHandler` ，在 `UICollectionViewDelegate` 的 `willDisplay/didEndDisplaying` 方法中调用 `IGListWorkingRangeHandler` 对应的方法：

```swift
- (void)willDisplayItemAtIndexPath:(NSIndexPath *)indexPath
                    forListAdapter:(IGListAdapter *)listAdapter;
- (void)didEndDisplayingItemAtIndexPath:(NSIndexPath *)indexPath
                         forListAdapter:(IGListAdapter *)listAdapter;

- (void)willDisplayItemAtIndexPath:(NSIndexPath *)indexPath
                    forListAdapter:(IGListAdapter *)listAdapter {
    _visibleSectionIndices.insert({
        .section = indexPath.section,
        .row = indexPath.row,
        .hash = indexPath.hash
    });

    [self _updateWorkingRangesWithListAdapter:listAdapter];
}

- (void)didEndDisplayingItemAtIndexPath:(NSIndexPath *)indexPath
                         forListAdapter:(IGListAdapter *)listAdapter {
    _visibleSectionIndices.erase({
        .section = indexPath.section,
        .row = indexPath.row,
        .hash = indexPath.hash
    });

    [self _updateWorkingRangesWithListAdapter:listAdapter];
}
```

为了效率更高， `IGListWorkingRangeHandler` 内部是基于 C++ 实现的，内部定义了 `_visibleSectionIndices` 和 `_workingRangeSectionControllers` 两个 `std::unordered_set` 的变量。每次更新 `Cell` 的显示隐藏状态时都会更新 `_visibleSectionIndices` ，然后再调用 `_updateWorkingRangesWithListAdapter:` 方法：

```swift
- (void)_updateWorkingRangesWithListAdapter:(IGListAdapter *)listAdapter {
    // 1. 由于需要顺序的 `set` ，所以这里使用了 `std::set` ；
    std::set<NSInteger> visibleSectionSet = std::set<NSInteger>();
    // 2. 插入所有可见的 `section` ；
    for (const _IGListWorkingRangeHandlerIndexPath &indexPath : _visibleSectionIndices) {
        visibleSectionSet.insert(indexPath.section);
    }

    NSInteger start;
    NSInteger end;
    // 3. 计算出开始和结束位置；
    if (visibleSectionSet.size() == 0) {
        start = 0;
        end = 0;
    } else {
        start = MAX(*visibleSectionSet.begin() - _workingRangeSize, 0);
        end = MIN(*visibleSectionSet.rbegin() + 1 + _workingRangeSize, (NSInteger)listAdapter.objects.count);
    }
     
    // 4. 创建新的 `workingRangeSectionControllers` ；
    _IGListWorkingRangeSectionControllerSet workingRangeSectionControllers (visibleSectionSet.size());
    for (NSInteger idx = start; idx < end; idx++) {
        id item = [listAdapter objectAtSection:idx];
        IGListSectionController *sectionController = [listAdapter sectionControllerForObject:item];
        workingRangeSectionControllers.insert({sectionController});
    }

    // 5. 遍历新的 `workingRangeSectionControllers` ，如果不在旧的 `_workingRangeSectionControllers` 中，
    // 则表示这个 `sectionController` 是新加入的，调用 `sectionControllerWillEnterWorkingRange` ；
    for (const _IGListWorkingRangeHandlerSectionControllerWrapper &wrapper : workingRangeSectionControllers) {
        auto it = _workingRangeSectionControllers.find(wrapper);
        if (it == _workingRangeSectionControllers.end()) {
            id <IGListWorkingRangeDelegate> workingRangeDelegate = wrapper.sectionController.workingRangeDelegate;
            [workingRangeDelegate listAdapter:listAdapter sectionControllerWillEnterWorkingRange:wrapper.sectionController];
        }
    }

    // 6. 遍历旧的 `_workingRangeSectionControllers` ，如果不在新的 `workingRangeSectionControllers` 中，
    // 则表示这个 `sectionController` 是已退出的，调用 `sectionControllerDidExitWorkingRange` ；
    for (const _IGListWorkingRangeHandlerSectionControllerWrapper &wrapper : _workingRangeSectionControllers) {
        auto it = workingRangeSectionControllers.find(wrapper);
        if (it == workingRangeSectionControllers.end()) {
            id <IGListWorkingRangeDelegate> workingRangeDelegate = wrapper.sectionController.workingRangeDelegate;
            [workingRangeDelegate listAdapter:listAdapter sectionControllerDidExitWorkingRange:wrapper.sectionController];
        }
    }

    _workingRangeSectionControllers = workingRangeSectionControllers;
}
```

可以看到由于 `workingRange` 是以 `Section` 为单位，所以无法提供精细到 `Cell `级别的预处理。这也是基于 `SectionController` 进行处理的缺点。

### DisplayHandler

`IGListDisplayHandler` 是 `IGListKit` 内部用于处理 `Cell` 显示/消失在屏幕上的相关事件。 `IGListAdapter` 在 `UICollectionViewDelegate` 的 `willDisplay/didEndDisplaying` 方法中调用 `IGListDisplayHandler` 对应的方法：

```swift
- (void)willDisplayCell:(UICollectionViewCell *)cell
         forListAdapter:(IGListAdapter *)listAdapter
      sectionController:(IGListSectionController *)sectionController
                 object:(id)object
              indexPath:(NSIndexPath *)indexPath;
- (void)didEndDisplayingCell:(UICollectionViewCell *)cell
              forListAdapter:(IGListAdapter *)listAdapter
           sectionController:(IGListSectionController *)sectionController
                   indexPath:(NSIndexPath *)indexPath;
- (void)willDisplaySupplementaryView:(UICollectionReusableView *)view
                      forListAdapter:(IGListAdapter *)listAdapter
                   sectionController:(IGListSectionController *)sectionController
                              object:(id)object
                           indexPath:(NSIndexPath *)indexPath;
- (void)didEndDisplayingSupplementaryView:(UICollectionReusableView *)view
                           forListAdapter:(IGListAdapter *)listAdapter
                        sectionController:(IGListSectionController *)sectionController
                                indexPath:(NSIndexPath *)indexPath;
```

`IGListDisplayHandler` 内部使用了 `NSCountedSet<IGListSectionController *> *visibleListSections` 来记录可见的 `IGListSectionController` ，跟 `NSMutableSet` 的不同之处在于， `NSCountedSet` 会记录每个 `Object` 添加的次数。 `IGListDisplayHandler` 还定义了一个 `NSMapTable *visibleViewObjectMap` 属性，用于处理 `UICollectionReusableView` 跟 `Object` 的对应关系。

`_pluckObjectForView:` 移除并返回 `UICollectionReusableView` 对应的 `Object` ：

```swift
- (id)_pluckObjectForView:(UICollectionReusableView *)view {
    NSMapTable *viewObjectMap = self.visibleViewObjectMap;
    id object = [viewObjectMap objectForKey:view];
    [viewObjectMap removeObjectForKey:view];
    return object;
}
```

`IGListDisplayHandler` 内部的 `willDisplay` 代码如下：

```swift
- (void)_willDisplayReusableView:(UICollectionReusableView *)view
                 forListAdapter:(IGListAdapter *)listAdapter
              sectionController:(IGListSectionController *)sectionController
                         object:(id)object
                      indexPath:(NSIndexPath *)indexPath {
    [self.visibleViewObjectMap setObject:object forKey:view];
    NSCountedSet *visibleListSections = self.visibleListSections;
    if ([visibleListSections countForObject:sectionController] == 0) {
        [sectionController willDisplaySectionControllerWithListAdapter:listAdapter];
        [listAdapter.delegate listAdapter:listAdapter willDisplayObject:object atIndex:indexPath.section];
    }
    [visibleListSections addObject:sectionController];
}
```

在 `willDisplay` 的处理中，如果 `countForObject` 为 0 则表示该 `sectionController` 即将要进入屏幕，随后调用 `sectionController` 和 `listAdapter.delegate` 的方法。然后可以看到调用 `[visibleListSections addObject:]` 添加对应的 `sectionController` ，由于 `visibleListSections` 是 `NSCountedSet` ，所以会记录 `sectionController` 的次数，可以配合后续的 `didEndingDisplay` 操作。

```swift
- (void)_didEndDisplayingReusableView:(UICollectionReusableView *)view
                      forListAdapter:(IGListAdapter *)listAdapter
                   sectionController:(IGListSectionController *)sectionController
                              object:(id)object
                           indexPath:(NSIndexPath *)indexPath {
    if (object == nil || sectionController == nil) {
        return;
    }
    const NSInteger section = indexPath.section;
    NSCountedSet *visibleSections = self.visibleListSections;
    [visibleSections removeObject:sectionController];
    if ([visibleSections countForObject:sectionController] == 0) {
        [sectionController didEndDisplayingSectionControllerWithListAdapter:listAdapter];
        [listAdapter.delegate listAdapter:listAdapter didEndDisplayingObject:object atIndex:section];
    }
}
```

可以看到在 `didEndDisplaying` 时， `visibleSections` 每次 `removeObject:sectionController` 都会使得 `sectionController` 的计数减一，只有当计数为 0 时才调用 `sectionController` 和 `listAdapter.delegate` 对应的方法。

`IGListDisplayHandler` 的内部实现为 `willDisplay/didEndDisplaying` 提供了两个层级的入口:

+ `IGListAdapter` 级别，通过设置 `adapter` 的 `id <IGListAdapterDelegate> delegate` ，可以获取整个 `UICollectionView` 的回调。
+ `IGListSectionController` ，通过设置 `id <IGListDisplayDelegate> displayDelegate` ，可以获取具体到某个 `sectionController` 的回调。也支持设置 `displayDelegate` 为 `IGListSectionController` 它自己，由于 `IGListSectionController` 跟 `Object` 是绑定的，所以在处理不同的 `ViewController` 中相同的 `Object` 时，我们不仅可以复用 `IGListSectionController` ，也可以复用 `displayDelegate` 的配置，进行一些曝光时长的统一配置。

### IGListSectionController总结

可以看到 `IGListSectionController` 作为 `IGListKit` 的基石，直接和数据层进行绑定，而且 `IGListKit` 还通过 `IGListSectionController` 进行各种扩展，支持以下特性：

- 支持范型特性，设置指定的数据类型；
- 支持快捷只显示单个 `Cell` 的 `Section` ；
- 支持数据流绑定， `Section` 内根据不同的数据刷新不同的 `Cell` ；
- 支持预处理，预处理的范围也可以进行设置；
- 支持设置显示时的相关回调，且可以基于 `IGListSectionController` 的层级进行操作。

## IGListAdapter

`IGListAdapter` 作为 `IGListKit` 的适配器，对外提供相关的刷新接口和一些通用方法，对内负责管理 `IGListSectionController` 和 `UICollectionView` ，调用 `dataSource` 和 `delegate` 。

![](/assets/images/swift-iglistkit-03.png)

它负责处理 `UICollectionView` 的 `DataSource` 和 `Delegate` ，所有 `DataSource/Delegate` 的相关方法都会在 `IGListAdapter` 内部消化完毕，调用方只需要设置 `IGListAdapter` 的 `dataSource` 和 `collectionView` 即可， `IGListAdapterDataSource `则负责给 `IGListAdapter` 提供数据源。

### 初始化

```swift
@protocol IGListAdapterDataSource <NSObject>

/// 根据不同的 adapter 返回需要展示在列表中的数据，一般情况下每个 UIViewController 只有一个 adapter
- (NSArray<id <IGListDiffable>> *)objectsForListAdapter:(IGListAdapter *)listAdapter;
/// 根据数据来返回新生成的对应的 IGListSectionController
/// IGListSectionController 应该在这里进行初始化，你也可以在这里传递其它数据给 IGListSectionController 。
/// 当 IGListAdapter 被创建，更新或者重新加载（ reloaded ）时，会初始化所有数据对应的 IGListSectionController 。 
/// IGListSectionController 会进行复用，可以通过 `-[IGListDiffable diffIdentifier]` 来阻止。
- (IGListSectionController *)listAdapter:(IGListAdapter *)listAdapter sectionControllerForObject:(id)object;
/// 当 UICollectionView 数据为空时就会显示这个方法返回的 UIView ，如果不想显示，可以直接返回 `nil` 。
- (nullable UIView *)emptyViewForListAdapter:(IGListAdapter *)listAdapter;

@end
```

`IGListAdapterDataSource` 作用和 `UICollectionViewDataSource` 类似，只不过设置对象变成了 `IGListAdapterDataSource` ，而且只需要提供数据源和 `IGListSectionController` 即可，不需要进行其它配置。

`IGListAdapter` 的初始化方法

```swift
- (instancetype)initWithUpdater:(id <IGListUpdatingDelegate>)updater
                 viewController:(UIViewController *)viewController
               workingRangeSize:(NSInteger)workingRangeSize {
    IGAssertMainThread();
    IGParameterAssert(updater);

    if (self = [super init]) {
		    // 1. 使用了 `NSMapTable` 而不是 `NSDictionary` ，因为这里的 `key` 是 `id<IGListDiffable>`  对象，不支持 `NSCopying` 协议，
        // 所以使用 `NSMapTable` ，通过 `id <IGListUpdatingDelegate>` 的 `objectLookupPointerFunctions` 方法来自定义 `hashFunction` 和 `isEqualFunction` ：
        NSPointerFunctions *keyFunctions = [updater objectLookupPointerFunctions];
        NSPointerFunctions *valueFunctions = [NSPointerFunctions pointerFunctionsWithOptions:NSPointerFunctionsStrongMemory];
        NSMapTable *table = [[NSMapTable alloc] initWithKeyPointerFunctions:keyFunctions valuePointerFunctions:valueFunctions capacity:0];
        _sectionMap = [[IGListSectionMap alloc] initWithMapTable:table];
	      // 2. `IGListDisplayHandler` ，提供 `UICollectionViewCell` 和  `UICollectionReusableView` 显示/隐藏相关的生命周期方法，内部调用 `IGListSectionController` 对应的方法
        _displayHandler = [IGListDisplayHandler new];
        // 3. `IGListWorkingRangeHandler` ，通过设置 `workingRangeSize` ，可以在 `UICollectionView` 滑动时做一些预处理工作
        _workingRangeHandler = [[IGListWorkingRangeHandler alloc] initWithWorkingRangeSize:workingRangeSize];
        // 4. `NSHashTable<id<IGListAdapterUpdateListener>> *_updateListeners` ，在 `UICollectionView` 完成更新操作后调用
        _updateListeners = [NSHashTable weakObjectsHashTable];
        // 5.  `NSMapTable<UICollectionReusableView *, IGListSectionController *> *_viewSectionControllerMap` ，
		  // 维护 `sectionController` 和 `UICollectionReusableView` 映射关系。
        _viewSectionControllerMap = [NSMapTable mapTableWithKeyOptions:NSMapTableObjectPointerPersonality | NSMapTableStrongMemory
                                                  valueOptions:NSMapTableStrongMemory];
        _updater = updater;
        _viewController = viewController;
        [IGListDebugger trackAdapter:self];
    }
    return self;
}
```

`objectLookupPointerFunctions` 的自定义 `hashFunction` 和 `isEqualFunction` ，由于 `object` 已经有了 `-diffIdentifier` ，所以可以基于这个方法进行判断：

```swift
static BOOL IGListIsEqual(const void *a, const void *b, NSUInteger (*size)(const void *item)) {
    const id<IGListDiffable, NSObject> left = (__bridge id<IGListDiffable, NSObject>)a;
    const id<IGListDiffable, NSObject> right = (__bridge id<IGListDiffable, NSObject>)b;
    return [left class] == [right class]
    && [[left diffIdentifier] isEqual:[right diffIdentifier]];
}

// 因为 diff 算法是基于 `-diffIdentifier` 进行计算，所以我们的映射表需要精确匹配这种行为
static NSUInteger IGListIdentifierHash(const void *item, NSUInteger (*size)(const void *item)) {
    return [[(__bridge id<IGListDiffable>)item diffIdentifier] hash];
}

- (NSPointerFunctions *)objectLookupPointerFunctions {
    NSPointerFunctions *functions = [NSPointerFunctions pointerFunctionsWithOptions:NSPointerFunctionsStrongMemory];
    functions.hashFunction = IGListIdentifierHash;
    functions.isEqualFunction = IGListIsEqual;
    return functions;
}
```

`IGListAdapter` 作为 `IGListKit` 的中心调度器，负责串联起 `IGListSectionController` ， `Model` ， `UICollectionView` 和 `UICollectionReusableView` 之间的关系，在设置 `UICollectionView` 时， `IGListAdapter` 就会进行对应的处理：

```swift
- (void)setCollectionView:(UICollectionView *)collectionView {
    IGAssertMainThread();

       // 1. 如果在Cell中设置UICollectionView时,有可能会多次设置IGListAdapter的 colleciontView，这里做一下判断，防止重复设置；
    if (_collectionView != collectionView || _collectionView.dataSource != self) {
        // 2. 每次关联UICollectionView和IGListAdapter，都需要清空之前的关联，
		    // 防止旧的IGListAdapter对UICollectionView进行更新，
        // 相关的 PR 在这里 [Prevent stale adapter:collectionView corruptions](https://github.com/Instagram/IGListKit/pull/517) 
        // 当在Cell中进行设置时：adapter.collectionView = cell.collectionView，
        // 有可能会有多个adapter链接到同一个collectionView，
        // 那么就可能会发生旧的adapter对当前UICollectionView进行修改的 bug ，
        // 所以这里需要对之前的adapter设置collectionView为nil
        static NSMapTable<UICollectionView *, IGListAdapter *> *globalCollectionViewAdapterMap = nil;
        if (globalCollectionViewAdapterMap == nil) {
            globalCollectionViewAdapterMap = [NSMapTable weakToWeakObjectsMapTable];
        }
        [globalCollectionViewAdapterMap removeObjectForKey:_collectionView];
        [[globalCollectionViewAdapterMap objectForKey:collectionView] setCollectionView:nil];
        [globalCollectionViewAdapterMap setObject:self forKey:collectionView];
        // 3. 清空已注册的Cell，Nib等；
        _registeredCellIdentifiers = [NSMutableSet new];
        _registeredNibNames = [NSMutableSet new];
        _registeredSupplementaryViewIdentifiers = [NSMutableSet new];
        _registeredSupplementaryViewNibNames = [NSMutableSet new];
        const BOOL settingFirstCollectionView = _collectionView == nil;
        _collectionView = collectionView;
        _collectionView.dataSource = self;
        if (@available(iOS 10.0, tvOS 10, *)) {
            _collectionView.prefetchingEnabled = NO;
        }
        [_collectionView.collectionViewLayout ig_hijackLayoutInteractiveReorderingMethodForAdapter:self];
        [_collectionView.collectionViewLayout invalidateLayout];
        // 4. 设置collectionView的delegate
        [self _updateCollectionViewDelegate];
        // 5. 如果是第一次设置collectionView则需要进行一些设置。
        if (settingFirstCollectionView) {
            [self _updateAfterPublicSettingsChange];
        }
    }
}
```

`_updateAfterPublicSettingsChange` 首先调用 `NSArray *objectsWithDuplicateIdentifiersRemoved(NSArray<id<IGListDiffable>> *objects)` 去除重复的 `Objects` ：

```swift
- (void)_updateAfterPublicSettingsChange {
    id<IGListAdapterDataSource> dataSource = _dataSource;
    if (_collectionView != nil && dataSource != nil) {
        NSArray *uniqueObjects = objectsWithDuplicateIdentifiersRemoved([dataSource objectsForListAdapter:self]);
        [self _updateObjects:uniqueObjects dataSource:dataSource];
    }
}
```

`IGListKit` 不支持 `Object` 间有相同的 `diffIdentifier` ，所以需要进行过滤，使用 `NSMapTable` 来进行记录，以 `diffIdentifier` 为 `key` 进行记录，如果 `identifierMap` 有记录，则不添加到 `uniqueObjects` 中：

```swift
static NSArray *objectsWithDuplicateIdentifiersRemoved(NSArray<id<IGListDiffable>> *objects) {
    if (objects == nil) {
        return nil;
    }

    NSMapTable *identifierMap = [NSMapTable strongToStrongObjectsMapTable];
    NSMutableArray *uniqueObjects = [NSMutableArray new];
    for (id<IGListDiffable> object in objects) {
        id diffIdentifier = [object diffIdentifier];
        id previousObject = [identifierMap objectForKey:diffIdentifier];
        if (diffIdentifier != nil
            && previousObject == nil) {
            [identifierMap setObject:object forKey:diffIdentifier];
            [uniqueObjects addObject:object];
        } else {
            IGLKLog(@"Duplicate identifier %@ for object %@ with object %@", diffIdentifier, object, previousObject);
        }
    }
    return uniqueObjects;
}
```

去除重复的 `Model` 后，再调用 `_updateObjects: dataSource:` 获取对应的 `IGListSectionController` ，并将 `IGListSectionController` 和 `Object` 串联起来：

```swift
- (void)_updateObjects:(NSArray *)objects dataSource:(id<IGListAdapterDataSource>)dataSource {
    // 1. 状态标记，防止在更新数据源过程中刷新collectionView
    _isInObjectUpdateTransaction = YES;
	  // 2. 更新数据源过程中所需要用到的数据组合
    NSMutableArray<IGListSectionController *> *sectionControllers = [NSMutableArray new];
    NSMutableArray *validObjects = [NSMutableArray new];
    IGListSectionMap *map = self.sectionMap;
    NSMutableSet *updatedObjects = [NSMutableSet new];
    // 3. 把当前的viewController和adapter存储到local thread dictionary中，以便在初始化 IGListSectionController时使用
    IGListSectionControllerPushThread(self.viewController, self);
    for (id object in objects) {
    // 4. 从map中获取对应的IGListSectionController，如果没有，则从dataSource生成的新的
        IGListSectionController *sectionController = [map sectionControllerForObject:object];
        if (sectionController == nil) {
            sectionController = [dataSource listAdapter:self sectionControllerForObject:object];
        }
        if (sectionController == nil) {
            IGLKLog(@"WARNING: Ignoring nil section controller returned by data source %@ for object %@.",
                    dataSource, object);
            continue;
        }

      // 5. 设置sectionController的collectionContext和viewController
		  // 防止sectioncontroller不是在-listAdapter:sectionControllerForObject:方法中创建的，
      // 导致collectionContext和viewController没有更新
        sectionController.collectionContext = self;
        sectionController.viewController = self.viewController;
      // 6. 如果找不到oldSection，则表示object是新增加的。如果新旧object不相等，则说明object有更新
        const NSInteger oldSection = [map sectionForObject:object];
        if (oldSection == NSNotFound || [map objectForSection:oldSection] != object) {
            [updatedObjects addObject:object];
        }
        [sectionControllers addObject:sectionController];
        [validObjects addObject:object];
    }
    // 7. 清除local thread dictionary的数据
    IGListSectionControllerPopThread();
	  // 8. 更新validObjects和sectionControllers的绑定关系
    [map updateWithObjects:validObjects sectionControllers:sectionControllers];
    // 9. 所有sectionControllers都已经加载完成，进行object更新工作
    for (id object in updatedObjects) {
        [[map sectionControllerForObject:object] didUpdateToObject:object];
    }
    [self _updateBackgroundViewShouldHide:![self _itemCountIsZero]];
    _isInObjectUpdateTransaction = NO;
}
```

### 更新数据

`IGListKit` 在数据更新时刷新界面的流程和普通的 `UICollectionView` 使用方式类似，首先是根据用户操作/网络请求等对数据进行调整，然后调用 `reloadData/performUpdates` 刷新 `UICollectionView` ，但是与系统的 `UICollectionView` 不同，我们不再需要手动去计算哪些 `Cell` 进行了刷新/删除/插入/移动和进行相关操作， `IGListKit` 会自动帮我们完成这件事，我们所需要做的只是更新数据，然后调用 `IGListAdapter` 的对应方法即可。 而 `IGListAdapter` 提供了三种刷新方式，下面具体展开说说。

#### performUpdatesAnimated:completion

`performUpdatesAnimated:completion:` ，等价于 `UICollectionView` 的 `performBatchUpdates:completion:` 方法，当数据源更新后，可以调用这个方法来进行局部刷新， `IGListAdapter` 内部会计算出新增/删除/更新的 `Object` 所对应的 `Section` 和位置，调用 `UICollectionView` 对应的方法。

```swift
- (void)performUpdatesAnimated:(BOOL)animated completion:(IGListUpdaterCompletion)completion {
    
    id<IGListAdapterDataSource> dataSource = self.dataSource;
    UICollectionView *collectionView = self.collectionView;
    // 1. 如果dataSource或者collectionView为nil时，直接返回，调用completion(NO)
    if (dataSource == nil || collectionView == nil) {
        if (completion) {
            completion(NO);
        }
        return;
    }
    // 2. 获取旧的objects，定义如何获取新的objects的block，
    // 延迟执行dataSource的objectsForListAdapter:方法，等到需要时再执行，保证获取到的  objects是最新的
    NSArray *fromObjects = self.sectionMap.objects;
    __weak __typeof__(self) weakSelf = self;
    IGListToObjectBlock toObjectsBlock = ^NSArray *{
        __typeof__(self) strongSelf = weakSelf;
        if (strongSelf == nil) {
            return nil;
        }
        return [dataSource objectsForListAdapter:strongSelf];
    };
    // 3. 这里在局部刷新布局信息时会用到，标记一下进入局部刷新流程
    [self _enterBatchUpdates];
    // 4. 调用id <IGListUpdatingDelegate> updater对应的方法更新collectionView，
    // 通过_collectionViewBlock来获取collectionView延迟到真正更新时才执行 block，确保获取到的 collectionView`是正确的
    [self.updater performUpdateWithCollectionViewBlock:[self _collectionViewBlock]
                                           fromObjects:fromObjects
                                        toObjectsBlock:toObjectsBlock
                                              animated:animated
                                 objectTransitionBlock:^(NSArray *toObjects) {
                                     // 5. 这里的toObjects是由update进行计算后得出的新的数据源，设置previousSectionMap，更新数据源
                                     weakSelf.previousSectionMap = [weakSelf.sectionMap copy];
                                     [weakSelf _updateObjects:toObjects dataSource:dataSource];
                                 } completion:^(BOOL finished) {
                                     // 6. 完成刷新，复原标记
                                     weakSelf.previousSectionMap = nil;

                                     [weakSelf _notifyDidUpdate:IGListAdapterUpdateTypePerformUpdates animated:animated];
                                     if (completion) {
                                         completion(finished);
                                     }
                                     [weakSelf _exitBatchUpdates];
                                 }];
}
```

#### reloadDataWithCompletion

`reloadDataWithCompletion:` 全局刷新，作用跟 `UICollectionView` 的 `reloadData` 方法类似，会移除掉所有旧的 `objects` 和 `IGListSectionController` ，需要注意的是会重新生成所有 `IGListSectionController` ，所以是个有可能非常耗时的操作，在调用这个方法前必须清楚知道这一前提，一般情况下推荐使用 `performUpdatesAnimated:completion:` 来进行刷新。`reloadData` 的实现比 `performUpdates` 的要简单很多，只需要调用 `update` 对应的方法，清空 `sectionMap` ，更新 `objects` 即可。

```swift
- (void)reloadDataWithCompletion:(nullable IGListUpdaterCompletion)completion {
    id<IGListAdapterDataSource> dataSource = self.dataSource;
    UICollectionView *collectionView = self.collectionView;
    if (dataSource == nil || collectionView == nil) {
        if (completion) {
            completion(NO);
        }
        return;
    }

    NSArray *uniqueObjects = objectsWithDuplicateIdentifiersRemoved([dataSource objectsForListAdapter:self]);

    __weak __typeof__(self) weakSelf = self;
    [self.updater reloadDataWithCollectionViewBlock:[self _collectionViewBlock]
                                  reloadUpdateBlock:^{
                                      [weakSelf.sectionMap reset];
                                      [weakSelf _updateObjects:uniqueObjects dataSource:dataSource];
                                  } completion:^(BOOL finished) {
                                      [weakSelf _notifyDidUpdate:IGListAdapterUpdateTypeReloadData animated:NO];
                                      if (completion) {
                                          completion(finished);
                                      }
                                  }];
}
```

#### reloadObjects

`reloadObjects:` 刷新 `objects` 所对应的 `section` ，在 `object` 有更新时进行调用，可以直接更新所对应的 `sections` ：

```swift
- (void)reloadObjects:(NSArray *)objects {
    NSMutableIndexSet *sections = [NSMutableIndexSet new];
    // 1. 使用_sectionMapUsingPreviousIfInUpdateBlock获取sectionMap，
    // 因为reloadObjects是有可能在batch update过程中调用，如果是在batch update则使用旧的 sectionMap
    IGListSectionMap *map = [self _sectionMapUsingPreviousIfInUpdateBlock:YES];
    for (id object in objects) {
        // 2. 根据object找到section，如果找不到则直接跳过
        const NSInteger section = [map sectionForObject:object];
        const BOOL notFound = section == NSNotFound;
        if (notFound) {
            continue;
        }
        [sections addIndex:section];
        // 3.根据section找一下object，如果新旧object不相等，map则更新object，
        // 同时更新sectionController的object
        if (object != [map objectForSection:section]) {
            [map updateObject:object];
            [[map sectionControllerForSection:section] didUpdateToObject:object];
        }
    }
    UICollectionView *collectionView = self.collectionView;
    [self.updater reloadCollectionView:collectionView sections:sections];
}
```

### 协议protocol

`IGListKit` 围绕 `IGListAdapter` 定义了了大量协议，提供了良好的扩展性和接口封装。下面逐一来进行分析。

#### IGListAdapterDelegate

当 `object` 在屏幕上出现/消失，会调用 `IGListKit` 的 `id <IGListAdapterDelegate> delegate` 的相关方法，但是由于是基于 `object` 的，所以粒度没有办法精确到每个 `Cell` ，所以需要 `Cell` 级别的粒度，可以使用 `IGListSectionController` 的 `id <IGListDisplayDelegate> displayDelegate` 。

```swift
NS_SWIFT_NAME(ListAdapterDelegate)
@protocol IGListAdapterDelegate <NSObject>

- (void)listAdapter:(IGListAdapter *)listAdapter willDisplayObject:(id)object atIndex:(NSInteger)index;
- (void)listAdapter:(IGListAdapter *)listAdapter didEndDisplayingObject:(id)object atIndex:(NSInteger)index;

@end
```

#### UICollectionViewDelegate & UIScrollViewDelegate

这两个需要放在一起说说，因为 `IGListKit` 对 `IGListAdapter` 的这两个 `delegate` 做了处理：

```swift
// 只接收UICollectionViewDelegate的回调，不接收UIScrollViewDelegate的回调
@property (nonatomic, nullable, weak) id <UICollectionViewDelegate> collectionViewDelegate;
// 只接收UIScrollViewDelegate的回调
@property (nonatomic, nullable, weak) id <UIScrollViewDelegate> scrollViewDelegate;
```

因为 `UICollectionViewDelegate` 是继承自 `UIScrollViewDelegate` ，所以 `id <UICollectionViewDelegate> collectionViewDelegate` 也会接收到 `UIScrollViewDelegate` 的回调，所以 `IGListKit` 使用一个 `NSProxy` 子类 `IGListAdapterProxy` 来对不同 `delegate` 的回调进行区分：

```swift
- (void)_createProxyAndUpdateCollectionViewDelegate {
    _collectionView.delegate = nil;

    self.delegateProxy = [[IGListAdapterProxy alloc] initWithCollectionViewTarget:_collectionViewDelegate
                                                                 scrollViewTarget:_scrollViewDelegate
                                                                      interceptor:self];
    [self _updateCollectionViewDelegate];
}

- (void)_updateCollectionViewDelegate {
    _collectionView.delegate = (id<UICollectionViewDelegate>)self.delegateProxy ?: self;
}
```

```swift
- (instancetype)initWithCollectionViewTarget:(nullable id<UICollectionViewDelegate>)collectionViewTarget
                            scrollViewTarget:(nullable id<UIScrollViewDelegate>)scrollViewTarget
                                 interceptor:(IGListAdapter *)interceptor {
    IGParameterAssert(interceptor != nil);
    if (self) {
        _collectionViewTarget = collectionViewTarget;
        _scrollViewTarget = scrollViewTarget;
        _interceptor = interceptor;
    }
    return self;
}

- (BOOL)respondsToSelector:(SEL)aSelector {
    // 先判断是否经过IGListAdapter进行处理，然后再判断_collectionViewTarget或者 _scrollViewTarget是否可以响应
    return isInterceptedSelector(aSelector)
    || [_collectionViewTarget respondsToSelector:aSelector]
    || [_scrollViewTarget respondsToSelector:aSelector];
}

- (id)forwardingTargetForSelector:(SEL)aSelector {
    // 先判断_interceptor是否可以处理，如果可以，就转发给_interceptor
    if (isInterceptedSelector(aSelector)) {
        return _interceptor;
    }
    // 因为UICollectionViewDelegate是UIScrollViewDelegate的子类，所以先检查 _scrollViewTarget是否可以响应，
    // 否则使用_collectionViewTarget
    return [_scrollViewTarget respondsToSelector:aSelector] ? _scrollViewTarget : _collectionViewTarget;
}
```

#### IGListAdapterPerformanceDelegate

`id <IGListAdapterPerformanceDelegate> performanceDelegate` 是为调用方提供了 `UICollectionView` 在滑动时会调用的方法耗时的回调，比如说监听获取 `Cell` 的耗时。

```swift
// 性能相关的 delegate ，
@property (nonatomic, nullable, weak) id <IGListAdapterPerformanceDelegate> performanceDelegate;
```

```swift
- (UICollectionViewCell *)collectionView:(UICollectionView *)collectionView cellForItemAtIndexPath:(NSIndexPath *)indexPath {
    id<IGListAdapterPerformanceDelegate> performanceDelegate = self.performanceDelegate;
    [performanceDelegate listAdapterWillCallDequeueCell:self];
    IGListSectionController *sectionController = [self sectionControllerForSection:indexPath.section];
    UICollectionViewCell *cell = [sectionController cellForItemAtIndex:indexPath.item];
    /// ....
    [performanceDelegate listAdapter:self didCallDequeueCell:cell onSectionController:sectionController atIndex:indexPath.item];
    return cell;
}
```

可以看到在 `cellForItemAtIndexPath` 的开头和结尾调用了 `performanceDelegate` 的方法。

#### IGListBatchContext

支持 `IGListBatchContext` 协议的对象为 `IGListSectionController` 提供了 `reload/insert/delete/move` 等方法，在 `IGListKit` 中，这个对象是 `IGListAdapter` ，只是 `IGListSectionController` 不知道这个对象的具体类型，只知道是 `id <IGListBatchContext>` 。

`-reloadInSectionController:atIndexes:` 方法负责重新加载 `IGListSectionController` 中 `indexes` 所对应的 `Cell` 。 `UICollectionView` 并不支持在 `batch updates` 中 `-reloadSections` 或者 `-reloadItemsAtIndexPaths:` ，内部实现为通过 `delete` 和 `insert` 操作实现，这块的实现在某些操作下可能会导致异常。 假设有个 `object` ， 对应的 `section` 为 2 ，`items` 数量为 4 ，如果需要对 `index` 为 1 的 `item` 进行 `reload` ，先要创建一个 `NSIndexPath` ，`item` 为 1， `section` 为 2 ，当执行 `-performBatchUpdates:` 时， `UICollectionView` 会删除和插入这个 `NSIndexPath` 。如果这时我们在 `position` 为 2 中插入了一个 `section` ，原有的 `seciton 2` 就会变成 `section 3` 。然而，插入的 `indexPath` 的 `section` 还是 2 。那么 `UICollectionView` 就会在 `section: 2 item: 1` 执行一个插入动画，这时候就会抛出一个异常。 为了避免这个问题， `IGListAdapter` 会根据 `sectionController` 的新旧来获取不同的 `NSIndexPath` ：

```swift
- (void)reloadInSectionController:(IGListSectionController *)sectionController atIndexes:(NSIndexSet *)indexes {
    UICollectionView *collectionView = self.collectionView;

    if (indexes.count == 0) {
        return;
    }
    [indexes enumerateIndexesUsingBlock:^(NSUInteger index, BOOL *stop) {
        NSIndexPath *fromIndexPath = [self indexPathForSectionController:sectionController index:index usePreviousIfInUpdateBlock:YES];
        NSIndexPath *toIndexPath = [self indexPathForSectionController:sectionController index:index usePreviousIfInUpdateBlock:NO];
        if (fromIndexPath != nil && toIndexPath != nil) {
            [self.updater reloadItemInCollectionView:collectionView fromIndexPath:fromIndexPath toIndexPath:toIndexPath];
        }
    }];
}
```

可以看到 `fromIndexPath` 是根据旧的数据源获取， `toIndexPath` 是新的数据源，同时还需要进行是否为 `nil` 的检查，因为 `sectionController` 有可能在批量更新中被删除了。

`-invalidateLayoutInSectionController:atIndexes:` 让 `IGListSectionController` 指定 `Cell` 布局信息失效：

```swift
- (void)invalidateLayoutInSectionController:(IGListSectionController *)sectionController atIndexes:(NSIndexSet *)indexes {
  
    UICollectionView *collectionView = self.collectionView;
    if (indexes.count == 0) {
        return;
    }
    NSArray *indexPaths = [self indexPathsFromSectionController:sectionController indexes:indexes usePreviousIfInUpdateBlock:NO];
    UICollectionViewLayout *layout = collectionView.collectionViewLayout;
    UICollectionViewLayoutInvalidationContext *context = [[[layout.class invalidationContextClass] alloc] init];
    [context invalidateItemsAtIndexPaths:indexPaths];
    [layout invalidateLayoutWithContext:context];
}
```

获取到 `indexPaths` 为新的 `objects` 所对应的 `indexPaths` ，这里是通过 `[layout.class invalidationContextClass]` ，因为 `layout` 有可能使用的是自定义的 `UICollectionViewLayoutInvalidationContext` 子类。

#### IGListCollectionContext

`IGListCollectionContext` 为 `IGListSectionController` 提供了 `UICollectionView` 的相关信息，如大小，复用，插入，删除，重新加载等。通过协议的方式可以把接口统一起来， `IGListSectionController`只能使用 `IGListCollectionContext` 提供的接口，它不知道也不需要知道 `IGListAdapter` 的存在。

```swift
- (__kindof UICollectionViewCell *)dequeueReusableCellOfClass:(Class)cellClass
                                          withReuseIdentifier:(NSString *)reuseIdentifier
                                         forSectionController:(IGListSectionController *)sectionController
                                                      atIndex:(NSInteger)index {
    UICollectionView *collectionView = self.collectionView;
    NSString *identifier = IGListReusableViewIdentifier(cellClass, nil, reuseIdentifier);
    NSIndexPath *indexPath = [self indexPathForSectionController:sectionController index:index usePreviousIfInUpdateBlock:NO];
    if (![self.registeredCellIdentifiers containsObject:identifier]) {
        [self.registeredCellIdentifiers addObject:identifier];
        [collectionView registerClass:cellClass forCellWithReuseIdentifier:identifier];
    }
    return [collectionView dequeueReusableCellWithReuseIdentifier:identifier forIndexPath:indexPath];
}
```

在复用 `Cell` 时， `IGListKit` 不需要先调用 `register` 方法来注册 `Cell` ， `IGListAdapter` 内部记录了已经注册过的 `Cell` ，如果没有注册过，就先调用 `UICollectionView` 的 `register` 方法来进行注册，然后再调用 `dequeueReusableCell` 方法来从复用池中获取 `Cell` 。这是一个非常舒服的特性，如果每次使用都需要注册 `Cell` ，那么当业务变得非常复杂时，可能需要注册大量的 `Cell` ，会出现一个屏幕都无法完全显示注册 `Cell` 的方法。同时 `UICollectionView` 或者 `UIViewController` 也不需要和 `Cell` 进行交互，当 `Cell` 需要调整时，我们只需要在 `IGListSectionController` 中进行处理，或者直接替换对应的 `IGListSectionController` 。 上面说到在获取 `Cell` 时， `IGListKit` 会自动帮我们判断是否需要注册对应的 `Cell` ，下面来看下具体是如何实现的：

```swift
NS_INLINE NSString *IGListReusableViewIdentifier(Class viewClass, NSString * _Nullable kind, NSString * _Nullable givenReuseIdentifier) {
    return [NSString stringWithFormat:@"%@%@%@", kind ?: @"", givenReuseIdentifier ?: @"", NSStringFromClass(viewClass)];
}

- (__kindof UICollectionViewCell *)dequeueReusableCellOfClass:(Class)cellClass
                                          withReuseIdentifier:(NSString *)reuseIdentifier
                                         forSectionController:(IGListSectionController *)sectionController
                                                      atIndex:(NSInteger)index {
    UICollectionView *collectionView = self.collectionView;
    NSString *identifier = IGListReusableViewIdentifier(cellClass, nil, reuseIdentifier);
    NSIndexPath *indexPath = [self indexPathForSectionController:sectionController index:index usePreviousIfInUpdateBlock:NO];
    if (![self.registeredCellIdentifiers containsObject:identifier]) {
        [self.registeredCellIdentifiers addObject:identifier];
        [collectionView registerClass:cellClass forCellWithReuseIdentifier:identifier];
    }
    return [collectionView dequeueReusableCellWithReuseIdentifier:identifier forIndexPath:indexPath];
}
```

这个方法是在 `IGListAdapter` 内， `IGListAdapter` 会使用 `NSMutableSet` 来记录所有注册过的 `Cell` 对应的 `reuseIdentifier` ，在调用 `dequeueReusableCellWithReuseIdentifier:forIndexPath:` 前会先判断 `registeredCellIdentifiers` 是否有包含这个 `identifier` ，如果没有则进行注册，而 `IGListReusableViewIdentifier` 会采用 `kind` ， `givenReuseIdentifier` 和 `viewClass` 进行拼接的方式，生成新的 `identifier` ，也就不可能存在相同的 `identifier` 。

如果需要在 `IGListSectionController` 内部对数据源进行修改和刷新视图， `IGListCollectionContext` 也提供了如下方法：

```swift
- (void)performBatchAnimated:(BOOL)animated updates:(void (^)(id<IGListBatchContext>))updates completion:(void (^)(BOOL))completion {
    [self _enterBatchUpdates];
    __weak __typeof__(self) weakSelf = self;
    [self.updater performUpdateWithCollectionViewBlock:[self _collectionViewBlock] animated:animated itemUpdates:^{
        // 更新isInUpdateBlock，执行block
        weakSelf.isInUpdateBlock = YES;
        updates(weakSelf);
        weakSelf.isInUpdateBlock = NO;
    } completion: ^(BOOL finished) {
        // 判断是否需要显示 emptyView 
        [weakSelf _updateBackgroundViewShouldHide:![weakSelf _itemCountIsZero]];
        [weakSelf _notifyDidUpdate:IGListAdapterUpdateTypeItemUpdates animated:animated];
        if (completion) {
            completion(finished);
        }
        [weakSelf _exitBatchUpdates];
    }];
}
```

`-performBatchAnimated:updates:completion:` 支持在执行多个 `Cell` 的相关操作。在 `updates block` 中更新 `sectionController` 的 `dataSource` ，然后调用 `IGListBatchContext` 的方法来插入/删除 `items` ：

```swift
// self 为 IGListSectionController
[self.collectionContext performBatchItemUpdates:^ (id<IGListBatchContext> batchContext>){
   [self.items addObject:newItem];
   [self.items removeObjectAtIndex:0];
   NSIndexSet *inserts = [NSIndexSet indexSetWithIndex:[self.items count] - 1];
   [batchContext insertInSectionController:self atIndexes:inserts];
   NSIndexSet *deletes = [NSIndexSet indexSetWithIndex:0];
   [batchContext deleteInSectionController:self atIndexes:deletes];
 } completion:nil];
```

#### IGListAdapterUpdateListener

`IGListKit` 支持上面提到的几种刷新方式， `IGListAdapterUpdateListener` 则提供了相关回调：

```swift
typedef NS_ENUM(NSInteger, IGListAdapterUpdateType) {
    // 调用-[IGListAdapter performUpdatesAnimated:completion:]
    IGListAdapterUpdateTypePerformUpdates,
    // 调用-[IGListAdapter reloadDataWithCompletion:]
    IGListAdapterUpdateTypeReloadData,
    // 在IGListSectionController中调用-[IGListCollectionContext performBatchAnimated:updates:completion:]
    IGListAdapterUpdateTypeItemUpdates,
};

@protocol IGListAdapterUpdateListener <NSObject>
- (void)listAdapter:(IGListAdapter *)listAdapter
    didFinishUpdate:(IGListAdapterUpdateType)update
           animated:(BOOL)animated;
@end
```

这个方法会在以下几种情况下调用

+ 执行 `-[IGListAdapter performUpdatesAnimated:completion:]` 的 `completion block` 前调用；
+ 执行 `-[IGListAdapter reloadDataWithCompletion:]` 后调用；
+ `IGListSectionController` 执行 `-[IGListCollectionContext performBatchAnimated:updates:completion:]` 方法后。

`IGListAdapter` 支持设置多个 `Listener` ，对外提供了两个方法来添加和移除 `Listener` :

```swift
- (void)addUpdateListener:(id<IGListAdapterUpdateListener>)updateListener;
- (void)removeUpdateListener:(id<IGListAdapterUpdateListener>)updateListener;
```

## Updater&Diff

### IGListAdapterUpdater

在初始化 `IGListAdapter` 时提供了一个 `id<IGListUpdatingDelegate> updater` 参数，调用者可以自己自定义一个支持 `IGListUpdatingDelegate` 协议的类，来实现 `IGListUpdatingDelegate` 的方法。 `IGListAdapter` 在更新 `UICollectionView` 和数据源时都是通过 `updater` 来进行操作， `IGListKit` 为我们提供了一个默认的 `updater` ： `IGListAdapterUpdater` ， `IGListAdapter` 支持 `UICollectionView` 的局部更新操作。

```swift
// 当更新逻辑执行完成时调用的block，finished表示更新是否完成。
typedef void (^IGListUpdatingCompletion)(BOOL finished);
// 当adapter对UICollectionView进行改动时调用，toObjects` 表示新的 `objects`
typedef void (^IGListObjectTransitionBlock)(NSArray *toObjects);
// 包含所有更新的block
typedef void (^IGListItemUpdateBlock)(void);
// adapter对UICollectionView进行reload是调用
typedef void (^IGListReloadUpdateBlock)(void);
// 返回转换后的objects
typedef NSArray * _Nullable (^IGListToObjectBlock)(void);
// 获取执行更新的UICollectionView
typedef UICollectionView * _Nullable (^IGListCollectionViewBlock)(void);
// IGListUpdatingDelegate用于处理section和row级别的更新，接口的实现需要对集合处理或者转发。
@protocol IGListUpdatingDelegate <NSObject>
// 用于在集合中寻找object
- (NSPointerFunctions *)objectLookupPointerFunctions;
/*
用于判断如何在objects进行转换。可以在objects直接执行diff， reload每个section ，或者直接调用 UICollectionView的-reloadData方法。
最后，UICollectionView必须要配置好toObjects数组中对应的每个section。
*/
- (void)performUpdateWithCollectionViewBlock:(IGListCollectionViewBlock)collectionViewBlock
                                 fromObjects:(nullable NSArray<id <IGListDiffable>> *)fromObjects
                              toObjectsBlock:(nullable IGListToObjectBlock)toObjectsBlock
                                    animated:(BOOL)animated
                       objectTransitionBlock:(IGListObjectTransitionBlock)objectTransitionBlock
                                  completion:(nullable IGListUpdatingCompletion)completion;

// 插入对应的 indexPaths
- (void)insertItemsIntoCollectionView:(UICollectionView *)collectionView indexPaths:(NSArray <NSIndexPath *> *)indexPaths;
// 删除对应的 indexPaths
- (void)deleteItemsFromCollectionView:(UICollectionView *)collectionView indexPaths:(NSArray <NSIndexPath *> *)indexPaths;
// 移动对应的 indexPath
- (void)moveItemInCollectionView:(UICollectionView *)collectionView
                   fromIndexPath:(NSIndexPath *)fromIndexPath
                     toIndexPath:(NSIndexPath *)toIndexPath;
// reload 对应的 fromIndexPath 和 toIndexPath
- (void)reloadItemInCollectionView:(UICollectionView *)collectionView
                     fromIndexPath:(NSIndexPath *)fromIndexPath
                       toIndexPath:(NSIndexPath *)toIndexPath;
// section 级别的处理，移动 index 对应的 section
- (void)moveSectionInCollectionView:(UICollectionView *)collectionView
                          fromIndex:(NSInteger)fromIndex
                            toIndex:(NSInteger)toIndex;
// 执行 reload data
- (void)reloadDataWithCollectionViewBlock:(IGListCollectionViewBlock)collectionViewBlock
                        reloadUpdateBlock:(IGListReloadUpdateBlock)reloadUpdateBlock
                               completion:(nullable IGListUpdatingCompletion)completion;

// reload 对应的 sections
- (void)reloadCollectionView:(UICollectionView *)collectionView sections:(NSIndexSet *)sections;

// 执行IGListItemUpdateBlock
- (void)performUpdateWithCollectionViewBlock:(IGListCollectionViewBlock)collectionViewBlock
                                    animated:(BOOL)animated
                                 itemUpdates:(IGListItemUpdateBlock)itemUpdates
                                  completion:(nullable IGListUpdatingCompletion)completion;

@end
```

`IGListAdapterUpdater` 内部提供了一套队列刷新机制，使用 `IGListBatchUpdates` 记录批量刷新的 `block` ：

```swift
@property (nonatomic, strong, readonly) NSMutableArray<void (^)(void)> *itemUpdateBlocks;
@property (nonatomic, strong, readonly) NSMutableArray<void (^)(BOOL)> *itemCompletionBlocks;
```

#### 批量与全局

`IGListAdapterUpdater` 提供的方法可以分为两种：

1.批量刷新，通过 `diff` 算法计算出需要进行操作的 `Cell` 或者 `Section` 。 `IGListCollectionViewBlock` 用于提供 `UICollectionView` ，通过 `block` 的方式来获取，可以保证在调用 `block` 时获取到 `UICollectionView` 是最新设置的。 `IGListToObjectBlock` 的作用也是一样的，保证获取的到 `toObjects` 是最新的：

```swift
- (void)performUpdateWithCollectionViewBlock:(IGListCollectionViewBlock)collectionViewBlock
                            		fromObjects:(NSArray *)fromObjects
                         		 toObjectsBlock:(IGListToObjectBlock)toObjectsBlock
                               	   animated:(BOOL)animated
                       objectTransitionBlock:(IGListObjectTransitionBlock)objectTransitionBlock
                                  completion:(IGListUpdatingCompletion)completion {
    self.fromObjects = self.fromObjects ?: self.pendingTransitionToObjects ?: fromObjects;
    self.toObjectsBlock = toObjectsBlock;
    self.queuedUpdateIsAnimated = self.queuedUpdateIsAnimated && animated;
    self.objectTransitionBlock = objectTransitionBlock;
    IGListUpdatingCompletion localCompletion = completion;
    if (localCompletion) {
        [self.completionBlocks addObject:localCompletion];
    }
    [self _queueUpdateWithCollectionViewBlock:collectionViewBlock];
}
```

2.全局刷新，作用类似于 `UICollectionView` 的 `reloadData` 方法：

```swift
- (void)reloadDataWithCollectionViewBlock:(IGListCollectionViewBlock)collectionViewBlock
                   reloadUpdateBlock:(IGListReloadUpdateBlock)reloadUpdateBlock
                          completion:(nullable IGListUpdatingCompletion)completion {
    IGListUpdatingCompletion localCompletion = completion;
    if (localCompletion) {
        [self.completionBlocks addObject:localCompletion];
    }
    self.reloadUpdates = reloadUpdateBlock;
    self.queuedReloadData = YES;
    [self _queueUpdateWithCollectionViewBlock:collectionViewBlock];
}
```

可以看到批量刷新和全局刷新的实现到最后都会调用 `_queueUpdateWithCollectionViewBlock` ，而在 `_queueUpdateWithCollectionViewBlock` 方法中会根据是否为 `reloadData` 来调用不同的方法：

```swift
- (void)_queueUpdateWithCollectionViewBlock:(IGListCollectionViewBlock)collectionViewBlock {
    __weak __typeof__(self) weakSelf = self;
    		// 这里在main_queue上使用dispatch_async的原因是如果在短时间内多次调用批量刷新的方法，
    		// 可以去掉多余的diff计算和页面刷新，只需要执行一次。
    dispatch_async(dispatch_get_main_queue(), ^{
        // 如果updater不是在IGListBatchUpdateStateIdle状态或者没有改变，则直接返回。 
        if (weakSelf.state != IGListBatchUpdateStateIdle
            || ![weakSelf hasChanges]) {
            return;
        }
        // 判断是否 hasQueuedReloadData 来调用不同的刷新方法
        if (weakSelf.hasQueuedReloadData) {
            [weakSelf performReloadDataWithCollectionViewBlock:collectionViewBlock];
        } else {
            [weakSelf performBatchUpdatesWithCollectionViewBlock:collectionViewBlock];
        }
    });
}
```

在进行批量更新操作时，如果 `state` 是 `IGListBatchUpdateStateExecutingBatchUpdateBlock` ，执行批量更新 `block` 的状态，则直接执行 `block` 即可，不需要添加到 `itemUpdateBlocks` 中：

```swift
if (self.state == IGListBatchUpdateStateExecutingBatchUpdateBlock) {
    itemUpdates();
} else {
    [batchUpdates.itemUpdateBlocks addObject:itemUpdates];
    self.queuedUpdateIsAnimated = self.queuedUpdateIsAnimated && animated;
    [self _queueUpdateWithCollectionViewBlock:collectionViewBlock];
}
```

#### 状态配置

在看 `perform` 方法实现前先看下状态的相关定义，以便更好理解整体流程。

```swift
typedef NS_ENUM (NSInteger, IGListBatchUpdateState) {
    IGListBatchUpdateStateIdle,
    IGListBatchUpdateStateQueuedBatchUpdate,
    IGListBatchUpdateStateExecutingBatchUpdateBlock,
    IGListBatchUpdateStateExecutedBatchUpdateBlock,
};
```

1. `IGListBatchUpdateStateIdle` ，空闲状态，即当前无 `perform` 任务；
2. `IGListBatchUpdateStateQueuedBatchUpdate` ，已加入到批量更新的状态中，防止内部在同一时间内多次调用 `performBatchUpdatesWithCollectionViewBlock:` 方法；
3. `IGListBatchUpdateStateExecutingBatchUpdateBlock` 正在执行批量更新操作；
4. `IGListBatchUpdateStateExecutedBatchUpdateBlock` 已经完成批量更新操作。

在整个更新流程中， `updater.state` 会在这四种状态间切换，在不同状态间执行重复刷新操作时， `updater` 会因应不同的状态调用不同的方法，这块的处理是为了保证 `UI` 跟数据源之间的一致性和减少多余的刷新操作。

在每次开始进行刷新操作前，都会记录复制一份 `updater` 的相关属性到本地变量中，同时会调用 `cleanStateBeforeUpdates` 方法清空属性，这样同时调用刷新方法也不会互相覆盖掉，彼此间的状态也不会互相影响：

```swift
- (void)cleanStateBeforeUpdates {
    self.queuedUpdateIsAnimated = YES;
    self.fromObjects = nil;
    self.toObjectsBlock = nil;
    self.reloadUpdates = nil;
    self.queuedReloadData = NO;
    self.objectTransitionBlock = nil;
    [self.completionBlocks removeAllObjects];
}
```

`updater` 提供了 `hasChanges` 来判断是否有改动，避免多余的操作和一直执行 `perform` 操作：

```swift
- (BOOL)hasChanges {
    return self.hasQueuedReloadData
    || [self.batchUpdates hasChanges]
    || self.fromObjects != nil
    || self.toObjectsBlock != nil;
}
```

#### 全局刷新

`performReloadDataWithCollectionViewBlock` 为 `reloadData` 时调用，不需要进行 `diff` 的计算，处理起来也简单一点。方法首先初始化相关本地变量，然后调用 `cleanStateBeforeUpdates` 方法清空属性，防止和其它刷新任务互相影响：

```swift
- (void)performReloadDataWithCollectionViewBlock:(IGListCollectionViewBlock)collectionViewBlock {
    id<IGListAdapterUpdaterDelegate> delegate = self.delegate;
    void (^reloadUpdates)(void) = self.reloadUpdates;
    IGListBatchUpdates *batchUpdates = self.batchUpdates;
    NSMutableArray *completionBlocks = [self.completionBlocks mutableCopy];
    [self cleanStateBeforeUpdates];
	  ...
}
```

设置 `executeCompletionBlocks` ，遍历 `completionBlocks` ，执行完毕后恢复 `state` 为 `IGListBatchUpdateStateIdle` ：

```swift
void (^executeCompletionBlocks)(BOOL) = ^(BOOL finished) {
    for (IGListUpdatingCompletion block in completionBlocks) {
        block(finished);
    }
    self.state = IGListBatchUpdateStateIdle;
};
```

判断 `collectionView` 是否为 `nil` ，如果为 `nil` 则直接返回：

```swift
UICollectionView *collectionView = collectionViewBlock();
if (collectionView == nil) {
    [self _cleanStateAfterUpdates];
    executeCompletionBlocks(NO);
    [_delegate listAdapterUpdater:self didFinishWithoutUpdatesWithCollectionView:collectionView];
    return;
}
```

设置 `state` 为 `IGListBatchUpdateStateExecutingBatchUpdateBlock` ，进入执行 `updateBlock` 的流程，如果有设置 `reloadUpdates` ，则执行 `reloadUpdates` 。即使是在 `reloadData` 流程中，也需要调用所有 `itemUpdateBlocks` ，因为调用方有可能在 `itemUpdateBlock` 中对数据进行修改，必须要保证数据源和 `UI` 一致。把 `batchUpdates.itemCompletionBlocks` 添加到 `completionBlocks` 中，保证所有的 `completionBlocks` 都会被执行。最后调用定义好的 `executeCompletionBlocks` ：

```swift
self.state = IGListBatchUpdateStateExecutingBatchUpdateBlock;
if (reloadUpdates) {
    reloadUpdates();
}
for (IGListItemUpdateBlock itemUpdateBlock in batchUpdates.itemUpdateBlocks) {
    itemUpdateBlock();
}
[completionBlocks addObjectsFromArray:batchUpdates.itemCompletionBlocks];
self.state = IGListBatchUpdateStateExecutedBatchUpdateBlock;
[self _cleanStateAfterUpdates];
[delegate listAdapterUpdater:self willReloadDataWithCollectionView:collectionView isFallbackReload:NO];
[collectionView reloadData];
[collectionView.collectionViewLayout invalidateLayout];
[collectionView layoutIfNeeded];
[delegate listAdapterUpdater:self didReloadDataWithCollectionView:collectionView isFallbackReload:NO];
executeCompletionBlocks(YES);
```

#### 批量刷新

`performBatchUpdatesWithCollectionViewBlock` 进行批量更新时，需要处理各个状态的边界逻辑，所以比 `performReloadDataWithCollectionViewBlock` 更加复杂，在代码中是个 204 行的函数，下面拆开来说下具体的实现：

1.首先创建本地变量来记录所有的相关的属性，防止在执行批量更新过程中，又再次调用了 `performBatchUpdatesWithCollectionViewBlock` 接口，导致原有的属性被覆盖， `cleanStateBeforeUpdates` 会将相关属性复原，确保对更新过程中的其它 `performBatchUpdatesWithCollectionViewBlock` 调用没影响：

```swift
- (void)performBatchUpdatesWithCollectionViewBlock:(IGListCollectionViewBlock)collectionViewBlock {
    id<IGListAdapterUpdaterDelegate> delegate = self.delegate;
    NSArray *fromObjects = [self.fromObjects copy];
    IGListToObjectBlock toObjectsBlock = [self.toObjectsBlock copy];
    NSMutableArray *completionBlocks = [self.completionBlocks mutableCopy];
    void (^objectTransitionBlock)(NSArray *) = [self.objectTransitionBlock copy];
    const BOOL animated = self.queuedUpdateIsAnimated;
    const BOOL allowsReloadingOnTooManyUpdates = self.allowsReloadingOnTooManyUpdates;
    const IGListExperiment experiments = self.experiments;
    IGListBatchUpdates *batchUpdates = self.batchUpdates;
    [self cleanStateBeforeUpdates];
}
```

2.如果 `collectionView` 为 `nil` 就直接返回，这块的处理和 `performReloadDataWithCollectionViewBlock` 是一致的，通过 `toObjectsBlock` 来获取 `toObjects` ，这里使用 `block` 的原因是可以保证在获取 `toObjects` 时对应的数据源是最新的：

```swift
UICollectionView *collectionView = collectionViewBlock();
if (collectionView == nil) {
    [self _cleanStateAfterUpdates];
    executeCompletionBlocks(NO);
    [_delegate listAdapterUpdater:self didFinishWithoutUpdatesWithCollectionView:collectionView];
    return;
}
NSArray *toObjects = nil;
if (toObjectsBlock != nil) {
    toObjects = objectsWithDuplicateIdentifiersRemoved(toObjectsBlock());
}
```

3.定义 `executeUpdateBlocks` ，首先设置 `state` 为 `IGListBatchUpdateStateExecutingBatchUpdateBlock` ，防止多次执行。然后在执行 `itemUpdateBlock` 前先调用 `objectTransitionBlock` ，使得数据源更新到最新的 `toObjects` ，以保证数据源跟 UI 一致。执行 `itemUpdateBlock` ，在 `itemUpdateBlock` 中处理 `NSIndexPath` 对应的插入，删除和刷新操作。最后把 `batchUpdates.itemCompletionBlocks` 添加到 `completionBlocks` 中：

```swift
void (^executeUpdateBlocks)(void) = ^{
    self.state = IGListBatchUpdateStateExecutingBatchUpdateBlock;
    if (objectTransitionBlock != nil) {
        objectTransitionBlock(toObjects);
    }
    for (IGListItemUpdateBlock itemUpdateBlock in batchUpdates.itemUpdateBlocks) {
        itemUpdateBlock();
    }
    [completionBlocks addObjectsFromArray:batchUpdates.itemCompletionBlocks];
    self.state = IGListBatchUpdateStateExecutedBatchUpdateBlock;
};
```

4.定义 `reloadDataFallback` ，如果 `collectionView` 所在的 `window` 不可见，则直接 `reloadData` ，跳过 `diff` 操作。在 `reloadDataFallback` 的最后，调用 `_queueUpdateWithCollectionViewBlock:` ，防止丢失一些批量更新过程中进行的更新操作，如果在下一个 Runloop 过程中没有更新操作， `_queueUpdateWithCollectionViewBlock` 会直接退出。设置 `pendingTransitionToObjects` 为 `toObjects` ，在后续的更新中 `pendingTransitionToObjects` 作为 `fromObjects` 使用：

```swift
void (^reloadDataFallback)(void) = ^{
    [delegate listAdapterUpdater:self willReloadDataWithCollectionView:collectionView isFallbackReload:YES];
    executeUpdateBlocks();
    [self _cleanStateAfterUpdates];
    [self _performBatchUpdatesItemBlockApplied];
    [collectionView reloadData];
    [collectionView layoutIfNeeded];
    executeCompletionBlocks(YES);
    [delegate listAdapterUpdater:self didReloadDataWithCollectionView:collectionView isFallbackReload:YES];
    [self _queueUpdateWithCollectionViewBlock:collectionViewBlock];
};
[self _beginPerformBatchUpdatesToObjects:toObjects];
if (self.allowsBackgroundReloading && collectionView.window == nil) {
    reloadDataFallback();
    return;
}
```

5.定义 `batchUpdatesBlock` ，放到 `-[UICollectionView performBatchUpdates:completion:]` 第一个 `block` 参数中，如果 `singleItemSectionUpdates` 为 `YES` ，即每个 `section` 中只有 1 个 `item` ，那么可以在 `section` 层面进行处理，直接调用 `UICollectionView` 的操作 `section` 的相关方法即可：

```swift
void (^batchUpdatesBlock)(IGListIndexSetResult *result) = ^(IGListIndexSetResult *result){
    executeUpdateBlocks();
    if (self.singleItemSectionUpdates) {
        [collectionView deleteSections:result.deletes];
        [collectionView insertSections:result.inserts];
        for (IGListMoveIndex *move in result.moves) {
            [collectionView moveSection:move.from toSection:move.to];
        }
        self.applyingUpdateData = [[IGListBatchUpdateData alloc]
                                   initWithInsertSections:result.inserts
                                   deleteSections:result.deletes
                                   moveSections:[NSSet setWithArray:result.moves]
                                   insertIndexPaths:@[]
                                   deleteIndexPaths:@[]
                                   updateIndexPaths:@[]
                                   moveIndexPaths:@[]];
    } else {
        self.applyingUpdateData = IGListApplyUpdatesToCollectionView(collectionView,
                                                                     result,
                                                                     self.batchUpdates,
                                                                     fromObjects,
                                                                     experiments,
                                                                     self.sectionMovesAsDeletesInserts,
                                                                     self.preferItemReloadsForSectionReloads);
    }
    [self _cleanStateAfterUpdates];
    [self _performBatchUpdatesItemBlockApplied];
};
```

6.在 `IGListApplyUpdatesToCollectionView` 中针对 `reload` 操作进行特殊处理。`sectionReloads` 在手动调用 `reload` 方法时会记录对应的 `section` ，合并 `diff` 和手动 `reloads` 的 `section` 到 `reloads` 中，同时如果有需要的话使用 `delete + insert` 代替 `move` ：

```swift
NSMutableIndexSet *reloads = [diffResult.updates mutableCopy];
[reloads addIndexes:batchUpdates.sectionReloads];
NSMutableIndexSet *inserts = [diffResult.inserts mutableCopy];
NSMutableIndexSet *deletes = [diffResult.deletes mutableCopy];
NSMutableArray<NSIndexPath *> *itemUpdates = [NSMutableArray new];
if (sectionMovesAsDeletesInserts) {
    for (IGListMoveIndex *move in moves) {
        [deletes addIndex:move.from];
        [inserts addIndex:move.to];
    }
    moves = [NSSet new];
}
```

如之前提到的在 `performBatchUpdates` 中 `reload` 是不安全的，所以只有在 `moves/inserts/deletes` 都为 0 时才执行 `reload` 操作，否则使用 `delete + insert` 代替：

```swift
if (preferItemReloadsForSectionReloads
    && moves.count == 0 && inserts.count == 0 && deletes.count == 0 && reloads.count > 0) {
    [reloads enumerateIndexesUsingBlock:^(NSUInteger sectionIndex, BOOL * _Nonnull stop) {
        NSMutableIndexSet *localIndexSet = [NSMutableIndexSet indexSetWithIndex:sectionIndex];
        if (sectionIndex < [collectionView numberOfSections]
            && sectionIndex < [collectionView.dataSource numberOfSectionsInCollectionView:collectionView]
            && [collectionView numberOfItemsInSection:sectionIndex] == [collectionView.dataSource collectionView:collectionView numberOfItemsInSection:sectionIndex]) {
            [itemUpdates addObjectsFromArray:convertSectionReloadToItemUpdates(localIndexSet, collectionView)];
        } else {
            IGListConvertReloadToDeleteInsert(localIndexSet, deletes, inserts, diffResult, fromObjects);
        }
    }];
} else {
    IGListConvertReloadToDeleteInsert(reloads, deletes, inserts, diffResult, fromObjects);
}
```

将 itemReloads 转换为 `itemDeletes` + `itemInserts` ，生成最后的 `updateData` ，`collectionView` 根据 `updateData` 对 `item` 进行操作， `ig_applyBatchUpdateData:` 内部调用对应的 `delete/insert/move/reload` 方法：

```swift
NSMutableArray<NSIndexPath *> *itemInserts = batchUpdates.itemInserts;
NSMutableArray<NSIndexPath *> *itemDeletes = batchUpdates.itemDeletes;
NSMutableArray<IGListMoveIndexPath *> *itemMoves = batchUpdates.itemMoves;

NSSet<NSIndexPath *> *uniqueDeletes = [NSSet setWithArray:itemDeletes];
NSMutableSet<NSIndexPath *> *reloadDeletePaths = [NSMutableSet new];
NSMutableSet<NSIndexPath *> *reloadInsertPaths = [NSMutableSet new];
for (IGListReloadIndexPath *reload in batchUpdates.itemReloads) {
    if (![uniqueDeletes containsObject:reload.fromIndexPath]) {
        [reloadDeletePaths addObject:reload.fromIndexPath];
        [reloadInsertPaths addObject:reload.toIndexPath];
    }
}
[itemDeletes addObjectsFromArray:[reloadDeletePaths allObjects]];
[itemInserts addObjectsFromArray:[reloadInsertPaths allObjects]];

IGListBatchUpdateData *updateData = [[IGListBatchUpdateData alloc] initWithInsertSections:inserts
                                                                           deleteSections:deletes
                                                                             moveSections:moves
                                                                         insertIndexPaths:itemInserts
                                                                         deleteIndexPaths:itemDeletes
                                                                         updateIndexPaths:itemUpdates
                                                                           moveIndexPaths:itemMoves];
[collectionView ig_applyBatchUpdateData:updateData];
return updateData;
```

7.设置 `fallbackWithoutUpdates` ，在 `collectionView.dataSource` 为 `nil` 时调用：

```swift
void (^fallbackWithoutUpdates)(void) = ^(void) {
    executeCompletionBlocks(NO);
    [delegate listAdapterUpdater:self didFinishWithoutUpdatesWithCollectionView:collectionView];
    [self _queueUpdateWithCollectionViewBlock:collectionViewBlock];
};
```

8.设置 `batchUpdatesCompletionBlock` ，放到 `-[UICollectionView performBatchUpdates:completion:]` 第二个 `block` 参数中：

```swift
void (^batchUpdatesCompletionBlock)(BOOL) = ^(BOOL finished) {
    IGListBatchUpdateData *oldApplyingUpdateData = self.applyingUpdateData;
    executeCompletionBlocks(finished);
    [delegate listAdapterUpdater:self didPerformBatchUpdates:oldApplyingUpdateData collectionView:collectionView];
    [self _queueUpdateWithCollectionViewBlock:collectionViewBlock];
};
```

9.把 `[UICollectionView performBatchUpdates` 封装起来，如果在 `batchUpdatesBlock` 处理时崩溃的了，显示出来的第一行 App 符号就不是 `block` 了。 `block` 生成的名字会包含行数，如果行数调整了，就会被标记为不同的崩溃，这会对崩溃记录造成影响：

```swift
void (^performUpdate)(IGListIndexSetResult *) = ^(IGListIndexSetResult *result){
    [delegate listAdapterUpdater:self
willPerformBatchUpdatesWithCollectionView:collectionView
                     fromObjects:fromObjects
                       toObjects:toObjects
              listIndexSetResult:result
                        animated:animated];
    IGListAdapterUpdaterPerformBatchUpdate(collectionView, animated, ^{
        batchUpdatesBlock(result);
    }, batchUpdatesCompletionBlock);
};
```

10.初始化 `tryToPerformUpdate` ， `tryToPerformUpdate` 会把之前设置好的 `block` ，设置一个 `try-catch` ，防止崩溃，根据边界情况判断是否需要 `fallback` ：

```swift
void (^tryToPerformUpdate)(IGListIndexSetResult *) = ^(IGListIndexSetResult *result){
    if (!IGListExperimentEnabled(experiments, IGListExperimentSkipLayoutBeforeUpdate)) {
        [collectionView layoutIfNeeded];
    }

    @try {
        if (collectionView.dataSource == nil) {
            fallbackWithoutUpdates();
        } else if (result.changeCount > 100 && allowsReloadingOnTooManyUpdates) {
            reloadDataFallback();
        } else {
            performUpdate(result);
        }
    } @catch (NSException *exception) {
        [delegate listAdapterUpdater:self
                      collectionView:collectionView
              willCrashWithException:exception
                         fromObjects:fromObjects
                           toObjects:toObjects
                          diffResult:result
                             updates:(id)self.applyingUpdateData];
        @throw exception;
    }
};
```

11.最后通过 `diff` 算法计算出 `IGListIndexSetResult` ，调用 `tryToPerformUpdate(result)` ：

```swift
const BOOL onBackgroundThread = IGListExperimentEnabled(experiments, IGListExperimentBackgroundDiffing);
[delegate listAdapterUpdater:self willDiffFromObjects:fromObjects toObjects:toObjects];
IGListAdapterUpdaterPerformDiffing(fromObjects, toObjects, IGListDiffEquality, experiments, onBackgroundThread, ^(IGListIndexSetResult *result){
    [delegate listAdapterUpdater:self didDiffWithResults:result onBackgroundThread:onBackgroundThread];
    tryToPerformUpdate(result);
});
```

调用顺序如下：

```swift
// 使用 fromObjects 和 toObjects 计算出 diff
IGListAdapterUpdaterPerformDiffing -> 
// 判断是否需要执行更新，
tryToPerformUpdate ->
// 执行更新
performUpdate -> 
// 调用UICollectionView的performBatchUpdates方法
IGListAdapterUpdaterPerformBatchUpdate -> 
// batchUpdatesBlock中执行executeUpdateBlocks
batchUpdatesBlock -> 
// 执行 UICollectionView section 和 items 的相关操作
[UICollectionView section 和 items 操作]
```

完成刷新后调用相关的 `block` ：

```swift
batchUpdatesCompletionBlock -> executeCompletionBlocks 
```

### IGListReloadDataUpdater

除了支持批量刷新的 `IGListAdapterUpdater` ，`IGListKit` 还提供了仅支持全局刷新的 `IGListReloadDataUpdater` ，实现非常简单，且执行的是 `[UICollectionView reloadData]` 。其所有 `IGListUpdatingDelegate` 的相关方法都会调用 `_synchronousReloadDataWithCollectionView:` 方法进行更新：

```swift
- (void)performUpdateWithCollectionViewBlock:(IGListCollectionViewBlock)collectionViewBlock
                            fromObjects:(NSArray *)fromObjects
                         toObjectsBlock:(IGListToObjectBlock)toObjectsBlock
                               animated:(BOOL)animated
                  objectTransitionBlock:(IGListObjectTransitionBlock)objectTransitionBlock
                             completion:(IGListUpdatingCompletion)completion {
    if (toObjectsBlock != nil) {
        NSArray *toObjects = toObjectsBlock() ?: @[];
        objectTransitionBlock(toObjects);
    }
    [self _synchronousReloadDataWithCollectionView:collectionViewBlock()];
    if (completion) {
        completion(YES);
    }
}

- (void)_synchronousReloadDataWithCollectionView:(UICollectionView *)collectionView {
    [collectionView reloadData];
    [collectionView layoutIfNeeded];
}
```

可以看到实现非常简单，如果只需要进行 `reloadData` ，可以使用 `IGListReloadDataUpdater` 替换掉 `IGListAdapterUpdater` 。

### Diff

在 iOS 还没有系统级地支持 Diff 特性的年代，在使用 `UITableView/UICollectionView` 时，当数据源发生变化，我们就需要手动根据数据源计算出变化的 `NSIndexPaths` 并进行更新，这个方法的时间复杂度一般是 `O(n^2)` ，在遍历旧数据内对新数据进行遍历，或者说直接 `reloadData` ，在 `UITableView/UICollectionView` 的复用机制下，只需要重新生成显示在屏幕的 `Cell` ，所带来的影响只是丢失了动画。而 `IGListKit` 的 `IGListDiff` 可以在时间复杂度 `O(n)` 的前提下为我们计算出对应的 `NSIndexPaths` 简单易易用，再也不需要直接 `reloadData` 。下面来说说 `IGListDiff` 的核心实现。

```swift
/// 记录 Diff 所需要的状态
struct IGListEntry {
    /// 记录旧数组中具有相同 hash 值的对象出现次数
    NSInteger oldCounter = 0;
    /// 记录新数组中具有相同 hash 值的对象出现次数
    NSInteger newCounter = 0;
    /// The indexes of the data in the old array
    /// 记录旧数组中当前 hash 对应的对象出现的位置 
    stack<NSInteger> oldIndexes;
    /// 数据是否有更新
    BOOL updated = NO;
};

/// 记录 IGListEntry 和位置（ index ）， index 默认为 NSNotFound
struct IGListRecord {
    IGListEntry *entry;
    mutable NSInteger index;
    IGListRecord() {
        entry = NULL;
        index = NSNotFound;
    }
};
```

1.首先先获取新旧数据所对应的数量: `newCount` 和 `oldCount` ，然后创建新旧数据所对应的 `NSMapTable` 

```swift
static id IGListDiffing(BOOL returnIndexPaths,
                        NSInteger fromSection,
                        NSInteger toSection,
                        NSArray<id<IGListDiffable>> *oldArray,
                        NSArray<id<IGListDiffable>> *newArray,
                        IGListDiffOption option) {
    const NSInteger newCount = newArray.count;
    const NSInteger oldCount = oldArray.count;
    NSMapTable *oldMap = [NSMapTable strongToStrongObjectsMapTable];
    NSMapTable *newMap = [NSMapTable strongToStrongObjectsMapTable];
}
```

2.如果 `newCount` 为 0 ，那么就是说 `oldArray` 的所有数据都需要进行删除，那么我们可以尽早返回，生成一个删除所有数据的 `IGListIndexPathResult/IGListIndexSetResult` ：

```swift
if (newCount == 0) {
    if (returnIndexPaths) {
        return [[IGListIndexPathResult alloc] initWithInserts:[NSArray new]
                                                      deletes:indexPathsAndPopulateMap(oldArray, fromSection, oldMap)
                                                      updates:[NSArray new]
                                                        moves:[NSArray new]
                                              oldIndexPathMap:oldMap
                                              newIndexPathMap:newMap];
    } else {
        [oldArray enumerateObjectsUsingBlock:^(id<IGListDiffable> obj, NSUInteger idx, BOOL *stop) {
            addIndexToMap(returnIndexPaths, fromSection, idx, obj, oldMap);
        }];
        return [[IGListIndexSetResult alloc] initWithInserts:[NSIndexSet new]
                                                     deletes:[NSIndexSet indexSetWithIndexesInRange:NSMakeRange(0, oldCount)]
                                                     updates:[NSIndexSet new]
                                                       moves:[NSArray new]
                                                 oldIndexMap:oldMap
                                                 newIndexMap:newMap];
    }
}
```

3.如果 `oldCount` 为 0 ，那么就是说 `newArray` 的所有数据都需要进行插入，那么我们可以尽早返回，生成一个插入所有数据的 `IGListIndexPathResult/IGListIndexSetResult` ：

```swift
if (oldCount == 0) {
    if (returnIndexPaths) {
        return [[IGListIndexPathResult alloc] initWithInserts:indexPathsAndPopulateMap(newArray, toSection, newMap)
                                                      deletes:[NSArray new]
                                                      updates:[NSArray new]
                                                        moves:[NSArray new]
                                              oldIndexPathMap:oldMap
                                              newIndexPathMap:newMap];
    } else {
        [newArray enumerateObjectsUsingBlock:^(id<IGListDiffable> obj, NSUInteger idx, BOOL *stop) {
            addIndexToMap(returnIndexPaths, toSection, idx, obj, newMap);
        }];
        return [[IGListIndexSetResult alloc] initWithInserts:[NSIndexSet indexSetWithIndexesInRange:NSMakeRange(0, newCount)]
                                                     deletes:[NSIndexSet new]
                                                     updates:[NSIndexSet new]
                                                       moves:[NSArray new]
                                                 oldIndexMap:oldMap
                                                 newIndexMap:newMap];
    }
}
```

4.如果 `newCount` 和 `oldCount` 都不为 0 ，那么就可以进行 Diff 计算，首先需要创建一个 `table` ，使用 `diffIdentifier` 为 `key` ， `IGListEntry` 为 `value` ，这里使用 `unordered_map` ，因为会比 `NSDictionary` 快很多。

```swift
unordered_map<id<NSObject>, IGListEntry, IGListHashID, IGListEqualID> table;
```

5.遍历 `newArray` 的数据，获取对应的 `entry` ， `entry` 的 `oldIndexes` 插入 `NSNotFound` ，且设置到 `newResultsArray` 中：

```swift
vector<IGListRecord> newResultsArray(newCount);
for (NSInteger i = 0; i < newCount; i++) {
    id<NSObject> key = IGListTableKey(newArray[i]);
    IGListEntry &entry = table[key];
    entry.newCounter++;
    entry.oldIndexes.push(NSNotFound);
    newResultsArray[i].entry = &entry;
}
```

6.遍历 `oldArray` 的数据，获取对应的 `entry` ， `entry` 的 `oldIndexes` 插入对应的位置 `i` ，且设置到 `oldResultsArray` 中：

```swift
vector<IGListRecord> oldResultsArray(oldCount);
for (NSInteger i = oldCount - 1; i >= 0; i--) {
    id<NSObject> key = IGListTableKey(oldArray[i]);
    IGListEntry &entry = table[key];
    entry.oldCounter++;
    entry.oldIndexes.push(i);
    oldResultsArray[i].entry = &entry;
}
```

7.处理那些在 `oldArray` 和 `newArray` 中都有出现的数据，遍历 `newResultsArray` ，根据 `entry` 的数据进行对比：

```swift
for (NSInteger i = 0; i < newCount; i++) {
    IGListEntry *entry = newResultsArray[i].entry;
		
    // 1. 获取原始位置 originalIndex , 如果 item 是插入的数据，那么 originalIndex 就为 NSNotFound
    const NSInteger originalIndex = entry->oldIndexes.top();
    entry->oldIndexes.pop();
    if (originalIndex < oldCount) {
        // 获取新旧数据的对象
        const id<IGListDiffable> n = newArray[i];
        const id<IGListDiffable> o = oldArray[originalIndex];
        switch (option) {
            case IGListDiffPointerPersonality:
                // 只通过指针进行比较
                if (n != o) {
                    entry->updated = YES;
                }
                break;
            case IGListDiffEquality:
                // 使用 `-[IGListDiffable isEqualToDiffableObject:]` 进行比较，以 n 和 o 指向的对象不同为前提
                if (n != o && ![n isEqualToDiffableObject:o]) {
                    entry->updated = YES;
                }
                break;
        }
    }
    if (originalIndex != NSNotFound
        && entry->newCounter > 0
        && entry->oldCounter > 0) {
        // 如果在 `newArray` 和 `oldArray` 中都出现，则进行位置的双向绑定
        newResultsArray[i].index = originalIndex;
        oldResultsArray[originalIndex].index = i;
    }
}
```

8.创建需要记录数据：

```swift
// 存储最后的 NSIndexPaths 或者 indexes
id mInserts, mMoves, mUpdates, mDeletes;
if (returnIndexPaths) {
    mInserts = [NSMutableArray<NSIndexPath *> new];
    mMoves = [NSMutableArray<IGListMoveIndexPath *> new];
    mUpdates = [NSMutableArray<NSIndexPath *> new];
    mDeletes = [NSMutableArray<NSIndexPath *> new];
} else {
    mInserts = [NSMutableIndexSet new];
    mMoves = [NSMutableArray<IGListMoveIndex *> new];
    mUpdates = [NSMutableIndexSet new];
    mDeletes = [NSMutableIndexSet new];
}
// 追踪删除的 items 的偏移量来计算 items 的移动位置vector<NSInteger> deleteOffsets(oldCount), insertOffsets(newCount);
NSInteger runningOffset = 0;
```

9.计算删除的数据：

```swift
for (NSInteger i = 0; i < oldCount; i++) { 
  deleteOffsets[i] = runningOffset; 
  const IGListRecord record = oldResultsArray[i]; 
  // 如果 record.index 为 NSNotFound ，则表示其没有在 newArray 中出现，已被删除 
  if (record.index == NSNotFound) { 
    addIndexToCollection(returnIndexPaths, mDeletes, fromSection, i); 
    runningOffset++; 
  }
	addIndexToMap(returnIndexPaths, fromSection, i, oldArray[i], oldMap); 
}
```

10.最后的计算

```swift
for (NSInteger i = 0; i < newCount; i++) {
    insertOffsets[i] = runningOffset;
    const IGListRecord record = newResultsArray[i];
    const NSInteger oldIndex = record.index;
    // 如果 `record.index` 为 `NSNotFound` ，则表示其没有在 `oldArray` 中出现，是新插入的
    if (record.index == NSNotFound) {
        addIndexToCollection(returnIndexPaths, mInserts, toSection, i);
        runningOffset++;
    } else {
        // 如果 record.entry-> updated 为 YES ，则表示
        if (record.entry->updated) {
            addIndexToCollection(returnIndexPaths, mUpdates, fromSection, oldIndex);
        }
        // 计算 indexes 是否匹配，据此来判断是否需要移动
		  // oldIndex - deleteOffset + insertOffset != i ，则位置发生变化，需要移动。
        const NSInteger insertOffset = insertOffsets[i];
        const NSInteger deleteOffset = deleteOffsets[oldIndex];
        if ((oldIndex - deleteOffset + insertOffset) != i) {
            id move;
            if (returnIndexPaths) {
                NSIndexPath *from = [NSIndexPath indexPathForItem:oldIndex inSection:fromSection];
                NSIndexPath *to = [NSIndexPath indexPathForItem:i inSection:toSection];
                move = [[IGListMoveIndexPath alloc] initWithFrom:from to:to];
            } else {
                move = [[IGListMoveIndex alloc] initWithFrom:oldIndex to:i];
            }
            [mMoves addObject:move];
        }
    }
    addIndexToMap(returnIndexPaths, toSection, i, newArray[i], newMap);
}
```

11.完成计算，返回结果：

```swift
if (returnIndexPaths) {
    return [[IGListIndexPathResult alloc] initWithInserts:mInserts
                                                  deletes:mDeletes
                                                  updates:mUpdates
                                                    moves:mMoves
                                          oldIndexPathMap:oldMap
                                          newIndexPathMap:newMap];
} else {
    return [[IGListIndexSetResult alloc] initWithInserts:mInserts
                                                 deletes:mDeletes
                                                 updates:mUpdates
                                                   moves:mMoves
                                             oldIndexMap:oldMap
                                             newIndexMap:newMap];
}
```

从上面的计算可以看出需要 5 次 `for` 循环进行遍历，也就是时间复杂度为 `O(5n)` ，在 `n` 足够大的情况下可以忽略，时间复杂度可视作 `O(n)` 。 这篇文章有使用两个数组作为例子进行说明 `IGListDiff` 是如何进行计算的： [IGListKit diff 实现简析](https://xiangwangfeng.com/2017/03/16/IGListKit-diff-实现简析/) 。

## IGListCollectionViewLayout

`IGListCollectionViewLayout` 其实不太算得上是 `IGListKIt` 的内容，它主要作用是提供一个可变宽度和高度的流式布局。看下来发现这块写得太好了，从里面可以学到很多如何高效编写一个自定义的 `UICollectionViewLayout` 的相关技巧。

`IGListCollectionViewLayout` 提供了一些常用的 `static` 的方法，用于在计算布局时根据滑动方向获取不同的属性，其中一个例子如下：

```swift
static CGFloat UIEdgeInsetsLeadingInsetInDirection(UIEdgeInsets insets, UICollectionViewScrollDirection direction) {
    switch (direction) {
        case UICollectionViewScrollDirectionVertical: return insets.top;
        case UICollectionViewScrollDirectionHorizontal: return insets.left;
    }
}
```

设置 `UICollectionViewLayoutAttributes` 的 `zIndex` ，这样使得 `FooterView` 可以在 `sticky` 时不被 `Cell` 覆盖：

```swift
static void adjustZIndexForAttributes(UICollectionViewLayoutAttributes *attributes) {
    const NSInteger maxZIndexPerSection = 1000;
    const NSInteger baseZIndex = attributes.indexPath.section * maxZIndexPerSection;

    switch (attributes.representedElementCategory) {
        case UICollectionElementCategoryCell:
            attributes.zIndex = baseZIndex + attributes.indexPath.item;
            break;
        case UICollectionElementCategorySupplementaryView:
            attributes.zIndex = baseZIndex + maxZIndexPerSection - 1;
            break;
        case UICollectionElementCategoryDecorationView:
            attributes.zIndex = baseZIndex - 1;
            break;
    }
}
```

这里取了个巧，假设每个 `Section` 的 `Item` 数量不超过 1000 个，每个 `Section` 的起始 `zIndex` 为 `baseZIndex` ，值为 `attributes.indexPath.section * maxZIndexPerSection` ，然后根据 `attributes.representedElementCategory` 进行判断：

1. `UICollectionElementCategoryCell` ，`baseZIndex + attributes.indexPath.item` ，根据 `indexPath.item` 进行叠加；
2. `UICollectionElementCategorySupplementaryView` ，位于每个 `Section` 的顶部，所以 `zIndex` 为 `baseZIndex + maxZIndexPerSection - 1` ；
3. `UICollectionElementCategoryDecorationView` 用于设置背景，所以应该位于最底部；

一般来说 iOS 应该很少出现单个 `Section` 超过 1000 ，如果出现了而又设置 `stickyHeaders` 为 `true` ，那么就可能会出现 `Cell` 把 `HeaderView` 覆盖的情况。

如何实现 `stickyHeaders` 功能。 `IGListKit` 自定义了一个 `UICollectionViewLayoutInvalidationContext` 的子类 `IGListCollectionViewLayoutInvalidationContext` ，用于在布局信息失效时提供相关变量：

```swift
@interface IGListCollectionViewLayoutInvalidationContext : UICollectionViewLayoutInvalidationContext
@property (nonatomic, assign) BOOL ig_invalidateSupplementaryAttributes;
@property (nonatomic, assign) BOOL ig_invalidateAllAttributes;
@end
```

`ig_invalidateSupplementaryAttributes` 表示 `Header` 和 `Footer` 相关布局信息都失效，需要重新计算； `ig_invalidateAllAttributes` 表示所有布局信息都失效，都需要重新计算；

如果需要自定义 `UICollectionViewLayoutInvalidationContext` ，需要重写下面的方法，返回对应的子类：

```swift
+ (Class)invalidationContextClass {
    return [IGListCollectionViewLayoutInvalidationContext class];
}
```

当 `UICollectionView` 的 `bounds` 将要发生变化时，会调用 `shouldInvalidateLayoutForBoundsChange:` 方法，如果返回 `YES` 则会调用 `invalidationContextForBoundsChange:` 获取新的 `IGListCollectionViewLayoutInvalidationContext` 。

```swift
- (BOOL)shouldInvalidateLayoutForBoundsChange:(CGRect)newBounds {
  
    const CGRect oldBounds = self.collectionView.bounds;
    // 如果 size 改变了，
    if (!CGSizeEqualToSize(oldBounds.size, newBounds.size)) {
        return YES;
    }
    // 2.
    if (CGRectGetMinInDirection(newBounds, self.scrollDirection) != CGRectGetMinInDirection(oldBounds, self.scrollDirection)) {
        return self.stickyHeaders;
    }
    return NO;
}
```

1. 如果 `size` 改变了， 布局肯定是会失效的，所以这里直接返回 `YES` ；
2. 如果滑动方向上的坐标改变了，则返回 `stickyHeaders` 的值，因为当 `stickyHeaders` 为 `YES` 时，我们需要重新计算 `Header` 的布局；

```swift
- (UICollectionViewLayoutInvalidationContext *)invalidationContextForBoundsChange:(CGRect)newBounds {
    const CGRect oldBounds = self.collectionView.bounds;
    IGListCollectionViewLayoutInvalidationContext *context =
    (IGListCollectionViewLayoutInvalidationContext *)[super invalidationContextForBoundsChange:newBounds];
    context.ig_invalidateSupplementaryAttributes = YES;
    if (!CGSizeEqualToSize(oldBounds.size, newBounds.size)) {
        context.ig_invalidateAllAttributes = YES;
    }
    return context;
}
```

整体流程如下： 

1. 创建一个自定义的 `IGListCollectionViewLayoutInvalidationContext` ，通过 `ig_invalidateSupplementaryAttributes` 来标记 `supplementary attributes` 失效； 
2.  在 `-shouldInvalidateLayoutForBoundsChange:` 中返回 `YES` ； 
3.  在 `-invalidationContextForBoundsChange:` 标记 `IGListCollectionViewLayoutInvalidationContext` 的 `ig_invalidateSupplementaryAttributes` 为 `YES` ； 
4.  在 `-invalidateLayoutWithContext:` 方法中，如果 `context` 的 `ig_invalidateSupplementaryAttributes` 为 `YES` ，则清除 `supplementaryAttributesCache` ； 
5. `-layoutAttributesForSupplementaryViewOfKind:atIndexPath:` 获取布局信息时，先检查 `supplementaryAttributesCache` 是否有对应的布局信息，如果没有，则重新生成；  
6. 确保 `-layoutAttributesForElementsInRect:` 通过 `-layoutAttributesForSupplementaryViewOfKind:atIndexPath:` 获取布局信息。