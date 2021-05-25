---
layout: post
title: "二叉树"
date: 2016-11-20 21:15:00.000000000 +09:00
categories: [算法]
tags: [算法, 二叉树]
---

**树**是数据结构中的重中之重，尤其以各类二叉树为学习的难点。一直以来，对于树的掌握都是模棱两可的状态，现在希望通过写一个关于二叉树的专题系列。在学习与总结的同时更加深入的了解掌握二叉树。本系列文章将着重介绍一般二叉树、完全二叉树、满二叉树、[线索二叉树](https://www.jianshu.com/p/3965a6e424f5)、[霍夫曼树](https://www.jianshu.com/p/5ad3e97d54a3)、[二叉排序树](https://www.jianshu.com/p/bbe133625c73)、平衡二叉树、红黑树、B树。希望各位读者能够关注专题，并给出相应意见，通过系列的学习做到心中有“树”。

本期的内容有:

- 基本概念：实现，深度 ，二叉查找树
- 遍历
- 苹果面试题：在iOS中展示二叉树

**结点概念**

**结点**是数据结构中的基础，是构成复杂数据结构的基本组成单位。

**树结点声明**

本系列文章中提及的结点专指树的结点。例如：结点A在图中表示为：

![](/aasets/images/al-tree-01.png)

**树的定义**

**树（Tree）**是n（n>=0)个结点的有限集。n=0时称为空树。在任意一颗非空树中：
 1）有且仅有一个特定的称为根（Root）的结点；
 2）当n>1时，其余结点可分为m(m>0)个互不相交的有限集T1、T2、......、Tn，其中每一个集合本身又是一棵树，并且称为根的子树。

此外，树的定义还需要强调以下两点：
 1）n>0时根结点是唯一的，不可能存在多个根结点，数据结构中的树只能有一个根结点。
 2）m>0时，子树的个数没有限制，但它们一定是互不相交的。
 示例树：
 图2.1为一棵普通的树：

![](/aasets/images/al-tree-02.png)

由树的定义可以看出，树的定义使用了递归的方式。递归在树的学习过程中起着重要作用，如果对于递归不是十分了解，建议先看看[递归算法](https://blog.csdn.net/feizaosyuacm/article/details/54919389)。

**结点的度**

结点拥有的子树数目称为结点的**度**。
图2.2中标注了图2.1所示树的各个结点的度。

![](/aasets/images/al-tree-03.png)

**结点关系**

结点子树的根结点为该结点的**孩子结点**。相应该结点称为孩子结点的**双亲结点**。
如上图，A为B的双亲结点，B为A的孩子结点。
同一个双亲结点的孩子结点之间互称**兄弟结点**。
结点B与结点C互为兄弟结点。

**结点层次**

从根开始定义起，根为第一层，根的孩子为第二层，以此类推。

![](/aasets/images/al-tree-04.png)

**树的深度**

树中结点的最大层次数称为树的深度或高度。如上图树的深度为4。

## 二叉树

**二叉树定义**

**二叉树**是n(n>=0)个结点的有限集合，该集合或者为空集（称为空二叉树），或者由一个根结点和两棵互不相交的、分别称为根结点的左子树和右子树组成。

![](/aasets/images/al-tree-05.png)

**二叉树特点**

由二叉树定义以及图示分析得出二叉树有以下特点：

+ 每个结点最多有两颗子树，所以二叉树中不存在度大于2的结点。
+ 左子树和右子树是有顺序的，次序不能任意颠倒。
+ 即使树中某结点只有一棵子树，也要区分它是左子树还是右子树。

**二叉树性质**

1. 在二叉树的第i层上最多有2i-1 个节点 。（i>=1）
2. 二叉树中如果深度为k,那么最多有2k-1个节点。(k>=1）
3. n0=n2+1  n0表示度数为0的节点数，n2表示度数为2的节点数。
    4）在完全二叉树中，具有n个节点的完全二叉树的深度为[log2n]+1，其中[log2n]是向下取整。
4. 若对含 n 个结点的完全二叉树从上到下且从左至右进行 1 至 n 的编号，则对完全二叉树中任意一个编号为 i 的结点有如下特性：

> (1) 若 i=1，则该结点是二叉树的根，无双亲, 否则，编号为 [i/2] 的结点为其双亲结点;
> (2) 若 2i>n，则该结点无左孩子，  否则，编号为 2i 的结点为其左孩子结点；
> (3) 若 2i+1>n，则该结点无右孩子结点，  否则，编号为2i+1 的结点为其右孩子结点。

**斜树**

所有的结点都只有左子树的二叉树叫左斜树。所有结点都是只有右子树的二叉树叫右斜树。这两者统称为斜树。

![](/aasets/images/al-tree-06.png)

**满二叉树**

满二叉树: 在一棵二叉树中。如果所有分支结点都存在左子树和右子树，并且所有叶子都在同一层上，这样的二叉树称为满二叉树。
 满二叉树的特点有：
 1）叶子只能出现在最下一层。出现在其它层就不可能达成平衡。
 2）非叶子结点的度一定是2。
 3）在同样深度的二叉树中，满二叉树的结点个数最多，叶子数最多。

![](/aasets/images/al-tree-07.png)

**完全二叉树**

完全二叉树: 对一颗具有n个结点的二叉树按层编号，如果编号为i(1<=i<=n)的结点与同样深度的满二叉树中编号为i的结点在二叉树中位置完全相同，则这棵二叉树称为完全二叉树。

![](/aasets/images/al-tree-08.png)

**特点**：
 1）叶子结点只能出现在最下层和次下层。
 2）最下层的叶子结点集中在树的左部。
 3）倒数第二层若存在叶子结点，一定在右部连续位置。
 4）如果结点度为1，则该结点只有左孩子，即没有右子树。
 5）同样结点数目的二叉树，完全二叉树深度最小。
 **注**：满二叉树一定是完全二叉树，但反过来不一定成立。

## 二叉树的存储结构

**顺序存储**

二叉树的顺序存储结构就是使用一维数组存储二叉树中的结点，并且结点的存储位置，就是数组的下标索引。

![](/aasets/images/al-tree-09.png)

如上图所示一颗完全二叉树采用顺序存储方式。

![](/aasets/images/al-tree-10.png)

当二叉树为完全二叉树时，结点数刚好填满数组。那么当二叉树不为完全二叉树时，采用顺序存储形式如何呢？

![](/aasets/images/al-tree-11.png)

如上图其中浅色结点表示结点不存在时，那么是怎样存储的呢?

![](/aasets/images/al-tree-12.png)

## 二叉树的特性

一个除头结点外，每个节点只有一个前驱，有零到两个后继的树即为二叉树。在二叉树中，一个节点可以有左节点或者左子树，也可以有右节点或者右子树。一些特殊的二叉树，比如斜二叉树、满二叉树、完全二叉树等等就不做过多赘述了。说这么多，不如看一张图来的直观。下方就是一个典型的二叉树。

![](/aasets/images/al-tree-13.png)

了解二叉树，理解其特性还是比较重要的。基于二叉树本身的逻辑结构，下方是二叉树这种数据结构所具备的特性。

- 特性1：

  在二叉树的第i层上至多有2^(i-1)（i >= 1）个节点

  - 这一特性比较好理解，如果层数是从零开始数的话，那么低i层上的节点数就是2^i，因为二叉树层与层之间的节点数是以2的指数幂进行增长的。如果根节点算是第0层的话，那么第n层的节点数就是2^n次幂。

- 特性2：

  深度为k的二叉树至多有2^k-1（k>=1）个节点

  - 这一特性也是比较好理解的, 由数学上的递加公式就可以很容易的推出来。由特性1易知每层最多有多少个节点，那么深度为k的话，说明一共有k层，那么共有节点数为：2^0 + 2^1 + 2^2 + 2^(k-1) = 2^k - 1。

- 特性3：

  二叉树的叶子节点数为n0, 度为2的节点数为n2, 那么n0 = n2 + 1

  - 这一特性也不难理解，推出n0 = n2 + 1这个公式并不难。我们假设叶子节点，也就是度数为0的节点的个数为n0, 度数为1的节点为n1, 度数为2的节点n2。那么二叉树的节点总数 n = n0 + n1 + n2。因为除了根节点外其余的节点入度都为1，所以二叉树的度数为n-1，当然度的个数可以使用出度来算，即为2*n2+n1，所以n-1=2*n2+n1。以n=n0+n1+n2与n-1=2*n2+n1这两个公式我们很容易的推出n0 = n2 + 1。

- 特性4：

  具有n个结点的完全二叉树的深度为log2n + 1 （向下取整，比如3.5，就取3）

  - 这个特性也是比较好理解的，基于完全二叉树的特点，我们假设完全二叉树的深度为k, 那么二叉树的结点个数的范围为2(k-1)-1 <= n <= 2k-1。由这个表达式我们很容易推出特性4。

## 二叉树的创建

首先介绍下二叉树。二叉树中每个节点最多有两个子节点，一般称为左子节点和右子节点，并且二叉树的子树有左右之分，其次序不能任意颠倒。下面是节点的Swift实现：

```swift
// 树节点
public class TreeNode {
    
    public var val: String
    public var left: TreeNode?
    public var right: TreeNode?
    
    public init(_ val: String) {
        self.val = val
        self.left = nil
        self.right = nil
    }
}

public class Tree {
    
    var values: [String]
    var index: Int = -1

    init(_ values: [String]) {
        self.values = values
    }
    
    func createTreeNode() -> TreeNode? {
        
        self.index += 1
        if self.index < self.values.count && self.index >= 0 {
            let value = self.values[self.index]
            if value == "" {
                 return nil
            } else {
                let rootNode = TreeNode(value)
                rootNode.left = createTreeNode()
                rootNode.right = createTreeNode()
                return rootNode
            }
        }
        return nil
    }
}
```

一般在面试中，会给定二叉树的根节点。要访问任何其他节点，只要从起始节点开始往左/右走即可。
在树中，节点的层次从根开始定义，根为第一层，树中节点的最大层次为树的**深度**。

```swift
// 计算树的最大深度
func maxDepth(_ root: TreeNode?) -> Int {
        
    guard let root = root else {
        return 0
    }
    return max(maxDepth(root.left), maxDepth(root.right)) + 1
}
```

面试中，最常见的是二叉查找树，它是一种特殊的二叉树。它的特点就是左子树中节点的值都小于根节点的值，右子树中节点的值都大于根节点的值。那么问题来了，给你一棵二叉树，怎么判断它是二叉查找树？我们根据定义，可以写出以下解法：

```swift
// 判断一颗二叉树是否为二叉查找树
func isValidBST(root: TreeNode?) -> Bool {
  return _helper(root, nil, nil)
}
    
private func _helper(node: TreeNode?, _ min: Int?, _ max: Int?) -> Bool {
  guard let node = node else {
    return true
  }
  // 所有右子节点都必须大于根节点
  if let min = min, node.val <= min {
    return false
  }
  // 所有左子节点都必须小于根节点
  if let max = max, node.val >= max {
    return false
  }
        
  return _helper(node.left, min, node.val) && _helper(node.right, node.val, max)
}
```

上面的代码有这几个点指点注意：

1. 二叉树本身是由递归定义的，所以原理上所有二叉树的题目都可以用递归来解
2. 二叉树这类题目很容易就会牵涉到往左往右走，所以写helper函数要想到有两个相对应的参数
3. 记得处理节点为nil的情况，尤其要注意根节点为nil的情况

## 二叉树遍历

**定义**

**二叉树的遍历**是指从二叉树的根结点出发，按照某种次序依次访问二叉树中的所有结点，使得每个结点被访问一次，且仅被访问一次。

聊二叉树怎么能没有二叉树的遍历呢，下方就会给出几种常见的二叉树的遍历方法。在遍历二叉树的方法中一般有先序遍历，中序遍历，后续遍历，层次遍历。本篇博客主要给出前三种遍历方式，而层次遍历会在图的部分进行介绍。二叉树的层次遍历其实与图的广度搜索是一样的，所以这部分放到图的相关博客中介绍。下方会给出几种遍历的具体方式，然后给出具体的代码实现。

二叉树的先、中、后遍历，这个先中后指的是遍历根节点的先后顺序。先序遍历：根左右，中序遍历：左根右，后序遍历：左右根。下方将详细介绍到。

**前序遍历**

关于先序遍历，上面已经介绍过一些了，接下来再进行细化一下。先序遍历，就是先遍历根节点然后再遍历左子树，最后遍历右子树。下图就是我们上面创建的二叉树的先序遍历的顺序，由下方的示例图就可以看出先序遍历的规则。一句话总结下方的结构图：根节点->左节点->右节点。下方先序遍历的顺序为：A B D 空 空 E 空 空 C 空 F 空 空 。

![](/aasets/images/al-tree-14.png)

上面给出了原理，接下来又到了代码实现的时候了。在树的遍历时，我们依然是采用递归的方式，因为无论是左子树还是右子树，都是二叉树的范畴。所以在进行二叉树遍历时，可以使用递归遍历的形式。而先序遍历莫非就是先遍历根节点，然后递归遍历左子树，最后遍历右子树。下方就是先序遍历的代码实现。在下方代码中，如果左节点或者右节点为空，那么我们就输出“空”。

```swift
// 先序遍历: 先遍历根节点然后再遍历左子树，最后遍历右子树。
func preOrderTraverse(_ node: TreeNode?){

    guard let node = node else {
        print("", terminator: "")
        return
    }
    print("\(node.val) ", terminator: "")
    preOrderTraverse(node.left)
    preOrderTraverse(node.right)
}
```

**中序遍历**

中序遍历，与先序遍历的不同之处在于，中序遍历是先遍历左子树，然后遍历根节点，最后遍历右子树。一句话总结：左子树->根节点->右子树。下方就是我们之前创建的树的中序遍历的结构图以及中序遍历的结果。

![](/aasets/images/al-tree-15.png)

中序遍历的代码实现与先序遍历的代码实现类似，都是使用递归的方式来实现的，只不过是先递归遍历左子树，然后遍历根节点，最后遍历右子树。下方就是中序遍历的代码具体实现。

```swift
// 中序遍历: 先遍历左子树，然后遍历根节点，最后遍历右子树。
func inOrderTraverse (_ node: TreeNode?) {
    guard let node = node else {
        print("", terminator: "")
        return
    }
    inOrderTraverse(node.left)
    print("\(node.val) ", terminator: "")
    inOrderTraverse(node.right)
}
```

**后序遍历**

接下来聊一下二叉树的后序遍历。如果上面这两种遍历方式理解的话，那么后序遍历也是比较好理解的。后序遍历是先遍历左子树，然后再遍历右子树，最后遍历根节点。与上方的表示方法一直，首先我们给出表示图，如下所示：

![](/aasets/images/al-tree-16.png)

后序遍历的代码就不做过多赘述了，与之前两种依然类似，只是换了一下遍历的顺序。下方就是二叉树后序遍历的代码实现。

```swift
// 后序遍历: 后序遍历是先遍历左子树，然后再遍历右子树，最后遍历根节点
func afterOrderTraverse (_ node: TreeNode?) {
    guard let node = node else {
        print("", terminator: "")
        return
    }
    afterOrderTraverse(node.left)
    afterOrderTraverse(node.right)
    print("\(node.val) ", terminator: "")
}
```

**层次遍历**

二叉树的层次遍历就不是二叉树这种数据结构所独有的了。后面的博客中我们会介绍到图这种数据结构，在图中有一个广度搜索，放到二叉树中就是层次遍历。也就是说二叉树的层次遍历，就是图中以二叉树的根节点为起始节点的广度搜索（BFS）。本篇博客就不给出具体的代码了，后面的博客会给出BFS的具体算法。当然在之前的博客中有图的BFS以及DFS。不过是C语言的实现。下方就是二叉树层次遍历的实例图。

![](/aasets/images/al-tree-17.png)

```swift
//层次遍历: 层次遍历相对上面的几个遍历实现起来要稍微复杂，层次遍历就是图中以二叉树的根节点为起始节点的广度搜索（BFS）
func levelOrder(_ root: TreeNode?){

    var result = [[TreeNode]]()
    var level = [TreeNode]()
    if let root = root {
        level.append(root)
    }
    while level.count != 0 {
        result.append(level)
        var nextLevel = [TreeNode]()
        for node in level {
            if let leftNode = node.left {
                nextLevel.append(leftNode)
            }
            if let rightNode = node.right {
                nextLevel.append(rightNode)
            }
        }
        level = nextLevel
    }

    let res = result.map { $0.map { $0.val }}
    print("\(res) ", terminator: "")
}
```

```swift
// 除了用递归方式遍历，可以用栈实现
func stackToTraversal(_ rootNode: TreeNode?) -> [String] {

    var result = [String]()
    var stack = [TreeNode]()
    var node = rootNode
    while !stack.isEmpty || node != nil {
        if node != nil {
            result.append(node!.val)
            stack.append(node!)
            node = node!.left
        } else {
            node = stack.removeLast().right
        }
    }
    return result
}
```

**测试**

```swift
// 二叉树
func treeNodeTest() {
    
    let values: [String] = ["A", "B", "D", "", "", "E", "", "", "C", "", "F", "", ""]
    let tree: Tree = Tree(values)
    let rootNode: TreeNode? = tree.createTreeNode()
    print("先序遍历")
    tree.preOrderTraverse(rootNode) // 先序遍历: A B D E C F -> A B D # # E # # C # F # #
    print("")
    print("中序遍历")
    tree.inOrderTraverse(rootNode)  // 中序遍历: D B E A C F  -> # D # B # E # A # C # F #
    print("")
    print("后序遍历")
    tree.afterOrderTraverse(rootNode) // 后序遍历: D E B F C A -> # # D # # E B # # # F C A
    print("")
    print("层次遍历")
    tree.levelOrder(rootNode)   // 层次遍历: [["A"], ["B", "C"], ["D", "E", "F"]]  -> A B C D E # F # # # # # #
    print("")
    print("栈实现遍历")
    print(tree.stackToTraversal(rootNode))
    print("App打印二叉树")
    print(tree.levelToOrder(rootNode))
    // print(tree.maxDepth(rootNode)) // 求深度
}
```

## 实战

> Given a binary tree, please design an iOS app to demo it.

首先一个简单的app是mvc架构，所以我们就要想，在View的层面上表示一棵二叉树？我们脑海中浮现树的结构是这样的：

![](/aasets/images/al-tree-18.png)

所以是不是在View的界面上，每个节点弄个UILabel来表示，然后用数学方法计算每个UIlabel对应的位置，从而完美的显示上图的样子？
 这个想法比较简单粗暴，是最容易想到，实现之后又是最直观展示一棵二叉树的，但是它有以下两个问题：

- 每个UILabel的位置计算起来比较麻烦；
- 如果一棵树有很多节点（比如1000个），那么当前界面就会显示不下了，这时候咋办？就算用UIScrollView来处理，整个树也会变得非常不直观，每个节点所对应的UILabel位置计算起来就会更费力。

要处理大量数据，我们就想到了UITableView。假如每一个cell对应一个节点，以及其左、右节点，那么我们就可以清晰的展示一棵树。比如上图这棵树，用UITableView就可以写成这样：

![](/aasets/images/al-tree-19.jpg)

其中"#"表示空节点。明眼人可以看出，我们是按照层级遍历的方式布局整个UITableView。这种做法解决了上面两个问题：

- 无需进行位置计算，UITableView提供复用Cell，效率大幅提高
- 面对很多节点的问题，可以先处理一部分数据，然后用处理infinite scroll的方式来加载剩余数据

接着问题来了，给你一棵二叉树，如何得到它的层级遍历？其实层级遍历就是图的广度优先遍历，而广度优先遍历很自然就会用到队列，所以我们不妨用队列来帮助实现树的层级遍历：

```swift
func levelToOrder(_ root: TreeNode?) -> [[String]] {
        
    var res = [[String]]()
    // 用数组来实现队列
    var queue = [TreeNode]()

    if let root = root {
        queue.append(root)
    }

    while queue.count > 0 {

        let size = queue.count
        var level = [String]()
        for _ in 0 ..< size {
            let node = queue.removeFirst()

            level.append(node.val)
            if let left = node.left {
                queue.append(left)
            }
            if let right = node.right {
                queue.append(right)
            }
        }
        res.append(level)
    }
    return res
}
```

## 总结

到这里为止，我们已经把重要的数据结构都分析了一遍。要知道，这些数据结构都不是单独存在的，我们在解决二叉树的问题时，用到了队列；解决数组的问题，也会用到字典或是栈。在真正面试或是日常Coding中，最低的时间复杂度是首要考虑，接着是优化空间复杂度，其次千万不要忘记考虑特殊情况。在Swift中，用let和var的地方要区分清楚，该不该定义数据为optional，有没有处理nil的情况都是很容易忽略的，希望大家多多练习，融会贯通。

[源码地址](<https://github.com/Jovins/Algorithm>)