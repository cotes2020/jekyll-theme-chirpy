  
  
  
- [Binary Tree](#binary-tree )
  - [Tree](#tree )
  - [樹的元素](#樹的元素 )
  - [定義](#定義 )
  - [程式碼](#程式碼 )
  - [集合關係](#集合關係 )
- [Intro](#intro )
  - [程式碼](#程式碼-1 )
  - [學習Binary Tree](#學習binary-tree )
- [Introduction](#introduction )
  - [Why Trees?](#why-trees )
  - [First Simple Tree](#first-simple-tree )
- [Properties](#properties )
- [Types of BT](#types-of-bt )
  - [Full Binary Tree](#full-binary-tree )
  - [Complete Binary Tree](#complete-binary-tree )
  - [Perfect Binary Tree](#perfect-binary-tree )
  - [Balanced Binary Tree](#balanced-binary-tree )
  - [degenerate / pathological tree](#degenerate--pathological-tree )
- [Binary Tree Traversal](#binary-tree-traversal )
  - [preorder traversal `N-L-R`](#preorder-traversal-n-l-r )
  - [inorder traversal `L-N-R` Binary search tree](#inorder-traversal-l-n-r-binary-search-tree )
  - [postorder traversal `L-R-N`](#postorder-traversal-l-r-n )
- [Binary Search Tree code (depth-first)](#binary-search-tree-code-depth-first )
  - [Binary Search Tree Search](#binary-search-tree-search )
  - [Binary Search Tree Insertion (Iterative method)](#binary-search-tree-insertion-iterative-method )
  - [Iterative `preOrder Traversal V-L-R` of Binary Tree](#iterative-preorder-traversal-v-l-r-of-binary-tree )
  - [Iterative `Inorder Traversal L-V-R` of Binary Tree](#iterative-inorder-traversal-l-v-r-of-binary-tree )
  - [Iterative `Postorder Traversal L-R-V` of Binary Tree](#iterative-postorder-traversal-l-r-v-of-binary-tree )
  - [Check Same Binary Tree](#check-same-binary-tree )
  - [Size Of Binary Tree `node numebr`](#size-of-binary-tree-node-numebr )
  - [Height Of Binary Tree](#height-of-binary-tree )
  - [Root To Leaf Sum Binary Tree](#root-to-leaf-sum-binary-tree )
  - [Check if Binary Tree is Binary Search Tree](#check-if-binary-tree-is-binary-search-tree )
- [Binary Search Tree code (Level Order -> queue)](#binary-search-tree-code-level-order---queue )
  - [Level Order Traversal (in one line)](#level-order-traversal-in-one-line )
  - [Level by Level Printing (in different line)](#level-by-level-printing-in-different-line )
  - [`Reverse` level order traversal binary tree](#reverse-level-order-traversal-binary-tree )
- [Handshaking Lemma and Interesting Tree Properties ??](#handshaking-lemma-and-interesting-tree-properties- )
- [Enumeration of Binary Trees ???](#enumeration-of-binary-trees- )
- [Insertion in a Binary Tree](#insertion-in-a-binary-tree )
  
- ref
  - [1](https://www.geeksforgeeks.org/binary-tree-data-structure/ )
  
---
  
#  Binary Tree
  
  
若熟悉`Linked List`(連結串列)將會更容易理解樹：
- `Linked list`是一維的線性結構(不是往前、就是往後)
- 樹(與Graph)則推廣成多維的結構。
  
linkedlist
  
![f1](https://i.imgur.com/mLBAp4m.png )
  
- A、B、C、D稱為node(節點)，用以代表資料(data)、狀態(state)。
- 連結各個node之間的連結(link)稱為edge，可能是單方向，或者雙向。
  
  
  
##  Tree
  
Tree(樹)是用以描述具有`階層結構(hierarchical structure)`的問題的首選，
- 階層結構意味著明確的先後次序，
- 例如，若要印出ABC三個字母的所有排列組合(permutation)
  
  
而樹的最根本特徵就是：
- 在樹的結構裡，`只有一個root(樹根)`，`並且不存在cycle`。
- 此特徵將衍生出另外兩項等價的性質：
  - 在樹中若要從root尋找特定node，一定只存在一條路徑(path)。
  - 每個node只會有一個parent。
  
  
  
##  樹的元素
  
  
  
針對node / vertex：
  
![f9](https://i.imgur.com/EVshcEh.png )
  
- `degree`(分歧度)：
  - 一個node擁有的subtree(子樹)的個數。
  - A的degree為3，F的degree為2，N的degree為0。
- `root`(樹根)：
  - 樹中最上層的node，也是唯一一個其parent為NULL的node。
  - A即為root。
- `external node/leaf`：
  - 沒有child/subtree的node。
  - G、H、J、K、L、M、N皆為leaf node。
- `internal node`：
  - 至少有一個child的node。
  - A、B、C、D、E、F、I皆為internal node。
  
  
- `parent <--> child`：
  - 以pointer說明，被指向者(pointed)為child，指向者(point to)為parent。
- `siblings：擁有相同parent的node們，互相稱兄道弟。`
  - B、C、D共同的parent為A，B、C、D即為彼此的sibling。
- `descendant`(子嗣)：
  - 站在A，所有能夠以「parent指向child」的方式找到的node，皆稱為A的descendant，因此整棵樹除了A以外皆為A的descendant。
  - 在F，能夠以「parent指向child」找到的node有L、M，則稱L、M為F的descendant。
- `ancestor`(祖先)：
  - 圖四中，站在K，所有能夠以「尋找parent」的方式找到的node，皆稱為K的ancestor，因此，E、B、A皆為K的ancestor。
- `path`(路徑)：
  - 由descendant與ancestor關係連結成的`edge`，例如A-B-E-K、A-C-F-N。
- `level`： root-2-3-4
  - 定義root的level為1，其餘node的level為其parent的level加一。
- `height of node`：
  - 某一node與其最長path上之descendant leaf node之間的edge數。
  - 例如，F的height為1，D的height為2，leaf node的height為0。
- `height of tree`：樹的height即為root的height。
  - 樹的height為A的height，等於3。
- `depth`：
  - 某一node與root之間的edge數。
  - 例如，F的depth為2，L的depth為3。
  
在樹中的traversal(尋訪)之時間複雜度(time complexity)會與height(樹高)有關。
  
---
  
##  定義
  
  
以下列出兩種互相等價的Tree(樹)的定義：
  
A. Tree(樹)是由一個或多個節點所組成的有限集合，並且滿足：
- 存在且只有一個稱為root(樹根)的節點；
- 其餘的節點可以分割成任意正整數個(包含零個)互斥(disjoint)的集合：T1、...、Tn，其中每一個集合也都滿足樹的定義，這些集合又稱為這棵樹的subtree(子樹)。
- B. Tree(樹)是由一個或多個nodes/vertices以及edge所組成，而且沒有cycle的集合(set)。
  
Forest(樹林)
- 由n≥0棵彼此互斥(disjoint)的Tree(樹)所形成的集合(Set)，即稱為Forest(樹林)。
- Forest(樹林)由多個Tree(樹)所組成，可以用來表示互斥集合(disjoint set)。
  
  
  
---
  
##  程式碼
  
以程式碼實作一棵樹，常用的手法為：先以`class TreeNode`(或是struct)定義出每顆node能夠指向多少subtree、攜帶哪些資料形態，再以另一個class Tree表示整棵樹，並以root作為樹的存取點：
  
  
---
  
##  集合關係
  
  
Tree(樹)位居承先啟後的重要戰略位置，資料結構之集合關係圖：
  
![f11](https://i.imgur.com/xqAyRXN.png )
  
  
本篇介紹的Tree(樹)並沒有限制child/ subtree的個數
- 理論上可以有多到超過記憶體空間的child node。
  - ![f1-1](https://i.imgur.com/wt3t5d0.png )
- 然而在實務上，較常使用每個node至多只有兩個child的樹，為`Binary Tree(二元樹)`。
  - ![f2](https://i.imgur.com/ngahlhP.png )
  - 樹上的每一個node之degree皆為2
  - 並稱兩個child pointer為left child和right child。
- 從Binary Tree再增加「鍵值(Key)大小規則」，即`Binary Search Tree(BST，二元搜尋樹)`。
- 以BST為基礎，在每個node上添加顏色(紅與黑)用以平衡樹的height，以減短搜尋時間，這種樹稱為`Red Black Tree(RBT，紅黑樹)`。
- 常見的平衡樹(balanced tree)還有：AVL tree、2-3-4 tree、Splay tree等等，請參考：Wikipedia：Self-balancing binary search tree
- 另一個方向，若打破「不能存在cycle」的限制，則從Tree推廣至圖(Graph)。
  
  
---
  
#  Intro
  
  
  
##  程式碼
  
  
  
```java
public class BinarySearchTree {
  
    // 根节点
    public static TreeNode root;
  
    public BinarySearchTree() {
        this.root = null;
    }
}
  
```
  
---
  
---
  
##  學習Binary Tree
  
  
- Binary Search Tree(BST)：
  - 在某些資料經常要增加、刪除的應用中，BST常用來做搜尋，
  - 例如許多程式語言的Library中的map和set。
- Binary Space Partition：
  - 應用於幾乎所有的3D電玩遊戲以決定哪些物件需要rendered。
- Binary Tries：
  - 應用於大多數high-bandwidth router(高頻寬路由器)以儲存router-tables。
- Heaps：
  - 用以實現高效率的priority queues(優先權佇列)，許多作業系統用來安排工作程序。請參考：Priority Queue：Binary Heap。
- Huffman Coding Tree：
  - 例如.jpeg、.mp3等壓縮技術皆使用Huffman編碼。(在一顆20MB的硬碟要價新台幣一萬元的時代，壓縮技術就是救世主。)
  
  
  
---
  
#  Introduction
  
  
##  Why Trees?
  
  
1. to store information that naturally forms a hierarchy. For example, the file system on a computer:
2. Trees (with some ordering e.g., BST) provide moderate access/search (quicker than Linked List and slower than arrays).
3. Trees provide moderate insertion/deletion (quicker than Arrays and slower than Unordered Linked Lists).
4. Like Linked Lists and unlike Arrays, Trees don’t have an upper limit on number of nodes as nodes are linked using pointers.
  
  
Main applications of trees include:
1. Manipulate hierarchical data.
2. Make information easy to search (see tree traversal).
3. Manipulate sorted lists of data.
4. As a workflow for compositing digital images for visual effects.
5. Router algorithms
6. Form of a multi-stage decision-making (see business chess).
  
  
##  First Simple Tree
  
Binary Tree: A tree whose elements have at most 2 children is called a binary tree. Since each element in a binary tree can have only 2 children, we typically name them the left and right child.
  
Summary: Tree is a hierarchical data structure. Main uses of trees include maintaining hierarchical data, providing moderate access and insert/delete operations. Binary trees are special cases of tree where every node has at most two children.
  
```java
/* Class containing left and right child of current node and key value*/
  
class Node
{
	int key;
	Node left, right;
  
	public Node(int item)
	{
		key = item;
		left = right = null;
	}
}
  
  
class BinaryTree
{
	// Root of Binary Tree
	Node root;
  
	// Constructors
	BinaryTree(int key)
	{
		root = new Node(key);
	}
  
	BinaryTree()
	{
		root = null;
	}
  
  
  // create a simple tree with 4 nodes
	public static void main(String[] args) {
  
		BinaryTree tree = new BinaryTree();
  
		/*create root*/
		tree.root = new Node(1);
  
		/* following is the tree after above statement
			1
			/ \
		null null	 */
  
		tree.root.left = new Node(2);
		tree.root.right = new Node(3);
  
		/* 2 and 3 become left and right children of 1
			1
			/ \
			2	 3
		/ \ / \
		null null null null */
  
  
		tree.root.left.left = new Node(4);
		/* 4 becomes left child of 2
					1
				/	 \
			2		 3
			/ \	 / \
			4 null null null
		/ \
		null null
		*/
	}
}
```
  
---
  
  
#  Properties
  
  
1) The maximum number of nodes at level ‘l’: 2^l.
   - level is number of nodes on path from root to the node (including root and node).
   - Level of root is 0, number of nodes = 2^0 = 1
  
  
2) Maximum number of nodes in a binary tree of height ‘h’: 2^h – 1.
   - height of a tree is maximum number of nodes on root to leaf path.
   - Height of a tree with single node is considered as 1.
   - In some books, height of the root is considered as 0. In this convention, the above formula becomes 2h+1 – 1
  
  
3) In a Binary Tree with N nodes, minimum possible height or minimum number of levels: Log2(N+1)
   - This can be directly derived from point 2 above.
   - If we consider the height of a leaf node is considered as 0, then above formula for minimum possible height becomes   ? Log2(N+1) ? – 1
  
  
4) A Binary Tree with L leaves has at least: Log2L ? + 1   levels
A Binary tree has maximum number of leaves (and minimum number of levels) when all levels are fully filled. Let all leaves be at level l, then below is true for number of leaves L.
  
   L   <=  2l-1  [From Point 1]
   l =   ? Log2L ? + 1
   where l is the minimum number of levels.
  
  
5) In Binary tree where every node has 0 or 2 children, number of leaf nodes is always one more than nodes with two children.
  
---
  
#  Types of BT
  
  
![Screen Shot 2021-09-15 at 8.28.09 PM](https://i.imgur.com/B41rB2I.png )
  
---
  
##  Full Binary Tree
  
- number of leaf nodes is the number of internal nodes plus 1:
  - Number of leaf nodes = Number of internal nodes + 1
  
```
               18
           /       \  
         15         30  
        /  \        /  \
      40    50    100   40
  
             18
           /    \   
         15     20    
        /  \       
      40    50   
    /   \
   30   50
  
               18
            /     \  
          40       30  
                   /  \
                 100   40
```
  
---
  
  
##  Complete Binary Tree
  
- node按照Full Binary Tree的次序排列(由上至下，由左至右)
  - 樹共有10個node，
  - 且這十個node正好填滿Full Binary Tree的前十個位置，
  - 則此樹為Complete Binary Tree。
- if all the levels are completely filled except possibly the last level and the last level has all keys as left as possible
  
  
```
               18
           /       \  
         15         30  
        /  \        /  \
      40    50    100   40
  
  
               18
           /       \  
         15         30  
        /  \        /  \
      40    50    100   40
     /  \   /
    8   7  9
```
  
圖四：這是一棵Complete Binary Tree。
![f4](https://i.imgur.com/K8LQjx1.png )
  
圖五：這不是一棵Complete Binary Tree。
![f5](https://i.imgur.com/ed7613u.png )
  
---
  
  
##  Perfect Binary Tree
  
- 所有internal node都有兩個subtree(child pointer)；
- 所有leaf node具有相同的level(或相同的height)。
- 由以上性質能夠推論出：
  - 若leaf node之level為n，整棵樹共有 2^n − 1個node。
- 並且，每個node與其child有以下關係：
  - 第i個node的left child之index為 2i；
  - 第i個node的right child之index為 2i+1；
  - 除了root之parent為NULL之外，第i個node的parent之index為 ⌊i/2⌋ 。
  
  
```
               18
           /       \  
         15         30  
        /  \        /  \
      40    50    100   40
  
  
               18
           /       \  
         15         30  
```
  
---
  
##  Balanced Binary Tree
  
  
- A binary tree is balanced if the height of the tree is O(Log n) where n is the number of nodes.
- For Example, the AVL tree maintains `O(Log n)` height by making sure that the difference between the heights of the left and right subtrees is almost 1.
- Red-Black trees maintain O(Log n) height by making sure that the number of Black nodes on every root to leaf paths is the same and there are no adjacent red nodes.
- performance-wise good: woest: O(log n) time for search, insert and delete.
  
  
---
  
##  degenerate / pathological tree
  
- A Tree where every internal node has one child. Such trees are performance-wise same as linked list.
  
```
      10
      /
    20
     \
     30
      \
      40     
```
  
---
  
#  Binary Tree Traversal
  
  
---
  
##  preorder traversal `N-L-R`
  
  
![Screen Shot 2020-07-23 at 01.29.46](https://i.imgur.com/bDWOafI.png )
  
10 -> 7 -> 6 -> 1 -> 8 -> 9 -> 11 -> 20 -> 14 -> 22
  
```
             10
           /     \  
          7       11  
        /  \        \
       6    8        20
      /      \      /  \    
     1        9    14  22
```
  
  
---
  
##  inorder traversal `L-N-R` Binary search tree
  
  
![Screen Shot 2020-07-23 at 01.33.45](https://i.imgur.com/q4wCoVb.png )
  
1 -> 6 -> 7 -> 8 -> 9 -> 10 -> 11 -> 14 -> 20 > 22
  
```
             10
           /     \  
          7       11  
        /  \        \
       6    8        20
      /      \      /  \    
     1        9    14  22
```
  
---
  
##  postorder traversal `L-R-N`
  
  
1 -> 6 -> 9 -> 8 -> 7 -> 14 -> 22 -> 20 -> 11 -> 10
  
```
             10
           /     \  
          7       11  
        /  \        \
       6    8        20
      /      \      /  \    
     1        9    14  22
```
  
  
---
  
#  Binary Search Tree code (depth-first)
  
  
##  Binary Search Tree Search
  
  
```java
  
public class BSTSearch {
  
    public Node search(Node root, int key) {
        if (root.data == null) { return null;}
        if (root.data == key) { return root;}
        else if (root.data > key) {
            search(root.left, key);
        }
        else {
            search(root.right, key);
        }
    }
  
    public static void main(String args[]){
  
        BinaryTree bt = new BinaryTree();
        Node root = null;
        root = bt.addNode(10, root);
        root = bt.addNode(20, root);
        root = bt.addNode(-10, root);
        root = bt.addNode(15, root);
        root = bt.addNode(0, root);
        root = bt.addNode(21, root);
        root = bt.addNode(-1, root);
  
        BSTSearch bstSearch = new BSTSearch();
  
        Node result = bstSearch.search(root, 21);
        assert result.data == 21;
  
        result = bstSearch.search(root, -1);
        assert result.data == 21;
  
        result = bstSearch.search(root, 11);
        assert result == null;
    }
}
  
```
  
---
  
##  Binary Search Tree Insertion (Iterative method)
  
  
worst: O(n)
  
```Java
  
public class BinaryTree{
  
    public Node insert(Node root, int data){
        Node node = new Node(data);
        if (root == null) {
            return node;
        }        
        Node parent = null;
        Node current = root;
        while (root != null) {
            parent = root;
            if (root.data < data) {
                root = root.right;
            }
            else {
                root = root.left;
            }
        }
        if (parent.data < data) {
            parent.right = data;
        }
        else {
            parent.left = data;
        }
        return parent;
  
        // if (root.data < key && root.next == null) {
        //     root.left.data == key
        // }
        // if (root.data > key && root.next == null) {
        //     root.left.data == key
        // }
        // if (root.next != null) {
        //     root = root.next;
        //     bstInsert(Node root.next, int key);
        // }
    }
}
```
  
---
  
  
  
##  Iterative `preOrder Traversal V-L-R` of Binary Tree
  
  
![Screen Shot 2020-07-24 at 14.39.24](https://i.imgur.com/8V6IjcI.png )
  
```java
public class TreeTraversals {
  
    public void preOrder(Node root){
        if(root == null){return;}
        System.out.print(root.data + " ");
        preOrder(root.left);
        preOrder(root.right);
    }
  
    public void preOrderItr(Node root){
        Deque<Node> stack = new LinkedList<Node>();
        stack.addFirst(root);
        while(!stack.isEmpty()){
            root = stack.pollFirst();
            System.out.print(root.data + " ");
  
            if(root.right != null){
                stack.addFirst(root.right);
            }
            if(root.left!= null){
                stack.addFirst(root.left);
            }
        }
    }
}
```
  
---
  
  
##  Iterative `Inorder Traversal L-V-R` of Binary Tree
  
  
O(n)
  
![Screen Shot 2020-07-25 at 00.40.01](https://i.imgur.com/ReKolwj.png )
  
```java
public class TreeTraversals {
  
    public void inOrder(Node root){
        if(root == null){return;}
        inOrder(root.left);
        System.out.print(root.data + " ");
        inOrder(root.right);
    }
  
    public void inorderItr(Node root){
        if (root==null) return;
        Stack<Node> s = new Stack<Node>();
        while (true) {
            if (root != null) {
                s.push(root);
                root=root.left;
            }
            else {
                if (s.isEmpty()) {
                    break;
                }
                root=s.pop();
                System.out.print(root);
                root = root.right;
            }
        }
    }
  
  
  
    public void inorderItr(Node root){
        Stack stack = new Stack();
        stack.push(root);
  
        if (root.left == null) {
            return stack.pop(root);
        }
        if (root.left != null) {
            return inorderItr(root.left);
            stack.pop(root)
            stack.pop(root.right)
        }
    }
  
  
  
  
    // public void inorderItr(Node root){
    //     Deque<Node> stack = new LinkedList<Node>();
    //     Node node = root;
    //     while(true){
    //         if(node != null){
    //             stack.addFirst(node);
    //             node = node.left;
    //         }
    //         else{
    //             if(stack.isEmpty()){
    //                 break;
    //             }
    //             node = stack.pollFirst();
    //             System.out.println(node.data);
    //             node = node.right;
    //         }
    //     }
    // }
}
```
  
---
  
##  Iterative `Postorder Traversal L-R-V` of Binary Tree
  
  
![Screen Shot 2020-07-24 at 13.57.34](https://i.imgur.com/EpqGSIW.png )
  
![Screen Shot 2020-07-24 at 14.00.03](https://i.imgur.com/fNYgO5h.png )
  
  
```java
public class TreeTraversals {
  
    public void postOrder(Node root){
        if(root == null){return;}
        postOrder(root.left);
        postOrder(root.right);
        System.out.print(root.data + " ");
    }
  
    // 1.
    public void iterPostOrder(Node root){
        if (root==null) {return;}
        Stack<Node> s1 = new Stack<Node>();
        Stack<Node> s2 = new Stack<Node>();
        s1.push(root);
        while (!s1.isEmpty()){
            root=s1.pop();
            s2.push(root);
            if (root.left != null) {
                s1.push(root.left);
            }
            if (root.right != null) {
                s1.push(root.right);
            }
        }
        while (!s2.isEmpty()){
            root=s2.pop();
            System.out.println(root.data)
        }
    }
  
    // 2.
    public void postOrderItr(Node root){
        Deque<Node> stack1 = new LinkedList<Node>();
        Deque<Node> stack2 = new LinkedList<Node>();
        stack1.addFirst(root);
        while(!stack1.isEmpty()){
            root = stack1.pollFirst();
            if(root.left != null){
                stack1.addFirst(root.left);
            }
            if(root.right != null){
                stack1.addFirst(root.right);
            }
            stack2.addFirst(root);
        }
        while(!stack2.isEmpty()){
            System.out.print(stack2.pollFirst().data + " ");
        }
    }
  
    public void postOrderItrOneStack(Node root){
        Node current = root;
        Deque<Node> stack = new LinkedList<>();
        while(current != null || !stack.isEmpty()){
            if(current != null){
                stack.addFirst(current);
                current = current.left;
            }else{
                Node temp = stack.peek().right;
                if (temp == null) {
                    temp = stack.poll();
                    System.out.print(temp.data + " ");
                    while (!stack.isEmpty() && temp == stack.peek().right) {
                        temp = stack.poll();
                        System.out.print(temp.data + " ");
                    }
                } else {
                    current = temp;
                }
            }
        }
    }
}
```
  
  
  
---
  
##  Check Same Binary Tree
  
  
![Screen Shot 2020-07-24 at 11.27.26](https://i.imgur.com/dcOA8T4.png )
time O(n)
  
```java
public class SameTree {
  
    public boolean sameTree(Node root1, Node root2){
        if(root1 == null && root2 == null){
            return true;
        }
        if(root1 == null || root2 == null){
            return false;
        }
        return root1.data == root2.data &&
                sameTree(root1.left, root2.left) &&
                sameTree(root1.right, root2.right);
    }
  
    public static void main(String args[]){
        BinaryTree bt = new BinaryTree();
        Node root1 = null;
        root1 = bt.addNode(10, root1);
        root1 = bt.addNode(20, root1);
        root1 = bt.addNode(15, root1);
        root1 = bt.addNode(2, root1);
  
        Node root2 = null;
        root2 = bt.addNode(10, root2);
        root2 = bt.addNode(20, root2);
        root2 = bt.addNode(15, root2);
        root2 = bt.addNode(2, root2);
  
        SameTree st = new SameTree();
        assert st.sameTree(root1, root2);
   }
}
```
  
  
---
  
##  Size Of Binary Tree `node numebr`
  
  
![Screen Shot 2020-07-24 at 13.30.14](https://i.imgur.com/mbLl3ws.png )
  
```java
public class SizeOfBinaryTree {
  
    public int size(Node root){
        if(root == null){
            return 0;
        }
        return size(root.left) + size(root.right) + 1;
    }
  
    public static void main(String args[]){
  
        BinaryTree bt = new BinaryTree();
        Node head = null;
        head = bt.addNode(10, head);
        head = bt.addNode(15, head);
        head = bt.addNode(5, head);
        head = bt.addNode(7, head);
        head = bt.addNode(19, head);
        head = bt.addNode(20, head);
        head = bt.addNode(-1, head);
  
        SizeOfBinaryTree sbt = new SizeOfBinaryTree();
        System.out.println(sbt.size(head));
    }
}
```
  
---
  
  
##  Height Of Binary Tree
  
  
time & space: O(n)
  
![Screen Shot 2020-07-24 at 13.29.49](https://i.imgur.com/WrgJy8X.png )
  
```java
public class SameTree {
  
    public int height(Node root){
        if(root == null){
            return 0;
        }
        int leftHeight  = height(root.left);
        int rightHeight = height(root.right);
        return Math.max(leftHeight, rightHeight) + 1;
    }
  
    // public int height(Node root){
    //     if(root.left == null && root.right == null) {
    //         return 1;
    //     }
    //     else {
    //         return 0;
    //     }
    //     return 1 + height(root.right) + height(root.left)
    // }
}
  
```
  
---
  
##  Root To Leaf Sum Binary Tree
  
  
![Screen Shot 2020-07-24 at 13.29.20](https://i.imgur.com/ezvZIHd.png )
  
```java
public class Root {
  
    public boolean RootToLeaf(Node root, int sum, list<Integer result>) {
  
        if (root == null) {return false;}
        if (root.left == null && root.right == null) {
            if (root.data == sum) {
                result.add(root.data);
                return true;
            }
            else {return false;}
        }
  
        if (RootToLeaf(Node root.left, int sum-root.data, list<Integer result>)) {
            result.add(root.data);
            return true;
        }
        else {return false;}
  
        if (RootToLeaf(Node root.right, int sum-root.data, list<Integer result>)) {
            result.add(root.data);
            return true;
        }
        else {return false;}     
        return false;   
    }
}
```
  
---
  
##  Check if Binary Tree is Binary Search Tree
  
  
![Screen Shot 2020-07-24 at 13.34.44](https://i.imgur.com/8uaDJ2C.png )
  
```java
public class Root {
  
    public boolean isBST(Node root, int min, int max) {
        if (root == null) {return true;}
        if (root.data < min || root.data > max) {return false;}
        return isBST(root.left, min, root.data) && isBST(root, root.data, max)
    }
}
```
  
---
  
#  Binary Search Tree code (Level Order -> queue)
  
  
![Screen Shot 2020-07-24 at 13.44.59](https://i.imgur.com/f5M3oF1.png )
  
time: O(n)
space: the size of the tree, O(n)
  
---
  
##  Level Order Traversal (in one line)
  
  
![Screen Shot 2020-07-24 at 13.53.38](https://i.imgur.com/nd0fEXX.png )
  
```java
public class LevelOrderTraversal {
  
    public void levelOrder(Node root){
  
        if(root == null){
            System.out.println("Please enter a valid tree!");
            return;
        }
  
        Queue<Node> queue = new LinkedList<Node>();
        queue.offer(root);
  
        while(queue.size() > 0){
            root = queue.poll();
            System.out.print(root.data + " ");
  
            if(root.left != null){
                queue.add(root.left);
            }
            if(root.right != null){
                queue.add(root.right);
            }
        }
    }
  
    public static void main(String args[]){
        LevelOrderTraversal loi = new LevelOrderTraversal();
        BinaryTree bt = new BinaryTree();
        Node head = null;
        head = bt.addNode(10, head);
        head = bt.addNode(15, head);
        head = bt.addNode(5, head);
        head = bt.addNode(7, head);
        head = bt.addNode(19, head);
        head = bt.addNode(20, head);
        head = bt.addNode(-1, head);
        loi.levelOrder(head);
    }
}
```
  
---
  
##  Level by Level Printing (in different line)
  
  
1. use 2 quesues
  
![Screen Shot 2020-07-25 at 00.49.14](https://i.imgur.com/eeISHog.png )
  
```java
  
void levelByLevelTwoQueue(Node root) {
    if (root==null) return;
    Queue<Node> q1 = new Queue<Node>();
    Queue<Node> q2 = new Queue<Node>();
    q1.add(root);
    while ( !q1.isEmpty() || !q2.isEmpty() ) {
        while (!q1.isEmpty()) {
            root=q1.pull();
            System.out.print(root);
            if (root.left != null) {q2.add(root.left);}
            if (root.right != null) {q2.add(root.right);}
        }
        System.out.println();
        while (!q2.isEmpty()) {
            root=q2.pull();
            System.out.print(root);
            if (root.left != null) {q1.add(root.left);}
            if (root.right != null) {q1.add(root.right);}
        }
        System.out.println();
    }
}
```
  
2. use 1 queue
  
![Screen Shot 2020-07-25 at 00.52.36](https://i.imgur.com/YvovZpO.png )
  
```java
void levelByLevelOneQueueUsingDelimiter(Node root) {
    if (root==null) return;
    Queue<Node> q = new Queue<Node>();
    q.add(root);
    while (!q.isEmpty()) {
        root=q.poll();
        if (root==null) {
            System.out.println();
            break
        }
        if (root != null) {
            System.out.print(root.data + " ");
            if (root.left != null) {
                q.add(root.left);
            }
            if (root.left != null) {
                q.add(root.left);
            }
        }
        q.add(null);
    }
}
  
public void levelByLevelOneQueueUsingDelimiter(Node root) {
    if (root == null) {
        return;
    }
    Queue<Node> q = new LinkedList<Node>();
    q.offer(root);
    q.offer(null);
    while (!q.isEmpty()) {
        root = q.poll();
        if (root != null) {
            System.out.print(root.data + " ");
            if (root.left != null) {
                q.offer(root.left);
            }
            if (root.right != null) {
                q.offer(root.right);
            }
        } else {
            if (!q.isEmpty()) {
                System.out.println();
                q.offer(null);
            }
        }
    }
}
```
  
3. 1 queue & 2 conter
  
![Screen Shot 2020-07-25 at 01.41.49](https://i.imgur.com/kcl3RcP.png )
  
![Screen Shot 2020-07-25 at 00.57.11](https://i.imgur.com/pl4FPNv.png )
  
```java
public void levelByLevelOneQueueUsingCount(Node root) {
    if (root == null) return;
  
    Queue<Node> q = new LinkedList<Node>();
    int levelCount = 1;
    int currentCount = 0;
    int current = 0;
  
    q.add(root);
  
    while (!q.isEmpty()) {
        while (levelCount > 0) {
            current = q.poll();
            if (current.left != null) {
                q.add(root.left);
                currentCount++;
            }
            if (current.right != null) {
                q.add(root.right);
                currentCount++;
            }
            System.out.print(current.data + " ");
            levelCount--;
        }
        System.out.println();
        levelCount = currentCount;
        currentCount = 0;
    }
  
}
```
  
---
  
##  `Reverse` level order traversal binary tree
  
  
![Screen Shot 2020-07-25 at 01.55.58](https://i.imgur.com/UJFL9fn.png )
  
```java
public void reverseLevelOrderTraversal(Node root) {
    if (root=null) return;
    Stack<Node> s = new Stack<Node>();
    Queue<Node> q = new Queue<Node>();
    q.add(root);
    while(!q.isEmpty()){
        current = q.poll();
        if (current.right != null) {
            q.add(root.right);
        }
        if (current.left != null) {
            q.add(root.left);
        }
        s.push(current)
    }
    while(!s.isEmpty()){
        System.out.print(s.pop().data + " ");
    }
}
```
  
---
  
  
#  Handshaking Lemma and Interesting Tree Properties ??
  
  
---
  
  
#  Enumeration of Binary Trees ???
  
  
A Binary Tree is `labeled` if every node is assigned a label
a Binary Tree is `unlabeled` if nodes are not assigned any label.
  
```
Below two are considered same unlabeled trees
    o                 o
  /   \             /   \
 o     o           o     o
  
Below two are considered different labeled trees
    A                C
  /   \             /  \
 B     C           A    B
```
  
How many different Unlabeled Binary Trees can be there with n nodes?
  
```
For n  = 1, there is only one tree
   o
  
For n  = 2, there are two possible trees
   o      o
  /        \  
 o          o
  
For n  = 3, there are five possible trees
    o      o           o         o      o
   /        \         /  \      /         \
  o          o       o    o     o          o
 /            \                  \        /
o              o                  o      o
```
  
  
The idea is to `consider all possible pair of counts for nodes in left and right subtrees` and multiply the counts for a particular pair. Finally add results of all pairs.
  
```
For example, let T(n) be count for n nodes.
T(0) = 1  [There is only 1 empty tree]
T(1) = 1
T(2) = 2
  
T(3) =  T(0)*T(2) + T(1)*T(1) + T(2)*T(0) = 1*2 + 1*1 + 2*1 = 5
  
T(4) =  T(0)*T(3) + T(1)*T(2) + T(2)*T(1) + T(3)*T(0)
     =  1*5 + 1*2 + 2*1 + 5*1
     =  14
```
  
The above pattern basically represents n’th Catalan Numbers. First few catalan numbers are 1 1 2 5 14 42 132 429 1430 4862,…
T(n)=\sum_{i=1}^{n}T(i-1)T(n-i)=\sum_{i=0}^{n-1}T(i)T(n-i-1)=C_n
Here,
T(i-1) represents number of nodes on the left-sub-tree
T(n−i-1) represents number of nodes on the right-sub-tree
  
n’th Catalan Number can also be evaluated using direct formula.
  
  
  
   T(n) = (2n)! / (n+1)!n!
Number of Binary Search Trees (BST) with n nodes is also same as number of unlabeled trees. The reason for this is simple, in BST also we can make any key as root, If root is i’th key in sorted order, then i-1 keys can go on one side and (n-i) keys can go on other side.
  
How many labeled Binary Trees can be there with n nodes?
To count labeled trees, we can use above count for unlabeled trees. The idea is simple, every unlabeled tree with n nodes can create n! different labeled trees by assigning different permutations of labels to all nodes.
  
Therefore,
  
Number of Labeled Tees = (Number of unlabeled trees) * n!
                       = [(2n)! / (n+1)!n!]  × n!
For example for n = 3, there are 5 * 3! = 5*6 = 30 different labeled trees
  
  
---
  
#  Insertion in a Binary Tree
  
  
![binary-tree-insertion](https://i.imgur.com/xjtBYaX.png )
  
```java
import java.util.LinkedList;
import java.util.Queue;
  
public class GFG {
  
	/* A binary tree node has key, pointer to
	left child and a pointer to right child */
	static class Node {
		int key;
		Node left, right;
  
		// constructor
		Node(int key) {
			this.key = key;
			left = null;
			right = null;
		}
  }
  
	static Node root;
	static Node temp = root;
  
	/* Inorder traversal of a binary tree*/
	static void inorder(Node temp) {
		if (temp == null)
			return;
		inorder(temp.left);
		System.out.print(temp.key+" ");
		inorder(temp.right);
	}
  
	/*function to insert element in binary tree */
	static void insert(Node temp, int key) {
		Queue<Node> q = new LinkedList<Node>();
		q.add(temp);
  
		// Do level order traversal until we find
		// an empty place.
		while (!q.isEmpty()) {
			temp = q.peek();
			q.remove();
  
			if (temp.left == null) {
				temp.left = new Node(key);
				break;
			} else
				q.add(temp.left);
  
			if (temp.right == null) {
				temp.right = new Node(key);
				break;
			} else
				q.add(temp.right);
		}
	}
  
	// Driver code
	public static void main(String args[])
	{
		root = new Node(10);
		root.left = new Node(11);
		root.left.left = new Node(7);
		root.right = new Node(9);
		root.right.left = new Node(15);
		root.right.right = new Node(8);
  
		System.out.print( "Inorder traversal before insertion:");
		inorder(root);
  
		int key = 12;
		insert(root, key);
  
		System.out.print("\nInorder traversal after insertion:");
		inorder(root);
	}
}
```
  
  
  
  
---
  
Deletion in a Binary Tree
BFS vs DFS for Binary Tree
Binary Tree (Array implementation)
AVL with duplicate keys
Applications of tree data structure
Applications of Minimum Spanning Tree Problem
Continuous Tree
Foldable Binary Trees
Expression Tree
Evaluation of Expression Tree
Symmetric Tree (Mirror Image of itself)
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
.
  