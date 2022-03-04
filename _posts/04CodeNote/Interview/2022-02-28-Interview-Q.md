---
title: 遇到的 Interview Quetsions
date: 2022-02-28 11:11:11 -0400
categories: [04CodeNote]
tags: [makefile]
math: true
image:
---

- [Question during interview](#question-during-interview)
  - [2022-2](#2022-2)

---

# Question during interview


## 2022-2

I will give you a tree. I'll give you an input 
- For every input will be in lines every line has two numbers, a and b. 
- So whenever you receive a line like that means that there is node a connected to node b.
- That's how I will describe the three to you. 
- I will guarantee to you that this is a three it's not internal graph.
- I want you to write the system that will answer quesryz of the form. What is the distance if I want to go from node A to node B? 
- Because it's tree you're gonna hear that every node is connected to every other node. 
- so I want you to write efficient code that will help to answer a query just like that. The tree can have 1000s of nodes and the queries that we need to ask maybe like millions or billions of queries, so I want something fast.

the input is an array of pairs 
- input (1,3),(1,4),(1,5),(3,6),(5,8)
- query: distance from 1 to 8.


```java 
class Node {
    int data;
    Node left = null, right = null;
    Node(int data) {
        this.data = data;
    }
}
 
class Main {
    
    public static boolean isNodePresent(Node root, Node node) {
        // base case
        if (root == null) {
            return false;
        }
        // if the node is found, return true
        if (root == node) {
            return true;
        }
        return isNodePresent(root.left, node) || isNodePresent(root.right, node);
    }
 

    public static int findLevel(Node root, Node node, int level) {
        // base case
        if (root == null) return Integer.MIN_VALUE; 

        // return level if the node is found
        if (root == node) return level; 
        // search node in the subtree
        int left = findLevel(root.left, node, level + 1);
        if (left != Integer.MIN_VALUE) return left;  
        return findLevel(root.right, node, level + 1);
    }
 
    // Function to find the lowest common ancestor of given nodes `x` and `y`,
    // where both `x` and `y` are present in the binary tree.
    public static Node findLCA(Node root, Node x, Node y) {
        // base case 1: if the tree is empty
        if (root == null) return null; 
        // base case 2: if either `x` or `y` is found
        if (root == x || root == y) return root; 
 
        // recursively check if `x` or `y` exists in the left subtree
        Node left = findLCA(root.left, x, y); 
        // recursively check if `x` or `y` exists in the right subtree
        Node right = findLCA(root.right, x, y);

        // if `x` is found in one subtree and `y` is found in the other subtree,
        // update lca to the current node
        if (left != null && right != null) return root; 
 
        // if `x` and `y` exist in the left subtree
        if (left != null) return left; 
        // if `x` and `y` exist in the right subtree
        if (right != null) return right; 
 
        return null;
    }
 
    // Function to find the distance between node `x` and node `y` in a
    // given binary tree rooted at `root` node
    public static int findDistance(Node root, Node x, Node y) {
        // `lca` stores the lowest common ancestor of `x` and `y`
        Node lca = null;

        int lev_x = findLevel(root, x, 0);
        int lev_y = findLevel(root, y, 0);

        if(lev_x == 0 or lev_y == 0)
 
        // call LCA procedure only if both `x` and `y` are present in the tree
        if (isNodePresent(root, y) && isNodePresent(root, x)) {
            lca = findLCA(root, x, y);
        }
        else return Integer.MIN_VALUE; 
 
        // return distance of `x` from lca + distance of `y` from lca
        return findLevel(lca, x, 0) + findLevel(lca, y, 0);
 
        /*
            The above statement is equivalent to the following:
 
            return findLevel(root, x, 0) + findLevel(root, y, 0) -
                    2*findLevel(root, lca, 0);
 
            We can avoid calling the `isNodePresent()` function by using
            return values of the `findLevel()` function to check if
            `x` and `y` are present in the tree or not.
        */
    }
 
    public static void main(String[] args) {
        /* Construct the following tree
              1
            /   \
           /     \
          2       3
           \     / \
            4   5   6
               /     \
              7       8
        */ 
        Node root = new Node(1);
        root.left = new Node(2);
        root.right = new Node(3);
        root.left.right = new Node(4);
        root.right.left = new Node(5);
        root.right.right = new Node(6);
        root.right.left.left = new Node(7);
        root.right.right.right = new Node(8); 
        // find the distance between node 7 and node 6
        System.out.print(findDistance(root, root.right.left.left, root.right.right));
    }
}

```















.