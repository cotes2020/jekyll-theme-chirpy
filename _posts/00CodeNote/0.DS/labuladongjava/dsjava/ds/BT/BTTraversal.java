package BT;

import java.util.Stack;

// Binary Tree Traversal

// preorder: V-L-R
// inorder: L-V-R
// postorder: L-R-V

public class BTTraversal {

// 1. preorder V L R

    public void preorder(Node node) {
        if (node != null) {
            System.out.print(node.data);
            preorder(node.left);
            preorder(node.right);
        }
    }


    public void preorderwithStack(Node node) {
        if(node == null) {
            return;
        }
        Stack<Node> s = new Stack<Node>();
        s.add(node);
        while(!s.isEmpty()) {
            node = s.pop();
            System.out.print(node.data + " ");
            if(node.right != null) {
                s.add(node.right);
            }
            if(node.left != null) {
                s.add(node.left);
            }
        }
    }



// 2. inorder 3 4 5 7 10 15 19 20

//        10
//     5     15
//   4   7      19
// 3              20

    public void inorder(Node node) {
        if (node != null) {
            inorder(node.left);
            System.out.print(node.data);
            inorder(node.right);
        }
    }

    public void inorderwithStack(Node node){
        if (node==null) return;
        Stack<Node> s = new Stack<Node>();
        while (true) {
            if (node != null) {
                s.push(node);
                node = node.left;   // to the left end
            }
            else {
                if (s.isEmpty()) {
                    break;
                }
                node = s.pop();
                System.out.print(node);
                node = node.right;
            }
        }
    }


// 3. postorder L - R - V

    public void postorder(Node node) {
        if (node != null) {
            inorder(node.left);
            inorder(node.right);
            System.out.print(node.data);
        }
    }


    public void postorderWithOneStack(Node node) {
        if(node == null) return;
        Stack<Node> s = new Stack<Node>();
        Node current = node;
        Node temp;
        while(current != null || !s.isEmpty()) {
            if(current != null) {
                s.push(current);
                current = current.left;
            }
            else {
                temp = s.peek().right;
                if(temp == null) {
                    temp = s.pop();
                    System.out.print(temp.data + " ");
                    while(!s.isEmpty() && temp == s.peek().right) {
                        temp = s.pop();
                        System.out.print(temp.data+ " ");
                    }
                }
                else {
                    current = temp;
                }
            }

        }
    }

    public void postorderWithTwoStack(Node node) {
        if (node == null) {
            System.out.print("The node is null.");
            return;
        }
        Stack<Node> s1 = new Stack<Node>();
        Stack<Node> s2 = new Stack<Node>();
        s1.push(node);
        while(!s1.isEmpty()) {
            node = s1.pop();
            s2.push(node);
            if(node.left != null) {
                s1.push(node.left);
            }
            if(node.right != null) {
                s1.push(node.right);
            }
        }
        while(!s2.isEmpty()) {
            node = s2.pop();
            System.out.print(node.data + " ");

        }
    }


    public static void main(String args[]){
        BinaryTree bt = new BinaryTree();
        Node root = null;
        root = bt.addNode(10, root);
        root = bt.addNode(15, root);
        root = bt.addNode(0, root);
        root = bt.addNode(5, root);
        root = bt.addNode(-1, root);
        root = bt.addNode(2, root);
        root = bt.addNode(6, root);

        BTTraversal traverse = new BTTraversal();
        System.out.println("preorderwithStack:");
        traverse.preorderwithStack(root);
        System.out.println();
        System.out.println("postorderWithTwoStack:");
        traverse.postorderWithTwoStack(root);
        System.out.println();
        System.out.println("postorderWithOneStack:");
        traverse.postorderWithOneStack(root);
    }

}
