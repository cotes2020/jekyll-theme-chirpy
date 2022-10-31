package BT;

import java.util.Deque;
import java.util.LinkedList;
import java.util.Stack;

public class BTTraversalLevelSpecialOrder {

    // Two stack to print in spiral way
    public void BTTraversalLevelSpecialOrderwithTwoStack(Node node) {
        if(node == null) {
            return;
        }
        Stack<Node> s1 = new Stack<Node>();
        Stack<Node> s2 = new Stack<Node>();
        s1.push(node);
        while (!s1.isEmpty() || !s2.isEmpty()) {
            while (!s1.isEmpty()) {
                node=s1.pop();
                if(node.left != null) {
                    s2.add(node.left);
                }
                if(node.right!= null) {
                    s2.add(node.right);
                }
                System.out.print(node.data + " ");
            }
            while (!s2.isEmpty()) {
                node=s2.pop();
                if(node.left != null) {
                    s1.add(node.left);
                }
                if(node.right!= null) {
                    s1.add(node.right);
                }
                System.out.print(node.data+ " ");
            }
        }
    }


    // One deque with count method to print tree in spiral order
    public void BTTraversalLevelSpecialOrderwithOneDeque(Node root) {
        if (root == null) {
            return;
        }
        Deque<Node> deque = new LinkedList<Node>();
        deque.offerFirst(root);
        int count = 1;
        boolean flip = true;
        while (!deque.isEmpty()) {
            int currentCount = 0;
            while (count > 0) {
                if (flip) {
                    root = deque.pollFirst();
                    System.out.print(root.data + " ");
                    if (root.left != null) {
                        deque.offerLast(root.left);
                        currentCount++;
                    }
                    if (root.right != null) {
                        deque.offerLast(root.right);
                        currentCount++;
                    }
                } else {
                    root = deque.pollLast();
                    System.out.print(root.data + " ");
                    if (root.right != null) {
                        deque.offerFirst(root.right);
                        currentCount++;
                    }
                    if (root.left != null) {
                        deque.offerFirst(root.left);
                        currentCount++;
                    }
                }
                count--;
            }
            flip = !flip;
            count = currentCount;
        }
    }


    public void levelByLevelWithOneDequeandCounter(Node node) {
        if(node == null) return;
        Node current;
        int levelcount = 1;
        int currentcount = 0;
        Deque<Node> dq = new LinkedList<>();
        dq.offerFirst(node);
        while(currentcount != 0) {
            while (levelcount != 0) {
                current = dq.pollFirst();
                System.out.print(current.data + " ");
                if(current.left != null){
                    dq.offerLast(current.left);
                    currentcount ++;
                }
                if(current.right != null){
                    dq.offerLast(current.right);
                    currentcount ++;
                }
                levelcount -= 1;
            }
            System.out.println();
            levelcount = currentcount;
            currentcount = 0;
        }
    }

    public static void main(String args[]) {
        BinaryTree bt = new BinaryTree();
        Node root = null;
        root = bt.addNode(10, root);
        root = bt.addNode(30, root);
        root = bt.addNode(25, root);
        root = bt.addNode(35, root);
        root = bt.addNode(-10, root);
        root = bt.addNode(0, root);
        root = bt.addNode(-20, root);
        root = bt.addNode(-15, root);
        root = bt.addNode(45, root);

        BTTraversalLevelSpecialOrder tt = new BTTraversalLevelSpecialOrder();
        System.out.println("Two stack method");
        tt.BTTraversalLevelSpecialOrderwithTwoStack(root);
        System.out.println("\nOne deque with count");
        tt.BTTraversalLevelSpecialOrderwithOneDeque(root);
        System.out.println("\nOne deque with delimiter");
        tt.levelByLevelWithOneDequeandCounter(root);
    }

}
