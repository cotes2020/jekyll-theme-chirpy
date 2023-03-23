package BT;


import java.util.LinkedList;
import java.util.Queue;
import java.util.Stack;


// from bootom line to top line

public class BTTraversalLevelPrintReversal {

    public void reverseLevelPrint(Node node) {
        if (node == null) return;
        Stack<Node> s = new Stack<Node>();
        Queue<Node> q = new LinkedList<>();
        q.add(node);
        while (!q.isEmpty()) {
            node = q.poll();

            if(node.right != null){
                q.offer(node.right);
            }
            if(node.left != null){
                q.offer(node.left);
            }

            s.add(node);
        }
        while(!s.isEmpty()){
            System.out.print(s.pop().data + " ");
        }
    }

    public static void main(String args[]){
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
        BTTraversalLevelPrintReversal rlo = new BTTraversalLevelPrintReversal();
        rlo.reverseLevelPrint(root);
    }
}
