package BT;


import java.util.LinkedList;
import java.util.Queue;

// use queue

public class BSTTraversalLevelOrder {

    public void levelOrderTraversal(Node node) {
        if(node == null) {
            System.out.println("Please enter a valid tree!");
            return;
        }

        Queue<Node> q = new LinkedList<Node>();
        q.add(node);
        // q.offer(node);

        while(!q.isEmpty()) {
            node = q.poll();          // put into the queue
            // System.out.print(node.data);
            System.out.print(node.data + " ");
            if(node.left != null) {
                q.add(node.left);     // put into the queue
            }
            if(node.right != null) {
                q.add(node.right);    // put into the queue
            }
        }
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

        BSTTraversalLevelOrder loi = new BSTTraversalLevelOrder();
        loi.levelOrderTraversal(head);
    }

}
