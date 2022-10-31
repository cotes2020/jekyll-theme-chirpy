package BT;


import java.util.LinkedList;
import java.util.Queue;

//        10
//     5     15
//   4   7      19
// 3              20



public class BTTraversalLevelByLevel {

    // Level Printing with two queue

    public void levelByLevelWithTwoQueue(Node node) {
        if(node == null) return;
        Node current = node;
        Queue<Node> q1 = new LinkedList<>();
        Queue<Node> q2 = new LinkedList<>();
        q1.add(node);
        while (!q1.isEmpty() || !q2.isEmpty()) {
            while (!q1.isEmpty()) {
                current = q1.poll();
                System.out.print(current.data+ " ");
                if(current.left != null){
                    q2.offer(current.left);
                }
                if(current.right != null){
                    q2.offer(current.right);
                }
            }
            System.out.println();
            while (!q2.isEmpty()) {
                current = q2.poll();
                System.out.print(current.data+ " ");
                if(current.left != null){
                    q1.offer(current.left);
                }
                if(current.right != null){
                    q1.offer(current.right);
                }
            }
            System.out.println();
        }
    }



    public void levelByLevelWithOneQueue(Node node) {
        if(node == null) return;
        Node current = node;
        Queue<Node> q = new LinkedList<>();
        q.offer(node);
        q.offer(null);
        while (!q.isEmpty()) {
            current = q.poll();
            if (current != null) {
                System.out.print(current.data+ " ");
                if(current.left != null){
                    q.offer(current.left);
                }
                if(current.right != null){
                    q.offer(current.right);
                }
            }
            else {
                if (!q.isEmpty()) {
                    System.out.println();
                    q.offer(null);
                }

            }
        }
    }



    public void levelByLevelWithOneQueueandCounter(Node node) {
        if(node == null) return;
        Node current;
        int levelcount = 1;
        int currentcount = 0;
        Queue<Node> q = new LinkedList<>();
        q.offer(node);
        while(!q.isEmpty()) {
            while (levelcount != 0) {
                current = q.poll();
                System.out.print(current.data + " ");
                if(current.left != null){
                    q.offer(current.left);
                    currentcount ++;
                }
                if(current.right != null){
                    q.offer(current.right);
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
        BTTraversalLevelByLevel tt = new BTTraversalLevelByLevel();
        BinaryTree bt = new BinaryTree();
        Node root = null;
        root = bt.addNode(10, root);
        root = bt.addNode(15, root);
        root = bt.addNode(0, root);
        root = bt.addNode(5, root);
        root = bt.addNode(-1, root);
        root = bt.addNode(2, root);
        root = bt.addNode(6, root);

        System.out.println("1. Two queue technique");
        tt.levelByLevelWithTwoQueue(root);
        System.out.println("\n2. One queue and delimiter");
        tt.levelByLevelWithOneQueue(root);
        System.out.println("\n\n3. One queue and count");
        tt.levelByLevelWithOneQueueandCounter(root);
    }


}
