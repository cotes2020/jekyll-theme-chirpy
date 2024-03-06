package BT;



public class HeightOfBT {

    public int heightofBT(Node root) {
        if (root == null) {
            return 0;
        }
        int lefthight = heightofBT(root.left) + 1;
        int righthight = heightofBT(root.right) + 1;
        return 1 + Math.max(lefthight, righthight);
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
        head = bt.addNode(21, head);
        System.out.println(bt.height(head));

    }
}
