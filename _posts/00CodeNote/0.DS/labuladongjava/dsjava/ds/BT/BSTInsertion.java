package BT;


// Binary Search Tree Insertion (Iterative method)


public class BSTInsertion {

    public Node insert(int data, Node root) {
        Node key = new Node();
        key.data = data;

        if (root == null) {return root;}

        Node parent = null;
        Node current = root;

        while (current != null) {
            parent = current;
            if (data < parent.data) {
                current = parent.left;
            }
            else {
                current = parent.right;
            }
        }
        if (data < parent.data) {
                parent.left = key;
        }
        else {
                parent.right = key;
        }

        return root;
    }

    public static void main(String args[]){
        BSTInsertion bt = new BSTInsertion();
        Node head = null;
        head = bt.insert(10, head);
        head = bt.insert(15, head);
        head = bt.insert(5, head);
        head = bt.insert(7, head);
        head = bt.insert(19, head);
        head = bt.insert(20, head);
        head = bt.insert(-1, head);
        head = bt.insert(21, head);
    }
}
