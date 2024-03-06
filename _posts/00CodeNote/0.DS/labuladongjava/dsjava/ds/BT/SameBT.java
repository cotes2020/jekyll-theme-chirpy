package BT;

// Check Same Binary Tree
// ![Screen Shot 2020-07-24 at 11.27.26](https://i.imgur.com/dcOA8T4.png)
// time O(n)

public class SameBT {

    public boolean sameBT(Node root1, Node root2) {
        if (root1 == null && root2 == null) {
            return true;
        }
        if (root1 == null || root2 == null) {
            return false;
        }
        return (root1.data == root2.data) && sameBT(root1.left, root2.left) && sameBT(root1.right, root2.right);
    }

    public static void main(final String args) {
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

        SameBT st = new SameBT();
        assert st.sameBT(root1, root2);
    }
}
