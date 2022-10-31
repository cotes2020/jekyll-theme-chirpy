package BT;


public class BSTLowestAncestor {

    // public void bstLowestAncestor(Node node, int key1, int key2) {
    //     if(node==null) {
    //         return;
    //     }
    //     if(node.data == key1 || node.data == key2 ) {
    //         System.out.println(node.data);
    //         return;
    //     }
    //     if(key1 < node.data && node.data < key2) {
    //         System.out.println(node.data);
    //         return;
    //     }
    //     bstLowestAncestor(node.left, key1, key2);
    //     bstLowestAncestor(node.right, key1, key2);
    // }

    public Node bstLowestAncestor(Node node, int p, int q) {
        if (node.data > Math.max(p, q)) {
            return bstLowestAncestor(node.left, p, q);
        } else if (node.data < Math.min(p, q)) {
            return bstLowestAncestor(node.right, p, q);
        } else {
            return node;
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
        BSTLowestAncestor go = new BSTLowestAncestor();
        root = go.bstLowestAncestor(root, 2, 15);
        System.out.println(root.data);
    }
}
