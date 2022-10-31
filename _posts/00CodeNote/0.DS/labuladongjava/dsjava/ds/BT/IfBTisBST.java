package BT;




public class IfBTisBST {


    public boolean isBST(Node root){
        return ifBTisBST(root, Integer.MIN_VALUE, Integer.MAX_VALUE);
    }

    private boolean ifBTisBST(Node node, int min, int max) {
        if(node == null) {return true;}
        if(node.data < min || node.data > max) {return false;}
        return ifBTisBST(node.left, min, node.data) &&  ifBTisBST(node.right, node.data, max);
    }

    public static void main(String args[]){
        BinaryTree bt = new BinaryTree();
        Node root = null;
        root = bt.addNode(10, root);
        root = bt.addNode(15, root);
        root = bt.addNode(-10, root);
        root = bt.addNode(17, root);
        root = bt.addNode(20, root);
        root = bt.addNode(0, root);

        IfBTisBST isBST = new IfBTisBST();
        assert isBST.isBST(root);
        System.out.println(isBST.isBST(root));
    }

}
