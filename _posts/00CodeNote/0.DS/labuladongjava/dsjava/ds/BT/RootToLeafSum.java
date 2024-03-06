package BT;


import java.util.ArrayList;
import java.util.List;


public class RootToLeafSum {

    public boolean issumexist(Node node, int key, List<Node> path) {

        if (node == null) {
            return false;
        }
        if (node.data == key) {
            path.add(node);
            return true;
        }
        if (issumexist(node.left, key - node.data, path) || issumexist(node.right, key - node.data, path)) {
            path.add(node);
            return true;
        }
        return false;

        // if(node.left == null && node.right == null){
        //     if(node.data == key){
        //         path.add(node);
        //         return true;
        //     }else{
        //         return false;
        //     }
        // }
        // if(issumexist(node.left, key-node.data, path) || issumexist(node.right, key - node.data, path)){
        //     path.add(node);
        //     return true;
        // }
        // return false;
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
        head = bt.addNode(4, head);
        head = bt.addNode(3, head);

        //        10
        //     5     15
        //   4   7      19
        // 3              20

        List<Node> result = new ArrayList<>();

        RootToLeafSum rtl = new RootToLeafSum();
        boolean r = rtl.issumexist(head, 22, result);
        if(r){
            System.out.println("Have path for sum ");
            result.forEach(node -> System.out.print(node.data + " "));
        }else{
            System.out.println("No path for sum ");
        }
    }
}
