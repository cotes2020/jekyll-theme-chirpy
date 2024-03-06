import java.util.ArrayList;
import java.util.List;

class Main {

    // int[][] input = {
    //     { 1, 3 },
    //     { 1, 2 },
    //     { 3, 5 }
    // };

    public static void build_tree(int[][] input) {
        Node root = new Node(input[0][0]);
        System.out.println("root: " + root.data);
        for(int[] pair:input){
            Node parent = findNode(root, pair[0]);
            if(parent != null) {
                parent.childs.add( new Node(pair[1]) );
            }
            else throw new IllegalArgumentException("Input array is null");
        }
    }

    public static Node findNode(Node root, int node_data) {
        Node target = null;

        // base case
        if (root == null) return null;
        // if the node is found, return true
        if (root.data == node_data) return root;

        List<Node> child_list = root.childs;
        for(Node child : child_list) {
            target = findNode(child, node_data);
            if(target != null) return target;
        }
        return target;
    }

    public static int findLevel(Node root, Node node, int level) {
        // base case
        if (root == null) return -1;
        // return level if the node is found
        if (root == node) return level;

        Integer target_level = -1;
        List<Node> child_list = root.childs;
        for(Node child : child_list) {
            target_level = findLevel(child, node, level+1);
            if(target_level != -1) return target_level;
        }
        return target_level;
    }


    // Function to find the distance between node `x` and node `y` in a
    // given binary tree rooted at `root` node
    public static int findDistance(Node root, int x, int y) {
        // `lca` stores the lowest common ancestor of `x` and `y`
        Node lca = null;

        Node node_x = findNode(root, x);
        Node node_y = findNode(root, y);

        int lev_x = findLevel(root, node_x, 0);
        int lev_y = findLevel(root, node_y, 0);

        if(lev_x == 0 || lev_y == 0) return -1;


        if(lev_x == -1) return -1;
        if(lev_y == -1) return -1;
        if(lev_x == -1 && lev_y == -1) return -1;


        // call LCA procedure only if both `x` and `y` are present in the tree
        if (findNode(root, y) !=null && findNode(root, x) != null) {
            lca = findLCA(root, node_x, node_y);
        }
        else return Integer.MIN_VALUE;

        // return distance of `x` from lca + distance of `y` from lca
        return findLevel(lca, node_x, 0) + findLevel(lca, node_y, 0);
    }

    public static void main(String[] args) {
        /* Construct the following tree
              1
            /   \
           /     \
          2       3
           \     / \
            4   5   6
               /     \
              7       8
        */
        int[][] input = {
            { 1, 3 },
            { 1, 2 },
            { 3, 5 }
        };

        build_tree(input);
        System.out.println("built the tree.");

        Node root = new Node(1);
        root.left = new Node(2);
        root.right = new Node(3);
        root.left.right = new Node(4);
        root.right.left = new Node(5);
        root.right.right = new Node(6);
        root.right.left.left = new Node(7);
        root.right.right.right = new Node(8);
        // find the distance between node 7 and node 6
        // System.out.print(findDistance(root, root.right.left.left, root.right.right));
        System.out.print(findDistance(root, 6, 7));
    }
}
