
class Main {

    public static boolean isNodePresent(Node root, int node_data) {
        // base case
        if (root == null) return false;
        // if the node is found, return true
        if (root.data == node_data) return true;
        return isNodePresent(root.left, node_data) || isNodePresent(root.right, node_data);
    }

    public static int findLevel(Node root, int node_data, int level) {
        // base case
        if (root == null) return -1;
        // return level if the node is found
        if (root.data == node_data) return level;

        // search node in the subtree
        int left = findLevel(root.left, node_data, level + 1);
        if (left != -1) return left;
        return findLevel(root.right, node_data, level + 1);
    }

    // Function to find the lowest common ancestor of given nodes `x` and `y`,
    // where both `x` and `y` are present in the binary tree.
    public static Node findLCA(Node root, int x, int y) {
        // base case 1: if the tree is empty
        if (root == null) return null;
        // base case 2: if either `x` or `y` is found
        if (root.data == x || root.data == y) return root;

        // recursively check if `x` or `y` exists in the left subtree
        Node left = findLCA(root.left, x, y);
        // recursively check if `x` or `y` exists in the right subtree
        Node right = findLCA(root.right, x, y);

        // if `x` is found in one subtree and `y` is found in the other subtree,
        // update lca to the current node
        if (left != null && right != null) return root;

        // if `x` and `y` exist in the left subtree
        if (left != null) return left;
        // if `x` and `y` exist in the right subtree
        if (right != null) return right;

        return null;
    }

    // Function to find the distance between node `x` and node `y` in a
    // given binary tree rooted at `root` node
    public static int findDistance(Node root, int x, int y) {
        // `lca` stores the lowest common ancestor of `x` and `y`
        Node lca = null;
        int lev_x = findLevel(root, x, 0);
        int lev_y = findLevel(root, y, 0);

        if(lev_x == -1) return -1;
        if(lev_y == -1) return -1;
        if(lev_x == -1 && lev_y == -1) return -1;

        // call LCA procedure only if both `x` and `y` are present in the tree
        // if (isNodePresent(root, y) && isNodePresent(root, x)) {
        //     lca = findLCA(root, x, y);
        // }
        // else return Integer.MIN_VALUE;
        lca = findLCA(root, x, y);

        // return distance of `x` from lca + distance of `y` from lca
        // return findLevel(lca, x, 0) + findLevel(lca, y, 0);
        return lev_x + lev_y - 2*findLevel(root, lca.data, 0);
        // avoid calling the `isNodePresent()` function by
        // using return values of the `findLevel()` function
        // to check if `x` and `y` are present in the tree or not.
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
