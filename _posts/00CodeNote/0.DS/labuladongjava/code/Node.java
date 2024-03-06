import java.util.ArrayList;
import java.util.List;

import javax.swing.text.AsyncBoxView.ChildState;

class Node {

    int data;
    Node left = null, right = null;
    List<Node> childs;

    Node(int data) {
        this.data = data;
        this.childs = new ArrayList<>();
    }

    Node(int data, int child_data) {
        this.data = data;
        this.childs.add(new Node(child_data));
    }

    public void print_childs_list() {
        List<Integer> childs_value = new ArrayList<>();
        for(Node child : this.childs) {
            childs_value.add(child.data);
        }
        System.out.println(childs_value);
    }
}
