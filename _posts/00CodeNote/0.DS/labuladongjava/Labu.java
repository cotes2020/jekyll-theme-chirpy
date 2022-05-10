package labuladongjava;

import java.util.Comparator;
import java.util.PriorityQueue; 

// import labuladongjava.PriorityQueue.PriorityQueue;
import labuladongjava.other.ListNode;
import labuladongjava.other.UF;

public class Labu {

    public Labu() {
        // private String customer;
        // private String bank;
        // private String account;
        // private int limit;
        // protected double balance;
    }


    public ListNode mergeTwoLists(ListNode l1, ListNode l2) {
        // 虚拟头结点
        ListNode dummy = new ListNode(-1), p = dummy;
        ListNode p1 = l1, p2 = l2;
        
        while (p1 != null && p2 != null) {
            // 比较 p1 和 p2 两个指针
            // 将值较小的的节点接到 p 指针
            if (p1.val > p2.val) {
                p.next = p2;
                p2 = p2.next;
            } else {
                p.next = p1;
                p1 = p1.next;
            }
            // p 指针不断前进
            p = p.next;
        }
        
        if (p1 != null) {
            p.next = p1;
        }
        if (p2 != null) {
            p.next = p2;
        }
        
        return dummy.next;
    }

    public ListNode mergeKLists(ListNode[] lists) {
        ListNode dummy = new ListNode(0), cur = dummy;
        if (lists.length == 0)return null;

        // Add all the list nodes to the min heap
        PriorityQueue<ListNode> minpq = new PriorityQueue<ListNode>(
            lists.length, 
            new Comparator<ListNode>() {
                public int compare(ListNode l1, ListNode l2) {
                    return l1.val - l2.val;
                }
            }
        );
        for (int i = 0; i < lists.length; i++) {
            if (lists[i] != null) minpq.offer(lists[i]);
        }
        while (!minpq.isEmpty()) {
            ListNode temp = minpq.poll();
            cur.next = temp;
            if (temp.next != null) minpq.offer(temp.next);
            cur = temp;
        }
        return dummy.next;
    }

    // public static void main(String[] args) {
    //     Labu run = new Labu();
    //     ListNode[][] lists = [[1,4,5],[1,3,4],[2,6]];
    //     ListNode ans = run.mergeKLists(lists);
    //     System.out.println(ans);
    // }

    


    public boolean equationsPossible(String[] equations) {
        // 26 个英文字母
        UF uf = new UF(26);
        // 先让相等的字母形成连通分量
        for (String eq : equations) {
            System.out.println(eq);
            if (eq.charAt(1) == '=') {
                char x = eq.charAt(0);
                char y = eq.charAt(3);
                uf.union(x - 'a', y - 'a');
                System.out.println(x);
                System.out.println(y);
                // System.out.println(x - 'a');
                // System.out.println(y - 'a');
            }
        }
        
        if (uf.connected('f' - 'a', 'f' - 'a')) System.out.println("yesc");
        if (uf.connected('b' - 'a', 'd' - 'a')) {System.out.println("yesb");} else {System.out.println("noyesb");}
        if (uf.connected('x' - 'a', 'z' - 'a')) {System.out.println("yesx");} else {System.out.println("noyesx");}
        
        // 检查不等关系是否打破相等关系的连通性
        for (String eq : equations) {
            if (eq.charAt(1) == '!') {
                char x = eq.charAt(0);
                char y = eq.charAt(3);
                // 如果相等关系成立，就是逻辑冲突
                if (uf.connected(x - 'a', y - 'a'))
                    System.out.println(x);
                    System.out.println(x);
                    System.out.println(y);
                    return false;
            }
        }
        return true;
    }
}



















