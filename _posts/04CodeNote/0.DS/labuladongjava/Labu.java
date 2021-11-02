package labuladongjava;

import java.util.Comparator;
import java.util.PriorityQueue;

// import labuladongjava.PriorityQueue.PriorityQueue;
import labuladongjava.other.ListNode;

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

}