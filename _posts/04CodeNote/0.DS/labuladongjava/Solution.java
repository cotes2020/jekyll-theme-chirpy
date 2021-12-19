package labuladongjava;

import java.lang.module.ModuleDescriptor.Builder;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Random;

import javax.sound.sampled.Mixer;

import labuladongjava.other.ListNode;

public class Solution { 
    public ListNode reverseBetween(ListNode head, int left, int right) {
        ListNode dummy=new ListNode(0,head);
        ListNode cur=head, lPre = dummy;
        for(int i=0; i<left; i++){
            lPre=cur;
            cur=cur.next;
        }
        ListNode prev = null, temp = null;
        for(int i = left; i<right; i++){
            temp = cur.next;
            cur.next=prev;
            prev=cur;
            cur=temp;
        } 
        lPre.next.next = temp;
        lPre.next = cur;
        return dummy.next;
    }

    public static void main(String[] args) {
        int[] nums = new int[]{1,2,2};
        // System.out.println(reverseBetween(nums)); 
    } 

    
}
