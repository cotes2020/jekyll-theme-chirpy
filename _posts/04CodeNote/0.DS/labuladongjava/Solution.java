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
import javax.swing.plaf.synth.SynthSpinnerUI;

import labuladongjava.other.ListNode;

public class Solution {
    public int countBinarySubstrings(String s) {
        int res=0, pre=1, cur=1;
        for(int i=0; i<s.length()-1; i++){
            System.out.print(s.charAt(i));
            System.out.println(s.charAt(i+1));
            if(s.charAt(i)==s.charAt(i+1)){
                cur++;
                // System.out.print("same");
                System.out.print(", cur ");
                System.out.print(cur);
                System.out.print(", pre ");
                System.out.print(pre);
                System.out.print(", res ");
                System.out.println(res);
            }
            else{
                res+=Math.min(pre, cur);
                System.out.print("notsame");
                System.out.print(", cur ");
                System.out.print(cur);
                System.out.print(", pre ");
                System.out.print(pre);
                System.out.print(", res ");
                System.out.println(res);
                pre=cur;
                cur=1;
            }
        }
        return res;
    }


    public static void main(String[] args) {
        // int[] nums = new int[]{1,2,2};
        // System.out.println(reverseBetween(nums)); 
        String s = new String("00110011");
        Solution res = new Solution();
        res.countBinarySubstrings(s);
    } 

    
}
