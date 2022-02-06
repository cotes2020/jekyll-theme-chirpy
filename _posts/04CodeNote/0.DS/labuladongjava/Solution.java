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
    public void duplicateZeros(int[] arr) {
        List<Integer> res = ArrayList<Integer>();
        int i=0, j=0;
        for (i=0; i<arr.length; i++){
            if (arr[i]==0){
                res.add(0);
                res.add(0);
            }
            else {
                res.add(arr[i]);
            }
            System.out.println();
        }
    }

    public static void main(String[] args) {
        int[] arr = [1,0,2,3,0,4,5,0];
        Solution res = new Solution();
        res.duplicateZeros(s);
    } 
}
