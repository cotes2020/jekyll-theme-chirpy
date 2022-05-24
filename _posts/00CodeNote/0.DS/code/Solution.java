package labuladongjava;
import labuladongjava.other.ListNode;

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


public class Solution { 
    public int dominantIndex(int[] nums) {
        int n=nums.length; 
        int slow=-1;
        int max = nums[0];
        for(int fast=0; fast<n; fast++) {
            if(nums[fast]>nums[max]) max=fast;
            // System.out.println("slow: " + nums[slow]);
            // System.out.println("fast: " + nums[fast]);


            if(nums[fast]>=nums[max]*2 && nums[fast]>nums[max]){
                slow=fast;
            } 
        }
        return slow;
        
    }

    public static void main(String[] args) {
        Solution res = new Solution();
        int[] nums = {1,2,3,4};
        int ans = res.dominantIndex(nums);
        System.out.println("ans:" + ans);
    } 
}
