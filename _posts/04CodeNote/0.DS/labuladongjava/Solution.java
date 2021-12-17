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

public class Solution { 
    public static int removeDuplicates(int[] nums) {
        if(nums.length==0) return 0; 
        int fast=1, slow=0;
        while(fast<nums.length){  
            System.out.println(nums[fast]);
            System.out.println(nums[slow]);
            if(nums[fast]!=nums[slow]){
                slow++;
                nums[slow] = nums[fast];
            }
            fast++; 
        }
        return slow+1;
    }

    public static void main(String[] args) {
        int[] nums = new int[]{1,2,2};
        System.out.println(removeDuplicates(nums)); 
    }

    
}
