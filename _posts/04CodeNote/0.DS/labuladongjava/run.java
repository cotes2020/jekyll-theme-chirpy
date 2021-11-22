package labuladongjava;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;

import javax.sound.sampled.Mixer;

public class run { 

    public static int f(int[] piles, int x){
        int hours = 0;
        for(int num : piles){
            hours += num/x;
            if(num%x>0) hours++;
        }
        return hours;
    }
    
    public static int minEatingSpeed(int[] piles, int H) {
        int left = 1, right=1000000000 + 1;
        while(left<right){
            int mid = left + (right-left)/2;
            if(f(piles, mid) <= H) right = mid-1;
            else left = mid+1;
        }
        return left;
    }

    public static void main(String[] args) {
        int[] piles = new int[]{30,11,23,4,20};
        int h = 5;

        // System.out.println(searchRange(s, target));
        System.out.println(minEatingSpeed(piles, h)); 
    }
}
