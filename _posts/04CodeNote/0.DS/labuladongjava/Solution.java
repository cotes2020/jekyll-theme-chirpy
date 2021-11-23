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

    HashMap<Integer,Integer>map;
    Random r;
    int range;
    public Solution(int n, int[] blacklist) {
        range = n-blacklist.length;
        r = new Random();
        map = new HashMap<>();
        for(int k: blacklist) map.put(k,-1); 
        int last = n-1;
        for(int k : blacklist) {
           if(k<range) {
            while(map.containsKey(last)) last--; 
               map.put(k,last);
               last--;
           }
            
        }
        
        
    }
    
    public int pick() {
        int val = r.nextInt(range);
        if(map.containsKey(val))
        {
            return map.get(val);
        }
        return val;
        
    }

    public static void main(String[] args) {
        int k = 4;
        int[] b = new int[]{2,1};
        build(k,b);

        // System.out.println(searchRange(s, target));
        System.out.println(pick()); 
    }

    
}
