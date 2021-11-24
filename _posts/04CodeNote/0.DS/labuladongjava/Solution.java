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
    public static int dp(int[] coins, int amount) { 
         
        int[] subCoin = new int[amount+1];
        Arrays.fill(subCoin, -1);
        subCoin[0]=0;
        
        for(int i=1; i<=amount;i++){
            int minC = Integer.MAX_VALUE;
            for(int coin:coins){
                int rest = i - coin;
                if(rest < 0) continue;
                if(subCoin[rest] ==-1) continue;
                minC = Math.min(minC, subCoin[rest]);
            }
            if(minC==Integer.MAX_VALUE) continue;
            subCoin[i]=minC+1;
            System.out.println(Arrays.toString(subCoin));
        }
        return subCoin[amount];
    }

    public static void main(String[] args) {
        int[] coins = new int[]{2,5,10,1};
        int amount = 27;    

        // System.out.println(searchRange(s, target));
        System.out.println(dp(coins, amount)); 
    }

    
}
