import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedList;
import java.util.List;

public class Solution {

    public List<List<Integer>> threeSum(int[] nums) {

        Arrays.sort(nums);
        List<List<Integer>> res = new LinkedList<>();
        // if length is less than 3, return empty result set
        int n = nums.length;
        if (n < 3 || nums[0] > 0) return res;

        // left to tight
        for(int i=0; i<n-2; i++){
            if (i == 0 || (i > 0 && nums[i] != nums[i - 1])){
                //  i j ------ k
                int j=i+1, k=n-1;
                while(j<k){
                    if(0==nums[i]+nums[j]+nums[k]) {
                        res.add(Arrays.asList(nums[i], nums[j], nums[k]));
                        // System.out.println("i:" + nums[i]+ ", j:" + nums[j]+ ", k:" +nums[k]);
                        while(j<k && nums[j]==nums[j+1]) j++;
                        while(j<k && nums[k]==nums[k-1]) k--;
                        j++;
                        k--;
                    }
                    // need smaller number
                    else if(0<nums[i]+nums[j]+nums[k]) k--;
                    // need bigger number
                    else j++;
                }
            }
        }
        return res;
    }

    public static void main(String[] args) {
        Solution res = new Solution();
        // String s = "amanaplanacanalpanama";
        // int[] nums = new int[]{-7,-3,-3,-1,-1,0,1,6};
        int[] nums = new int[]{1,0,-1};
        List<List<Integer>> ans = res.threeSum(nums);
        System.out.println("ans:" + ans);
    }
}
