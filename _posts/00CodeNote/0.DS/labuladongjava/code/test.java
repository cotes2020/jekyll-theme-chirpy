import java.util.*;

public class Test {

    public int maxArea(int[] height) {
        int max = 0;
        for(int i=0;i<height.length-1; i++) {
            int curxa = i, curya=height[i];
            for(int j=i+1; j<height.length; j++) {
                int curxb = j, curyb=height[j];
                max = Math.max(max, (curxb-curxa) * Math.min(curya,curyb));
                System.out.println("xa:"+curxa+"ya:"+curya+"xb:"+curxb+"yb:"+curyb);
                System.out.println("max:"+max);

            }
        }
        return max;
    }

    public static void main(String[] args) {
        System.out.println("run");
        Test run = new Test();
        int[] nums = new int[]{0,2};
        // int target = 8;
        int ans = run.maxArea(nums);
        System.out.println(ans);

        // for(List<Integer> x : ans) System.out.println(x);
        // for(int x : ans) System.out.println(x);
    }

}
