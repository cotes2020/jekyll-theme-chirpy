 
public class Test {

    public int[] searchRange(int[] nums, int target) { 
        int[] result = new int[2];
        result[0] = findFirst(nums, target);
        int y = findFirst(nums, target+1);
        System.out.println("y: " + y); 

        result[1] = y==-1? nums.length-1: y-1 ;
        return result;
    }

    
    public int findFirst(int[] nums, int target) {
        int res=-1;
        int l=0, r=nums.length-1;
        while(l<=r){
            int m=(l+r)/2;
            if(target<=nums[m]) r=m-1;
            else l=m+1;
            if(nums[m]==target) res=m; 
        }
        return res;
    }

    public static void main(String[] args) {
        System.out.println("run");
        Test run = new Test();
        int[] nums = new int[]{5,8,8,8,8,10};
        int target = 8;
        int[] ans = run.searchRange(nums, target);
        // System.out.println(ans); 
        
        for(int x : ans) System.out.println(x);
    }
    
}
