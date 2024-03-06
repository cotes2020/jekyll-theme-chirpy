import java.io.File;
import java.util.ArrayList;

public class Recursion {

    // 5! = 5 · 4 · 3 · 2 · 1
    // 2! = 2 · 1
    // 1! = 1
    public static int factorial(int n) throws IllegalArgumentException {
        if(n<0) throw new IllegalArgumentException();
        else if(n==0) return 1;
        else return factorial(n-1) * n;
    }


    // Drawing an English Ruler
    // public static void drawRuler(int nlnches, int majorLength) {
    //     drawLine(majotLength, 0);
    //     for(int j=1; j<=nlnches; j++){
    //         drawInterval(majorLength - 1);
    //         drawLine(majorLength, j);
    //     }
    // }

    // 1 2 3 4 5 6 7 8 9
    public static boolean binarySearch(int[] data, int target, int low, int high) {
        if(low>high) return false;
        int mid = (low + high)/2;
        if(data[mid]==target) return true;
        else if(data[mid]>target) return binarySearch(data, target, low, mid-1);
        else return binarySearch(data, target, mid+1, high);
    }

    // FILE SYSTEM
    public static long diskUsage(File root) {
        long disk_usage = root.length();
        if(root.isDirectory()) {
            for(String file: root.list()) {
                File child = new File(root, file);
                disk_usage += diskUsage(child);
            }
        }
        System.out.println(disk_usage + "\t" + root);
        return disk_usage;
    }


    // Returns the sum of the first n integers of the given array
    public static int linearSum(int[ ] data, int n) {
        if (n == 0) return 0;
        else return linearSum(data, n−1) + data[n−1];
    }

    // Reverses the contents of subarray data[low] through data[high] inclusive.
    public static void reverseArray(int[ ] data, int low, int high) {
        if(low<high){
            int temp = data[low];
            data[low] = data[high];
            data[high] = temp;
            reverseArray(data, low+1, high-1);
        }
    }





    public static void main(String[] args) {
        System.out.println("hi");
        Recursion pr = new Recursion();

        // int[] data = new int[]{1,2,3,4,5,6,7,8,9};
        // boolean ans1 = Recursion.binarySearch(data, 6, 0, data.length-1);
        // System.out.println(ans1);


        File root = new File("/Users/luo/Documents/GitHub/ocholuo.github.io/_posts/04CodeNote/0.DS/labuladongjava/dsjava/recursion");
        long ans2 = Recursion.diskUsage(root);
        System.out.println(ans2);
    }
}
