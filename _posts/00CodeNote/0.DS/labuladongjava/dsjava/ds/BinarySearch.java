package labuladongjava.other;

// 二分搜索


public class BinarySearch {

    public int binarysearch(int array[], int key) {
        int left = 0;
        int right = array.length - 1;
        while(left <= right) {
            int mid = (left+right)/2;

            if(key == array[mid]) {
                return mid;
            }
            else if(key > array[mid]) {
                left = mid + 1;
            }
            else {
                right = mid - 1;
            }
        }
        return -1;
    }
}
