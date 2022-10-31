public class Sort {


    public static void insertionSort(char[] data) {
        int n = data.length;
        for(i=0; i<n-1; i++){
            char cur = data[i];
            int j = k;
            while(j>0 && data[j-1] > data[j]){
                data[j] = data[j-1];
                j--;
            }
            data[j] = cur;
        }
    }
}
