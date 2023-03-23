
import java.util.*;

public class Sort{


    // Bubble sort
	// push the biggedt to the end
	public void bubbleSort(int arr[]) {
		int i, j, temp;
        boolean swapped;
        int n = arr.length;
		for (i = 0; i < n - 1; i++) {
			swapped = false;
			for (j = 0; j < n - i - 1; j++){
				if (arr[j] > arr[j + 1]) { // swap arr[j] and arr[j+1]
					temp = arr[j];
					arr[j] = arr[j + 1];
					arr[j + 1] = temp;
					swapped = true;
				}
			}
			// IF no two elements were swapped by inner loop, then break
			if (swapped == false)
				break;
		}
	}


	// SelectionSort
	// find the smallesr from arr[0]
	public void SelectionSort(int[] num) {
	    for (int k=0; k < in.size(); j++) {
	        int minIndex = k;
	        for (int i=k+1; i < num.length; i++) {
	            if (num[minIndex] > num[i]) {
	                minIndex = i;
	            }
	        }
	        int temp = num[k];
	        num[k] = num[minIndex];
	        num[minIndex] = temp;
	    }
	}


    // ==================================================================

    // Insertion Sort
    // The array is virtually split into a sorted and an unsorted part.
    // Values from the unsorted part are picked and placed at the correct position in the sorted part.
    // Algorithm
    // To sort an array of size n in ascending order:
    // 1: Iterate from arr[1] to arr[n] over the array.  --> key
    // 2: Compare key arr[i] to its predecessor arr[i - 1] arr[j]
    // 3: If the key is smaller than its predecessor
    //    predecessor arr[j] position up
    //    j--
    //    Compare key to predecessor arr[j]
    // 4: If the key is bigger than predecessor
    //    arr[j + 1] = key
    // time O(n^2); space O(1)

    public void insertionSort(int arr[]) {
        int len = arr.length;
        for (int i = 1; i < len; i++) {
            int key = arr[i];
            int j = i - 1;
            while (j >= 0 && arr[j] > key) {
                arr[j + 1] = arr[j];
                j = j - 1;
            }
            arr[j + 1] = key;
        }
        return;
    }


    // ==================================================================

    // QuickSort
    // Divide and Conquer algorithm.
    // It picks an element as pivot and partitions the given array around the picked
    // pivot. There are many different versions of quickSort that pick pivot in
    // different ways.
    // - Always pick first element as pivot.
    // - Always pick last element as pivot (implemented below)
    // - Pick a random element as pivot.
    // - Pick median as pivot.
    // time O(n^2); space O(n)

    public void quickSort(int arr[], int low, int height){
        if (low >= height){
            return;
        }
        int pivot = part(arr, low, height);
        quickSort(arr, low, pivot - 1);
        quickSort(arr, pivot + 1, height);
    }

    public int part(int arr[], int low, int height){
        int pivot = arr[height];
        int small = low;
        int big = height - 1;
        while (true){
            while (small < height && arr[small] <= pivot){   // all left smaller then pivot
                small++;
            }
            while (low <= big && pivot < arr[big]){  // find first bigger in right
                big--;
            }
            if (big < small){
                swap(arr, small, height);
                break;
            }
            swap(arr, small, big);
        }
        return small;
    }

    public void swap(int[] arr, int small, int big) {
        int curr = arr[small];
        arr[small] = arr[big];
        arr[big] = curr;
    }





    public void merge(int arr[], int l, int m, int r) {
        int n1 = m - l + 1; // sizes of two subarrays to be merged
        int n2 = r - m;

        int L[] = new int[n1];
        int R[] = new int[n2];

        /* Copy data to temp arrays */
        for (int i = 0; i < n1; ++i) {
            L[i] = arr[l + i];
        }
        for (int j = 0; j < n2; ++j) {
            R[j] = arr[m + 1 + j];
        }

        /* Merge the temp arrays */
        // Initial indexes of first and second subarrays
        int i = 0, j = 0;
        // Initial index of merged subarry array
        int k = l;
        while (i < n1 && j < n2) {
            if (L[i] <= R[j]) {
                arr[k] = L[i];
                i++;
            } else {
                arr[k] = R[j];
                j++;
            }
            k++;
        }
        /* Copy remaining elements of L[] if any */
        while (i < n1) {
            arr[k] = L[i];
            i++;
            k++;
        }
        /* Copy remaining elements of R[] if any */
        while (j < n2) {
            arr[k] = R[j];
            j++;
            k++;
        }
    }

    // Main function that sorts arr[l..r] using
    // merge()
    public void mergesort(int arr[], int low, int height) {
        if (low < height) {
            int m = (low + height) / 2;
            mergesort(arr, low, m);
            mergesort(arr, m + 1, height);
            merge(arr, low, m, height);
        }
    }
    // MergeSort

    public static void mergeSort(int[] array, int[] copy, int start, int end) {
        if (start == end) {
            return; // 如果数组长度为0则直接返回
        }

        int middle = (end - start) / 2; // 将数组分为左右两部分，分别对左右两部分进行排序
        mergeSort(array, copy, start, start + middle);
        mergeSort(array, copy, start + middle + 1, end);

        int copyindex = end;
        int i = start + middle;
        int j = end;

        // 将左右两个排好序的数组进行排序，将结果保存到copy数组中
        while (i >= start && j >= start + middle + 1) {
            // 将左右数组由大到小复制到copy数组中
            if (array[i] > array[j]) {
                copy[copyindex--] = array[i--];
            } else {
                copy[copyindex--] = array[j--];
            }
        }
        while (i >= start) {
            copy[copyindex--] = array[i--];
            // 因为左数组中剩下的肯定比copy数组中最小的还小，如果左边的数组还有，则将其复制到copy数组中，
        }

        while (j >= start + middle + 1) {
            copy[copyindex--] = array[j--];
            // 因为右数组中剩下的肯定比copy数组中最小的还小如果右边的数组还有，则将其复制到copy数组中
        }
        for (i = start; i <= end; i++) {
            array[i] = copy[i];
            // 将copy数组复制到array数组中。
        }
    }

    public void printArray(int arr[]) {
        int n = arr.length;
        for (int i = 0; i < n; ++i){
            System.out.print(arr[i] + " ");
        }
        System.out.println();
    }

    public static void main(String args[]) {
        int arr1[] = { 12, 11, 13, 5, 6, 1 };
        int arr2[] = { 12, 11, 13, 5, 6, 1 };
        int arr3[] = { 12, 11, 13, 5, 6, 1 };
        Sort pr = new Sort();
        System.out.println("bubbleSort:");
        pr.bubbleSort(arr1);
        pr.printArray(arr1);
        System.out.println("insertionSort:");
        pr.insertionSort(arr1);
        pr.printArray(arr1);
        System.out.println("quickSort:");
        pr.quickSort(arr2, 0, 5);
        pr.printArray(arr2);
        System.out.println("MergeSort:");
        pr.mergesort(arr3, 0, arr3.length - 1);
        pr.printArray(arr2);
    }
}
