---
layout: post
title: "排序"
date: 2018-12-03 19:30:00.000000000 +09:00
categories: [算法]
tags: [算法, 排序]
---

排序算法是《数据结构与算法》中最基本的算法之一。排序算法可以分为内部排序和外部排序，内部排序是数据记录在内存中进行排序，而外部排序是因排序的数据很大，一次不能容纳全部的排序记录，在排序过程中需要访问外存。常见的内部排序算法有：插入排序、希尔排序、选择排序、冒泡排序、归并排序、快速排序、堆排序、基数排序等。

我们平常用的排序算法一般就以下几种：

|   名称   | 时间复杂度 | 空间复杂度 | 是否稳定 |
| :------: | :--------: | :--------: | :------: |
| 计数排序 |   O(n+k)   |   O(n+k)   |    是    |
| 冒泡排序 |   O(n^2)   |    O(1)    |    是    |
| 插入排序 |   O(n^2)   |    O(1)    |    是    |
| 选择排序 |   O(n^2)   |    O(1)    |    否    |
|  堆排序  |  O(nlogn)  |    O(1)    |    否    |
| 归并排序 |  O(nlogn)  |    O(1)    |    是    |
| 快速排序 |  O(nlogn)  |    O(1)    |    否    |
| 希尔排序 |   O(n^2)   |    O(1)    |    否    |
|  桶排序  |    O(n)    |    O(k)    |    是    |

## 计数排序

**计数排序Counting Sort**是一个非基于比较的排序算法，其核心在于将输入的数据值转化为键存储在额外开辟的数组空间中。 作为一种线性时间复杂度的排序，计数排序要求输入的数据必须是有确定范围的整数。该算法于1954年由 Harold H. Seward 提出。它的优势在于在`对一定范围内的整数排序`时，快于任何比较排序算法。

**算法复杂度**

时间平均复杂度：O(n+k) 最坏复杂度:O(n+k) 最好复杂度: O(n+k) 空间复杂度: O(n+k) 稳定

**排序思路**

- 1.找出待排序数组最大值
- 2.定义一个索引最大值为待排序数组最大值的数组
- 3.遍历待排序数组, 将待排序数组遍历到的值作新数组索引
- 4.在新数组对应索引存储值原有基础上+1

![](/assets/images/al-sort-01.png)

- C代码实现:

```c
int main()
{
    // 待排序数组
    int nums[5] = {3, 1, 2, 0, 3};
    // 用于排序数组
    int newNums[4] = {0};
    // 计算待排序数组长度
    int len = sizeof(nums) / sizeof(nums[0]);
    // 遍历待排序数组
    for(int i = 0; i < len; i++){
        // 取出待排序数组当前值
        int index = nums[i];
        // 将待排序数组当前值作为排序数组索引
        // 将用于排序数组对应索引原有值+1
        newNums[index] = newNums[index] +1;
    }
    
    // 计算待排序数组长度
    int len2 = sizeof(newNums) / sizeof(newNums[0]);
    int index = 0;
    // 输出排序数组索引, 就是排序之后结果
    for(int i = 0; i < len2; i++){
        for(int j = 0; j < newNums[i]; j++){
            nums[index] = i;
          	index += 1;
        }
    }
    return 0;
}
```

+ Swift代码实现

```swift
// 计数排序
// array: 待排序数组
// maxValue: 数组中最大值
func countSort(array: inout [NSInteger], maxValue: NSInteger) {

    var nums = [NSInteger](repeating: 0, count: maxValue + 1)
    // 遍历带排序的数组
    for i in 0..<array.count {
        // 取出带排序的数组当前值
        let value = array[i]
        // 将待排序数组当前值作为排序数组索引
        // 将用于排序数组对应索引原有值+1，主要目的->考虑数组中有重复数字
        nums[value] += 1
    }

    // 还原排序结果到待排序数组
    var index = 0
    for i in 0..<nums.count {
        for _ in 0..<nums[i] {
            array[index] = i
            index += 1
        }
    }
}
```

## 冒泡排序

**冒泡排序Bubble Sort**是一种简单的排序算法。它重复 地走访过要排序的数列,一次比较两个元素,如果他们的顺序错误就把他们交换过来。走访数列的工作是重复地进行直到没有再需要交换,也就是说该数列已经排序完成。这个算法的名字由来是因为越小的元素会经由交换慢慢“浮”到数列的顶端。

![](/assets/images/al-sort-02.gif)

**算法复杂度**

时间平均复杂度：O(n^2) 最坏复杂度:O(n^2) 最好复杂度: O(n) 空间复杂度: O(1) 稳定

**排序思路**

- 假设按照升序排序
- 1.从第0个元素开始, 每次都用相邻两个元素进行比较
- 2.一旦发现后面一个元素小于前面一个元素就交换位置
- 3.经过一轮比较之后最后一个元素就是最大值
- 4.排除最后一个元素, 以此类推, 每次比较完成之后最大值都会出现再被比较所有元素的最后
- 直到当前元素没有可比较的元素, 排序完成

C语言代码实现:

```c
// 冒泡排序
void bubbleSort(int numbers[], int length) {
    for (int i = 0; i < length; i++) {
        // -1防止`角标越界`: 访问到了不属于自己的索引
        for (int j = 0; j < length - i - 1; j++) {
           //  1.用当前元素和相邻元素比较
            if (numbers[j] < numbers[j + 1]) {
                //  2.一旦发现小于就交换位置
                swapEle(numbers, j, j + 1);
            }
        }
    }
}
// 交换两个元素的值, i/j需要交换的索引
void swapEle(int array[], int i, int j) {
    int temp = array[i];
    array[i] = array[j];
    array[j] = temp;
}
```

Swift语言代码实现:

```swift
// 冒泡排序
// array: 待排序数组
// 降序
func bubleSort(_ array: inout [NSInteger]) {

    for i in 0..<array.count {
        for j in 0..<array.count - i - 1 {

            if array[j] < array[j + 1] {
                swaps(&array, j, j + 1)
            }
        }
    }
}
// 交换两个数
func swaps<T>(_ chars: inout [T], _ s: Int, _ e: Int) {
    (chars[s], chars[e]) = (chars[e], chars[s])
}
```

## 插入排序

**插入排序Insertion-Sort**的算法描述是一种简单直观的排序算法。它的工作原理是通过构建有序序列，对于未排序数据，在已排序序列中从后向前扫描，找到相应位置并插入。

![](/assets/images/al-sort-03.gif)

**算法复杂度**

时间平均复杂度：O(n^2) 最坏复杂度:O(n^2) 最好复杂度: O(n) 空间复杂度: O(1) 稳定

**排序思路**

- 假设按照升序排序
- 1.从索引为1的元素开始向前比较, 一旦前面一个元素大于自己就让前面的元素先后移动
- 2.直到没有可比较元素或者前面的元素小于自己的时候, 就将自己插入到当前空出来的位置

C语言代码实现:

```c
int main()
{
    // 待排序数组
    int nums[5] = {3, 1, 2, 0, 3};
    // 0.计算待排序数组长度
    int len = sizeof(nums) / sizeof(nums[0]);

    //  1.从第一个元素开始依次取出所有用于比较元素
    for (int i = 1; i < len; i++)
    {
        // 2.取出用于比较元素
        int temp = nums[i];
        int j = i;
        while(j > 0){
            // 3.判断元素是否小于前一个元素
            if(temp < nums[j - 1]){
                // 4.让前一个元素向后移动一位
                nums[j] = nums[j - 1];
            }else{
                break;
            }
            j--;
        }
        // 5.将元素插入到空出来的位置
        nums[j] = temp;
    }
}
```

```c
int main()
{
    // 待排序数组
    int nums[5] = {3, 1, 2, 0, 3};
    // 0.计算待排序数组长度
    int len = sizeof(nums) / sizeof(nums[0]);

    //  1.从第一个元素开始依次取出所有用于比较元素
    for (int i = 1; i < len; i++)
    {
        // 2.遍历取出前面元素进行比较
        for(int j = i; j > 0; j--)
        {
            // 3.如果前面一个元素大于当前元素,就交换位置
            if(nums[j-1] > nums[j]){
                int temp = nums[j];
                nums[j] = nums[j - 1];
                nums[j - 1] = temp;
            }else{
                break;
            }
        }
    }
}
```

Swift语言代码实现:

```swift
// 插入排序
// array: 待排序数组
// 升序
func insertSort(_ array: inout [NSInteger]) {

    for i in 1..<array.count {
        for j in 0..<i {
            let k = i - j
            if array[k - 1] > array[k] {
                swaps(&array, k - 1, k)
            } else {
                break
            }
        }
    }
}
// 交换两个数
func swaps<T>(_ chars: inout [T], _ s: Int, _ e: Int) {
    (chars[s], chars[e]) = (chars[e], chars[s])
}
```

## 选择排序

**选择排序Selection sort**是一种简单直观的排序算法。它的工作原理如下。首先在未排序序列中找到最小元素,存放到排序序列的起始位置,然后,再从剩余未排序元素中继续寻找最小元素,然后放到排序序列末尾。以此类推,直到所有元素均排序完毕。

![](/assets/images/al-sort-04.gif)

**算法复杂度**

时间平均复杂度：O(n^2) 最坏复杂度:O(n^2) 最好复杂度: O(n^2) 空间复杂度: O(1) 不稳定

**排序思路**

- 假设按照升序排序.
- 1.用第0个元素和后面所有元素依次比较.
- 2.判断第0个元素是否大于当前被比较元素, 一旦小于就交换位置.
- 3.第0个元素和后续所有元素比较完成后, 第0个元素就是最小值.
- 4.排除第0个元素, 用第1个元素重复1~3操作, 比较完成后第1个元素就是倒数第二小的值.
- 以此类推, 直到当前元素没有可比较的元素, 排序完成.

C语言代码实现:

```c
// 选择排序(大到小)
void selectSort(int numbers[], int length) {
    
    // 外循环为什么要-1?
    // 最后一位不用比较, 也没有下一位和它比较, 否则会出现错误访问
    for (int i = 0; i < length; i++) {
        for (int j = i; j < length - 1; j++) {
            // 1.用当前元素和后续所有元素比较
            if (numbers[i] < numbers[j + 1]) {
                //  2.一旦发现小于就交换位置
                swapEle(numbers, i, j + 1);
            }
        }
    }
}
// 交换两个元素的值, i/j需要交换的索引
void swapEle(int array[], int i, int j) {
    int temp = array[i];
    array[i] = array[j];
    array[j] = temp;
}
```

Swift语言代码实现:

```swift
// 选择排序
// array: 待排序数组
// 降序
func selectSort(_ array: inout [NSInteger]) {

    for i in 0..<array.count {

        for j in i..<array.count - 1 {

            if array[i] < array[j + 1] {

                swaps(&array, i, j + 1)
            }
        }
    }
}
```

## 堆排序

**堆排序Heapsort**是指利用堆这种数据结构所设计的一种排序算法。堆积是一个近似完全二叉树的结构，并同时满足堆积的性质：即子结点的键值或索引总是小于（或者大于）它的父节点。

> 堆（英语：heap)是计算机科学中一类特殊的数据结构的统称
> 堆总是满足下列性质： 1. 堆中某个节点的值总是不大于或不小于其父节点的值； 2. 堆总是一棵完全二叉树
> 将根节点最大的堆叫做最大堆或大根堆，根节点最小的堆叫做最小堆或小根堆

> 若设二叉树的深度为h，除第 h 层外，其它各层 (1～h-1) 的结点数都达到最大个数，第 h 层所有的结点都连续集中在最左边，这就是完全二叉树。

**算法复杂度**

时间平均复杂度：O(nlog2^n) 最坏复杂度:O(nlog2^n) 最好复杂度: O(nlog2^n) 空间复杂度: O(1) 不稳定

**算法过程描述**

- <1>将初始待排序关键字序列(R1,R2….Rn)构建成大顶堆，此堆为初始的无序区；
- <2>将堆顶元素R[1]与最后一个元素R[n]交换，此时得到新的无序区(R1,R2,……Rn-1)和新的有序区(Rn),且满足R[1,2…n-1]<=R[n]；
- <3>由于交换后新的堆顶R[1]可能违反堆的性质，因此需要对当前无序区(R1,R2,……Rn-1)调整为新堆，然后再次将R[1]与无序区最后一个元素交换，得到新的无序区(R1,R2….Rn-2)和新的有序区(Rn-1,Rn)。不断重复此过程直到有序区的元素个数为n-1，则整个排序过程完成。

**堆排序步骤一**

构造初始堆。将给定无序序列构造成一个大顶堆（一般升序采用大顶堆，降序采用小顶堆)。

1. 假设给定无序序列结构如下:

![](/assets/images/al-sort-05.png)

2. 此时我们从最后一个非叶子结点开始（叶结点自然不用调整，第一个非叶子结点 arr.length/2-1=5/2-1=1，也就是下面的6结点），从左至右，从下至上进行调整。

![](/assets/images/al-sort-06.png)

3. 找到第二个非叶节点4，由于[4,9,8]中9元素最大，4和9交换。

![](/assets/images/al-sort-07.png)

4. 这时，交换导致了子根[4,5,6]结构混乱，继续调整，[4,5,6]中6最大，交换4和6。

![](/assets/images/al-sort-08.png)

**堆排序步骤二**

将堆顶元素与末尾元素进行交换，使末尾元素最大。然后继续调整堆，再将堆顶元素与末尾元素交换，得到第二大元素。如此反复进行交换、重建、交换。

1. 将堆顶元素9和末尾元素4进行交换

![](/assets/images/al-sort-09.png)

2. 重新调整结构，使其继续满足堆定义

![](/assets/images/al-sort-10.png)

3. 再将堆顶元素8与末尾元素5进行交换，得到第二大元素8.

![](/assets/images/al-sort-11.png)

后续过程，继续进行调整，交换，如此反复进行，最终使得整个序列有序

![](/assets/images/al-sort-12.png)

再简单总结下堆排序的基本思路：

+ 将无需序列构建成一个堆，根据升序降序需求选择大顶堆或小顶堆;

+ 将堆顶元素与末尾元素交换，将最大元素"沉"到数组末端;

+ 重新调整结构，使其满足堆定义，然后继续交换堆顶元素与当前末尾元素，反复执行调整+交换步骤，直到整个序列有序。

**代码实现**

```swift
// 堆排序
var len: Int = 0
func headSort(_ array: inout Array<NSInteger>) {

    // 构建大顶堆
    for i in (0...(array.count / 2 - 1)).reversed() {
        adjustHead(&array, i, array.count)
    }
    // 调整堆结构+交换堆顶元素与末尾元素
    for j in (1...(array.count - 1)).reversed() {
        swaps(&array, 0, j)         // 将堆顶元素与末尾元素进行交换
        adjustHead(&array, 0, j)    // 重新对堆进行调整
    }
}

// 调整大顶堆（仅是调整过程，建立在大顶堆已构建的基础上）
func adjustHead(_ array: inout Array<NSInteger>, _ index: Int, _ len: Int) {

    var index = index
    let temp = array[index] // 取出当前元素
    var k = 2 * index + 1
    while k < len {    // 从i结点的左子结点开始，也就是2i+1处开始
        if k + 1 < len && array[k] < array[k + 1] {     // 如果左子结点小于右子结点，k指向右子结点
            k += 1
        }
        if array[k] > temp { // 如果子节点大于父节点，将子节点值赋给父节点（不用进行交换）
            array[index] = array[k]
            index = k
        } else {
            break
        }
        k = k * 2 + 1
    }
    array[index] = temp   	// 将temp值放到最终的位置
}

// 交换两个数
func swaps<T>(_ chars: inout [T], _ s: Int, _ e: Int) {
    (chars[s], chars[e]) = (chars[e], chars[s])
}
```

## 归并排序

**归并排序Merge Sort**是建立在归并操作上的一种有效的排序算法。该算法是采用分治法（Divide and Conquer）的一个非常典型的应用。将已有序的子序列合并，得到完全有序的序列；即先使每个子序列有序，再使子序列段间有序。若将两个有序表合并成一个有序表，称为2-路归并。

**算法复杂度**

时间平均复杂度：O(nlog2^n) 最坏复杂度:O(nlog2^n) 最好复杂度: O(nlog2^n) 空间复杂度: O(n) 稳定

**算法过程描述**

- <1>把长度为n的输入序列分成两个长度为n/2的子序列；
- <2>对这两个子序列分别采用归并排序；
- <3>将两个排序好的子序列合并成一个最终的排序序列。

**图解归并排序**

![](/assets/images/al-sort-13.png)

![](/assets/images/al-sort-14.png)

**代码实现**

```swift
// 归并排序（MERGE-SORT）是利用归并的思想实现的排序方法，该算法采用经典的分治（divide-and-conquer）策略（
// 分治法将问题分(divide)成一些小的问题然后递归求解，而治(conquer)的阶段则将分的阶段得到的各答案"修补"在一起，即分而治之)。
func mergeSort(array: [Int]) -> [Int] {
    var helper = Array(repeating: 0, count: array.count)
    var array = array
    mergeSort(&array, &helper, 0, array.count - 1)
    return array
}

func mergeSort(_ array: inout [Int], _ helper: inout [Int], _ low: Int, _ high: Int) {
    guard low < high else {
        return
    }

    let middle = (high - low) / 2 + low
    mergeSort(&array, &helper, low, middle)
    mergeSort(&array, &helper, middle + 1, high)
    merge(&array, &helper, low, middle, high)
}

func merge(_ array: inout [Int], _ helper: inout [Int], _ low: Int, _ middle: Int, _ high: Int) {
    // copy both halves into a helper array
    for i in low ... high {
        helper[i] = array[i]
    }

    var helperLeft = low
    var helperRight = middle + 1
    var current = low

    // iterate through helper array and copy the right one to original array
    while helperLeft <= middle && helperRight <= high {
        if helper[helperLeft] <= helper[helperRight] {
            array[current] = helper[helperLeft]
            helperLeft += 1
        } else {
            array[current] = helper[helperRight]
            helperRight += 1
        }
        current += 1
    }

    // handle the rest
    guard middle - helperLeft >= 0 else {
        return
    }
    for i in 0 ... middle - helperLeft {
        array[current + i] = helper[helperLeft + i]
    }
}
```

## 快速排序

**快速排序Quick Sort**的基本思想：通过一趟排序将待排记录分隔成独立的两部分，其中一部分记录的关键字均比另一部分的关键字小，则可分别对这两部分记录继续进行排序，以达到整个序列有序。

**算法复杂度**

时间平均复杂度：O(nlog2^n) 最坏复杂度:O(n^2) 最好复杂度: O(nlog2^n) 空间复杂度: O(nlog2^n) 不稳定

**算法过程描述**

快速排序使用分治法来把一个串（list）分为两个子串（sub-lists）。具体算法描述如下：

- <1>从数列中挑出一个元素，称为 “基准”（pivot）；
- <2>重新排序数列，所有元素比基准值小的摆放在基准前面，所有元素比基准值大的摆在基准的后面（相同的数可以到任一边）。在这个分区退出之后，该基准就处于数列的中间位置。这个称为分区（partition）操作；
- <3>递归地（recursive）把小于基准值元素的子数列和大于基准值元素的子数列排序。

Swift代码实现

```swift
// 快速排序
func quickSort(_ array: [Int]) -> [Int] {

    guard array.count > 1 else {
        return array
    }
    let privot = array[array.count / 2]
    let left = array.filter { $0 < privot }
    let middle = array.filter { $0 == privot }
    let right = array.filter { $0 > privot }
    return quickSort(left) + middle + quickSort(right)
}
```

## 希尔排序

**希尔排序Shell Sort**1959年Shell发明，第一个突破O(n2)的排序算法，是简单插入排序的改进版。它与插入排序的不同之处在于，它会优先比较距离较远的元素。希尔排序又叫缩小增量排序。

![](/assets/images/al-sort-15.gif)

**算法复杂度**

时间平均复杂度：O(n^1.3) 最坏复杂度:O(n^2) 最好复杂度: O(n) 空间复杂度: O(1) 不稳定

**算法过程描述**

先将整个待排序的记录序列分割成为若干子序列分别进行直接插入排序，具体算法描述：

- <1>选择一个增量序列t1，t2，…，tk，其中ti>tj，tk=1；
- <2>按增量序列个数k，对序列进行k 趟排序；
- <3>每趟排序，根据对应的增量ti，将待排序列分割成若干长度为m 的子序列，分别对各子表进行直接插入排序。仅增量因子为1 时，整个序列作为一个表来处理，表长度即为整个序列的长度。

排序思路:

- 1.希尔排序可以理解为插入排序的升级版, 先将待排序数组按照指定步长划分为几个小数组
- 2.利用插入排序对小数组进行排序, 然后将几个排序的小数组重新合并为原始数组
- 3.重复上述操作, 直到步长为1时,再利用插入排序排序即可

C语言代码实现:

```c
int main()
{
    // 待排序数组
    int nums[5] = {3, 1, 2, 0, 3};
    // 0.计算待排序数组长度
    int len = sizeof(nums) / sizeof(nums[0]);

// 2.计算步长
    int gap = len / 2;
    do{
        //  1.从第一个元素开始依次取出所有用于比较元素
        for (int i = gap; i < len; i++)
        {
            // 2.遍历取出前面元素进行比较
            int j = i;
            while((j - gap) >= 0)
            {
                printf("%i > %i\n", nums[j - gap], nums[j]);
                // 3.如果前面一个元素大于当前元素,就交换位置
                if(nums[j - gap] > nums[j]){
                    int temp = nums[j];
                    nums[j] = nums[j - gap];
                    nums[j - gap] = temp;
                }else{
                    break;
                }
                j--;
            }
        }
        // 每个小数组排序完成, 重新计算步长
        gap = gap / 2;
    }while(gap >= 1);
}
```

Swift代码实现

```swift
// 希尔排序
func shellSort(_ array: inout Array<NSInteger>) {

    var gap = array.count / 2
    repeat {

        // 1.从第一个元素开始依次取出所有用于比较元素
        for i in gap..<array.count {

            // 2.遍历取出前面元素进行比较
            var j = i
            while j - gap >= 0 {

                if array[j - gap] > array[j] {
                    swaps(&array, j - gap, j)
                } else {
                    break
                }
                j -= 1
            }
        }
        // 每个小数组排序完成, 重新计算步长
        gap = gap / 2
    } while gap >= 1
}
// 交换两个数
func swaps<T>(_ chars: inout [T], _ s: Int, _ e: Int) {
    (chars[s], chars[e]) = (chars[e], chars[s])
}
```

## 排序实战

直接来看一道Facebook, Google, Linkedin都考过的面试题。

> 已知有很多会议，如果有这些会议时间有重叠，则将它们合并。
>  比如有一个会议的时间为3点到5点，另一个会议时间为4点到6点，那么合并之后的会议时间为3点到6点

解决算法题目第一步永远是把具体问题抽象化。这里每一个会议我们已知开始时间和结束时间，就可以写一个类来定义它：

```swift
public class MeetingTime {
  public var start: Int
  public var end: Int
  public init(_ start: Int, _ end: Int) {
    self.start = start
    self.end = end
  }
}
```

然后题目说已知有很多会议，就是说我们已知有一个MeetingTime的数组、所以题目就转化为写一个函数，输入为一个MeetingTime的数组，输出为一个将原数组中所有重叠时间都处理过的新数组。

```swift
func merge(meetingTimes: [MeetingTime]) -> [MeetingTime] {}
```

下面来分析一下题目怎么解。最基本的思路是遍历一次数组，然后归并所有重叠时间。举个例子：[[1, 3], [5, 6], [4, 7], [2, 3]]。这里我们可以发现[1, 3]和[2, 3]可以归并为[1, 3]，[5, 6]和[4, 7]可以归并为[5, 7]。所以这里就提出一个要求：要将所有**可能重叠的时间尽量放在一起**，这样遍历的时候可以就可以从前往后一个接着一个的归并。于是很自然的想到 -- 按照会议开始的时间排序。

Swift代码实现

```swift
public class MeetingTime {
    
    public var start: Int
    public var end: Int
    public init(_ start: Int, _ end: Int) {
        self.start = start
        self.end = end
    }
}

public class Meeting {
    
    func merge(meetingTimes: [MeetingTime]) -> [MeetingTime] {
        
        var meetings = meetingTimes
        // 处理特殊情况
        guard meetingTimes.count > 1 else {
            return meetingTimes
        }
        
        // 排序
        meetings = meetings.sorted() {
            if $0.start != $1.start {
                return $0.start < $1.start
            } else {
                return $0.end < $1.end
            }
        }
        
        // 新建结果数组
        var res = [MeetingTime]()
        res.append(meetings[0])
        
        // 遍历排序后的原数组，并与结果数组归并
        for i in 1..<meetings.count {
            let last = res[res.count - 1]
            let current = meetingTimes[i]
            if current.start > last.end {
                res.append(current)
            } else {
                last.end = max(last.end, current.end)
            }
        }
        
        return res
    }
}
```

## 展望

排序在Swift中的应用场景很多，比如tableView中对于dataSource的处理。当然很多时候，排序都是和搜索，尤其是二分搜索配合使用。下期探讨搜索的时候，会对排序进行进一步拓展。

[源码地址](<https://github.com/Jovins/Algorithm>)