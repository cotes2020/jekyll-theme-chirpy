# alist = [54,26, 93,17, 77,31, 44,55, 20]
# alist = [19, 1, 9, 7, 3, 10, 13, 15, 8, 12]
alist = [18, 54, 296, 13, 17, 8, 6]
n = len(alist)


def bubbleSort(alist):
    print(alist)
    compare_num = 0
    change_num = 0
    for passnum in range(len(alist) - 1, 0, -1):
        # print(alist[n])
        for i in range(passnum):
            compare_num += 1
            if alist[i] > alist[i + 1]:
                temp = alist[i]
                alist[i] = alist[i + 1]
                alist[i + 1] = temp
                change_num += 1
        # print(alist)
    print("compare_num", compare_num)
    print("change_num", change_num)


# print(" =============== bubbleSort ")
# compare_pre=n*n/2 - n/2
# print("n:", n, " time: n^2/2 - n/2", compare_pre)
# bubbleSort(alist)
# # print(alist)


def selectionSort(alist):
    compare_num = 0
    change_num = 0
    for fillslot in range(len(alist) - 1, 0, -1):
        positionOfMax = 0
        for location in range(1, fillslot + 1):
            # print(alist)
            # print(alist[location])
            compare_num += 1
            if alist[location] > alist[positionOfMax]:
                positionOfMax = location
        temp = alist[fillslot]
        alist[fillslot] = alist[positionOfMax]
        alist[positionOfMax] = temp
        change_num += 1
    print("compare_num", compare_num)
    print("change_num", change_num)


# print(" =============== selectionSort ")
# compare_pre=n*n/2 - n/2
# print("n:", n, " time: n^2/2 - n/2", compare_pre)
# selectionSort(alist)
# # print(alist)


def shellSort(alist):
    sublistcount = len(alist) // 2
    while sublistcount > 0:
        # print("sublistcount:", sublistcount)
        for startposition in range(sublistcount):
            # print("startposition:", startposition)
            gapInsertionSort(alist, startposition, sublistcount)
            # gapInsertionSort(alist, 0, 4)
            # gapInsertionSort(alist, 1, 4)
            # gapInsertionSort(alist, 2, 4)
            # gapInsertionSort(alist, 3, 4)
        print("After increments of size", sublistcount, "The list is", alist)
        sublistcount = sublistcount // 2


def gapInsertionSort(alist, start, gap):
    # gapInsertionSort(alist, 1, 4)
    for i in range(start + gap, len(alist), gap):
        # for i in range(5, 9, 3): 5678
        position = i
        currentvalue = alist[i]
        while position >= gap and alist[position - gap] > currentvalue:
            alist[position] = alist[position - gap]
            position = position - gap
        alist[position] = currentvalue


# print(" =============== shellSort ")
# shellSort(alist)
# print(alist)


def mergeSort(alist):
    print("Splitting ", alist)
    if len(alist) > 1:
        mid = len(alist) // 2
        lefthalf = alist[:mid]
        righthalf = alist[mid:]

        mergeSort(lefthalf)
        mergeSort(righthalf)

        i = 0
        j = 0
        k = 0
        while i < len(lefthalf) and j < len(righthalf):
            if lefthalf[i] <= righthalf[j]:
                alist[k] = lefthalf[i]
                i = i + 1
            else:
                alist[k] = righthalf[j]
                j = j + 1
            k = k + 1

        while i < len(lefthalf):
            alist[k] = lefthalf[i]
            i = i + 1
            k = k + 1

        while j < len(righthalf):
            alist[k] = righthalf[j]
            j = j + 1
            k = k + 1
    print("Merging ", alist)


# print(" =============== mergeSort ")
# alist = [21, 1, 26, 45, 29, 28, 2, 9, 16, 49, 39, 27, 43, 34, 46, 40]
# mergeSort(alist)
# print(alist)


def quickSort(alist):
    quickSortHelper(alist, 0, len(alist) - 1)


def quickSortHelper(alist, first, last):
    n = 0
    if first < last:
        n += 1
        splitpoint = partition(alist, first, last)
        quickSortHelper(alist, first, splitpoint - 1)
        quickSortHelper(alist, splitpoint + 1, last)


def partition(alist, first, last):
    print(alist)
    pivotvalue = alist[first]

    leftmark = first + 1
    rightmark = last

    done = False
    while not done:
        while leftmark <= rightmark and alist[leftmark] <= pivotvalue:
            leftmark = leftmark + 1

        while leftmark <= rightmark and alist[rightmark] >= pivotvalue:
            rightmark = rightmark - 1

        if leftmark > rightmark:
            done = True
        else:
            temp = alist[leftmark]
            alist[leftmark] = alist[rightmark]
            alist[rightmark] = temp

    temp = alist[first]
    alist[first] = alist[rightmark]
    alist[rightmark] = temp
    return rightmark


# print(" =============== quickSort ")
# alist = [14, 17, 13, 15, 19, 10, 3, 16, 9, 12]
# quickSort(alist)
# print(alist)


# -------------------------------------- Excercises -------------------------------------------------
# Set up a random experiment to test the difference between a sequential search and a binary search on a list of integers.


# -------------------------------------- Excercises -------------------------------------------------
# Use the binary search functions given in the text (recursive and iterative).
# Generate a random, ordered list of integers and do a benchmark analysis for each one.
# What are your results? Can you explain them?


# -------------------------------------- Excercises -------------------------------------------------
# Implement the binary search using recursion without the slice operator. Recall that you will need to pass the list along with the starting and ending index values for the sublist. Generate a random, ordered list of integers and do a benchmark analysis.


# -------------------------------------- Excercises -------------------------------------------------
# Implement the len method (__len__) for the hash table Map ADT implementation.


# -------------------------------------- Excercises -------------------------------------------------
# Implement the in method (__contains__) for the hash table Map ADT implementation.


# -------------------------------------- Excercises -------------------------------------------------
# How can you delete items from a hash table that uses chaining for collision resolution? How about if open addressing is used? What are the special circumstances that must be handled? Implement the del method for the HashTable class.


# -------------------------------------- Excercises -------------------------------------------------
# In the hash table map implementation, the hash table size was chosen to be 101. If the table gets full, this needs to be increased. Re-implement the put method so that the table will automatically resize itself when the loading factor reaches a predetermined value (you can decide the value based on your assessment of load versus performance).


# -------------------------------------- Excercises -------------------------------------------------
# Implement quadratic probing as a rehash technique.


# -------------------------------------- Excercises -------------------------------------------------
# Using a random number generator, create a list of 500 integers. Perform a benchmark analysis using some of the sorting algorithms from this chapter. What is the difference in execution speed?


# -------------------------------------- Excercises -------------------------------------------------
# Implement the bubble sort using simultaneous assignment.


# -------------------------------------- Excercises -------------------------------------------------
# A bubble sort can be modified to “bubble” in both directions. The first pass moves “up” the list, and the second pass moves “down.” This alternating pattern continues until no more passes are necessary. Implement this variation and describe under what circumstances it might be appropriate.


# -------------------------------------- Excercises -------------------------------------------------
# Implement the selection sort using simultaneous assignment.


# -------------------------------------- Excercises -------------------------------------------------
# Perform a benchmark analysis for a shell sort, using different increment sets on the same list.


# -------------------------------------- Excercises -------------------------------------------------
# Implement the mergeSort function without using the slice operator.


# -------------------------------------- Excercises -------------------------------------------------
# One way to improve the quick sort is to use an insertion sort on lists that have a small length (call it the “partition limit”). Why does this make sense? Re-implement the quick sort and use it to sort a random list of integers. Perform an analysis using different list sizes for the partition limit.


# -------------------------------------- Excercises -------------------------------------------------
# Implement the median-of-three method for selecting a pivot value as a modification to quickSort. Run an experiment to compare the two techniques.
