
[toc]

---

# Back-End Developer intern

3 question:

---

## 1. Merge sorted arrays

Input:
arr1[] = { 1, 3, 4, 5},
arr2[] = {2, 4, 6, 8}
Output: arr3[] = {1, 2, 3, 4, 4, 5, 6, 8}



```py
a = [1,2,3,0,0,0]
m = 3
b = [2,5,6]
n = 3

a = [0]
m = 0
b = [2,5,6]
n = 3


# ----------------- solution 1: -----------------
# add 2 lists and sort
def merge(nums1, m, nums2, n):
    nums1[m:] = nums2
    nums1.sort()
# Runtime: 28 ms, faster than 96.18% of Python3 online submissions for Merge Sorted Array.
# Memory Usage: 14.2 MB, less than 59.72% of Python3 online submissions for Merge Sorted Array.


# ----------------- solution 2: -----------------
# iterate the biggest on to the end
# once one finish, add the rest.
def merge(nums1, m, nums2, n):
    listL = m+n-1
    while m > 0 and n > 0:
        if nums1[m-1] > nums2[n-1]:
            nums1[listL] = nums1[m-1]
            # print("nums1 number", nums1[listL])
            m -= 1
        else:
            nums1[listL] = nums2[n-1]
            # print("nums2 number", nums1[listL])
            # print("n:", n)
            n -= 1
        listL -= 1

    # m <= 0: when nums1 finsih
    # n != 0 : rest of put to the head nums1[:n]
    # n <= 0: when nums2 finsih
    # m != 0 : keep those m
    if m <= 0:  
        nums1[:n] = nums2[:n]
    print(nums1)
# Runtime: 36 ms, faster than 62.64% of Python3 online submissions for Merge Sorted Array.
# Memory Usage: 14.2 MB, less than 59.72% of Python3 online submissions for Merge Sorted Array.



# ----------------- solution 2: -----------------
# iterate the biggest on to the end
# once one finish, add the rest.
def merge(nums1, m, nums2, n):
    listL = m + n - 1
    m, n = m-1, n-1
    while n>=0 and m>=0:
        if nums1[m] > nums2[n]:
            nums1[listL] = nums1[m]
            m -= 1
        else:
            nums1[listL] = nums2[n]
            n -= 1
        listL -= 1
    if n>-1:
        nums1[0:n+1] = nums2[0:n+1]
    print(nums1)
merge(a, m, b, n)




# ----------------- solution 3: -----------------
# (O(n1 * n2) Time
# O(1) Extra Space)
# Create an array arr3[] of size n1 + n2.
# Copy all n1 elements of arr1[] to arr3[]
# Traverse arr2[] and one by one insert elements (like insertion sort) of arr3[] to arr1[]. This step take O(n1 * n2) time.
# We have discussed implementation of above method in Merge two sorted arrays with O(1) extra space
def merge(nums1, m, nums2, n):
    nums3 = [0] * (m + n)
    i,j,k = 0,0,0
    # Traverse both array
    while i < m and j < n:
        # print(i,j)
        # iterate the biggest on to the end
        if nums1[i] < nums2[j]:
            nums3[k] = nums1[i]
            k = k + 1
            i = i + 1
            # print(nums3)
        else:
            nums3[k] = nums2[j]
            k = k + 1
            j = j + 1
            # print(nums3)
    if i >= m:
        # print("nums1 end")
        nums3[i+j:] = nums2[j:len(nums2)-n]
    if j >= n:
        # print("nums2 end")
        nums3[i+j:] = nums1[i:len(nums1)-m]
    print(nums3)
```


---

## 2. Rearrange an array in order – smallest, largest, 2nd smallest, 2nd largest, ..

Examples:

Input : arr[] = [5, 8, 1, 4, 2, 9, 3, 7, 6]

Output :arr[] = {1, 9, 2, 8, 3, 7, 4, 6, 5}

Input : arr[] = [1, 2, 3, 4]

Output :arr[] = {1, 4, 2, 3}


```py
def rearrangeArray(arr, n) :

    # Sorting the array elements
    arr.sort()

    # To store modified array
    tempArr = [0] * (n + 1)

    # Adding numbers from sorted array to new array accordingly
    ArrIndex = 0

    # Traverse from begin and end simultaneously  
    i = 0
    j = n-1

    while(i <= n // 2 or j > n // 2 ) :
        tempArr[ArrIndex] = arr[i]
        ArrIndex += 1
        tempArr[ArrIndex] = arr[j]
        ArrIndex += 1
        i += 1
        j -= 1

    # Modifying original array
    for i in range(0, n) :
        arr[i] = tempArr[i]


```


---

## 3. Print all triplets with given sum

find triplets in array whose sum is equal to a given number.

```bash
Examples:

Input: arr[] = {0, -1, 2, -3, 1}
        sum = -2
Output:  0 -3  1
        -1  2 -3
# 0 + (-3) + 1 = -2
# (-1) + 2 + (-3) = -2

Input: arr[] = {1, -2, 1, 0, 5}
       sum = 0
Output: 1 -2  1
# 1 + (-2) + 1 = 0
```

```py

# ------------solution 1: loop ljk------------
def findTriplets(arr, n, sum):
    for i in range(0, n - 2):  
        for j in range(i + 1, n - 1):  
            for k in range(j + 1, n):
                if (arr[i] + arr[j] + arr[k] == sum):  
                    print(arr[i], " ", arr[j], " ", arr[k])
# Time Complexity: O(n3).
# As three nested for loops have been used.
# Auxiliary Space: O(1).
# As no data structure has been used for storing values.




# ------------solution 2: Hashing------------
import math as mt
# function to print triplets with given sum
def findTriplets(arr, n, Sum):
    for i in range(n - 1):
        # Find all pairs with Sum equals
        # to "Sum-arr[i]"
        s = dict()
        for j in range(i + 1, n):
            x = Sum - (arr[i] + arr[j])
            if x in s.keys():
                print(x, arr[i], arr[j])
            else:
                s[arr[j]] = 1



# ------------solution 3: Two-pointer------------
def triplets(target, d):
    # if less than 3 ele, nothing to count
    len_a = len(a)
    if len(a) < 3:
        return 0

    # Remove elements in arr larger than x
    d = [i for i in a if i < x]

    d.sort()
    count = 0
    # every number in the list
    for i in range(0, len(d) - 2):
        head = i
        mid = head + 1
        tail = len(d) - 1
        # when mid tail wasnot same:
        while(mid < tail):
            print(d[i], d[mid], d[tail])
            sum = d[head] + d[mid] + d[tail]
            # 1,2,3,4,5
            # 1,2,5
            # target=8
            # 1,3,4
            if (sum == target):
                mid +=1
                tail -= 1
                count += 1
                print("yes, sum: ", sum)
            # 1,2,3,4
            # 1,2,4
            # target=7
            # 1,3,4
            elif (sum < target):
                mid +=1
            # 1,2,3,4,5,98
            # 1,2,98
            # target=7
            # 1,2,5
            else:
                tail -= 1
    print(count)
    return count

d = [1,2,3,4,5]
target = 8

count = triplets(target,d)
print(count)




# ------------solution 4: steven------------
def triplets(a, x):
    # if less than 3 ele, nothing to count
    len_a = len(a)
    if len(a) < 3:
        return 0

    # Remove elements in arr larger than x
    a = [i for i in a if i < x]
    a.sort()

    # i is element 1, j is 2, k is 3
    i, j = 0, 1
    count = 0
    while i <= len_a - 2:
        # Iterate through k while i, j remain static
        for k in range(j + 1, len(a)):
            if a[i] + a[j] + a[k] <= x:
                print(a[i], a[j], a[k])
                count += 1
        # increment j since we exhausted k
        j += 1
        # increment i and j once j hits end of arr
        if j == len_a - 1:
            i += 1
            j = i + 1
        # once i hits end, we return all the elements we counted, end of loop
        if i == len_a - 2:
            return len(triplets)


# -----------

def triplets(target, d):
    d.sort()
    count = 0
    # every number in the list
    for i in range(0, len(d) - 2):
        head = i
        mid = head + 1
        tail = len(d) - 1
        # when mid tail wasnot same:
        while(mid < tail):
            print(d[i], d[mid], d[tail])
            sum = d[head] + d[mid] + d[tail]
            # 1,2,3,4,5
            # 1,2,5
            # target=8
            # 1,3,4
            if (sum == target):
                mid +=1
                tail -= 1
                count += 1
                print("yes, sum: ", sum)
            # 1,2,3,4
            # 1,2,4
            # target=7
            # 1,3,4
            elif (sum < target):
                mid +=1
            # 1,2,3,4,5,98
            # 1,2,98
            # target=7
            # 1,2,5
            else:
                tail -= 1
    print(count)
    return count

d = [1,2,3,4,5]
target = 8

count = triplets(target,d)
print(count)
```

---


# Back-End Developer

3 question:


---

## 1. Partitioning Array


![Screen Shot 2020-12-12 at 21.27.29](https://i.imgur.com/cdore1Y.png)

![Screen Shot 2020-12-12 at 21.27.48](https://i.imgur.com/DV363VC.png)


```py

from collections import Counter

def solve(k, nums):
    if k == 0 or len(nums) == 0:
        return "YES"
    if len(nums)%k != 0:
        return "NO"
    counter = Counter(nums)
    print(counter)
    for entry in counter:
        print(entry)
        print(counter[entry])
        if counter[entry] > len(nums)/k:
            return "NO"
    return "YES"


if __name__ == '__main__':
    print(solve(2,[1,2,3,4])) #yes
    print(solve(2,[1,2,3,3])) #yes
    print(solve(3,[1,2,3,4])) #no
    print(solve(3,[3,3,3,6,6,6,9,9,9]))#yes
    print(solve(1,[]))#yes
    print(solve(1,[1]))#yes
    print(solve(2,[1,2]))#yes
```
---


## 2. calculate the get response


![Screen Shot 2020-12-12 at 21.54.06](https://i.imgur.com/wRBMNHM.png)

```
unicomp6.unicomp.net - - [01/Jul/1995:00:00:06 -0400] "GET /shuttle/countdown/ HTTP/1.0" 200 5001
unicomp6.unicomp.net - - [01/Jul/1995:00:00:06 -0400] "GET /shuttle/countdown/ HTTP/1.0" 200 5002
unicomp6.unicomp.net - - [01/Jul/1995:00:00:06 -0400] "GET /shuttle/countdown/ HTTP/1.0" 200 3988
```


```py
# read the string filename
filename = "/Users/luo/Downloads/GitHub/ocholuo.github.io/_posts/Lab/interview/num.txt"
print(filename)

# open the file
with open(filename) as file:
    bcount = 0
    bsum = 0
    for line in file:
        # find the bytes amount for each line
        bindex = line.rfind(" ") + 1
        bnum = int(line[bindex:])
        if(bnum > 5000 ):
            bcount += 1
            bsum += bnum
        print(bcount)
        print(bsum)

    file = open("bytes_" + filename, "w")
    file.write( str(bcount)+ "\n" + str(bsum) + "\n")
    file.close()
```



---




## 3. shopper the list


![Screen Shot 2020-12-12 at 22.16.47](https://i.imgur.com/VKP2hK7.png)


```
#!/bin/python3

import math
import os
import random
import re
import sys


#
# Complete the 'getNumberOfOptions' function below.
#
# The function is expected to return a LONG_INTEGER.
# The function accepts following parameters:
#  1. INTEGER_ARRAY priceOfJeans
#  2. INTEGER_ARRAY priceOfShoes
#  3. INTEGER_ARRAY priceOfSkirts
#  4. INTEGER_ARRAY priceOfTops
#  5. INTEGER budgeted
#

# def findNumber(targetN, targetL):
#     listL = len(targetL)

#     # if only one number, add 1 or 0 after compare it
#     if(listL==1):
#        if(targetN>=targetL[0]): return 1
#        else: return 0

#     # use point, mid, tail to check
#     # how many number is smaller then taretN
#     point=0
#     mid = listL//2
#     tail = listL-1
#     while ( point < mid and mid < tail):
#         print("point, mid, tail: ", targetL[point], targetL[mid], targetL[tail])
#         print("targetL[mid]: ", targetL[mid])
#         print("targetN: ", targetN)

#         # exclusion to same time
#         if(targetL[point]==targetN):
#             print("findNumber: ", point+1)
#             return point + 1
#         if(targetL[tail]==targetN):
#             print("findNumber: ", tail+1)
#             return tail + 1
#         if(targetL[mid]==targetN):
#             print("findNumber: ", mid+1)
#             return mid + 1

#         # comparing:
#         if(targetL[mid]>targetN):
#             print("mid>targetN")
#             tail = mid
#             mid += (tail - point)//2
#         if(targetL[mid]<targetN):
#             # print("mid<targetN")
#             point = mid
#             mid += (tail - point)//2
#     print("---------", point+1, " numbers are smaller than ", targetN)
#     return point + 1


def getNumberOfOptions(priceOfJeans, priceOfShoes, priceOfSkirts, priceOfTops, budgeted):
    # Write your code here
    # print(priceOfJeans, priceOfShoes, priceOfSkirts, priceOfTops)

    # Remove elements in arr larger than budgeted
    # if any of them become empty, return 0
    priceOfJeans = [i for i in priceOfJeans if i < budgeted]
    if(priceOfJeans==[]): return 0
    priceOfShoes = [i for i in priceOfShoes if i < budgeted]
    if(priceOfShoes==[]): return 0
    priceOfSkirts = [i for i in priceOfSkirts if i < budgeted]
    if(priceOfSkirts==[]): return 0
    priceOfTops = [i for i in priceOfTops if i < budgeted]
    if(priceOfTops==[]): return 0

    priceOfJeans.sort()
    priceOfShoes.sort()
    priceOfSkirts.sort()
    priceOfTops.sort()
    # check:
    # print(priceOfJeans)
    # print(priceOfShoes)
    # print(priceOfSkirts)
    # print(priceOfTops)


    # --------------- solution 1 ---------------
    count = 0
    for a in priceOfJeans:
        for b in priceOfShoes:
            for c in priceOfSkirts:
                for d in priceOfTops:
                    if(a + b + c + d <= budgeted):
                        count +=1
    return count

    # --------------- solution 2 ---------------
    # count = 0
    # print("test")
    # for a in priceOfJeans:
    #     # print("testa")
    #     for b in priceOfShoes:
    #         print("testB")
    #         for c in priceOfSkirts:
    #             print("a,b,c: " , a,b,c)
    #             targetN = budgeted-a-b-c
    #             print("targetN: ", targetN)
    #             numD = findNumber(targetN, priceOfTops)
    #             print("num is: ", numD)
    #             count+=numD
    #             print("---------------count is: ", count)
    # print("count: ", count)
    # return count





if __name__ == '__main__':
```





```py






def findNumber(targetN, targetL):
    listL = len(targetL)

    if(listL==1):
        if(targetN>=targetL[0]): return 1
        else: return 0

    point=0
    mid = listL//2
    tail = listL-1
    a=0
    # while ( a < 2 ):
    while ( point < mid and mid < tail):
        print("point, mid, tail: ", targetL[point], targetL[mid], targetL[tail])
        print("targetL[mid]: ", targetL[mid])
        print("targetN: ", targetN)

        if(targetL[mid]==targetN):
            print("findNumber: ", mid+1)
            return mid + 1
            # print("point, mid, tail: ", targetL[point], targetL[mid], targetL[tail])
        if(targetL[tail]==targetN):
            print("findNumber: ", tail+1)
            return tail + 1
        if(targetL[point]==targetN):
            print("findNumber: ", point+1)
            return point + 1

        if(targetL[mid]>targetN):
            print("mid>targetN")
            tail = mid
            mid += (tail - point)//2
            # print("point, mid, tail: ", targetL[point], targetL[mid], targetL[tail])
        if(targetL[mid]<targetN):
            # print("mid<targetN")
            point = mid
            mid += (tail - point)//2
            print(point,mid,tail)
            # print("point, mid, tail: ", targetL[point], targetL[mid], targetL[tail])
        a+=1

    print("findNumber: ", point+1)
    return point + 1

# listL = [24,35,67,68,69,70,87,88,90]
# targetN = 90
# count = findNumber(targetN, listL)


# listL = [1,2,3]
# targetN = 3
# count = findNumber(targetN, listL)


# listL = [4]
# targetN = 3
# count = findNumber(targetN, listL)


def getNumberOfOptions(priceOfJeans, priceOfShoes, priceOfSkirts, priceOfTops, budgeted):
    # Write your code here
    # print(priceOfJeans, priceOfShoes, priceOfSkirts, priceOfTops)

    # Remove elements in arr larger than budgeted
    priceOfJeans = [i for i in priceOfJeans if i < budgeted]
    if(priceOfJeans==[]): return 0
    priceOfShoes = [i for i in priceOfShoes if i < budgeted]
    if(priceOfShoes==[]): return 0
    priceOfSkirts = [i for i in priceOfSkirts if i < budgeted]
    if(priceOfSkirts==[]): return 0
    priceOfTops = [i for i in priceOfTops if i < budgeted]
    if(priceOfTops==[]): return 0

    priceOfJeans.sort()
    priceOfShoes.sort()
    priceOfSkirts.sort()
    priceOfTops.sort()

    print(priceOfJeans)
    print(priceOfShoes)
    print(priceOfSkirts)
    print(priceOfTops)

    count = 0
    print("test")
    for a in priceOfJeans:
        # print("testa")
        for b in priceOfShoes:
            print("testB")
            for c in priceOfSkirts:
                print("a,b,c: " , a,b,c)
                targetN = budgeted-a-b-c
                print("targetN: ", targetN)
                numD = findNumber(targetN, priceOfTops)
                print("num is: ", numD)
                count+=numD
                print("---------------count is: ", count)
    print("count: ", count)
    return count


priceOfJeans = [1]
priceOfShoes = [4]
priceOfSkirts = [3]
priceOfTops = [1]
budgeted = 3

count = getNumberOfOptions(priceOfJeans, priceOfShoes, priceOfSkirts, priceOfTops, budgeted)
print("count: ", count)


```







.
