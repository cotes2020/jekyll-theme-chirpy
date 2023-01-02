---
title: 遇到的 Interview Quetsions
date: 2022-02-28 11:11:11 -0400
categories: [00CodeNote]
tags: [makefile]
math: true
image:
---

- [Question during interview](#question-during-interview)
  - [AWS](#aws)
    - [Streaming packet processing](#streaming-packet-processing)
    - [calculate number of 1s in the binary form.](#calculate-number-of-1s-in-the-binary-form)
    - [1 亚麻order](#1-亚麻order)
  - [Apple](#apple)
    - [2022-2](#2022-2)
    - [encrypt the ebs by default](#encrypt-the-ebs-by-default)
  - [IBM](#ibm)
    - [1. Merge sorted arrays](#1-merge-sorted-arrays)
    - [2. Rearrange an array in order – smallest, largest, 2nd smallest, 2nd largest, ..](#2-rearrange-an-array-in-order--smallest-largest-2nd-smallest-2nd-largest-)
    - [3. Print all triplets with given sum](#3-print-all-triplets-with-given-sum)
    - [1. Partitioning Array](#1-partitioning-array)
    - [2. calculate the get response](#2-calculate-the-get-response)
    - [3. shopper the list](#3-shopper-the-list)
  - [redfine](#redfine)
    - [Implement a RESTful API](#implement-a-restful-api)

---

# Question during interview

---

## AWS

---


### Streaming packet processing

receive and read the offset with data

start read with 0.

A packet is defined with two elements

- {offset:1, data [12, 4]}
- {offset:0, data [0]}


```java
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;

public class Amazon {
    // value
    private HashMap<Integer, int[]> map = new HashMap<>();
    private int offset = 0;
    private int index = 0;

    public void receive(int offsetnumber, int[] data) {
        // check the parameter
        // start it
        map.put(offsetnumber, data);
        System.out.println("adding: key: " + offsetnumber + ", value: " + Arrays.toString(data));
    }

    public void read(){
        Object ans;
        ArrayList<Object> res = readdata();
        if((Boolean) res.get(0)) ans = res.get(1);
        else ans = null;
        System.out.println(ans);
    }

    public ArrayList<Object> readdata(){
        ArrayList<Object> res = new ArrayList<Object>();
        if(map.containsKey(offset)) {
            res.add(true);
            int[] datas = map.get(offset);
            int n = datas.length;
            if(n==1){
                res.add(datas[0]);
                // System.out.println("offset: " + offset + ", one value: " + ans);
                offset++;
                return res;
            }
            res.add(datas[index]);
            if(index == n-1) {
                // System.out.println("offset: " + offset + ", last value: " + ans);
                index=0;
                offset++;
                return res;
            }
            // System.out.println("offset: " + offset + ", current value: " + ans);
            index++;
            return res;
        }
        // System.out.println("The offset is not stored: " + offset);
        res.add(false);
        return res;
    }

    public static void main(String[] args) {
        Amazon run = new Amazon();
        run.read();

        int offsetnumber = 1;
        int[] data = {12, 4};
        run.receive(offsetnumber, data);

        run.read();

        int offsetnumber2 = 0;
        int[] data2 = {100};
        run.receive(offsetnumber2, data2);

        run.read();
        run.read();
        run.read();
        run.read();
        run.read();
        run.read();
    }
}
```


---

###  calculate number of 1s in the binary form.

```py
numinput = int(raw_input())
calnum= str(bin(numinput))
num=calnum.count('1')
print(num)


def NumberOf1(self, n):
    if n >= 0:
        return bin(n).count('1')
    else:
        return bin(n & 0xffffffff).count('1')

```

---

### 1 亚麻order

亚麻的每个订单第一段为认证号，后面接纯字母代表prime order； 后面接纯数字代表non-prime order。要求给prime order 按照字典顺序排在前面，non-prime order按照其原始顺序加到队尾。


```java
// Example:
// Input:
// {
//     ["a1 9 2 3 1"],
//     ["g1 act car"],
//     ["zo4 4 7"],
//     ["ab1 off key dog"],
//     ["a8 act zoo"]
// }

// Output:
// {
//     ["g1 act car"],
//     ["a8 act zoo"],
//     ["ab1 off key dog"],
//     ["a1 9 2 3 1"],
//     ["zo4 4 7"]
// }


import java.util.List;
import java.util.*;
// import java.util.Stack;

public class solution200927{


    // solution
    public String[] reorderLogFiles(String[] logs) {

        // compare(log1, log2)
        // (text, text):
        // <0; = log2, log1
        // >0; = log1, log2
        // (text, num): nochange -1 <0
        // (num, text): change 1 > 0

        Comparator<String> myComp = new Comparator<String>() {

            @Override
            public int compare(String log1, String log2) {
                // split each log into two parts: <identifier, content>
                String[] split1 = log1.split(" ", 2); // before index 2, split by " "
                String[] split2 = log2.split(" ", 2);

                boolean isDigit1 = Character.isDigit(split1[1].charAt(0));
                boolean isDigit2 = Character.isDigit(split2[1].charAt(0));

                // case 1). both logs are letter-logs
                if (!isDigit1 && !isDigit2) {
                    // first compare the content
                    int alphcompare = split1[1].compareTo(split2[1]);
                    if (alphcompare != 0){
                        return alphcompare;
                        // > 0 split2, split1
                        // < 0 split1, split2
                    }
                    // logs of same content, compare the identifiers
                    // 2nd part same, compare first part
                    return split1[0].compareTo(split2[0]);
                }

                // case 2). one of logs is digit-log
                if (!isDigit1 && isDigit2)
                    // the letter-log comes before digit-logs
                    return -1;
                else if (isDigit1 && !isDigit2)
                    return 1;
                else
                    // case 3). both logs are digit-log
                    return 0;
            }
        };

        Arrays.sort(logs, myComp);
        return logs;
    }



    // my
    String alph = "abcdefghijklmnopqrstuvwxyz";

    public boolean checkprime(String order){
        int startindex = order.indexOf(" ");
        String startwrd = order.substring(startindex+1, startindex+2);
        if(alph.indexOf(startwrd)==-1){
            System.out.println("not prime");
            return false;
        }
        else{
            System.out.println("prime");
            return true;
        }
    }


    public List<String> compare(List<String> result){
        int n = result.size()-1;
        for(int i = 0; i < n-1 ; ++i) {
            for (int j = i + 1; j < n; ++j) {
            String currOrder = result.get(i);
            if(result.get(i).compareTo(result.get(j)) > 0){
                String temp = result.get(i);
                result.set(i) = result.get(j);
                result.set(i+1) = result.get(i);
            }
        }
        return List<String> result;
    }

    List<String> prioritizedOrders(int numOrders, List<String> orderList) {
        List<String> result = new ArrayList<String>();
        for(int i=0; i<numOrders; i++) {
            String currstr = orderList.get(i);
            if(Boolean.TRUE.equals(checkprime(currstr))){
                result.add(currstr);
            }
        }

        result = compare(result);
        // for(String ans : result){
        //     System.out.println(ans);
        // }
        return result;
    }


    public static void main(String[] args) {
        List<String> orderList = Arrays.asList("a abc", "a 12", "a echo");
        int numOrders = orderList.size();

        solution200927 pr = new solution200927();

        System.out.println("hello");

        List<String> result = pr.prioritizedOrders(numOrders, orderList);
        // System.out.println("result is: " + result);

    }
}
```


python solution


```py
numOrders= 6
orderList = [
  ["zld","93","12"],
  ["fp","kindle","book"],
  ["10a","echo","show"],
  ["17g","12","25","6"],
  ["ab1","kindle","book"],
  ["125","echo","dot","second","generation"]
]

# numOrders= 4
# orderList = [
#   ["mi2","jog","mid","pet"],
#   ["wz3","34","54","398"],
#   ["al","alps","cow","bar"],
#   ["x4","45","21","7"]
# ]



# Divide into separate list O(n)
nonprimelist, primelist = [],[]
for i in orderList:
  if (''.join(i[1:])).isdigit():
    nonprimelist.append(i)
  else:
    primelist.append(i)

# Sorting with multiple keys O(n*log(n))
primelist.sort(key=lambda x:(' '.join(x[1:]), x[0]))
print(primelist + nonprimelist)
```


---

## Apple


### 2022-2

I will give you a tree. I'll give you an input
- For every input will be in lines every line has two numbers, a and b.
- So whenever you receive a line like that means that there is node a connected to node b.
- That's how I will describe the three to you.
- I will guarantee to you that this is a three it's not internal graph.
- I want you to write the system that will answer quesryz of the form. What is the distance if I want to go from node A to node B?
- Because it's tree you're gonna hear that every node is connected to every other node.
- so I want you to write efficient code that will help to answer a query just like that. The tree can have 1000s of nodes and the queries that we need to ask maybe like millions or billions of queries, so I want something fast.

the input is an array of pairs
- input (1,3),(1,4),(1,5),(3,6),(5,8)
- query: distance from 1 to 8.


```java
class Node {
    int data;
    Node left = null, right = null;
    Node(int data) {
        this.data = data;
    }
}

class Main {

    public static boolean isNodePresent(Node root, Node node) {
        // base case
        if (root == null) {
            return false;
        }
        // if the node is found, return true
        if (root == node) {
            return true;
        }
        return isNodePresent(root.left, node) || isNodePresent(root.right, node);
    }


    public static int findLevel(Node root, Node node, int level) {
        // base case
        if (root == null) return Integer.MIN_VALUE;

        // return level if the node is found
        if (root == node) return level;
        // search node in the subtree
        int left = findLevel(root.left, node, level + 1);
        if (left != Integer.MIN_VALUE) return left;
        return findLevel(root.right, node, level + 1);
    }

    // Function to find the lowest common ancestor of given nodes `x` and `y`,
    // where both `x` and `y` are present in the binary tree.
    public static Node findLCA(Node root, Node x, Node y) {
        // base case 1: if the tree is empty
        if (root == null) return null;
        // base case 2: if either `x` or `y` is found
        if (root == x || root == y) return root;

        // recursively check if `x` or `y` exists in the left subtree
        Node left = findLCA(root.left, x, y);
        // recursively check if `x` or `y` exists in the right subtree
        Node right = findLCA(root.right, x, y);

        // if `x` is found in one subtree and `y` is found in the other subtree,
        // update lca to the current node
        if (left != null && right != null) return root;

        // if `x` and `y` exist in the left subtree
        if (left != null) return left;
        // if `x` and `y` exist in the right subtree
        if (right != null) return right;

        return null;
    }

    // Function to find the distance between node `x` and node `y` in a
    // given binary tree rooted at `root` node
    public static int findDistance(Node root, Node x, Node y) {
        // `lca` stores the lowest common ancestor of `x` and `y`
        Node lca = null;

        int lev_x = findLevel(root, x, 0);
        int lev_y = findLevel(root, y, 0);

        if(lev_x == 0 or lev_y == 0)

        // call LCA procedure only if both `x` and `y` are present in the tree
        if (isNodePresent(root, y) && isNodePresent(root, x)) {
            lca = findLCA(root, x, y);
        }
        else return Integer.MIN_VALUE;

        // return distance of `x` from lca + distance of `y` from lca
        return findLevel(lca, x, 0) + findLevel(lca, y, 0);

        /*
            The above statement is equivalent to the following:

            return findLevel(root, x, 0) + findLevel(root, y, 0) -
                    2*findLevel(root, lca, 0);

            We can avoid calling the `isNodePresent()` function by using
            return values of the `findLevel()` function to check if
            `x` and `y` are present in the tree or not.
        */
    }

    public static void main(String[] args) {
        /* Construct the following tree
              1
            /   \
           /     \
          2       3
           \     / \
            4   5   6
               /     \
              7       8
        */
        Node root = new Node(1);
        root.left = new Node(2);
        root.right = new Node(3);
        root.left.right = new Node(4);
        root.right.left = new Node(5);
        root.right.right = new Node(6);
        root.right.left.left = new Node(7);
        root.right.right.right = new Node(8);
        // find the distance between node 7 and node 6
        System.out.print(findDistance(root, root.right.left.left, root.right.right));
    }
}

```


---


### encrypt the ebs by default


```py
def lambda_handler(event, context):
    # set the region
    region = 'us-east-1'

    # set the client
    client = boto3.client('ec2', region)

    response = client.enable_ebs_encryption_by_default()

    # result =
    print("Default EBS Encryption setup for region", region,": ", response['EbsEncryptionByDefault'])

    return {
        'statusCode': 200,
        'body': json.dumps(event)
    }
```



---


## IBM

---


### 1. Merge sorted arrays

> Back-End Developer
> 3 question:

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

### 2. Rearrange an array in order – smallest, largest, 2nd smallest, 2nd largest, ..

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

### 3. Print all triplets with given sum

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


### 1. Partitioning Array


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


### 2. calculate the get response


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




### 3. shopper the list


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


---


## redfine



---


### Implement a RESTful API

Instructions

Implement a RESTful API that supports 4 operations:
- adding a user,
- authenticating,
- retrieving user details,
- and logging out.
- The user details should be stored in a persistent back-end data store such as Sqlite or even a text file.
- Password and session generation mechanisms should follow current secure development best practices.

While many mechanisms exist for `secure storage and generation`, this exercise is to determine how you would implement these solutions for yourself.
While you may write in any language you want, an example of libraries and frameworks to use in a Python project would be using `hashlib`, but not Flask’s HTTPBasicAuth - we would like to see how you implement those mechanisms.
Do not `JWT tokens` unless you’re implementing the algorithms, management, and token creation yourself.

Finally, include a `README` file that describes your project, the overall flow for a user, why you made any specific architectural decisions, and what you would change given appropriate frameworks, libraries, etc.

Don’t spend more than 3 hours on this - the code is not meant to be production-ready, it is meant to show your understanding of best practices and how these interesting challenges work in your code.
No deployment or infrastructure is necessary, code running on localhost is sufficient.


Testing
Test data will be username, Firstname Lastname, password, and Mother’s Favorite Search Engine.

Some possible values to test with might be:
gh0st,William L. Simon,,Searx
jet-setter,Frank Abignale,r0u7!nG,Bing
kvothe,Patrick Rothfuss,3##Heel7sa*9-zRwT,Duck Duck Go
tpratchett,Terry Pratchett,Thats Sir Terry to you!,Google
lmb,Lois McMaster Bujold,null,Yandex


Final Project
The final project deliverable should include
- `the source code` of your application,
- a `README` file targeting other engineers,
- and a plain `text dump of your data store with test data loaded`,
- and at least 2 users logged in.
- This can be in a tar file, zip file, or as separate attachments.




```py
def lambda_handler(event, context):
    # test one regions example
    regions = ['us-east-1']

    # set the region
    region =

    # set the client
    client = boto3.client('ec2', region)

    response = client.enable_ebs_encryption_by_default()


    # result =

    print("Default EBS Encryption setup for region", region,": ", response['EbsEncryptionByDefault'])

    return {
        'statusCode': 200,
        'body': json.dumps(event)
    }
```

```py
# If I was to give you 2 ordered sets of any data type you like (array, set, list, etc),
# such as 2,4,6,8 and 3,5,7,9 can you combine them into 1 ordered set so the final set
# is ordered 2,3,4,5,6,7,8,9.

# Extended challenge: can you write the logic to integrate the two sets of ordered numbers
# manually, so that while they are being merged they are in order (eg whever you are in
# merging the sets, the final set remains in incremental order)

def sortThem():
  #  example:
  a = [2,4,6,8]
  b = [3,5,7,9]
  # a = a+b
  # a.sort()
  # print(a)

  c = []

  i = 0
  j = 0

  while i < len(a):
    if (a[i] < b[j]):
      # print("a is smaller")
      # c[i]=a[i] #c.add(a[i])
      c.append(a[i])
      i+=1
    else:
      c.append(b[j])
      j+=1
    print(i,j)

  c = c + a[i:] + b[j:]

  print(c)

sortThem()

```







.
