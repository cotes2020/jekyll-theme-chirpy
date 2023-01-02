# =============== 有序数组去重
from collections import OrderedDict
from typing import List


# Method 1 ----- new list
def removeDuplicates(test_list):
    res = []
    for i in test_list:
        if i not in res:
            res.append(i)


# Method 2 ----- new list
def removeDuplicates(test_list):
    res = []
    [res.append(x) for x in test_list if x not in res]


# Method 3 ------ set(x)
def removeDuplicates(test_list):
    # the ordering of the element is lost
    test_list = list(set(test_list))


# Method 4 ------ Using list comprehension + enumerate()
def removeDuplicates(test_list):
    [i for n, i in enumerate(test_list)]


# Method 5 : Using collections.OrderedDict.fromkeys()
def removeDuplicates(test_list):
    list(OrderedDict.fromkeys(test_list))
    # maintain the insertion order as well
    list(dict.fromkeys(test_list))


# Method 6 ------ 快慢指针
def removeDuplicates(test_list):
    # Runtime: 72 ms, faster than 99.60% of Python3 online submissions for Remove Duplicates from Sorted Array.
    # Memory Usage: 15.7 MB, less than 45.93% of Python3 online submissions for Remove Duplicates from Sorted Array.
    fast, slow = 0, 0
    if len(test_list) == 0:
        return 0
    while fast < len(test_list):
        print(test_list)
        print(test_list[fast])

        if test_list[slow] != test_list[fast]:
            slow += 1
            test_list[slow] = test_list[fast]
        fast += 1
    print(test_list[0 : slow + 1])
    return slow + 1


# removeDuplicates([0,0,1,2,2,3,3])


# =============== 有序链表去重
from basic import LinkedList, Node


# 两个指针
# Runtime: 40 ms, faster than 84.87% of Python3 online submissions for Remove Duplicates from Sorted List.
# Memory Usage: 14.2 MB, less than 56.16% of Python3 online submissions for Remove Duplicates from Sorted List.
def deleteDuplicates(LL):
    if not LL:
        return 0
    slow, fast = LL.head, LL.head
    if LL.head == None:
        return LL.head
    while fast != None:
        if slow.val != fast.val:
            slow.next = fast
            slow = slow.next
        fast = fast.next
    slow.next = None
    # print(LL.val)
    return LL


# 一个指针
def deleteDuplicates(LL):
    cur = LL.head
    while cur:
        while cur.next and cur.val == cur.next.val:
            cur.next = cur.next.next  # skip duplicated node
        cur = cur.next  # not duplicate of current node, move to next node
    return LL


# nice for if the values weren't sorted in the linked list
def deleteDuplicates(LL):
    dic = {}
    node = LL.head
    while node:
        dic[node.val] = dic.get(node.val, 0) + 1
        node = node.next
    node = LL.head
    while node:
        tmp = node
        for _ in range(dic[node.val]):
            tmp = tmp.next
        node.next = tmp
        node = node.next
    return LL


# recursive
def deleteDuplicates(LL):
    if not LL.head:
        return LL
    if LL.head.next is not None:
        if LL.head.val == LL.head.next.val:
            LL.head.next = LL.head.next.next
            deleteDuplicates(LL.head)
        else:
            deleteDuplicates(LL.head.next)
    return LL


# LL = LinkedList()
# list_num = [0,0,1,2,2,3,3,4]
# for i in list_num:
#     LL.insert(i)
# LL.printLL()

# LL = deleteDuplicates(LL)
# LL.printLL()


# =============== 移除元素
# Runtime: 32 ms, faster than 81.50% of Python3 online submissions for Remove Element.
# Memory Usage: 14.2 MB, less than 47.25% of Python3 online submissions for Remove Element.
def removeElement(nums: List[int], val: int) -> int:
    slow, fast = 0, 0
    while fast < len(nums):
        if nums[fast] != val:
            nums[slow] = nums[fast]
            slow += 1
        fast += 1
    print(nums)
    print(nums[0:slow])


# removeElement([0,0,1,2,2,3,3], 2)


# =============== 移除0
# 两个指针
def moveZeroes(nums: List[int]) -> None:
    # Runtime: 188 ms, faster than 17.89% of Python3 online submissions for Move Zeroes.
    # Memory Usage: 15.6 MB, less than 7.33% of Python3 online submissions for Move Zeroes.
    slow, fast = 0, 0
    if nums == []:
        return []
    while fast < len(nums):
        print(nums[fast])
        if nums[fast] != 0:
            nums[slow] = nums[fast]
            slow += 1
        fast += 1
    for i in range(slow, len(nums)):
        nums[i] = 0
    print(nums)


# 一个指针
def moveZeroes(nums: List[int]) -> None:
    # Runtime: 172 ms, faster than 25.48% of Python3 online submissions for Move Zeroes.
    # Memory Usage: 15.4 MB, less than 24.21% of Python3 online submissions for Move Zeroes.
    slow = 0
    if nums == []:
        return []
    for i in range(len(nums)):
        if nums[i] != 0:
            nums[slow] = nums[i]
            slow += 1
        i += 1
    for i in range(slow, len(nums)):
        nums[i] = 0
    print(nums)


def moveZeroes(self, nums: List[int]) -> None:
    # Runtime: 248 ms, faster than 13.91% of Python3 online submissions for Move Zeroes.
    # Memory Usage: 15.2 MB, less than 88.67% of Python3 online submissions for Move Zeroes.
    slow = 0
    leng = len(nums)
    if nums == []:
        return []
    for i in range(leng):
        if nums[i] != 0:
            nums[slow] = nums[i]
            slow += 1
    for i in range(slow, leng):
        nums[i] = 0
    return nums


# Runtime: 260 ms, faster than 13.33% of Python3 online submissions for Move Zeroes.
# Memory Usage: 15.5 MB, less than 24.34% of Python3 online submissions for Move Zeroes.
def moveZeroes(nums: List[int]) -> None:
    slow = 0
    for i in range(len(nums)):
        if nums[i] != 0:
            nums[slow], nums[i] = nums[i], nums[slow]
            slow += 1


# moveZeroes([0,1,0,3,12])


# =============== 和为k的子数组
# Given an array of integers nums and an integer k,
# return the total number of continuous subarrays whose sum equals to k.
# Example 1:
# Input: nums = [1,1,1], k = 2
# Output: 2
# Example 2:
# Input: nums = [1,2,3], k = 3
# Output: 2

# 1. Brute force O(N^3) - TLE

# 2. Sliding Window O(N^2) - only works if the range of nums ∈ Z+ (+ve integers)
def subarraySum(self, nums: List[int], k: int) -> int:
    if len(nums) == 1:
        if nums[0] == k:
            return 1
        else:
            return 0
    # O(n^2) solution
    count = 0
    i, j = 0, 1
    while j < len(nums):
        sub = nums[i : j + 1]
        if sum(sub) == k:
            # increment count and then disturbe the window again
            count += 1  # --- or return True
            j += 1  # disturbacne
        # --- expand
        elif sum(sub) < k:
            j += 1
        # shrink
        else:
            i += 1
    return count  # return False


# [1,2,1,3] and k = 3
# running sums = [1,3,4,7]
# from 1->4, there is increase of k
# from 4->7, there is an increase of k.
# So, we've found 2 subarrays of sums=k.
def subarraySum(self, nums, k):
    count, sums = 0, 0
    d = dict()
    d[0] = 1
    for i in range(len(nums)):
        sums += nums[i]
        count += d.get(sums - k, 0)
        # how many this sums shows up
        d[sums] = d.get(sums, 0) + 1
    return count


def subarraySum(nums: List[int], k: int) -> int:
    print("------", nums)
    sumlist = [nums[0]]
    if sumlist == [k]:
        return 1
    for i in range(1, len(nums)):
        sumlist.append(nums[i - 1] + nums[i])
        print(nums[i - 1] + nums[i])
    i = sumlist.count(k) + nums.count(k)
    print(sumlist)
    return i


# print(subarraySum([1,-1,0], 0))
# print(subarraySum([-1,-1,1], 0))
# print(subarraySum([1,1,1], 2))
# print(subarraySum([1,2,3], 3))


# =============== 304. 二维区域和检索 - 矩阵不可变
# https://www.youtube.com/watch?v=PwDqpOMwg6U
# [
#   [[ [3,0,1,4,2],[5,6,3,2,1],[1,2,0,1,5],[4,1,0,1,7],[1,0,3,0,5] ]],
#   [2,1,4,3],[1,1,2,2],[1,2,2,4]
# ]
class NumMatrix:
    def __init__(self, matrix):
        if not matrix:
            return 0
        n, m = len(matrix), len(matrix[0])
        self.sums = [[0 for j in range(m + 1)] for i in range(n + 1)]
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                self.sums[i][j] = (
                    matrix[i - 1][j - 1]
                    + self.sums[i][j - 1]
                    + self.sums[i - 1][j]
                    - self.sums[i - 1][j - 1]
                )
        print(" -------- sums:")
        for i in self.sums:
            print(i)

    def sumRegion(self, row1: int, col1: int, row2: int, col2: int) -> int:
        #     self                     +      左                      +       up                  +      小角
        sum = (
            self.sums[row2 + 1][col2 + 1]
            - self.sums[row2 + 1][col1]
            - self.sums[row1][col2 + 1]
            + self.sums[row1][col1]
        )
        print(sum)
        return sum


# numMatrix = NumMatrix([[3, 0, 1, 4, 2], [5, 6, 3, 2, 1], [1, 2, 0, 1, 5], [4, 1, 0, 1, 7], [1, 0, 3, 0, 5]])
# numMatrix.sumRegion(2, 1, 4, 3); # return 8 (i.e sum of the red rectangle)
# numMatrix.sumRegion(1, 1, 2, 2); # return 11 (i.e sum of the green rectangle)
# numMatrix.sumRegion(1, 2, 2, 4); # return 12 (i.e sum of the blue rectangle)


# ===============


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


class Solution:
    # iteratively
    def mergeTwoLists(self, l1, l2):
        # while
        dummy = ListNode(0)
        p = dummy
        while l1 and l2:
            if l1.val < l2.val:
                p.next = l1
                l1 = l1.next
            else:
                p.next = l2
                l2 = l2.next
            p = p.next
        p.next = l1 or l2
        return dummy.next

    # recursively
    def mergeTwoLists(self, l1, l2):
        if not l1 or not l2:
            return l1 or l2
        if l1.val < l2.val:
            l1.next = self.mergeTwoLists(l1.next, l2)
            return l1
        else:
            l2.next = self.mergeTwoLists(l1, l2.next)
            return l2

    # recursively
    def mergeTwoLists(self, a, b):
        if a and b:
            if a.val > b.val:
                a, b = b, a
            a.next = self.mergeTwoLists(a.next, b)
        return a or b

    # in-place, iteratively
    def mergeTwoLists(self, l1, l2):
        if None in (l1, l2):
            return l1 or l2
        dummy = cur = ListNode(0)
        dummy.next = l1
        while l1 and l2:
            if l1.val < l2.val:
                l1 = l1.next
            else:
                nxt = cur.next
                cur.next = l2
                tmp = l2.next
                l2.next = nxt
                l2 = tmp
            cur = cur.next
        cur.next = l1 or l2
        return dummy.next


# ===============

# 23. Merge k Sorted Lists
# # You are given an array of k linked-lists lists, each linked-list is sorted in ascending order.
# Merge all the linked-lists into one sorted linked-list and return it.
# Example 1:
# Input: lists = [[1,4,5],[1,3,4],[2,6]]
# Output: [1,1,2,3,4,4,5,6]
# Explanation: The linked-lists are:
# [
#   1->4->5,
#   1->3->4,
#   2->6
# ]
# merging them into one sorted list:
# 1->1->2->3->4->4->5->6


class Solution:
    def mergeKLists(self, lists):
        if not lists:
            return None
        if len(lists) == 1:
            return lists[0]

        mid = len(lists) // 2
        l, r = self.mergeKLists(lists[:mid]), self.mergeKLists(lists[mid:])
        return self.merge(l, r)

    def merge(self, l, r):
        dummy = p = ListNode()
        while l and r:
            if l.val < r.val:
                p.next = l
                l = l.next
            else:
                p.next = r
                r = r.next
            p = p.next
        p.next = l or r
        return dummy.next

    def merge1(self, l, r):
        if not l or not r:
            return l or r
        if l.val < r.val:
            l.next = self.merge(l.next, r)
            return l
        r.next = self.merge(l, r.next)
        return r


# ===============


# ===============


# ===============


# ===============


# ===============


# ===============


# ===============


# ===============


# ===============


# ===============


# ===============


# ===============


# ===============
