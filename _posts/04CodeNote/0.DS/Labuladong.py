

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
    res = [i for n, i in enumerate(test_list)]

# Method 5 : Using collections.OrderedDict.fromkeys()
def removeDuplicates(test_list):
    res = list(OrderedDict.fromkeys(test_list))
    # maintain the insertion order as well
    res = list(dict.fromkeys(test_list))

# Method 6 ------ 快慢指针
def removeDuplicates(test_list):
    # Runtime: 72 ms, faster than 99.60% of Python3 online submissions for Remove Duplicates from Sorted Array.
    # Memory Usage: 15.7 MB, less than 45.93% of Python3 online submissions for Remove Duplicates from Sorted Array.
    fast, slow = 0,0
    if len(test_list) == 0: return 0
    while fast < len(test_list):
        print(test_list)
        print(test_list[fast])

        if test_list[slow] != test_list[fast]:
            slow +=1
            test_list[slow] = test_list[fast]
        fast += 1
    print(test_list[0:slow+1])
    return slow+1

# removeDuplicates([0,0,1,2,2,3,3])







# =============== 有序链表去重
from basic import LinkedList, Node

# 两个指针
# Runtime: 40 ms, faster than 84.87% of Python3 online submissions for Remove Duplicates from Sorted List.
# Memory Usage: 14.2 MB, less than 56.16% of Python3 online submissions for Remove Duplicates from Sorted List.
def deleteDuplicates(LL):
    if not LL: return 0
    slow, fast = LL.head, LL.head
    if LL.head == None: return LL.head
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
            cur.next = cur.next.next     # skip duplicated node
        cur = cur.next     # not duplicate of current node, move to next node
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
    if not LL.head: return LL
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
    slow, fast = 0,0
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
    slow, fast = 0,0
    if nums == []: 
        return []
    while fast < len(nums):
        print(nums[fast])
        if nums[fast] != 0:
            nums[slow] = nums[fast]
            slow+=1
        fast+=1
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
            slow+=1
        i+=1
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
            slow+=1
    for i in range(slow, leng):
        nums[i] = 0
    return nums

# Runtime: 260 ms, faster than 13.33% of Python3 online submissions for Move Zeroes.
# Memory Usage: 15.5 MB, less than 24.34% of Python3 online submissions for Move Zeroes.
def moveZeroes(nums: List[int]) -> None:
    slow = 0
    for i in range(len(nums)):
        if nums[i] != 0:
            nums[slow],nums[i] = nums[i],nums[slow]
            slow +=1

# moveZeroes([0,1,0,3,12])





# =============== 和为k的子数组
# 1. Brute force O(N^3) - TLE
def subarraySum(self, nums: List[int], k: int) -> int: 
	count = 0
	for i in range(len(nums)):
		for j in range(i, len(nums)):
			if sum(nums[i:j+1]) == k: count += 1
	return count

# 2. Sliding Window O(N^2) - only works if the range of nums ∈ Z+ (+ve integers)
def subarraySum(self, nums: List[int], k: int) -> int: 
    if len(nums) == 1:
       if nums[0] == k: return 1
       else: return 0
    # O(n^2) solution
    count = 0
    i, j = 0, 1
    while j < len(nums):
        sub = nums[i:j+1]
        if sum(sub) == k: 
            # increment count and then disturbe the window again
            count += 1 # --- or return True
            j += 1 # disturbacne
        # --- expand
        elif sum(sub) < k: j += 1
        # shrink
        else: i += 1
    return count # return False         


def subarraySum(nums: List[int], k: int) -> int:
    print("------", nums)
    sumlist = [nums[0]]
    if sumlist == [k]: return 1
    for i in range(1,len(nums)):
        sumlist.append(nums[i-1]+nums[i])
        print(nums[i-1]+nums[i])
    i = sumlist.count(k) + nums.count(k)
    print(sumlist)
    return i
       
       
print(subarraySum([1,-1,0], 0))
print(subarraySum([-1,-1,1], 0))
print(subarraySum([1,1,1], 2))
print(subarraySum([1,2,3], 3))

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



# =============== 



# =============== 



# =============== 



















# .