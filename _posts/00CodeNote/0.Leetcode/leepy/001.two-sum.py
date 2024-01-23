# 1. Two Sum

# Given an array of integers, return indices of the two numbers such that they add up to a specific target.
# You may assume that each input would have exactly one solution, and you may not use the same element twice.
# 挑两个和为 target 的数字


# Example:
# nums = [2, 7, 11, 15]
# target = 9
# Because nums[0] + nums[1] = 2 + 7 = 9,
# return [0, 1].


# l1 = ["eat","sleep","repeat"]
# print (list(enumerate(l1)))


# ======================== Python
from typing import List


def twoSum(nums, target):
    """
    :type nums: List[int]
    :type target: int
    :rtype: List[int]
    """
    remain = []
    for i in range(len(nums)):
        if nums[i] in remain:
            return [remain.index(nums[i]), i]
        remain.append(target - nums[i])


def twoSum(nums, target):
    ans = {}
    # for every num
    # [(0, 'eat'), (1, 'sleep'), (2, 'repeat')]
    for i, num in list(enumerate(nums)):
        if num in ans.keys():
            return [ans[num], i]
        # the other part
        ans[target - num] = i


def twoSum(nums: List[int], target: int) -> List[int]:
    hashmap = {}
    for idx, value in enumerate(nums):
        if (target - value) in hashmap:
            return [idx, hashmap[target - value]]
        hashmap[value] = idx


nums = [2, 7, 11, 15]
target = 9
print(twoSum(nums=nums, target=target))


# # ---------------------- solution java   Approach 1: Brute Force  ----------------------
# # 挨个 轮回三次 算数字是否相符
# class Solution {
#     public static int[] twoSum(int[] nums, int target) {
#         int[] result = new int[2];
#         if (nums == null || nums.length == 0) {
#             return result;
#         }
#         for (int i = 0; i < nums.length; i++) {
#             for (int j = i + 1; j < nums.length; j++) {
#                 if (nums[i] == target - nums[j]) {
#                     result[0] = i;
#                     result[1] = j;
#                     return result;
#                 }
#             }
#         }
#         return result;
#     }
# }
# # Runtime: 89 ms, faster than 10.51% of Java online submissions for Two Sum.
# # Memory Usage: 39.5 MB, less than 71.20% of Java online submissions for Two Sum.


# public int[] twoSum(int[] nums, int target) {
#     for (int i = 0; i < nums.length; i++) {
#         for (int j = i + 1; j < nums.length; j++) {
#             if (nums[j] == target - nums[i]) {
#                 return new int[] { i, j };
#             }
#         }
#     }
#     throw new IllegalArgumentException("No two sum solution");
# }
# # Runtime: 110 ms, faster than 6.76% of Java online submissions for Two Sum.
# # Memory Usage: 40.8 MB, less than 10.59% of Java online submissions for Two Sum.
# # Time complexity : O(n^2) For each element, we try to find its complement by looping through the rest of array which takes O(n)O(n) time. Therefore, the time complexity is O(n^2)
# # Space complexity : O(1)O(1).


# # ---------------------- solution java   Approach 2: Two-pass Hash Table  ----------------------
# A hash table.
# more efficient way to check if the complement exists in the array.
# the best way to maintain a mapping of each element in the array to its index.
# reduce the look up time from O(n) to O(1) by trading space for speed. A hash table is built exactly for this purpose, it supports fast look up in near constant time.
#
#
# class Solution {
#     public int[] twoSum(int[] nums, int target) {
#         Map<Integer, Integer> map = new HashMap<>();
#
#         for (int i = 0; i < nums.length; i++) {
#             map.put(nums[i], i);
#             #  {value=index}
#         }
#
#         for (int i = 0; i < nums.length; i++) {
#             int complement = target - nums[i];
#             if (map.containsKey(complement) && map.get(complement) != i) {
#                 return new int[] { i, map.get(complement) };
#             }
#         }
#         throw new IllegalArgumentException("No two sum solution");
#     }
# }
# Runtime: 2 ms, faster than 76.21% of Java online submissions for Two Sum.
# Memory Usage: 39.4 MB, less than 77.83% of Java online submissions for Two Sum.


# Approach 3: One-pass Hash Table
# It turns out we can do it in one-pass. While we iterate and inserting elements into the table, we also look back to check if current elements complement already exists in the table. If it exists, we have found a solution and return immediately.


# public int[] twoSum(int[] nums, int target) {
#     Map<Integer, Integer> map = new HashMap<>();
#     for (int i = 0; i < nums.length; i++) {
#         int complement = target - nums[i];
#         if (map.containsKey(complement)) {
#             return new int[] { map.get(complement), i };
#         }
#         map.put(nums[i], i);
#     }
#     throw new IllegalArgumentException("No two sum solution");
# }
# Runtime: 1 ms, faster than 99.96% of Java online submissions for Two Sum.
# Memory Usage: 41.7 MB, less than 5.73% of Java online submissions for Two Sum.

# Time complexity : O(n). We traverse the list containing nn elements only once. Each look up in the table costs only O(1)O(1) time.
#
# Space complexity : O(n). The extra space required depends on the number of items stored in the hash table, which stores at most nn elements.


# Java 8, only 3ms runtime (99.94% faster than all submissions):
#
# class Solution {
#     public int[] twoSum(int[] nums, int target) {
#       Hashtable<Integer, Integer> hashTable = new Hashtable<Integer, Integer>();
#       int i = 0;
#       while ((i < nums.length) && (hashTable.get(nums[i]) == null)) {
#         hashTable.put(target - nums[i], i);
#         i++;
#       }
#       if (i < nums.length) {
#         return new int[]{hashTable.get(nums[i]),i};
#       }
#       return null;
#     }
# }


# # ---------------------- solution py ----------------------
# class Solution(object):
#     def twoSum(nums, target):
#         dictionary = dict()
#         pos = 0
#         while pos < len(nums):
#             if (target - nums[pos]) not in dictionary:
#                 dictionary[nums[pos]] = pos
#                 pos += 1
#             else:
#                 return [dictionary[target - nums[pos]], pos]
# Runtime: 28 ms, faster than 98.66% of Python online submissions for Two Sum.
# Memory Usage: 14.1 MB, less than 5.13% of Python online submissions for Two Sum.


# # ---------------------- solution py ----------------------
# enumerate(), (index, value) 阅过放入dic
# 在看目标结果是否已在dic里,
# class Solution(object):
#     def twoSum(nums, target):
#         a ={}
#         for index, num in enumerate(nums):
#             if target-num in a:
#                 return [a[target - num], index]
#             else:
#                 a[num] = index
# # Runtime: 28 ms, faster than 98.66% of Python online submissions for Two Sum.
# # Memory Usage: 14.2 MB, less than 5.13% of Python online submissions for Two Sum.


# # ---------------------- solution py ----------------------
# 如果可以 sorted（），算两遍，往中间移动
# 但是无规律的 list 不符合
# class Solution(object):
#     def twoSum(nums, target):
#         if len(nums) == 0:
#             return []
#         nums = sorted(nums)
#         head = 0
#         end = len(nums)-1
#         print(end)
#         while head < end:
#             sum = nums[head] + nums[end]
#             if (sum > target):
#                 end -=1
#                 print(nums[head], nums[end], "sum > target")
#             elif (sum < target):
#                 head +=1
#                 print(nums[head], nums[end], "sum < target")
#             elif (sum == target):
#                 print(nums[head], nums[end], "sum = target")
#                 return [head, end]
#         return []


# if __name__ == '__main__':
#     # begin
#     s = Solution()
#     # print(s.twoSum([2, 7, 11, 15], 9))
#     print(s.twoSum([3,2,4], 6))
