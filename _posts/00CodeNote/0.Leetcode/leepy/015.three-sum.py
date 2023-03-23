#


# 15. 3Sum
# Medium

# Given an array nums of n integers, are there elements a, b, c in nums such that a + b + c = 0? Find all unique triplets in the array which gives the sum of zero.

# Notice that the solution set must not contain duplicate triplets.


# Example 1:
# Input: nums = [-1,0,1,2,-1,-4]
# Output: [[-1,-1,2],[-1,0,1]]

# Example 2:
# Input: nums = []
# Output: []

# Example 3:
# Input: nums = [0]
# Output: []


# # ---------------------- solution py ----------------------
# # 3 sum > 2 sum
# class Solution:
#     def threeSum(self, nums: List[int]) -> List[List[int]]:
#         # exception: (make it slow)
#         # if len(nums) < 3:
#         #     return []

#         nums.sort()
#         n = len(nums)
#         res = []
#         for i in range(n-2):
#             #  [1,2,3] no way to become 0
#             if (nums[i] + nums [i+1] + nums[i+2] > 0):
#                 return res
#             #  [-100, ....., 1, 13] no way to become 0
#             if (nums[i] + nums[n-2] + nums[n-1] < 0):
#                 continue

#             #  [-100, -100, -1, -1, 2, 3]  minus the duplicate
#             if i > 0 and nums[i] == nums[i-1]:
#                 continue


#             # 3sum: 1 num + 2sum (2 pointer left and right)
#             l , r = i+1, len(nums)-1
#             while l < r:
#                 s = nums[i] + nums[l] + nums[r]
#                 if s < 0:
#                     l +=1
#                 elif s > 0:
#                     r -= 1
#                 else:
#                     res.append((nums[i], nums[l], nums[r]))

#                     # minus duplicate

#                     # [-100, -100, -1, -1, 1, 1, 2, 3]
#                     while l < r and nums[l] == nums[l+1]:
#                         l += 1

#                     # # [-100, -100, -1, -1, 1, 1, 2, 3]
#                     while l < r and nums[r] == nums[r-1]:
#                         r -= 1

#                     l += 1; r -= 1
#         return res
# # Runtime: 680 ms, faster than 91.53% of Python3 online submissions for 3Sum.
# # Memory Usage: 16.9 MB, less than 96.04% of Python3 online submissions for 3Sum.


# ---------------------- solution py ----------------------
class Solution(object):
        # exception:
        if len(nums) < 3:
            return []

        nums.sort()
        res = set()
        target = 0

        # loop from x to n-2
        for index, x in enumerate(nums[:-2]):

            # [1,2,3] no way to become 0
            if (nums[index] + nums[index+1] + nums[index+2] > 0):
                return res
            #  [-100, ....., 1, 13] no way to become 0
            if (nums[index] + nums[-2] + nums[-1] < 0):
                continue

            # minus duplicate
            if index >= 1 and x == nums[index-1]:
                continue

            dic = {}
            # loop from x+1 to end
            for y in nums[index+1:]:
                if target-x-y not in dic:
                    # 2 number sum
                    dic[y] = 1
                else:
                    res.add((x, target-x-y, y))
        return res
# Runtime: 596 ms, faster than 97.83% of Python3 online submissions for 3Sum.
# Memory Usage: 18.1 MB, less than 16.93% of Python3 online submissions for 3Sum.


if __name__ == '__main__':
    # begin
    s = Solution()
    print(s.threeSum([-1, 0, 1, 2, -1, -4])
