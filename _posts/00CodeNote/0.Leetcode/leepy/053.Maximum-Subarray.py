# 53. Maximum Subarray
# easy
# Given an integer array nums, find the contiguous subarray (containing at least one number) which has the largest sum and return its sum.

# Follow up: If you have figured out the O(n) solution, try coding another solution using the divide and conquer approach, which is more subtle.

# Example 1:
# Input: nums = [-2,1,-3,4,-1,2,1,-5,4]
# Output: 6
# Explanation: [4,-1,2,1] has the largest sum = 6.


# # solution ----------------------------- slide widow, add one by one
# # O(n^3) O(n^2)
# def maxSubArray(nums):
#     maxSum = max(nums[0], nums[-1])
#     i,j=0,13
#     while i <= len(nums)-2:
#         currSum = nums[i]
#         maxSum = max(maxSum, currSum)
#         j=i+1
#         while j <= len(nums)-1:
#             currSum += nums[j]
#             maxSum = max(maxSum, currSum)
#             print(nums[i], nums[j], currSum)
#             j+=1
#         i+=1
#     print(maxSum)
#     return maxSum


# # solution ----------------------------- adding one, one by one, no matter + -
# # Runtime: 52 ms, faster than 49.54% of Python online submissions for Maximum Subarray.
# # Memory Usage: 14.3 MB, less than 52.01% of Python online submissions for Maximum Subarray.
# def maxSubArray(nums):
#     if not nums:
#         return 0
#     startSum = maxSum = nums[0]
#     for i in nums[1:]:
#         startSum = max(i, startSum + i)
#         maxSum = max(maxSum, startSum)
#         print("i:", i, "startSum:", startSum, "maxSum:", maxSum)
#     return maxSum


# # solution ----------------------------- adding one, one by one, separate + -
# # Runtime: 36 ms, faster than 99.72% of Python online submissions for Maximum Subarray.
# # Memory Usage: 14.3 MB, less than 36.42% of Python online submissions for Maximum Subarray.
# def maxSubArray(self, nums):
#     is_all_neg, curr, maxcurr = True, 0, float("-inf")
#     # if all negative: just return the bigest one.
#     for n in nums:
#         maxcurr = max(n, maxcurr)
#         if n >= 0:
#             is_all_neg = False
#             break
#     if is_all_neg:
#         return maxcurr

#     # if not all negative
#     maxcurr = 0
#     for n in nums:
#         curr += n
#         if curr < 0:
#             curr = 0
#         # curr = max(curr, n)
#         maxcurr = max(maxcurr, curr)
#     return maxcurr
# # not good, because calculate the max for more turns


# # solution ----------------------------- adding one, one by one, separate + -
# # Runtime: 52 ms, faster than 49.54% of Python online submissions for Maximum Subarray.
# # Memory Usage: 14.2 MB, less than 52.01% of Python online submissions for Maximum Subarray.
# def maxSubArray(nums):
#     maxSub, curSum = nums[0], 0
#     for n in nums:
#         if curSum < 0:
#             curSum = 0
#         # if all negative, just return the bigest one, stop adding the negative one.
#         # if not negative, add the sum
#         curSum += n
#         maxSub = max(maxSub, curSum)
#     return maxSub


# # solution 4 -----------------------------
# # Runtime: 44 ms, faster than 87.85% of Python online submissions for Maximum Subarray.
# # Memory Usage: 14.4 MB, less than 36.42% of Python online submissions for Maximum Subarray.
# def maxSubArray(nums):
#     for i in range(1,len(nums)):
#         nums[i] = max(nums[i], nums[i]+nums[i-1])
#     return max(nums)


# # solution 2 ----------------------------- linea
# def maxSubArray(nums):
#     ans = nums[0]
#     for i in range(len(nums)-1):
#         if i == 0:
#             ans = nums[i]
#         else:
#             lastSum = maxSubArray(nums[0:i])
#             maxSum = max(nums[i], nums[i] + lastSum)
#             print("nums[i]:", nums[i], "lastSum:", lastSum, "maxSum:", maxSum)
#             ans = max(maxSum, ans)
#         print("test")
#         print(nums, ans)
#     return ans


# nums = [-9, -2, -3, -4]
# nums = [1]
nums = [-2, 1, -3, 4, -1, 2, 1, -5, 4]
# nums = [-2,1]
# nums = [2,1]
# nums = [-1,0,-2]
max = maxSubArray(nums)
print(max)

# a = nums[0:1]
# print(a)
