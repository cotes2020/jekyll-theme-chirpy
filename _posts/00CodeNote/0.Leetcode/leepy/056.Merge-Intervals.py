# 56. Merge Intervals
# Medium

# Given an array of intervals where intervals[i] = [starti, endi], merge all overlapping intervals, and return an array of the non-overlapping intervals that cover all the intervals in the input.

# Example 1:
# Input: intervals = [[1,3],[2,6],[8,10],[15,18]]
# Output: [[1,6],[8,10],[15,18]]
# Explanation: Since intervals [1,3] and [2,6] overlaps, merge them into [1,6].

# Example 2:
# Input: intervals = [[1,4],[4,5]]
# Output: [[1,5]]
# Explanation: Intervals [1,4] and [4,5] are considered overlapping.

# # ---------------------- solution py ----------------------
class Solution:
    def merge(self, intervals):
        if len(intervals) == 0:
            return []
        intervals.sort(key=lambda x: x[0])
        merged = []
        for interval in intervals:
            # if the list of merged intervals is empty
            # or if the current interval does not overlap with the previous,
            # simply append it.
            if not merged or merged[-1][1] < interval[0]:
                merged.append(interval)
            else:
                # otherwise, there is overlap, merge the current and previous intervals.
                merged[-1][1] = max(merged[-1][1], interval[1])
        return merged


# # Runtime: 60 ms, faster than 93.79% of Python online submissions for Merge Intervals.
# # Memory Usage: 15.6 MB, less than 30.00% of Python online submissions for Merge Intervals.


# ---------------------- solution py ----------------------
# def merge(self, intervals):
#     if len(intervals) == 0: return []
#     out = []
#     for i in sorted(intervals, key=lambda i: i.start):
#         # if out not empty or overlap a[1,2] > b[3,4]
#         if out and out[-1].end >= i.start:
#             out[-1].end = max(out[-1].end, i.end)
#         # if out empty or not overlap
#         else:
#             out += i,
#     return out
# # Runtime: 60 ms, faster than 93.79% of Python online submissions for Merge Intervals.
# # Memory Usage: 15.9 MB, less than 30.00% of Python online submissions for Merge Intervals.
