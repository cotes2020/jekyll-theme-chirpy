# 242. Valid Anagram
# Easy

# Given two strings s and t , write a function to determine if t is an anagram of s.

# Example 1:
# Input: s = "anagram", t = "nagaram"
# Output: true


# Example 2:
# Input: s = "rat", t = "car"
# Output: false


# # ---------------------- solution java ----------------------
# # find the char list, sort, see if equals
# public boolean isAnagram(String s, String t) {
#     if (s.length() != t.length()) {
#         return false;
#     }
#     char[] str1 = s.toCharArray();
#     char[] str2 = t.toCharArray();
#     Arrays.sort(str1);
#     Arrays.sort(str2);
#     return Arrays.equals(str1, str2);
# }
# # Complexity analysis
# # Time complexity : O(n \log n)O(nlogn). Assume that nn is the length of ss, sorting costs O(n \log n)O(nlogn) and comparing two strings costs O(n)O(n). Sorting time dominates and the overall time complexity is O(n \log n)O(nlogn).
# # Space complexity : O(1)O(1). Space depends on the sorting implementation which, usually, costs O(1)O(1) auxiliary space if heapsort is used. Note that in Java, toCharArray() makes a copy of the string so it costs O(n)O(n) extra space, but we ignore this for complexity analysis because:
# # It is a language dependent detail.
# # It depends on how the function is designed. For example, the function parameter types can be changed to char[].


# ---------------------- solution py ----------------------
class Solution:
    def isAnagram(self, s: str, t: str) -> bool:
        # exception:
        if len(s) != len(t):
            return False
        if s == t:
            return True
        return sorted(s) == sorted(t)


# Runtime: 40 ms, faster than 86.87% of Python3 online submissions for Valid Anagram.
# Memory Usage: 14.8 MB, less than 21.44% of Python3 online submissions for Valid Anagram.


# ---------------------- solution py ----------------------
# from collections import Counter
# def isAnagram(self, s: str, t: str) -> bool:
#         return Counter(s) == Counter(t)


# ---------------------- solution py ----------------------
# class Solution(object):
#     def isAnagram(self, s, t):
#         """
#         :type s: str
#         :type t: str
#         :rtype: bool
#         """
#         if len(s) != len(t):
#             return False

#         chars = {}
#         for i in s:
#             if i in chars:
#                 chars[i] += 1
#             else:
#                 chars[i] = 1
#         for x in t:
#             if x in chars:
#                 chars[x] -= 1
#             else:
#                 return False
#         for c in chars:
#             if chars[c] != 0:
#                 return False
#         return True


#
