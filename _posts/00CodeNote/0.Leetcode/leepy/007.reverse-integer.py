# 7. Reverse Integer

# Given a 32-bit signed integer, reverse digits of an integer.

# Example 1:
# Input: 123
# Output: 321

# Example 2:
# Input: -123
# Output: -321

# Example 3:
# Input: 120
# Output: 21

# Note:
# Assume we are dealing with an environment which could only store integers within the 32-bit signed integer range: [−231,  231 − 1]. For the purpose of this problem, assume that your function returns 0 when the reversed integer overflows.


def reverse(x):
    if x > 0:  # handle positive numbers
        a = int(str(x)[::-1])
    if x <= 0:  # handle negative numbers
        a = -1 * int(str(x * -1)[::-1])
    # handle 32 bit overflow
    mina = -(2**31)
    maxa = 2**31 - 1
    if a not in range(mina, maxa):
        return 0
    else:
        return a


# Runtime: 24 ms, faster than 95.74% of Python3 online submissions for Reverse Integer.
# Memory Usage: 13.8 MB, less than 5.26% of Python3 online submissions for Reverse Integer.


class Solution:
    def reverse(self, x):
        s = str(abs(x))  # Absolute value
        reversed = int(s[::-1])
        if reversed > 2147483647:
            return 0
        return reversed if x > 0 else (reversed * -1)


# Runtime: 20 ms, faster than 99.27% of Python3 online submissions for Reverse Integer.
# Memory Usage: 13.8 MB, less than 5.26% of Python3 online submissions for Reverse Integer.
