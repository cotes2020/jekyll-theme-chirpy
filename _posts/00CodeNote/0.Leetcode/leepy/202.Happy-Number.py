# 202. Happy Number

# Write an algorithm to determine if a number n is happy.
# A happy number is a number defined by the following process:

# Starting with any positive integer, replace the number by the sum of the squares of its digits.
# Repeat the process until the number equals 1 (where it will stay), or it loops endlessly in a cycle which does not include 1.
# Those numbers for which this process ends in 1 are happy.
# Return true if n is a happy number, and false if not.


# Example 1:
# Input: n = 19
# Output: true
# Explanation:
# 1^2 + 9^2 = 82
# 8^2 + 2^2 = 68
# 6^2 + 8^2 = 100
# 1^2 + 0^2 + 02 = 1

# Example 2:
# Input: n = 2
# Output: false


class Solution:
    def isHappy(self, n):
        # set for all number that wont provide 1
        seen = set()

        def helper(x):
            res = 0
            while x > 9:
                # divmod(3, 8) =  (0, 3)
                # divmod(5, 5) =  (1, 0)
                x, y = divmod(x, 10)
                res += y**2
                res += x**2
            if res in seen:
                return False
            if res == 1:
                return True
            seen.add(res)
            return helper(res)

        return helper(n)


def isHappy(self, n):
    mem = set()
    while n != 1:
        n = sum([int(i) ** 2 for i in str(n)])
        if n in mem:
            return False
        else:
            mem.add(n)
    else:
        return True
