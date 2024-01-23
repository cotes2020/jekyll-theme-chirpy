# 70. Climbing Stairs
# Easy
# You are climbing a staircase. It takes n steps to reach the top.
# Each time you can either climb 1 or 2 steps. In how many distinct ways can you climb to the top?

# Example 1:
# Input: n = 2
# Output: 2
# Explanation: There are two ways to climb to the top.
# 1. 1 step + 1 step
# 2. 2 steps

# Example 2:
# Input: n = 3
# Output: 3
# Explanation: There are three ways to climb to the top.
# 1. 1 step + 1 step + 1 step
# 2. 1 step + 2 steps
# 3. 2 steps + 1 step
# Constraints:
# 1 <= n <= 45

# botoon up
# solution -----------------
# Runtime: 19 ms, faster than 49.07% of Python online submissions for Climbing Stairs.
# Memory Usage: 13.5 MB, less than 35.35% of Python online submissions for Climbing Stairs.
def climbStairs(n):
    """
    :type n: int
    :rtype: int
    """
    dp = {}
    dp[0] = 1
    dp[1] = 1
    i = 2
    while i <= n:
        dp[i] = dp[i - 1] + dp[i - 2]
        i += 1
    return dp[n]


def climbStairs(n):
    dp = {0: 1, 1: 1, 2: 2, 3: 3, 4: 5}
    i = 5
    while i <= n:
        dp[i] = dp[i - 1] + dp[i - 2]
        i += 1
    return dp[n]


# botoon up
# solution -----------------
# Runtime: 16 ms, faster than 77.16% of Python online submissions for Climbing Stairs.
# Memory Usage: 13.5 MB, less than 11.97% of Python online submissions for Climbing Stairs.
def climbStairs(self, n):
    one, two = 1, 1
    for i in range(n - 1):
        tem = one
        one = one + two
        two = tem
    return one


# botoon up
# solution -----------------
# Runtime: 8 ms, faster than 99.55% of Python online submissions for Climbing Stairs.
# Memory Usage: 13.3 MB, less than 88.05% of Python online submissions for Climbing Stairs.
def climbStairs(self, n):
    if n in [1, 2, 3]:
        return n
    else:
        dic = {3: 3, 4: 5}
        for i in range(5, n + 1):
            dic[i] = dic.get(i - 1) + dic.get(i - 2)
    return dic.get(n)


climbStairs(6)
