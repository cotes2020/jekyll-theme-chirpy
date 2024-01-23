# 94. Binary Tree Inorder Traversal
# Easy
# Given the root of a binary tree, return the inorder traversal of its nodes' values.

# Example 1:
# Input: root = [1,null,2,3]
# Output: [1,3,2]
#    1
#     \
#      2
#     /
#    3


# Example 2:
# Input: root = []
# Output: []

# Example 3:
# Input: root = [1]
# Output: [1]

# Example 4:
# Input: root = [1,2]
# Output: [2,1]
#      1
#     /
#    2

# Example 5:
# Input: root = [1,null,2]
# Output: [1,2]
#    1
#     \
#      2

# Constraints:
# The number of nodes in the tree is in the range [0, 100].
# -100 <= Node.val <= 100


# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class Solution:

    # Runtime: 24 ms, faster than 21.04% of Python online submissions for Binary Tree Inorder Traversal.
    # Memory Usage: 13.4 MB, less than 76.61% of Python online submissions for Binary Tree Inorder Traversal.
    def inorderTraversal(self, root):
        stack = []
        result = []
        while root or stack:
            while root:
                stack.append(root)
                root = root.left
            root = stack.pop()
            result.append(root.val)
            root = root.right
        return result


# Runtime: 16 ms, faster than 76.59% of Python online submissions for Binary Tree Inorder Traversal.
# Memory Usage: 13.6 MB, less than 19.47% of Python online submissions for Binary Tree Inorder Traversal.


class Solution:
    def __init__(self):
        self.final = []

    def traversal(self, root):
        if root:
            self.traversal(root.left)
            self.final.append(root.val)
            self.traversal(root.right)

    def inorderTraversal(self, root):
        self.traversal(root)
        return self.final


# solution using stack and BT
# Runtime: 16 ms, faster than 76.59% of Python online submissions for Binary Tree Inorder Traversal.
# Memory Usage: 13.6 MB, less than 5.98% of Python online submissions for Binary Tree Inorder Traversal.
class Solution:
    def inorderTraversal(self, root):
        if root is None:
            return None
        curr = root
        ans = []
        stack = []
        stack.append(root)
        while stack != []:
            if curr is None:
                curr = stack.pop()
                ans.append(curr.val)
                curr = curr.right
            else:
                stack.append(curr)
                curr = curr.left
        ans.pop()
        return ans


# Runtime: 12 ms, faster than 95.47% of Python online submissions for Binary Tree Inorder Traversal.
# Memory Usage: 13.3 MB, less than 76.61% of Python online submissions for Binary Tree Inorder Traversal.
class Solution:
    def inorderTraversal(self, root):
        def dfs(node, ans):
            if node.left:
                dfs(node.left)
            ans.append(node.val)
            if node.right:
                dfs(node.right)

        ans = []
        if root:
            dfs(root, ans)

        return ans

    # Runtime: 38 ms, faster than 5.25% of Python online submissions for Binary Tree Inorder Traversal.
    # Memory Usage: 13.4 MB, less than 76.61% of Python online submissions for Binary Tree Inorder Traversal.
    # Runtime: 12 ms, faster than 95.47% of Python online submissions for Binary Tree Inorder Traversal.
    # Memory Usage: 13.4 MB, less than 48.55% of Python online submissions for Binary Tree Inorder Traversal.

    def inorderTraversal(self, root):
        if not root:
            return []
        return (
            self.inorderTraversal(root.left)
            + [root.val]
            + self.inorderTraversal(root.right)
        )

    def Solution(root):
        return (
            [] if not root else Solution(root.left) + [root.val] + Solution(root.right)
        )

    def inorderTraversal(self, root):
        return (
            []
            if not root
            else self.inorderTraversal(root.left)
            + [root.val]
            + self.inorderTraversal(root.right)
        )
