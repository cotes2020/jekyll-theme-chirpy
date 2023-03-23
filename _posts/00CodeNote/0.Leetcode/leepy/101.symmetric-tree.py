# 101. Symmetric Tree
# Easy
# Given the root of a binary tree, check whether it is a mirror of itself (i.e., symmetric around its center).

# Example 1:
#           1
#        /     \
#      2         2
#     /  \      /  \
#   3     4    4     3
# Input: root = [1,2,2,3,4,4,3]
# Output: true

# Example 2:
#           1
#        /     \
#      2         2
#        \         \
#         3         3
# Input: root = [1,2,2,null,3,null,3]
# Output: false

# Constraints:
# The number of nodes in the tree is in the range [1, 1000].
# -100 <= Node.val <= 100

# Follow up: Could you solve it both recursively and iteratively?


# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


# Top-down Iterative
# Runtime: 39 ms, faster than 8.35% of Python online submissions for Symmetric Tree.
# Memory Usage: 13.4 MB, less than 99.60% of Python online submissions for Symmetric Tree.
class Solution:
    def isSymmetric(self, root):
        if not root:
            return True
        stack1, stack2 = [root.left], [root.right]
        while stack1 and stack2:
            root1, root2 = stack1.pop(), stack2.pop()
            if (root1, root2) == (None, None):
                continue
            elif not (root1 and root2):
                return False
            elif root1.val != root2.val:
                return False
            stack1.append(root1.right)
            stack1.append(root1.left)
            stack2.append(root2.left)
            stack2.append(root2.right)
        return not (stack1 or stack2)

    # (Faster, More Memory)
    def isSymmetric(self, root):
        if root.left is None and root.right is None:
            return True
        if root.left and root.right:
            left_tree = root.left
            right_tree = root.right

            queue_left = deque([left_tree])
            queue_right = deque([right_tree])

            while len(queue_left) and len(queue_right):
                p_left = queue_left.pop()
                p_right = queue_right.pop()

                if p_left.val != p_right.val:
                    return False

                if p_left.left and p_right.right:
                    queue_left.appendleft(p_left.left)
                    queue_right.appendleft(p_right.right)
                elif p_left.left or p_right.right:
                    return False

                if p_left.right and p_right.left:
                    queue_left.appendleft(p_left.right)
                    queue_right.appendleft(p_right.left)
                elif p_left.right or p_right.left:
                    return False
            return True
        return False


# Top-down Recursion
# Runtime: 34 ms, faster than 12.71% of Python online submissions for Symmetric Tree.
# Memory Usage: 13.5 MB, less than 92.74% of Python online submissions for Symmetric Tree.
class Solution:
    def isSymmetric(self, root):
        if not root:
            return True
        return self.preorder(root.left, root.right)

    def preorder(self, root1, root2):
        # none first, none do not have val
        if (root1, root2) == (None, None):
            return True
        elif not (root1 and root2):
            return False
        elif root1.val != root2.val:
            return False
        return self.preorder(root1.left, root2.right) and self.preorder(
            root2.left, root1.right
        )


# Runtime: 20 ms, faster than 82.46% of Python online submissions for Symmetric Tree.
# Memory Usage: 13.9 MB, less than 6.32% of Python online submissions for Symmetric Tree.
class Solution:
    def traverse(self, root1, root2):
        if (root1, root2) == (None, None):
            return True
        if root1 is None or root2 is None:
            return False
        if root1.val != root2.val:
            return False
        left = self.traverse(root1.left, root2.right)
        right = self.traverse(root1.right, root2.left)
        if left and right and root1.val == root2.val:
            return True
        return False

    def isSymmetric(self, root):
        if root is None:
            return True
        return self.traverse(root.left, root.right)

    def isSymmetric(self, root):
        def pprint(now):
            # if not now:
            if now is None:
                return ["_"]

            if now.left is None and now.right is None:
                return [str(now.val)]

            return pprint(now.left) + [str(now.val)] + pprint(now.right)

        if (
            root is not None
            and root.left is not None
            and root.right is not None
            and root.left.val != root.right.val
        ):
            return False

        return pprint(root.left) == pprint(root.right)[::-1]


# Runtime: 22 ms, faster than 60.46% of Python online submissions for Symmetric Tree.
# Memory Usage: 13.7 MB, less than 44.59% of Python online submissions for Symmetric Tree.


class Solution:
    def isSymmetric(self, root):
        # edge cases
        if not root.left or not root.right:
            if not root.left and not root.right:
                return True
            else:
                return False
        if root.left.val != root.right.val:
            return False

        # initialize two queues
        q_left = [root.left]
        q_right = [root.right]

        # run normal bfs on the left subtree
        # run reversed bfs on the right subtree

        while len(q_left):
            new_left, new_right = [], []
            for node_l, node_r in zip(q_left, q_right):

                # check structure
                if node_l.left or node_r.right:
                    if node_l.left and node_r.right:
                        # check values
                        if node_l.left.val == node_r.right.val:
                            new_left.append(node_l.left)
                            new_right.append(node_r.right)
                        else:
                            return False
                    else:
                        return False

                # same as before
                if node_l.right or node_r.left:
                    if node_l.right and node_r.left:
                        if node_l.right.val == node_r.left.val:
                            new_left.append(node_l.right)
                            new_right.append(node_r.left)
                        else:
                            return False
                    else:
                        return False

            q_left = new_left
            q_right = new_right

        return True
