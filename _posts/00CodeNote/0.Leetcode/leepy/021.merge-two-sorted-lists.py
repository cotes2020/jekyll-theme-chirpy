# 21. Merge Two Sorted Lists
# Easy
# Merge two sorted linked lists and return it as a sorted list. The list should be made by splicing together the nodes of the first two lists.

# Example 1:
# Input: l1 = [1,2,4], l2 = [1,3,4]
# Output: [1,1,2,3,4,4]

# Example 2:
# Input: l1 = [], l2 = []
# Output: []

# Example 3:
# l1 = []
# l2 = [0]
# Output: [0]

# Constraints:
# The number of nodes in both lists is in the range [0, 50].
# -100 <= Node.val <= 100
# Both l1 and l2 are sorted in non-decreasing order.


class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None


# Runtime O(M+N) where N is the size of the smaller list,
# we only have N iterations in our while loop,
# then simply append the remaining list to sorted list

# worse case, go though all the nodes
# O(M+N)
# N = total number of the nodes
# this happens when the values in the lists are of same range.


# Runtime: 41 ms, faster than 11.61% of Python online submissions for Merge Two Sorted Lists.
# Memory Usage: 13.4 MB, less than 86.32% of Python online submissions for Merge Two Sorted Lists.
def mergeTwoLists(self, l1, l2):
    """
    :type l1: ListNode
    :type l2: ListNode
    :rtype: ListNode
    """
    dummy = ListNode(0)  # for pointing to the sorted list that are making
    head = dummy
    # ListNode{val: 0, next: None}
    # ListNode{val: 0, next: None}

    # [1,2,4]
    # [1,3,4]
    # ListNode{val: 1, next: ListNode{val: 2, next: ListNode{val: 4, next: None}}}
    # ListNode{val: 1, next: ListNode{val: 3, next: ListNode{val: 4, next: None}}}

    while l1 and l2:
        if l1.val <= l2.val:
            dummy.next = l1
            l1 = l1.next
        else:
            dummy.next = l2
            l2 = l2.next
        dummy = dummy.next
        # ListNode{val: 1, next: ListNode{val: 2, next: ListNode{val: 4, next: None}}}
        # ListNode{val: 1, next: ListNode{val: 3, next: ListNode{val: 4, next: None}}}
        # ListNode{val: 2, next: ListNode{val: 4, next: None}}
        # ListNode{val: 3, next: ListNode{val: 4, next: None}}
        # ListNode{val: 4, next: None}

    dummy.next = l1 or l2
    # if any l1 or l2 = None
    # dummy point to the remaning list

    return head.next
    # as dummy point to 0 define initially


l1 = ListNode(0)
l2 = ListNode(0)

print(mergeTwoLists(l1, l2, l2))
