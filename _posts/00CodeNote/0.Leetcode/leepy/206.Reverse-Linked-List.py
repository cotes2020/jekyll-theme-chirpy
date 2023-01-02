# 206. Reverse Linked List

# Reverse a singly linked list.

# Example:
# Input: 1->2->3->4->5->NULL
# Output: 5->4->3->2->1->NULL

# Follow up:
# A linked list can be reversed either iteratively or recursively. Could you implement both?


# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next


# solution 1 ----------------------------- linea 3 point
# Time complexity = O(n)
# Space compexity - O(1)
class Solution:
    def reverseList(self, head):
        if head is None:
            return head
        prev, cur, ahead = None, head, head.next
        while ahead is not None:
            cur.next = prev

            prev = cur
            cur = ahead
            ahead = ahead.next
        cur.next = prev
        head = cur
        return head


# Runtime: 16 ms, faster than 98.66% of Python online submissions for Reverse Linked List.
# Memory Usage: 15.4 MB, less than 55.90% of Python online submissions for Reverse Linked List.


# solution 2 ----------------------------- linea 3 point
# def reverseList(self, head):
#     prev = None
#     while head:
#         head.next = prev
#         prev = head
#         head = head.next
#         # head.next, prev, head = prev, head, head.next
#     return prev
# Runtime: 24 ms, faster than 72.49% of Python online submissions for Reverse Linked List.
# Memory Usage: 15.2 MB, less than 94.24% of Python online submissions for Reverse Linked List.


def reverseList(self, head):
    if not head:
        return head  # Empty.
    if not head.next:
        return head  # We reached end.
    # Traverse to end, orig_head is now end node.
    orig_head = self.reverseList(head.next)
    head.next.next = head  # Swap head with right node.
    head.next = None  # So we don't wind up in infinite loop.
    return orig_head  # Very last thing returned. End node!


# Runtime: 24 ms, faster than 72.49% of Python online submissions for Reverse Linked List.
# Memory Usage: 18.9 MB, less than 10.50% of Python online submissions for Reverse Linked List.
