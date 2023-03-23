class Node:
    def __init__(self, data):
        "constructor to initiate this object"
        # store data
        self.val = data
        # store reference (next item)
        self.next = None
        return

    def has_value(self, value):
        "method to compare the value with the node data"
        if self.val == value:
            return True
        else:
            return False


# A Linked List class with a single head node


class LinkedList:
    def __init__(self):
        self.head = None

    # insertion method for the linked list
    def insert(self, data):
        newNode = Node(data)
        if self.head:
            current = self.head
            while current.next:
                current = current.next
            current.next = newNode
        else:
            self.head = newNode

    # print method for the linked list
    def printLL(self):
        current = self.head
        list = []
        while current:
            #   print(current.val)
            list.append(current.val)
            current = current.next
        print(list)
