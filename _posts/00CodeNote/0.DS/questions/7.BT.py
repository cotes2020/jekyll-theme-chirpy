# https://runestone.academy/runestone/books/published/pythonds/Trees/DiscussionQuestions.html

# 7.21. Discussion Questions

# -------------------------------------- Excercises -------------------------------------------------
# 1. Draw the tree structure resulting from the following set of tree function calls:
# >>> r = BinaryTree(3)
# >>> insertLeft(r,4)
# [3, [4, [], []], []]
# >>> insertLeft(r,5)
# [3, [5, [4, [], []], []], []]
# >>> insertRight(r,6)
# [3, [5, [4, [], []], []], [6, [], []]]
# >>> insertRight(r,7)
# [3, [5, [4, [], []], []], [7, [], [6, [], []]]]
# >>> setRootVal(r,9)
# >>> insertLeft(r,11)
# [9, [11, [5, [4, [], []] , []], []], [7, [], [6, [], []]]]
#          ---------------------               -----------
#     -------------------------------   ---------------------
#             9
#         11       7
#     5               6
# 4


# -------------------------------------- Excercises -------------------------------------------------
# 2. Trace the algorithm for creating an expression tree for the expression (4∗8)/6−3.
from pythonds.basic import Stack
from pythonds.trees import BinaryTree

# def calculate(a, operator, b):
#     if operator == '+': return a + b
#     elif operator == '-': return a - b
#     elif operator == '/': return a // b
#     else: return a * b


def expression_tree(input_word):
    char_list = input_word.split()
    tStack = Stack()
    eTree = BinaryTree("")
    tStack.push(eTree)
    cur_tree = eTree
    for char in char_list:
        if char == "(":
            cur_tree.insertLeft("")
            tStack.push(cur_tree)
            cur_tree = cur_tree.getLeftChild()
        elif char == ")":
            cur_tree = tStack.pop()
        elif char in ["+", "-", "*", "/"]:
            cur_tree.setRootVal(char)
            cur_tree.insertRight("")
            tStack.push(cur_tree)
            cur_tree = cur_tree.getRightChild()
        else:
            print(char)
            try:
                cur_tree.setRootVal(int(char))
                cur_tree = tStack.pop()
            except ValueError:
                raise ValueError(f"token '{char}' is not a valid integer")
    return eTree


# input_word = '( 4 * 8 ) / 6 - 3'
# etree = expression_tree(input_word)
# print(etree)


# -------------------------------------- Excercises -------------------------------------------------
# 3. Consider the following list of integers: [1,2,3,4,5,6,7,8,9,10].
# Show the binary search tree resulting from inserting the integers in the list.
alist = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]


class BST:
    def __init__(self, i):
        self.value = i
        self.lchild = None
        self.rchild = None
        self.parent = None

    def set_value(self, value):
        self.value = value

    def get_value(self):
        return self.value

    def set_lchild(self, node):
        self.lchild = node
        node.parent = self

    def set_rchild(self, node):
        self.rchild = node
        node.parent = self

    def get_lchild(self):
        return self.lchild

    def get_rchild(self):
        return self.rchild

    def put_node(self, node, i):
        new_node = BST(i)
        print("====node.value", node.value)

        if node == None or node.value == None:
            node = new_node
            print("item ", i, "is child of node", node.value)

        if i == node.value:
            return node

        elif i > node.value:
            print(i, ">", node.value)
            if node.rchild == None:
                print("item ", i, "is rchild of node", node.value)
                node.rchild = new_node
                node.set_rchild(new_node)
            else:
                node.put_node(node.rchild, i)
        else:
            print(i, "<", node.value)
            if node.lchild == None:
                print("item ", i, "is lchild of node", node.value)
                node.lchild = new_node
                node.set_lchild(new_node)
            else:
                node.put_node(node.lchild, i)
        return node


# eTree = BST(None)
# alist = [1,2,3,4,5,6,7,8,9,10]
# alist = [17, 5, 35, 2, 16, 29, 38, 33, 19]
# for i in alist:
#     print(i)
#     eTree = eTree.put_node(eTree, i)


# -------------------------------------- Excercises -------------------------------------------------
# 4. Consider the following list of integers: [10,9,8,7,6,5,4,3,2,1].
# Show the binary search tree resulting from inserting the integers in the list.
# eTree = BST(None)
# alist = [10,9,8,7,6,5,4,3,2,1]
# for i in alist:
#     print(i)
#     eTree = eTree.put_node(eTree, i)


# -------------------------------------- Excercises -------------------------------------------------
# 5. Generate a random list of integers.
# Show the binary heap tree resulting from inserting the integers on the list one at a time.
class BHT:
    def __init__(self):
        self.heapList = [0]
        self.currentSize = 0

    def insert(self, k):
        print("insert ", k)
        self.heapList.append(k)
        print("self.heapList: ", self.heapList)
        self.currentSize += 1
        self.percUp(self.currentSize)
        print(self.heapList)

    def percUp(self, i):
        while i // 2 > 0:
            # 第一个数字大    8，5, 6-> 5，6, 8
            if self.heapList[i] < self.heapList[i // 2]:
                print(
                    " change the position ",
                    self.heapList[i],
                    "<-",
                    self.heapList[i // 2],
                )
                tmp = self.heapList[i // 2]
                self.heapList[i // 2] = self.heapList[i]
                self.heapList[i] = tmp
            i = i // 2


# bhTree = BHT()
# alist = [5,9,11,14,18,19,21,33,17,27]
# for i in alist:
#     bhTree.insert(i)
# bhTree.insert(7)

# insert  5
# self.heapList:  [0, 5]
# [0, 5]
# insert  9
# self.heapList:  [0, 5, 9]
# [0, 5, 9]
# insert  11
# self.heapList:  [0, 5, 9, 11]
# [0, 5, 9, 11]
# insert  14
# self.heapList:  [0, 5, 9, 11, 14]
# [0, 5, 9, 11, 14]
# insert  18
# self.heapList:  [0, 5, 9, 11, 14, 18]
# [0, 5, 9, 11, 14, 18]
# insert  19
# self.heapList:  [0, 5, 9, 11, 14, 18, 19]
# [0, 5, 9, 11, 14, 18, 19]
# insert  21
# self.heapList:  [0, 5, 9, 11, 14, 18, 19, 21]
# [0, 5, 9, 11, 14, 18, 19, 21]
# insert  33
# self.heapList:  [0, 5, 9, 11, 14, 18, 19, 21, 33]
# [0, 5, 9, 11, 14, 18, 19, 21, 33]
# insert  17
# self.heapList:  [0, 5, 9, 11, 14, 18, 19, 21, 33, 17]
# [0, 5, 9, 11, 14, 18, 19, 21, 33, 17]
# insert  27
# self.heapList:  [0, 5, 9, 11, 14, 18, 19, 21, 33, 17, 27]
# [0, 5, 9, 11, 14, 18, 19, 21, 33, 17, 27]
# insert  7
# self.heapList:  [0, 5, 9, 11, 14, 18, 19, 21, 33, 17, 27, 7]
#  change the position  7 <- 18
#  change the position  7 <- 9
# [0, 5, 7, 11, 14, 9, 19, 21, 33, 17, 27, 18]


# -------------------------------------- Excercises -------------------------------------------------
# 6. Using the list from the previous question, show the binary heap tree resulting from using the list as a parameter to the buildHeap method. Show both the tree and list form.


# -------------------------------------- Excercises -------------------------------------------------
# 7. Draw the binary search tree that results from inserting the following keys in the order given:
# 68,88,61,89,94,50,4,76,66, and 82.


# -------------------------------------- Excercises -------------------------------------------------
# 8. Generate a random list of integers.
# Draw the binary search tree resulting from inserting the integers on the list.


# -------------------------------------- Excercises -------------------------------------------------
# 9. Consider the following list of integers: [1,2,3,4,5,6,7,8,9,10].
# Show the binary heap resulting from inserting the integers one at a time.
# bhTree = BHT()
# alist = [1,2,3,4,5,6,7,8,9,10]
# for i in alist:
#     bhTree.insert(i)

# insert  1
# self.heapList:  [0, 1]
# [0, 1]
# insert  2
# self.heapList:  [0, 1, 2]
# [0, 1, 2]
# insert  3
# self.heapList:  [0, 1, 2, 3]
# [0, 1, 2, 3]
# insert  4
# self.heapList:  [0, 1, 2, 3, 4]
# [0, 1, 2, 3, 4]
# insert  5
# self.heapList:  [0, 1, 2, 3, 4, 5]
# [0, 1, 2, 3, 4, 5]
# insert  6
# self.heapList:  [0, 1, 2, 3, 4, 5, 6]
# [0, 1, 2, 3, 4, 5, 6]
# insert  7
# self.heapList:  [0, 1, 2, 3, 4, 5, 6, 7]
# [0, 1, 2, 3, 4, 5, 6, 7]
# insert  8
# self.heapList:  [0, 1, 2, 3, 4, 5, 6, 7, 8]
# [0, 1, 2, 3, 4, 5, 6, 7, 8]
# insert  9
# self.heapList:  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
# [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
# insert  10
# self.heapList:  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]


# -------------------------------------- Excercises -------------------------------------------------
# 10. Consider the following list of integers: [10,9,8,7,6,5,4,3,2,1].
# Show the binary heap resulting from inserting the integers one at a time.

# bhTree = BHT()
# alist = [10,9,8,7,6,5,4,3,2,1]
# for i in alist:
#     bhTree.insert(i)

# insert  10
# self.heapList:  [0, 10]
# [0, 10]
# insert  9
# self.heapList:  [0, 10, 9]
#  change the position  9 <- 10
# [0, 9, 10]
# insert  8
# self.heapList:  [0, 9, 10, 8]
#  change the position  8 <- 9
# [0, 8, 10, 9]
# insert  7
# self.heapList:  [0, 8, 10, 9, 7]
#  change the position  7 <- 10
#  change the position  7 <- 8
# [0, 7, 8, 9, 10]
# insert  6
# self.heapList:  [0, 7, 8, 9, 10, 6]
#  change the position  6 <- 8
#  change the position  6 <- 7
# [0, 6, 7, 9, 10, 8]
# insert  5
# self.heapList:  [0, 6, 7, 9, 10, 8, 5]
#  change the position  5 <- 9
#  change the position  5 <- 6
# [0, 5, 7, 6, 10, 8, 9]
# insert  4
# self.heapList:  [0, 5, 7, 6, 10, 8, 9, 4]
#  change the position  4 <- 6
#  change the position  4 <- 5
# [0, 4, 7, 5, 10, 8, 9, 6]
# insert  3
# self.heapList:  [0, 4, 7, 5, 10, 8, 9, 6, 3]
#  change the position  3 <- 10
#  change the position  3 <- 7
#  change the position  3 <- 4
# [0, 3, 4, 5, 7, 8, 9, 6, 10]
# insert  2
# self.heapList:  [0, 3, 4, 5, 7, 8, 9, 6, 10, 2]
#  change the position  2 <- 7
#  change the position  2 <- 4
#  change the position  2 <- 3
# [0, 2, 3, 5, 4, 8, 9, 6, 10, 7]
# insert  1
# self.heapList:  [0, 2, 3, 5, 4, 8, 9, 6, 10, 7, 1]
#  change the position  1 <- 8
#  change the position  1 <- 3
#  change the position  1 <- 2
# [0, 1, 2, 5, 4, 3, 9, 6, 10, 7, 8]


# -------------------------------------- Excercises -------------------------------------------------
# 11. Consider the two different techniques we used for implementing traversals of a binary tree.
# Why must we check before the call to preorder when implementing as a method, whereas we could check inside the call when implementing as a function?


# -------------------------------------- Excercises -------------------------------------------------
# 12. Show the function calls needed to build the following binary tree.
# class BHT():
#     def __init__(self):
#         self.heapList = [0]
#         self.currentSize = 0
#         self.value = [0]

#     def insert(self, value, k):
#         print("insert ", k)
#         self.heapList.append(k)
#         self.value.append(value)
#         print("self.heapList: ", self.heapList)
#         self.currentSize += 1
#         self.percUp(self.currentSize)
#         print(self.heapList)
#         print(self.value)

#     def percUp(self, i):
#         while i // 2 > 0:
#             # 第一个数字大    8，5, 6-> 5，6, 8
#             if self.heapList[i] < self.heapList[i // 2]:
#                 print(" change the position ", self.heapList[i], "<-", self.heapList[i // 2])
#                 tmp = self.heapList[i // 2]
#                 self.heapList[i // 2] = self.heapList[i]
#                 self.heapList[i] = tmp
#                 tmp = self.value[i // 2]
#                 self.value[i // 2] = self.value[i]
#                 self.value[i] = tmp
#             i= i//2

# lan_dic = {"language":0, "complied": 1, "interpreted":2, "C":3, "Java":4, "Python": 5, "Scheme":6}
# bhTree = BHT()
# for i in lan_dic:
#     print(lan_dic[i])
#     bhTree.insert(i, lan_dic[i])

# 0
# insert  0
# self.heapList:  [0, 0]
# [0, 0]
# [0, 'language']
# 1
# insert  1
# self.heapList:  [0, 0, 1]
# [0, 0, 1]
# [0, 'language', 'complied']
# 2
# insert  2
# self.heapList:  [0, 0, 1, 2]
# [0, 0, 1, 2]
# [0, 'language', 'complied', 'interpreted']
# 3
# insert  3
# self.heapList:  [0, 0, 1, 2, 3]
# [0, 0, 1, 2, 3]
# [0, 'language', 'complied', 'interpreted', 'C']
# 4
# insert  4
# self.heapList:  [0, 0, 1, 2, 3, 4]
# [0, 0, 1, 2, 3, 4]
# [0, 'language', 'complied', 'interpreted', 'C', 'Java']
# 5
# insert  5
# self.heapList:  [0, 0, 1, 2, 3, 4, 5]
# [0, 0, 1, 2, 3, 4, 5]
# [0, 'language', 'complied', 'interpreted', 'C', 'Java', 'Python']
# 6
# insert  6
# self.heapList:  [0, 0, 1, 2, 3, 4, 5, 6]
# [0, 0, 1, 2, 3, 4, 5, 6]
# [0, 'language', 'complied', 'interpreted', 'C', 'Java', 'Python', 'Scheme']


# -------------------------------------- Excercises -------------------------------------------------
# Given the following tree, perform the appropriate rotations to bring it back into balance.
#       B
#   A       E
#           |
#       D       F
#   c


class BST:
    def __init__(self, i):
        self.value = i
        self.bvalue = 0
        self.lchild = None
        self.rchild = None
        self.parent = None

    def set_value(self, value):
        self.value = value

    def get_value(self):
        return self.value

    def get_bvalue(self):
        return self.bvalue

    def set_lchild(self, node):
        self.lchild = node
        node.parent = self

    def set_rchild(self, node):
        self.rchild = node
        node.parent = self

    def get_lchild(self):
        return self.lchild

    def get_rchild(self):
        return self.rchild

    def islChild(self, node):
        return node.parent.lchild == node

    def isrChild(self, node):
        return node.parent.rchild == node

    def put_node(self, node, i):
        new_node = BST(i)
        print("====node.value", node.value)

        if node == None or node.value == None:
            node = new_node
            print("item ", i, "is child of node", node.value, "bvalue ", node.bvalue)

        if i == node.value:
            return node

        elif i > node.value:
            print(i, ">", node.value)
            if node.rchild == None:
                # node.rchild = new_node
                node.set_rchild(new_node)
                print(
                    "item ", i, "is rchild of node", node.value, "bvalue ", node.bvalue
                )
                new_node.updateBalance(new_node)

            else:
                node.put_node(node.rchild, i)
        else:
            print(i, "<", node.value)
            if node.lchild == None:
                # node.lchild = new_node
                node.set_lchild(new_node)
                print(
                    "item ", i, "is lchild of node", node.value, "bvalue ", node.bvalue
                )
                new_node.updateBalance(new_node)
            else:
                node.put_node(node.lchild, i)

        return node

    def updateBalance(self, node):
        print(node.value, "bvalue is ", node.bvalue)
        if node.bvalue > 1 or node.bvalue < -1:
            node.reban(node)
        if node.parent != None:
            if node.islChild(node):
                node.parent.bvalue += 1
                print(node.parent.value, " bvalue +1: ", node.bvalue)
            elif node.isrChild(node):
                node.parent.bvalue -= 1
                print(node.parent.value, " bvalue -1: ", node.bvalue)
            if node.parent.bvalue != 0:
                node.updateBalance(node.parent)

    def reban(self, node):
        # node.hight
        return


#       B
#   A       E
#           |
#       D       F
#   c
bTree = BST("B")
bTree = bTree.put_node(bTree, "A")
bTree = bTree.put_node(bTree, "E")
bTree = bTree.put_node(bTree, "D")
bTree = bTree.put_node(bTree, "F")
bTree = bTree.put_node(bTree.rchild.lchild, "C")


# -------------------------------------- Excercises -------------------------------------------------
# Using the following as a starting point, derive the equation that gives the updated balance factor for node D.


# 7.22. Programming Exercises


# -------------------------------------- Excercises -------------------------------------------------
# Extend the buildParseTree function to handle mathematical expressions that do not have spaces between every character.


# -------------------------------------- Excercises -------------------------------------------------
# Modify the buildParseTree and evaluate functions to handle boolean statements (and, or, and not). Remember that “not” is a unary operator, so this will complicate your code somewhat.


# -------------------------------------- Excercises -------------------------------------------------
# Using the findSuccessor method, write a non-recursive inorder traversal for a binary search tree.


# -------------------------------------- Excercises -------------------------------------------------
# Modify the code for a binary search tree to make it threaded. Write a non-recursive inorder traversal method for the threaded binary search tree. A threaded binary tree maintains a reference from each node to its successor.


# -------------------------------------- Excercises -------------------------------------------------
# Modify our implementation of the binary search tree so that it handles duplicate keys properly. That is, if a key is already in the tree then the new payload should replace the old rather than add another node with the same key.


# -------------------------------------- Excercises -------------------------------------------------
# Create a binary heap with a limited heap size. In other words, the heap only keeps track of the n most important items. If the heap grows in size to more than n items the least important item is dropped.


# -------------------------------------- Excercises -------------------------------------------------
# Clean up the printexp function so that it does not include an ‘extra’ set of parentheses around each number.


# -------------------------------------- Excercises -------------------------------------------------
# Using the buildHeap method, write a sorting function that can sort a list in 𝑂(𝑛log𝑛) time.


# -------------------------------------- Excercises -------------------------------------------------
# Write a function that takes a parse tree for a mathematical expression and calculates the derivative of the expression with respect to some variable.


# -------------------------------------- Excercises -------------------------------------------------
# Implement a binary heap as a max heap.


# -------------------------------------- Excercises -------------------------------------------------
# Using the BinaryHeap class, implement a new class called PriorityQueue. Your PriorityQueue class should implement the constructor, plus the enqueue and dequeue methods.
