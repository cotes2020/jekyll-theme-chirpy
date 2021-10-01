


# https://runestone.academy/runestone/books/published/pythonds/Trees/DiscussionQuestions.html

# 7.21. Discussion Questions



# -------------------------------------- Excercises -------------------------------------------------
# Draw the tree structure resulting from the following set of tree function calls:
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
# Trace the algorithm for creating an expression tree for the expression (4∗8)/6−3. 
from _typeshed import Self
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
    eTree = BinaryTree('')
    tStack.push(eTree)
    cur_tree = eTree
    for char in char_list:
        if char == "(":
            cur_tree.insertLeft('')
            tStack.push(cur_tree)
            cur_tree = cur_tree.getLeftChild()
        elif char == ')':
            cur_tree = tStack.pop()
        elif char in ['+', '-', '*', '/']:
            cur_tree.setRootVal(char)
            cur_tree.insertRight('')
            tStack.push(cur_tree)
            cur_tree = cur_tree.getRightChild()
        else:
            print(char)
            try:
                cur_tree.setRootVal(int(char))
                cur_tree = tStack.pop()
            except ValueError:
                raise ValueError("token '{}' is not a valid integer".format(char))
    return eTree
# input_word = '( 4 * 8 ) / 6 - 3'
# etree = expression_tree(input_word)
# print(etree)



# -------------------------------------- Excercises -------------------------------------------------
# Consider the following list of integers: [1,2,3,4,5,6,7,8,9,10]. Show the binary search tree resulting from inserting the integers in the list.
alist = [1,2,3,4,5,6,7,8,9,10]

from pythonds.trees import BinaryTree

class BST:
    def __init__(self, i):
        self.value = i
        self.lchild = None
        self.rchild = None 
        self.parent = None

    def set_value(self, value): self.value = value
    
    def get_value(self): return self.value

    def add_lchild(self, node): 
        self.lchild = node
        node.parent = self

    def add_rchild(self, node): 
        self.rchild = node
        node.parent = self

    def get_lchild(self): return self.lchild
    def get_rchild(self): return self.rchild

    def put_node(self, node, i): 
        new_node = BST(i)
        print("====node.value", node.value)

        if node == None or node.value == None:
            node = new_node
            print("item ", i, "is child of node", node.value)
            
        if i == node.value: return node

        elif i > node.value:
            print(i, ">", node.value)
            if node.rchild == None: 
                print("item ", i, "is rchild of node", node.value)
                node.rchild = new_node
                node.add_rchild(new_node)
            else: node.put_node(node.rchild, i)
        else:
            print(i, "<", node.value)
            if node.lchild == None: 
                print("item ", i, "is lchild of node", node.value)
                node.lchild = new_node
                node.add_lchild(new_node)
            else: node.put_node(node.lchild, i)
        return node

# eTree = BST(None)
# alist = [1,2,3,4,5,6,7,8,9,10]
# alist = [17, 5, 35, 2, 16, 29, 38, 33, 19]
# for i in alist:
#     print(i)
#     eTree = eTree.put_node(eTree, i)
 

# -------------------------------------- Excercises -------------------------------------------------
# Consider the following list of integers: [10,9,8,7,6,5,4,3,2,1]. Show the binary search tree resulting from inserting the integers in the list.
eTree = BST(None)
alist = [10,9,8,7,6,5,4,3,2,1]
for i in alist:
    print(i)
    eTree = eTree.put_node(eTree, i)



# -------------------------------------- Excercises -------------------------------------------------
# Generate a random list of integers. Show the binary heap tree resulting from inserting the integers on the list one at a time.
class BHT():

    def __init__(self, value):
        self.value = value
        self.lchild = None
        self.rchild = None
        self.parent = None

    def has_lchild(self): return self.lchild != None
    def has_rchild(self): return self.rchild != None
    
    def add(self, alist, index):
        new_node = BHT(alist[index])
        if self.value == None:
            return new_node
        else:
            if not self.has_lchild and not self.has_rchild:
                self.lchild == new_node
                new_node.parent = self
            if self.has_lchild and self.has_rchild:
                self.add(self.lchild, alist, index)
            if self.has_lchild and not self.has_rchild:
                self.rchild == new_node
                new_node.parent = self
        return self


def build_BHT(self, bht, alist, index):
    new_node = BHT(alist[index])
    if bht.value == None:
        return new_node
    else:
        pre_BHT = build_BHT(self, alist, bht, index-1)
        new_BHT = pre_BHT.add(alist, index)
    return new_BHT

bhTree = BHT(None)
alist = [10,9,8,7,6,5,4,3,2,1]
for i in alist:
    print(i)
    eTree = bhTree.add(i)


# -------------------------------------- Excercises -------------------------------------------------
# Using the list from the previous question, show the binary heap tree resulting from using the list as a parameter to the buildHeap method. Show both the tree and list form.



# -------------------------------------- Excercises -------------------------------------------------
# Draw the binary search tree that results from inserting the following keys in the order given: 68,88,61,89,94,50,4,76,66, and 82.



# -------------------------------------- Excercises -------------------------------------------------
# Generate a random list of integers. Draw the binary search tree resulting from inserting the integers on the list.



# -------------------------------------- Excercises -------------------------------------------------
# Consider the following list of integers: [1,2,3,4,5,6,7,8,9,10]. Show the binary heap resulting from inserting the integers one at a time.



# -------------------------------------- Excercises -------------------------------------------------
# Consider the following list of integers: [10,9,8,7,6,5,4,3,2,1]. Show the binary heap resulting from inserting the integers one at a time.



# -------------------------------------- Excercises -------------------------------------------------
# Consider the two different techniques we used for implementing traversals of a binary tree. Why must we check before the call to preorder when implementing as a method, whereas we could check inside the call when implementing as a function?



# -------------------------------------- Excercises -------------------------------------------------
# Show the function calls needed to build the following binary tree.
 


# -------------------------------------- Excercises -------------------------------------------------
# Given the following tree, perform the appropriate rotations to bring it back into balance.

 


# -------------------------------------- Excercises -------------------------------------------------
# Using the following as a starting point, derive the equation that gives the updated balance factor for node D.