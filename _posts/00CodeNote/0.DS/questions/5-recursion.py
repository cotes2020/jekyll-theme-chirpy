# --------------------------------------------------------------------------------------------------
# check the reverse words
def reverse(s):
    # print(s)
    if len(s) <= 1:
        s = s
    elif len(s) <=2:
        s = s[1] + s[0]
    else:
        s = reverse(s[1:]) + s[0]
    # print(s)
    return s

# print(reverse("hello")=="olleh")
# print(reverse("l")=="l")
# print(reverse("follow")=="wollof")
# print(reverse("")=="")


# --------------------------------------------------------------------------------------------------
# check the mirror words
def removeWhite(s):
    s = s.replace(' ', '').replace("'",'').replace('"','')
    return s

def isPal(s):
    if len(s) <= 1:
        # print(s)
        return True
    if len(s) == 2:
        # print(s)
        return s[0] == s[-1]
    else:
        return isPal(s[0]+s[-1]) and isPal(s[1:-1])

# print(isPal("x"))
# print(isPal("radar"))
# print(isPal("hello"))
# print(isPal(""))
# print(isPal("hannah"))
# print(isPal(removeWhite("madam i'm adam")))

# testEqual(isPal(removeWhite("x")),True)
# testEqual(isPal(removeWhite("radar")),True)
# testEqual(isPal(removeWhite("hello")),False)
# testEqual(isPal(removeWhite("")),True)
# testEqual(isPal(removeWhite("hannah")),True)
# testEqual(isPal(removeWhite("madam i'm adam")),True)



# -------------------------------------- Exercises -------------------------------------------------
# exchange the coins

def dpMakeChange(coinValueList,change,minCoins,coinsUsed):
   for cents in range(change+1):
      coinCount = cents
      newCoin = 0
      for j in [c for c in coinValueList if c <= cents]:
            if minCoins[cents-j] + 1 <= coinCount:
               coinCount = minCoins[cents-j]+1
               newCoin = j
      minCoins[cents] = coinCount
      coinsUsed[cents] = newCoin
   print(minCoins)
   print(coinsUsed)
   return minCoins[change]
# Making change for 63 requires

# amnt = 63
# clist = [1,5,10,21,25]
# coinsUsed = [0]*(amnt+1)
# coinCount = [0]*(amnt+1)
# print("Making change for",amnt,"requires")
# print(dpMakeChange(clist,amnt,coinCount,coinsUsed),"coins")

# minCoins: change for 0, for 1, for 2 ....
# [0, 1, 2, 3, 4, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 2, 3, 4, 5, 6, 2, 1, 2, 3, 4, 1, 2, 3, 4, 5, 2, 2, 3, 4, 5, 2, 3, 4, 5, 6, 3, 3, 2, 3, 4, 3, 2, 3, 4, 5, 2, 3, 3, 4, 5, 3, 3, 4, 5, 6, 3, 4, 4, 3]

# [1, 1, 1, 1, 1, 5, 1, 1, 1, 1, 10, 1, 1, 1, 1, 5, 1, 1, 1, 1, 10, 21, 1, 1, 1, 25, 1, 1, 1, 1, 5, 10, 1, 1, 1, 10, 1, 1, 1, 1, 5, 10, 21, 1, 1, 10, 21, 1, 1, 1, 25, 1, 10, 1, 1, 5, 10, 1, 1, 1, 10, 1, 10, 21]

# printCoins that walks backward through the table to print out the value of each coin used. This shows the algorithm in action solving the problem for our friends in Lower Elbonia. The first two lines of main set the amount to be converted and create the list of coins used. The next two lines create the lists we need to store the results. coinsUsed is a list of the coins used to make change, and coinCount is the minimum number of coins used to make change for the amount corresponding to the position in the list.
def printCoins(coinsUsed,change):
   coin = change
   while coin > 0:
      thisCoin = coinsUsed[coin]
      print(thisCoin)
      coin = coin - thisCoin

def main():
    amnt = 63
    clist = [1,5,10,21,25]
    coinsUsed = [0]*(amnt+1)
    coinCount = [0]*(amnt+1)

    print('Making change for',amnt,'requires')
    print(dpMakeChange(clist,amnt,coinCount,coinsUsed),'coins')
    print('They are:')
    printCoins(coinsUsed,amnt)
    print('The used list is as follows:')
    print(coinsUsed)

# main()




# -------------------------------------- Exercises -------------------------------------------------
# Write a recursive function to compute the factorial of a number.
# Factorial of a non-negative integer, is multiplication of all integers smaller than or equal to n.
# For example factorial of 6 is 6*5*4*3*2*1 which is 720.

# recursion:
def factorial_recursion(number):
    if number == 1:
        factorial = 1
    else:
        factorial = number * factorial_recursion(number-1)
    print(factorial)
    return factorial

def factorial_recursion(number):
    if number <= 1: return 1
    return number * factorial_recursion(number-1)

def factorial_recursion(number):
    return 1 if (number == 1 or number == 0) else number*factorial_recursion(number-1)

# Iterative Solution: O(n)
# Factorial can also be calculated iteratively as recursion can be costly for large numbers.
# Here we have shown the iterative approach using both for and while loop.
def factorial(n):
    res = 1
    for i in range(2, n+1):
        res *= i
    return res

def factorial(n):
    if(n == 0): return 1
    i = n
    fact = 1
    while(n / i != n):
        fact = fact * i
        i -= 1
    return fact
# print(factorial_recursion(6))


# Factorial of a large number
# not possible to store these many digits even if we use long long int.
# Input : 100
# Output : 933262154439441526816992388562667004-
#          907159682643816214685929638952175999-
#          932299156089414639761565182862536979-
#          208272237582511852109168640000000000-
#          00000000000000

# Input :50
# Output : 3041409320171337804361260816606476884-
#          4377641568960512000000000000
# 1. use an array to store individual digits of the result. The idea is to use basic mathematics for multiplication.





import random
# -------------------------------------- Exercises -------------------------------------------------
# Modify the recursive tree program using one or all of the following ideas:
# Modify the thickness of the branches so that as the branchLen gets smaller, the line gets thinner.
# Modify the color of the branches so that as the branchLen gets very short it is colored like a leaf.
# Modify the angle used in turning the turtle so that at each branch point the angle is selected at random in some range. For example choose the angle between 15 and 45 degrees. Play around to see what looks good.
# Modify the branchLen recursively so that instead of always subtracting the same amount you subtract a random amount in some range.
# If you implement all of the above ideas you will have a very realistic looking tree.
import turtle


def tree(branchLen,t, wid, color):
    min_len = random.randint(14,17)
    # min_len = 15
    if branchLen > 5:
        angle = random.randint(15,45)
        t.width(wid)
        t.color(color)
        r,g,b = color
        t.forward(branchLen)
        t.right(angle)
        tree(branchLen-min_len, t, wid-5, (r+10,g+20,b+10))
        t.left(angle*2)
        tree(branchLen-min_len, t, wid-5, (r+10,g+20,b+10))
        # back to center
        t.right(angle)
        t.backward(branchLen)

def main():
  t = turtle.Turtle()
  myWin = turtle.Screen()
  myWin.colormode(255)
  t.left(90)
  t.up()
  t.backward(100)
  t.down()
  t.color((50,100,20))
  tree(85, t, 25, (50,100,20))
  myWin.exitonclick()
# main()




# -------------------------------------- Exercises -------------------------------------------------
# Write a recursive function to compute the Fibonacci sequence. How does the performance of the recursive function compare to that of an iterative version?

def Fibonacci(number):
    arr = {}
    arr[0] = 0
    arr[1] = 1
    arr[2] = 1
    arr[3] = 3
    arr[4] = 5
    if number in arr.keys():
        return arr[number]
    for i in range(5, number+1):
        arr[i] = i-1 + i-2
    return arr[number]
# print(Fibonacci(8))




# -------------------------------------- Exercises -------------------------------------------------
# Write a program to solve the following problem:
# You have two jugs: a 4-gallon jug and a 3-gallon jug.
# Neither of the jugs have markings on them.
# There is a pump that can be used to fill the jugs with water.
# How can you get exactly two gallons of water in the 4-gallon jug?

# The operations you can perform are:
# Empty a Jug
# Fill a Jug
# Pour water from one jug to the other until one of the jugs is either empty or full.

# m < n
# Solution 1 (Always pour from m liter jug into n liter jug)
# Fill the m litre jug and empty it into n liter jug.
# Whenever the m liter jug becomes empty fill it.
# Whenever the n liter jug becomes full empty it.
# Repeat steps 1,2,3 till either n liter jug or the m liter jug contains d litres of water.

class Tree:
    def __init__(self, data=None):
      self.child = []
      self.parent = None
      self.data = data

    def printTree(self):
        if self.child == []:
            print('root ', self.data, 'have no child')
        else:
            root = self
            print('root ', root.data, 'have child')
            leaf = {}
            child_1st_list = []
            child_1st = root.child
            for i in child_1st:
                print('1st child:', i.data)
                self.printChild(i)

    def printNode(self, Node):
        print(Node.data)

    def printChild(self, node):
        if node.child == []:
            print(node.data, 'has no child')
        else:
            child_list = []
            for i in node.child:
                child_list.append(i.data)
            leaf = {'parent': node.data, 'child': child_list}
            print(leaf)

    def insert(self, parent_node, child_node):
        if parent_node.data != None:
            child_node.parent = parent_node
            parent_node.child.append(child_node)
        else:
            print('not tree yet')
            print('setup tree')
            self.data = child_node.data

# t = Tree((0,0))
# # root.printTree()

# child1 = Tree((0,1))
# print(" +++++++ add child 1")
# t.insert(t, child1)
# # t.printTree()
# # t.printChild(t)

# child2 = Tree((0,2))
# print(" +++++++ add child 2")
# t.insert(t, child2)
# # t.printChild(child1)

# child3 = Tree((0,3))
# print(" +++++++ add child 3")
# t.insert(child1, child3)
# t.printChild(child1)

# t.printTree()


def add_water(big, small, big_size, small_size):
    # Whenever the m liter jug becomes empty fill it.
    if small == 0:
        print('refill small')
        small = small_size
    # Whenever the n liter jug becomes full empty it.
    if big == big_size:
        print('empty big')
        big = 0
    # Fill the m litre jug and empty it into n liter jug.
    if big+small < big_size:
        print('big <- all small')
        big = small
        small = 0
    else:
        print('big <- small, small still have some')
        small = big+small - big_size
        big = big_size
    print('end:', big, small)
    return (big, small)

def jugs(target, jug_list):
    big, small = (0,0)
    big_size, small_size = jug_list
    number_dic = {}

    i = 0
    while target != big:
        print(' ============= step:', i)
        print('start:', big, small)
        big, small = add_water(big, small, big_size, small_size)
        if big not in number_dic.keys():
            number_dic[big] = i
        target_index = i
        i += 1

    for i in number_dic.keys():
        print('for number: ', i, 'need step: ', number_dic[i])

    print(' ============= for target: ', target, 'need step: ', target_index)

# jugs(1, (4, 3))


# -------------------------------------- Exercises -------------------------------------------------
# Write a program that solves the following problem:
# Three missionaries and three cannibals come to a river and find a boat that holds two people.
# Everyone must get across the river to continue on the journey.
# However, if the cannibals ever outnumber > the missionaries on either bank, the missionaries will be eaten.
# Find a series of crossings that will get everyone safely to the other side of the river.

# initial configuration (3 missionaries and cannibals)


# start = {'c': 3, 'm': 3, 'b':1}
# end = {'c': 0, 'm': 0, 'b':0}
# on_boat = {'c': 0, 'm': 0}

class MandC():
    def __init__(self, start, end, on_boat):
        #左岸传教士数量
        self.m = start['m']
        #左岸野人数量
        self.c = start['c']
        # b = 1: 船在左岸；
        # b = 0: 船在右岸
        self.b = start['b']
        self.g = 0
        self.f = 0  #f = g+h
        self.father = None
        self.node = [m, c, b]

    def print_state(self, start, end, on_boat):
        print(start, end, on_boat)

    ### decorator
    @classmethod
    def root(cls):
        return cls((3,3,1))

    def is_legal(self, location):
        # location = start/end
        missionaries, cannibals = location['m'], location['c']
        if 0 <= missionaries <= 3 and 0 <= cannibals <= 3:
            return True
        else: return False

    def is_solution(self):
        return start == {'c': 0, 'm': 0}

    def is_failure(self, start):
        missionaries, cannibals = start['m'], start['c']
        # boat = self.state_vars[2]

        ### missionaries on right side AND more cannibals than missionaries
        if missionaries > 0 or cannibals > 0:
            return True
        return

    def get_possible_moves(self):
        # moves = [(1, 0), (2, 0), (0, 1), (0, 2), (1, 1)]
        # 3 action
        action_dict = {
            'right': [
                # 2 on boat
                {'c': 1, 'm': 1, 'b': 1},
                {'c': 0, 'm': 2, 'b': 1},
                {'c': 2, 'm': 0, 'b': 1},
                # 1 on boat
                {'c': 0, 'm': 1, 'b': 1},
                {'c': 1, 'm': 0, 'b': 1},
            ],
            'left': [
                # 2 on boat
                {'c': 1, 'm': 1, 'b': 0},
                {'c': 0, 'm': 2, 'b': 0},
                {'c': 2, 'm': 0, 'b': 0},
                # 1 on boat
                {'c': 0, 'm': 1, 'b': 0},
                {'c': 1, 'm': 0, 'b': 0},
            ]
        }
        return action_dict

    def __str__(self):
        return 'MCState[{}]'.format(self.state_vars)

    def __repr__(self):
        return str(self)


    def check_status(self, start, end, action, possible_list):
        if start == {'c': 0, 'm': 0, 'b': 0}:
            status = 'finished'
            return [action], True
        elif start['c'] > start['m'] and start['m'] != 0:
            status = 'GG'
        elif end['c'] > end['m'] and end['m'] != 0:
            status = 'GG'
        else:
            status = 'safe'
            possible_list.append(action)
        print(status)
        return possible_list, False


    def gogogo(self, from_loc, to_loc, move):
        """
        from_loc = {'c': 3, 'm': 3, 'b':1}
        to_loc = {'c': 0, 'm': 0, 'b':0}
        move : {'c': 1, 'm': 1},
        """
        mock_from = dict(from_loc)
        mock_to = dict(to_loc)
        if mock_from['c'] >= move['c'] and mock_from['m'] >= move['m']:
            print(' ============ ', mock_from, ' - boat move: ', move)
            mock_from['c'] = mock_from['c'] - move['c']
            mock_from['m'] = mock_from['m'] - move['m']
            mock_to['c'] = mock_to['c'] + move['c']
            mock_to['m'] = mock_to['m'] + move['m']
            mock_from['b'] = 0
            mock_to['b'] = 1
        else:
            print(' ============ boat move can not be done:', move)
        return mock_from, mock_to


    def movement(self, start, end):
        action_dict = self.get_possible_moves()
        possible_list = []
        boat = start['b']

        if boat == 1:
            for action in action_dict['right']:
                print('++++++ boat from start to end')
                print('++++++ form', start)
                mock_from, mock_to = self.gogogo(start, end, action)
                print(' ============ start_status: ', mock_from)
                print(' ============ end_status: ', mock_to)
                possible_list, stop_process = \
                    self.check_status(mock_from, mock_to, action, possible_list)
        else:
            for action in action_dict['left']:
                print('++++++ boat from end to start')
                print('++++++ form', end)
                mock_from, mock_to = self.gogogo(end, start, action)
                print(' ============ start_status: ', mock_to)
                print(' ============ end_status: ', mock_from)
                possible_list, stop_process = \
                    self.check_status(mock_to, mock_from, action, possible_list)
            if stop_process:
                return start, end, possible_list, stop_process
        print(possible_list)
        return start, end, possible_list, stop_process


    def main(self, start, end):
        action_dict = self.get_possible_moves()
        for action in action_dict['right']:
            if start['c'] == action['c'] \
                and start['m'] == action['m'] \
                and start['b'] == action['b']:
                print('action = one boat')
                return [action]
        # if start == {'c': 1, 'm': 1, 'b':1}:
        #     action = 'one more boat'
        # if start == {'c': 0, 'm': 1, 'b':1}:
        #     action = 'one more boat'
        # if start == {'c': 0, 'm': 2, 'b':1}:
        #     action = 'one more boat'
        # if start == {'c': 1, 'm': 0, 'b':1}:
        #     action = 'one more boat'
        # if start == {'c': 2, 'm': 0, 'b':1}:
        #     action = 'one more boat'
        else:
            start, end, possible_list, stop_process = self.movement(start, end)
            if stop_process == True:
                return possible_list
            else:
                print('need more')
                for i in possible_list:
            # possible_list = self.main(start, end)
        return possible_list

# start = {'c': 3, 'm': 3, 'b':1}
# start = {'c': 1, 'm': 2, 'b':1}
start = {'c': 1, 'm': 1, 'b':1}
end = {'c': 0, 'm': 0, 'b':0}
on_boat = {'c': 0, 'm': 0}

run = MandC(start, end, on_boat)
# run.movement(start, end)

# possible_list = run.move(start, end)
# print("possible_list:", possible_list)

action = run.main(start, end)
print('action:', action)




class State():
    def __init__(self, start, end, on_boat):
        #左岸传教士数量
        self.m = start['m']
        #左岸野人数量
        self.c = start['c']
        # b = 1: 船在左岸；
        # b = 0: 船在右岸
        self.b = start['b']
        self.g = 0
        self.f = 0  #f = g+h
        self.father = None
        self.node = [m, c, b]


M = 3  # 传教士
C = 3  # 野人
K = 2  # 船的最大容量

child = []  # child用来存所有的拓展节点
open_list = []  # open表
closed_list = []  # closed表


init = State(M, C, 1)  # 初始节点
goal = State(0, 0, 0)  # 目标

#0 ≤ m ≤ 3,0 ≤ c ≤ 3, b ∈ {0,1}, 左岸m > c(m 不为 0 时), 右岸3-m > 3-c(m 不为 3 时)
def safe(s):
    if s.m > M or s.m < 0 \
       or s.c > C or s.c < 0 \
       or (s.m != 0 and s.m < s.c) \
       or (s.m != M and M - s.m < C - s.c):
        return False
    else:
        return True

# 启发函数
def h(s):
    return s.m + s.c - K * s.b
    # return M - s.m + C - s.c

def equal(a, b):
    if a.node == b.node:
        return 1,b
    else:
        return 0,b

# 判断当前状态与父状态是否一致
def back(new, s):
    if s.father is None:
        return False
    #判断当前状态与祖先状态是否一致
    c=b=s.father
    while(1):
        a,c=equal(new, b)
        if a:
            return True
        b=c.father
        if b is None:
            return False
# 将open_list以f值进行排序
def open_sort(l):
    the_key = operator.attrgetter('f')  # 指定属性排序的key
    l.sort(key=the_key)


# 扩展节点时在open表和closed表中找原来是否存在相同mcb属性的节点
def in_list(new, l):
    for item in l:
        if new.node == item.node:
            return True, item
    return False, None


def A_star(s):
    A=[]
    global open_list, closed_list
    open_list = [s]
    closed_list = []
    #print(len(open_list))
    # print （'closed list:'）  # 选择打印open表或closed表变化过程
    #print(s.node)
    #a=1
    while(1):  # open表非空
        #get = open_list[0]  # 取出open表第一个元素get
        for i in open_list:
            if i.node == goal.node:  # 判断是否为目标节点
                A.append(i)
                open_list.remove(i)
        if not(open_list):
            break
        get=open_list[0]
        open_list.remove(get)  # 将get从open表移出
        closed_list.append(get)  # 将get加入closed表

        # 以下得到一个get的新子节点new并考虑是否放入openlist
        for i in range(M+1):  # 上船传教士
            for j in range(C+1):  # 上船野人
                # 船上非法情况
                if i + j == 0 or i + j > K or (i != 0 and i < j):
                    continue
                #a=a+1
                if get.b == 1:  # 当前船在左岸，下一状态统计船在右岸的情况
                    new = State(get.m - i, get.c - j, 0)
                    child.append(new)
                    #print(1)
                else:  # 当前船在右岸，下一状态统计船在左岸的情况
                    new = State(get.m + i, get.c + j, 1)
                    child.append(new)
                    #print(2)
                #优先级：not>and>ture。如果状态不安全或者要拓展的节点与当前节点的父节点状态一致。
                if not safe(new) or back(new, get):  # 状态非法或new折返了
                    child.pop()
                #如果要拓展的节点满足以上情况，将它的父亲设为当前节点，计算f，并对open_list排序
                else:
                    new.father = get
                    new.g = get.g + 1  #与起点的距离
                    new.f = get.g + h(get)  # f = g + h

                    open_list.append(new)
                    #print(len(open_list))
                    open_sort(open_list)
        # 打印open表或closed表
        #for o in open_list:
        # for o in closed_list:
            #print(o)
            #print(o.node)
           # print(o.father)
        #print(a)
    return(A)


# 递归打印路径
def printPath(f):
    if f is None:
        return
    printPath(f.father)
    #注意print()语句放在递归调用前和递归调用后的区别。放在后实现了倒叙输出
    print(f.node )


if __name__ == '__main__':
    print ('有%d个传教士，%d个野人，船容量:%d' % (M, C, K))
    final = A_star(init)
    print('有{}种方案'.format(len(final)))
    if final:
        for i in(final):
            print ('有解，解为：')
            printPath(i)
    else:
        print ('无解！')



# -------------------------------------- Exercises -------------------------------------------------
# Suppose you are a computer scientist/art thief who has broken into a major art gallery.
# All you have with you to haul out your stolen art is your knapsack which only holds W pounds of art,
# but for every piece of art you know its value and its weight.
# Write a dynamic programming function to help you maximize your profit.
# Here is a sample problem for you to use to get started:
# Suppose your knapsack can hold a total weight of 20. You have 5 items as follows:
# item     weight      value
#   1        2           3
#   2        3           4
#   3        4           8
#   4        5           8
#   5        9          10




# -------------------------------------- Exercises -------------------------------------------------
# This problem is called the string edit distance problem, and is quite useful in many areas of research. Suppose that you want to transform the word “algorithm” into the word “alligator.” For each letter you can either copy the letter from one word to another at a cost of 5, you can delete a letter at cost of 20, or insert a letter at a cost of 20. The total cost to transform one word into another is used by spell check programs to provide suggestions for words that are close to one another. Use dynamic programming techniques to develop an algorithm that gives you the smallest edit distance between any two words.
