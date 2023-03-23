from pythonds.basic import Stack

# class Stack:
#     """Stack implementation as a list"""
#     def __init__(self): self.items = []
#     def is_empty(self): return not bool(self.items)
#     def push(self, item): self.items.append(item)
#     def pop(self): return self.items.pop()
#     # append() and pop() operations were both O(1)
#     def peek(self): return self.items[-1]
#     def size(self): return len(self.items)


# use stack -> Infix-to-Postfix
# (A+B+D)*C -> (AB+D+)C*
# A*B+C*D -> AB*+CD*
def infixToPostfix(infixexpr):
    # Assume the infix expression is a string of tokens delimited by spaces.
    # The operator tokens are *, /, +, and -, along with the left and right parentheses, ( and ).
    # The operand tokens are the single-tokenacter identifiers A, B, C, and so on.
    # The following steps will produce a string of tokens in postfix order.
    prec = {"*": 3, "/": 3, "+": 2, "-": 2, "(": 1, "**": 4}
    operand_tokens = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

    # Create an empty stack called opStack for keeping operators. Create an empty list for output.
    opStack = Stack()
    postfixList = []
    # Convert the input infix string to a list by using the string method split.
    token_list = infixexpr.split()

    # Scan the token list from left to right.
    for token in token_list:
        print(token)
        # If the token is an operand, append it to the end of the output list.
        if token in operand_tokens:
            postfixList.append(token)
            print("postfixList.append:", token, "add operand")
        # If the token is a left parenthesis, push it on the opStack.
        elif token == "(":
            opStack.push(token)
            print("opStack.push:", token)
        # If the token is a right parenthesis,
        #   pop the opStack until the corresponding left parenthesis is removed.
        #   Append each operator to the end of the output list.
        elif token == ")":
            topToken = opStack.pop()
            print("opStack.pop:", token)
            while topToken != "(":
                postfixList.append(topToken)
                print("postfixList.append:", token, postfixList)
                topToken = opStack.pop()

        # If the token is an operator, *, /, +, or -, push it on the opStack.
        #   However, first remove any operators already on the opStack that have higher or equal precedence and append them to the output list.
        else:
            while (not opStack.isEmpty()) and (prec[opStack.peek()] >= prec[token]):
                postfixList.append(opStack.pop())
                print("postfixList.append:", token, postfixList)
            opStack.push(token)
            print("opStack.push for caculator:", token)
    # When the input expression has been completely processed, check the opStack. Any operators still on the stack can be removed and appended to the end of the output list.
    while not opStack.isEmpty():
        out = opStack.pop()
        postfixList.append(out)
        print("postfixList.append:", out, postfixList)
    return " ".join(postfixList)


# def infixToPostfix(infixexpr):
#   prec = {}
#   prec["**"] = 4
#   prec["*"] = 3
#   prec["/"] = 3
#   prec["+"] = 2
#   prec["-"] = 2
#   prec["("] = 1

#   opStack = Stack()
#   postfixList = []
#   tokenList = infixexpr.split()

#   for token in tokenList:
#     if token in "ABCDEFGHIJKLMNOPQRSTUVWXYZ" or token in "0123456789":
#       postfixList.append(token)
#     elif token == '(':
#       opStack.push(token)
#     elif token == ')':
#       topToken = opStack.pop()
#       while topToken != '(':
#         postfixList.append(topToken)
#         topToken = opStack.pop()
#     else:
#       while (not opStack.isEmpty()) and (prec[opStack.peek()] >= prec[token]):
#         postfixList.append(opStack.pop())
#       opStack.push(token)

#   while not opStack.isEmpty():
#     postfixList.append(opStack.pop())

#   return " ".join(postfixList)

# print(infixToPostfix("( A + B ) * C"))
# print(infixToPostfix("( A + B / E ) * ( C + D )"))
# print(infixToPostfix("A * B + C * D"))
# print(infixToPostfix("( A + B ) * C - ( D - E ) * ( F + G )"))
# print(infixToPostfix("10 + 3 * 5 / (16 - 4)"))
print(infixToPostfix("5 * 3 ** ( 4 - 2 )"))


# use stack -> calculate Postfix
def postfixEval(postfixExpr):
    # Create an empty stack called operandStack.
    operandStack = Stack()
    # Convert the string to a list by using the string method split.
    token_list = postfixExpr.split()
    # Scan the token list from left to right.
    for token in token_list:
        # If the token is an operand, convert it from a string to an integer and push the value onto the operandStack.
        if token in "0123456789":
            operandStack.push(int(token))
        # If the token is an operator, *, /, +, or -, it will need two operands. Pop the operandStack twice.
        # The first pop is the second operand and the second pop is the first operand.
        # Perform the arithmetic operation.
        # Push the result back on the operandStack.
        else:
            second_ope = operandStack.pop()
            first_ope = operandStack.pop()
            result = doMath(token, first_ope, second_ope)
            print(first_ope, token, second_ope, result)
            operandStack.push(result)
    # When the input expression has been completely processed, the result is on the stack. Pop the operandStack and return the value.
    return operandStack.pop()


def doMath(op, op1, op2):
    if op == "*":
        return op1 * op2
    elif op == "/":
        return op1 / op2
    elif op == "+":
        return op1 + op2
    else:
        return op1 - op2


# print(postfixEval('7 8 + 3 2 + /'))
# print(postfixEval('17 10 + 3 * 9 /'))
