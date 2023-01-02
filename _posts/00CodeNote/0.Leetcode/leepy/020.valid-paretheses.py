# 20. Valid Parentheses
# Easy
# Given a string s containing just the characters '(', ')', '{', '}', '[' and ']', determine if the input string is valid.

# An input string is valid if:
# Open brackets must be closed by the same type of brackets.
# Open brackets must be closed in the correct order.

# Example 1:
# Input: s = "()"
# Output: true

# Example 2:
# Input: s = "()[]{}"
# Output: true

# Example 3:
# Input: s = "(]"
# Output: false

# Example 4:
# Input: s = "([)]"
# Output: false

# Example 5:
# Input: s = "{[]}"
# Output: true

# Constraints:
# 1 <= s.length <= 104
# s consists of parentheses only '()[]{}'.


class Stack:
    def __init__(self):
        self.items = []

    def is_empty(self):
        return self.items == []

    def stack(self, item):
        self.items.append(item)

    def pop(self):
        return self.items.pop()

    def peek(self):
        return self.items[-1]

    def size(self):
        return len(self.items)


# use dic + stack
def isValid(s):
    out_dict = {"(": ")", "[": "]", "{": "}"}
    stack = []
    for char in s:
        if char in out_dict:
            # if opening bracket add into stack
            stack.append(char)
        elif (not stack) or (char != out_dict[stack.pop()]):
            # if closing bracket map with the correct opening bracket
            return False
    return stack == []
    # if len(stack) == 0:
    #     return True
    # return False
    # Check if all the opening brackets are closed


# use dic + stack
def isValid(s):
    map = {"(": ")", "{": "}", "[": "]"}
    stack = []
    for ch in s:
        if len(stack) == 0:
            stack.append(ch)
        else:
            if map.get(stack[-1]) == ch:
                stack.pop()
            else:
                stack.append(ch)
    return stack == []


# use stack
def par_checker(symbol_string):
    s = Stack()
    map = {"(": ")", "[": "]", "{": "}"}
    for symbol in "([{":
        if symbol in map:
            s.stack(symbol)
        elif (not s.is_empty()) and symbol == map.get(s.peek()):
            s.pop()
        else:
            return False
    return s.is_empty()


# stack
def matches(opener, closer):
    openers = "({["
    closers = ")}]"
    return openers.index(opener) == closers.index(closer)


def par_checker(symbol_string):
    s = Stack()
    for i in symbol_string:
        if i in "([{":
            s.stack(i)
        elif (not s.is_empty()) and matches(s.pop(), i):
            continue
        else:
            return False
    return s.is_empty()


# replace/remove the pair
def isValid(self, s: str) -> bool:
    newstr = ""
    for i in s:
        if i in ["(", ")", "{", "}", "[", "]"]:
            newstr += i
    while "()" in newstr or "{}" in newstr or "[]" in newstr:
        newstr = newstr.replace("()", "").replace("{}", "").replace("[]", "")
    return newstr == ""


# # Runtime: 40 ms, faster than 17.44% of Python3 online submissions for Valid Parentheses.
# # Memory Usage: 13.7 MB, less than 6.09% of Python3 online submissions for Valid Parentheses.


def isValid(self, s: str) -> bool:
    while "()" in s or "{}" in s or "[]" in s:
        s = s.replace("()", "").replace("{}", "").replace("[]", "")
    return s == ""


# # Runtime: 80 ms, faster than 6.27% of Python3 online submissions for Valid Parentheses.
# # Memory Usage: 14 MB, less than 5.22% of Python3 online submissions for Valid Parentheses.


# == JAVA==
# import java.util.HashMap;
# import java.util.Stack;
# class Solution {
#     // Hash table that takes care of the mappings.
#     private HashMap<Character, Character> mappings;

#     // Initialize hash map with mappings. This simply makes the code easier to read.
#     public Solution() {
#       this.mappings = new HashMap<Character, Character>();
#       this.mappings.put(')', '(');
#       this.mappings.put('}', '{');
#       this.mappings.put(']', '[');
#     }

#     public boolean isValid(String s) {

#       // Initialize a stack to be used in the algorithm.
#       Stack<Character> stack = new Stack<Character>();

#       for (int i = 0; i < s.length(); i++) {
#         char c = s.charAt(i);

#         // If the current character is a closing bracket.
#         // Input: ( )[]{}, ([)]
#         if (this.mappings.containsKey(c)) {

#           // Get the top element of the stack. If the stack is empty, set a dummy value of '#'
#           char topElement = stack.empty() ? '#' : stack.pop();

#           // If the mapping for this bracket doesn't match the stack's top element, return false.
#           if (topElement != this.mappings.get(c)) {
#             return false;
#           }
#         } else {
#           // If it was an opening bracket, stack to the stack.
#           stack.stack(c);
#         }
#       }
#       // If the stack still contains elements, then it is an invalid expression.
#       return stack.isEmpty();
#     }

# 	public Object findDisappearedNumbers(int[] arr) {
# 		return null;
# 	}
# }


# print(par_checker("((())(()"))  #expected False
# print(par_checker("((()))"))  # expected True
# print(par_checker("((()()))"))  # expected True
# print(par_checker("(()"))  # expected False
# print(par_checker(")("))  # expected False

print(isValid("()"))
