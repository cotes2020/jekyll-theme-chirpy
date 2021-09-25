# from test import testEqual

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

# testEqual(reverse("hello"),"olleh")
# testEqual(reverse("l"),"l")
# testEqual(reverse("follow"),"wollof")
# testEqual(reverse(""),"")

# print(reverse("hello")=="olleh")
# print(reverse("l")=="l")
# print(reverse("follow")=="wollof")
# print(reverse("")=="")



def removeWhite(s):
    s = s.replace(" ", "").replace("'","").replace('"','')
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

print(isPal("x"))
print(isPal("radar"))
print(isPal("hello"))
print(isPal(""))
print(isPal("hannah"))
print(isPal(removeWhite("madam i'm adam")))

# testEqual(isPal(removeWhite("x")),True)
# testEqual(isPal(removeWhite("radar")),True)
# testEqual(isPal(removeWhite("hello")),False)
# testEqual(isPal(removeWhite("")),True)
# testEqual(isPal(removeWhite("hannah")),True)
# testEqual(isPal(removeWhite("madam i'm adam")),True)

