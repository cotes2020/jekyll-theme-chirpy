---
layout: post
title: "有效的括号"
date: 2019-06-08 21:11:00.000000000 +09:00
categories: [算法]
tags: [算法, 有效的括号]
---

> 给定一个只包括 `'('`，`')'`，`'{'`，`'}'`，`'['`，`']'` 的字符串，判断字符串是否有效。
>
> 有效字符串需满足：
>
> 1. 左括号必须用相同类型的右括号闭合。
> 2. 左括号必须以正确的顺序闭合。
>
> 注意空字符串可被认为是有效字符串。

**示例 1:**

```
输入: "()"
输出: true
```

**示例 2:**

```
输入: "()[]{}"
输出: true
```

**示例 3:**

```
输入: "(]"
输出: false
```

**示例 4:**

```
输入: "([)]"
输出: false
```

**示例 5:**

```
输入: "{[]}"
输出: true
```

**分析**

这道题让我们验证输入的字符串是否为括号字符串，包括大括号，中括号和小括号。

**方法**

使用**栈**

- 遍历输入字符串
- 如果当前字符为左半边括号时，则将其压入栈中
- 如果遇到右半边括号时，**分类讨论：**
- 1）如栈不为空且为对应的左半边括号，则取出栈顶元素，继续循环
- 2）若此时栈为空，则直接返回false
- 3）若不为对应的左半边括号，反之返回false

关于有效括号表达式的一个有趣属性是有效表达式的子表达式也应该是有效表达式。（不是每个子表达式）例如:

![](/assets/images/al-Valid-Parentheses-01.png)

**动画演示**

![](/assets/images/al-Valid-Parentheses-02.gif)

**复杂度分析**

- 时间复杂度：O(n)*O*(*n*)，因为我们一次只遍历给定的字符串中的一个字符并在栈上进行 O(1)*O*(1) 的推入和弹出操作。
- 空间复杂度：O(n)*O*(*n*)，当我们将所有的开括号都推到栈上时以及在最糟糕的情况下，我们最终要把所有括号推到栈上。例如 `((((((((((`。

**代码实现**

```swift
func isValid(_ s: String) -> Bool {
        
    var stack = [Character]()
    for char in s {

        if char == "(" || char == "[" || char == "{" {

            stack.append(char)
        } else if char == ")" {

            guard stack.count != 0 && stack.removeLast() == "(" else {
                return false
            }
        } else if char == "]" {

            guard stack.count != 0 && stack.removeLast() == "[" else {
                return false
            }
        } else if char == "}" {

            guard stack.count != 0 && stack.removeLast() == "{" else {
                return false
            }
        }
    }
    return stack.isEmpty
}
```

[源码地址](<https://github.com/Jovins/Algorithm>)