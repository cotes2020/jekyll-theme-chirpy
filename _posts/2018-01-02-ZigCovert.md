---
layout: post
title: "Z 字形变换"
date: 2018-01-02 21:22:00.000000000 +09:00
categories: [算法]
tags: [算法, Z 字形变换]
---

> 将一个给定字符串根据给定的行数，以从上往下、从左到右进行 Z 字形排列。

比如输入字符串为 `"LEETCODEISHIRING"` 行数为 3 时，排列如下：

```
L   C   I   R
E T O E S I I G
E   D   H   N
```

之后，你的输出需要从左往右逐行读取，产生出一个新的字符串，比如：`"LCIRETOESIIGEDHN"`。

请你实现这个将字符串进行指定行数变换的函数：

```
string convert(string s, int numRows);
```

**示例 1:**

```
输入: s = "LEETCODEISHIRING", numRows = 3
输出: "LCIRETOESIIGEDHN"
```

**示例 2:**

```
输入: s = "LEETCODEISHIRING", numRows = 4
输出: "LDREOEIIECIHNTSG"
解释:

L     D     R
E   O E   I I
E C   I H   N
T     S     G
```

**思路**

Z字形，就是两种状态，一种垂直向下，还有一种斜向上

控制好边界情况就可以了。

**规律**

![](/assets/images/al-ZigCovert-01.png)

![](/assets/images/al-ZigCovert-02.png)

如上图所示，我们发现规律：

1. 每一个Z字的首字母差，`len = 2 * numRows - 2` 位置
2. 除去首尾两行，每个 Z 字每行有两个字母，索引号关系为`index = index + 2 * (numRows - i - 1)`

**复杂度分析**

- 时间复杂度：O(log(n + m))
- 空间复杂度：O(1) 。

**代码实现**

```swift
func convert(_ s: String, _ numRows: Int) -> String {
        
    if numRows == 1 {
        return s
    }
    var res: [Character] = []
    var chars: [Character] = Array(s)	// 字符串转数组
    let count = chars.count
    let len = 2 * numRows - 2 // 每个Z首字母差长度
    for i in 0..<numRows {

        var index = i
        while index < count {

            res.append(chars[index])
            if i != 0 && i != numRows - 1 { // 排除首尾

                let k = index + 2 * (numRows - i - 1)
                if k < count {
                    res.append(chars[k])
                }
            }
            index += len    // 循环下一个Z
        }
    }
    return String(res)
}
```

[源码地址](https://github.com/Jovins/Algorithm)