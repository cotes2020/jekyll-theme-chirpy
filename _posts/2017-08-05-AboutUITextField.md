---
title: 关于 UITextField 和 键盘 的通知、代理调用顺序
date: 2017-08-05 10:36:54
categories: iOS
tags: UI
---


### 从点击UITextField到键盘弹出完成，调用代理方法或发通知的顺序
```
1. textFieldShouldBeginEditing:                （调代理）
2. textFieldDidBeginEditing:                   （调代理）
3. UITextFieldTextDidBeginEditingNotification  （发通知）
4. UIKeyboardWillChangeFrameNotification       （发通知）
5. UIKeyboardWillShowNotification              （发通知）
6. UIKeyboardDidShowNotification               （发通知）

```


### 在UITextField中编辑（输入或者删除）文字时，调用代理方法或发通知的顺序
```
// 在此方法中取得的文本框文字是输入或删除之前的
1. textField:shouldChangeCharactersInRange:replacementString:  （调代理）

// 在此通知中取得的文本框文字是输入或删除之后的
2. UITextFieldTextDidChangeNotification                        （发通知）

```

### 在UITextField结束编辑时，调用代理方法或发通知的顺序
```
1. textFieldShouldEndEditing:              （调代理）
2. UIKeyboardWillChangeFrameNotification   （发通知）
3. UIKeyboardWillHideNotification          （发通知）
4. textFieldDidEndEditing:                 （调代理）
5. UITextFieldTextDidEndEditingNotification（发通知）
6. UIKeyboardDidHideNotification           （发通知）

```

<br>
<br>
<br>

