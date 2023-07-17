---
title: How to Traverse Trees in Rust
author: Hulua
date: 2023-07-16 20:55:00 +0800
categories: [Rust]
tags: [rust]
---

Today learned some something new about Rust when trying to use Rust to traverse binary trees. There are some pattern differences while using Rust to traverse binary trees compared with other languages. One main difference is that Rust does not provide NULL (or Nil) pointer. The alternative to refer to a case without children is None of Option. Also the definition of TreeNode is somewhat different. 

Let's first look at the tree node definition:

```rust
pub struct TreeNode<T> {
    value: T,
    left: Option<Box<TreeNode<T>>>,
    right: Option<Box<TreeNode<T>>>,
}
```

Here we can see we wrap the TreeNode inside a Box pointer. and outside the pointer we wrapped it in an Option, and we have a generic parameter T. Next, we can implment a utility function to create a leaf node as:

```Rust
impl<T> TreeNode<T> {
    fn leaf(value: T) -> Option<Box<TreeNode<T>>> {
        Some(Box::new(Self {
            value,
            left: None,
            right: None,
        }))
    }
}
```

With that, we can implement a basic inorder traverse function:


```rust
impl<T> TreeNode<T> {
    pub fn traverse_inorder1(&self, f: &mut impl FnMut(&T)) {
        if let Some(left) = &self.left {
            left.traverse_inorder1(f);
        }
        f(&self.value);
        if let Some(right) = &self.right {
            right.traverse_inorder1(f);
        }
    }
}
```

In this version, we implement the `traverse_inorder1` function for the TreeNode, note the 2nd parameter is a closure with the form of ` f: &mut impl FnMut(&T)`, it means the function that will visit the value of the node. The function body is like other languages where we use recursion to traverse the tree.

So far, although there are some new things using Rust to traverse binary tree, what really makes things intersting is the usage of `ControlFlow`. In rust core, it is defined as:

```rust

#[stable(feature = "control_flow_enum_type", since = "1.55.0")]
// ControlFlow should not implement PartialOrd or Ord, per RFC 3058:
// https://rust-lang.github.io/rfcs/3058-try-trait-v2.html#traits-for-controlflow
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ControlFlow<B, C = ()> {
    /// Move on to the next phase of the operation as normal.
    #[stable(feature = "control_flow_enum_type", since = "1.55.0")]
    #[lang = "Continue"]
    Continue(C),
    /// Exit the operation without running subsequent phases.
    #[stable(feature = "control_flow_enum_type", since = "1.55.0")]
    #[lang = "Break"]
    Break(B),
    // Yes, the order of the variants doesn't match the type parameters.
    // They're in this order so that `ControlFlow<A, B>` <-> `Result<B, A>`
    // is a no-op conversion in the `Try` implementation.
}
```
 
In short, it provide one way to either continue something or break something. This enum has 2 generic parameters. B is for the case when we want to break the execution and return its value (type B), while for the case of continue execution, since usually no return value, so usually `C=()`.

Now given one requirement, inorder traverse the binary tree and stop whenever hit a negative node. Then what should we do?. Taking advantage of `ControlFlow`, we can change the closure like below:

```rust
    pub fn traverse_inorder2<B>(&self, f: &mut impl FnMut(&T) -> ControlFlow<B>) -> ControlFlow<B> {
        if let Some(left) = &self.left {
            left.traverse_inorder2(f)?;
        }
        f(&self.value)?;
        if let Some(right) = &self.right {
            right.traverse_inorder2(f)?;
        }
        ControlFlow::Continue(())
    }

```

Here the function signature changed to `pub fn traverse_inorder2<B>(&self, f: &mut impl FnMut(&T) -> ControlFlow<B>) -> ControlFlow<B> `, here we can the generic parameter B, which indicates the type to return when break (assume hit negative node), and the closure also return the `ControlFlow`, so as the whole function.

The whole test code is below:
```rust
use std::ops::ControlFlow;

pub struct TreeNode<T> {
    value: T,
    left: Option<Box<TreeNode<T>>>,
    right: Option<Box<TreeNode<T>>>,
}

impl<T> TreeNode<T> {
    pub fn traverse_inorder1(&self, f: &mut impl FnMut(&T)) {
        if let Some(left) = &self.left {
            left.traverse_inorder1(f);
        }
        f(&self.value);
        if let Some(right) = &self.right {
            right.traverse_inorder1(f);
        }
    }

    pub fn traverse_inorder2<B>(&self, f: &mut impl FnMut(&T) -> ControlFlow<B>) -> ControlFlow<B> {
        if let Some(left) = &self.left {
            left.traverse_inorder2(f)?;
        }
        f(&self.value)?;
        if let Some(right) = &self.right {
            right.traverse_inorder2(f)?;
        }
        ControlFlow::Continue(())
    }

    fn leaf(value: T) -> Option<Box<TreeNode<T>>> {
        Some(Box::new(Self {
            value,
            left: None,
            right: None,
        }))
    }
}

fn main() {
    let node = TreeNode {
        value: 0,
        left: TreeNode::leaf(1),
        right: Some(Box::new(TreeNode {
            value: -1,
            left: TreeNode::leaf(5),
            right: TreeNode::leaf(2),
        })),
    };

    let mut sum1 = 0;
    node.traverse_inorder1(&mut |val| {
        sum1 += val;
    });
    println!("The sum1 is {}", sum1);

    let mut sum2 = 0;
    let res = node.traverse_inorder2(&mut |val| {
        if *val < 0 {
            ControlFlow::Break(*val)
        } else {
            sum2 += *val;
            ControlFlow::Continue(())
        }
    });
    println!("The sum2 is {}", sum2);
    println!("The res is {:?}", res);
}
```

So, to break the execution when hit a negative node, we pass

```rust
&mut |val| {
        if *val < 0 {
            ControlFlow::Break(*val)
        } else {
            sum2 += *val;
            ControlFlow::Continue(())
        }
    }
```
this closure to the function. This is an interesting way to control execution flow in Rust.