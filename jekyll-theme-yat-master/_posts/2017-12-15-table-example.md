---
layout: post
title: Table example
subtitle:
categories: example
tags: [table]
---

## Table example as below

**For now, these extended features are provided:**

* Cells spanning multiple columns
* Cells spanning multiple rows
* Cells text align separately
* Table header not required
* Grouped table header rows or data rows

### Rowspan and Colspan
^^ in a cell indicates it should be merged with the cell above.  
This feature is contributed by [pmccloghrylaing](https://github.com/pmccloghrylaing).  

| Stage | Direct Products | ATP Yields |
| ----: | --------------: | ---------: |
|Glycolysis | 2 ATP                   ||
|^^         | 2 NADH      | 3--5 ATP   |
|Pyruvaye oxidation | 2 NADH | 5 ATP   |
|Citric acid cycle  | 2 ATP           ||
|^^                 | 6 NADH | 15 ATP  |
|^^                 | 2 FADH | 3 ATP   |
| 30--32 ATP                         |||

[ Net ATP yields per hexose]

### Multiline
A backslash at end to join cell contents with the following lines.  
This feature is contributed by [Lucas-C](https://github.com/Lucas-C).

|:     Easy Multiline     :|||
|:------ |:------ |:-------- |
| Apple  | Banana |  Orange  \
| Apple  | Banana |  Orange  \
| Apple  | Banana |  Orange
| Apple  | Banana |  Orange  \
| Apple  | Banana |  Orange  |
| Apple  | Banana |  Orange  |


### Headerless
Table header can be eliminated.

|--|--|--|--|--|--|--|--|
|♜ |  |♝ |♛ |♚ |♝ |♞ |♜ |
|  |♟ |♟ |♟ |  |♟ |♟ |♟ |
|♟ |  |♞ |  |  |  |  |  |
|  |♗ |  |  |♟ |  |  |  |
|  |  |  |  |♙ |  |  |  |
|  |  |  |  |  |♘ |  |  |
|♙ |♙ |♙ |♙ |  |♙ |♙ |♙ |
|♖ |♘ |♗ |♕ |♔ |  |  |♖ |


```markdown
|:     Fruits \|\| Food           :|||
|:-------- |:-------- |:------------ |
| Apple    |: Apple  :|    Apple     \
| Banana   |  Banana  |    Banana    \
| Orange   |  Orange  |    Orange    |
|:   Rowspan is 5   :||:  How's it? :|
|^^   A. Peach       ||^^ 1. Fine    |
|^^   B. Orange      ||^^ 2. Bad  $I = \int \rho R^{2} dV$     |
|^^   C. Banana      ||   It's OK! ![example image][my-image]  |
```

### Text Alignment
Table cell can be set alignment separately.

| \:Fruits         ||  Food   :|
|:-------- |:------ |:-------- |
| Apple    | Banana |  Orange  |
| Apple    | Banana |  Orange  |


|          | Fruits\::        ||
|:-------- |:------ |:-------- |
| Apple    | Banana |  Orange  |
| Apple    | Banana |  Orange  |


|: \:Fruits       :||          |:       Food     :||
|:-------- |:------ |:-------- |:-------- |:------ |
| Apple    | Banana |  Orange  |:   Strawberry    :|
| Apple  &  Banana || ^^       |    Peach        :||


|: \:Fruits       :||          |:       Food     :||
| Apple    | Banana |  Orange  |:   Strawberry    :|


|:     Fruits \|\| Food           :|||
|:-------- |:-------- |:------------ |
| Apple    |: Apple  :|    Apple     \
| Banana   |  Banana  |    Banana    \
| Orange   |  Orange  |    Orange    |
|:   Rowspan is 5   :||:  How's it? :|
|^^   A. Peach       ||^^ 1. Fine    |
|^^   B. Orange      ||^^ 2. Bad  $I = \int \rho R^{2} dV$     |
|^^   C. Banana      ||   It's OK! ![example image][my-image]  |

[my-image]: http://www.unexpected-vortices.com/sw/rippledoc/example-image.jpg "An exemplary image"

